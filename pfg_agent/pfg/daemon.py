"""Интерактивный демон PFG-Agent: событийно-управляемый жизненный цикл.

Три режима работы (см. ТЗ, раздел Lifecycle):
1. Режим реакции   — событие в шине (message/file_inbox/task/cron) -> agent.run_task;
2. Автономный цикл — при простое внутренняя мотивация сама ставит цели;
3. Фаза сна        — тяжёлая консолидация по расписанию: VACUUM БД, дистилляция
                     рёбер в символические формулы, компиляция навыков, апоптоз связей.

Демон запускается один раз (`python -m pfg.cli daemon` или `repl`) и живёт, пока
не получит SIGINT/SIGTERM или команду `stop` в REPL. В ожидании событий потребляет
~0% CPU (блокирующий poll с таймаутом).
"""
from __future__ import annotations

import logging
from pathlib import Path
import signal
import threading
import time
from typing import Any

from pfg.agent import AgentResult, PFGAgent
from pfg.config import AgentConfig
from pfg.events import EventBus, Event, FileInboxWatcher, TaskQueue, TelegramListenerStub
from pfg.motivation import MotivationEngine, get_disk_free_gb
from pfg.scheduler import Scheduler, default_jobs
from pfg.utils import JsonlLogger, now_iso

logger = logging.getLogger("pfg.daemon")


class SleepPhase:
    """Ночная консолидация: сжатие памяти, дистилляция, навыки, апоптоз связей."""

    def __init__(self, agent: PFGAgent) -> None:
        self.agent = agent
        # свой логгер на корень проекта — не зависит от cwd процесса
        log_dir = Path(agent.config.log_dir)
        if not log_dir.is_absolute():
            log_dir = Path(agent.config.root) / log_dir
        self.log = JsonlLogger(log_dir)

    def run(self) -> dict[str, Any]:
        a = self.agent
        report: dict[str, Any] = {"started_at": now_iso()}
        try:
            # 1. Синаптический апоптоз + рост графа (уже реализовано в graph/evolve)
            report["evolve"] = a.evolve()
            # 2. Рефлексия по накопленным эпизодам -> уроки
            lessons = a.reflection.reflect()
            report["lessons"] = len(lessons)
            # 3. Сжатие БД (VACUUM) — умеренно, чтобы беречь SD-карту
            try:
                a.storage.execute("VACUUM")
                report["vacuum"] = "ok"
            except Exception as exc:  # noqa: BLE001 — VACUUM может не идти внутри транзакции
                report["vacuum"] = f"skipped: {exc}"
            # 4. Сохранение графа
            a.save()
            report["graph"] = a.graph.stats()
        except Exception:  # noqa: BLE001
            logger.exception("фаза сна завершилась с ошибкой")
            report["error"] = "sleep phase failed"
        report["finished_at"] = now_iso()
        self.log.write("learning.jsonl", {"level": "INFO", "source": "sleep",
                                       "message": "sleep phase finished", "data": report})
        logger.info("фаза сна завершена: %s", {k: v for k, v in report.items() if k != "evolve"})
        return report


class AutonomousWorker:
    """Автономный цикл: при простое агент сам генерирует цели через мотивацию."""

    def __init__(self, agent: PFGAgent, bus: EventBus, task_queue: TaskQueue) -> None:
        self.agent = agent
        self.bus = bus
        self.task_queue = task_queue
        self.motivation = MotivationEngine(agent.config)
        self.goals_generated = 0
        self.last_goal: str | None = None

    def work_once(self) -> AgentResult | None:
        """Один шаг автономной работы. Возвращает результат задачи или None (спит)."""
        status = self.agent.status()
        pending = sum(1 for t in self.task_queue.load() if t.get("status") == "pending")
        candidates = 0
        try:
            candidates = len(self.agent.skills.find_repetitive_subgraphs(min_usage=10))
        except Exception:  # noqa: BLE001
            pass
        db_path = self.agent.config.resolve_path(self.agent.config.memory.sqlite_path)
        db_mb = db_path.stat().st_size / (1024 ** 2) if db_path.exists() else 0.0
        goal = self.motivation.generate_internal_goal(
            agent_status=status,
            pending_tasks=pending,
            skill_candidates=candidates,
            disk_free_gb=get_disk_free_gb(self.agent.config.root),
            db_size_mb=db_mb,
            unsaved_steps=int(status.get("graph", {}).get("steps_since_save", 0)),
        )
        if goal is None:
            return None
        self.goals_generated += 1
        self.last_goal = goal.goal
        self.bus.emit("internal", {"goal": goal.to_dict()}, source="motivation", priority=8)
        return self.agent.run_task(goal.goal)


class AgentDaemon:
    """Главный событийный цикл агента (Event-Driven Daemon)."""

    def __init__(self, config: AgentConfig | None = None,
                 agent: PFGAgent | None = None) -> None:
        self.config = config or (agent.config if agent else AgentConfig.load())
        self.agent = agent or PFGAgent(self.config)
        self.bus = EventBus()
        root = self.config.root
        self.inbox_watcher = FileInboxWatcher(
            self.bus, inbox_dir=root / "data" / "inbox", root=root,
            interval_s=float(getattr(self.config.daemon, "inbox_scan_interval_s", 5.0)))
        self.telegram = TelegramListenerStub(self.bus, feed_path=root / "data" / "inbox" / "telegram.jsonl")
        self.task_queue = TaskQueue(self.bus, queue_path=root / "data" / "task_queue.json",
                                    outbox_dir=root / "sandbox" / "outbox")
        self.scheduler = Scheduler(self.bus)
        for name, interval, kind in default_jobs(self.config):
            self.scheduler.add_job(name, interval, kind)
        self.autonomous = AutonomousWorker(self.agent, self.bus, self.task_queue)
        self.sleep = SleepPhase(self.agent)
        self.log = JsonlLogger(self.config.log_dir)
        self.is_alive = False
        self._events_handled = 0
        self._last_activity = time.time()
        self._idle_threshold = float(getattr(self.config.daemon, "idle_before_autonomous_s", 30.0))
        self._install_signal_handlers()

    # ---------- обработка событий ----------
    def handle_event(self, ev: Event) -> AgentResult | None:
        """Режим реакции: превращает событие в задачу агента (если содержательное)."""
        self._last_activity = time.time()
        self._events_handled += 1
        goal = self._event_to_goal(ev)
        if not goal:
            return None
        logger.info("событие %s (%s) -> цель: %s", ev.id, ev.type, goal[:80])
        result = self.agent.run_task(goal)
        if ev.type == "task" and ev.payload.get("task_id"):
            self.task_queue.complete(ev.payload["task_id"],
                                     f"# Отчёт по задаче {ev.payload['task_id']}\n\n"
                                     f"Цель: {goal}\nУспех: {result.success}\n"
                                     f"Ответ: {result.answer}\nВремя: {now_iso()}\n")
        self.log.write("actions.jsonl", {"level": "INFO", "source": "daemon",
                                         "message": "event handled",
                                         "data": {"event": ev.to_dict(), "success": result.success}})
        return result

    @staticmethod
    def _event_to_goal(ev: Event) -> str:
        p = ev.payload or {}
        if ev.type == "message":
            return str(p.get("text", ""))[:500]
        if ev.type == "file_inbox":
            if p.get("error"):
                return ""
            text = str(p.get("goal") or p.get("text", "")).strip()
            return text[:500]
        if ev.type == "task":
            return str(p.get("payload", ""))[:500]
        return ""

    def _on_cron(self, ev: Event) -> None:
        kind = (ev.payload or {}).get("kind", "")
        if kind == "autosave":
            self.agent.save()
        elif kind == "scan_inbox":
            self.inbox_watcher.scan_once()
            self.task_queue.poll_into_bus()
            self.telegram.poll_once()
        elif kind == "sleep":
            self.sleep.run()
        elif kind == "autonomous":
            # автономная цель только если внешних событий давно нет
            if time.time() - self._last_activity >= self._idle_threshold:
                self.autonomous.work_once()

    # ---------- главный цикл ----------
    def run_forever(self) -> None:
        """Бесконечный event loop по псевдокоду ТЗ: реакция -> автономия -> сон."""
        self.bus.subscribe("cron", self._on_cron)
        self.is_alive = True
        self.inbox_watcher.start()
        self.task_queue.poll_into_bus()
        logger.info("демон '%s' запущен; режимы: реакция/автономия/сон", self.config.name)
        poll_s = float(getattr(self.config.daemon, "poll_timeout_s", 2.0))
        try:
            while self.is_alive:
                self.scheduler.tick()
                ev = self.bus.poll(timeout=poll_s)
                if ev is not None and ev.type != "cron":
                    self.handle_event(ev)
                elif ev is None and time.time() - self._last_activity >= self._idle_threshold * 2:
                    # шина пуста достаточно долго — один шаг автономии за итерацию
                    self.autonomous.work_once()
                    self._last_activity = time.time()
        finally:
            self.shutdown()

    def run_once(self, timeout: float = 0.0) -> AgentResult | None:
        """Один виток цикла (для тестов): tick + poll + реакция/автономия."""
        self.bus.subscribe("cron", self._on_cron)
        self.scheduler.tick()
        ev = self.bus.poll(timeout=timeout)
        if ev is not None and ev.type != "cron":
            return self.handle_event(ev)
        return None

    def stop(self) -> None:
        self.is_alive = False

    def shutdown(self) -> None:
        self.is_alive = False
        self.inbox_watcher.stop()
        self.scheduler.stop()
        try:
            self.agent.save()
            self.agent.close()
        except Exception:  # noqa: BLE001
            logger.exception("ошибка при корректном завершении демона")
        logger.info("демон остановлен; обработано событий: %d", self._events_handled)

    def _install_signal_handlers(self) -> None:
        def _handler(signum: int, _frame: Any) -> None:
            logger.info("получен сигнал %s — корректное завершение", signum)
            self.stop()

        for sig in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                pass  # не главный поток или платформа без сигнала — не критично

    def status(self) -> dict[str, Any]:
        return {
            "alive": self.is_alive,
            "events_handled": self._events_handled,
            "bus_pending": self.bus.pending,
            "idle_s": round(time.time() - self._last_activity, 1),
            "autonomous_goals": self.autonomous.goals_generated,
            "last_goal": self.autonomous.last_goal,
            "jobs": self.scheduler.list_jobs(),
            "agent": self.agent.status(),
        }


class ReplSession:
    """Интерактивная REPL-сессия поверх того же демона (режим 1 без systemd).

    Команды: любое сообщение -> задача агента; /status, /queue <цель>, /sleep,
    /auto, /quit. Читает stdin построчно — блокирующий input, 0% CPU в простое.
    """

    def __init__(self, daemon: AgentDaemon | None = None) -> None:
        self.daemon = daemon or AgentDaemon()

    def handle_line(self, line: str) -> str | None:
        line = line.strip()
        if not line:
            return None
        if line == "/quit":
            self.daemon.stop()
            return "Пока. Я продолжу жить в фоне, когда запустишь `pfg daemon`."
        if line == "/status":
            import json
            return json.dumps(self.daemon.status(), ensure_ascii=False, indent=2, default=str)
        if line.startswith("/queue "):
            task = self.daemon.task_queue.add(line[len("/queue "):])
            return f"Задача {task['id']} добавлена в очередь."
        if line == "/sleep":
            rep = self.daemon.sleep.run()
            return f"Фаза сна выполнена: уроков={rep.get('lessons')}, vacuum={rep.get('vacuum')}"
        if line == "/auto":
            res = self.daemon.autonomous.work_once()
            return f"Автономная цель: {res.goal} -> {res.answer}" if res else "Мотивация молчит — делать нечего."
        result = self.daemon.agent.run_task(line)
        return f"[{ 'OK' if result.success else 'FAIL' }, conf={result.confidence:.2f}] {result.answer}"

    def run(self) -> int:
        print(f"PFG-Agent '{self.daemon.config.name}' — REPL. /status /queue /sleep /auto /quit")
        self.daemon.is_alive = True
        while self.daemon.is_alive:
            try:
                line = input("> ")
            except (EOFError, KeyboardInterrupt):
                break
            reply = self.handle_line(line)
            if reply is not None:
                print(reply)
        self.daemon.shutdown()
        return 0
