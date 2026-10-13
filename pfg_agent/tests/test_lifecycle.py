"""Тесты жизненного цикла: Event Bus, очередь задач, мотивация, планировщик, демон."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfg.config import AgentConfig
from pfg.daemon import AgentDaemon, ReplSession, SleepPhase
from pfg.events import EventBus, Event, FileInboxWatcher, TaskQueue, TelegramListenerStub
from pfg.motivation import MotivationEngine
from pfg.scheduler import Scheduler


@pytest.fixture()
def tmp_config(tmp_path: Path) -> AgentConfig:
    cfg = AgentConfig.load(root=tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "sandbox").mkdir(exist_ok=True)
    return cfg


# ---------- EventBus ----------
def test_bus_publish_poll_order_by_priority() -> None:
    bus = EventBus()
    bus.emit("message", {"text": "low"}, priority=9)
    bus.emit("message", {"text": "high"}, priority=1)
    first = bus.poll(timeout=0.5)
    assert first is not None and first.payload["text"] == "high"
    second = bus.poll(timeout=0.5)
    assert second is not None and second.payload["text"] == "low"
    assert bus.poll(timeout=0.05) is None  # пустая шина — спим, не падаем


def test_bus_subscriber_error_isolation() -> None:
    bus = EventBus()
    seen = []

    def bad(_ev: Event) -> None:
        raise RuntimeError("сбой подписчика")

    def good(ev: Event) -> None:
        seen.append(ev.id)

    bus.subscribe("message", bad)
    bus.subscribe("message", good)
    ev = bus.emit("message", {"text": "x"})
    polled = bus.poll(timeout=0.5)
    assert polled is not None
    assert seen == [ev.id]  # хороший подписчик сработал несмотря на падение плохого


# ---------- Inbox watcher ----------
def test_inbox_watcher_turns_file_into_goal_event(tmp_config: AgentConfig) -> None:
    bus = EventBus()
    inbox = tmp_config.root / "data" / "inbox"
    watcher = FileInboxWatcher(bus, inbox_dir=inbox, root=tmp_config.root)
    (inbox).mkdir(parents=True, exist_ok=True)
    (inbox / "note.txt").write_text("Запомни, что курс биткоина важен", encoding="utf-8")
    n = watcher.scan_once()
    assert n == 1
    ev = bus.poll(timeout=0.5)
    assert ev is not None and ev.type == "file_inbox"
    assert "биткоина" in ev.payload["text"]
    # файл убран из inbox (перемещён в processed) — не обработаем дважды
    assert not (inbox / "note.txt").exists()


# ---------- Task queue ----------
def test_task_queue_add_claim_complete(tmp_config: AgentConfig) -> None:
    bus = EventBus()
    tq = TaskQueue(bus, queue_path=tmp_config.root / "data" / "task_queue.json",
                   outbox_dir=tmp_config.root / "sandbox" / "outbox")
    task = tq.add("найди 3 способа заработка на RPi4", type_="research", priority="low")
    tasks = json.loads((tmp_config.root / "data" / "task_queue.json").read_text())
    assert any(t["id"] == task["id"] and t["status"] == "pending" for t in tasks)
    moved = tq.poll_into_bus()
    assert moved == 1
    ev = bus.poll(timeout=0.5)
    assert ev is not None and ev.type == "task" and ev.priority == 7
    tq.complete(task["id"], "# отчёт")
    report = tmp_config.root / "sandbox" / "outbox" / f"{task['id']}.md"
    assert report.exists() and "отчёт" in report.read_text()


# ---------- Telegram stub ----------
def test_telegram_stub_reads_feed(tmp_config: AgentConfig) -> None:
    bus = EventBus()
    feed = tmp_config.root / "data" / "inbox" / "telegram.jsonl"
    feed.parent.mkdir(parents=True, exist_ok=True)
    stub = TelegramListenerStub(bus, feed)
    feed.write_text(json.dumps({"chat_id": 1, "text": "Какой любимый цвет?"}) + "\n", encoding="utf-8")
    assert stub.poll_once() == 1
    ev = bus.poll(timeout=0.5)
    assert ev is not None and ev.type == "message" and "цвет" in ev.payload["text"]
    assert stub.poll_once() == 0  # без дублей


# ---------- Мотивация ----------
def test_motivation_prefers_profit_when_tasks_pending(tmp_config: AgentConfig) -> None:
    mot = MotivationEngine(tmp_config)
    status = {"recent_lessons": 0, "episodes": 0, "safety_violations": 0,
              "graph": {"nodes": 8, "edges": 64}}
    goal = mot.generate_internal_goal(agent_status=status, pending_tasks=3)
    assert goal is not None
    assert "очередь" in goal.goal.lower() or "навык" in goal.goal.lower()


def test_motivation_curiosity_on_errors(tmp_config: AgentConfig) -> None:
    mot = MotivationEngine(tmp_config)
    status = {"recent_lessons": 5, "episodes": 20, "safety_violations": 2,
              "graph": {"nodes": 8, "edges": 12}}
    goal = mot.generate_internal_goal(agent_status=status)
    assert goal is not None
    assert "ошибк" in goal.goal.lower() or "связ" in goal.goal.lower()


def test_motivation_returns_none_when_quiet(tmp_config: AgentConfig) -> None:
    mot = MotivationEngine(tmp_config)
    mot.min_priority = 0.9  # почти недостижимый порог
    status = {"recent_lessons": 0, "episodes": 0, "safety_violations": 0,
              "graph": {"nodes": 8, "edges": 64}}
    assert mot.generate_internal_goal(agent_status=status) is None


# ---------- Scheduler ----------
def test_scheduler_fires_due_jobs_as_cron_events() -> None:
    bus = EventBus()
    sch = Scheduler(bus)
    job = sch.add_job("t", interval_s=10, kind="autosave")
    job.next_run = time.time() - 1  # созрела
    assert sch.tick() == 1
    ev = bus.poll(timeout=0.5)
    assert ev is not None and ev.type == "cron" and ev.payload["kind"] == "autosave"
    assert sch.tick() == 0  # следующая ещё не созрела


# ---------- Демон: полный цикл реакции ----------
def test_daemon_handles_message_event(tmp_config: AgentConfig) -> None:
    daemon = AgentDaemon(tmp_config)
    try:
        daemon.bus.subscribe("cron", daemon._on_cron)
        daemon.telegram.feed.parent.mkdir(parents=True, exist_ok=True)
        daemon.telegram.feed.write_text(
            json.dumps({"chat_id": 7, "text": "Запомни, что любимый цвет синий"}) + "\n",
            encoding="utf-8")
        daemon.telegram.poll_once()
        res = daemon.run_once(timeout=1.0)
        assert res is not None and res.success
        # событие-цель выполнено агентом, факт в памяти
        daemon.agent.memory.retrieve("любимый цвет")
        hits = daemon.agent.memory.retrieve("любимый цвет")
        assert hits and "синий" in hits[0][0].text
        st = daemon.status()
        assert st["events_handled"] >= 1
    finally:
        daemon.shutdown()


def test_daemon_completes_queued_task_with_report(tmp_config: AgentConfig) -> None:
    daemon = AgentDaemon(tmp_config)
    try:
        task = daemon.task_queue.add("Создай файл sandbox/queued.txt с текстом hi")
        daemon.task_queue.poll_into_bus()
        handled = 0
        while True:
            ev = daemon.bus.poll(timeout=0.5)
            if ev is None:
                break
            if ev.type == "task":
                daemon.handle_event(ev)
                handled += 1
        assert handled == 1
        report = daemon.task_queue.outbox / f"{task['id']}.md"
        assert report.exists(), "отчёт по задаче должен попасть в sandbox/outbox/"
        tasks = daemon.task_queue.load()
        assert next(t for t in tasks if t["id"] == task["id"])["status"] == "done"
        assert (tmp_config.root / "sandbox" / "queued.txt").read_text().strip() == "hi"
    finally:
        daemon.shutdown()


# ---------- Фаза сна ----------
def test_sleep_phase_runs_and_reports(tmp_config: AgentConfig) -> None:
    daemon = AgentDaemon(tmp_config)
    try:
        daemon.agent.run_task("Запомни, что 2+2 это проверка сна")
        report = daemon.sleep.run()
        assert "evolve" in report and "vacuum" in report
        assert report.get("error") is None, report
        logf = Path(tmp_config.log_dir if Path(tmp_config.log_dir).is_absolute() else tmp_config.root / tmp_config.log_dir) / "learning.jsonl"
        assert logf.exists() and "sleep phase finished" in logf.read_text()
    finally:
        daemon.shutdown()


# ---------- REPL ----------
def test_repl_commands(tmp_config: AgentConfig) -> None:
    daemon = AgentDaemon(tmp_config)
    session = ReplSession(daemon)
    try:
        assert session.handle_line("") is None
        r = session.handle_line("Запомни, что любимый цвет синий")
        assert r and r.startswith("[OK")
        r2 = session.handle_line("/queue найти способ оптимизации")
        assert r2 and "добавлена" in r2
        r3 = session.handle_line("/status")
        assert r3 and '"agent"' in r3
        assert session.handle_line("/quit") is not None
        assert daemon.is_alive is False
    finally:
        daemon.shutdown()
