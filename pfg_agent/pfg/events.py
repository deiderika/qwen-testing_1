"""Event Bus PFG-Agent: шина событий для событийно-управляемого демона (раздел Lifecycle).

Источники событий:
* FileInboxWatcher — новые файлы в data/inbox/ (пользователь «скидывает» задачи/данные);
* TaskQueue        — очередь задач data/task_queue.json;
* CronScheduler    — периодические таймеры (см. scheduler.py);
* ManualSource     — события, добавленные из REPL / CLI / тестов.

Безопасность: все путиwatcher'ов ограничены root проекта; содержимое inbox читается
как текст с лимитом размера; сеть и shell здесь не используются вообще.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pfg.utils import new_id, now_iso

logger = logging.getLogger("pfg.events")


@dataclass
class Event:
    """Одно событие шины."""
    id: str
    type: str                      # message | file_inbox | task | cron | internal
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    ts: str = field(default_factory=now_iso)
    priority: int = 5              # 1 — самый срочный

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "payload": self.payload,
                "source": self.source, "ts": self.ts, "priority": self.priority}


class EventBus:
    """Потокобезопасная приоритетная очередь событий с подписками."""

    def __init__(self, maxsize: int = 10_000) -> None:
        self._queue: "queue.PriorityQueue[tuple[int, float, Event]]" = queue.PriorityQueue(maxsize=maxsize)
        self._subscribers: dict[str, list[Callable[[Event], None]]] = {}
        self._lock = threading.Lock()
        self._processed = 0

    def publish(self, event: Event) -> bool:
        """Ставит событие в очередь. Возвращает False, если очередь переполнена."""
        try:
            self._queue.put_nowait((event.priority, time.time(), event))
            logger.debug("событие %s (%s) опубликовано", event.id, event.type)
            return True
        except queue.Full:
            logger.warning("EventBus переполнен, событие %s отброшено", event.id)
            return False

    def emit(self, type_: str, payload: dict[str, Any] | None = None,
             source: str = "manual", priority: int = 5) -> Event:
        """Удобная публикация события по типу и полезной нагрузке."""
        ev = Event(id=f"ev_{new_id()}", type=type_, payload=payload or {},
                   source=source, priority=priority)
        self.publish(ev)
        return ev

    def poll(self, timeout: float = 1.0) -> Event | None:
        """Ждёт событие до timeout секунд; 0 — неблокирующий опрос. Спит без CPU-нагрузки."""
        try:
            _, _, ev = self._queue.get(timeout=timeout if timeout > 0 else None) \
                if timeout > 0 else self._queue.get_nowait()
        except queue.Empty:
            return None
        self._processed += 1
        self._notify(ev)
        return ev

    def _notify(self, ev: Event) -> None:
        with self._lock:
            subs = list(self._subscribers.get(ev.type, [])) + list(self._subscribers.get("*", []))
        for cb in subs:
            try:
                cb(ev)
            except Exception:  # noqa: BLE001 — подписчик не должен ронять шину
                logger.exception("ошибка подписчика на тип %s", ev.type)

    def subscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    @property
    def processed(self) -> int:
        return self._processed


class FileInboxWatcher:
    """Следит за папкой inbox: каждый новый файл превращается в событие file_inbox.

    Поддерживает два формата: произвольный текстовый файл (содержимое -> payload.text)
    и .json с готовой структурой {goal/type/payload}. Файл перемещается в processed/,
    чтобы не обрабатывать его дважды (SD-friendly: никаких удалений).
    """

    def __init__(self, bus: EventBus, inbox_dir: Path, root: Path | None = None,
                 max_kb: int = 64, interval_s: float = 2.0) -> None:
        self.bus = bus
        self.inbox = Path(inbox_dir)
        self.root = Path(root) if root else self.inbox.parent.parent
        self.max_bytes = max_kb * 1024
        self.interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # processed/ — соседняя папка, иначе glob("*") в inbox захватит её файлы при rename
        self.processed_dir = self.inbox.parent / "inbox_processed"

    def start(self) -> None:
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, name="inbox-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def scan_once(self) -> int:
        """Один проход по inbox (используется и в тестах, и демоном без потоков)."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for path in sorted(self.inbox.glob("*")):
            if not path.is_file() or path.name.endswith(".tmp"):
                continue
            try:
                size = path.stat().st_size
                if size > self.max_bytes:
                    self._archive(path, {"text": "", "truncated": True})
                    self.bus.emit("file_inbox", {"path": str(path), "error": "file too large"},
                                  source="inbox_watcher", priority=7)
                    count += 1
                    continue
                raw = path.read_text(encoding="utf-8", errors="replace")
                try:
                    rel = str(path.relative_to(self.root))
                except ValueError:
                    rel = path.name
                payload: dict[str, Any] = {"path": rel, "text": raw[:4000]}
                if path.suffix == ".json":
                    try:
                        data = json.loads(raw)
                        if isinstance(data, dict):
                            payload.update({k: v for k, v in data.items() if k in ("goal", "type", "payload", "priority")})
                    except json.JSONDecodeError:
                        pass
                self.bus.emit(payload.get("type", "file_inbox") if path.suffix == ".json" else "file_inbox",
                              payload, source="inbox_watcher", priority=int(payload.get("priority", 4)))
                self._archive(path, {"text": raw})
                count += 1
            except OSError:
                logger.exception("не удалось обработать файл inbox: %s", path)
        return count

    def _archive(self, path: Path, _info: dict[str, Any]) -> None:
        try:
            dest = self.processed_dir / f"{int(time.time())}_{path.name}"
            path.rename(dest)
        except OSError:
            logger.warning("не удалось переместить %s в processed/", path)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.scan_once()


class TaskQueue:
    """Очередь задач data/task_queue.json — пользователь скидывает задачи без ожидания ответа.

    Формат элемента: {"id", "type", "payload", "priority", "added_at"}.
    Приоритеты: high=2, low=7, normal/default=5. Выполненная задача получает
    status=done и результат кладётся в sandbox/outbox/<id>.md.
    """

    PRIORITY_MAP = {"high": 2, "normal": 5, "low": 7}

    def __init__(self, bus: EventBus, queue_path: Path, outbox_dir: Path) -> None:
        self.bus = bus
        self.path = Path(queue_path)
        self.outbox = Path(outbox_dir)
        self.outbox.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            logger.warning("task_queue повреждён, игнорируем: %s", self.path)
            return []

    def save(self, tasks: list[dict[str, Any]]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, goal: str, type_: str = "generic", priority: str = "normal") -> dict[str, Any]:
        """Добавляет задачу в файл очереди (может вызываться извне другим процессом)."""
        task = {"id": f"task_{new_id()}", "type": type_, "payload": goal,
                "priority": priority, "added_at": now_iso(), "status": "pending"}
        tasks = self.load()
        tasks.append(task)
        self.save(tasks)
        return task

    def poll_into_bus(self) -> int:
        """Забирает pending-задачи из файла и публикует их в шину как события `task`."""
        tasks = self.load()
        moved = 0
        for t in tasks:
            if t.get("status", "pending") == "pending":
                t["status"] = "claimed"
                self.bus.emit("task", {"task_id": t["id"], "type": t.get("type", "generic"),
                                       "payload": t.get("payload", ""),
                                       "priority": self.PRIORITY_MAP.get(t.get("priority", "normal"), 5)},
                              source="task_queue",
                              priority=self.PRIORITY_MAP.get(t.get("priority", "normal"), 5))
                moved += 1
        if moved:
            self.save(tasks)
        return moved

    def complete(self, task_id: str, report: str) -> None:
        """Отмечает задачу выполненной и пишет отчёт в outbox."""
        tasks = self.load()
        for t in tasks:
            if t.get("id") == task_id:
                t["status"] = "done"
                t["finished_at"] = now_iso()
        self.save(tasks)
        try:
            safe_id = "".join(c for c in task_id if c.isalnum() or c in "_-")
            (self.outbox / f"{safe_id}.md").write_text(report, encoding="utf-8")
        except OSError:
            logger.exception("не удалось записать отчёт в outbox для %s", task_id)


class TelegramListenerStub:
    """Заглушка Telegram-листанера.

    Честная реализация требует сети и токена, что по умолчанию запрещено
    (allow_network=false). Вместо молчаливого нарушения безопасности мы
    читаем входящие сообщения из data/inbox/telegram.jsonl — этот файл может
    наполнять внешний мост (например, отдельный процесс-бот на другой машине
    или python-telegram-bot при явном включении сети). Каждая строка:
    {"chat_id": ..., "text": "..."} -> событие message.
    """

    def __init__(self, bus: EventBus, feed_path: Path) -> None:
        self.bus = bus
        self.feed = Path(feed_path)
        self._offset = 0

    def poll_once(self) -> int:
        if not self.feed.exists():
            return 0
        n = 0
        try:
            lines = self.feed.read_text(encoding="utf-8").splitlines()
            for line in lines[self._offset:]:
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = str(msg.get("text", ""))[:2000]
                if text:
                    self.bus.emit("message", {"text": text, "chat_id": msg.get("chat_id"),
                                              "channel": "telegram_stub"},
                                  source="telegram", priority=2)
                    n += 1
            self._offset = len(lines)
        except OSError:
            logger.exception("telegram feed недоступен")
        return n
