"""Фоновый планировщик задач PFG-Agent.

Простой cron-подобный планировщик без внешних зависимостей: периодические задачи
(интервал в секундах) превращаются в события шины `cron` с заданным payload.
Демон подписывает обработчики на эти события (см. daemon.py).

Также здесь живёт «фаза сна» — тяжёлая ночная консолидация, вызываемая по расписанию:
[VACUUM] БД, символическая дистилляция рёбер, компиляция навыков из паттернов,
апоптоз неактивных связей (удаление старых малоиспользуемых рёбер через prune_and_grow).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pfg.events import EventBus
from pfg.utils import utc_iso

logger = logging.getLogger("pfg.scheduler")


@dataclass
class ScheduledJob:
    """Периодическая задача."""
    name: str
    interval_s: float
    action: dict[str, Any]            # payload для события cron: {"kind": "...", ...}
    next_run: float = 0.0
    runs: int = 0
    last_result: str | None = None
    enabled: bool = True


class Scheduler:
    """Планировщик фоновых задач поверх Event Bus.

    tick() вызывается главным циклом демона; он неблокирующий и сам решает,
    какие джобы созрели. optional start_thread() — для REPL/тестов.
    """

    def __init__(self, bus: EventBus, jobs: list[ScheduledJob] | None = None) -> None:
        self.bus = bus
        self.jobs: list[ScheduledJob] = list(jobs or [])
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def add_job(self, name: str, interval_s: float, kind: str,
                extra: dict[str, Any] | None = None, first_delay_s: float | None = None) -> ScheduledJob:
        job = ScheduledJob(name=name, interval_s=max(1.0, float(interval_s)),
                           action={"kind": kind, **(extra or {})},
                           next_run=time.time() + (first_delay_s if first_delay_s is not None else interval_s))
        with self._lock:
            self.jobs.append(job)
        logger.info("запланирована задача '%s' каждые %.0f с (kind=%s)", name, interval_s, kind)
        return job

    def remove_job(self, name: str) -> bool:
        with self._lock:
            before = len(self.jobs)
            self.jobs = [j for j in self.jobs if j.name != name]
            return len(self.jobs) < before

    def tick(self, now: float | None = None) -> int:
        """Публикует в шину все созревшие задачи. Возвращает число публикаций."""
        t = now if now is not None else time.time()
        fired = 0
        with self._lock:
            due = [j for j in self.jobs if j.enabled and j.next_run <= t]
        for job in due:
            self.bus.emit("cron", {**job.action, "job": job.name}, source="scheduler", priority=6)
            job.runs += 1
            job.next_run = t + job.interval_s
            fired += 1
        return fired

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [{"name": j.name, "interval_s": j.interval_s, "kind": j.action.get("kind"),
                     "runs": j.runs, "next_run": utc_iso(j.next_run), "enabled": j.enabled}
                    for j in self.jobs]

    # ---------- потоковый режим (для тестов/REPL вне основного цикла демона) ----------
    def start_thread(self, poll_s: float = 1.0) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def _loop() -> None:
            while not self._stop.wait(poll_s):
                try:
                    self.tick()
                except Exception:  # noqa: BLE001
                    logger.exception("ошибка scheduler tick")

        self._thread = threading.Thread(target=_loop, name="scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


def default_jobs(config: Any) -> list[tuple[str, float, str]]:
    """Стандартный набор фоновых задач демона (интервалы берутся из конфига daemon)."""
    d = getattr(config, "daemon", None)

    def g(name: str, default: float) -> float:
        return float(getattr(d, name, default))

    return [
        ("autonomous_goal", g("autonomous_interval_s", 900.0), "autonomous"),
        ("sleep_phase", g("sleep_interval_s", 21600.0), "sleep"),
        ("autosave", g("autosave_interval_s", 300.0), "autosave"),
        ("scan_inbox", g("inbox_scan_interval_s", 5.0), "scan_inbox"),
    ]
