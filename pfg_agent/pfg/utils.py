"""Утилиты: идентификаторы, время, JSONL-логи."""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def new_id(prefix: str = "") -> str:
    """Генерирует короткий уникальный идентификатор."""
    uid = uuid.uuid4().hex[:12]
    return f"{prefix}{uid}" if prefix else uid


def now_ts() -> int:
    """Текущее unix-время в секундах."""
    return int(time.time())


def now_iso() -> str:
    """Текущее время в ISO-8601 (UTC)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_iso(ts: float | None = None) -> str:
    """ISO-строка из unix-времени (UTC)."""
    t = time.time() if ts is None else ts
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class JsonlLogger:
    """Приложение JSON-событий в JSONL-файл. Безопасно при отсутствии прав на запись."""

    def __init__(self, log_dir: str | Path = "logs", enabled: bool = True) -> None:
        self.log_dir = Path(log_dir)
        self.enabled = enabled
        if self.enabled:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.enabled = False

    def write(self, filename: str, record: dict[str, Any]) -> None:
        """Добавляет одну запись в JSONL-файл; ошибки логов не должны ронять агент."""
        if not self.enabled:
            return
        record = {"ts": utc_iso(), **record}
        try:
            with (self.log_dir / filename).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass


def dump_json(obj: Any, path: str | Path) -> None:
    """Сохраняет объект в JSON-файл с human-readable форматированием."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    """Читает JSON-файл."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
