"""SQLite-хранилище PFG-Agent: узлы, рёбра, факты, эпизоды, навыки, метрики, события."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from pfg.utils import now_ts

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    vector BLOB NOT NULL,
    bias REAL,
    self_gain REAL,
    created_at INTEGER,
    last_used_at INTEGER,
    usage_count INTEGER,
    metadata TEXT
);
CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    src_node_id TEXT NOT NULL,
    dst_node_id TEXT NOT NULL,
    params BLOB NOT NULL,
    salience REAL,
    usage_count INTEGER,
    active INTEGER,
    frozen INTEGER,
    symbolic_expr TEXT,
    created_at INTEGER,
    last_used_at INTEGER,
    metadata TEXT
);
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    key_vector BLOB,
    value_vector BLOB,
    importance REAL,
    confidence REAL,
    source TEXT,
    created_at INTEGER,
    last_used_at INTEGER
);
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    created_at INTEGER,
    goal TEXT,
    context TEXT,
    actions TEXT,
    result TEXT,
    reward REAL,
    error TEXT,
    lesson TEXT
);
CREATE TABLE IF NOT EXISTS skills (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    code_path TEXT,
    test_path TEXT,
    active INTEGER,
    created_at INTEGER,
    metric_before REAL,
    metric_after REAL,
    metadata TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at INTEGER,
    name TEXT,
    value REAL,
    metadata TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at INTEGER,
    level TEXT,
    source TEXT,
    message TEXT,
    metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_node_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_node_id);
CREATE INDEX IF NOT EXISTS idx_facts_created ON facts(created_at);
-- Внешние источники данных (раздел "Data Ingest"): репутация домена, статистика
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    domain TEXT,
    reputation REAL,
    total_facts INTEGER,
    conflicts INTEGER,
    last_used_at INTEGER
);
CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT,
    status TEXT,
    bytes INTEGER,
    facts_extracted INTEGER,
    trust REAL,
    created_at INTEGER
);
CREATE TABLE IF NOT EXISTS cache_entries (
    url_hash TEXT PRIMARY KEY,
    etag TEXT,
    fetched_at INTEGER,
    payload_path TEXT
);
-- Расширение facts для карантина/доверия: добавляем колонки мягко (migragtion-safe)
"""

# Мягкая миграция: старые БД могут не иметь новых колонок у facts — добавляем при необходимости
_FACTS_MIGRATION = [
    ("trust", "REAL DEFAULT 0.8"),
    ("quarantined", "INTEGER DEFAULT 0"),
    ("suspicious", "INTEGER DEFAULT 0"),
    ("corroborated_by", "TEXT DEFAULT ''"),
    ("last_confirmed_at", "INTEGER DEFAULT 0"),
]


def vec_to_blob(v: np.ndarray) -> bytes:
    """Сериализует float32-вектор в компактный BLOB."""
    return np.asarray(v, dtype=np.float32).tobytes(order="C")


def blob_to_vec(b: bytes | None) -> np.ndarray:
    """Десериализует BLOB обратно в float32-вектор."""
    if b is None:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(bytes(b), dtype=np.float32).copy()


class Storage:
    """Тонкая обёртка над SQLite с batch-friendly интерфейсом."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Мягко добавляет новые колонки в facts, если их нет (совместимость со старыми БД)."""
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(facts)")}
        for name, decl in _FACTS_MIGRATION:
            if name not in cols:
                try:
                    self.conn.execute(f"ALTER TABLE facts ADD COLUMN {name} {decl}")  # noqa: S608 — имена из кода
                except sqlite3.OperationalError:
                    pass

    # ---------- низкоуровневые ----------
    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        cur = self.conn.execute(sql, params)
        cur.row_factory = None
        return cur.fetchall()

    # ---------- high-level ----------
    def upsert_node(self, node: dict[str, Any]) -> None:
        self.execute(
            """INSERT INTO nodes (id, type, vector, bias, self_gain, created_at, last_used_at, usage_count, metadata)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET vector=excluded.vector, bias=excluded.bias,
                 self_gain=excluded.self_gain, last_used_at=excluded.last_used_at,
                 usage_count=excluded.usage_count, metadata=excluded.metadata""",
            (
                node["id"], node["type"], vec_to_blob(node["vector"]), node.get("bias", 0.0),
                node.get("self_gain", 1.0), node.get("created_at", 0), node.get("last_used_at", 0),
                node.get("usage_count", 0), json.dumps(node.get("metadata", {}), ensure_ascii=False),
            ),
        )

    def upsert_edge(self, edge: dict[str, Any]) -> None:
        self.execute(
            """INSERT INTO edges (id, src_node_id, dst_node_id, params, salience, usage_count,
                 active, frozen, symbolic_expr, created_at, last_used_at, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET params=excluded.params, salience=excluded.salience,
                 usage_count=excluded.usage_count, active=excluded.active, frozen=excluded.frozen,
                 symbolic_expr=excluded.symbolic_expr, last_used_at=excluded.last_used_at,
                 metadata=excluded.metadata""",
            (
                edge["id"], edge["src_node_id"], edge["dst_node_id"], edge["params_blob"],
                edge.get("salience", 0.5), edge.get("usage_count", 0), int(edge.get("active", True)),
                int(edge.get("frozen", False)), edge.get("symbolic_expr"),
                edge.get("created_at", 0), edge.get("last_used_at", 0),
                json.dumps(edge.get("metadata", {}), ensure_ascii=False),
            ),
        )

    def delete_edge(self, edge_id: str) -> None:
        self.execute("DELETE FROM edges WHERE id=?", (edge_id,))

    def add_fact(self, fact: dict[str, Any]) -> None:
        self.execute(
            """INSERT INTO facts (id, text, key_vector, value_vector, importance, confidence,
                 source, created_at, last_used_at, trust, quarantined, suspicious,
                 corroborated_by, last_confirmed_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET text=excluded.text, key_vector=excluded.key_vector,
                 value_vector=excluded.value_vector, importance=excluded.importance,
                 confidence=excluded.confidence, last_used_at=excluded.last_used_at,
                 trust=excluded.trust, quarantined=excluded.quarantined,
                 suspicious=excluded.suspicious, corroborated_by=excluded.corroborated_by,
                 last_confirmed_at=excluded.last_confirmed_at""",
            (
                fact["id"], fact["text"], vec_to_blob(fact["key_vector"]),
                vec_to_blob(fact.get("value_vector", np.zeros(0))),
                fact.get("importance", 0.5), fact.get("confidence", 0.8), fact.get("source", "agent"),
                fact.get("created_at", 0), fact.get("last_used_at", 0),
                fact.get("trust", fact.get("confidence", 0.8)),
                int(fact.get("quarantined", False)), int(fact.get("suspicious", False)),
                ",".join(fact.get("corroborated_by", []) or []),
                fact.get("last_confirmed_at", fact.get("created_at", 0)),
            ),
        )

    # ---------- внешние источники / fetch / кэш ----------
    def upsert_source(self, domain: str, reputation: float | None = None,
                      total_facts_delta: int = 0, conflicts_delta: int = 0) -> None:
        row = self.query("SELECT id, reputation, total_facts, conflicts FROM sources WHERE domain=?", (domain,))
        ts = now_ts()
        if row:
            sid, rep, tf, cf = row[0]
            self.execute(
                "UPDATE sources SET reputation=?, total_facts=?, conflicts=?, last_used_at=? WHERE id=?",
                (reputation if reputation is not None else rep,
                 int(tf or 0) + total_facts_delta, int(cf or 0) + conflicts_delta, ts, sid),
            )
        else:
            self.execute(
                "INSERT INTO sources (id, domain, reputation, total_facts, conflicts, last_used_at)"
                " VALUES (?,?,?,?,?,?)",
                (f"src_{domain}", domain, reputation if reputation is not None else 0.5,
                 max(0, total_facts_delta), max(0, conflicts_delta), ts),
            )

    def get_sources(self) -> list[tuple]:
        return self.query("SELECT id, domain, reputation, total_facts, conflicts, last_used_at FROM sources")

    def log_fetch(self, url: str, status: str, nbytes: int, facts_extracted: int, trust: float) -> None:
        self.execute(
            "INSERT INTO fetch_log (url, status, bytes, facts_extracted, trust, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (url[:500], status, nbytes, facts_extracted, trust, now_ts()),
        )

    def fetch_quota_today(self) -> tuple[int, int]:
        """Возвращает (запросов_сегодня, байт_сегодня) по часовой квоте за последние 24ч."""
        since = now_ts() - 86400
        rows = self.query(
            "SELECT COUNT(*), COALESCE(SUM(bytes),0) FROM fetch_log WHERE created_at >= ?", (since,))
        n, b = rows[0]
        hour_rows = self.query(
            "SELECT COUNT(*) FROM fetch_log WHERE created_at >= ?", (now_ts() - 3600,))
        return int(hour_rows[0][0]), int(b or 0)

    def cache_get(self, url_hash: str) -> tuple | None:
        rows = self.query(
            "SELECT url_hash, etag, fetched_at, payload_path FROM cache_entries WHERE url_hash=?", (url_hash,))
        return rows[0] if rows else None

    def cache_put(self, url_hash: str, etag: str | None, payload_path: str | None) -> None:
        self.execute(
            """INSERT INTO cache_entries (url_hash, etag, fetched_at, payload_path) VALUES (?,?,?,?)
               ON CONFLICT(url_hash) DO UPDATE SET etag=excluded.etag,
                 fetched_at=excluded.fetched_at, payload_path=excluded.payload_path""",
            (url_hash, etag or "", now_ts(), payload_path or ""),
        )

    def add_episode(self, ep: dict[str, Any]) -> None:
        self.execute(
            """INSERT OR REPLACE INTO episodes (id, created_at, goal, context, actions, result,
                 reward, error, lesson) VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                ep["id"], ep.get("created_at", 0), ep.get("goal", ""),
                json.dumps(ep.get("context", []), ensure_ascii=False),
                json.dumps(ep.get("actions", []), ensure_ascii=False, default=str),
                json.dumps(ep.get("result", ""), ensure_ascii=False, default=str),
                ep.get("reward", 0.0), ep.get("error"), ep.get("lesson"),
            ),
        )

    def add_skill(self, sk: dict[str, Any]) -> None:
        self.execute(
            """INSERT OR REPLACE INTO skills (id, name, description, code_path, test_path, active,
                 created_at, metric_before, metric_after, metadata) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                sk["id"], sk["name"], sk.get("description", ""), sk.get("code_path"),
                sk.get("test_path"), int(sk.get("active", False)), sk.get("created_at", 0),
                sk.get("metric_before"), sk.get("metric_after"),
                json.dumps(sk.get("metadata", {}), ensure_ascii=False),
            ),
        )

    def log_metric(self, name: str, value: float, ts: int, metadata: dict | None = None) -> None:
        self.execute(
            "INSERT INTO metrics (created_at, name, value, metadata) VALUES (?,?,?,?)",
            (ts, name, float(value), json.dumps(metadata or {}, ensure_ascii=False)),
        )

    def log_event(self, level: str, source: str, message: str, ts: int, metadata: dict | None = None) -> None:
        self.execute(
            "INSERT INTO events (created_at, level, source, message, metadata) VALUES (?,?,?,?,?)",
            (ts, level, source, message, json.dumps(metadata or {}, ensure_ascii=False)),
        )

    def close(self) -> None:
        self.conn.close()
