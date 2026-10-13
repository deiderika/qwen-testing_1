"""Память PFG-Agent: рабочая, семантическая (факты), эпизодическая, процедурная (навыки).

Эмбеддинги — компактный хеш-вектор (hashing trick + n-grams), без внешних моделей.
Релевантность: score = cos(q,k) + a*importance + b*recency + g*usage.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pfg.config import AgentConfig
from pfg.math_basis import cosine_similarity
from pfg.storage import Storage, blob_to_vec
from pfg.utils import new_id, now_ts

logger = logging.getLogger("pfg.memory")

_WORD_RE = re.compile(r"[\w]+", re.UNICODE)


def embed_text(text: str, dim: int) -> np.ndarray:
    """Хеш-эмбеддинг текста: слова + символьные биграммы, L2-нормированный float32."""
    vec = np.zeros(dim, dtype=np.float64)
    text_low = text.lower()
    tokens = _WORD_RE.findall(text_low)
    grams = list(tokens)
    for w in tokens:
        grams.extend(w[i : i + 2] for i in range(max(1, len(w) - 1)))
    for g in grams:
        h = hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if h[4] & 1 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec.astype(np.float32)


@dataclass
class Fact:
    id: str
    text: str
    key_vector: np.ndarray
    value_vector: np.ndarray
    importance: float = 0.5
    confidence: float = 0.8
    source: str = "agent"
    created_at: int = field(default_factory=now_ts)
    last_used_at: int = field(default_factory=now_ts)
    trust: float = 0.8            # доверие к источнику (внешние данные < 1)
    quarantined: bool = False     # факт из недоверенного контура
    suspicious: bool = False      # в исходном тексте найдены приказы/инъекции


@dataclass
class Episode:
    id: str
    goal: str
    context: list[str]
    actions: list[dict[str, Any]]
    result: str
    reward: float
    error: str | None = None
    lesson: str | None = None
    created_at: int = field(default_factory=now_ts)


class Memory:
    """Фасад над SQLite-хранилищем: факты, эпизоды, рабочая память."""

    def __init__(self, config: AgentConfig, storage: Storage) -> None:
        self.config = config
        self.storage = storage
        self.dim = config.graph.node_dim
        # рабочая память (volatile)
        self.current_goal: str = ""
        self.recent_actions: list[dict[str, Any]] = []
        self.working_slots: dict[str, np.ndarray] = {}

    # ---------- факты ----------
    def store_fact(self, text: str, importance: float = 0.6, source: str = "user",
                   confidence: float = 0.9) -> str:
        """Запись факта с дедупликацией по косинусной близости ключей."""
        key = embed_text(text, self.dim)
        dup = self._find_duplicate(key)
        if dup is not None:
            fid, old_text, kv, vv, imp, conf, src, created, used = dup[:9]
            new_imp = min(1.0, imp + 0.1)
            self.storage.add_fact({
                "id": fid, "text": text, "key_vector": blob_to_vec(kv),
                "value_vector": blob_to_vec(vv), "importance": new_imp,
                "confidence": min(1.0, conf + 0.05), "source": src,
                "created_at": created, "last_used_at": now_ts(),
            })
            logger.info("факт %s обновлён (дубликат)", fid)
            return fid
        fid = f"fact_{new_id()}"
        self.storage.add_fact({
            "id": fid, "text": text, "key_vector": key, "value_vector": key,
            "importance": importance, "confidence": confidence, "source": source,
            "created_at": now_ts(), "last_used_at": now_ts(),
        })
        return fid

    def store_external_fact(self, text: str, source: str, trust: float, confidence: float,
                            quarantined: bool = True, suspicious: bool = False,
                            corroborated_by: list[str] | None = None,
                            importance: float = 0.4) -> str:
        """Факт из недоверенного источника: всегда в карантине, доверие = trust.

        Дубль помечается префиксом 'dup:' — пайплайн использует это для статистики,
        при этом last_confirmed_at и corroborated_by обновляются (подтверждение поднимает статус).
        """
        key = embed_text(text, self.dim)
        dup = self._find_duplicate(key)
        corr = [c for c in (corroborated_by or []) if c and c != source]
        if dup is not None:
            row = tuple(dup)
            fid, _t, kv, vv, imp, conf, src, created, used = row[:9]
            extra = self._extra_cols(row[9:])
            # подтверждение тем же смыслом из другого источника => поднимаем trust/confidence
            agree = len(corr) > 0
            new_trust = min(1.0, extra["trust"] + (0.15 if agree else 0.0))
            new_conf = min(1.0, max(conf or 0.5, confidence) + (0.1 if agree else 0.0))
            self.storage.add_fact({
                "id": fid, "text": text, "key_vector": blob_to_vec(kv),
                "value_vector": blob_to_vec(vv), "importance": min(1.0, (imp or 0.4) + 0.05),
                "confidence": new_conf, "source": src, "created_at": created,
                "last_used_at": now_ts(), "trust": new_trust,
                "quarantined": extra["quarantined"] or int(quarantined),
                "suspicious": extra["suspicious"] or int(suspicious),
                "corroborated_by": sorted(set((extra["corroborated_by"], *corr))),
                "last_confirmed_at": now_ts(),
            })
            return f"dup:{fid}"
        fid = f"fact_{new_id()}"
        self.storage.add_fact({
            "id": fid, "text": text, "key_vector": key, "value_vector": key,
            "importance": importance, "confidence": confidence, "source": source,
            "created_at": now_ts(), "last_used_at": now_ts(),
            "trust": trust, "quarantined": int(quarantined), "suspicious": int(suspicious),
            "corroborated_by": corr, "last_confirmed_at": now_ts(),
        })
        return fid

    @staticmethod
    def _extra_cols(rest: tuple) -> dict[str, Any]:
        """Достаёт колонки trust/quarantined/suspicious/corroborated_by/last_confirmed_at из строки facts."""
        keys = ("trust", "quarantined", "suspicious", "corroborated_by", "last_confirmed_at")
        out: dict[str, Any] = {"trust": 0.8, "quarantined": 0, "suspicious": 0,
                               "corroborated_by": "", "last_confirmed_at": 0}
        for i, k in enumerate(keys):
            if i < len(rest) and rest[i] is not None:
                out[k] = rest[i]
        if isinstance(out["corroborated_by"], str):
            out["corroborated_by"] = [x for x in out["corroborated_by"].split(",") if x]
        return out

    def _find_duplicate(self, key: np.ndarray, threshold: float = 0.97) -> tuple | None:
        for row in self.storage.query("SELECT * FROM facts"):
            if cosine_similarity(key, blob_to_vec(row[2])) >= threshold:
                return row
        return None

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[Fact, float]]:
        """Поиск фактов: cos + alpha*importance + beta*recency + gamma*usage.

        Карантинные/подозрительные внешние факты получают штраф к score —
        они видны агенту как сведения с пометкой, но не вытесняют собственные знания.
        """
        mcfg = self.config.memory
        top_k = top_k or mcfg.retrieval_top_k
        q = embed_text(query, self.dim)
        now = now_ts()
        scored: list[tuple[float, Fact]] = []
        for row in self.storage.query("SELECT * FROM facts"):
            fid, text, kv, vv, imp, conf, src, created, used = row[:9]
            extra = self._extra_cols(row[9:])
            kvec = blob_to_vec(kv)
            age = max(0, now - int(created or 0))
            recency = math.exp(-mcfg.recency_lambda * age)
            usage = min(1.0, 0.1 * math.log1p(int(used or 0) + 1))
            score = (
                cosine_similarity(q, kvec)
                + mcfg.w_importance * float(imp or 0.5)
                + mcfg.w_recency * recency
                + mcfg.w_usage * usage
            )
            if extra["quarantined"]:
                score -= 0.15
            if extra["suspicious"]:
                score -= 0.25
            fact = Fact(id=fid, text=text, key_vector=kvec, value_vector=blob_to_vec(vv),
                        importance=float(imp or 0.5), confidence=float(conf or 0.8),
                        source=src or "agent", created_at=int(created or 0),
                        last_used_at=int(used or 0),
                        trust=float(extra["trust"]), quarantined=bool(extra["quarantined"]),
                        suspicious=bool(extra["suspicious"]))
            scored.append((score, fact))
        scored.sort(key=lambda t: t[0], reverse=True)
        out = scored[:top_k]
        for _, f in out:
            self.storage.execute("UPDATE facts SET last_used_at=? WHERE id=?", (now_ts(), f.id))
        return [(f, s) for s, f in out]

    def all_facts(self) -> list[Fact]:
        rows = self.storage.query("SELECT * FROM facts ORDER BY created_at")
        out: list[Fact] = []
        for r in rows:
            extra = self._extra_cols(r[9:])
            out.append(Fact(id=r[0], text=r[1], key_vector=blob_to_vec(r[2]), value_vector=blob_to_vec(r[3]),
                            importance=float(r[4] or 0), confidence=float(r[5] or 0), source=r[6] or "",
                            created_at=int(r[7] or 0), last_used_at=int(r[8] or 0),
                            trust=float(extra["trust"]), quarantined=bool(extra["quarantined"]),
                            suspicious=bool(extra["suspicious"])))
        return out

    # ---------- эпизоды ----------
    def store_episode(self, ep: Episode) -> str:
        self.storage.add_episode({
            "id": ep.id, "created_at": ep.created_at, "goal": ep.goal,
            "context": ep.context, "actions": ep.actions, "result": ep.result,
            "reward": ep.reward, "error": ep.error, "lesson": ep.lesson,
        })
        self._trim("episodes", self.config.memory.max_episodes)
        return ep.id

    def recent_episodes(self, limit: int = 20) -> list[dict[str, Any]]:
        import json
        rows = self.storage.query(
            "SELECT id, created_at, goal, context, actions, result, reward, error, lesson"
            " FROM episodes ORDER BY created_at DESC LIMIT ?", (limit,))
        return [
            {"id": r[0], "created_at": r[1], "goal": r[2], "context": json.loads(r[3] or "[]"),
             "actions": json.loads(r[4] or "[]"), "result": json.loads(r[5] or '""'),
             "reward": r[6], "error": r[7], "lesson": r[8]}
            for r in rows
        ]

    # ---------- уроки ----------
    def store_lesson(self, text: str, importance: float = 0.7) -> str:
        return self.store_fact(f"УРОК: {text}", importance=importance, source="reflection")

    def _trim(self, table: str, max_rows: int) -> None:
        cnt = self.storage.query(f"SELECT COUNT(*) FROM {table}")[0][0]  # noqa: S608 — таблица из белого списка кода
        if cnt > max_rows:
            self.storage.execute(
                f"DELETE FROM {table} WHERE rowid NOT IN ("  # noqa: S608
                f" SELECT rowid FROM {table} ORDER BY created_at DESC LIMIT ?)", (max_rows,),
            )

    # ---------- рабочая память ----------
    def set_goal(self, goal: str) -> None:
        self.current_goal = goal

    def push_action(self, action_record: dict[str, Any]) -> None:
        self.recent_actions.append(action_record)
        self.recent_actions = self.recent_actions[-16:]

    def vector_for(self, node_id: str, text: str) -> np.ndarray:
        """Внешний вклад M_i: эмбеддинг текста для узла памяти."""
        v = embed_text(text, self.dim)
        self.working_slots[node_id] = v
        return v
