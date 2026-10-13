"""Curiosity-контур: агент сам решает, ЧТО именно ему изучить (информационный пробел).

Формула приоритета запроса:

    Priority(q) = Gap(q) * Relevance(q) * Timeliness(q) / Cost(q)

  Gap(q)        = 1 - PredictionConfidence(q)   — чем хуже предсказание, тем больше пробел
  Relevance(q)  = cos(q, GoalVector)            — связь с текущей целью
  Timeliness(q) = 1 если данные устарели, иначе 0.3
  Cost(q)       = λ1*bytes + λ2*latency + parse_difficulty

Важно: модуль порождает только ГИПОТЕЗЫ запросов (topic + candidate URL из белого списка)
и никогда не выполняет действия сам. Решение о fetch принимает планировщик через SafetyGuard
(действие fetch_data имеет risk=high и требует подтверждения человека).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pfg.config import AgentConfig
from pfg.math_basis import cosine_similarity
from pfg.memory import Memory, embed_text

logger = logging.getLogger("pfg.curiosity")


@dataclass
class InfoGap:
    """Один информационный пробел — кандидат на изучение."""
    topic: str
    node_id: str | None
    gap: float           # 1 - confidence
    relevance: float     # cos к цели
    timeliness: float    # 1.0 устарело / 0.3 свежо
    cost: float          # ожидаемая цена запроса (>0)
    priority: float      # итог формулы
    suggested_url: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"topic": self.topic, "node": self.node_id, "gap": round(self.gap, 3),
                "relevance": round(self.relevance, 3), "timeliness": round(self.timeliness, 2),
                "cost": round(self.cost, 4), "priority": round(self.priority, 3),
                "suggested_url": self.suggested_url, "reason": self.reason}


@dataclass
class CuriosityReport:
    goal: str
    gaps: list[InfoGap] = field(default_factory=list)
    network_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"goal": self.goal, "network_enabled": self.network_enabled,
                "gaps": [g.to_dict() for g in self.gaps]}


class CuriosityEngine:
    """Находит узлы/факты с максимальной ошибкой предсказания и предлагает что изучить."""

    def __init__(self, config: AgentConfig, memory: Memory, graph: Any = None,
                 storage: Any = None) -> None:
        self.config = config
        self.ccfg = config.curiosity
        self.memory = memory
        self.graph = graph
        self.storage = storage
        self.dim = config.graph.node_dim

    # ---------- оценка уверенности предсказания для узла ----------
    def node_confidence(self, node_id: str) -> float:
        """Эвристика: насколько граф «уверен» в состоянии узла.

        Используется норма активации последнего состояния (чем ближе |x| к насыщению tanh,
        тем увереннее) плюс usage_count как признак изученности. Если графа нет — 0.5.
        """
        if self.graph is None:
            return 0.5
        state = getattr(self.graph, "last_states", {}).get(node_id)
        if state is None or not len(state):
            return 0.5
        sat = float(np.mean(np.tanh(np.abs(state))))          # 0..~0.76
        node = self.graph.nodes.get(node_id)
        usage = min(1.0, 0.1 * float(np.log1p(node.usage_count if node else 0)))
        return float(min(1.0, 0.7 * sat + 0.3 * usage))

    def estimate_cost(self, est_bytes: int = 20_000, est_latency_s: float = 2.0) -> float:
        c = (self.ccfg.lambda_bytes * est_bytes
             + self.ccfg.lambda_latency * est_latency_s
             + self.ccfg.parse_difficulty)
        return max(0.05, c)

    # ---------- генерация пробелов ----------
    def find_gaps(self, goal: str, top_n: int | None = None) -> list[InfoGap]:
        """Возвращает отсортированные по Priority пробелы (не ниже gap_threshold)."""
        top_n = top_n or self.ccfg.max_goals_per_cycle
        goal_vec = embed_text(goal, self.dim)
        gaps: list[InfoGap] = []
        now = __import__("time").time()
        # 1. По узлам графа: низкая уверенность + релевантность цели
        if self.graph is not None:
            for nid, node in self.graph.nodes.items():
                conf = self.node_confidence(nid)
                gap = 1.0 - conf
                if gap < self.ccfg.gap_threshold:
                    continue
                rel = max(0.0, cosine_similarity(node.vector, goal_vec))
                tl = 1.0  # состояние узла всегда «про актуальность модели мира»
                cost = self.estimate_cost()
                prio = gap * max(rel, 0.05) * tl / cost
                gaps.append(InfoGap(topic=f"Уточнить состояние концепта '{nid}'",
                                    node_id=nid, gap=gap, relevance=rel, timeliness=tl,
                                    cost=cost, priority=prio,
                                    reason=f"confidence={conf:.2f}"))
        # 2. По старым внешним фактам: устаревшие данные => Timeliness=1
        stale_after = self.ccfg.stale_after_s
        for f in self.memory.all_facts():
            if not f.quarantined:
                continue
            age = max(0.0, now - f.created_at)
            tl = 1.0 if age > stale_after else 0.3
            conf = max(f.confidence, f.trust)
            gap = 1.0 - conf
            if gap < self.ccfg.gap_threshold and tl < 1.0:
                continue
            rel = max(0.0, cosine_similarity(embed_text(f.text, self.dim), goal_vec))
            cost = self.estimate_cost()
            prio = gap * max(rel, 0.05) * tl / cost
            topic = f"Обновить внешний факт: {f.text[:80]}"
            url = self._suggest_url(f.source)
            gaps.append(InfoGap(topic=topic, node_id=None, gap=gap, relevance=rel,
                                timeliness=tl, cost=cost, priority=prio, suggested_url=url,
                                reason=f"trust={conf:.2f}, age={int(age)}s"))
        gaps.sort(key=lambda g: g.priority, reverse=True)
        # фильтруем откровенно мусорные (< порога) и берём топ-N
        return [g for g in gaps if g.priority > 0.0][:top_n]

    def _suggest_url(self, source: str) -> str | None:
        """Candidate URL строго из белого списка; никаких произвольных доменов."""
        allowed = self.config.network.allowed_domains
        if not allowed:
            return None
        m = re.match(r"https?://([^/]+)", source or "")
        if m and any(m.group(1) == d or m.group(1).endswith("." + d) for d in allowed):
            return source
        return f"https://{allowed[0]}"  # первый разрешённый домен как точка поиска

    def report(self, goal: str, top_n: int | None = None) -> CuriosityReport:
        """Собрать полный отчёт о пробелах для цели (используется CLI-командой curiosity)."""
        gaps = self.find_gaps(goal, top_n=top_n)
        return CuriosityReport(goal=goal, gaps=gaps,
                               network_enabled=self.config.network.enabled)

    # ---------- безопасная интеграция с мотивацией ----------
    def motivation_goals(self, report: CuriosityReport) -> list[str]:
        """Формулирует цели так, чтобы они НЕ содержали приказов и сетевых действий.

        Если сеть выключена — цель всегда локальная («переформулировать пробел как
        внутреннюю задачу рефлексии»). Внешний fetch возможен только через явное
        действие с подтверждением человека.
        """
        out: list[str] = []
        net_on = self.config.network.enabled
        for g in report.gaps:
            if net_on and g.suggested_url:
                out.append(f"Подготовить план обновления знаний по пробелу: {g.topic}"
                           f" (кандидат источника — только из белого списка)")
            else:
                out.append(f"Локально переосмыслить пробел: {g.topic}"
                           " (сеть выключена — опереться на память и self-play)")
        return out[: self.ccfg.max_goals_per_cycle]
