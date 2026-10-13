"""Рефлексия PFG-Agent: анализ ошибок, уроки, эволюция важности рёбер.

Запускается после каждой задачи (или командой `reflect`). Анализирует эпизоды:
ошибки, низкую уверенность, повторяющиеся неудачи, длинные планы. Результаты:
уроки в память, +/- salience рёбер, предложения новых рёбер/узлов/навыков.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from pfg.config import AgentConfig
from pfg.graph import PlasticFunctionGraph
from pfg.memory import Memory
from pfg.utils import JsonlLogger, new_id

logger = logging.getLogger("pfg.reflection")


@dataclass
class Lesson:
    source_episode_id: str
    text: str
    importance: float = 0.7
    suggested_changes: list[dict[str, Any]] = field(default_factory=list)
    lesson_id: str = field(default_factory=new_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.lesson_id, "episode": self.source_episode_id,
            "text": self.text, "importance": self.importance,
            "suggested_changes": self.suggested_changes,
        }


class ReflectionEngine:
    """Извлекает уроки из опыта и мягко перестраивает граф под них."""

    def __init__(self, config: AgentConfig, graph: PlasticFunctionGraph, memory: Memory) -> None:
        self.config = config
        self.graph = graph
        self.memory = memory
        self.log = JsonlLogger(config.log_dir)
        self.last_lessons: list[Lesson] = []
        self._goal_fail_counter: Counter[str] = Counter()

    # ---------- основной вход ----------
    def reflect(self, goal: str = "", plan: Any = None, results: Any = None,
                observation: str = "") -> list[Lesson]:
        """Рефлексия по завершённой задаче + по последним эпизодам. Возвращает созданные уроки."""
        lessons: list[Lesson] = []
        success = bool(results) and all(getattr(r, "success", True) for r in results)
        errors = [getattr(r, "error", None) for r in (results or []) if getattr(r, "error", None)]

        # 1. Ошибка исполнения -> урок + понижение salience у последних активных рёбер пути
        if errors:
            lessons.append(Lesson(
                source_episode_id=goal[:40],
                text=f"Действие по цели '{goal[:60]}' завершилось с ошибкой: {errors[0][:200]}. "
                     f"В следующий раз проверить аргументы заранее.",
                importance=0.8,
                suggested_changes=[{"type": "lower_salience", "reason": "failed_action"}],
            ))
            self._adjust_salience(delta=-0.05)
            self._goal_fail_counter[goal[:40]] += 1
        elif success and results:
            # успех — укрепляем часто используемые рёбра (закрепление успешного подграфа)
            self._adjust_salience(delta=+0.02)

        # 2. Низкая уверенность плана
        if plan is not None and getattr(plan, "confidence", 1.0) < 0.35:
            lessons.append(Lesson(
                source_episode_id=goal[:40],
                text=f"План для '{goal[:60]}' имел низкую уверенность "
                     f"({plan.confidence:.2f}); стоит уточнять цель у пользователя.",
                importance=0.6,
                suggested_changes=[{"type": "add_edge", "hint": "goal->ask_user"}],
            ))

        # 3. Слишком длинный план
        if plan is not None and getattr(plan, "length", 0) > 6:
            lessons.append(Lesson(
                source_episode_id=goal[:40],
                text="План длиннее 6 шагов — candidate на компиляцию навыка для сокращения.",
                importance=0.7,
                suggested_changes=[{"type": "create_skill", "reason": "long_plan"}],
            ))

        # 4. Повторяющиеся неудачи одной цели
        repeated = [g for g, c in self._goal_fail_counter.items() if c >= 2]
        for g in repeated:
            lessons.append(Lesson(
                source_episode_id=g,
                text=f"Цель '{g}' проваливалась {self._goal_fail_counter[g]} раза — нужен новый подход или навык.",
                importance=0.9,
                suggested_changes=[{"type": "propose_node", "hint": "alternative_strategy"}],
            ))
            self._goal_fail_counter[g] = 0  # сброс, чтобы не плодить уроки

        # 5. Рутинный анализ прошлых эпизодов
        lessons.extend(self._scan_episodes())

        for lesson in lessons:
            self.memory.store_lesson(lesson.text, importance=lesson.importance)
            self.log.write("learning.jsonl", {
                "level": "INFO", "source": "reflection", "message": "lesson stored",
                "data": lesson.to_dict(),
            })
        self.last_lessons = lessons
        if lessons:
            logger.info("рефлексия: %d уроков", len(lessons))
        return lessons

    # ---------- вспомогательное ----------
    def _scan_episodes(self) -> list[Lesson]:
        """Ищет в недавних эпизодах незакрытые ошибки без урока."""
        out: list[Lesson] = []
        for ep in self.memory.recent_episodes(limit=10):
            if ep.get("error") and not ep.get("lesson"):
                out.append(Lesson(
                    source_episode_id=ep["id"],
                    text=f"Эпизод '{str(ep.get('goal',''))[:50]}' завершился ошибкой "
                         f"'{str(ep['error'])[:120]}' — урок записан автоматически.",
                    importance=0.65,
                ))
                self.storage_set_lesson(ep["id"], out[-1].text)
        return out

    def storage_set_lesson(self, episode_id: str, lesson_text: str) -> None:
        try:
            self.memory.storage.execute(
                "UPDATE episodes SET lesson=? WHERE id=?", (lesson_text, episode_id))
        except Exception:  # noqa: BLE001
            logger.warning("не удалось проставить урок эпизоду %s", episode_id)

    def _adjust_salience(self, delta: float) -> None:
        """Мягкое усиление/ослабление самых используемых рёбер (процедурное закрепление)."""
        edges = sorted(self.graph.edges.values(), key=lambda e: e.usage_count, reverse=True)
        for e in edges[: max(1, len(edges) // 10)]:
            self.graph.update_salience(e.id, delta)

    def suggest_structure_growth(self) -> list[dict[str, Any]]:
        """Предложения новых рёбер: пары узлов с высокой совместной активностью, но слабыми связями."""
        suggestions: list[dict[str, Any]] = []
        threshold = self.config.learning.edge_add_threshold
        coact = self.graph.coactivation_counts()
        linked = {(e.src_node_id, e.dst_node_id) for e in self.graph.edges.values()}
        for key, count in sorted(coact.items(), key=lambda kv: -kv[1])[:20]:
            src, dst = key.split("|")
            if (dst, src) in linked or count < 5:
                continue
            strength = min(1.0, count / 100.0)
            if strength >= threshold:
                suggestions.append({"type": "add_edge", "src": dst, "dst": src,
                                    "strength": round(strength, 3)})
        return suggestions[:8]

    def apply_growth(self) -> int:
        """Безопасно применяет предложения роста (только add_edge, лимит max_edges учитывается графом)."""
        added = 0
        for s in self.suggest_structure_growth():
            try:
                self.graph.add_edge(s["src"], s["dst"])
                added += 1
            except (KeyError, RuntimeError):
                continue
        if added:
            self.log.write("learning.jsonl", {
                "level": "INFO", "source": "reflection",
                "message": "graph grew", "data": {"added_edges": added},
            })
        return added
