"""Внутренняя мотивация PFG-Agent (Intrinsic Motivation).

Формула приоритета внутренней цели:

    Priority = w_curiosity * Curiosity + w_need * Need + w_profit * Profit

* Curiosity — «любопытство»: ошибки предсказания графа, накопленные уроки,
  нерешённые аномалии в логах/эпизодах, неисследованные концепты.
* Need      — «потребность»: нехватка ресурсов (место на диске, размер БД,
  накопившиеся незакрытые эпизоды, устаревший граф без сохранения).
* Profit    — «выгода»: ожидаемая польза от внешней задачи из очереди
  (pending-задачи task_queue) или от оптимизации (компиляция навыков ускоряет граф).

Модуль чистый: никаких побочных эффектов, только чтение состояния и генерация
строковых целей для агента. Безопасность: пороги и веса берутся из конфига;
агент никогда не получает цель вида "удали/скачай/запусти shell" — только
разрешённые операции (reflect/evolve/recall/изучение sandbox).
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("pfg.motivation")


@dataclass
class Drive:
    """Одна внутренняя потребность с текущей интенсивностью 0..1."""
    name: str
    intensity: float
    reason: str


@dataclass
class InternalGoal:
    """Сгенерированная агентом внутренняя цель."""
    goal: str
    priority: float
    drives: list[Drive] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "priority": round(self.priority, 4),
            "drives": [{"name": d.name, "intensity": round(d.intensity, 3),
                        "reason": d.reason} for d in self.drives],
        }


class MotivationEngine:
    """Генератор внутренних целей на основе состояния системы."""

    def __init__(self, config: Any) -> None:
        self.config = config
        m = getattr(config, "motivation", None)
        # веса формулы; при отсутствии секции — разумные значения по умолчанию
        self.w_curiosity = float(getattr(m, "w_curiosity", 0.4))
        self.w_need = float(getattr(m, "w_need", 0.35))
        self.w_profit = float(getattr(m, "w_profit", 0.25))
        self.min_priority = float(getattr(m, "min_priority", 0.15))
        self.disk_free_min_gb = float(getattr(m, "disk_free_min_gb", 0.5))
        self.max_db_mb = float(getattr(m, "max_db_mb", 200))
        self._last_goals: list[str] = []   # антизацикливание: не повторять одно и то же

    # ---------- оценка состояний (Curiosity / Need / Profit) ----------
    def curiosity(self, agent_status: dict[str, Any]) -> Drive:
        """Любопытство: много уроков/ошибок/необработанных эпизодов => хочется исследовать."""
        lessons = int(agent_status.get("recent_lessons", 0))
        episodes = int(agent_status.get("episodes", 0))
        violations = int(agent_status.get("safety_violations", 0))
        edges = int(agent_status.get("graph", {}).get("edges", 0))
        nodes = max(1, int(agent_status.get("graph", {}).get("nodes", 1)))
        load_factor = min(1.0, edges / max(1, nodes * 8))  # недоисследованный граф
        raw = 0.1 * lessons + 0.01 * violations * 5 + 0.05 * min(episodes, 20) \
              + 0.3 * (1.0 - load_factor)
        intensity = min(1.0, raw)
        reason = f"уроки={lessons}, эпизоды={episodes}, нарушения={violations}, плотность графа={load_factor:.2f}"
        return Drive("curiosity", intensity, reason)

    def need(self, disk_free_gb: float, db_size_mb: float, unsaved_steps: int) -> Drive:
        """Потребность: ресурсы на исходе => надо оптимизироваться/консолидироваться."""
        parts: list[str] = []
        score = 0.0
        if disk_free_gb < self.disk_free_min_gb:
            score += 0.6
            parts.append(f"мало места ({disk_free_gb:.2f} ГБ)")
        if db_size_mb > self.max_db_mb:
            score += 0.4
            parts.append(f"БД большая ({db_size_mb:.0f} МБ)")
        if unsaved_steps > 500:
            score += 0.2
            parts.append("граф давно не сохранялся")
        return Drive("need", min(1.0, score), "; ".join(parts) or "ресурсы в норме")

    def profit(self, pending_tasks: int, idle_skills_candidates: int) -> Drive:
        """Выгода: есть внешние задачи или явные кандидаты навыка => работа окупается."""
        score = min(1.0, 0.35 * pending_tasks + 0.25 * idle_skills_candidates)
        reason = f"задач в очереди={pending_tasks}, кандидатов навыков={idle_skills_candidates}"
        return Drive("profit", score, reason)

    # ---------- выбор цели ----------
    def _candidates(self, drives: list[Drive], agent_status: dict[str, Any]) -> list[tuple[float, str]]:
        """Превращает набор потребностей в конкретные безопасные цели."""
        cur = next((d for d in drives if d.name == "curiosity"), None)
        nd = next((d for d in drives if d.name == "need"), None)
        pr = next((d for d in drives if d.name == "profit"), None)
        out: list[tuple[float, str]] = []
        if cur and cur.intensity > 0.2:
            out.append((self.w_curiosity * cur.intensity,
                        "Проанализируй последние ошибки и уроки и обобщи их в память"))
            out.append((self.w_curiosity * cur.intensity * 0.8,
                        "Изучи часто активируемые связи графа и предложи новую ветку знаний"))
        if nd and nd.intensity > 0.3:
            out.append((self.w_need * nd.intensity,
                        "Выполни ночную консолидацию: сожми память и упорядочь навыки"))
        if pr and pr.intensity > 0.1:
            out.append((self.w_profit * pr.intensity,
                        "Проверь очередь задач и возьми самую важную pending задачу"))
            out.append((self.w_profit * pr.intensity * 0.7,
                        "Скомпилируй повторяющийся подграф в быстрый навык"))
        # фоновая гигиена всегда немного интересна
        out.append((0.05, "Обнови важность связей графа на основе последних эпизодов"))
        return out

    def generate_internal_goal(self, agent_status: dict[str, Any],
                               pending_tasks: int = 0,
                               skill_candidates: int = 0,
                               disk_free_gb: float = 999.0,
                               db_size_mb: float = 0.0,
                               unsaved_steps: int = 0) -> InternalGoal | None:
        """Возвращает лучшую внутреннюю цель или None, если делать нечего."""
        drives = [
            self.curiosity(agent_status),
            self.need(disk_free_gb, db_size_mb, unsaved_steps),
            self.profit(pending_tasks, skill_candidates),
        ]
        best_score, best_goal = 0.0, ""
        for score, goal in self._candidates(drives, agent_status):
            if goal in self._last_goals[-5:]:
                score *= 0.2  # штраф за повтор — стимул разнообразить поведение
            if score > best_score:
                best_score, best_goal = score, goal
        if not best_goal or best_score < self.min_priority:
            logger.debug("мотивация: целей нет (best=%.3f < %.3f)", best_score, self.min_priority)
            return None
        self._last_goals.append(best_goal)
        self._last_goals = self._last_goals[-20:]
        goal_obj = InternalGoal(goal=best_goal, priority=best_score, drives=drives)
        logger.info("внутренняя цель (p=%.2f): %s", best_score, best_goal)
        return goal_obj


def get_disk_free_gb(path: Any) -> float:
    """Свободное место на разделе пути проекта (для оценки Need)."""
    try:
        usage = shutil.disk_usage(str(path))
        return usage.free / (1024 ** 3)
    except OSError:
        return 999.0
