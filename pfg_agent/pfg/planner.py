"""Планировщик PFG-Agent: парсинг цели, симуляция действий через граф, выбор лучшего.

План строится из базовых действий (раздел 15.1). Для каждого кандидата граф
прогоняется на simulation_horizon шагов; считываются узлы reward/risk/cost/
success_probability и вычисляется score = w_r*reward - w_risk*risk - w_c*cost.
При близких оценках выбирается более безопасное, затем более дешёвое действие.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pfg.config import AgentConfig
from pfg.graph import PlasticFunctionGraph
from pfg.math_basis import sigmoid
from pfg.safety import Action
from pfg.utils import new_id

logger = logging.getLogger("pfg.planner")

BASE_ACTIONS = [
    "remember", "recall", "search_memory", "read_file", "write_file",
    "list_files", "calculate", "run_skill", "create_skill", "ask_user", "wait",
]


@dataclass
class PlanStep:
    action: Action
    expected_reward: float = 0.0
    expected_risk: float = 0.0
    expected_cost: float = 0.0
    success_probability: float = 0.5
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.action.to_dict(),
            "expected": {
                "reward": round(self.expected_reward, 4),
                "risk": round(self.expected_risk, 4),
                "cost": round(self.expected_cost, 4),
                "success_probability": round(self.success_probability, 4),
            },
            "score": round(self.score, 4),
        }


@dataclass
class Plan:
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    expected_benefit: float = 0.0
    risk: float = 0.0
    cost: float = 0.0
    confidence: float = 0.5
    plan_id: str = field(default_factory=new_id)

    @property
    def length(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.plan_id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "expected_benefit": round(self.expected_benefit, 4),
            "risk": round(self.risk, 4),
            "cost": round(self.cost, 4),
            "confidence": round(self.confidence, 4),
            "alternatives": self.alternatives,
        }


# ---------- разбор цели ----------
_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
_OP_WORDS = {
    "прибавь": "+", "плюс": "+", "add": "+", "сложи": "+", "сумма": "+",
    "вычти": "-", "минус": "-", "subtract": "-",
    "умножь": "*", "умножить": "*", "умножение": "*", "multiply": "*", "на сколько раз больше": "*",
    "подели": "/", "раздели": "/", "divide": "/", "поделить": "/",
    "+": "+", "-": "-", "*": "*", "/": "/", "x": "*", ":": "/",
}


def _detect_op(g: str) -> str | None:
    """Определяет операцию по словам (длинные слова раньше коротких) или символам."""
    for word in sorted(_OP_WORDS, key=len, reverse=True):
        if word.isalpha() and word in g:
            return _OP_WORDS[word]
    for ch in "+-*/":
        if ch in g:
            return {"+": "+", "-": "-", "*": "*", "/": "/"}[ch]
    return None


def parse_goal(goal: str) -> dict[str, Any]:
    """Лёгкий rule-based парсер намерений (замена облачному LLM)."""
    g = goal.lower().strip()
    intent = "chat"
    args: dict[str, Any] = {}

    if re.search(r"(запомни|сохрани факт|не забудь|помни что)", g):
        intent = "remember"
        m = re.search(r"(?:что|:)\s*(.+)$", goal, re.IGNORECASE | re.DOTALL)
        args["text"] = (m.group(1) if m else goal).strip().rstrip(".")
    elif any(op in g for op in _OP_WORDS) and len(_NUM_RE.findall(g)) >= 2:
        nums = [float(n.replace(",", ".")) for n in _NUM_RE.findall(g)]
        op = "+"
        for word, symbol in _OP_WORDS.items():
            if word in g:
                op = symbol
                break
        intent = "calculate"
        args.update({"a": nums[0], "b": nums[1], "op": op})
    elif re.search(
        r"(какой|какая|какое|какие|когда|где|сколько|кто|почему|зачем|как часто|"
        r"что ты знаешь|вспомни|напомни|помнишь)",
        g,
    ):
        intent = "recall"
        args["query"] = goal.strip()
    elif re.search(r"(создай файл|запиши|сохрани файл|напиши в файл)", g):
        intent = "write_file"
        m_path = re.search(r"(sandbox[\w./\\-]*|\S+\.\w+)", goal)
        m_content = re.search(r"(?:текстом|содержимым|:)\s*[\"']?([^\"'\n]+)[\"']?\s*$", goal)
        args["path"] = m_path.group(1) if m_path else "sandbox/output.txt"
        args["content"] = m_content.group(1).strip() if m_content else goal.strip()
    elif re.search(r"(прочитай|открой|покажи содержимое|что в файле)", g):
        intent = "read_file"
        m_path = re.search(r"(sandbox[\w./\\-]*|\S+\.\w+)", goal)
        args["path"] = m_path.group(1) if m_path else ""
    elif re.search(r"(список файлов|ls|покажи файлы|какие файлы)", g):
        intent = "list_files"
        args["path"] = "sandbox"
    elif re.search(r"(сколько шагов|дойди|сетк|grid|путь к цели)", g):
        intent = "gridworld"
    return {"intent": intent, "args": args, "raw": goal}


class Planner:
    """Строит план: кандидатов оценивает симуляцией графа, выбирает лучших."""

    def __init__(self, config: AgentConfig, graph: PlasticFunctionGraph) -> None:
        self.config = config
        self.pcfg = config.planner
        self.graph = graph
        self.dim = config.graph.node_dim
        # узлы-счётчики для симуляции
        self.reward_node = graph.add_node("reward", "plan_reward")
        self.risk_node = graph.add_node("risk", "plan_risk")
        self.cost_node = graph.add_node("abstract", "plan_cost")
        self.succ_node = graph.add_node("concept", "plan_success_prob")
        self._action_vecs: dict[str, np.ndarray] = {}

    def action_vector(self, action_name: str) -> np.ndarray:
        """Детерминированный вектор действия из хеша имени (стабилен между запусками)."""
        if action_name not in self._action_vecs:
            rng = np.random.default_rng(abs(hash(action_name)) % (2**32))
            self._action_vecs[action_name] = (rng.standard_normal(self.dim) * 0.5).astype(np.float32)
        return self._action_vecs[action_name]

    # ---------- симуляция ----------
    def simulate_action(self, action_name: str) -> dict[str, float]:
        """Прогон графа с действием на входе; читает reward/risk/cost/success."""
        horizon = self.pcfg.simulation_horizon
        vec = self.action_vector(action_name)
        sensor = next((n for n in self.graph.nodes.values() if n.type == "sensor"), None)
        inputs: dict[str, np.ndarray] = {}
        if sensor is not None:
            inputs[sensor.id] = vec
        traj = self.graph.simulate(inputs, horizon)
        last = traj[-1] if traj else {}

        def readout(node_id: str) -> float:
            v = last.get(node_id)
            if v is None:
                node = self.graph.nodes.get(node_id)
                v = node.vector if node is not None else np.zeros(self.dim)
            return float(sigmoid(np.asarray(v, dtype=np.float32).mean()))

        out = {
            "reward": readout(self.reward_node),
            "risk": readout(self.risk_node),
            "cost": readout(self.cost_node),
            "success_probability": readout(self.succ_node),
        }
        # эвристика: неизвестные опасные категории не должны выигрывать симуляцию
        if action_name in ("network_request", "shell_command", "money_transfer"):
            out["risk"] = max(out["risk"], 0.9)
        return out

    def score_action(self, action_name: str, sim: dict[str, float]) -> tuple[float, float]:
        """score = w_r*reward - w_risk*risk - w_c*cost; tie-break пары (безопасность, стоимость)."""
        w = self.pcfg
        score = w.reward_weight * sim["reward"] - w.risk_weight * sim["risk"] - w.cost_weight * sim["cost"]
        return score, sim["success_probability"]

    # ---------- построение плана ----------
    def make_plan(self, state: dict[str, Any]) -> Plan:
        """Собирает план под распарсенную цель. state = {'intent':..., 'args':...}."""
        intent = state.get("intent", "chat")
        args = state.get("args", {})
        candidates = self._candidate_actions(intent)
        evaluated: list[PlanStep] = []
        for name in candidates[: self.pcfg.max_actions]:
            sim = self.simulate_action(name)
            score, succ = self.score_action(name, sim)
            step = PlanStep(
                action=Action(name=name, args=dict(args), purpose=f"выполнить '{intent}'"),
                expected_reward=sim["reward"], expected_risk=sim["risk"],
                expected_cost=sim["cost"], success_probability=succ, score=score,
            )
            evaluated.append(step)

        evaluated.sort(key=lambda s: s.score, reverse=True)
        best_steps = self._select_sequence(evaluated, intent)
        plan = Plan(goal=state.get("raw", intent), steps=best_steps)
        if best_steps:
            plan.expected_benefit = float(np.mean([s.expected_reward for s in best_steps]))
            plan.risk = float(max(s.expected_risk for s in best_steps))
            plan.cost = float(sum(s.expected_cost for s in best_steps))
            plan.confidence = float(np.clip(np.mean([s.success_probability for s in best_steps]), 0, 1))
        plan.alternatives = [
            {"action": s.action.name, "score": round(s.score, 4)} for s in evaluated[len(best_steps):][:5]
        ]
        logger.info("план %s: %d шагов, уверенность %.2f", plan.plan_id, plan.length, plan.confidence)
        return plan

    def _select_sequence(self, evaluated: list[PlanStep], intent: str) -> list[PlanStep]:
        """Из лучших кандидатов собирает осмысленную последовательность с tie-break по безопасности."""
        if not evaluated:
            return []
        by_name = {s.action.name: s for s in evaluated}
        seq_names: list[str] = []
        if intent == "write_file":
            seq_names = ["write_file", "read_file"]
        elif intent == "remember":
            seq_names = ["remember"]
        elif intent == "recall":
            seq_names = ["search_memory", "recall"]
        elif intent == "read_file":
            seq_names = ["read_file"]
        elif intent == "list_files":
            seq_names = ["list_files"]
        elif intent == "calculate":
            seq_names = ["calculate"]
        elif intent == "gridworld":
            seq_names = ["run_skill", "calculate"]
        else:
            top = evaluated[0]
            seq_names = [top.action.name] if top.score > 0 else ["ask_user"]

        steps = [by_name[n] for n in seq_names if n in by_name]
        if not steps:
            steps = [evaluated[0]]
        # tie-break: среди шагов с близким score выбираем более безопасный/дешёвый
        merged: dict[str, PlanStep] = {}
        tol = self.pcfg.tie_tolerance
        for s in evaluated:
            cur = merged.get(s.action.name)
            if cur is None or abs(s.score - cur.score) <= tol:
                if cur is None or (s.expected_risk, s.expected_cost) < (cur.expected_risk, cur.expected_cost):
                    merged[s.action.name] = s
            if s.score > (cur.score if cur else -1e9):
                merged[s.action.name] = s
        return [merged.get(s.action.name, s) for s in steps]

    def _candidate_actions(self, intent: str) -> list[str]:
        """Кандидаты для симуляции: релевантные + соседние по графу."""
        primary = {
            "remember": ["remember", "search_memory", "wait"],
            "recall": ["search_memory", "recall", "ask_user", "wait"],
            "write_file": ["write_file", "read_file", "remember", "ask_user"],
            "read_file": ["read_file", "list_files", "ask_user"],
            "list_files": ["list_files", "read_file"],
            "calculate": ["calculate", "run_skill", "ask_user"],
            "gridworld": ["run_skill", "calculate", "list_files"],
            "chat": ["recall", "search_memory", "ask_user", "wait"],
        }.get(intent, ["ask_user", "wait"])
        ordered = sorted(primary, key=lambda a: -self.simulate_action(a)["reward"])
        return ordered
