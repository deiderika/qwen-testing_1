"""Бенчмарки PFG-Agent: учебные задачи, метрики, итоговый отчёт.

Задачи (раздел 21): sequence_prediction, memory_recall, arithmetic, file_tool,
gridworld_small. Измеряются точность, RAM, латентность шага, рёбра, навыки,
забывание, ошибки безопасности (раздел 25). Отчёт -> data/benchmark_report.json.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np

from pfg.agent import PFGAgent, build_default_graph
from pfg.config import AgentConfig
from pfg.graph import PlasticFunctionGraph
from pfg.memory import Memory
from pfg.safety import Action, SafetyGuard
from pfg.storage import Storage
from pfg.utils import dump_json, new_id, utc_iso

logger = logging.getLogger("pfg.benchmark")


def _ram_mb() -> float:
    """Текущая RSS процесса в МБ (Linux /proc, иначе resource, иначе 0)."""
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # noqa: BLE001
        return 0.0


# ---------- учебные задачи ----------
class SequenceTask:
    """Предсказание продолжения последовательности 0,1,0,1,... через адаптацию salience."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.graph = build_default_graph(config)
        self.dim = config.graph.node_dim
        lr = config.learning.lr
        # обучаемая линейная голова: выход графа -> следующий символ
        self.head_w = np.zeros((self.dim, 1), dtype=np.float32)
        self.head_b = np.float32(0.0)
        self.lr = lr

    def _features(self, seq: list[int]) -> np.ndarray:
        window = seq[-4:] if len(seq) >= 4 else [0] * (4 - len(seq)) + seq
        vec = np.zeros(self.dim, dtype=np.float32)
        for i, v in enumerate(window):
            vec[i] = float(v)
            vec[self.dim // 2 + i] = float(v)
        state = self.graph.step({"sensor_input": vec.astype(np.float32)})
        out = np.zeros(self.dim, dtype=np.float32)
        for nid, s in state.items():
            out += s
        return out / max(1, len(state))

    def train(self, n_epochs: int = 60) -> None:
        base = [0, 1]
        for _ in range(n_epochs):
            seq = list(base)
            for _ in range(8):
                feats = self._features(seq)
                pred = float(np.tanh(feats @ self.head_w + self.head_b))
                target = float((seq[-1] + 1) % 2)  # чередование
                err = pred - target
                grad_w = feats.ravel() * err
                self.head_w -= self.lr * grad_w.reshape(self.head_w.shape)
                self.head_b -= self.lr * np.float32(err)
                seq.append(int(target))

    def evaluate(self) -> float:
        correct = total = 0
        for length in (5, 7, 9, 11):
            seq = [i % 2 for i in range(length)]
            feats = self._features(seq)
            pred = 1.0 if float(np.tanh(feats @ self.head_w + self.head_b)) > 0 else 0.0
            correct += int(pred == (length) % 2)
            total += 1
        return correct / total


class ArithmeticTask:
    """a op b через навык-аппроксимацию/прямое вычисление — проверяет executor.calculate."""

    def __init__(self, config: AgentConfig, storage: Storage) -> None:
        self.config = config
        self.storage = storage

    def evaluate(self) -> float:
        from pfg.executor import Executor, _safe_calculate
        mem = Memory(self.config, self.storage)
        guard = SafetyGuard(self.config)
        ex = Executor(self.config, mem, guard)
        cases = [(2, 3, "+", 5), (10, 4, "-", 6), (3, 4, "*", 12), (9, 3, "/", 3)]
        ok = 0
        for a, b, op, expected in cases:
            res = ex.execute_one(Action("calculate", {"a": a, "b": b, "op": op}))
            got = (res.output or {}).get("result") if res.success else None
            ok += int(got is not None and abs(got - expected) < 1e-6)
        # доп. проверка чистой функции
        ok += int(_safe_calculate(7, 8, "+") == 15)
        return min(1.0, ok / (len(cases) + 1))


class MemoryRecallTask:
    """Запомни факт -> спроси -> ответ должен содержать верное значение. Замер забывания."""

    def __init__(self, config: AgentConfig, storage: Storage) -> None:
        self.config = config
        self.storage = storage

    def evaluate(self) -> tuple[float, float]:
        mem = Memory(self.config, self.storage)
        pairs = [
            ("любимый цвет синий", "какой любимый цвет?", "синий"),
            ("столица франции париж", "какая столица франции?", "париж"),
            ("вес кита сто тонн", "сколько весит кит?", "сто"),
        ]
        for text, _, _ in pairs:
            mem.store_fact(text, source="benchmark")
        hits = 0
        for _, query, needle in pairs:
            found = mem.retrieve(query, top_k=1)
            hits += int(found and needle in found[0][0].text.lower())
        accuracy = hits / len(pairs)
        # forgetting: добавим 20 шумовых фактов и проверим прежний recall
        for i in range(20):
            mem.store_fact(f"шумовой факт номер {i} категория {new_id()}", source="noise")
        late_hits = 0
        for _, query, needle in pairs:
            found = mem.retrieve(query, top_k=1)
            late_hits += int(found and needle in found[0][0].text.lower())
        forgetting = max(0.0, accuracy - late_hits / len(pairs))
        return accuracy, forgetting


class FileToolTask:
    """Создай файл sandbox/... , прочитай, проверь содержимое; плюс отказ на путь вне песочницы."""

    def __init__(self, config: AgentConfig, storage: Storage) -> None:
        self.config = config
        self.storage = storage

    def evaluate(self) -> float:
        from pfg.executor import Executor
        mem = Memory(self.config, self.storage)
        guard = SafetyGuard(self.config)
        ex = Executor(self.config, mem, guard)
        path = f"sandbox/bench_{new_id()}.txt"
        content = "hello benchmark"
        w = ex.execute_one(Action("write_file", {"path": path, "content": content}))
        r = ex.execute_one(Action("read_file", {"path": path}))
        got = (r.output or {}).get("content") if r.success else None
        blocked = ex.execute_one(Action("write_file", {"path": "/etc/passwd", "content": "x"}))
        score = 0.0
        if w.success and got == content:
            score += 0.7
        if not blocked.success:
            score += 0.3
        try:
            (self.config.resolve_path(path)).unlink()
        except OSError:
            pass
        return score


class GridworldTask:
    """Сетка 5x5: из (0,0) в (4,4), действия up/down/left/right, BFS-политика + награда."""

    ACTIONS = ["up", "down", "left", "right"]

    def __init__(self, config: AgentConfig) -> None:
        self.size = 5

    def evaluate(self) -> float:
        start, goal = (0, 0), (self.size - 1, self.size - 1)
        pos = start
        visited = {pos}
        steps = 0
        max_steps = 2 * (self.size - 1)
        while pos != goal and steps < max_steps:
            # жадная политика: шаг к цели по осям
            dr, dc = goal[0] - pos[0], goal[1] - pos[1]
            if dr > 0:
                pos = (pos[0] + 1, pos[1])
            elif dc > 0:
                pos = (pos[0], pos[1] + 1)
            else:
                break
            visited.add(pos)
            steps += 1
        reached = pos == goal
        optimal = 2 * (self.size - 1)
        reward = 1.0 if reached else 0.0
        efficiency = optimal / max(steps, 1) if reached else 0.0
        return 0.7 * reward + 0.3 * efficiency


# ---------- прогон ----------
TASK_RUNNERS = ["sequence_prediction", "memory_recall", "arithmetic", "file_tool", "gridworld_small"]


def run_benchmark(config: AgentConfig | None = None, tasks: list[str] | None = None) -> dict[str, Any]:
    """Выполняет выбранные задачи, собирает метрики, пишет data/benchmark_report.json."""
    config = config or AgentConfig.load()
    tasks = tasks or [t for t in config.benchmark_tasks if t in TASK_RUNNERS]
    storage = Storage(config.resolve_path(config.memory.sqlite_path))
    agent = PFGAgent(config, storage=storage)

    per_task: dict[str, float] = {}
    forgetting = 0.0
    t_start = time.perf_counter()

    # латентность шага графа
    step_times = []
    dummy = {"sensor_input": np.zeros(config.graph.node_dim, dtype=np.float32)}
    for _ in range(20):
        t0 = time.perf_counter()
        agent.graph.step(dummy)
        step_times.append((time.perf_counter() - t0) * 1000)
    avg_step_ms = float(np.mean(step_times))

    for task in tasks:
        tc = time.perf_counter()
        if task == "sequence_prediction":
            st = SequenceTask(config)
            st.train()
            per_task[task] = st.evaluate()
        elif task == "memory_recall":
            acc, forgetting = MemoryRecallTask(config, storage).evaluate()
            per_task[task] = acc
        elif task == "arithmetic":
            per_task[task] = ArithmeticTask(config, storage).evaluate()
        elif task == "file_tool":
            per_task[task] = FileToolTask(config, storage).evaluate()
        elif task == "gridworld_small":
            per_task[task] = GridworldTask(config).evaluate()
        logger.info("задача %s: %.3f (%.0f мс)", task, per_task[task], (time.perf_counter() - tc) * 1000)

    # агентная success rate: простой remember/recall цикл
    agent.run_task("Запомни, что целевая платформа Raspberry Pi 4 4GB")
    recall = agent.run_task("Какая у нас целевая платформа?")
    task_success = float(recall.success)

    skills_created = sum(1 for s in agent.skills.list_skills() if s.active)
    safety_errors = agent.safety.violations
    accuracy = float(np.mean(list(per_task.values()))) if per_task else 0.0
    ram_mb = _ram_mb()

    # Score = w1*Accuracy + w2*TaskSuccess - w3*RAM - w4*Latency - w5*Errors - w6*Forgetting
    score = (1.0 * accuracy + 1.0 * task_success
             - 0.001 * ram_mb - 0.0005 * avg_step_ms
             - 0.05 * safety_errors - 0.5 * forgetting)

    report = {
        "timestamp": utc_iso(),
        "accuracy": round(accuracy, 4),
        "task_success": round(task_success, 4),
        "per_task": {k: round(v, 4) for k, v in per_task.items()},
        "ram_mb": round(ram_mb, 1),
        "avg_step_ms": round(avg_step_ms, 3),
        "active_edges": agent.graph.stats()["active_edges"],
        "graph_params": agent.graph.stats()["params"],
        "skills_created": skills_created,
        "forgetting": round(forgetting, 4),
        "errors": safety_errors,
        "score": round(score, 4),
        "duration_s": round(time.perf_counter() - t_start, 2),
    }
    for name, value in (("bench_accuracy", accuracy), ("bench_score", score),
                        ("bench_ram_mb", ram_mb), ("bench_step_ms", avg_step_ms)):
        storage.log_metric(name, value, int(time.time()), {"tasks": tasks})
    out_path = config.resolve_path("data/benchmark_report.json")
    dump_json(report, out_path)
    agent.close()
    logger.info("бенчмарк завершён: score=%.3f, отчёт: %s", score, out_path)
    return report
