"""Пластичный функциональный граф (Plastic Function Graph) — ядро интеллекта PFG-Agent.

Узел хранит состояние h_i; рёбра несут не просто веса, а функции (базис Чебышёва),
гейты внимания и накапливаемую важность (salience). Инференс — чистый NumPy (float32).
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pfg.config import AgentConfig
from pfg.math_basis import edge_function, edge_gate, edge_input, layer_norm
from pfg.utils import new_id, now_ts

logger = logging.getLogger("pfg.graph")

NODE_TYPES = {"concept", "sensor", "action", "memory", "reward", "risk", "skill", "abstract"}


@dataclass
class Node:
    id: str
    type: str
    vector: np.ndarray
    bias: float = 0.0
    self_gain: float = 1.0
    created_at: int = field(default_factory=now_ts)
    last_used_at: int = 0
    usage_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "type": self.type, "vector": self.vector.tolist(),
            "bias": self.bias, "self_gain": self.self_gain, "created_at": self.created_at,
            "last_used_at": self.last_used_at, "usage_count": self.usage_count,
            "metadata": self.metadata,
        }


@dataclass
class Edge:
    id: str
    src_node_id: str
    dst_node_id: str
    input_projection: np.ndarray   # a_e
    input_bias: float              # b_e
    basis_coeffs: np.ndarray       # c_{e,k}
    output_vector: np.ndarray      # v_e
    gate_key: np.ndarray           # k_e
    gate_bias: float               # b_gate
    salience: float = 0.5          # S_e
    usage_count: int = 0
    created_at: int = field(default_factory=now_ts)
    last_used_at: int = 0
    active: bool = True
    frozen: bool = False
    symbolic_expr: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def params_blob(self) -> bytes:
        """Сериализация обучаемых параметров в компактный BLOB."""
        buf = io.BytesIO()
        for arr in (self.input_projection, self.basis_coeffs, self.output_vector, self.gate_key):
            np.save(buf, np.asarray(arr, dtype=np.float32), allow_pickle=False)
        return buf.getvalue()

    @classmethod
    def from_params_blob(cls, blob: bytes, meta: dict[str, Any]) -> "Edge":
        buf = io.BytesIO(blob)
        a = np.load(buf)
        c = np.load(buf)
        v = np.load(buf)
        k = np.load(buf)
        return cls(input_projection=a, input_bias=float(meta.pop("input_bias", 0.0)),
                   basis_coeffs=c, output_vector=v, gate_key=k,
                   gate_bias=float(meta.pop("gate_bias", 0.0)), **meta)

    def param_base(self) -> dict[str, Any]:
        return {"input_bias": self.input_bias, "gate_bias": self.gate_bias}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "src_node_id": self.src_node_id, "dst_node_id": self.dst_node_id,
            "input_projection": self.input_projection.tolist(), "input_bias": self.input_bias,
            "basis_coeffs": self.basis_coeffs.tolist(), "output_vector": self.output_vector.tolist(),
            "gate_key": self.gate_key.tolist(), "gate_bias": self.gate_bias,
            "salience": self.salience, "usage_count": self.usage_count,
            "created_at": self.created_at, "last_used_at": self.last_used_at,
            "active": self.active, "frozen": self.frozen, "symbolic_expr": self.symbolic_expr,
            "metadata": self.metadata,
        }


class PlasticFunctionGraph:
    """Граф пластичных функциональных связей с TopK-гейтированием и salience-обучением."""

    def __init__(self, config: AgentConfig, rng: np.random.Generator | None = None) -> None:
        self.config = config
        self.dim = config.graph.node_dim
        self.rng = rng or np.random.default_rng(config.seed)
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        self._incoming: dict[str, list[str]] = {}   # dst_id -> [edge_id]
        self._outgoing: dict[str, list[str]] = {}   # src_id -> [edge_id]
        self._queries: dict[str, np.ndarray] = {}   # узловые q_i для гейтов
        self.last_states: dict[str, np.ndarray] = {}  # состояния после последнего шага (для curiosity)
        self.step_count = 0
        self.steps_since_save = 0

    # ---------- структура ----------
    def add_node(self, node_type: str, node_id: str | None = None) -> str:
        if node_type not in NODE_TYPES:
            raise ValueError(f"неизвестный тип узла: {node_type}")
        if len(self.nodes) >= self.config.graph.max_nodes:
            raise RuntimeError("достигнут max_nodes")
        nid = node_id or f"{node_type}_{new_id()}"
        if nid in self.nodes:
            return nid
        vec = (self.rng.standard_normal(self.dim) * 0.1).astype(np.float32)
        self.nodes[nid] = Node(id=nid, type=node_type, vector=vec,
                               self_gain=float(self.rng.uniform(0.7, 1.0)))
        self._queries[nid] = (self.rng.standard_normal(self.dim) * 0.1).astype(np.float32)
        self._incoming.setdefault(nid, [])
        self._outgoing.setdefault(nid, [])
        return nid

    def add_edge(self, src_id: str, dst_id: str) -> str:
        if src_id not in self.nodes or dst_id not in self.nodes:
            raise KeyError("узлы должны существовать до создания ребра")
        if len(self.edges) >= self.config.graph.max_edges:
            raise RuntimeError("достигнут max_edges")
        eid = f"e_{new_id()}"
        scale = 1.0 / np.sqrt(self.dim)
        edge = Edge(
            id=eid, src_node_id=src_id, dst_node_id=dst_id,
            input_projection=(self.rng.standard_normal(self.dim) * scale).astype(np.float32),
            input_bias=float(self.rng.normal(0, 0.1)),
            basis_coeffs=self._default_basis_coeffs(),
            output_vector=(self.rng.standard_normal(self.dim) * scale).astype(np.float32),
            gate_key=(self.rng.standard_normal(self.dim) * scale).astype(np.float32),
            gate_bias=float(self.rng.normal(0, 0.1)),
            salience=0.5,
        )
        self.edges[eid] = edge
        self._incoming[dst_id].append(eid)
        self._outgoing[src_id].append(eid)
        return eid

    def _default_basis_coeffs(self) -> np.ndarray:
        """Инициализация f_e близко к линейной функции: T_1 + малый шум."""
        order = self.config.graph.basis_order
        coeffs = np.zeros(order + 1, dtype=np.float32)
        if order >= 1:
            coeffs[1] = 1.0
        else:
            coeffs[0] = 1.0
        coeffs += (self.rng.standard_normal(order + 1) * 0.05).astype(np.float32)
        return coeffs

    def remove_edge(self, edge_id: str) -> None:
        edge = self.edges.pop(edge_id, None)
        if edge is None:
            return
        self._incoming[edge.dst_node_id].remove(edge_id)
        self._outgoing[edge.src_node_id].remove(edge_id)

    def incoming(self, node_id: str) -> list[Edge]:
        return [self.edges[eid] for eid in self._incoming.get(node_id, []) if eid in self.edges]

    # ---------- вычисление ----------
    def step(self, inputs: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        """Один шаг симуляции графа по алгоритму из ТЗ (раздел 12.2)."""
        inputs = inputs or {}
        cfg = self.config.graph
        ts = now_ts()
        new_states: dict[str, np.ndarray] = {}

        for nid, node in self.nodes.items():
            scored: list[tuple[float, Edge, float]] = []
            for edge in self.incoming(nid):
                if not edge.active:
                    continue
                src = self.nodes[edge.src_node_id]
                x = edge_input(src.vector, edge.input_projection, edge.input_bias)
                if edge.symbolic_expr and edge.frozen:
                    f = _eval_symbolic(edge.symbolic_expr, float(x))
                else:
                    f = float(edge_function(x, edge.basis_coeffs))
                g = float(edge_gate(edge.salience, self._queries[nid], edge.gate_key,
                                    edge.gate_bias, cfg.node_dim))
                scored.append((g, edge, f))

            scored.sort(key=lambda item: item[0], reverse=True)
            top = scored[: cfg.top_k_edges]

            contribution = np.zeros(cfg.node_dim, dtype=np.float32)
            for g, edge, f in top:
                contribution += np.float32(g * f) * edge.output_vector
                edge.usage_count += 1
                edge.last_used_at = ts

            external = np.asarray(inputs.get(nid, np.zeros(cfg.node_dim)), dtype=np.float32)
            pre = node.self_gain * node.vector + contribution + external + node.bias
            updated = node.vector + np.float32(cfg.step_size) * np.tanh(pre)
            state = layer_norm(updated) if cfg.use_layer_norm else updated
            node.last_used_at = ts
            node.usage_count += 1
            new_states[nid] = state.astype(np.float32)

        for nid, st in new_states.items():
            self.nodes[nid].vector = st
        self.last_states = new_states
        self.step_count += 1
        self.steps_since_save += 1
        return new_states

    def simulate(self, inputs_seq: list[dict[str, np.ndarray]] | dict[str, np.ndarray],
                 horizon: int) -> list[dict[str, np.ndarray]]:
        """Развёртка графа на horizon шагов. inputs_seq — список входов по шагам или один вход."""
        if isinstance(inputs_seq, dict):
            inputs_seq = [inputs_seq] * horizon
        traj: list[dict[str, np.ndarray]] = []
        for t in range(horizon):
            inp = inputs_seq[t] if t < len(inputs_seq) else {}
            traj.append(self.step(inp))
        return traj

    # ---------- обучение структуры ----------
    def update_salience(self, edge_id: str, delta: float) -> None:
        edge = self.edges.get(edge_id)
        if edge is None or edge.frozen:
            return
        edge.salience = float(np.clip(edge.salience + delta, 0.0, 2.0))

    def decay_salience(self) -> None:
        """Лёгкаяforgetting-регуляризация важных связей."""
        d = self.config.learning.salience_decay
        for e in self.edges.values():
            if not e.frozen:
                e.salience *= d

    def prune_and_grow(self) -> dict[str, int]:
        """Эволюция структуры: удаление слабых старых рёбер (добавление — снаружи, через add_edge)."""
        removed = 0
        lcfg = self.config.learning
        age_limit = lcfg.min_edge_age_for_removal
        for eid in list(self.edges):
            e = self.edges[eid]
            if e.frozen:
                continue
            # возраст рёбра оцениваем по накопленному использованию (дешёвый прокси для RPi)
            if e.salience < lcfg.edge_remove_threshold and e.usage_count > age_limit:
                self.remove_edge(eid)
                removed += 1
        self.decay_salience()
        return {"removed": removed}

    def coactivation_counts(self) -> dict[str, int]:
        """Счётчики совместной активности узлов — сырьё для обнаружения повторяющихся подграфов."""
        counts: dict[str, int] = {}
        for eid, e in self.edges.items():
            if e.usage_count > 0:
                key = f"{e.src_node_id}|{e.dst_node_id}"
                counts[key] = e.usage_count
        return counts

    # ---------- сериализация ----------
    def export_npz(self, path: str) -> None:
        """Экспорт всех параметров в .npz — лёгкий формат для RPi/NumPy-инференса."""
        arrays: dict[str, np.ndarray] = {}
        for n in self.nodes.values():
            arrays[f"node::{n.id}::vector"] = n.vector
        for e in self.edges.values():
            arrays[f"edge::{e.id}::a"] = e.input_projection
            arrays[f"edge::{e.id}::c"] = e.basis_coeffs
            arrays[f"edge::{e.id}::v"] = e.output_vector
            arrays[f"edge::{e.id}::k"] = e.gate_key
        meta = {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [{**e.to_dict(),
                       "input_projection": None, "basis_coeffs": None,
                       "output_vector": None, "gate_key": None} for e in self.edges.values()],
            "step_count": self.step_count,
        }
        import json
        arrays["__meta__"] = np.array([json.dumps(meta)], dtype=object)
        np.savez_compressed(path, **arrays, allow_pickle=True)

    def save(self, storage: Any) -> None:
        """Сохранение полного состояния в SQLite-хранилище (storage: Storage)."""
        self.steps_since_save = 0
        for n in self.nodes.values():
            storage.upsert_node({**n.to_dict(), "vector": n.vector})
        for e in self.edges.values():
            storage.upsert_edge({
                "id": e.id, "src_node_id": e.src_node_id, "dst_node_id": e.dst_node_id,
                "params_blob": e.params_blob(), "salience": e.salience,
                "usage_count": e.usage_count, "active": e.active, "frozen": e.frozen,
                "symbolic_expr": e.symbolic_expr, "created_at": e.created_at,
                "last_used_at": e.last_used_at, "metadata": {**e.metadata, **e.param_base()},
            })

    def load(self, storage: Any) -> None:
        """Загрузка состояния из SQLite."""
        import json
        self.nodes.clear(); self.edges.clear(); self._incoming.clear(); self._outgoing.clear()
        from pfg.storage import blob_to_vec
        for row in storage.query("SELECT * FROM nodes"):
            (nid, ntype, vector, bias, self_gain, created_at, last_used, usage, meta) = row
            node = Node(id=nid, type=ntype, vector=blob_to_vec(vector), bias=float(bias or 0.0),
                        self_gain=float(self_gain or 1.0), created_at=int(created_at or 0),
                        last_used_at=int(last_used or 0), usage_count=int(usage or 0),
                        metadata=json.loads(meta or "{}"))
            self.nodes[nid] = node
            self._queries.setdefault(nid, (self.rng.standard_normal(self.dim) * 0.1).astype(np.float32))
            self._incoming.setdefault(nid, [])
            self._outgoing.setdefault(nid, [])
        for row in storage.query("SELECT * FROM edges"):
            (eid, src, dst, params, salience, usage, active, frozen, sym, created_at, last_used, meta) = row
            m = json.loads(meta or "{}")
            base = {"input_bias": float(m.get("input_bias", 0.0)), "gate_bias": float(m.get("gate_bias", 0.0))}
            extra = {k: v for k, v in m.items() if k not in ("input_bias", "gate_bias")}
            # Нормализация legacy-формул: старые записи могли содержать математическую
            # нотацию '^' (XOR для eval). Приводим к Python-синтаксису '**' при загрузке.
            if isinstance(sym, str) and "^" in sym and "**" not in sym:
                sym = sym.replace("^", "**")
            edge = Edge.from_params_blob(params, {
                "id": eid, "src_node_id": src, "dst_node_id": dst, **base,
                "salience": float(salience or 0.5), "usage_count": int(usage or 0),
                "active": bool(active), "frozen": bool(frozen), "symbolic_expr": sym,
                "created_at": int(created_at or 0), "last_used_at": int(last_used or 0),
                "metadata": extra,
            })
            self.edges[eid] = edge
            self._incoming.setdefault(dst, []).append(eid)
            self._outgoing.setdefault(src, []).append(eid)

    def stats(self) -> dict[str, Any]:
        active = sum(1 for e in self.edges.values() if e.active)
        n_params = sum(
            e.input_projection.size + e.basis_coeffs.size + e.output_vector.size + e.gate_key.size
            for e in self.edges.values()
        )
        return {
            "nodes": len(self.nodes), "edges": len(self.edges), "active_edges": active,
            "frozen_edges": sum(1 for e in self.edges.values() if e.frozen),
            "params": int(n_params), "steps": self.step_count, "node_dim": self.dim,
            "steps_since_save": self.steps_since_save,
        }


# ---------- символическая выжимка (раздел 20) ----------
_CANDIDATE_FORMULAS = ["0", "c", "a*x+b", "a*x^2+b*x+c", "a*tanh(b*x+c)+d"]


def fit_symbolic(xs: np.ndarray, ys: np.ndarray, max_error: float = 0.05) -> tuple[str, float] | None:
    """Пытается аппроксимировать x->f(x) простой формулой. Возвращает (expr, error) или None."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size < 4:
        return None
    best: tuple[str, float] | None = None
    # константа
    err_c = float(np.mean(np.abs(ys - ys.mean())))
    best = ("%.4f" % float(ys.mean()), err_c)
    for deg, label in ((1, "poly1"), (2, "poly2")):
        try:
            coeffs, err = np.polyfit(xs, ys, deg), None
            pred = np.polyval(coeffs, xs)
            err = float(np.mean(np.abs(pred - ys)))
        except Exception:  # noqa: BLE001 — polyfit может упасть на вырожденных данных
            continue
        expr = _poly_expr(coeffs)
        if best is None or err < best[1]:
            best = (expr, err)
    _ = label  # для ясности: перебираем полиномы степеней 1 и 2
    if best and best[1] <= max_error:
        return best
    return None


def _poly_expr(coeffs: np.ndarray) -> str:
    parts = []
    deg = len(coeffs) - 1
    for i, c in enumerate(coeffs):
        p = deg - i
        if abs(c) < 1e-4:
            continue
        term = f"{c:.4f}"
        if p >= 1:
            # Python-совместимое возведение в степень (**), а не математическая нотация (^),
            # иначе eval() трактует ^ как XOR и падает.
            term += "*x" if p == 1 else f"*x**{p}"
        parts.append(term)
    return " + ".join(parts).replace("+ -", "- ") if parts else "0"


_SAFE_MATH = {"tanh": np.tanh, "sin": np.sin, "cos": np.cos, "exp": np.exp, "abs": abs}


def _eval_symbolic(expr: str, x: float) -> float:
    """Безопасное вычисление зафиксированного символического выражения (только числа и x)."""
    # Обратная совместимость: старые сохранённые формулы могли содержать математическую
    # нотацию '^' (например "0.5*x^2"). Для eval это XOR, поэтому транслируем в '**'.
    safe_expr = expr.replace("^", "**")
    env = {"x": x, "__builtins__": {}, **_SAFE_MATH}
    try:
        return float(eval(safe_expr, env, {"x": x}))  # noqa: S307 — expr создаётся самим агентом из whitelist-формул
    except Exception:  # noqa: BLE001
        logger.warning("не удалось вычислить symbolic_expr %r", expr)
        return float(x)


def distill_edges(graph: "PlasticFunctionGraph", n_samples: int = 64,
                  max_error: float = 0.05) -> int:
    """Периодическая символическая выжимка рёбер: тяжёлый базис -> быстрая формула."""
    rng = graph.rng
    distilled = 0
    for edge in graph.edges.values():
        if edge.frozen or not edge.active:
            continue
        src = graph.nodes.get(edge.src_node_id)
        if src is None:
            continue
        xs = rng.uniform(-1.0, 1.0, size=n_samples).astype(np.float32)
        ys = np.array([float(edge_function(np.float32(x), edge.basis_coeffs)) for x in xs], dtype=np.float64)
        result = fit_symbolic(xs, ys, max_error=max_error)
        if result is not None:
            expr, err = result
            edge.symbolic_expr = expr
            edge.frozen = True
            edge.metadata["distill_error"] = err
            distilled += 1
    return distilled
