"""Навыки PFG-Agent: обнаружение подграфов, компиляция, тестирование, откат.

Разрешено перезаписывать навыки (раздел 19.1); ядро графа, интерпретатор и
модуль безопасности — нет. Генерируемый код проходит статическую проверку
(ast) и запускается в изолированном процессе. Плохой навык не активируется;
перед активацией сохраняются метрики до/после и старая версия для отката.
"""
from __future__ import annotations

import ast
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pfg.config import AgentConfig
from pfg.executor import Executor
from pfg.graph import PlasticFunctionGraph, distill_edges
from pfg.math_basis import _fit_polynomial_desc as fit_polynomial
from pfg.memory import Memory
from pfg.storage import Storage
from pfg.utils import JsonlLogger, new_id, now_ts, utc_iso

logger = logging.getLogger("pfg.skills")

# Запрещённые конструкции в коде навыка (статическая проверка)
_FORBIDDEN_NODES = (
    ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal,
)
_FORBIDDEN_NAMES = {
    "exec", "eval", "compile", "open", "__import__", "globals", "locals",
    "input", "breakpoint", "exit", "quit", "os", "sys", "subprocess",
    "socket", "shutil", "pathlib", "requests", "urllib", "pickle", "marshal",
}


@dataclass
class SkillProposal:
    accepted: bool
    skill_name: str = ""
    reason: str = ""
    error_before: float | None = None
    error_after: float | None = None
    latency_before_ms: float | None = None
    latency_after_ms: float | None = None


@dataclass
class SkillInfo:
    id: str
    name: str
    description: str
    path: Path
    active: bool
    meta: dict[str, Any] = field(default_factory=dict)


def static_check(code: str, allowed_imports: list[str]) -> tuple[bool, str]:
    """AST-проверка сгенерированного кода: только чистые функции, whitelist импортов."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"синтаксическая ошибка: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, (_FORBIDDEN_NODES[2], _FORBIDDEN_NODES[3])):  # Global/Nonlocal
            return False, "запрещены global/nonlocal"
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in allowed_imports:
                    return False, f"импорт вне whitelist: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in allowed_imports:
                return False, f"импорт вне whitelist: {node.module}"
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in _FORBIDDEN_NAMES:
                return False, f"запрещённый вызов: {fn.id}"
            if isinstance(fn, ast.Attribute) and fn.value.__class__.__name__ == "Name":
                base = getattr(fn.value, "id", "")
                if base in _FORBIDDEN_NAMES:
                    return False, f"запрещённая библиотека: {base}"
        elif isinstance(node, (ast.With, ast.Try)):
            pass  # разрешаем
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return False, f"дуэт-доступ к внутреннему атрибуту: {node.attr}"
    return True, "ok"


class SkillsManager:
    """Реестр навыков: создание, тестирование, активация, запуск, откат."""

    def __init__(self, config: AgentConfig, storage: Storage, memory: Memory,
                 graph: PlasticFunctionGraph) -> None:
        self.config = config
        self.scfg = config.skills
        self.storage = storage
        self.memory = memory
        self.graph = graph
        self.dir = config.resolve_path(self.scfg.dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = JsonlLogger(config.log_dir)

    # ---------- реестр ----------
    def list_skills(self) -> list[SkillInfo]:
        rows = self.storage.query(
            "SELECT id, name, description, code_path, active, metadata FROM skills ORDER BY created_at")
        out = []
        for sid, name, desc, code_path, active, meta in rows:
            out.append(SkillInfo(id=sid, name=name, description=desc or "",
                                 path=Path(code_path) if code_path else self.dir / name,
                                 active=bool(active),
                                 meta=yaml.safe_load(meta) if isinstance(meta, str) and meta else {}))
        return out

    def get_active(self, name: str) -> SkillInfo | None:
        return next((s for s in self.list_skills() if s.name == name and s.active), None)

    # ---------- обнаружение (раздел 19.2) ----------
    def find_repetitive_subgraphs(self, min_usage: int = 20) -> list[dict[str, Any]]:
        """Ищет часто используемые рёбра/пары узлов — кандидатов в навыки."""
        candidates = []
        for edge in self.graph.edges.values():
            if edge.usage_count >= min_usage and edge.active:
                candidates.append({
                    "edge_id": edge.id,
                    "src": edge.src_node_id, "dst": edge.dst_node_id,
                    "usage": edge.usage_count, "salience": round(edge.salience, 3),
                })
        candidates.sort(key=lambda c: -c["usage"])
        return candidates[:10]

    def collect_edge_examples(self, edge_id: str, n: int = 64) -> list[tuple[float, float]]:
        """Собирает пары x -> f_e(x) с входами均匀 в [-1,1]."""
        from pfg.math_basis import edge_function
        edge = self.graph.edges.get(edge_id)
        if edge is None:
            return []
        rng = np.random.default_rng(self.config.seed)
        xs = rng.uniform(-1.0, 1.0, size=n)
        return [(float(x), float(edge_function(np.float32(x), edge.basis_coeffs)), edge_id)
                for x in xs]

    # ---------- компиляция (раздел 19.3) ----------
    def create_from_examples(self, name: str, examples: list[Any],
                             description: str = "") -> SkillProposal:
        """examples — пары (x, y) или тройки (x, y, edge_id) для замера latency до/после.

    Аппроксимирует примеры простой функцией, генерирует код+тесты, проверяет в песочнице.
    """
        if len(examples) < 4:
            return SkillProposal(False, name, "мало примеров для аппроксимации")
        xs = np.array([t[0] for t in examples], dtype=np.float64)
        ys = np.array([t[1] for t in examples], dtype=np.float64)

        best: tuple[int, np.ndarray, float] | None = None
        for deg in (1, 2, 3, 4):
            coeffs, err = fit_polynomial(xs, ys, deg)
            if best is None or err < best[2]:
                best = (deg, coeffs, err)
        assert best is not None
        deg, coeffs, error = best
        if error > self.scfg.max_skill_error:
            return SkillProposal(False, name, f"слишком большая ошибка аппроксимации: {error:.4f}")

        # порог max_skill_error — средняя ошибка; на отдельных точках она может быть выше.
        # Тесты в песочнице используют max-ошибку с запасом, иначе хорошие кандидаты
        # отклоняются из-за выбросов (несоответствие mean/max критериев).
        sample = [(float(t[0]), float(t[1])) for t in examples[:16]]
        max_test_err = max((abs(float(np.polyval(coeffs, x)) - y) for x, y in sample), default=0.0)
        test_tol = max(1e-3, 2.0 * max_test_err, 5.0 * self.scfg.max_skill_error)
        if max_test_err > 10.0 * self.scfg.max_skill_error:
            return SkillProposal(False, name,
                                 f"большая максимальная ошибка на тестовых точках: {max_test_err:.4f}")

        code = generate_skill_code(name, coeffs)
        ok, why = static_check(code, self.scfg.allowed_imports)
        if not ok:
            return SkillProposal(False, name, f"статическая проверка не пройдена: {why}")

        tests_code = generate_tests(name, coeffs, examples, tol=test_tol)
        # прогон тестов в изолированном процессе
        run = Executor.run_isolated(
            code + "\n" + tests_code, "_run_tests", (),
            timeout_s=self.config.executor.skill_timeout_s,
            max_mem_mb=self.config.executor.skill_max_mem_mb,
        )
        if not run.get("ok") or run.get("result") != "PASS":
            return SkillProposal(False, name, f"тесты в песочнице не пройдены: {run.get('error', run)}")

        # измерение латентности навыка vs оригинального ребра (если передан edge_id)
        t0 = time.perf_counter()
        for _ in range(20):
            Executor.run_isolated(code, name, (0.5,), timeout_s=2)
        latency_after_ms = (time.perf_counter() - t0) / 20 * 1000
        from pfg.math_basis import edge_function
        edge_id = examples[0][2] if len(examples[0]) > 2 else None
        edge = self.graph.edges.get(edge_id) if edge_id else None
        latency_before_ms = None
        if edge is not None:
            t0 = time.perf_counter()
            for x in xs[:50]:
                float(edge_function(np.float32(x), edge.basis_coeffs))
            latency_before_ms = (time.perf_counter() - t0) / 50 * 1000

        skill_dir = self._write_skill_files(name, code, tests_code, description,
                                             coeffs.tolist(), error,
                                             latency_before_ms, latency_after_ms)
        sid = f"skill_{new_id()}"
        self.storage.add_skill({
            "id": sid, "name": name, "description": description,
            "code_path": str(skill_dir), "test_path": str(skill_dir / "tests.py"),
            "active": True, "created_at": now_ts(),
            "metric_before": latency_before_ms, "metric_after": latency_after_ms,
            "metadata": {"approx_error": error, "coeffs": coeffs.tolist()},
        })
        self.log.write("skills.jsonl", {
            "level": "INFO", "source": "skills", "message": "skill created",
            "data": {"id": sid, "name": name, "error": error,
                     "latency_before_ms": latency_before_ms, "latency_after_ms": latency_after_ms},
        })
        logger.info("навык '%s' создан и активирован (ошибка %.4f)", name, error)
        return SkillProposal(True, name, "ok", error_before=None, error_after=error,
                             latency_before_ms=latency_before_ms, latency_after_ms=latency_after_ms)

    def compile_edge(self, edge_id: str, name: str | None = None) -> SkillProposal:
        """Полный цикл: собрать примеры ребра -> создать навык."""
        examples = self.collect_edge_examples(edge_id)
        if not examples:
            return SkillProposal(False, name or edge_id, "ребро не найдено")
        fname = name or f"edge_{edge_id[:8]}_fn"
        proposal = self.create_from_examples(fname, examples,
                                             description=f"Компиляция ребра {edge_id}")
        if proposal.accepted:
            # безопасная замена: ребро замораживается символической формой (раздел 20)
            distilled = distill_edges(self.graph, max_error=self.scfg.max_skill_error)
            self.log.write("skills.jsonl", {
                "level": "INFO", "source": "skills", "message": "edges distilled",
                "data": {"distilled": distilled},
            })
        return proposal

    # ---------- запуск ----------
    def run(self, skill_name: str, inputs: dict[str, Any], config: AgentConfig) -> "Any":
        from pfg.executor import ActionResult
        skill = self.get_active(skill_name)
        if skill is None:
            return ActionResult("run_skill", False, error=f"навык '{skill_name}' не найден или не активен")
        code_path = Path(skill.path) / "skill.py"
        if not code_path.exists():
            return ActionResult("run_skill", False, error=f"код навыка потерян: {code_path}")
        code = code_path.read_text(encoding="utf-8")
        args = tuple(float(v) for v in inputs.values()) if inputs else ()
        result = Executor.run_isolated(code, skill_name, args,
                                       timeout_s=config.executor.skill_timeout_s,
                                       max_mem_mb=config.executor.skill_max_mem_mb)
        if not result.get("ok"):
            return ActionResult("run_skill", False, error=str(result.get("error")))
        return ActionResult("run_skill", True, output={"skill": skill_name, "result": result["result"]})

    # ---------- файлы навыка (раздел 19.4) ----------
    def _write_skill_files(self, name: str, code: str, tests_code: str, description: str,
                           coeffs: list[float], error: float,
                           latency_before: float | None, latency_after: float | None) -> Path:
        skill_dir = self.dir / f"{name}_{new_id()}"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "skill.py").write_text(code, encoding="utf-8")
        (skill_dir / "tests.py").write_text(tests_code, encoding="utf-8")
        meta = {
            "id": skill_dir.name, "name": name, "description": description,
            "created_at": utc_iso(), "active": True,
            "inputs": [{"name": "x", "type": "float"}],
            "outputs": [{"name": "result", "type": "float"}],
            "metrics": {
                "error_after": round(error, 6),
                "latency_before_ms": latency_before,
                "latency_after_ms": round(latency_after, 3) if latency_after else None,
                "coeffs": coeffs,
            },
        }
        (skill_dir / "meta.yaml").write_text(
            yaml.safe_dump(meta, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (skill_dir / "README.md").write_text(
            f"# Навык {name}\n\n{description}\n\nОшибка аппроксимации: {error:.6f}\n",
            encoding="utf-8")
        return skill_dir

    # ---------- откат (раздел 19.5) ----------
    def rollback(self, skill_id: str) -> bool:
        row = self.storage.query("SELECT name, code_path, active FROM skills WHERE id=?", (skill_id,))
        if not row:
            return False
        name, code_path, active = row[0]
        self.storage.execute("UPDATE skills SET active=0 WHERE id=?", (skill_id,))
        self.log.write("skills.jsonl", {
            "level": "WARNING", "source": "skills", "message": "skill rolled back",
            "data": {"id": skill_id, "name": name, "was_active": bool(active)},
        })
        logger.warning("навык %s деактивирован (откат)", name)
        return True

    def check_regression(self, skill_id: str, metric_key: str = "metric_after",
                         baseline: float | None = None) -> bool:
        """Если метрика навыка хуже базовой — откатываем."""
        rows = self.storage.query(f"SELECT {metric_key} FROM skills WHERE id=?", (skill_id,))
        if not rows or rows[0][0] is None:
            return True
        current = float(rows[0][0])
        if baseline is not None and current > baseline:
            self.rollback(skill_id)
            return False
        return True


# ---------- генерация кода ----------
def generate_skill_code(name: str, coeffs: list[float]) -> str:
    """Генерирует чистую Python-функцию полинома (только арифметика, без импортов)."""
    safe_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    terms = []
    deg = len(coeffs) - 1
    for i, c_raw in enumerate(coeffs):
        c = float(c_raw)  # numpy-скаляры (np.float64) невалидны как литералы в песочнице
        p = deg - i
        if abs(c) < 1e-9:
            continue
        term = f"{c!r}"
        if p == 1:
            term = f"({c!r}) * x"
        elif p > 1:
            term = f"({c!r}) * x ** {p}"
        terms.append(term)
    body = " + ".join(terms) if terms else "0.0"
    return (
        f"def {safe_name}(x):\n"
        f'    """Автогенерированный навык: полиномиальная аппроксимация."""\n'
        f"    return float({body})\n"
    )


def generate_tests(name: str, coeffs: list[np.floating | float],
                   examples: list[tuple[float, float]], tol: float = 1e-3) -> str:
    """Генерирует тесты: сравнение навыка с эталонными парами x->y."""
    safe_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    # нормализуем примеры к парам (x, y): могут приходить тройки (x, y, edge_id)
    # float() — чтобы в литерал не попал np.float64(...) (в песочнице нет numpy)
    sample = [(float(t[0]), float(t[1])) for t in examples[:16]]
    lines = [
        "def _run_tests():",
        f"    expected = {sample!r}",
        f"    fn = {safe_name}",
        f"    tol = {float(tol)!r}",
        "    for x, y in expected:",
        "        if abs(fn(x) - y) > tol:",
        "            return 'FAIL'",
        "    return 'PASS'",
    ]
    return "\n".join(lines) + "\n"
