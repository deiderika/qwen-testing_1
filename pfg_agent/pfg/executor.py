"""Исполнитель PFG-Agent: выполняет только разрешённые действия в песочнице.

Файловые операции — строго внутри sandbox/ (проверка повторная, defence-in-depth).
Сгенерированный код навыков запускается в отдельном процессе с ограничениями
времени и памяти (RLIMIT_AS при доступности), без сети и shell.
"""
from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pfg.config import AgentConfig
from pfg.memory import Memory
from pfg.safety import Action, SafetyGuard
from pfg.utils import JsonlLogger, new_id

logger = logging.getLogger("pfg.executor")


@dataclass
class ActionResult:
    action: str
    success: bool
    output: Any = None
    error: str | None = None
    duration_hint: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action, "success": self.success,
            "output": _jsonable(self.output), "error": self.error,
            "metadata": self.metadata,
        }


def _jsonable(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


_CALC_RE = re.compile(r"^\s*[-+*/%\d.,()\s]+\s*$")


def _safe_calculate(a: float, b: float, op: str) -> float:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if b == 0:
            raise ZeroDivisionError("деление на ноль")
        return a / b
    raise ValueError(f"неизвестная операция: {op}")


def _sandbox_path(config: AgentConfig, raw: str) -> Path:
    """Разрешает путь только внутри allowed_dirs; иначе PermissionError."""
    from pfg.safety import is_inside_sandbox
    if not is_inside_sandbox(raw, config.executor.allowed_dirs, config.root):
        raise PermissionError(f"путь вне песочницы: {raw}")
    p = Path(raw)
    return p if p.is_absolute() else (config.root / p)


def _isolated_target(conn, code, fname, args, timeout_s, max_mem_mb):
    """Точка входа дочернего процесса (модульный уровень — picklable для spawn)."""
    try:
        try:  # мягкие лимиты ресурсов (Unix); на Windows молча пропускаем
            import resource
            # Python-интерпретатор сам по себе резервирует ~60-90MB виртуальной
            # памяти (RLIMIT_AS), поэтому лимит берётся с запасом, иначе spawn-
            # процесс падает с MemoryError ещё до выполнения кода навыка.
            mem_bytes = int((max_mem_mb + 384) * 1024 * 1024)
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            if hard != resource.RLIM_INFINITY:
                mem_bytes = min(mem_bytes, hard)
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            resource.setrlimit(resource.RLIMIT_CPU, (timeout_s, timeout_s + 1))
            resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        except (ImportError, ValueError, OSError):
            pass
        env: dict[str, Any] = {"__builtins__": _restricted_builtins(), "math": math}
        exec(compile(code, "<skill>", "exec"), env)  # noqa: S102 — код прошёл статическую проверку
        fn = env.get(fname)
        if not callable(fn):
            raise NameError(f"функция {fname} не определена")
        conn.send({"ok": True, "result": fn(*args)})
    except BaseException as exc:  # noqa: BLE001
        conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        try:
            conn.close()
        except OSError:
            pass


class Executor:
    """Безопасное выполнение действий плана."""

    def __init__(self, config: AgentConfig, memory: Memory, safety: SafetyGuard) -> None:
        self.config = config
        self.memory = memory
        self.safety = safety
        self.audit = JsonlLogger(config.log_dir)
        self.skill_registry: Any = None  # подставляется агентом (SkillsManager)
        self.web_search: Any = None      # подставляется агентом (WebSearchEngine)

    # ---------- публичный API ----------
    def execute(self, actions: list[Action]) -> list[ActionResult]:
        results: list[ActionResult] = []
        for a in actions:
            results.append(self.execute_one(a))
        return results

    def execute_one(self, action: Action) -> ActionResult:
        decision = self.safety.check(action)
        if not decision.allowed:
            res = ActionResult(action.name, False, error=f"запрещено безопасностью: {decision.reason}",
                               metadata={"needs_approval": decision.needs_approval})
            self._log(action, res)
            return res
        handler = getattr(self, f"_do_{action.name}", None)
        if handler is None:
            res = ActionResult(action.name, False, error=f"нет обработчика для '{action.name}'")
        else:
            try:
                res = handler(action)
            except Exception as exc:  # noqa: BLE001 — ошибка действия не роняет агента
                logger.exception("действие %s упало", action.name)
                res = ActionResult(action.name, False, error=f"{type(exc).__name__}: {exc}")
        self._log(action, res)
        self.memory.push_action(res.to_dict())
        return res

    def _log(self, action: Action, res: ActionResult) -> None:
        """Аудит исполнения: JSONL + (опционально) SQLite-события."""
        self.audit.write("actions.jsonl", {
            "level": "INFO" if res.success else "WARNING",
            "source": "executor",
            "message": "action completed",
            "data": {"action": action.name, "success": res.success, "error": res.error},
        })

    # ---------- обработчики ----------
    def _do_remember(self, action: Action) -> ActionResult:
        text = str(action.args.get("text", "")).strip()
        if not text:
            return ActionResult("remember", False, error="пустой текст факта")
        fid = self.memory.store_fact(text, importance=0.7, source="user")
        return ActionResult("remember", True, output={"fact_id": fid, "text": text})

    def _do_recall(self, action: Action) -> ActionResult:
        query = str(action.args.get("query", "") or action.args.get("text", ""))
        hits = self.memory.retrieve(query, top_k=3)
        return ActionResult("recall", bool(hits),
                            output=[{"id": f.id, "text": f.text, "score": round(s, 3)} for f, s in hits])

    def _do_search_memory(self, action: Action) -> ActionResult:
        return self._do_recall(action)

    def _do_web_search(self, action: Action) -> ActionResult:
        """Мультиисточниковый поиск. Движок подставляется агентом (executor.web_search)."""
        query = str(action.args.get("query", "") or action.args.get("text", ""))
        if not query:
            return ActionResult("web_search", False, error="пустой поисковый запрос")
        engine = getattr(self, "web_search", None)
        if engine is None:
            return ActionResult("web_search", False, error="веб-поиск не подключён (нет fetch-контура)")
        rep = engine.search_logged(query)
        return ActionResult("web_search", rep.success,
                            output=rep.to_dict(),
                            error="; ".join(rep.errors) if rep.errors and not rep.success else None)

    def _do_write_file(self, action: Action) -> ActionResult:
        path = _sandbox_path(self.config, str(action.args.get("path", "")))
        content = str(action.args.get("content", ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ActionResult("write_file", True, output={"path": str(path), "bytes": len(content.encode())})

    def _do_read_file(self, action: Action) -> ActionResult:
        path = _sandbox_path(self.config, str(action.args.get("path", "")))
        if not path.exists():
            return ActionResult("read_file", False, error=f"файл не найден: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        return ActionResult("read_file", True, output={"path": str(path), "content": text[:4096]})

    def _do_list_files(self, action: Action) -> ActionResult:
        path = _sandbox_path(self.config, str(action.args.get("path", "sandbox")))
        if not path.exists():
            return ActionResult("list_files", True, output=[])
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
        return ActionResult("list_files", True, output=entries)

    def _do_delete_file(self, action: Action) -> ActionResult:
        path = _sandbox_path(self.config, str(action.args.get("path", "")))
        if path.is_file():
            path.unlink()
            return ActionResult("delete_file", True, output={"deleted": str(path)})
        return ActionResult("delete_file", False, error="файл не найден")

    def _do_calculate(self, action: Action) -> ActionResult:
        try:
            a = float(action.args.get("a", 0))
            b = float(action.args.get("b", 0))
            op = str(action.args.get("op", "+"))
            expr = str(action.args.get("expr", ""))
            if expr and _CALC_RE.match(expr):
                value = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 — whitelist: только цифры и операторы
            else:
                value = _safe_calculate(a, b, op)
            return ActionResult("calculate", True, output={"a": a, "b": b, "op": op, "result": value})
        except (ValueError, ZeroDivisionError) as exc:
            return ActionResult("calculate", False, error=str(exc))

    def _do_ask_user(self, action: Action) -> ActionResult:
        return ActionResult("ask_user", True, output={"question": action.args.get("question", "Уточните задачу.")})

    def _do_wait(self, action: Action) -> ActionResult:
        return ActionResult("wait", True, output={"waited": True})

    def _do_run_skill(self, action: Action) -> ActionResult:
        skill_name = str(action.args.get("skill", action.args.get("name", "")))
        if self.skill_registry is None:
            return ActionResult("run_skill", False, error="реестр навыков не подключён")
        return self.skill_registry.run(skill_name, action.args.get("inputs", {}), self.config)

    def _do_create_skill(self, action: Action) -> ActionResult:
        if self.skill_registry is None:
            return ActionResult("create_skill", False, error="реестр навыков не подключён")
        proposal = self.skill_registry.create_from_examples(
            name=str(action.args.get("name", f"skill_{new_id()}")),
            examples=action.args.get("examples", []),
            description=str(action.args.get("description", "")),
        )
        return ActionResult("create_skill", proposal.accepted,
                            output={"skill": proposal.skill_name, "reason": proposal.reason})

    # ---------- изолированный запуск кода ----------
    @staticmethod
    def run_isolated(func_code: str, func_name: str, call_args: tuple,
                     timeout_s: int = 5, max_mem_mb: int = 128) -> dict[str, Any]:
        """Запуск функции из исходника в отдельном процессе с лимитами времени/памяти.

        Возвращает {'ok': bool, 'result'|'error': ...}. Сеть и файлы вне песочницы
        недоступны: процесс получает только код и аргументы, builtins ограничены.
        """
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(target=_isolated_target,
                           args=(child_conn, func_code, func_name, call_args,
                                 timeout_s, max_mem_mb), daemon=True)
        proc.start()
        child_conn.close()
        got = parent_conn.poll(timeout_s)
        payload = parent_conn.recv() if got else {"ok": False, "error": "timeout"}
        proc.join(1.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(1.0)
            payload = {"ok": False, "error": "process killed (limit exceeded)"}
        parent_conn.close()
        return payload


_SAFE_BUILTINS_NAMES = (
    "abs", "min", "max", "sum", "round", "len", "range", "int", "float", "str",
    "bool", "list", "tuple", "dict", "set", "frozenset", "enumerate", "zip",
    "map", "filter", "sorted", "reversed", "isinstance", "print", "any", "all",
    "divmod", "pow", "repr", "format", "Exception", "ValueError", "ZeroDivisionError",
    "TypeError", "KeyError", "IndexError", "RuntimeError", "StopIteration",
)


def _restricted_builtins() -> dict[str, Any]:
    import builtins
    return {name: getattr(builtins, name) for name in _SAFE_BUILTINS_NAMES if hasattr(builtins, name)}



