"""Модуль безопасности PFG-Agent: правила, уровни риска, подтверждения, аудит.

Безопасность важнее автономности (принцип 2.3). По умолчанию запрещены:
сеть, shell, финансовые операции, доступ к файлам вне песочницы.
Каждое решение пишется в JSONL-аудит и SQLite-события.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pfg.config import AgentConfig
from pfg.utils import JsonlLogger, new_id, now_ts

logger = logging.getLogger("pfg.safety")

RISK_LEVELS = ("low", "medium", "high", "critical")
_RISK_ORDER = {r: i for i, r in enumerate(RISK_LEVELS)}

# Категории действий по уровням риска (раздел 17.1–17.2)
ACTION_RISK: dict[str, str] = {
    "remember": "low",
    "recall": "low",
    "search_memory": "low",
    "read_file": "low",
    "list_files": "low",
    "calculate": "low",
    "wait": "low",
    "ask_user": "low",
    "write_file": "medium",
    "run_skill": "medium",
    "create_skill": "medium",
    "network_request": "high",
    "shell_command": "high",
    "delete_file": "high",
    "money_transfer": "critical",
}

# Опасные паттерны в аргументах (shell/путь) — блокируются всегда
_SHELL_DENY_RE = re.compile(
    r"rm\s+-rf|sudo|mkfs|dd\s+if=|:\(\)\s*\{.*\};|shutdown|reboot|"
    r"/etc/|/root/|id_rsa|\.ssh|\.gnupg|wallet|private[_-]?key|"
    r"env|printenv|export\s|curl\s|wget\s|nc\s|chmod\s+777",
    re.IGNORECASE,
)

SENSITIVE_PATH_PARTS = ("/etc", "/root", "/home", "~", ".ssh", ".gnupg", "id_rsa", "wallet", ".env", "token", "secret", "key")


@dataclass
class Action:
    """Действие агента: имя, аргументы, назначение."""
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    purpose: str = ""
    action_id: str = field(default_factory=new_id)

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.name, "args": self.args, "purpose": self.purpose, "id": self.action_id}


@dataclass
class SafetyDecision:
    allowed: bool
    risk: str = "low"
    reason: str = ""
    needs_approval: bool = False
    action_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed, "risk": self.risk, "reason": self.reason,
            "needs_approval": self.needs_approval, "action_id": self.action_id,
        }


def is_inside_sandbox(path: str | Path, allowed_dirs: list[str], root: Path) -> bool:
    """Проверяет, что путь после нормализации остаётся внутри разрешённого каталога."""
    try:
        resolved = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    except OSError:
        return False
    for allowed in allowed_dirs:
        adir = Path(allowed)
        if not adir.is_absolute():
            adir = (root / adir).resolve()
        else:
            adir = adir.resolve()
        try:
            resolved.relative_to(adir)
            return True
        except ValueError:
            continue
    return False


def has_sensitive_path(path: str) -> bool:
    p = str(path).lower()
    return any(part in p for part in SENSITIVE_PATH_PARTS)


class SafetyGuard:
    """Страж безопасности: проверяет действия, требует подтверждений, ведёт аудит."""

    def __init__(self, config: AgentConfig, approvals: set[str] | None = None) -> None:
        self.config = config
        self.ecfg = config.executor
        self.approvals: set[str] = approvals or set()  # категории, одобренные человеком в этой сессии
        self.audit = JsonlLogger(config.log_dir)
        self.storage: Any = None  # опционально: Storage для записи событий
        self.violations = 0

    # ---------- публичный API ----------
    def check(self, action: Action) -> SafetyDecision:
        decision = self._check(action)
        decision.action_id = action.action_id
        self._audit(action, decision)
        return decision

    def check_plan(self, actions: list[Action]) -> tuple[list[Action], list[SafetyDecision]]:
        """Возвращает (разрешённые действия, все решения). План обрезается на первом запрете high/critical."""
        allowed: list[Action] = []
        decisions: list[SafetyDecision] = []
        for a in actions:
            d = self.check(a)
            decisions.append(d)
            if d.allowed:
                allowed.append(a)
            elif d.risk in ("high", "critical"):
                logger.warning("план прерван на действии %s: %s", a.name, d.reason)
                break
        return allowed, decisions

    def grant_approval(self, category: str) -> None:
        """Человек подтвердил категорию действий (например 'network')."""
        self.approvals.add(category)
        logger.info("выдано подтверждение на категорию: %s", category)

    # ---------- внутренняя логика ----------
    def _category_of(self, action_name: str) -> str | None:
        if action_name == "network_request":
            return "network"
        if action_name == "shell_command":
            return "shell"
        if action_name == "money_transfer":
            return "money"
        return None

    def _check(self, action: Action) -> SafetyDecision:
        name = action.name
        risk = ACTION_RISK.get(name, "high")  # неизвестное действие — высокий риск

        # 1. Неизвестные действия запрещены
        if name not in ACTION_RISK:
            return SafetyDecision(False, "high", f"неизвестное действие: {name}")

        # 2. Файловые операции — только песочница
        if name in ("write_file", "read_file", "list_files", "delete_file"):
            raw_path = str(action.args.get("path", ""))
            if not raw_path:
                return SafetyDecision(False, risk, "не указан путь файла")
            if has_sensitive_path(raw_path):
                return SafetyDecision(False, "high", f"запрещённый чувствительный путь: {raw_path}")
            if not is_inside_sandbox(raw_path, self.ecfg.allowed_dirs, self.config.root):
                return SafetyDecision(False, risk, f"путь вне песочницы: {raw_path}")

        # 3. Shell: запрещён по умолчанию; опасные паттерны — навсегда
        if name == "shell_command":
            cmd = str(action.args.get("command", ""))
            if _SHELL_DENY_RE.search(cmd):
                return SafetyDecision(False, "critical", "команда содержит опасный паттерн")
            if not self.ecfg.allow_shell:
                return SafetyDecision(False, "high", "shell отключён конфигурацией")
            return self._approval_gate(action, "shell", risk)

        # 4. Сеть
        if name == "network_request":
            if not self.ecfg.allow_network:
                return SafetyDecision(False, "high", "сеть отключена конфигурацией")
            return self._approval_gate(action, "network", risk)

        # 5. Финансы
        if name == "money_transfer":
            if not self.ecfg.allow_money:
                return SafetyDecision(False, "critical", "финансовые операции отключены")
            return self._approval_gate(action, "money", risk)

        # 6. Остальные low/medium — разрешены (с проверкой путей выше, если применимо)
        return SafetyDecision(True, risk, "ok")

    def _approval_gate(self, action: Action, category: str, risk: str) -> SafetyDecision:
        needs = category in self.ecfg.require_human_approval_for
        if needs and category not in self.approvals:
            return SafetyDecision(False, risk,
                                  f"действие '{action.name}' требует подтверждения человека ({category})",
                                  needs_approval=True)
        return SafetyDecision(True, risk, f"подтверждено человеком: {category}" if needs else "ok")

    # ---------- аудит ----------
    def _audit(self, action: Action, decision: SafetyDecision) -> None:
        if not decision.allowed:
            self.violations += 1
        record = {
            "level": "INFO" if decision.allowed else "WARNING",
            "source": "safety",
            "message": "action checked",
            "data": {
                "goal": getattr(self, "_current_goal", ""),
                "action": action.name,
                "args_keys": sorted(action.args.keys()),
                "risk": decision.risk,
                "allowed": decision.allowed,
                "reason": decision.reason,
                "needs_approval": decision.needs_approval,
            },
        }
        self.audit.write("actions.jsonl", record)
        if self.storage is not None:
            try:
                self.storage.log_event(record["level"], "safety", record["message"],
                                       now_ts(), record["data"])
            except Exception:  # noqa: BLE001 — аудит не должен ронять агента
                logger.exception("не удалось записать событие аудита")

    def set_goal_context(self, goal: str) -> None:
        self._current_goal = goal

    @staticmethod
    def risk_greater(a: str, b: str) -> bool:
        return _RISK_ORDER.get(a, 99) > _RISK_ORDER.get(b, 99)
