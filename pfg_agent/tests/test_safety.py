"""Тесты безопасности: песочница, сеть/shell/деньги, подтверждения, аудит."""
import pytest

from pfg.config import AgentConfig
from pfg.safety import Action, SafetyGuard, is_inside_sandbox


@pytest.fixture()
def guard(tmp_path) -> SafetyGuard:
    cfg = AgentConfig.load()
    cfg.root = tmp_path
    cfg.log_dir = str(tmp_path / "logs")
    (tmp_path / "sandbox").mkdir(exist_ok=True)
    return SafetyGuard(cfg)


def test_write_inside_sandbox_allowed(guard):
    d = guard.check(Action("write_file", {"path": "sandbox/a.txt", "content": "x"}))
    assert d.allowed and d.risk == "medium"


def test_write_outside_sandbox_blocked(guard):
    d = guard.check(Action("write_file", {"path": "../evil.txt", "content": "x"}))
    assert not d.allowed
    d2 = guard.check(Action("write_file", {"path": "/tmp/evil.txt", "content": "x"}))
    assert not d2.allowed


def test_traversal_blocked(guard):
    d = guard.check(Action("read_file", {"path": "sandbox/../../etc/passwd"}))
    assert not d.allowed


def test_sensitive_paths_blocked(guard):
    for p in ("sandbox/../.ssh/id_rsa", ".env", "sandbox/token.txt"):
        d = guard.check(Action("read_file", {"path": p}))
        assert not d.allowed, p


def test_network_blocked_by_default(guard):
    d = guard.check(Action("network_request", {"url": "http://example.com"}))
    assert not d.allowed and d.risk == "high"


def test_shell_blocked_by_default_and_dangerous_always(guard):
    d = guard.check(Action("shell_command", {"command": "echo hi"}))
    assert not d.allowed
    # даже если shell включён — опасная команда блокируется навсегда
    guard.ecfg.allow_shell = True
    d2 = guard.check(Action("shell_command", {"command": "rm -rf /"}))
    assert not d2.allowed and d2.risk == "critical"
    d3 = guard.check(Action("shell_command", {"command": "sudo reboot"}))
    assert not d3.allowed


def test_shell_needs_human_approval(guard):
    guard.ecfg.allow_shell = True
    d = guard.check(Action("shell_command", {"command": "echo hi"}))
    assert not d.allowed and d.needs_approval
    guard.grant_approval("shell")
    d2 = guard.check(Action("shell_command", {"command": "echo hi"}))
    assert d2.allowed


def test_money_blocked(guard):
    d = guard.check(Action("money_transfer", {"amount": 100}))
    assert not d.allowed and d.risk == "critical"


def test_unknown_action_blocked(guard):
    d = guard.check(Action("launch_missiles", {}))
    assert not d.allowed


def test_low_risk_actions_allowed(guard):
    for name, args in [("remember", {"text": "x"}), ("recall", {"query": "x"}),
                       ("calculate", {"a": 1, "b": 2, "op": "+"}), ("wait", {})]:
        assert guard.check(Action(name, args)).allowed


def test_audit_written(guard, tmp_path):
    guard.check(Action("write_file", {"path": "/etc/x", "content": "y"}))
    log_file = tmp_path / "logs" / "actions.jsonl"
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert lines and '"allowed": false' in lines[-1]


def test_check_plan_truncates_on_high_risk(guard):
    actions = [
        Action("remember", {"text": "ok"}),
        Action("network_request", {"url": "http://x"}),
        Action("remember", {"text": "не должен выполниться"}),
    ]
    allowed, decisions = guard.check_plan(actions)
    assert [a.name for a in allowed] == ["remember"]
    assert len(decisions) == 2  # план обрезан на запретe high


def test_is_inside_sandbox_helper(tmp_path):
    (tmp_path / "sandbox").mkdir()
    assert is_inside_sandbox("sandbox/a.txt", ["sandbox"], tmp_path)
    assert not is_inside_sandbox("skills/a.py", ["sandbox"], tmp_path)
    assert not is_inside_sandbox("sandbox/../skills/a.py", ["sandbox"], tmp_path)
