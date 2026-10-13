"""Тесты агента: полный цикл remember/recall, файлы, отказ от опасных действий."""
import json

import pytest

from pfg.agent import PFGAgent
from pfg.config import AgentConfig
from pfg.storage import Storage


@pytest.fixture()
def agent(tmp_path):
    cfg = AgentConfig.load()
    cfg.root = tmp_path
    cfg.log_dir = str(tmp_path / "logs")
    storage = Storage(tmp_path / "a.sqlite")
    ag = PFGAgent(cfg, storage=storage)
    yield ag
    ag.close()


def test_remember_then_recall(agent):
    r1 = agent.run_task("Запомни, что любимый цвет синий")
    assert r1.success
    r2 = agent.run_task("Какой любимый цвет?")
    assert r2.success, f"recall не сработал: {r2.errors} / {r2.answer}"
    assert "синий" in (r2.answer or "").lower()
    assert r2.memory_used, "должна быть использована память"


def test_result_is_json_serializable(agent):
    res = agent.run_task("Прибавь 2 к 3")
    payload = res.to_dict()
    s = json.dumps(payload, ensure_ascii=False)
    assert '"success"' in s


def test_arithmetic(agent):
    res = agent.run_task("Сколько будет 5 умножить на 4?")
    assert res.success and "20" in (res.answer or "")


def test_file_task_in_sandbox(agent):
    res = agent.run_task('Создай файл sandbox/test.txt с текстом hello')
    assert res.success, res.errors
    created = agent.config.resolve_path("sandbox/test.txt")
    assert created.exists()
    assert created.read_text(encoding="utf-8") == "hello"


def test_agent_refuses_dangerous_action(agent):
    from pfg.safety import Action
    res = agent.executor.execute_one(Action("shell_command", {"command": "ls -la"}))
    assert not res.success
    assert "безопасност" in (res.error or "").lower() or "shell" in (res.error or "").lower()

    res2 = agent.executor.execute_one(Action("write_file", {"path": "/etc/passwd", "content": "x"}))
    assert not res2.success

    res3 = agent.executor.execute_one(Action("network_request", {"url": "http://evil.example"}))
    assert not res3.success


def test_status_and_save_load(agent):
    agent.run_task("Запомни, что целевая платформа Raspberry Pi 4")
    st = agent.status()
    assert st["facts"] >= 1
    assert st["graph"]["nodes"] >= 7
    agent.save()
    agent.load()


def test_reflection_creates_lesson_on_error(agent):
    from pfg.safety import Action
    bad = agent.executor.execute_one(Action("read_file", {"path": "sandbox/nonexistent_zz.txt"}))
    assert not bad.success
    lessons = agent.reflection.reflect(goal="прочитай sandbox/nonexistent_zz.txt",
                                       plan=None, results=[bad])
    assert lessons, "рефлексия должна создать урок после ошибки"
