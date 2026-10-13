"""Тесты навыков: создание, статическая проверка, запуск в песочнице, плохой навык, откат."""
import numpy as np
import pytest

from pfg.agent import build_default_graph
from pfg.config import AgentConfig
from pfg.memory import Memory
from pfg.skills import SkillsManager, generate_skill_code, static_check
from pfg.storage import Storage


@pytest.fixture()
def manager(tmp_path):
    cfg = AgentConfig.load()
    cfg.root = tmp_path
    cfg.log_dir = str(tmp_path / "logs")
    storage = Storage(tmp_path / "s.sqlite")
    graph = build_default_graph(cfg)
    mem = Memory(cfg, storage)
    mgr = SkillsManager(cfg, storage, mem, graph)
    yield mgr, cfg, storage, graph
    storage.close()


def test_static_check_blocks_imports():
    ok, _ = static_check("def f(x):\n    return x + 1\n", ["math"])
    assert ok
    bad, why = static_check("import os\ndef f(x):\n    return os.system('rm -rf /')\n", ["math"])
    assert not bad and "whitelist" in why
    bad2, _ = static_check("def f(x):\n    return open('/etc/passwd').read()\n", ["math"])
    assert not bad2
    bad3, _ = static_check("def f(x):\n    return eval(x)\n", ["math"])
    assert not bad3


def test_generate_skill_code_runs(manager):
    mgr, cfg, _, _ = manager
    code = generate_skill_code("lin_fn", [2.0, 1.0])  # y = 2x + 1
    from pfg.executor import Executor
    out = Executor.run_isolated(code, "lin_fn", (3.0,), timeout_s=10)
    assert out["ok"], out
    assert abs(out["result"] - 7.0) < 1e-6


def test_isolated_run_blocks_timeout_and_open(manager):
    from pfg.executor import Executor
    out = Executor.run_isolated("def slow():\n    while True:\n        pass\n", "slow", (), timeout_s=2)
    assert not out["ok"]
    out2 = Executor.run_isolated("def evil():\n    return open('/etc/passwd').read()\n", "evil", (), timeout_s=5)
    assert not out2["ok"]  # open недоступен в restricted builtins


def test_create_good_skill(manager):
    mgr, cfg, _, graph = manager
    xs = np.linspace(-1, 1, 40)
    ys = 0.7 * xs + 0.1
    examples = [(float(x), float(y)) for x, y in zip(xs, ys)]
    proposal = mgr.create_from_examples("sum_approx", examples, "линейная аппроксимация")
    assert proposal.accepted, proposal.reason
    skills = mgr.list_skills()
    assert any(s.name == "sum_approx" and s.active for s in skills)
    skill_dir = next(s.path for s in skills if s.name == "sum_approx")
    assert (skill_dir / "skill.py").exists()
    assert (skill_dir / "meta.yaml").exists()
    assert (skill_dir / "tests.py").exists()
    assert (skill_dir / "README.md").exists()
    # запуск навыка через реестр
    res = mgr.run("sum_approx", {"x": 0.5}, cfg)
    assert res.success
    assert abs(res.output["result"] - 0.45) < 0.05


def test_bad_skill_not_activated(manager):
    mgr, cfg, _, _ = manager
    rng = np.random.default_rng(1)
    xs = rng.uniform(-1, 1, 40)
    ys = rng.uniform(-1, 1, 40)  # шум — полином не впишется в допуск
    examples = [(float(x), float(y)) for x, y in zip(xs, ys)]
    proposal = mgr.create_from_examples("noise_skill", examples)
    assert not proposal.accepted, "навык с высокой ошибкой нельзя активировать"
    assert not any(s.name == "noise_skill" for s in mgr.list_skills())


def test_too_few_examples_rejected(manager):
    mgr, _, _, _ = manager
    proposal = mgr.create_from_examples("tiny", [(0.1, 0.2), (0.3, 0.4)])
    assert not proposal.accepted


def test_rollback(manager):
    mgr, cfg, _, _ = manager
    xs = np.linspace(-1, 1, 30)
    examples = [(float(x), float(2 * x)) for x in xs]
    proposal = mgr.create_from_examples("to_rollback", examples)
    assert proposal.accepted
    sid = next(s.id for s in mgr.list_skills() if s.name == "to_rollback")
    assert mgr.rollback(sid)
    assert mgr.get_active("to_rollback") is None


def test_compile_edge_cycle(manager):
    mgr, cfg, _, graph = manager
    edge = next(iter(graph.edges.values()))
    # сделаем ребро линейным, чтобы аппроксимация прошла допуск
    edge.basis_coeffs = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    proposal = mgr.compile_edge(edge.id, name="compiled_edge")
    assert proposal.accepted, proposal.reason
