"""Тесты пластичного функционального графа."""
import numpy as np
import pytest

from pfg.config import AgentConfig
from pfg.graph import PlasticFunctionGraph, distill_edges, fit_symbolic


@pytest.fixture()
def config(tmp_path) -> AgentConfig:
    cfg = AgentConfig.load()
    cfg.root = tmp_path
    return cfg


def make_graph(config, n=8):
    g = PlasticFunctionGraph(config)
    ids = [g.add_node("concept") for _ in range(n)]
    for i in range(n):
        g.add_edge(ids[i], ids[(i + 1) % n])
    return g, ids


def test_add_node_and_edge(config):
    g, ids = make_graph(config, 4)
    assert len(g.nodes) == 4
    assert len(g.edges) == 4
    assert g.nodes[ids[0]].type == "concept"


def test_unknown_node_type_rejected(config):
    g = PlasticFunctionGraph(config)
    with pytest.raises(ValueError):
        g.add_node("hacker")


def test_step_changes_states(config):
    g, _ = make_graph(config, 6)
    before = {nid: v.vector.copy() for nid, v in g.nodes.items()}
    for _ in range(5):
        g.step({})
    changed = any(not np.allclose(g.nodes[nid].vector, before[nid]) for nid in g.nodes)
    assert changed  # шаг графа меняет состояния узлов
    assert all(np.all(np.isfinite(g.nodes[nid].vector)) for nid in g.nodes)


def test_topk_limits_contributions(config):
    config.graph.top_k_edges = 2
    g = PlasticFunctionGraph(config)
    dst = g.add_node("concept", "dst")
    srcs = [g.add_node("sensor", f"s{i}") for i in range(10)]
    for s in srcs:
        g.add_edge(s, dst)
    # после шага только 2 ребра получили использование на этом узле за шаг
    used_before = sum(e.usage_count for e in g.incoming(dst))
    g.step({})
    used_after = sum(e.usage_count for e in g.incoming(dst))
    assert used_after - used_before <= 2


def test_remove_edge(config):
    g, ids = make_graph(config, 3)
    eid = list(g.edges)[0]
    g.remove_edge(eid)
    assert eid not in g.edges


def test_salience_update_and_prune(config):
    g, _ = make_graph(config, 4)
    eid = list(g.edges)[0]
    g.update_salience(eid, 0.5)
    assert g.edges[eid].salience > 0.5
    g.edges[eid].salience = 0.01
    g.edges[eid].usage_count = config.learning.min_edge_age_for_removal + 1
    stats = g.prune_and_grow()
    assert stats["removed"] >= 1


def test_save_load_roundtrip(config, tmp_path):
    from pfg.storage import Storage
    storage = Storage(tmp_path / "g.sqlite")
    g, _ = make_graph(config, 5)
    g.step({})
    g.save(storage)
    g2 = PlasticFunctionGraph(config)
    g2.load(storage)
    assert set(g2.nodes) == set(g.nodes)
    assert set(g2.edges) == set(g.edges)
    for nid in g.nodes:
        assert np.allclose(g2.nodes[nid].vector, g.nodes[nid].vector)
    storage.close()


def test_export_npz(config, tmp_path):
    g, _ = make_graph(config, 4)
    path = tmp_path / "graph.npz"
    g.export_npz(str(path))
    data = np.load(str(path), allow_pickle=True)
    keys = list(data.files)
    assert any(k.startswith("node::") for k in keys)
    assert any(k.startswith("edge::") for k in keys)


def test_simulate_horizon(config):
    g, _ = make_graph(config, 4)
    traj = g.simulate({}, horizon=5)
    assert len(traj) == 5


def test_symbolic_fit_and_distill(config):
    xs = np.linspace(-1, 1, 40)
    ys = 0.73 * xs + 0.12
    result = fit_symbolic(xs, ys, max_error=0.05)
    assert result is not None
    expr, err = result
    assert err < 0.05

    g, _ = make_graph(config, 3)
    # ребро с линейным базисом должно выжиматься
    edge = next(iter(g.edges.values()))
    edge.basis_coeffs = np.array([0.12, 0.73, 0, 0, 0, 0], dtype=np.float32)
    n = distill_edges(g, max_error=0.05)
    assert n >= 1
    assert edge.frozen and edge.symbolic_expr
