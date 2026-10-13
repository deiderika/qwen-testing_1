"""Тесты памяти: факты, поиск, дедупликация, эпизоды."""
import pytest

from pfg.config import AgentConfig
from pfg.memory import Episode, Memory, embed_text
from pfg.storage import Storage


@pytest.fixture()
def memory(tmp_path) -> Memory:
    cfg = AgentConfig.load()
    cfg.root = tmp_path
    storage = Storage(tmp_path / "m.sqlite")
    mem = Memory(cfg, storage)
    yield mem
    storage.close()


def test_embed_text_normalized():
    import numpy as np
    v = embed_text("любимый цвет синий", 32)
    assert v.shape == (32,)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_store_and_retrieve_fact(memory):
    fid = memory.store_fact("любимый цвет синий", source="user")
    hits = memory.retrieve("какой любимый цвет?", top_k=1)
    assert hits, "факт не найден"
    assert "синий" in hits[0][0].text
    assert hits[0][0].id == fid


def test_duplicate_updates_not_duplicates(memory):
    fid1 = memory.store_fact("столица франции париж")
    fid2 = memory.store_fact("столица франции париж")
    assert fid1 == fid2, "дубликат должен обновлять существующий факт"
    facts = memory.all_facts()
    matching = [f for f in facts if f.text == "столица франции париж"]
    assert len(matching) == 1


def test_relevance_prefers_related(memory):
    memory.store_fact("любимый цвет синий")
    memory.store_fact("рецепт борща на зиму")
    hits = memory.retrieve("какой любимый цвет?", top_k=1)
    assert "цвет" in hits[0][0].text


def test_episode_roundtrip(memory):
    ep = Episode(id="ep_test1", goal="тест", context=["a"], actions=[{"action": "wait"}],
                 result="ok", reward=1.0, error=None)
    memory.store_episode(ep)
    eps = memory.recent_episodes(limit=5)
    assert any(e["id"] == "ep_test1" for e in eps)


def test_lesson_stored_as_fact(memory):
    memory.store_lesson("нельзя писать вне песочницы", importance=0.9)
    hits = memory.retrieve("песочница запрет", top_k=2)
    assert any("УРОК" in f.text for f, _ in hits)


def test_working_memory(memory):
    memory.set_goal("цель")
    assert memory.current_goal == "цель"
    memory.push_action({"action": "wait", "success": True})
    assert memory.recent_actions[-1]["action"] == "wait"
