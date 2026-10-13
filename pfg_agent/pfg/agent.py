"""Агент PFG-Agent: объединяет граф, память, планировщик, исполнитель, рефлексию, безопасность.

Основной цикл (раздел 14.1): цель -> контекст -> состояние -> план -> проверка
безопасности -> исполнение -> наблюдение -> эпизод -> рефлексия -> ответ.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pfg.config import AgentConfig
from pfg.executor import ActionResult, Executor
from pfg.graph import PlasticFunctionGraph
from pfg.memory import Episode, Memory
from pfg.planner import Planner, parse_goal
from pfg.reflection import ReflectionEngine
from pfg.safety import Action, SafetyGuard
from pfg.skills import SkillsManager
from pfg.storage import Storage
from pfg.utils import JsonlLogger, new_id

logger = logging.getLogger("pfg.agent")


@dataclass
class AgentResult:
    goal: str
    success: bool
    answer: str | None = None
    actions: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    memory_used: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success, "answer": self.answer,
            "confidence": round(self.confidence, 3),
            "memory_used": self.memory_used,
            "actions": self.actions, "errors": self.errors,
            "metadata": self.metadata,
        }


def build_default_graph(config: AgentConfig) -> PlasticFunctionGraph:
    """Мини-граф по умолчанию: сенсоры -> концепты -> reward/risk; loop-рёбра для динамики."""
    g = PlasticFunctionGraph(config)
    sensor_in = g.add_node("sensor", "sensor_input")
    concept_a = g.add_node("concept", "concept_a")
    concept_b = g.add_node("concept", "concept_b")
    abstract = g.add_node("abstract", "abstract_state")
    mem_node = g.add_node("memory", "memory_node")
    reward = g.add_node("reward", "value_reward")
    risk = g.add_node("risk", "value_risk")
    for src, dst in [(sensor_in, concept_a), (sensor_in, concept_b),
                     (concept_a, abstract), (concept_b, abstract),
                     (abstract, reward), (abstract, risk),
                     (mem_node, concept_a), (reward, concept_b), (risk, concept_a)]:
        g.add_edge(src, dst)
    # self-loop рёбра для поддержания динамики состояний
    for n in (concept_a, concept_b, abstract):
        g.add_edge(n, n)
    return g


class PFGAgent:
    """Автономный агент на пластичном функциональном графе."""

    def __init__(self, config: AgentConfig | None = None, storage: Storage | None = None,
                 graph: PlasticFunctionGraph | None = None) -> None:
        self.config = config or AgentConfig.load()
        self._owns_storage = storage is None
        self.storage = storage or Storage(self.config.resolve_path(self.config.memory.sqlite_path))
        self.graph = graph or build_default_graph(self.config)
        self.memory = Memory(self.config, self.storage)
        self.safety = SafetyGuard(self.config)
        self.safety.storage = self.storage
        self.planner = Planner(self.config, self.graph)
        self.executor = Executor(self.config, self.memory, self.safety)
        self.skills = SkillsManager(self.config, self.storage, self.memory, self.graph)
        self.executor.skill_registry = self.skills
        self.reflection = ReflectionEngine(self.config, self.graph, self.memory)
        self.log = JsonlLogger(self.config.log_dir)
        # интернет-контур (по умолчанию выключен; включается в конфиге осознанно)
        self.ingest: Any = None
        self.web_search: Any = None
        try:
            from pfg.fetcher import Fetcher
            from pfg.ingest import IngestPipeline
            from pfg.websearch import WebSearchEngine

            fetcher = Fetcher(self.config, self.storage)
            self.ingest = IngestPipeline(self.config, self.memory, self.storage, fetcher=fetcher,
                                         safety_log=lambda rec: self.log.write("safety.jsonl", {
                                             "level": "WARNING", "source": "ingest",
                                             "message": rec.get("event", ""), "data": rec}))
            if self.config.network.enabled and self.websearch_possible():
                self.web_search = WebSearchEngine(self.config, self.ingest, fetcher)
                self.executor.web_search = self.web_search
        except Exception as exc:  # noqa: BLE001 — отсутствие контура не роняет агента
            logger.warning("интернет-контур не инициализирован: %s", exc)
        logger.info("агент '%s' инициализирован: %s", self.config.name, self.graph.stats())

    def websearch_possible(self) -> bool:
        return bool(self.config.network.allowed_domains) or not self.config.network.deny_by_default

    # ---------- автономный интернет-поиск ----------
    _QUESTION_HINTS = re.compile(
        r"(когда|где|сколько|почему|что такое|кто такой|кто такая|какой|какая|какие|"
        r"чем отличается|расскажи|назови|who |what |when |where |how much|why)", re.IGNORECASE)

    def _should_auto_search(self, state: dict[str, Any], parsed: dict[str, Any],
                            results: list[ActionResult]) -> bool:
        """Эвристика «мне не хватает знаний»: вопрос + пустой/слабый результат поиска по памяти.

        Срабатывает только если включён сетевой контур и есть веб-движок.
        """
        if self.web_search is None or not self.config.network.enabled:
            return False
        intent = parsed.get("intent", "chat")
        if intent in ("remember", "write_file", "calculate", "gridworld"):
            return False  # это не запрос знания
        # уже явно искали в сети — не дублируем
        if any(r.action == "web_search" for r in results):
            return False
        goal = parsed.get("raw") or ""
        is_question = "?" in goal or bool(self._QUESTION_HINTS.search(goal))
        if not is_question:
            return False
        # память пуста или найденное слабо релевантно вопросу
        hits = self.memory.retrieve(goal, top_k=self.config.memory.retrieval_top_k)
        if hits:
            best_score = max(s for _, s in hits)
            best_text = max(hits, key=lambda h: h[1])[0].text
            # пересечение слов вопроса и факта — дешёвый прокси релевантности
            q_words = set(re.findall(r"[а-яa-z]{4,}", goal.lower()))
            f_words = set(re.findall(r"[а-яa-z]{4,}", best_text.lower()))
            overlap = len(q_words & f_words) / max(1, len(q_words))
            if best_score > 0.55 and overlap >= 0.3:
                return False  # в памяти есть хороший ответ
        return True

    def _autonomous_web_search(self, query: str) -> dict[str, Any] | None:
        """Агент сам решает сходить в сеть: safety-гейт → поиск → ActionResult для compose_answer."""
        action = Action(name="web_search", args={"query": query}, purpose="автономный поиск знания")
        decision = self.safety.check(action)
        self.log.write("actions.jsonl", {
            "level": "INFO", "source": "agent.autosearch", "message": "intrinsic web search",
            "data": {"query": query[:200], "allowed": decision.allowed, "risk": decision.risk},
        })
        if not decision.allowed:
            return None
        try:
            rep = self.web_search.search(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("автономный поиск упал: %s", exc)
            return None
        out = {
            "success": rep.success, "answer": rep.answer,
            "sources": rep.sources, "confidence": rep.confidence,
            "corroborated": rep.corroborated, "auto": True,
        }
        result = ActionResult(action="web_search", success=rep.success,
                              output=out, error=None if rep.success else "источники не ответили")
        return {"decision": decision, "result": result, "report": rep}

    # ---------- основной цикл ----------
    def run_task(self, goal: str) -> AgentResult:
        state = self.encode_goal(goal)
        parsed = state["parsed"]
        plan = self.planner.make_plan(parsed)
        safe_actions, decisions = self.safety.check_plan([s.action for s in plan.steps])
        results = self.executor.execute(safe_actions)
        # автономный fallback: если в памяти ничего не найдено и это вопрос — агент сам идёт в интернет
        web_report = None
        if self._should_auto_search(state, parsed, results):
            query = (parsed.get("args", {}).get("query") or goal).strip()
            web_report = self._autonomous_web_search(query)
            if web_report is not None:
                decisions.append(web_report["decision"])
                results.append(web_report["result"])
        observation = self.observe(results)
        answer, success, confidence = self.compose_answer(parsed, results, observation,
                                                          web_report=web_report)

        ep = Episode(id=f"ep_{new_id()}", goal=goal,
                     context=[f.id for f, _ in self.memory.retrieve(goal, top_k=5)],
                     actions=[r.to_dict() for r in results],
                     result=observation,
                     reward=1.0 if success else -0.5,
                     error="; ".join(r.error for r in results if r.error) or None)
        self.memory.store_episode(ep)
        lessons = self.reflection.reflect(goal=goal, plan=plan, results=results, observation=observation)
        if lessons and ep.lesson is None:
            try:
                self.storage.execute("UPDATE episodes SET lesson=? WHERE id=?",
                                     (lessons[0].text[:300], ep.id))
            except Exception:  # noqa: BLE001
                pass
        self.graph.decay_salience()

        res = AgentResult(
            goal=goal, success=success, answer=answer,
            actions=[r.to_dict() for r in results],
            confidence=confidence,
            memory_used=ep.context[:5],
            errors=[r.error for r in results if r.error],
            metadata={"plan": plan.to_dict(), "decisions": [d.to_dict() for d in decisions],
                      "episode_id": ep.id, "lessons": len(lessons)},
        )
        self.log.write("events.jsonl", {
            "level": "INFO" if success else "WARNING",
            "source": "agent", "message": "task finished",
            "data": {"goal": goal[:120], "success": success, "confidence": round(confidence, 3)},
        })
        return res

    # ---------- компоненты цикла ----------
    def encode_goal(self, goal: str) -> dict[str, Any]:
        """Внешние входы I_i и M_i: эмбеддинг цели в сенсорный узел + вклад памяти."""
        parsed = parse_goal(goal)
        self.memory.set_goal(goal)
        self.safety.set_goal_context(goal)
        vec = self.memory.vector_for("memory_node", goal)
        inputs = {"sensor_input": vec}
        hits = self.memory.retrieve(goal, top_k=3)
        if hits:
            mem_vec = np.mean([f.key_vector for f, _ in hits], axis=0).astype(np.float32)
            inputs["memory_node"] = mem_vec
        self.graph.step(inputs)
        return {"parsed": parsed, "inputs": inputs, "context_facts": [f.text for f, _ in hits]}

    def observe(self, results: list[ActionResult]) -> str:
        parts = []
        for r in results:
            if r.success:
                parts.append(f"{r.action}: {_short(r.output)}")
            else:
                parts.append(f"{r.action} FAILED: {r.error}")
        obs = " | ".join(parts) if parts else "нет выполненных действий"
        # наблюдение влияет на граф: успешность входит как внешний сигнал
        signal = 1.0 if all(r.success for r in results) and results else -1.0
        self.graph.step({"sensor_input": np.full(self.config.graph.node_dim, signal * 0.2, dtype=np.float32)})
        return obs[:2000]

    def compose_answer(self, parsed: dict[str, Any], results: list[ActionResult],
                       observation: str, *, web_report: dict[str, Any] | None = None,
                       ) -> tuple[str, bool, float]:
        intent = parsed.get("intent", "chat")
        ok_results = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        success = bool(results) and not failed

        # авто-ответ из интернета имеет приоритет над слабым ответом из памяти
        if web_report is not None:
            out = web_report["result"].output or {}
            if out.get("success") and out.get("answer"):
                src_n = len(out.get("sources", []))
                tag = "сам нашёл в интернете" + (", подтверждено несколькими источниками"
                                                 if out.get("corroborated") else "")
                return f"{out['answer']} ({tag}, источников: {src_n})", True, \
                    round(float(out.get("confidence", 0.6)) + 0.05, 2)

        if intent == "remember" and ok_results:
            out = ok_results[0].output or {}
            return f"Запомнил: {out.get('text', '')}", True, 0.9
        if intent == "recall":
            for r in ok_results:
                if isinstance(r.output, list) and r.output:
                    best = r.output[0]["text"]
                    return f"По моим данным: {best}", True, 0.85
            # память пуста — если включён интернет-контур, пробуем найти сами
            if self.web_search is not None and self.config.network.enabled:
                query = parsed.get("args", {}).get("query") or parsed.get("raw", "")
                rep = self._autonomous_web_search(query.strip())
                if rep is not None:
                    out = rep["result"].output or {}
                    if out.get("success") and out.get("answer"):
                        src_n = len(out.get("sources", []))
                        tag = "сам нашёл в интернете" + (", подтверждено несколькими источниками"
                                                         if out.get("corroborated") else "")
                        return f"{out['answer']} ({tag}, источников: {src_n})", True, \
                            round(float(out.get("confidence", 0.6)) + 0.05, 2)
            return "Я не нашёл это в памяти.", False, 0.3
        if intent == "web_search":
            for r in ok_results:
                out = r.output if isinstance(r.output, dict) else {}
                if out.get("success") and out.get("answer"):
                    src_n = len(out.get("sources", []))
                    return f"{out['answer']} (источников: {src_n})", True, float(out.get("confidence", 0.6))
            err = results[0].error if results and results[0].error else "источники не ответили"
            return f"Веб-поиск не удался: {err}", False, 0.3
        if intent == "calculate":
            for r in ok_results:
                if isinstance(r.output, dict) and "result" in r.output:
                    val = r.output["result"]
                    val = int(val) if float(val).is_integer() else val
                    return f"Результат: {val}", True, 0.95
        if intent in ("write_file", "read_file", "list_files"):
            return f"Готово. {observation}", success, 0.8 if success else 0.4
        if ok_results:
            return observation, success, 0.7
        if failed:
            return f"Не удалось: {failed[0].error}", False, 0.5
        return "Нет выполнимых действий.", False, 0.2

    # ---------- обслуживание ----------
    def evolve(self) -> dict[str, Any]:
        """Эволюционный проход: рост/обрезка графа, символическая выжимка, компиляция навыка."""
        info: dict[str, Any] = {}
        info["pruned"] = self.graph.prune_and_grow()
        info["edges_added"] = self.reflection.apply_growth()
        candidates = self.skills.find_repetitive_subgraphs(min_usage=10)
        compiled = []
        for cand in candidates[:2]:
            proposal = self.skills.compile_edge(cand["edge_id"])
            compiled.append({"skill": proposal.skill_name, "accepted": proposal.accepted,
                             "reason": proposal.reason})
        info["skills_compiled"] = compiled
        self.log.write("learning.jsonl", {
            "level": "INFO", "source": "agent", "message": "evolve pass", "data": info,
        })
        return info

    def save(self) -> None:
        self.graph.save(self.storage)

    def load(self) -> None:
        self.graph.load(self.storage)

    def status(self) -> dict[str, Any]:
        facts = len(self.memory.all_facts())
        eps = self.storage.query("SELECT COUNT(*) FROM episodes")[0][0]
        skills = self.skills.list_skills()
        return {
            "name": self.config.name,
            "graph": self.graph.stats(),
            "facts": facts,
            "episodes": int(eps),
            "skills_total": len(skills),
            "skills_active": sum(1 for s in skills if s.active),
            "safety_violations": self.safety.violations,
            "recent_lessons": len(self.reflection.last_lessons),
        }

    def close(self) -> None:
        if self._owns_storage:
            try:
                self.graph.save(self.storage)
            finally:
                self.storage.close()


def _short(obj: Any, limit: int = 160) -> str:
    s = str(obj)
    return s if len(s) <= limit else s[:limit] + "…"
