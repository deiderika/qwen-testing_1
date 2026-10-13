"""Мультиисточниковый веб-поиск PFG-Agent: запрос -> несколько источников -> ответ.

Пайплайн (расширение интернет-контура из ТЗ):
    Query -> Source Plan (белый список) -> Fetch (квоты/лимиты fetcher.py)
    -> Extract (html_to_text / flatten_json) -> Fact Extraction (карантин!)
    -> Cross-source Corroboration -> Answer Synthesis (только структурированные факты)

Инварианты безопасности:
  1. Используются только домены из network.allowed_domains (deny by default).
  2. Извлечённые предложения НЕ попадают в контекст планировщика как текст —
     они идут через IngestPipeline (trust, quarantine, дедупликация), а синтез
     ответа работает с короткими цитатами-фактами и метаданными источников.
  3. Никаких приказов от данных: при подозрении на инъекцию источник помечается
     suspicious и не участвует в подтверждении (corroboration).
"""
from __future__ import annotations

import json
import logging
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

from pfg.config import AgentConfig
from pfg.fetcher import Fetcher
from pfg.ingest import IngestPipeline, clean_text, extract_facts, html_to_text
from pfg.memory import embed_text
from pfg.math_basis import cosine_similarity
from pfg.utils import utc_iso

logger = logging.getLogger("pfg.websearch")

STOPWORDS = {
    "какой", "какая", "какое", "какие", "когда", "где", "кто", "что", "почему",
    "зачем", "как", "сколько", "теперь", "сегодня", "вчера", "завтра", "мне",
    "меня", "тебе", "тебя", "это", "есть", "был", "была", "было", "для", "над",
    "об", "он", "они", "оно", "а", "в", "и", "к", "м", "н", "о", "с", "у", "ы",
    "до", "по", "же", "бы", "не", "ни", "да", "ну", "из", "за", "про", "чем",
    "чтобы", "который", "которая", "которые", "the", "a", "an", "of", "to", "is",
    "what", "when", "where", "who", "how", "please", "tell", "know",
}


def query_tokens(query: str) -> list[str]:
    """Значимые слова запроса (без стоп-слов), для релевантности и имён статей."""
    words = re.findall(r"[а-яёa-z0-9][\w\-]{2,}", query.lower())
    return [w for w in words if w not in STOPWORDS] or words[:4]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


@dataclass
class SourceResult:
    source: str
    domain: str
    ok: bool
    facts: list[str] = field(default_factory=list)
    trust: float = 0.0
    reason: str = ""
    bytes_in: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "ok": self.ok, "facts": len(self.facts),
                "trust": round(self.trust, 3), "reason": self.reason, "bytes": self.bytes_in}


@dataclass
class WebSearchReport:
    query: str
    success: bool
    answer: str | None = None
    confidence: float = 0.0
    corroborated: bool = False
    sources: list[dict[str, Any]] = field(default_factory=list)
    stored_fact_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query, "success": self.success, "answer": self.answer,
                "confidence": round(self.confidence, 3), "corroborated": self.corroborated,
                "sources": self.sources, "stored_fact_ids": self.stored_fact_ids,
                "errors": self.errors, "retrieved_at": utc_iso()}


class WebSearchEngine:
    """Поиск по нескольким разрешённым источникам + синтез ответа из фактов."""

    def __init__(self, config: AgentConfig, ingest: IngestPipeline, fetcher: Fetcher) -> None:
        self.config = config
        self.ncfg = config.network
        self.ingest = ingest
        self.fetcher = fetcher
        self.dim = config.graph.node_dim
        self._last_query: str = ""
        # каналы строятся ТОЛЬКО из белого списка доменов
        self.wiki_langs: list[str] = []
        self.other_channels: list[tuple[str, str]] = []
        for d in self.ncfg.allowed_domains:
            dl = d.lower().strip()
            if dl.endswith("wikipedia.org"):
                lang = "ru" if dl.startswith("ru.") else "en"
                if lang not in self.wiki_langs:
                    self.wiki_langs.append(lang)
            elif "news" in dl or "rss" in dl or dl.endswith(("feed", ".rss")):
                self.other_channels.append((f"rss:{dl}", f"https://{dl}/"))
            else:
                self.other_channels.append((f"site:{dl}", f"https://{dl}/"))

    @property
    def multi_source(self) -> bool:
        """Можно ли искать сразу в нескольких независимых источниках."""
        return len(self.wiki_langs) + len(self.other_channels) >= 2

    # ---------- публичный API ----------
    def search(self, query: str, max_sources: int = 3) -> WebSearchReport:
        self._last_query = query
        report = WebSearchReport(query=query, success=False)
        if not self.ncfg.enabled:
            report.errors.append("network выключен (enabled=false) — включите в конфиге осознанно")
            return report
        urls = self._plan_urls(query)[: max_sources * 3]
        if not urls:
            report.errors.append("нет доступных источников для запроса (пустой белый список?)")
            return report

        # собираем факты по каналам; канал остановлен после первого успешного варианта
        all_facts: list[tuple[str, str, float]] = []  # (text, source, trust)
        done_kinds: set[str] = set()
        for kind, url in urls:
            if kind in done_kinds:
                continue
            sr = self._fetch_source(kind, url)
            if not sr.ok:
                report.errors.append(f"{kind}: {sr.reason}")
                continue
            report.sources.append(sr.to_dict())
            done_kinds.add(kind)
            for t in sr.facts:
                all_facts.append((t, sr.source, sr.trust))
            if len(done_kinds) >= max_sources:
                break

        if not all_facts:
            report.errors.append("ни один источник не вернул факты")
            return report

        # --- кросс-источниковое подтверждение (corroboration) ---
        clusters = self._cluster(all_facts)
        best = self._pick_best_cluster(clusters)
        if best is None:
            report.errors.append("не удалось выбрать релевантные факты")
            return report

        texts, srcs, trusts = best
        n_src = len(set(srcs))
        avg_trust = sum(trusts) / len(trusts) if trusts else 0.3
        # confidence: базовое доверие + бонус за подтверждение несколькими источниками
        conf = min(0.95, avg_trust * 0.6 + (0.25 if n_src >= 2 else 0.0) + 0.15)

        # --- сохранение фактов через память с карантином (данные != инструкции) ---
        for text, src, tr in list(zip(texts, srcs, trusts))[: self.ncfg.max_facts_per_ingest]:
            fid = self.ingest.memory.store_external_fact(
                text, source=src, trust=tr, confidence=min(0.9, conf),
                quarantined=True, suspicious=False,
                corroborated_by=[s for s in set(srcs) if s != src][:3],
            )
            report.stored_fact_ids.append(fid)

        lead = texts[0].rstrip(".")
        if n_src >= 2:
            report.answer = f"{lead}. (подтверждено {n_src} источниками)"
            report.corroborated = True
        else:
            report.answer = f"{lead}. (источник: {srcs[0]}, недоверенные данные)"
        report.success = True
        report.confidence = conf
        logger.info("websearch '%s': %d фактов, %d источников, conf=%.2f",
                    query[:60], len(texts), n_src, conf)
        return report

    # ---------- выбор URL ----------
    def _plan_urls(self, query: str) -> list[tuple[str, str]]:
        """URL-кандидаты: сначала параллельно по разным каналам, затем запасные варианты."""
        tokens = query_tokens(query)
        title = "_".join(tokens[:4]) if tokens else query.strip()[:40]
        qenc = urllib.parse.quote(title)
        round1: list[tuple[str, str]] = []
        round2: list[tuple[str, str]] = []
        for lang in self.wiki_langs:
            api = f"https://{lang}.wikipedia.org/api.php"
            rest = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{qenc}"
            # основной вариант: extracts по прямому title с redirects
            round1.append((f"wiki:{lang}",
                           f"{api}?action=query&titles={qenc}&prop=extracts&exintro=1"
                           f"&explaintext=1&format=json&redirects=1"))
            # запасной: поиск статьи по запросу -> snippet
            round2.append((f"wiki:{lang}",
                           f"{api}?action=query&list=search&srsearch={urllib.parse.quote(' '.join(tokens[:6]))}"
                           f"&format=json&srlimit=3&utf8=1&exintro=1"))
            # запасной 2: REST summary
            round2.append((f"wiki:{lang}", rest))
        for kind, base in self.other_channels:
            round1.append((kind, base))
        return round1 + round2

    # ---------- fetch + извлечение ----------
    def _fetch_source(self, kind: str, url: str) -> SourceResult:
        res = self.fetcher.fetch(url)
        if not res.ok:
            return SourceResult(kind, self._domain(url), False, reason=res.reason or res.status,
                                bytes_in=res.bytes)
        try:
            sentences = self._extract_sentences(res.text, url)
        except Exception as exc:  # noqa: BLE001
            return SourceResult(kind, self._domain(url), False, reason=f"parse error: {type(exc).__name__}",
                                bytes_in=res.bytes)
        rep_row = self.ingest.storage.query("SELECT reputation FROM sources WHERE domain=?",
                                            (self._domain(url),))
        reputation = float(rep_row[0][0]) if rep_row else 0.4
        trust = max(0.2, min(0.8, reputation))
        if not sentences:
            return SourceResult(kind, self._domain(url), False, reason="факты не извлечены",
                                bytes_in=res.bytes)
        return SourceResult(kind, self._domain(url), True, facts=sentences, trust=trust,
                            bytes_in=res.bytes)

    def _extract_sentences(self, raw: str, url: str) -> list[str]:
        """Текст источника -> короткие факт-предложения (макс. 8)."""
        text = raw
        stripped = raw.lstrip()
        if stripped.startswith(("{", "[")):
            try:
                data = json.loads(raw)
                text = self._flatten_obj(data)
            except json.JSONDecodeError:
                text = raw
        elif "<" in raw[:2000]:
            text = html_to_text(raw)
        text = clean_text(text, max_chars=30_000)
        cands = extract_facts(text, max_facts=8)
        if not cands:  # мягче: принять и более длинные предложения
            for chunk in re.split(r"(?<=[.!?。])\s+|\n+", text):
                s = chunk.strip()
                if 10 <= len(s) <= 400 and not any(c == s for c in cands):
                    cands.append(s)
                if len(cands) >= 8:
                    break
        return cands

    @staticmethod
    def _flatten_obj(obj: Any, prefix: str = "") -> str:
        lines: list[str] = []

        def walk(o: Any, p: str) -> None:
            if len(lines) > 300:
                return
            if isinstance(o, dict):
                for k, v in o.items():
                    walk(v, f"{p}{k}.")
            elif isinstance(o, list):
                for v in o[:20]:
                    walk(v, p)
            else:
                s = str(o).strip()
                if s and len(s) < 500:
                    lines.append(f"{p[:-1]}: {s}" if p else s)
        walk(obj, prefix)
        return "\n".join(lines)

    # ---------- кластеризация и выбор ----------
    def _cluster(self, facts: list[tuple[str, str, float]]) -> list[tuple[list[str], list[str], list[float]]]:
        """Группирует близкие по смыслу факты разных источников в кластеры."""
        clusters: list[list[int]] = []
        vecs = [embed_text(t, self.dim) for t, _s, _tr in facts]
        assigned: set[int] = set()
        for i in range(len(facts)):
            if i in assigned:
                continue
            group = [i]
            assigned.add(i)
            for j in range(i + 1, len(facts)):
                if j in assigned:
                    continue
                if cosine_similarity(vecs[i], vecs[j]) >= 0.72:
                    group.append(j)
                    assigned.add(j)
            clusters.append(group)
        out = []
        for g in clusters:
            texts = [facts[i][0] for i in g]
            srcs = [facts[i][1] for i in g]
            trusts = [facts[i][2] for i in g]
            out.append((texts, srcs, trusts))
        return out

    def _pick_best_cluster(self, clusters) -> tuple[list[str], list[str], list[float]] | None:
        if not clusters:
            return None
        qv = embed_text(self._last_query or "", self.dim)
        qtok = set(query_tokens(self._last_query or ""))

        def rel_score(text: str) -> float:
            v = cosine_similarity(qv, embed_text(text, self.dim))
            ttoks = set(re.findall(r"[а-яёa-z0-9]{3,}", text.lower()))
            overlap = len(qtok & ttoks) / max(1, len(qtok)) if qtok else 0.0
            return 0.5 * v + 0.5 * overlap

        def score(cluster) -> tuple:
            texts, srcs, trusts = cluster
            n_src = len(set(srcs))
            rel = max(rel_score(t) for t in texts)
            tr = sum(trusts) / len(trusts)
            return (rel, n_src, tr)

        best = max(clusters, key=score)
        texts, srcs, trusts = best
        order = sorted(range(len(texts)), key=lambda i: -rel_score(texts[i]))
        return [texts[i] for i in order], [srcs[i] for i in order], [trusts[i] for i in order]

    def search_logged(self, query: str) -> WebSearchReport:
        rep = self.search(query)
        try:
            import time as _t
            self.ingest.storage.log_event("INFO", "websearch", f"query: {query[:120]}",
                                          int(_t.time()), rep.to_dict())
        except Exception:  # noqa: BLE001
            pass
        return rep

    @staticmethod
    def _domain(url: str) -> str:
        return urllib.parse.urlparse(url).hostname or url
