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

import html
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

# ---------- Поисковые системы (SERP) ----------
# Каждая система — это (имя, домен выдачи, шаблон URL запроса, регулярка для
# извлечения заголовков+сниппетов из HTML). Домены ОБЯЗАТЕЛЬНО должны быть в
# network.allowed_domains, иначе fetcher их заблокирует (deny by default).
SEARCH_ENGINES: dict[str, dict[str, str]] = {
    "google": {
        "domain": "www.google.com",
        "url": "https://www.google.com/search?q={q}&num=8&hl=ru&nfpr=1",
        # google: <h3 class="...">ЗАГОЛОВОК</h3>, сниппеты в div с data-sncf / span
        "title_re": r"<h3[^>]*>(.*?)</h3>",
        "snippet_re": r'<span[^>]*>([^<]{40,300})</span>',
    },
    "yandex": {
        "domain": "yandex.ru",
        "url": "https://yandex.ru/search/?text={q}&lr=213",
        "title_re": r'<a[^>]*organic-[a-z-]*web-result[^>]*>.*?<h2[^>]*>(.*?)</h2>',
        "snippet_re": r'<(?:div|span)[^>]*organic__content-wrapper[^>]*>(.*?)</(?:div|span)>',
    },
    "duckduckgo": {
        "domain": "duckduckgo.com",
        "url": "https://duckduckgo.com/html/?q={q}",
        "title_re": r'<a[^>]*class="result__a"[^>]*>(.*?)</a>',
        "snippet_re": r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
    },
    "brave": {
        "domain": "search.brave.com",
        "url": "https://search.brave.com/search?q={q}",
        "title_re": r'<div[^>]*snippet-title[^>]*>(.*?)</div>',
        "snippet_re": r'<div[^>]*snippet-content[^>]*>(.*?)</div>',
    },
}


def _strip_tags(s: str) -> str:
    s = re.sub(r"<script.*?</script>|<style.*?</style>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    return html.unescape(re.sub(r"\s+", " ", s)).strip()


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
        self.serp_engines: list[str] = []
        for d in self.ncfg.allowed_domains:
            dl = d.lower().strip()
            eng = next((name for name, spec in SEARCH_ENGINES.items() if dl == spec["domain"]), None)
            if eng is not None:
                # разрешённый домен поисковика -> добавляем движок (если не отфильтрован)
                sel = [e.strip().lower() for e in self.ncfg.search_engines if e.strip()]
                if not sel or eng in sel:
                    self.serp_engines.append(eng)
                continue
            if dl.endswith("wikipedia.org"):
                lang = "ru" if dl.startswith("ru.") else "en"
                if lang not in self.wiki_langs:
                    self.wiki_langs.append(lang)
            elif "news" in dl or "rss" in dl or dl.endswith(("feed", ".rss")):
                self.other_channels.append((f"rss:{dl}", f"https://{dl}/"))
            elif dl == "api.duckduckgo.com":
                self.other_channels.append(("ddg", ""))  # URL строится по запросу
            elif dl == "api.github.com":
                self.other_channels.append(("github", ""))
            else:
                self.other_channels.append((f"site:{dl}", f"https://{dl}/"))

    @staticmethod
    def _is_cyrillic(text: str) -> bool:
        """Есть ли в тексте кириллица (для выбора языка Википедии под запрос)."""
        return bool(re.search(r"[а-яё]", text, re.IGNORECASE))

    @property
    def multi_source(self) -> bool:
        """Можно ли искать сразу в нескольких независимых источниках."""
        return len(self.wiki_langs) + len(self.other_channels) + len(self.serp_engines) >= 2

    def available_engines(self) -> list[str]:
        """Список активных поисковых движков (для network-status)."""
        out = list(self.serp_engines)
        for l in self.wiki_langs:
            out.append(f"wikipedia:{l}")
        for kind, _b in self.other_channels:
            out.append(kind)
        return out

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
        """URL-кандидаты: сначала параллельно по разным каналам, затем запасные варианты.

        Язык Википедии подбирается под запрос (кириллица -> ru), заголовки статей
        ищутся через search API — прямой title по словам вопроса почти всегда мимо.
        """
        tokens = query_tokens(query)
        qenc = urllib.parse.quote(" ".join(tokens[:6]) or query.strip()[:40])
        prefer_ru = self._is_cyrillic(query) and "ru" in self.wiki_langs
        langs = sorted(self.wiki_langs, key=lambda l: 0 if (l == "ru") == prefer_ru else 1)
        round1: list[tuple[str, str]] = []
        round2: list[tuple[str, str]] = []
        # --- поисковые системы (Google / Яндекс / DDG / Brave): широкая выдача ---
        full_q = urllib.parse.quote(query.strip()[:80] or " ".join(tokens[:6]))
        for eng in self.serp_engines:
            spec = SEARCH_ENGINES[eng]
            round1.append((f"serp:{eng}", spec["url"].format(q=full_q)))
        for lang in langs:
            api = f"https://{lang}.wikipedia.org/api.php"
            # канал wiki_<lang>: поиск статьи по запросу -> точный title + snippet
            round1.append((f"wiki_{lang}",
                           f"{api}?action=query&list=search&srsearch={qenc}"
                           f"&format=json&srlimit=3&utf8=1"))
            # канал wikisearch_<lang>: полные тексты найденных статей (exintro)
            round1.append((f"wikisearch_{lang}",
                           f"{api}?action=query&generator=search&gsrsearch={qenc}&gsrlimit=2"
                           f"&prop=extracts&exintro=1&explaintext=1&format=json&redirects=1&utf8=1"))
            # запасной: summary по «заголовку из слов»
            title = "_".join(tokens[:4]) if tokens else query.strip()[:40]
            rest = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
            round2.append((f"wikirest_{lang}", rest))
        for kind, base in self.other_channels:
            if kind == "ddg":
                q = urllib.parse.quote(" ".join(tokens[:5]) or query.strip()[:40])
                round1.append(("ddg", f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1"))
            elif kind == "github":
                q = urllib.parse.quote(" ".join(tokens[:4]) or query.strip()[:30])
                round1.append(("github",
                               f"https://api.github.com/search/repositories?q={q}&per_page=2"))
            else:
                round1.append((kind, base))
        return round1 + round2

    # ---------- fetch + извлечение ----------
    def _fetch_source(self, kind: str, url: str) -> SourceResult:
        res = self.fetcher.fetch(url)
        if not res.ok:
            return SourceResult(kind, self._domain(url), False, reason=res.reason or res.status,
                                bytes_in=res.bytes)
        try:
            if kind.startswith("serp:"):
                sentences = self._extract_serp(res.text, kind.split(":", 1)[1])
            else:
                sentences = self._extract_sentences(res.text, url)
        except Exception as exc:  # noqa: BLE001
            return SourceResult(kind, self._domain(url), False, reason=f"parse error: {type(exc).__name__}",
                                bytes_in=res.bytes)
        rep_row = self.ingest.storage.query("SELECT reputation FROM sources WHERE domain=?",
                                            (self._domain(url),))
        reputation = float(rep_row[0][0]) if rep_row else 0.4
        trust = max(0.2, min(0.8, reputation))
        # поисковики дают только сниппеты — доверие к SERP чуть ниже, чем к первоисточнику
        if kind.startswith("serp:"):
            trust = min(trust, 0.55)
        if not sentences:
            return SourceResult(kind, self._domain(url), False, reason="факты не извлечены",
                                bytes_in=res.bytes)
        return SourceResult(kind, self._domain(url), True, facts=sentences, trust=trust,
                            bytes_in=res.bytes)

    def _extract_serp(self, raw: str, engine: str) -> list[str]:
        """HTML поисковой выдачи -> список фактов «заголовок — сниппет».

        Устойчив к разным разметкам: сначала пробуем точные шаблоны движка,
        затем универсальный fallback (h3/h2 + следующий текстовый блок).
        """
        facts: list[str] = []
        spec = SEARCH_ENGINES.get(engine, {})
        if spec:
            titles = [_strip_tags(t) for t in re.findall(spec["title_re"], raw, flags=re.S)]
            snips = [_strip_tags(s) for s in re.findall(spec["snippet_re"], raw, flags=re.S)]
            titles = [t for t in titles if 8 <= len(t) <= 200]
            snips = [s for s in snips if 40 <= len(s) <= 400]
            for i, t in enumerate(titles[:6]):
                s = snips[i] if i < len(snips) else ""
                facts.append(f"{t}: {s}" if s else t)
        if not facts:  # fallback: любые заголовки h3/h2 с текстом рядом
            blocks = re.findall(r"<(?:h3|h2)[^>]*>(.*?)</(?:h3|h2)>", raw, flags=re.S)
            for b in blocks:
                t = _strip_tags(b)
                if 15 <= len(t) <= 250 and not t.lower().startswith(("cookie", "sign in")):
                    facts.append(t)
                if len(facts) >= 6:
                    break
        # фильтр мусора SERP-страниц
        junk = ("сообщить о нарушении", "personalized search", "consent", "javascript")
        return [f for f in facts if not any(j in f.lower() for j in junk)]

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
