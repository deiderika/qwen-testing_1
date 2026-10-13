"""Пайплайн Data Ingest: внешний контент -> карантин -> факты -> память.

Конвейер (жёсткие фильтры, раздел 4 ТЗ на интернет-контур):
    Trigger -> Policy Check -> Fetch -> Extract -> Clean -> Quarantine
    -> Fact Extraction -> Trust & Corroboration -> Deduplicate -> Embed+Store
    -> Log + Update Salience

Ключевые инварианты безопасности:
  1. Данные НИКОГДА не становятся инструкциями: в планировщик/контекст агента
     попадают только извлечённые факты с метаданными (source/trust/confidence),
     а не исходный текст и не HTML.
  2. Если текст помечен как подозрительный (приказы, инъекции) — включается
     строгий режим извлечения (только короткие строки "сущность: значение"),
     факт сохраняется с quarantined=1 и пониженным доверием.
  3. Действия уровня medium+ не могут быть инициированы недоверенными данными
     (см. safety.py: источник-факт не может создать действие).
"""
from __future__ import annotations

import json
import logging
import re
import urllib.parse
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

from pfg.config import AgentConfig
from pfg.memory import Memory
from pfg.storage import Storage
from pfg.trust import (
    TrustInputs, compute_trust, corroboration_ratio, detect_conflict,
    freshness, quarantine_block, suspicion_score, trust_to_confidence,
)
from pfg.utils import now_ts, utc_iso

logger = logging.getLogger("pfg.ingest")


# ---------- 4. Extract: HTML -> текст ----------
class _TextExtractor(HTMLParser):
    """Лёгкий извлекатель текста из HTML без внешних зависимостей (RPi4-friendly)."""

    _SKIP = {"script", "style", "noscript", "template", "svg", "iframe", "nav", "footer", "header"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in ("p", "br", "div", "li", "h1", "h2", "h3", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self.parts.append(data.strip())

    @property
    def text(self) -> str:
        return "\n".join(self.parts)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:  # noqa: BLE001 — битый HTML не должен ронять ingest
        return re.sub(r"<[^>]+>", " ", html)
    return parser.text


# ---------- 5. Clean ----------
_WS_RE = re.compile(r"[ \t\u00a0]+")
_NL_RE = re.compile(r"\n\s*\n+")


def clean_text(text: str, max_chars: int = 60_000) -> str:
    """Нормализует пробелы, склеивает пустые строки, обрезает по лимиту."""
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n", text)
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if len(ln) >= 8)
    return text[:max_chars]


# ---------- 7. Fact Extraction ----------
_SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")
_LABELLED_RE = re.compile(r"^([^:\n]{3,60}):\s*(.{3,200})$")


def extract_facts(text: str, max_facts: int = 15, strict: bool = False) -> list[str]:
    """Извлекает 3..max фактов из чистого текста.

    Обычный режим: предложения разумной длины.
    Строгий режим (strict=True, для подозрительных источников): принимаются
    только «метки значений» вида "Сущность: значение" и строки без глагольных приказов —
    фактически цифры/названия/даты. Это ограничивает площадь инъекций.
    """
    out: list[str] = []
    seen: set[str] = set()
    candidates: list[str] = []
    for chunk in _SENT_SPLIT.split(text):
        s = chunk.strip().strip("\"'“”«»")
        if not (20 <= len(s) <= 300):
            continue
        candidates.append(s)
    for s in candidates:
        if strict:
            m = _LABELLED_RE.match(s)
            if not m:
                continue
            s = f"{m.group(1).strip()}: {m.group(2).strip()}"
        norm = s.lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(s)
        if len(out) >= max_facts:
            break
    return out


@dataclass
class IngestReport:
    source: str
    ok: bool
    facts_stored: int = 0
    facts_deduped: int = 0
    suspicious: bool = False
    trust: float = 0.0
    quarantined: bool = False
    bytes_in: int = 0
    reason: str = ""
    fact_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "ok": self.ok, "facts_stored": self.facts_stored,
                "facts_deduped": self.facts_deduped, "suspicious": self.suspicious,
                "trust": round(self.trust, 3), "quarantined": self.quarantined,
                "bytes_in": self.bytes_in, "reason": self.reason, "fact_ids": self.fact_ids}


class IngestPipeline:
    """Конвейер ingestion: принимает текст/URL, выдаёт факты в памяти с доверием."""

    def __init__(self, config: AgentConfig, memory: Memory, storage: Storage,
                 fetcher: Any = None, safety_log: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.config = config
        self.memory = memory
        self.storage = storage
        self.fetcher = fetcher
        self.safety_log = safety_log or (lambda rec: None)
        self.tcfg = config.trust
        self.ncfg = config.network

    # ---------- публичные точки входа ----------
    def ingest_text(self, text: str, source: str = "local", domain: str = "") -> IngestReport:
        """Ингест локального текста (inbox, knowledge_bundle, выгрузки)."""
        return self._run(text, source=source, domain=domain or source, nbytes=len(text.encode()))

    def ingest_file(self, path: str | Path) -> IngestReport:
        p = Path(path)
        raw = p.read_text(encoding="utf-8", errors="replace")
        if p.suffix in (".html", ".htm"):
            raw = html_to_text(raw)
        if p.suffix == ".json":
            raw = self._flatten_json(raw)
        return self.ingest_text(raw, source=f"file:{p.name}", domain="local-file")

    def ingest_url(self, url: str) -> IngestReport:
        """Fetch + ingest. Отклоняется, если fetch-политика запрещает (по умолчанию — всегда)."""
        if self.fetcher is None:
            return IngestReport(url, False, reason="fetcher не подключён")
        res = self.fetcher.fetch(url)
        if not res.ok:
            return IngestReport(url, False, reason=res.reason or res.status, bytes_in=res.bytes)
        domain = urllib.parse.urlparse(url).hostname or url
        text = res.text
        if "json" in (res.etag or "").lower() or text.lstrip().startswith(("{", "[")):
            text = self._flatten_json(text)
        elif "<" in text[:2000]:
            text = html_to_text(text)
        report = self._run(text, source=url, domain=domain, nbytes=res.bytes)
        # обновляем fetch_log числом извлечённых фактов
        try:
            self.storage.log_fetch(url, res.status, res.bytes, report.facts_stored + report.facts_deduped,
                                   report.trust)
        except Exception:  # noqa: BLE001
            pass
        return report

    def ingest_bundle(self, path: str | Path) -> list[IngestReport]:
        """Перенос знаний с ПК: JSON-бандл [{"text":..., "source":...}, ...]."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        reports = []
        items = data if isinstance(data, list) else data.get("facts", [])
        for item in items[: self.ncfg.max_facts_per_ingest * 4]:
            if isinstance(item, str):
                reports.append(self.ingest_text(item, source=f"bundle:{Path(path).name}"))
            elif isinstance(item, dict) and "text" in item:
                reports.append(self.ingest_text(str(item["text"]),
                                                source=str(item.get("source", "bundle"))))
        return reports

    # ---------- ядро конвейера ----------
    def _run(self, raw_text: str, source: str, domain: str, nbytes: int) -> IngestReport:
        text = clean_text(raw_text)
        if len(text) < 20:
            return IngestReport(source, False, reason="после очистки осталось слишком мало текста",
                                bytes_in=nbytes)
        # 6. Quarantine-детектор приказов
        susp, hits = suspicion_score(text)
        suspicious = susp >= 0.3
        if suspicious:
            self.safety_log({
                "event": "instruction_like_text_detected",
                "source": source, "patterns": hits, "suspicion": round(susp, 2),
                "action_taken": "strict extraction mode; trust penalty applied",
            })
            logger.warning("обнаружены приказоподобные паттерны в %s: %s — строгий режим", source, hits)
        # 7. Fact extraction (strict для подозрительного контента)
        facts = extract_facts(text, max_facts=self.ncfg.max_facts_per_ingest, strict=suspicious)
        if not facts:
            return IngestReport(source, False, reason="не удалось извлечь факты",
                                suspicious=suspicious, bytes_in=nbytes)
        # компоненты доверия
        rep_row = self.storage.query("SELECT reputation FROM sources WHERE domain=?", (domain,))
        reputation = float(rep_row[0][0]) if rep_row else (0.9 if domain in ("user", "local-file", "local") else 0.4)
        existing_texts = [f.text for f in self.memory.all_facts()][-400:]
        sim_fn: Callable[[str, str], float] | None = None
        try:
            from pfg.memory import embed_text
            from pfg.math_basis import cosine_similarity

            def sim_fn(a: str, b: str) -> float:  # noqa: D418
                dim = self.config.graph.node_dim
                return cosine_similarity(embed_text(a, dim), embed_text(b, dim))
        except Exception:  # noqa: BLE001
            sim_fn = None
        stored, deduped, ids = 0, 0, []
        agg_conflicts = 0.0
        for fact_text in facts:
            conflict = detect_conflict(fact_text, existing_texts, sim_fn)
            corr_sources = self._corroborating_sources(fact_text, existing_texts, sim_fn)
            inputs = TrustInputs(
                reputation=reputation,
                corroboration=corroboration_ratio(len(corr_sources), 1 if conflict > 0 else 0),
                freshness=freshness(0.0, self.tcfg.freshness_lambda),
                conflict=conflict,
                suspicion=susp,
            )
            trust = compute_trust(inputs, self.tcfg)
            agg_conflicts = max(agg_conflicts, conflict)
            confidence = trust_to_confidence(trust, self.tcfg.min_trust_for_low_risk,
                                             t_full=max(self.tcfg.min_trust_for_fact_confidence + 0.3, 0.5))
            fid = self.memory.store_external_fact(
                fact_text, source=source, trust=trust, confidence=confidence,
                quarantined=True, suspicious=suspicious, corroborated_by=corr_sources,
            )
            existed = fid.startswith("dup:")
            deduped += int(existed)
            stored += int(not existed)
            ids.append(fid)
            if conflict > 0.5:
                self.storage.upsert_source(domain, conflicts_delta=1)
        self.storage.upsert_source(domain, total_facts_delta=stored)
        avg_trust = compute_trust(TrustInputs(reputation, 0.0, 1.0, agg_conflicts, susp), self.tcfg)
        # карантиновый блок сохраняем только как лог-артефакт (НЕ в контекст планировщика)
        qblock = quarantine_block(text[:2000], source, utc_iso(), avg_trust)
        logger.info("ingest %s: %d фактов (%d дублей), trust=%.2f, suspicious=%s",
                    source, stored, deduped, avg_trust, suspicious)
        _ = qblock  # карантин нужен для аудита; в память идут только структурированные факты
        return IngestReport(source, True, facts_stored=stored, facts_deduped=deduped,
                            suspicious=suspicious, trust=avg_trust, quarantined=True,
                            bytes_in=nbytes, fact_ids=ids)

    def _corroborating_sources(self, fact_text: str, existing_texts: list[str],
                               sim_fn: Callable[[str, str], float] | None) -> list[str]:
        """Источники уже сохранённых похожих фактов — подтверждение (corroboration)."""
        if sim_fn is None:
            return []
        out: list[str] = []
        for f in self.memory.all_facts()[-300:]:
            if f.text != fact_text and sim_fn(f.text, fact_text) >= 0.9 and f.source not in out:
                out.append(f.source)
        return out[:5]

    @staticmethod
    def _flatten_json(raw: str) -> str:
        """JSON API -> плоский текст 'ключ: значение' для извлечения фактов."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        lines: list[str] = []

        def walk(obj: Any, prefix: str = "") -> None:
            if len(lines) > 400:
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    walk(v, f"{prefix}{k}.")
            elif isinstance(obj, list):
                for i, v in enumerate(obj[:50]):
                    walk(v, f"{prefix}[{i}].")
            else:
                lines.append(f"{prefix.rstrip('.')}: {obj}")
        walk(data)
        return "\n".join(lines)
