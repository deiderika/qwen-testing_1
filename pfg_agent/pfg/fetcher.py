"""Fetch-контур PFG-Agent: HTTP с белым списком доменов, квотами и лимитами.

Политика (раздел "Самая серьёзная проблема: недоверенные данные"):
  * network.enabled=false по умолчанию — любой fetch отклоняется;
  * deny_by_default + белый список allowed_domains;
  * жёсткие лимиты: байты на запрос, байты в сутки, запросы в час, таймаут;
  * только http/https, без редиректов вне белого списка;
  * robots.txt уважается, если respect_robots_txt=true;
  * каждый запрос пишется в fetch_log (для network-status и квот).

Модуль НЕ является действием планировщика: он вызывается явно (CLI ingest/fetch)
или автономным циклом через разрешённое действие `fetch_data` (risk=high, требует подтверждения).
"""
from __future__ import annotations

import hashlib
import logging
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from pfg.config import AgentConfig

logger = logging.getLogger("pfg.fetcher")


@dataclass
class FetchResult:
    url: str
    ok: bool
    status: str            # "ok" | "denied_policy" | "error" | "cached"
    text: str = ""
    etag: str | None = None
    bytes: int = 0
    latency_s: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"url": self.url, "ok": self.ok, "status": self.status,
                "bytes": self.bytes, "latency_s": round(self.latency_s, 3),
                "reason": self.reason, "facts_extracted": None}


class DomainPolicy:
    """Белый список доменов: deny by default."""

    def __init__(self, config: AgentConfig) -> None:
        self.cfg = config.network
        self.allowed = {d.lower().strip() for d in self.cfg.allowed_domains if d.strip()}

    def domain_of(self, url: str) -> str:
        try:
            parsed = urllib.parse.urlparse(url)
        except ValueError:
            return ""
        if parsed.scheme not in ("http", "https"):
            return ""
        host = (parsed.hostname or "").lower()
        # urlparse сам отделяет userinfo (user@host), так что трюк с '@' здесь безопасен
        return host

    def is_allowed(self, url: str) -> tuple[bool, str]:
        if not self.cfg.enabled:
            return False, "network выключен (enabled=false) — включите осознанно в конфиге"
        host = self.domain_of(url)
        if not host:
            return False, "некорректный URL или схема не http/https"
        if self.cfg.deny_by_default and host not in self.allowed \
                and not any(host.endswith("." + d) for d in self.allowed):
            return False, f"домен '{host}' не в белом списке"
        return True, "ok"

    def quota_left(self, storage: Any) -> tuple[bool, str]:
        """Проверяет почасовую и суточную квоты по fetch_log."""
        hour_requests, day_bytes = storage.fetch_quota_today()
        if hour_requests >= self.cfg.max_requests_per_hour:
            return False, f"исчерпана часовая квота ({hour_requests}/{self.cfg.max_requests_per_hour})"
        if day_bytes >= self.cfg.max_total_bytes_per_day:
            return False, f"исчерпана суточная квота байт ({day_bytes}/{self.cfg.max_total_bytes_per_day})"
        return True, "ok"


class UrlCache:
    """Кэш URL: хеш -> (etag, fetched_at, payload_path). Не качать один URL чаще cache_min_age_s."""

    def __init__(self, config: AgentConfig, storage: Any) -> None:
        self.storage = storage
        self.cache_dir = config.resolve_path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_age = config.network.cache_min_age_s

    @staticmethod
    def hash_url(url: str) -> str:
        return hashlib.blake2b(url.encode("utf-8"), digest_size=16).hexdigest()

    def fresh(self, url: str) -> tuple[bool, str | None]:
        """Возвращает (есть_свежая_копия, текст_payload)."""
        row = self.storage.cache_get(self.hash_url(url))
        if not row:
            return False, None
        _h, etag, fetched_at, payload_path = row
        if int(fetched_at or 0) + self.min_age > int(time.time()):
            try:
                from pathlib import Path
                p = Path(payload_path)
                if p.exists():
                    return True, p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
        return False, None

    def store(self, url: str, etag: str | None, text: str) -> str:
        from pathlib import Path
        h = self.hash_url(url)
        path = self.cache_dir / f"{h}.txt"
        path.write_text(text, encoding="utf-8")
        self.storage.cache_put(h, etag, str(path))
        return str(path)


_ROBOTS_CACHE: dict[str, tuple[float, set[str]]] = {}


def _robots_disallow_set(base: str, user_agent: str, ttl: float = 3600.0) -> set[str]:
    """Загружает (с кэшем на ttl секунд) набор Disallow-путей из robots.txt для нашего UA."""
    now = time.time()
    entry = _ROBOTS_CACHE.get(base)
    if entry and entry[0] > now:
        return entry[1]
    disallowed: set[str] = set()
    try:
        req = urllib.request.Request(f"{base}/robots.txt", headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310 — base из белого списка
            body = resp.read(20000).decode("utf-8", errors="replace")
        in_block = False
        ua_short = user_agent.split("/")[0].lower()
        for line in body.splitlines():
            low = line.strip().lower()
            if low.startswith("user-agent"):
                ua = low.split(":", 1)[-1].strip()
                in_block = ua in ("*", ua_short)
            elif in_block and low.startswith("disallow"):
                path = low.split(":", 1)[-1].strip()
                if path:
                    disallowed.add(path)
    except Exception:  # noqa: BLE001 — недоступный robots.txt не блокирует fetch
        pass
    _ROBOTS_CACHE[base] = (now + ttl, disallowed)
    return disallowed


def robots_blocks(url: str, user_agent: str) -> bool:
    """True, если robots.txt запрещает этот URL."""
    parsed = urllib.parse.urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    disallowed = _robots_disallow_set(base, user_agent)
    path = parsed.path or "/"
    return any(path.startswith(d) for d in disallowed)


class Fetcher:
    """HTTP-клиент с политикой, квотами, лимитами и кэшем."""

    def __init__(self, config: AgentConfig, storage: Any) -> None:
        self.config = config
        self.ncfg = config.network
        self.storage = storage
        self.policy = DomainPolicy(config)
        self.cache = UrlCache(config, storage)

    def fetch(self, url: str) -> FetchResult:
        # 1. политика домена
        ok, reason = self.policy.is_allowed(url)
        if not ok:
            logger.warning("fetch отклонён: %s (%s)", url, reason)
            self.storage.log_fetch(url, "denied_policy", 0, 0, 0.0)
            return FetchResult(url, False, "denied_policy", reason=reason)
        # 2. квоты
        ok, reason = self.policy.quota_left(self.storage)
        if not ok:
            self.storage.log_fetch(url, "denied_quota", 0, 0, 0.0)
            return FetchResult(url, False, "denied_policy", reason=reason)
        # 3. robots.txt
        if self.ncfg.respect_robots_txt and robots_blocks(url, self.ncfg.user_agent):
            self.storage.log_fetch(url, "denied_robots", 0, 0, 0.0)
            return FetchResult(url, False, "denied_policy", reason="robots.txt запрещает этот путь")
        # 4. кэш
        fresh_hit, cached_text = self.cache.fresh(url)
        if fresh_hit and cached_text is not None:
            return FetchResult(url, True, "cached", text=cached_text, reason="killed by cache")
        # 5. сам запрос (без редиректов наружу: простой RedirectValidator)
        t0 = time.time()
        allowed_hosts = {self.policy.domain_of(url)} | self.policy.allowed
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": self.ncfg.user_agent,
                "Accept": "text/plain,text/html,application/json;q=0.8,*/*;q=0.5",
            })
            opener = urllib.request.build_opener(_NoOfflistRedirect(allowed_hosts))
            with opener.open(req, timeout=self.ncfg.timeout_seconds) as resp:
                final_url = resp.geturl()
                if self.policy.domain_of(final_url) not in allowed_hosts \
                        and not self.policy.is_allowed(final_url)[0]:
                    raise PermissionError(f"редирект вне белого списка: {final_url}")
                raw = resp.read(self.ncfg.max_bytes_per_request + 1)
                etag = resp.headers.get("ETag")
                ctype = (resp.headers.get("Content-Type") or "").lower()
        except Exception as exc:  # noqa: BLE001 — сеть никогда не роняет агента
            lat = time.time() - t0
            self.storage.log_fetch(url, "error", 0, 0, 0.0)
            return FetchResult(url, False, "error", latency_s=lat, reason=f"{type(exc).__name__}: {exc}")
        lat = time.time() - t0
        if len(raw) > self.ncfg.max_bytes_per_request:
            self.storage.log_fetch(url, "too_large", len(raw), 0, 0.0)
            return FetchResult(url, False, "error", latency_s=lat,
                               reason=f"ответ больше лимита {self.ncfg.max_bytes_per_request} байт")
        if any(t in ctype for t in ("image/", "video/", "audio/")):
            self.storage.log_fetch(url, "binary_rejected", len(raw), 0, 0.0)
            return FetchResult(url, False, "error", latency_s=lat,
                               reason="бинарный контент (медиа) запрещён на RPi4")
        text = raw.decode("utf-8", errors="replace")
        self.cache.store(url, etag, text)
        self.storage.log_fetch(url, "ok", len(raw), 0, 0.0)  # facts_extracted обновит ingest
        logger.info("fetch ok: %s (%d байт, %.2f с)", url, len(raw), lat)
        return FetchResult(url, True, "ok", text=text, etag=etag, bytes=len(raw), latency_s=lat)


class _NoOfflistRedirect(urllib.request.HTTPRedirectHandler):
    """Не даёт редиректу увести запрос за пределы белого списка."""

    def __init__(self, allowed_hosts: set[str]) -> None:
        super().__init__()
        self.allowed_hosts = {h for h in allowed_hosts if h}

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, D102
        host = urllib.parse.urlparse(newurl).hostname or ""
        if host not in self.allowed_hosts:
            raise urllib.error.HTTPError(req.full_url, code, f"redirect off-list: {host}", headers, fp)  # type: ignore[arg-type]
        return super().redirect_request(req, fp, code, msg, headers, newurl)
