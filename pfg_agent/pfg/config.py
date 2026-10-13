"""Конфигурация PFG-Agent: загрузка YAML, валидация, значения по умолчанию."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


@dataclass
class GraphConfig:
    node_dim: int = 32
    max_nodes: int = 128
    max_edges: int = 4096
    basis_order: int = 5
    top_k_edges: int = 8
    activation: str = "tanh"
    use_layer_norm: bool = True
    step_size: float = 0.5


@dataclass
class LearningConfig:
    lr: float = 0.01
    salience_decay: float = 0.999
    edge_add_threshold: float = 0.7
    edge_remove_threshold: float = 0.05
    min_edge_age_for_removal: int = 100


@dataclass
class MemoryConfig:
    sqlite_path: str = "data/pfg.sqlite"
    max_facts: int = 100_000
    max_episodes: int = 10_000
    retrieval_top_k: int = 8
    recency_lambda: float = 1e-4
    w_importance: float = 0.2
    w_recency: float = 0.1
    w_usage: float = 0.05


@dataclass
class PlannerConfig:
    simulation_horizon: int = 5
    max_actions: int = 16
    reward_weight: float = 1.0
    risk_weight: float = 1.0
    cost_weight: float = 0.3
    tie_tolerance: float = 0.05


@dataclass
class ExecutorConfig:
    allow_network: bool = False
    allow_shell: bool = False
    allow_money: bool = False
    allowed_dirs: list[str] = field(default_factory=lambda: ["sandbox"])
    require_human_approval_for: list[str] = field(default_factory=lambda: ["money", "network", "shell"])
    skill_timeout_s: int = 5
    skill_max_mem_mb: int = 128


@dataclass
class SkillsConfig:
    dir: str = "skills"
    max_code_size_kb: int = 64
    allowed_imports: list[str] = field(default_factory=lambda: ["math", "json", "re", "datetime", "typing"])
    max_skill_error: float = 0.05


@dataclass
class DaemonConfig:
    """Параметры событийного демона и фоновых циклов (Lifecycle)."""
    poll_timeout_s: float = 2.0
    idle_before_autonomous_s: float = 30.0
    autonomous_interval_s: float = 900.0
    sleep_interval_s: float = 21600.0        # 6 часов — «ночная» консолидация
    autosave_interval_s: float = 300.0
    inbox_scan_interval_s: float = 5.0


@dataclass
class MotivationConfig:
    """Веса формулы Priority = w_c*Curiosity + w_n*Need + w_p*Profit."""
    w_curiosity: float = 0.4
    w_need: float = 0.35
    w_profit: float = 0.25
    min_priority: float = 0.15
    disk_free_min_gb: float = 0.5
    max_db_mb: float = 200.0


@dataclass
class NetworkConfig:
    """Внешний контур данных: по умолчанию ВЫКЛЮЧЕН, белый список доменов, квоты."""
    enabled: bool = False
    deny_by_default: bool = True
    allowed_domains: list[str] = field(default_factory=list)
    max_bytes_per_request: int = 60_000
    max_total_bytes_per_day: int = 5_000_000
    max_requests_per_hour: int = 30
    timeout_seconds: int = 15
    respect_robots_txt: bool = True
    user_agent: str = "PFG-Agent/0.1 (autonomous research; contact: local)"
    cache_min_age_s: int = 900          # не качать один URL чаще, чем раз в 15 минут
    max_facts_per_ingest: int = 15
    min_facts_per_ingest: int = 1
    # Поисковые системы для запроса «найди в интернете». Пусто = все известные.
    # Значения: google, yandex, duckduckgo, brave, wikipedia (последняя — через API).
    search_engines: list[str] = field(default_factory=list)


@dataclass
class TrustConfig:
    """Trust = w1*Reputation + w2*Corroboration + w3*Freshness - w4*Conflict - w5*Suspicion."""
    w_reputation: float = 0.30
    w_corroboration: float = 0.25
    w_freshness: float = 0.25
    w_conflict: float = 0.30
    w_suspicion: float = 0.40
    freshness_lambda: float = 1e-5
    min_trust_for_low_risk: float = 0.2
    min_trust_for_fact_confidence: float = 0.5
    conflict_penalty: float = 0.3
    suspicion_penalty: float = 0.4


@dataclass
class CuriosityConfig:
    """Приоритет информационного пробела: Priority = Gap*Relevance*Timeliness / Cost."""
    gap_threshold: float = 0.4
    max_goals_per_cycle: int = 3
    cost_weight: float = 0.5
    lambda_bytes: float = 1e-5      # стоимость за байт
    lambda_latency: float = 0.01    # стоимость за секунду задержки
    parse_difficulty: float = 0.3   # базовая стоимость парсинга
    stale_after_s: int = 86400      # данные старше суток считаются устаревшими


@dataclass
class AgentConfig:
    name: str = "pfg-small"
    seed: int = 42
    log_level: str = "INFO"
    graph: GraphConfig = field(default_factory=GraphConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    motivation: MotivationConfig = field(default_factory=MotivationConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    trust: TrustConfig = field(default_factory=TrustConfig)
    curiosity: CuriosityConfig = field(default_factory=CuriosityConfig)
    benchmark_tasks: list[str] = field(
        default_factory=lambda: [
            "sequence_prediction",
            "memory_recall",
            "arithmetic",
            "file_tool",
            "gridworld_small",
        ]
    )
    log_dir: str = "logs"
    root: Path = field(default_factory=Path.cwd)

    @classmethod
    def load(cls, path: str | Path | None = None, root: Path | None = None) -> "AgentConfig":
        """Загружает конфигурацию из YAML с валидацией ключевых параметров."""
        cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
        raw: dict[str, Any] = {}
        if cfg_path.exists():
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

        def section(name: str, cls_: type, defaults: dict[str, Any] | None = None) -> Any:
            data = dict(raw.get(name, {}) or {})
            if defaults:
                for k, v in defaults.items():
                    data.setdefault(k, v)
            valid = set(cls_.__dataclass_fields__)  # type: ignore[attr-defined]
            unknown = set(data) - valid
            if unknown:
                logging.getLogger("pfg.config").warning("Неизвестные ключи в секции %s: %s", name, sorted(unknown))
                data = {k: v for k, v in data.items() if k in valid}
            return cls_(**data)

        agent_sec = raw.get("agent", {}) or {}
        bench = raw.get("benchmark", {}) or {}
        logging_sec = raw.get("logging", {}) or {}
        mem_defaults = {}
        cfg = cls(
            name=agent_sec.get("name", cls.name),
            seed=int(agent_sec.get("seed", 42)),
            log_level=str(agent_sec.get("log_level", "INFO")).upper(),
            graph=section("graph", GraphConfig),
            learning=section("learning", LearningConfig),
            memory=section("memory", MemoryConfig, mem_defaults),
            planner=section("planner", PlannerConfig),
            executor=section("executor", ExecutorConfig),
            skills=section("skills", SkillsConfig),
            daemon=section("daemon", DaemonConfig),
            motivation=section("motivation", MotivationConfig),
            network=section("network", NetworkConfig),
            trust=section("trust", TrustConfig),
            curiosity=section("curiosity", CuriosityConfig),
            benchmark_tasks=list(bench.get("tasks", cls().benchmark_tasks)),
            log_dir=str(logging_sec.get("dir", "logs")),
            root=Path(root) if root else Path.cwd(),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Проверяет согласованность и безопасность конфигурации."""
        g = self.graph
        if not (1 <= g.node_dim <= 512):
            raise ValueError(f"node_dim вне диапазона 1..512: {g.node_dim}")
        if not (0 < g.top_k_edges <= 64):
            raise ValueError(f"top_k_edges вне диапазона 1..64: {g.top_k_edges}")
        if not (0 <= g.basis_order <= 16):
            raise ValueError(f"basis_order вне диапазона 0..16: {g.basis_order}")
        if g.max_nodes > 4096:
            raise ValueError("max_nodes > 4096 не поддерживается на целевом железе")
        if self.learning.salience_decay <= 0 or self.learning.salience_decay > 1:
            raise ValueError("salience_decay должен быть в (0, 1]")
        if self.planner.simulation_horizon > 50:
            raise ValueError("simulation_horizon слишком велик для RPi4 (>50)")
        # Безопасность: включённые опасные права только явно; никогда implied.
        for cat in ("network", "shell", "money"):
            flag = getattr(self.executor, f"allow_{cat}")
            if flag and cat not in self.executor.require_human_approval_for:
                raise ValueError(
                    f"allow_{cat}=true требует обязательного require_human_approval_for '{cat}'"
                )
        # Внешний контур данных: enabled без белого списка — подозрительно, но не фатально;
        # запрещённые значения квот — ошибка конфигурации.
        n = self.network
        if n.max_bytes_per_request < 1 or n.max_requests_per_hour < 1 or n.timeout_seconds < 1:
            raise ValueError("некорректные квоты network (должны быть >= 1)")
        if n.enabled and n.deny_by_default and not n.allowed_domains:
            logging.getLogger("pfg.config").warning(
                "network.enabled=true, но allowed_domains пуст — любой fetch будет отклонён")

    def resolve_path(self, rel: str) -> Path:
        """Относительный путь конфигурации -> абсолютный от корня проекта."""
        p = Path(rel)
        return p if p.is_absolute() else self.root / p
