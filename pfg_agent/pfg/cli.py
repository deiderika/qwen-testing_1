"""CLI PFG-Agent: python -m pfg.cli <команда>.

Команды: init, ask, task, train, reflect, evolve, benchmark, export-graph, status.
Внутренний результат — JSON-совместимый; человекочитаемый вывод сверху.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from pfg import __version__


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        stream=sys.stderr,
    )


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def cmd_init(args: argparse.Namespace) -> int:
    from pfg.agent import PFGAgent
    from pfg.config import AgentConfig
    config = AgentConfig.load(args.config)
    agent = PFGAgent(config)
    agent.save()
    agent.close()
    _print_json({"ok": True, "message": "инфраструктура инициализирована",
                 "db": str(config.resolve_path(config.memory.sqlite_path))})
    return 0


def _agent(args: argparse.Namespace):
    from pfg.agent import PFGAgent
    from pfg.config import AgentConfig
    config = AgentConfig.load(args.config)
    _setup_logging(config.log_level if not args.verbose else "DEBUG")
    agent = PFGAgent(config)
    try:
        agent.load()
    except Exception:  # noqa: BLE001 — пустая база до init — норма
        pass
    return agent


def cmd_ask(args: argparse.Namespace) -> int:
    agent = _agent(args)
    try:
        result = agent.run_task(args.goal)
        agent.save()
        payload = result.to_dict()
        payload["goal"] = result.goal
        if result.answer:
            print(f"\n🤖 {result.answer}\n", file=sys.stderr)
        _print_json(payload)
        return 0 if result.success else 1
    finally:
        agent.close()


def cmd_task(args: argparse.Namespace) -> int:
    return cmd_ask(args)


def cmd_train(args: argparse.Namespace) -> int:
    from pfg.benchmark import SequenceTask, ArithmeticTask, MemoryRecallTask, FileToolTask, GridworldTask
    from pfg.config import AgentConfig
    from pfg.storage import Storage
    config = AgentConfig.load(args.config)
    _setup_logging(config.log_level)
    storage = Storage(config.resolve_path(config.memory.sqlite_path))
    task = args.task
    result: dict[str, Any] = {"task": task}
    if task == "sequence":
        st = SequenceTask(config)
        st.train(n_epochs=args.epochs)
        result["accuracy"] = st.evaluate()
    elif task == "arithmetic":
        result["accuracy"] = ArithmeticTask(config, storage).evaluate()
    elif task == "memory":
        acc, forgetting = MemoryRecallTask(config, storage).evaluate()
        result.update({"accuracy": acc, "forgetting": forgetting})
    elif task == "files":
        result["accuracy"] = FileToolTask(config, storage).evaluate()
    elif task == "gridworld":
        result["accuracy"] = GridworldTask(config).evaluate()
    else:
        print(f"неизвестная учебная задача: {task}", file=sys.stderr)
        return 2
    _print_json(result)
    return 0


def cmd_reflect(args: argparse.Namespace) -> int:
    agent = _agent(args)
    try:
        lessons = agent.reflection.reflect(goal="(ручная рефлексия)")
        growth = agent.reflection.apply_growth()
        agent.save()
        _print_json({"lessons": [l.to_dict() for l in lessons], "edges_added": growth})
        return 0
    finally:
        agent.close()


def cmd_evolve(args: argparse.Namespace) -> int:
    agent = _agent(args)
    try:
        info = agent.evolve()
        agent.save()
        _print_json(info)
        return 0
    finally:
        agent.close()


def cmd_benchmark(args: argparse.Namespace) -> int:
    from pfg.benchmark import run_benchmark
    from pfg.config import AgentConfig
    config = AgentConfig.load(args.config)
    _setup_logging(config.log_level)
    report = run_benchmark(config)
    print("\n=== Бенчмарк PFG-Agent ===", file=sys.stderr)
    for k, v in report.items():
        print(f"  {k}: {v}", file=sys.stderr)
    _print_json(report)
    return 0


def cmd_export_graph(args: argparse.Namespace) -> int:
    agent = _agent(args)
    try:
        out = args.out or "data/graph_export.npz"
        path = str(agent.config.resolve_path(out))
        agent.graph.export_npz(path)
        json_path = path.rsplit(".", 1)[0] + ".json"
        from pfg.utils import dump_json
        dump_json({
            "nodes": [n.to_dict() for n in agent.graph.nodes.values()],
            "edges": [e.to_dict() for e in agent.graph.edges.values()],
            "stats": agent.graph.stats(),
        }, json_path)
        _print_json({"npz": path, "json": json_path, "stats": agent.graph.stats()})
        return 0
    finally:
        agent.close()


def cmd_status(args: argparse.Namespace) -> int:
    agent = _agent(args)
    try:
        _print_json(agent.status())
        return 0
    finally:
        agent.close()


# ---------- Lifecycle-команды (демон, REPL, очередь, сон) ----------

def cmd_daemon(args: argparse.Namespace) -> int:
    """Запускает событийный демон: реакция -> автономия -> фаза сна."""
    from pfg.config import AgentConfig
    from pfg.daemon import AgentDaemon
    config = AgentConfig.load(args.config)
    daemon = AgentDaemon(config)
    if args.once:
        daemon.run_once(timeout=args.timeout)
        _print_json(daemon.status())
        daemon.shutdown()
        return 0
    daemon.run_forever()
    return 0


def cmd_repl(args: argparse.Namespace) -> int:
    from pfg.config import AgentConfig
    from pfg.daemon import AgentDaemon, ReplSession
    config = AgentConfig.load(args.config)
    return ReplSession(AgentDaemon(config)).run()


def cmd_queue(args: argparse.Namespace) -> int:
    """queue add "цель" [--priority high|normal|low] | queue list"""
    from pfg.config import AgentConfig
    from pfg.events import EventBus, TaskQueue
    config = AgentConfig.load(args.config)
    tq = TaskQueue(EventBus(), queue_path=config.resolve_path("data/task_queue.json"),
                   outbox_dir=config.resolve_path("sandbox/outbox"))
    if args.action == "add":
        task = tq.add(args.goal, type_=args.type, priority=args.priority)
        _print_json(task)
        return 0
    _print_json(tq.load())
    return 0


def cmd_sleep(args: argparse.Namespace) -> int:
    """Разовый запуск фазы сна (ночная консолидация)."""
    from pfg.agent import PFGAgent
    from pfg.daemon import SleepPhase
    agent = _agent(args)
    try:
        report = SleepPhase(agent).run()
        _print_json(report)
        return 0
    finally:
        agent.close()


def cmd_ingest(args: argparse.Namespace) -> int:
    """Загрузить внешний текст/файл в память через карантинный конвейер ingestion."""
    from pathlib import Path
    from pfg.ingest import IngestPipeline
    agent = _agent(args)
    try:
        pipeline = IngestPipeline(agent.config, agent.memory, agent.storage)
        p = Path(args.path)
        if p.is_dir():
            reports = [pipeline.ingest_file(f).to_dict() for f in sorted(p.glob("*")) if f.is_file()]
        else:
            reports = [pipeline.ingest_file(p).to_dict()]
        agent.save()
        _print_json({"ok": True, "reports": reports})
        return 0
    finally:
        agent.close()


def cmd_web(args: argparse.Namespace) -> int:
    """Запуск веб-обёртки: чат с агентом в браузере."""
    from pfg.web import run_web
    config_path = args.config or "config/default.yaml"
    return run_web(config_path, host=args.host, port=args.port)


def cmd_search(args: argparse.Namespace) -> int:
    """Мультиисточниковый поиск в интернете (белый список + карантин + corroboration)."""
    agent = _agent(args)
    try:
        if agent.web_search is None:
            _print_json({"ok": False,
                         "error": "интернет-контур выключен: включите network.enabled и allowed_domains "
                                  "в конфиге (по умолчанию запрещено осознанно)"})
            return 1
        rep = agent.web_search.search_logged(args.query)
        agent.save()
        _print_json(rep.to_dict())
        print(f"\n>>> {rep.answer or 'ответ не найден'}  (уверенность {rep.confidence:.2f})")
        return 0 if rep.success else 1
    finally:
        agent.close()


def cmd_network_status(args: argparse.Namespace) -> int:
    """Состояние интернет-контура: квоты, трафик, источники."""
    agent = _agent(args)
    try:
        hour_requests, day_bytes = agent.storage.fetch_quota_today()
        sources = [dict(r) for r in agent.storage.get_sources()]
        recent = [dict(r) for r in agent.storage.query(
            "SELECT url, status, bytes, facts_extracted, trust, created_at "
            "FROM fetch_log ORDER BY id DESC LIMIT 10")]
        nc = agent.config.network
        _print_json({
            "enabled": nc.enabled,
            "deny_by_default": nc.deny_by_default,
            "allowed_domains": nc.allowed_domains,
            "quota": {"requests_this_hour": hour_requests, "max_per_hour": nc.max_requests_per_hour,
                      "bytes_today": day_bytes, "max_bytes_per_day": nc.max_total_bytes_per_day},
            "web_search_attached": agent.web_search is not None,
            "sources": sources,
            "recent_fetches": recent,
        })
        return 0
    finally:
        agent.close()


def cmd_curiosity(args: argparse.Namespace) -> int:
    """Что агент хочет изучить: информационные пробелы, отсортированные по Priority."""
    from pfg.curiosity import CuriosityEngine
    agent = _agent(args)
    try:
        eng = CuriosityEngine(agent.config, agent.memory, graph=agent.graph,
                              storage=agent.storage)
        goal = args.goal or "понять мир вокруг и улучшить собственные навыки"
        report = eng.report(goal)
        payload = report.to_dict()
        payload["motivation_goals"] = eng.motivation_goals(report)
        _print_json(payload)
        if not report.gaps:
            print("\n>>> Информационных пробелов выше порога не найдено — агент «всё знает» "
                  "в рамках текущей цели. Экономим ресурсы.")
        else:
            print(f"\n>>> Топ пробелов для цели: {goal!r}")
            for i, g in enumerate(report.gaps, 1):
                url = f" [кандидат: {g.suggested_url}]" if g.suggested_url else ""
                print(f"  {i}. (priority={g.priority:.2f}) {g.topic}"
                      f" | gap={g.gap:.2f} rel={g.relevance:.2f}{url}")
            if not agent.config.network.enabled:
                print("  (сеть выключена — пробелы будут переосмыслены локально через память/self-play)")
        return 0
    finally:
        agent.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pfg.cli", description="PFG-Agent: компактный саморазвивающийся ИИ")
    p.add_argument("--config", help="путь к YAML-конфигурации", default=None)
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG-логирование")
    p.add_argument("--version", action="version", version=f"pfg-agent {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="создать базу, папки и базовый граф").set_defaults(func=cmd_init)

    sp = sub.add_parser("ask", help="задать вопрос/цель агенту")
    sp.add_argument("goal")
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("task", help="поручить агенту задачу")
    sp.add_argument("goal")
    sp.set_defaults(func=cmd_task)

    sp = sub.add_parser("train", help="обучение на учебной задаче")
    sp.add_argument("task", choices=["sequence", "arithmetic", "memory", "files", "gridworld"])
    sp.add_argument("--epochs", type=int, default=60)
    sp.set_defaults(func=cmd_train)

    sub.add_parser("reflect", help="рефлексия по последним эпизодам").set_defaults(func=cmd_reflect)
    sub.add_parser("evolve", help="эволюция графа и компиляция навыков").set_defaults(func=cmd_evolve)
    sub.add_parser("benchmark", help="прогон бенчмарков и отчёт").set_defaults(func=cmd_benchmark)

    sp = sub.add_parser("export-graph", help="экспорт весов в npz+json (для RPi)")
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_export_graph)

    sub.add_parser("status", help="состояние агента").set_defaults(func=cmd_status)

    sp = sub.add_parser("daemon", help="запустить событийный демон (режимы: реакция/автономия/сон)")
    sp.add_argument("--once", action="store_true", help="один виток цикла (отладка)")
    sp.add_argument("--timeout", type=float, default=1.0, help="таймаут poll для --once, сек")
    sp.set_defaults(func=cmd_daemon)

    sub.add_parser("repl", help="интерактивная REPL-сессия с агентом").set_defaults(func=cmd_repl)

    sp = sub.add_parser("queue", help="очередь задач data/task_queue.json")
    sp.add_argument("action", choices=["add", "list"])
    sp.add_argument("goal", nargs="?", default="", help="цель задачи (для add)")
    sp.add_argument("--type", default="generic")
    sp.add_argument("--priority", default="normal", choices=["high", "normal", "low"])
    sp.set_defaults(func=cmd_queue)

    sub.add_parser("sleep", help="разовый запуск фазы сна (консолидация)").set_defaults(func=cmd_sleep)

    sp = sub.add_parser("ingest", help="загрузить внешний текст/файл в память через карантинный конвейер")
    sp.add_argument("path", help="путь к файлу (txt/html/json) или data/inbox/")
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("web", help="запустить веб-интерфейс (чат в браузере)")
    sp.add_argument("--host", default="127.0.0.1", help="адрес (по умолчанию только localhost)")
    sp.add_argument("--port", type=int, default=8765)
    sp.set_defaults(func=cmd_web)

    sp = sub.add_parser("search", help="мультиисточниковый поиск в интернете (белый список + карантин)")
    sp.add_argument("query", help="что искать")
    sp.set_defaults(func=cmd_search)

    sub.add_parser("network-status", help="квоты, трафик и источники интернет-контура").set_defaults(
        func=cmd_network_status)

    sp = sub.add_parser("curiosity", help="показать, что агент хочет изучить (информационные пробелы)")
    sp.add_argument("goal", nargs="?", default=None,
                    help="текущая цель для оценки релевантности (опционально)")
    sp.set_defaults(func=cmd_curiosity)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nпрервано пользователем", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("pfg.cli").exception("ошибка команды")
        _print_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
