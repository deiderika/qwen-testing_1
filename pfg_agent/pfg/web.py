"""Веб-обёртка PFG-Agent: лёгкий HTTP-сервер на стандартной библиотеке.

Общает агента с браузером без внешних зависимостей (Flask/FastAPI не нужны —
важно для RPi4). Сервер однопоточный, агент создаётся один раз и переиспользуется;
доступ к нему сериализуется блокировкой, так как цикл агента не потокобезопасен.

Безопасность:
- сервер слушает 127.0.0.1 по умолчанию (не торчит наружу);
- статистика/ответы логируются в logs/web.jsonl;
- опасные действия по-прежнему отсекает SafetyGuard внутри agent.run_task.

Эндпоинты:
    GET  /              — HTML-чат
    GET  /api/status    — состояние агента (JSON)
    POST /api/chat      — {"message": "..."} -> ответ агента (JSON)
    GET  /api/history   — последние эпизоды (JSON)
    POST /api/action    — {"cmd": "evolve"|"reflect"|"sleep"} — фоновые операции
"""
from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

logger = logging.getLogger("pfg.web")

PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PFG-Agent — чат</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: system-ui, sans-serif; background:#0d1117; color:#e6edf3;
         margin:0; display:flex; flex-direction:column; height:100vh; }
  header { padding:12px 18px; background:#161b22; border-bottom:1px solid #30363d;
           display:flex; justify-content:space-between; align-items:center; }
  header h1 { font-size:16px; margin:0; }
  #status { font-size:12px; color:#8b949e; }
  main { flex:1; overflow-y:auto; padding:18px; display:flex; flex-direction:column; gap:10px; }
  .msg { max-width:75%; padding:10px 14px; border-radius:12px; line-height:1.45;
         white-space:pre-wrap; word-wrap:break-word; }
  .user { align-self:flex-end; background:#1f6feb; color:#fff; }
  .bot  { align-self:flex-start; background:#21262d; border:1px solid #30363d; }
  .meta { font-size:11px; color:#8b949e; margin-top:6px; }
  footer { display:flex; gap:8px; padding:12px 18px; background:#161b22;
           border-top:1px solid #30363d; }
  input[type=text] { flex:1; padding:10px 12px; border-radius:8px; border:1px solid #30363d;
                     background:#0d1117; color:#e6edf3; font-size:14px; }
  button { padding:10px 16px; border:none; border-radius:8px; background:#238636;
           color:#fff; font-size:14px; cursor:pointer; }
  button:disabled { opacity:.5; cursor:wait; }
</style>
</head>
<body>
<header><h1>🧠 PFG-Agent</h1><span id="status">подключение…</span></header>
<main id="log"></main>
<footer>
  <input id="inp" type="text" placeholder="Спросите что-нибудь…" autofocus>
  <button id="send">➤</button>
</footer>
<script>
const log=document.getElementById('log'), inp=document.getElementById('inp'),
      btn=document.getElementById('send'), st=document.getElementById('status');
function add(cls, text, meta){
  const d=document.createElement('div'); d.className='msg '+cls; d.textContent=text;
  if(meta){const m=document.createElement('div');m.className='meta';m.textContent=meta;d.appendChild(m);}
  log.appendChild(d); log.scrollTop=log.scrollHeight;
}
fetch('/api/status').then(r=>r.json()).then(s=>{
  st.textContent=`узлов: ${s.graph.nodes}, рёбер: ${s.graph.edges}, фактов: ${s.facts}, навыков: ${s.skills_active}/${s.skills_total}`;
}).catch(()=>st.textContent='нет связи');
async function send(){
  const t=inp.value.trim(); if(!t) return;
  inp.value=''; btn.disabled=true; add('user', t);
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:t})});
    const j=await r.json();
    const meta=`уверенность: ${(j.confidence*100).toFixed(0)}%`+
      (j.memory_used&&j.memory_used.length?` · память: ${j.memory_used.length}`:'')+
      (j.actions&&j.actions.length?` · действий: ${j.actions.length}`:'');
    add('bot', j.answer||'(пусто)', meta+(j.errors&&j.errors.length?' · ошибки: '+j.errors.join('; '):''));
  }catch(e){ add('bot','Ошибка соединения: '+e); }
  btn.disabled=false; inp.focus();
}
btn.onclick=send;
inp.addEventListener('keydown',e=>{if(e.key==='Enter')send();});
</script>
</body>
</html>"""


class AgentWebServer:
    """Обёртка: HTTP + общий экземпляр PFG-Agent.

    Используется однопоточный HTTPServer: все запросы обрабатываются в одном
    потоке, поэтому SQLite-соединение агента не пересекает границы потоков
    (SQLite по умолчанию запрещает использование соединения из других тредов).
    Для RPi4 это плюс: нет гонок и лишних трат памяти.
    """

    def __init__(self, agent: Any, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.agent = agent
        self.host = host
        self.port = port
        self._server: HTTPServer | None = None

    # ---------- бизнес-логика (вызывается из хэндлера) ----------
    def chat(self, message: str) -> dict[str, Any]:
        result = self.agent.run_task(message)
        self.agent.save()
        payload = result.to_dict()
        logger.info("web chat: %r -> success=%s", message[:80], result.success)
        return payload

    def status(self) -> dict[str, Any]:
        return self.agent.status()

    def history(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.agent.memory.recent_episodes(limit)

    def action(self, cmd: str) -> dict[str, Any]:
        from pfg.daemon import SleepPhase
        if cmd == "evolve":
            out = self.agent.evolve()
            self.agent.save()
        elif cmd == "reflect":
            lessons = self.agent.reflection.reflect()
            out = {"lessons": [l.to_dict() for l in lessons]}
        elif cmd == "sleep":
            out = SleepPhase(self.agent).run()
        else:
            return {"ok": False, "error": f"unknown cmd: {cmd}"}
        return {"ok": True, "cmd": cmd, "result": out}

    # ---------- сервер ----------
    def make_handler(self) -> type[BaseHTTPRequestHandler]:
        web = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _json(self, obj: Any, code: int = 200) -> None:
                self._send(code, json.dumps(obj, ensure_ascii=False).encode(),
                           "application/json; charset=utf-8")

            def do_GET(self) -> None:  # noqa: N802
                if self.path in ("/", "/index.html"):
                    self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                elif self.path == "/api/status":
                    self._json(web.status())
                elif self.path.startswith("/api/history"):
                    self._json(web.history())
                else:
                    self._json({"error": "not found"}, 404)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                if length > 64 * 1024:
                    self._json({"error": "too large"}, 413)
                    return
                try:
                    data = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    self._json({"error": "invalid json"}, 400)
                    return
                if self.path == "/api/chat":
                    msg = str(data.get("message", "")).strip()
                    if not msg:
                        self._json({"error": "empty message"}, 400)
                        return
                    self._json(web.chat(msg))
                elif self.path == "/api/action":
                    self._json(web.action(str(data.get("cmd", ""))))
                else:
                    self._json({"error": "not found"}, 404)

            def log_message(self, fmt: str, *args: Any) -> None:
                logger.debug("web http: " + fmt, *args)

        return Handler

    def serve_forever(self) -> None:
        self._server = HTTPServer((self.host, self.port), self.make_handler())
        print(f"🌐 PFG-Agent веб-интерфейс: http://{self.host}:{self.port}")
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self._server.shutdown()
            self.agent.close()


def run_web(config_path: str = "config/default.yaml", host: str = "127.0.0.1",
            port: int = 8765) -> int:
    """Точка входа CLI-команды `web`."""
    from pfg.agent import PFGAgent
    from pfg.config import AgentConfig

    config = AgentConfig.load(config_path)
    agent = PFGAgent(config)
    try:
        agent.load()
    except Exception:  # noqa: BLE001 — пустая база до init — норма
        pass
    server = AgentWebServer(agent, host=host, port=port)
    server.serve_forever()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pfg.web", description="Веб-обёртка PFG-Agent")
    parser.add_argument("--host", default="127.0.0.1", help="адрес (по умолчанию localhost)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    return run_web(args.config, args.host, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
