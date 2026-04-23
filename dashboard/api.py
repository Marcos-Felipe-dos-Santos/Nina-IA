"""
dashboard.api
=============
Backend FastAPI para o dashboard de monitoramento da Nina IA.
Endpoints REST e WebSocket para comunicacao em tempo real.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dashboard.events import event_bus

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


class InjectMessageRequest(BaseModel):
    """Payload JSON para injecao de mensagem no pipeline."""

    message: str


class ForceTopicRequest(BaseModel):
    """Payload JSON para injecao de topico forcado."""

    topic: str


def create_app() -> FastAPI:
    """Cria e configura a aplicacao FastAPI."""
    app = FastAPI(
        title="Nina IA - Dashboard",
        description="Monitoramento em tempo real da assistente Nina IA",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve a pagina principal do dashboard."""
        index_path = _STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path, media_type="text/html")
        return HTMLResponse("<h1>Nina IA Dashboard</h1><p>index.html nao encontrado.</p>")

    @app.get("/status")
    async def get_status() -> Dict[str, Any]:
        """Retorna o estado atual da Nina."""
        return event_bus.get_status()

    @app.get("/history")
    async def get_history() -> Dict[str, Any]:
        """Retorna as ultimas 20 trocas de conversa."""
        history = event_bus.get_history(20)
        return {"conversations": history, "total": len(history)}

    @app.get("/memories")
    def get_memories() -> Dict[str, Any]:
        """Retorna lista de memorias salvas no ChromaDB."""
        try:
            from core.memory.manager import get_memory_manager

            manager = get_memory_manager()
            total = manager.count()

            if total == 0:
                return {"memories": [], "total": 0}

            manager._inicializar_sem_embeddings()
            resultados = manager._collection.get(include=["documents", "metadatas"])

            memorias = []
            ids = resultados.get("ids", [])
            docs = resultados.get("documents", [])
            metas = resultados.get("metadatas", [])

            for doc_id, doc, meta in zip(ids, docs, metas):
                memorias.append(
                    {
                        "id": doc_id,
                        "summary": doc,
                        "timestamp": meta.get("timestamp", ""),
                        "user_msg": meta.get("user_msg", ""),
                        "nina_response": meta.get("nina_response", ""),
                    }
                )

            return {"memories": memorias, "total": len(memorias)}

        except Exception as exc:
            logger.error("Erro ao buscar memorias: %s", exc)
            return {"memories": [], "total": 0, "error": str(exc)}

    @app.delete("/memories/{memory_id}")
    def delete_memory(memory_id: str) -> Dict[str, Any]:
        """Deleta uma memoria especifica pelo ID."""
        try:
            from core.memory.manager import get_memory_manager

            manager = get_memory_manager()
            manager._inicializar_sem_embeddings()
            manager._collection.delete(ids=[memory_id])
            return {"status": "ok", "deleted": memory_id}

        except Exception as exc:
            logger.error("Erro ao deletar memoria: %s", exc)
            return {"status": "error", "error": str(exc)}

    @app.get("/metrics")
    async def get_metrics() -> Dict[str, Any]:
        """Retorna metricas de latencia agregadas."""
        return event_bus.get_metrics()

    @app.get("/tools/log")
    async def get_tool_log() -> Dict[str, Any]:
        """Retorna log das ultimas tools executadas."""
        log = event_bus.get_tool_log(20)
        return {"tools": log, "total": len(log)}

    @app.post("/inject")
    async def inject_message(req: InjectMessageRequest) -> Dict[str, Any]:
        """Injeta uma fala do usuario diretamente no pipeline."""
        event_bus.inject_message(req.message)
        return {"status": "ok", "message": "Mensagem injetada ao pipeline"}

    @app.post("/force_topic")
    async def force_topic(req: ForceTopicRequest) -> Dict[str, Any]:
        """Injeta um contexto forcado na proxima consulta do LLM."""
        event_bus.set_forced_topic(req.topic)
        return {"status": "ok", "topic": "Topico forcado configurado"}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """Stream de eventos em tempo real para o dashboard."""
        await websocket.accept()
        queue = event_bus.subscribe()

        try:
            await websocket.send_json(
                {
                    "type": "initial_state",
                    "data": event_bus.get_status(),
                }
            )

            while True:
                event = await queue.get()
                await websocket.send_json(event)

        except WebSocketDisconnect:
            logger.info("WebSocket desconectado.")
        except Exception as exc:
            logger.error("Erro no WebSocket: %s", exc)
        finally:
            event_bus.unsubscribe(queue)

    return app


app = create_app()


def start_dashboard(host: str = "0.0.0.0", port: int = 8000) -> threading.Thread:
    """Inicia o dashboard em uma thread separada."""

    def _run():
        uvicorn.run(app, host=host, port=port, log_level="warning")

    thread = threading.Thread(
        target=_run,
        daemon=True,
        name="Dashboard-Server",
    )
    thread.start()

    logger.info("Dashboard iniciado em http://%s:%s", host, port)
    print(f"Dashboard: http://localhost:{port}")

    return thread
