"""
dashboard.api
=============
Backend FastAPI para o dashboard de monitoramento da Nina IA.
Endpoints REST + WebSocket para comunicação em tempo real.
"""

import asyncio
import json
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

from dashboard.events import NinaState, event_bus

logger = logging.getLogger(__name__)

# Diretório dos arquivos estáticos
_STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    """Cria e configura a aplicação FastAPI.

    Returns:
        Instância FastAPI configurada com rotas e middleware.
    """
    app = FastAPI(
        title="Nina IA — Dashboard",
        description="Monitoramento em tempo real da assistente Nina IA",
        version="1.0.0",
    )

    # CORS para desenvolvimento
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Montar arquivos estáticos
    _STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # ── Rotas REST ──────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve a página principal do dashboard."""
        index_path = _STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path, media_type="text/html")
        return HTMLResponse("<h1>Nina IA Dashboard</h1><p>index.html não encontrado.</p>")

    @app.get("/status")
    async def get_status() -> Dict[str, Any]:
        """Retorna o estado atual da Nina.

        Returns:
            JSON com estado, contadores e timestamp.
        """
        return event_bus.get_status()

    @app.get("/history")
    async def get_history() -> Dict[str, Any]:
        """Retorna as últimas 20 trocas de conversa.

        Returns:
            JSON com lista de conversas e total.
        """
        history = event_bus.get_history(20)
        return {"conversations": history, "total": len(history)}

    @app.get("/memories")
    def get_memories() -> Dict[str, Any]:
        """Retorna lista de memórias salvas no ChromaDB.

        Returns:
            JSON com lista de memórias e total.
        """
        try:
            from core.memory.manager import get_memory_manager

            manager = get_memory_manager()
            total = manager.count()

            if total == 0:
                return {"memories": [], "total": 0}

            manager._inicializar_sem_embeddings()
            resultados = manager._collection.get(
                include=["documents", "metadatas"],
            )

            memorias = []
            ids = resultados.get("ids", [])
            docs = resultados.get("documents", [])
            metas = resultados.get("metadatas", [])

            for doc_id, doc, meta in zip(ids, docs, metas):
                memorias.append({
                    "id": doc_id,
                    "summary": doc,
                    "timestamp": meta.get("timestamp", ""),
                    "user_msg": meta.get("user_msg", ""),
                    "nina_response": meta.get("nina_response", ""),
                })

            return {"memories": memorias, "total": len(memorias)}

        except Exception as e:
            logger.error(f"Erro ao buscar memórias: {e}")
            return {"memories": [], "total": 0, "error": str(e)}

    @app.delete("/memories/{memory_id}")
    def delete_memory(memory_id: str) -> Dict[str, Any]:
        """Deleta uma memória específica pelo ID.

        Args:
            memory_id: ID da memória no ChromaDB.

        Returns:
            JSON com status da operação.
        """
        try:
            from core.memory.manager import get_memory_manager

            manager = get_memory_manager()
            manager._inicializar_sem_embeddings()
            manager._collection.delete(ids=[memory_id])

            return {"status": "ok", "deleted": memory_id}

        except Exception as e:
            logger.error(f"Erro ao deletar memória: {e}")
            return {"status": "error", "error": str(e)}

    @app.get("/metrics")
    async def get_metrics() -> Dict[str, Any]:
        """Retorna métricas de latência agregadas.

        Returns:
            JSON com médias, min, max por etapa.
        """
        return event_bus.get_metrics()

    @app.get("/tools/log")
    async def get_tool_log() -> Dict[str, Any]:
        """Retorna log das últimas tools executadas.

        Returns:
            JSON com lista de execuções e total.
        """
        log = event_bus.get_tool_log(20)
        return {"tools": log, "total": len(log)}

    # ── Injeções (Controle) ─────────────────────────────────

    class InjectMessageRequest(BaseModel):
        message: str

    class ForceTopicRequest(BaseModel):
        topic: str

    @app.post("/inject")
    async def inject_message(req: InjectMessageRequest) -> Dict[str, Any]:
        """Injeta uma fala do usuário diretamente no pipeline."""
        event_bus.inject_message(req.message)
        return {"status": "ok", "message": "Mensagem injetada ao pipeline"}

    @app.post("/force_topic")
    async def force_topic(req: ForceTopicRequest) -> Dict[str, Any]:
        """Injeta um contexto forçado na próxima consulta do LLM."""
        event_bus.set_forced_topic(req.topic)
        return {"status": "ok", "topic": "Tópico forçado configurado"}

    # ── WebSocket ───────────────────────────────────────────

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket para stream de eventos em tempo real.

        O frontend se conecta e recebe atualizações automáticas
        de estado, novas conversas e execuções de tools.
        """
        await websocket.accept()
        queue = event_bus.subscribe()

        try:
            # Enviar estado inicial
            await websocket.send_json({
                "type": "initial_state",
                "data": event_bus.get_status(),
            })

            # Loop de eventos
            while True:
                event = await queue.get()
                await websocket.send_json(event)

        except WebSocketDisconnect:
            logger.info("WebSocket desconectado.")
        except Exception as e:
            logger.error(f"Erro no WebSocket: {e}")
        finally:
            event_bus.unsubscribe(queue)

    return app


# Instância global do app
app = create_app()


def start_dashboard(host: str = "0.0.0.0", port: int = 8000) -> threading.Thread:
    """Inicia o dashboard em uma thread separada.

    Args:
        host: Host para bind.
        port: Porta do servidor.

    Returns:
        Thread do servidor.
    """
    def _run():
        uvicorn.run(app, host=host, port=port, log_level="warning")

    thread = threading.Thread(
        target=_run,
        daemon=True,
        name="Dashboard-Server",
    )
    thread.start()

    logger.info(f"Dashboard iniciado em http://{host}:{port}")
    print(f"🌐 Dashboard: http://localhost:{port}")

    return thread
