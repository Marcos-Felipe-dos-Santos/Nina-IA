"""
dashboard.events
================
Sistema de eventos em tempo real para o dashboard.
Barramento centralizado que conecta o pipeline ao frontend via WebSocket.
"""

import asyncio
import logging
import time
from collections import deque
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class NinaState(str, Enum):
    """Estados possíveis da Nina."""
    IDLE = "idle"           # Aguardando
    LISTENING = "listening"  # Ouvindo (VAD detectou fala)
    THINKING = "thinking"    # Processando LLM
    SPEAKING = "speaking"    # Sintetizando/reproduzindo TTS
    ERROR = "error"          # Erro no pipeline


class EventBus:
    """Barramento de eventos em tempo real da Nina IA.

    Centraliza estado, histórico de conversas, métricas e
    logs de tools. Emite eventos para os WebSockets do dashboard.

    Uso singleton: importe `event_bus` para acessar a instância global.
    """

    def __init__(self) -> None:
        """Inicializa o barramento de eventos."""
        # Estado atual
        self._state: NinaState = NinaState.IDLE

        # Histórico de conversas (últimas 50)
        self._history: deque = deque(maxlen=50)

        # Log de tools executadas (últimas 50)
        self._tool_log: deque = deque(maxlen=50)

        # Métricas de latência (acumuladas)
        self._latency_samples: Dict[str, List[float]] = {
            "STT": [], "LLM": [], "TTS": []
        }

        # Forçar tópico (contexto)
        self._forced_topic: Optional[str] = None

        # Fila de injeções via painel
        self._injected_messages: deque = deque()

        # Assinantes WebSocket: lista de tuplas (Queue, EventLoop)
        self._subscribers: List[tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = []

        logger.info("EventBus inicializado.")

    # ── Injeções de Painel ──────────────────────────────────
    @property
    def forced_topic(self) -> Optional[str]:
        return self._forced_topic

    def set_forced_topic(self, topic: str) -> None:
        """Define um tópico prioritário a ser injetado."""
        self._forced_topic = topic
        logger.info(f"Tópico forçado definido: {topic}")
        # Notificar o dashboard em tempo real
        self._emit_event({
            "type": "injection_active",
            "data": {"type": "topic", "text": topic, "timestamp": datetime.now().isoformat()}
        })

    def inject_message(self, message: str) -> None:
        """Enfileira uma mensagem para contornar o microfone."""
        self._injected_messages.append(message)
        logger.info(f"Mensagem injetada: {message}")
        # Notificar o dashboard
        self._emit_event({
            "type": "injection_active",
            "data": {"type": "message", "text": message, "timestamp": datetime.now().isoformat()}
        })

    def pop_injected_message(self) -> Optional[str]:
        """Remove e retorna a mensagem injetada mais antiga."""
        if self._injected_messages:
            return self._injected_messages.popleft()
        return None

    # ── Estado ──────────────────────────────────────────────
    @property
    def state(self) -> str:
        """Retorna o estado atual da Nina."""
        return self._state.value

    def set_state(self, state: NinaState) -> None:
        """Atualiza o estado e emite evento.

        Args:
            state: Novo estado.
        """
        self._state = state
        self._emit_event({
            "type": "state_change",
            "state": state.value,
            "timestamp": datetime.now().isoformat(),
        })

    # ── Histórico ───────────────────────────────────────────
    def add_conversation(
        self,
        user_msg: str,
        nina_response: str,
        latencies: Optional[Dict[str, float]] = None,
    ) -> None:
        """Adiciona uma troca de conversa ao histórico.

        Args:
            user_msg: Mensagem do usuário.
            nina_response: Resposta da Nina.
            latencies: Dicionário com latências {STT, LLM, TTS}.
        """
        entry = {
            "user": user_msg,
            "nina": nina_response,
            "timestamp": datetime.now().isoformat(),
            "latencies": latencies or {},
        }
        self._history.append(entry)

        # Acumular latências para métricas
        if latencies:
            for key, val in latencies.items():
                if key in self._latency_samples and val is not None:
                    self._latency_samples[key].append(val)

        self._emit_event({
            "type": "new_conversation",
            "data": entry,
        })

    def get_history(self, n: int = 20) -> List[Dict[str, Any]]:
        """Retorna as últimas N conversas.

        Args:
            n: Número máximo de conversas.

        Returns:
            Lista de dicionários com user, nina, timestamp, latencies.
        """
        items = list(self._history)
        return items[-n:]

    # ── Tools ───────────────────────────────────────────────
    def log_tool(self, tool_name: str, params: Dict, result: str) -> None:
        """Registra a execução de uma tool.

        Args:
            tool_name: Nome da ferramenta.
            params: Parâmetros usados.
            result: Resultado da execução.
        """
        entry = {
            "tool": tool_name,
            "params": params,
            "result": result[:200],
            "timestamp": datetime.now().isoformat(),
        }
        self._tool_log.append(entry)

        self._emit_event({
            "type": "tool_executed",
            "data": entry,
        })

    def get_tool_log(self, n: int = 20) -> List[Dict]:
        """Retorna as últimas N execuções de tools."""
        items = list(self._tool_log)
        return items[-n:]

    # ── Métricas ────────────────────────────────────────────
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de latência agregadas.

        Returns:
            Dicionário com médias, mínimos e máximos por etapa.
        """
        metrics = {}
        for key, samples in self._latency_samples.items():
            if samples:
                metrics[key] = {
                    "avg_ms": round(sum(samples) / len(samples), 1),
                    "min_ms": round(min(samples), 1),
                    "max_ms": round(max(samples), 1),
                    "count": len(samples),
                }
            else:
                metrics[key] = {
                    "avg_ms": 0, "min_ms": 0, "max_ms": 0, "count": 0
                }

        # Total
        all_avgs = [m["avg_ms"] for m in metrics.values() if m["count"] > 0]
        metrics["TOTAL"] = {
            "avg_ms": round(sum(all_avgs), 1),
            "count": min(m["count"] for m in metrics.values()) if metrics else 0,
        }

        return metrics

    # ── WebSocket ───────────────────────────────────────────
    def subscribe(self) -> asyncio.Queue:
        """Registra um novo assinante WebSocket.

        Returns:
            Queue assíncrona (limitada a 100 eventos) que receberá os eventos.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        loop = asyncio.get_running_loop()
        self._subscribers.append((queue, loop))
        logger.info(f"Novo assinante WS. Total: {len(self._subscribers)}")
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Remove um assinante WebSocket."""
        for sub in self._subscribers[:]:
            if sub[0] is queue:
                self._subscribers.remove(sub)
                logger.info(f"Assinante WS removido. Total: {len(self._subscribers)}")

    def _emit_event(self, event: Dict[str, Any]) -> None:
        """Emite um evento para todos os assinantes via call_soon_threadsafe.

        Remove automaticamente subscribers cujo event loop foi fechado.

        Args:
            event: Dicionário do evento.
        """
        dead_subscribers = []

        for sub in self._subscribers:
            queue, loop = sub
            try:
                if loop.is_closed():
                    dead_subscribers.append(sub)
                    continue

                def _put(q=queue, e=event):
                    try:
                        q.put_nowait(e)
                    except asyncio.QueueFull:
                        pass

                # Se for o mesmo loop, executa direto
                current_loop = None
                try:
                    current_loop = asyncio.get_running_loop()
                except RuntimeError:
                    pass

                if loop is current_loop:
                    _put()
                else:
                    loop.call_soon_threadsafe(_put)
            except Exception as e:
                logger.error(f"Erro ao emitir evento WS: {e}")
                dead_subscribers.append(sub)

        # Limpar subscribers orfãos
        for sub in dead_subscribers:
            if sub in self._subscribers:
                self._subscribers.remove(sub)
                logger.info(f"Subscriber orfão removido. Total: {len(self._subscribers)}")

    # ── Status ──────────────────────────────────────────────
    def get_status(self) -> Dict[str, Any]:
        """Retorna o status completo da Nina.

        Returns:
            Dicionário com estado, contadores e uptime.
        """
        return {
            "state": self._state.value,
            "total_conversations": len(self._history),
            "total_tools_executed": len(self._tool_log),
            "active_websockets": len(self._subscribers),
            "timestamp": datetime.now().isoformat(),
        }


# Instância global (singleton)
event_bus = EventBus()
