"""
core.pipeline
=============
Pipeline principal da Nina IA.
Une STT -> Memoria -> LLM (com Tools) -> TTS em um loop continuo.
Emite eventos para o dashboard em tempo real.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional

from core.avatar.emotion import detect_emotion
from core.avatar.vtube import VTubeController
from core.llm.client import NinaLLM
from core.memory.manager import get_memory_manager
from core.stt.microphone import MicrophoneCapture
from core.stt.transcriber import WhisperTranscriber
from core.tools.executor import ToolExecutor
from core.tts.edge_synthesizer import EdgeTTSSynthesizer
from core.tts.synthesizer import KokoroSynthesizer
from core.utils.config_loader import carregar_config
from core.utils.latency import LatencyTracker
from dashboard.events import NinaState, event_bus

logger = logging.getLogger(__name__)


def _truncar_para_voz(texto: str, limite: int = 200) -> str:
    """Trunca o texto na ultima pontuacao natural antes do limite.

    Args:
        texto: Texto original a ser convertido em audio.
        limite: Numero maximo de caracteres desejado.

    Returns:
        Texto reduzido de forma natural para leitura em voz.
    """
    texto_limpo = " ".join((texto or "").split()).strip()
    if len(texto_limpo) <= limite:
        return texto_limpo

    pontuacoes = ".!?:;,"
    ultimo_corte = -1
    for indice, caractere in enumerate(texto_limpo[:limite]):
        if caractere in pontuacoes:
            ultimo_corte = indice

    if ultimo_corte >= 0:
        return texto_limpo[: ultimo_corte + 1].strip()

    ultimo_espaco = texto_limpo.rfind(" ", 0, limite)
    if ultimo_espaco > 0:
        return texto_limpo[:ultimo_espaco].strip()

    return texto_limpo[:limite].strip()


class NinaPipeline:
    """Pipeline completo: ouvir -> lembrar -> pensar -> responder -> salvar."""

    def __init__(self) -> None:
        """Inicializa todos os componentes do pipeline."""
        logger.info("Inicializando pipeline Nina IA...")

        self.config = carregar_config()
        self.microfone = MicrophoneCapture()
        self.transcritor = WhisperTranscriber()
        self.memoria = get_memory_manager()
        self.tools = ToolExecutor()
        self.llm = NinaLLM()
        self.tts_provider = self.config.get("tts", {}).get("provider", "kokoro").lower()
        self.sintetizador = self._criar_sintetizador()
        self.tracker = LatencyTracker()

        self.llm.configurar_tools(self.tools.get_tools_for_gemini())

        self.avatar_enabled = self.config.get("avatar", {}).get("enabled", False)
        if self.avatar_enabled:
            port = self.config.get("avatar", {}).get("vtube_port", 8001)
            plugin_name = self.config.get("avatar", {}).get("plugin_name", "Nina IA")
            self.vtube = VTubeController(port=port, plugin_name=plugin_name)
        else:
            self.vtube = None

        logger.info("Pipeline inicializado com sucesso.")

    def _criar_sintetizador(self):
        """Cria o sintetizador conforme o provider ativo."""
        if self.tts_provider == "edge":
            try:
                logger.info("Inicializando Edge TTS como provider principal.")
                return EdgeTTSSynthesizer()
            except Exception as exc:
                logger.warning(
                    "Falha ao inicializar Edge TTS. Fazendo fallback para Kokoro. Erro: %s",
                    exc,
                )
                self.tts_provider = "kokoro"

        logger.info("Inicializando Kokoro como provider de TTS.")
        return KokoroSynthesizer()

    def _executar_tts_com_fallback(self, texto: str) -> float:
        """Executa o TTS ativo e usa Kokoro se o Edge falhar."""
        try:
            return self.sintetizador.sintetizar_e_reproduzir(texto)
        except Exception as exc:
            if getattr(self, "tts_provider", "kokoro") == "kokoro":
                raise

            logger.warning(
                "Edge TTS falhou durante a sintese. Fazendo fallback para Kokoro. Erro: %s",
                exc,
            )
            self.tts_provider = "kokoro"
            self.sintetizador = KokoroSynthesizer()
            return self.sintetizador.sintetizar_e_reproduzir(texto)

    def _timestamp(self) -> str:
        """Retorna timestamp formatado para exibicao."""
        return datetime.now().strftime("%H:%M:%S")

    def _injetar_memorias(self, texto_usuario: str) -> str:
        """Busca memorias relevantes e as injeta na mensagem do usuario."""
        if not self.memoria.enabled:
            return texto_usuario

        contexto_memoria = self.memoria.formatar_memorias_para_prompt(texto_usuario)

        if contexto_memoria:
            print(f"[{self._timestamp()}] Memorias encontradas!")
            logger.info("Memorias injetadas: %s", contexto_memoria[:100])
            return f"{contexto_memoria}\n\n{texto_usuario}"

        return texto_usuario

    async def processar_turno(self) -> Optional[str]:
        """Processa um turno completo do pipeline."""
        self.tracker.resetar()

        injected_msg = event_bus.pop_injected_message()
        if injected_msg:
            print(f"\n[{self._timestamp()}] Mensagem injetada via painel...")
            event_bus.set_state(NinaState.LISTENING)
            texto_usuario = injected_msg
        else:
            print(f"\n[{self._timestamp()}] Aguardando fala...")
            event_bus.set_state(NinaState.LISTENING)
            audio = self.microfone.gravar_com_vad(timeout_segundos=30.0)

            if audio is None:
                print(f"[{self._timestamp()}] Nenhuma fala detectada.")
                event_bus.set_state(NinaState.IDLE)
                return None

            print(f"[{self._timestamp()}] Transcrevendo...")
            self.tracker.iniciar("STT")
            texto_usuario, _ = self.transcritor.transcrever_array(
                audio,
                sample_rate=self.microfone.sample_rate,
            )
            self.tracker.finalizar("STT")

            if not texto_usuario.strip():
                print(f"[{self._timestamp()}] Audio vazio ou inaudivel.")
                event_bus.set_state(NinaState.IDLE)
                return None

        print(f'[{self._timestamp()}] Voce: "{texto_usuario}"')

        if event_bus.forced_topic:
            topico = event_bus.forced_topic
            print(f"[{self._timestamp()}] Injetando topico: {topico}")
            texto_usuario = f"[CONTEXTO PRIORITARIO OBRIGATORIO: {topico}]\n\n{texto_usuario}"
            event_bus._forced_topic = None

        print(f"[{self._timestamp()}] Buscando memorias...")
        texto_com_memoria = self._injetar_memorias(texto_usuario)

        print(f"[{self._timestamp()}] Pensando...")
        event_bus.set_state(NinaState.THINKING)
        self.tracker.iniciar("LLM")
        resposta = await self.llm.ask(texto_com_memoria)
        self.tracker.finalizar("LLM")

        if resposta.strip():
            resposta_para_voz = _truncar_para_voz(resposta)
            print(f"[{self._timestamp()}] Sintetizando resposta...")
            event_bus.set_state(NinaState.SPEAKING)
            self.tracker.iniciar("TTS")

            if self.avatar_enabled and self.vtube and self.vtube.connected:
                keywords = self.config.get("avatar", {}).get("keywords", {})
                emocao = detect_emotion(resposta, keywords)
                print(f"[{self._timestamp()}] Emocao detectada: {emocao}")

                await asyncio.gather(
                    self.vtube.trigger_expression(emocao),
                    asyncio.to_thread(self._executar_tts_com_fallback, resposta_para_voz),
                )
            else:
                await asyncio.to_thread(self._executar_tts_com_fallback, resposta_para_voz)

            self.tracker.finalizar("TTS")

        print(f"[{self._timestamp()}] Salvando na memoria...")
        self.memoria.save_conversation(texto_usuario, resposta)

        latencies = {
            "STT": self.tracker.obter_latencia("STT"),
            "LLM": self.tracker.obter_latencia("LLM"),
            "TTS": self.tracker.obter_latencia("TTS"),
        }
        event_bus.add_conversation(texto_usuario, resposta, latencies)

        self.tracker.exibir()
        event_bus.set_state(NinaState.IDLE)

        return resposta

    async def executar(self) -> None:
        """Executa o loop principal do pipeline."""
        mem_count = self.memoria.count() if self.memoria.enabled else 0

        print("\n" + "=" * 60)
        print("  Nina IA - Assistente de Voz")
        print("  Pipeline: STT -> Memoria -> LLM (Gemini) -> TTS")
        print(f"  Modelo: {self.llm.model}")
        print(f"  Memorias: {mem_count} salvas")
        print(f"  Ferramentas: {self.tools.registry.count()} ativas")
        print("  Pressione Ctrl+C para sair")
        print("=" * 60)

        try:
            if self.avatar_enabled and self.vtube:
                print(f"\n[{self._timestamp()}] Conectando ao VTube Studio...")
                await self.vtube.connect()

            while True:
                try:
                    await self.processar_turno()
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    logger.error("Erro no turno do pipeline: %s", exc, exc_info=True)
                    print(f"\n[{self._timestamp()}] Erro no turno: {exc}")
                    event_bus.set_state(NinaState.ERROR)
                    await asyncio.sleep(1)
                    event_bus.set_state(NinaState.IDLE)

        except KeyboardInterrupt:
            print(f"\n\n[{self._timestamp()}] Nina IA encerrada. Ate logo!")

        finally:
            self.microfone.encerrar()

    def executar_sync(self) -> None:
        """Versao sincrona do loop principal."""
        asyncio.run(self.executar())
