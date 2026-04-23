"""
core.pipeline
=============
Pipeline principal da Nina IA.
Une STT → Memória → LLM (com Tools) → TTS em um loop contínuo.
Emite eventos para o dashboard em tempo real.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional

from core.llm.client import NinaLLM
from core.memory.manager import get_memory_manager
from core.stt.microphone import MicrophoneCapture
from core.stt.transcriber import WhisperTranscriber
from core.tools.executor import ToolExecutor
from core.tts.synthesizer import KokoroSynthesizer
from core.utils.latency import LatencyTracker
from core.utils.config_loader import carregar_config
from core.avatar.vtube import VTubeController
from core.avatar.emotion import detect_emotion
from dashboard.events import NinaState, event_bus

logger = logging.getLogger(__name__)


class NinaPipeline:
    """Pipeline completo: ouvir → lembrar → pensar (com tools) → responder → salvar.

    Orquestra os módulos STT, Memória, LLM, Tools e TTS em um loop
    contínuo, exibindo timestamps e latências de cada etapa.
    Emite eventos para o dashboard via EventBus.

    Attributes:
        microfone: Captura de áudio com VAD.
        transcritor: Transcrição WhisperX.
        memoria: Gerenciador de memória de longo prazo.
        llm: Cliente LLM Gemini.
        tools: Executor de ferramentas.
        sintetizador: Síntese Kokoro TTS.
        tracker: Rastreador de latência.
    """

    def __init__(self) -> None:
        """Inicializa todos os componentes do pipeline."""
        logger.info("Inicializando pipeline Nina IA...")

        self.microfone = MicrophoneCapture()
        self.transcritor = WhisperTranscriber()
        self.memoria = get_memory_manager()
        self.tools = ToolExecutor()
        self.llm = NinaLLM()
        self.sintetizador = KokoroSynthesizer()
        self.tracker = LatencyTracker()

        # Configurar tools no LLM para function calling
        self.llm.configurar_tools(self.tools.get_tools_for_gemini())

        # Configuração do Avatar
        self.config = carregar_config()
        self.avatar_enabled = self.config.get("avatar", {}).get("enabled", False)
        if self.avatar_enabled:
            port = self.config.get("avatar", {}).get("vtube_port", 8001)
            plugin_name = self.config.get("avatar", {}).get("plugin_name", "Nina IA")
            self.vtube = VTubeController(port=port, plugin_name=plugin_name)
        else:
            self.vtube = None

        logger.info("Pipeline inicializado com sucesso.")

    def _timestamp(self) -> str:
        """Retorna timestamp formatado para exibição.

        Returns:
            String no formato HH:MM:SS.
        """
        return datetime.now().strftime("%H:%M:%S")

    def _injetar_memorias(self, texto_usuario: str) -> str:
        """Busca memórias relevantes e as injeta na mensagem do usuário.

        Args:
            texto_usuario: Mensagem original do usuário.

        Returns:
            Mensagem enriquecida com memórias, ou a original
            se não houver memórias relevantes.
        """
        if not self.memoria.enabled:
            return texto_usuario

        contexto_memoria = self.memoria.formatar_memorias_para_prompt(texto_usuario)

        if contexto_memoria:
            print(f"[{self._timestamp()}] 📚 Memórias encontradas!")
            logger.info(f"Memórias injetadas: {contexto_memoria[:100]}")
            return f"{contexto_memoria}\n\n{texto_usuario}"

        return texto_usuario

    async def processar_turno(self) -> Optional[str]:
        """Processa um turno completo do pipeline.

        Fluxo: ouvir → lembrar → pensar (com tools) → responder → salvar.
        Emite eventos de estado para o dashboard.

        Returns:
            Texto da resposta da Nina, ou None se não houve fala.
        """
        self.tracker.resetar()

        # ── 1. Captura de áudio ou Injeção ──
        injected_msg = event_bus.pop_injected_message()
        if injected_msg:
            print(f"\n[{self._timestamp()}] 💉 Mensagem injetada via painel...")
            event_bus.set_state(NinaState.LISTENING)
            audio = None
            texto_usuario = injected_msg
        else:
            print(f"\n[{self._timestamp()}] 🎤 Aguardando fala...")
            event_bus.set_state(NinaState.LISTENING)
            audio = self.microfone.gravar_com_vad(timeout_segundos=30.0)

            if audio is None:
                print(f"[{self._timestamp()}] ⏳ Nenhuma fala detectada.")
                event_bus.set_state(NinaState.IDLE)
                return None

            # ── 2. Transcrição STT ──
            print(f"[{self._timestamp()}] 📝 Transcrevendo...")
            self.tracker.iniciar("STT")
            texto_usuario, _ = self.transcritor.transcrever_array(
                audio, sample_rate=self.microfone.sample_rate
            )
            self.tracker.finalizar("STT")

            if not texto_usuario.strip():
                print(f"[{self._timestamp()}] 🔇 Áudio vazio ou inaudível.")
                event_bus.set_state(NinaState.IDLE)
                return None

        print(f"[{self._timestamp()}] 👤 Você: \"{texto_usuario}\"")

        # ── 2.5 Injetar tópico forçado (se houver) ──
        if event_bus.forced_topic:
            topico = event_bus.forced_topic
            print(f"[{self._timestamp()}] 🎯 Injetando tópico: {topico}")
            texto_usuario = f"[CONTEXTO PRIORITÁRIO OBRIGATÓRIO: {topico}]\n\n{texto_usuario}"
            event_bus._forced_topic = None

        # ── 3. Buscar memórias relevantes ──
        print(f"[{self._timestamp()}] 🧠 Buscando memórias...")
        texto_com_memoria = self._injetar_memorias(texto_usuario)

        # ── 4. Processamento LLM (com function calling) ──
        print(f"[{self._timestamp()}] 💭 Pensando...")
        event_bus.set_state(NinaState.THINKING)
        self.tracker.iniciar("LLM")
        resposta = await self.llm.ask(texto_com_memoria)
        self.tracker.finalizar("LLM")

        # ── 5. Síntese TTS e Expressão do Avatar ──
        if resposta.strip():
            print(f"[{self._timestamp()}] 🔊 Sintetizando resposta...")
            event_bus.set_state(NinaState.SPEAKING)
            self.tracker.iniciar("TTS")
            
            if self.avatar_enabled and self.vtube and self.vtube.connected:
                # Detectar emoção localmente
                keywords = self.config.get("avatar", {}).get("keywords", {})
                emocao = detect_emotion(resposta, keywords)
                print(f"[{self._timestamp()}] 🎭 Emoção detectada: {emocao}")
                
                # Disparar expressão e iniciar TTS ao mesmo tempo
                await asyncio.gather(
                    self.vtube.trigger_expression(emocao),
                    asyncio.to_thread(self.sintetizador.sintetizar_e_reproduzir, resposta)
                )
            else:
                await asyncio.to_thread(self.sintetizador.sintetizar_e_reproduzir, resposta)
                
            self.tracker.finalizar("TTS")

        # ── 6. Salvar conversa na memória ──
        print(f"[{self._timestamp()}] 💾 Salvando na memória...")
        self.memoria.save_conversation(texto_usuario, resposta)

        # ── 7. Emitir evento para o dashboard ──
        latencies = {
            "STT": self.tracker.obter_latencia("STT"),
            "LLM": self.tracker.obter_latencia("LLM"),
            "TTS": self.tracker.obter_latencia("TTS"),
        }
        event_bus.add_conversation(texto_usuario, resposta, latencies)

        # ── 8. Exibir latências e voltar ao idle ──
        self.tracker.exibir()
        event_bus.set_state(NinaState.IDLE)

        return resposta

    async def executar(self) -> None:
        """Executa o loop principal do pipeline.

        Loop contínuo que processa turnos de conversa até
        o usuário pressionar Ctrl+C.
        """
        mem_count = self.memoria.count() if self.memoria.enabled else 0

        print("\n" + "=" * 60)
        print("  🤖 Nina IA — Assistente de Voz")
        print("  Pipeline: STT → Memória → LLM (Gemini) → TTS")
        print(f"  Modelo: {self.llm.model}")
        print(f"  Memórias: {mem_count} salvas")
        print(f"  {self.tools.list_tools_summary()}")
        print("  Pressione Ctrl+C para sair")
        print("=" * 60)

        try:
            if self.avatar_enabled and self.vtube:
                print(f"\n[{self._timestamp()}] 🎥 Conectando ao VTube Studio...")
                await self.vtube.connect()
                
            while True:
                try:
                    await self.processar_turno()
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error(f"Erro no turno do pipeline: {e}", exc_info=True)
                    print(f"\n[{self._timestamp()}] ⚠️  Erro no turno: {e}")
                    event_bus.set_state(NinaState.ERROR)
                    await asyncio.sleep(1)
                    event_bus.set_state(NinaState.IDLE)

        except KeyboardInterrupt:
            print(f"\n\n[{self._timestamp()}] 👋 Nina IA encerrada. Até logo!")

        finally:
            self.microfone.encerrar()

    def executar_sync(self) -> None:
        """Versão síncrona do loop principal.

        Conveniência para chamar a partir de main.py sem
        gerenciar o event loop manualmente.
        """
        asyncio.run(self.executar())
