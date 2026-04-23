"""
Nina IA — Assistente pessoal de voz com IA
============================================
Ponto de entrada principal do sistema.

Pipeline completo:
    🎤 STT (WhisperX) → 📚 Memória (ChromaDB/RAG) → 🧠 LLM (Gemini)
    → 🔧 Tools (Function Calling) → 👁️ Visão (Gemini Vision) → 🔊 TTS (Kokoro)

Dashboard: http://localhost:8000
"""

import atexit
import logging
import signal
import sys
import time
from datetime import datetime

from core.utils.config_loader import carregar_config


# ── Configuração de logging ─────────────────────────────────
def configurar_logging(nivel: str = "INFO") -> None:
    """Configura o sistema de logging global.

    Args:
        nivel: Nível de log (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, nivel.upper(), logging.INFO),
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Silenciar logs ruidosos do PyTorch, Lightning e libs internas
    for lib in ("torch", "lightning", "pytorch_lightning", "torchaudio",
                "numba", "httpcore", "httpx", "urllib3", "filelock"):
        logging.getLogger(lib).setLevel(logging.ERROR)


# ── Warmup dos modelos ───────────────────────────────────────
def warmup_modelos(config: dict) -> dict:
    """Pré-carrega modelos pesados na VRAM/RAM antes do loop principal.

    Exibe progresso no terminal durante o carregamento.
    Retorna dict com as instâncias pré-aquecidas.

    Args:
        config: Dicionário de configuração.

    Returns:
        Dicionário com instâncias pré-carregadas dos componentes.
    """
    componentes = {}
    total_etapas = 4
    inicio = time.perf_counter()

    # [1/4] WhisperX STT
    print(f"\n  [1/{total_etapas}] ⏳ Carregando WhisperX STT...", end=" ", flush=True)
    t0 = time.perf_counter()
    from core.stt.transcriber import WhisperTranscriber
    transcritor = WhisperTranscriber()
    transcritor._carregar_modelo()
    componentes["transcritor"] = transcritor
    print(f"✅ ({time.perf_counter() - t0:.1f}s)")

    # [2/4] silero-vad
    print(f"  [2/{total_etapas}] ⏳ Carregando silero-vad...", end=" ", flush=True)
    t0 = time.perf_counter()
    from core.stt.microphone import MicrophoneCapture
    microfone = MicrophoneCapture()
    microfone._carregar_vad()
    componentes["microfone"] = microfone
    print(f"✅ ({time.perf_counter() - t0:.1f}s)")

    # [3/4] Kokoro TTS
    print(f"  [3/{total_etapas}] ⏳ Carregando Kokoro TTS...", end=" ", flush=True)
    t0 = time.perf_counter()
    from core.tts.synthesizer import KokoroSynthesizer
    sintetizador = KokoroSynthesizer()
    sintetizador._carregar_pipeline()
    componentes["sintetizador"] = sintetizador
    print(f"✅ ({time.perf_counter() - t0:.1f}s)")

    # [4/4] ChromaDB + Embeddings (se habilitado)
    if config.get("memory", {}).get("enabled", True):
        print(f"  [4/{total_etapas}] ⏳ Carregando ChromaDB + embeddings...", end=" ", flush=True)
        t0 = time.perf_counter()
        from core.memory.manager import get_memory_manager
        memoria = get_memory_manager()
        memoria._inicializar()
        componentes["memoria"] = memoria
        print(f"✅ ({time.perf_counter() - t0:.1f}s)")
    else:
        print(f"  [4/{total_etapas}] ⏭️  Memória desabilitada, pulando.")

    total_s = time.perf_counter() - inicio
    print(f"\n  🚀 Warmup completo em {total_s:.1f}s — todos os modelos na VRAM.\n")

    return componentes


# ── Banner de inicialização ─────────────────────────────────
def exibir_banner(config: dict) -> None:
    """Exibe o banner de inicialização com status dos módulos.

    Args:
        config: Dicionário de configuração carregado do config.yaml.
    """
    # Detectar módulos ativos
    modulos = {
        "🎤 STT (WhisperX)": True,
        "🔊 TTS (Kokoro)": True,
        "🧠 LLM (Gemini)": True,
        "📚 Memória (ChromaDB)": config.get("memory", {}).get("enabled", True),
        "🔧 Tools (Function Calling)": True,
        "👁️  Visão (Gemini Vision)": config.get("vision", {}).get("enabled", True),
        "🎭 Avatar (VTube Studio)": config.get("avatar", {}).get("enabled", False),
        "🌐 Dashboard (FastAPI)": True,
    }

    import os
    config_llm = config.get("llm", {})
    provider_llm = os.environ.get("LLM_PROVIDER", config_llm.get("provider", "gemini")).lower()
    
    if provider_llm == "ollama":
        modelo_llm = f"Ollama ({config_llm.get('ollama_model', 'llama3')})"
    else:
        modelo_llm = f"Gemini ({config_llm.get('gemini_model', config_llm.get('model', 'gemini-1.5-flash'))})"

    modelo_stt = config.get("stt", {}).get("model_name", "small")
    modo_visao = config.get("vision", {}).get("mode", "manual")

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║                                                          ║")
    print("║     🤖  N I N A   I A  —  Assistente de Voz (v1.0.0)     ║")
    print("║                                                          ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║                                                          ║")
    print(f"║    Provider LLM: {provider_llm.upper():<39} ║")

    for nome, ativo in modulos.items():
        status = "✅" if ativo else "❌"
        # Padding para alinhar
        linha = f"  {status} {nome}"
        print(f"║  {linha:<56}║")

    print("║                                                          ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║                                                          ║")
    print(f"║    Modelo LLM: {modelo_llm:<41}║")
    print(f"║    Modelo STT: whisperx/{modelo_stt:<35}║")
    print(f"║    Visão:      modo {modo_visao:<36}║")
    print(f"║    Dashboard:  http://localhost:8000                     ║")
    print("║                                                          ║")
    print("║    Pressione Ctrl+C para encerrar                        ║")
    print("║                                                          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


# ── Desligamento gracioso ────────────────────────────────────
class GracefulShutdown:
    """Gerencia o encerramento seguro do sistema.

    Salva memórias pendentes, fecha conexões e para threads.
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._encerrado = False

    def registrar_pipeline(self, pipeline) -> None:
        """Registra o pipeline para encerramento.

        Args:
            pipeline: Instância do NinaPipeline.
        """
        self._pipeline = pipeline

    def encerrar(self, signum=None, frame=None) -> None:
        """Executa o desligamento gracioso.

        Args:
            signum: Número do sinal (SIGINT/SIGTERM).
            frame: Stack frame (não usado).
        """
        if self._encerrado:
            return
        self._encerrado = True

        ts = datetime.now().strftime("%H:%M:%S")

        print(f"\n\n[{ts}] 🛑 Iniciando desligamento gracioso...")

        # 1. Fechar microfone
        if self._pipeline:
            try:
                self._pipeline.microfone.encerrar()
                print(f"[{ts}]   ✅ Microfone encerrado")
            except Exception as e:
                print(f"[{ts}]   ⚠️  Microfone: {e}")

        # 2. Salvar memórias pendentes
        if self._pipeline and self._pipeline.memoria.enabled:
            try:
                print(f"[{ts}]   ✅ Memórias salvas "
                      f"({self._pipeline.memoria.count()} total)")
            except Exception as e:
                print(f"[{ts}]   ⚠️  Memórias: {e}")

        # 3. Desconectar VTube Studio
        if self._pipeline and self._pipeline.vtube and self._pipeline.vtube.connected:
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self._pipeline.vtube.disconnect())
                loop.close()
                print(f"[{ts}]   ✅ VTube Studio desconectado")
            except Exception as e:
                print(f"[{ts}]   ⚠️  VTube Studio: {e}")

        # 4. Informar estado ao dashboard
        try:
            from dashboard.events import NinaState, event_bus
            event_bus.set_state(NinaState.IDLE)
            print(f"[{ts}]   ✅ Dashboard notificado")
        except Exception:
            pass

        print(f"[{ts}] 👋 Nina desligando... Até logo!")
        print()


# Instância global de shutdown
_shutdown = GracefulShutdown()


# ── Ponto de entrada principal ───────────────────────────────
def main() -> None:
    """Função principal — ponto de entrada do programa.

    Fluxo:
    1. Carrega config.yaml
    2. Configura logging (com filtros de ruído)
    3. Pré-carrega modelos na VRAM (warmup com progresso)
    4. Exibe banner com módulos ativos (APÓS warmup)
    5. Inicia dashboard em thread separada
    6. Inicializa e executa o pipeline de voz
    7. Captura Ctrl+C com desligamento gracioso
    """
    # 1. Carregar configuração
    config = carregar_config()
    nivel_log = config.get("general", {}).get("log_level", "INFO")
    configurar_logging(nivel_log)

    # 2. Warmup — modelos carregam ANTES do banner
    print("  🔄 Inicializando Nina IA...")
    componentes = warmup_modelos(config)

    # 3. Exibir banner (modelos já estão prontos)
    exibir_banner(config)

    # 4. Registrar handlers de sinal
    signal.signal(signal.SIGINT, _shutdown.encerrar)
    signal.signal(signal.SIGTERM, _shutdown.encerrar)
    atexit.register(_shutdown.encerrar)

    # 5. Iniciar dashboard em thread separada
    from dashboard.api import start_dashboard
    start_dashboard(host="0.0.0.0", port=8000)

    # 6. Inicializar e executar pipeline (reaproveita componentes do warmup)
    try:
        from core.pipeline import NinaPipeline

        pipeline = NinaPipeline()

        # Substituir componentes lazy pelos pré-aquecidos
        if "transcritor" in componentes:
            pipeline.transcritor = componentes["transcritor"]
        if "microfone" in componentes:
            pipeline.microfone = componentes["microfone"]
        if "sintetizador" in componentes:
            pipeline.sintetizador = componentes["sintetizador"]

        _shutdown.registrar_pipeline(pipeline)

        pipeline.executar_sync()

    except KeyboardInterrupt:
        _shutdown.encerrar()

    except Exception as e:
        logging.error(f"Erro fatal: {e}")
        _shutdown.encerrar()
        sys.exit(1)


if __name__ == "__main__":
    main()

