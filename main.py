"""
Nina IA - Assistente pessoal de voz com IA
==========================================
Ponto de entrada principal do sistema.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import sys
import time
from datetime import datetime

from core.utils.config_loader import carregar_config


def configurar_logging(nivel: str = "INFO") -> None:
    """Configura o sistema de logging global."""
    logging.basicConfig(
        level=getattr(logging, nivel.upper(), logging.INFO),
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    for lib in (
        "torch",
        "lightning",
        "pytorch_lightning",
        "torchaudio",
        "numba",
        "httpcore",
        "httpx",
        "urllib3",
        "filelock",
    ):
        logging.getLogger(lib).setLevel(logging.ERROR)


def warmup_modelos(config: dict) -> dict:
    """Pre-carrega os componentes pesados antes do loop principal."""
    componentes = {}
    total_etapas = 4
    inicio = time.perf_counter()

    print(f"\n  [1/{total_etapas}] Carregando WhisperX STT...", end=" ", flush=True)
    t0 = time.perf_counter()
    from core.stt.transcriber import WhisperTranscriber

    transcritor = WhisperTranscriber()
    transcritor._carregar_modelo()
    componentes["transcritor"] = transcritor
    print(f"OK ({time.perf_counter() - t0:.1f}s)")

    print(f"  [2/{total_etapas}] Carregando silero-vad...", end=" ", flush=True)
    t0 = time.perf_counter()
    from core.stt.microphone import MicrophoneCapture

    microfone = MicrophoneCapture()
    microfone._carregar_vad()
    componentes["microfone"] = microfone
    print(f"OK ({time.perf_counter() - t0:.1f}s)")

    tts_config = config.get("tts", {})
    tts_provider = tts_config.get("provider", "kokoro").lower()
    tts_label = "Edge Neural pt-BR" if tts_provider == "edge" else "Kokoro"
    print(f"  [3/{total_etapas}] Carregando {tts_label}...", end=" ", flush=True)
    t0 = time.perf_counter()
    if tts_provider == "edge":
        from core.tts.edge_synthesizer import EdgeTTSSynthesizer

        sintetizador = EdgeTTSSynthesizer()
    else:
        from core.tts.synthesizer import KokoroSynthesizer

        sintetizador = KokoroSynthesizer()
    sintetizador._carregar_pipeline()
    componentes["sintetizador"] = sintetizador
    print(f"OK ({time.perf_counter() - t0:.1f}s)")

    if config.get("memory", {}).get("enabled", True):
        print(
            f"  [4/{total_etapas}] Carregando ChromaDB + embeddings...",
            end=" ",
            flush=True,
        )
        t0 = time.perf_counter()
        from core.memory.manager import get_memory_manager

        memoria = get_memory_manager()
        memoria._inicializar()
        componentes["memoria"] = memoria
        print(f"OK ({time.perf_counter() - t0:.1f}s)")
    else:
        print(f"  [4/{total_etapas}] Memoria desabilitada, pulando.")

    total_s = time.perf_counter() - inicio
    print(f"\n  Warmup completo em {total_s:.1f}s.\n")

    return componentes


def exibir_banner(config: dict) -> None:
    """Exibe o banner de inicializacao com status dos modulos."""
    config_tts = config.get("tts", {})
    provider_tts = config_tts.get("provider", "kokoro").lower()
    nome_tts = "TTS (Edge Neural pt-BR)" if provider_tts == "edge" else "TTS (Kokoro)"

    modulos = {
        "STT (WhisperX)": True,
        nome_tts: True,
        "LLM (Gemini)": True,
        "Memoria (ChromaDB)": config.get("memory", {}).get("enabled", True),
        "Tools (Function Calling)": True,
        "Visao (Gemini Vision)": config.get("vision", {}).get("enabled", True),
        "Avatar (VTube Studio)": config.get("avatar", {}).get("enabled", False),
        "Dashboard (FastAPI)": True,
    }

    config_llm = config.get("llm", {})
    provider_llm = os.environ.get("LLM_PROVIDER", config_llm.get("provider", "gemini")).lower()

    if provider_llm == "ollama":
        modelo_llm = f"Ollama ({config_llm.get('ollama_model', 'llama3')})"
    else:
        modelo_llm = f"Gemini ({config_llm.get('gemini_model', 'gemini-1.5-flash')})"

    modelo_stt = config.get("stt", {}).get("model_name", "small")
    modo_visao = config.get("vision", {}).get("mode", "manual")
    descricao_tts = (
        f"Edge ({config_tts.get('edge_voice', 'pt-BR-FranciscaNeural')})"
        if provider_tts == "edge"
        else f"Kokoro ({config_tts.get('kokoro_voice', config_tts.get('voice', 'pf_dora'))})"
    )

    print()
    print("=" * 60)
    print("  N I N A   I A  - Assistente de Voz (v1.0.0)")
    print("=" * 60)
    print(f"  Provider LLM: {provider_llm.upper()}")
    for nome, ativo in modulos.items():
        status = "OK" if ativo else "OFF"
        print(f"  [{status}] {nome}")
    print("-" * 60)
    print(f"  Modelo LLM: {modelo_llm}")
    print(f"  Modelo STT: whisperx/{modelo_stt}")
    print(f"  TTS ativo:  {descricao_tts}")
    print(f"  Visao:      modo {modo_visao}")
    print("  Dashboard:  http://localhost:8000")
    print("  Pressione Ctrl+C para encerrar")
    print("=" * 60)
    print()


class GracefulShutdown:
    """Gerencia o encerramento seguro do sistema."""

    def __init__(self) -> None:
        self._pipeline = None
        self._encerrado = False

    def registrar_pipeline(self, pipeline) -> None:
        """Registra o pipeline para encerramento."""
        self._pipeline = pipeline

    def encerrar(self, signum=None, frame=None) -> None:
        """Executa o desligamento gracioso."""
        if self._encerrado:
            return
        self._encerrado = True

        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n\n[{ts}] Iniciando desligamento gracioso...")

        if self._pipeline:
            try:
                self._pipeline.microfone.encerrar()
                print(f"[{ts}] Microfone encerrado")
            except Exception as exc:
                print(f"[{ts}] Microfone: {exc}")

        if self._pipeline and self._pipeline.memoria.enabled:
            try:
                print(f"[{ts}] Memorias salvas ({self._pipeline.memoria.count()} total)")
            except Exception as exc:
                print(f"[{ts}] Memorias: {exc}")

        if self._pipeline and self._pipeline.vtube and self._pipeline.vtube.connected:
            try:
                import asyncio

                loop = asyncio.new_event_loop()
                loop.run_until_complete(self._pipeline.vtube.disconnect())
                loop.close()
                print(f"[{ts}] VTube Studio desconectado")
            except Exception as exc:
                print(f"[{ts}] VTube Studio: {exc}")

        try:
            from dashboard.events import NinaState, event_bus

            event_bus.set_state(NinaState.IDLE)
            print(f"[{ts}] Dashboard notificado")
        except Exception:
            pass

        print(f"[{ts}] Nina desligando. Ate logo!")
        print()


_shutdown = GracefulShutdown()


def main() -> None:
    """Funcao principal do programa."""
    config = carregar_config()
    nivel_log = config.get("general", {}).get("log_level", "INFO")
    configurar_logging(nivel_log)

    print("  Inicializando Nina IA...")
    componentes = warmup_modelos(config)
    exibir_banner(config)

    signal.signal(signal.SIGINT, _shutdown.encerrar)
    signal.signal(signal.SIGTERM, _shutdown.encerrar)
    atexit.register(_shutdown.encerrar)

    from dashboard.api import start_dashboard

    start_dashboard(host="0.0.0.0", port=8000)

    try:
        from core.pipeline import NinaPipeline

        pipeline = NinaPipeline()

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
    except Exception as exc:
        logging.error("Erro fatal: %s", exc)
        _shutdown.encerrar()
        sys.exit(1)


if __name__ == "__main__":
    main()
