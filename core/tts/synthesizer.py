"""
core.tts.synthesizer
====================
Síntese de voz com Kokoro TTS.
Gera áudio a partir de texto e reproduz via sounddevice.
"""

import logging
import time
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd

from core.utils.config_loader import obter_secao

logger = logging.getLogger(__name__)


class KokoroSynthesizer:
    """Sintetiza fala usando Kokoro TTS.

    Carrega o pipeline uma vez (lazy loading) e reutiliza para
    sínteses subsequentes.

    Attributes:
        lang_code: Código de idioma Kokoro ('a' = inglês, 'p' = pt-br).
        voice: Nome da voz (ex: 'pf_dora', 'pm_alex').
        speed: Velocidade da fala (1.0 = normal).
        sample_rate: Taxa de amostragem da saída (24000 Hz padrão do Kokoro).
    """

    def __init__(self) -> None:
        """Inicializa o sintetizador com parâmetros do config.yaml."""
        config_tts = obter_secao("tts")

        self.lang_code: str = config_tts.get("lang_code", "p")
        self.voice: str = config_tts.get("voice", "pf_dora")
        self.speed: float = config_tts.get("speed", 1.0)
        self.sample_rate: int = config_tts.get("sample_rate", 24000)

        # Pipeline será carregado sob demanda
        self._pipeline = None

        logger.info(
            f"KokoroSynthesizer configurado: "
            f"lang_code={self.lang_code}, voice={self.voice}, "
            f"speed={self.speed}, sample_rate={self.sample_rate}"
        )

    def _carregar_pipeline(self):
        """Carrega o pipeline Kokoro (lazy loading).

        Returns:
            Pipeline Kokoro inicializado.
        """
        if self._pipeline is None:
            from kokoro import KPipeline

            logger.info(
                f"Carregando pipeline Kokoro "
                f"(lang_code='{self.lang_code}')..."
            )

            self._pipeline = KPipeline(lang_code=self.lang_code)

            logger.info("Pipeline Kokoro carregado com sucesso.")

        return self._pipeline

    def sintetizar(self, texto: str) -> Tuple[np.ndarray, float]:
        """Sintetiza texto em áudio.

        Args:
            texto: Texto a ser convertido em fala.

        Returns:
            Tupla (audio_array, latencia_ms).
            audio_array é um numpy float32 com taxa de 24kHz.
        """
        pipeline = self._carregar_pipeline()

        logger.info(f"Sintetizando: '{texto[:60]}...'")

        inicio = time.perf_counter()

        # Kokoro retorna um gerador de (grafemas, fonemas, audio)
        # Concatenamos todos os chunks para obter o áudio completo
        chunks_audio = []
        for grafemas, fonemas, audio_chunk in pipeline(
            texto, voice=self.voice, speed=self.speed
        ):
            if audio_chunk is not None:
                chunks_audio.append(audio_chunk)

        fim = time.perf_counter()
        latencia_ms = (fim - inicio) * 1000

        if not chunks_audio:
            logger.error("Kokoro não gerou áudio para o texto fornecido.")
            return np.array([], dtype=np.float32), latencia_ms

        # Concatenar todos os chunks
        audio_completo = np.concatenate(chunks_audio)

        duracao_audio = len(audio_completo) / self.sample_rate
        logger.info(
            f"Síntese concluída em {latencia_ms:.0f}ms "
            f"(áudio: {duracao_audio:.2f}s)"
        )

        return audio_completo, latencia_ms

    def reproduzir(self, audio: np.ndarray) -> None:
        """Reproduz um array de áudio via sounddevice.

        Args:
            audio: Array numpy float32 com o áudio a reproduzir.
        """
        if len(audio) == 0:
            logger.warning("Áudio vazio, nada para reproduzir.")
            return

        duracao = len(audio) / self.sample_rate
        logger.info(f"Reproduzindo áudio ({duracao:.2f}s)...")
        print(f"🔊 Reproduzindo áudio ({duracao:.2f}s)...")

        sd.play(audio, samplerate=self.sample_rate)
        sd.wait()  # Aguardar término da reprodução

        logger.info("Reprodução concluída.")

    def sintetizar_e_reproduzir(self, texto: str) -> float:
        """Sintetiza e reproduz o texto em uma única chamada.

        Args:
            texto: Texto a ser convertido e reproduzido.

        Returns:
            Latência da síntese em milissegundos (não inclui reprodução).
        """
        audio, latencia_ms = self.sintetizar(texto)
        self.reproduzir(audio)
        return latencia_ms
