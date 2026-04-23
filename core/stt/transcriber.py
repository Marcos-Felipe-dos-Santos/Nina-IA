"""
core.stt.transcriber
====================
Transcrição de áudio com WhisperX.
Recebe áudio (numpy array ou arquivo) e retorna texto + latência.
"""

import logging
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from core.utils.config_loader import obter_secao

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Transcreve áudio usando WhisperX.

    Carrega o modelo uma vez (lazy loading) e reutiliza para
    transcrições subsequentes, minimizando latência.

    Attributes:
        model_name: Nome do modelo Whisper (tiny, base, small, medium, large-v3).
        device: Dispositivo de computação ('cuda' ou 'cpu').
        compute_type: Tipo de computação ('float16', 'int8', etc.).
        language: Idioma forçado para transcrição.
    """

    def __init__(self) -> None:
        """Inicializa o transcritor com parâmetros do config.yaml."""
        config_stt = obter_secao("stt")

        self.model_name: str = config_stt.get("model_name", "small")
        self.device: str = config_stt.get("device", "cuda")
        self.compute_type: str = config_stt.get("compute_type", "float16")
        self.language: Optional[str] = config_stt.get("language", "pt")
        self.batch_size: int = config_stt.get("batch_size", 16)

        # Modelo será carregado sob demanda
        self._modelo = None

        logger.info(
            f"WhisperTranscriber configurado: "
            f"model={self.model_name}, device={self.device}, "
            f"compute_type={self.compute_type}, language={self.language}"
        )

    def _carregar_modelo(self) -> Any:
        """Carrega o modelo WhisperX (lazy loading).

        Returns:
            Modelo WhisperX carregado.
        """
        if self._modelo is None:
            import whisperx

            logger.info(
                f"Carregando modelo WhisperX '{self.model_name}' "
                f"no dispositivo '{self.device}'..."
            )

            self._modelo = whisperx.load_model(
                self.model_name,
                self.device,
                compute_type=self.compute_type,
                language=self.language,
            )

            logger.info("Modelo WhisperX carregado com sucesso.")

        return self._modelo

    def transcrever_array(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> Tuple[str, float]:
        """Transcreve um array numpy de áudio.

        Args:
            audio: Array numpy float32 normalizado [-1.0, 1.0].
            sample_rate: Taxa de amostragem do áudio.

        Returns:
            Tupla (texto_transcrito, latencia_ms).
        """
        import whisperx

        modelo = self._carregar_modelo()

        logger.info(
            f"Transcrevendo áudio: {len(audio)} amostras "
            f"({len(audio) / sample_rate:.2f}s)..."
        )

        inicio = time.perf_counter()

        # WhisperX espera áudio carregado via whisperx.load_audio ou
        # um numpy array float32. Precisamos salvar temporariamente
        # se a taxa de amostragem não for a esperada pelo modelo.
        if sample_rate != 16000:
            # Salvar como WAV temporário e recarregar com resampling
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                audio_int16 = (audio * 32767).astype(np.int16)
                with wave.open(tmp_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_int16.tobytes())

            audio_processado = whisperx.load_audio(tmp_path)
            Path(tmp_path).unlink(missing_ok=True)
        else:
            audio_processado = audio

        # Transcrever
        resultado = modelo.transcribe(
            audio_processado, batch_size=self.batch_size
        )

        fim = time.perf_counter()
        latencia_ms = (fim - inicio) * 1000

        # Extrair texto dos segmentos
        segmentos = resultado.get("segments", [])
        texto = " ".join(seg.get("text", "").strip() for seg in segmentos).strip()

        logger.info(
            f"Transcrição concluída em {latencia_ms:.0f}ms: '{texto[:80]}...'"
            if len(texto) > 80
            else f"Transcrição concluída em {latencia_ms:.0f}ms: '{texto}'"
        )

        return texto, latencia_ms

    def transcrever_arquivo(self, caminho: str | Path) -> Tuple[str, float]:
        """Transcreve um arquivo de áudio.

        Args:
            caminho: Caminho para o arquivo de áudio (WAV, MP3, etc.).

        Returns:
            Tupla (texto_transcrito, latencia_ms).

        Raises:
            FileNotFoundError: Se o arquivo não existir.
        """
        import whisperx

        caminho = Path(caminho)
        if not caminho.exists():
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {caminho}")

        modelo = self._carregar_modelo()

        logger.info(f"Transcrevendo arquivo: {caminho}")

        inicio = time.perf_counter()

        # Carregar e processar áudio
        audio = whisperx.load_audio(str(caminho))
        resultado = modelo.transcribe(audio, batch_size=self.batch_size)

        fim = time.perf_counter()
        latencia_ms = (fim - inicio) * 1000

        # Extrair texto
        segmentos = resultado.get("segments", [])
        texto = " ".join(seg.get("text", "").strip() for seg in segmentos).strip()

        logger.info(
            f"Transcrição do arquivo concluída em {latencia_ms:.0f}ms: '{texto[:80]}'"
        )

        return texto, latencia_ms
