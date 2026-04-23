"""
core.stt.microphone
===================
Captura de áudio do microfone com PyAudio.
Integra silero-vad para detecção automática de fala (VAD).
"""

import logging
import wave
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudio
import torch

from core.utils.config_loader import obter_secao

logger = logging.getLogger(__name__)


class MicrophoneCapture:
    """Captura áudio do microfone com suporte a VAD (Voice Activity Detection).

    Usa PyAudio para captura contínua e silero-vad para detectar
    automaticamente quando o usuário começa e para de falar.

    Attributes:
        sample_rate: Taxa de amostragem em Hz.
        chunk_size: Número de amostras por frame.
        vad_threshold: Limiar de confiança para detecção de fala.
    """

    def __init__(self) -> None:
        """Inicializa a captura com parâmetros do config.yaml."""
        config_vad = obter_secao("vad")
        config_stt = obter_secao("stt")

        self.sample_rate: int = config_vad.get("sample_rate", 16000)
        self.chunk_size: int = config_vad.get("chunk_size", 512)
        self.vad_threshold: float = config_vad.get("threshold", 0.5)
        self.min_silence_ms: int = config_vad.get("min_silence_duration_ms", 700)
        self.speech_pad_ms: int = config_vad.get("speech_pad_ms", 300)
        self.device_index: Optional[int] = config_stt.get("device_index")

        # PyAudio será inicializado sob demanda
        self._pa: Optional[pyaudio.PyAudio] = None

        # Modelo VAD será carregado sob demanda
        self._vad_model = None
        self._vad_iterator = None

        logger.info(
            f"MicrophoneCapture configurado: "
            f"sample_rate={self.sample_rate}, chunk_size={self.chunk_size}"
        )

    def _inicializar_pyaudio(self) -> pyaudio.PyAudio:
        """Inicializa o PyAudio (lazy loading)."""
        if self._pa is None:
            self._pa = pyaudio.PyAudio()
        return self._pa

    def _carregar_vad(self) -> None:
        """Carrega o modelo silero-vad (lazy loading)."""
        if self._vad_model is None:
            from silero_vad import load_silero_vad, VADIterator

            logger.info("Carregando modelo silero-vad...")
            self._vad_model = load_silero_vad()
            self._vad_iterator = VADIterator(
                self._vad_model,
                sampling_rate=self.sample_rate,
                threshold=self.vad_threshold,
                min_silence_duration_ms=self.min_silence_ms,
                speech_pad_ms=self.speech_pad_ms,
            )
            logger.info("Modelo silero-vad carregado com sucesso.")

    def gravar_segundos(self, duracao_segundos: float = 3.0) -> np.ndarray:
        """Grava um número fixo de segundos do microfone.

        Args:
            duracao_segundos: Duração da gravação em segundos.

        Returns:
            Array numpy float32 normalizado com o áudio capturado.
        """
        pa = self._inicializar_pyaudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
        )

        total_frames = int(self.sample_rate * duracao_segundos / self.chunk_size)
        logger.info(
            f"Gravando {duracao_segundos}s de áudio "
            f"({total_frames} frames de {self.chunk_size} amostras)..."
        )

        frames: list[bytes] = []
        for _ in range(total_frames):
            dados = stream.read(self.chunk_size, exception_on_overflow=False)
            frames.append(dados)

        stream.stop_stream()
        stream.close()

        # Converter para float32 normalizado
        audio_int16 = np.frombuffer(b"".join(frames), dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        logger.info(
            f"Gravação concluída: {len(audio_float32)} amostras "
            f"({len(audio_float32) / self.sample_rate:.2f}s)"
        )

        return audio_float32

    def gravar_com_vad(self, timeout_segundos: float = 30.0) -> Optional[np.ndarray]:
        """Grava do microfone com detecção automática de fala via VAD.

        Aguarda o usuário começar a falar, captura até detectar silêncio,
        e retorna o áudio da fala completa.

        Args:
            timeout_segundos: Tempo máximo de espera antes de desistir.

        Returns:
            Array numpy float32 com o áudio da fala, ou None se timeout.
        """
        self._carregar_vad()
        pa = self._inicializar_pyaudio()

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
        )

        logger.info("Aguardando fala (VAD ativo)...")
        print("🎤 Ouvindo... (fale algo)")

        frames_coletados: list[np.ndarray] = []
        fala_detectada = False
        max_chunks = int(self.sample_rate * timeout_segundos / self.chunk_size)

        try:
            for i in range(max_chunks):
                dados = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_int16 = np.frombuffer(dados, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                audio_tensor = torch.from_numpy(audio_float32)

                # Alimentar o VAD
                resultado_vad = self._vad_iterator(
                    audio_tensor, return_seconds=False
                )

                # Coletar frames durante a fala
                if resultado_vad is not None:
                    if "start" in resultado_vad:
                        fala_detectada = True
                        logger.info("Fala detectada pelo VAD.")
                        print("🗣️  Fala detectada!")

                    if "end" in resultado_vad and fala_detectada:
                        logger.info("Fim da fala detectado pelo VAD.")
                        print("🔇 Fim da fala detectado.")
                        break

                # Sempre coletar frames após detectar fala
                if fala_detectada:
                    frames_coletados.append(audio_float32)

            # Resetar o VAD para próximo uso
            if self._vad_iterator is not None:
                self._vad_iterator.reset_states()

        finally:
            stream.stop_stream()
            stream.close()

        if not frames_coletados:
            logger.warning("Nenhuma fala detectada dentro do timeout.")
            return None

        audio_completo = np.concatenate(frames_coletados)
        logger.info(
            f"Áudio capturado: {len(audio_completo)} amostras "
            f"({len(audio_completo) / self.sample_rate:.2f}s)"
        )

        return audio_completo

    def salvar_wav(self, audio: np.ndarray, caminho: str | Path) -> Path:
        """Salva um array de áudio float32 como arquivo WAV.

        Args:
            audio: Array numpy float32 normalizado [-1.0, 1.0].
            caminho: Caminho do arquivo WAV de saída.

        Returns:
            Path do arquivo salvo.
        """
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)

        # Converter de volta para int16
        audio_int16 = (audio * 32767).astype(np.int16)

        with wave.open(str(caminho), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())

        logger.info(f"Áudio salvo em: {caminho}")
        return caminho

    def encerrar(self) -> None:
        """Libera os recursos do PyAudio."""
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
            logger.info("PyAudio encerrado.")
