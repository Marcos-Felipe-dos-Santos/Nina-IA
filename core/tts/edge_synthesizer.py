"""
core.tts.edge_synthesizer
=========================
Sintese de voz com Edge TTS e reproducao dual opcional.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import sounddevice as sd
import soundfile as sf

from core.utils.config_loader import obter_secao

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TEMP_DIR = _PROJECT_ROOT / "data" / "temp"
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


class EdgeTTSSynthesizer:
    """Sintetiza voz com Edge TTS mantendo a interface do Kokoro."""

    def __init__(self) -> None:
        """Inicializa o provider Edge usando o config.yaml."""
        config_tts = obter_secao("tts")

        self.voice: str = config_tts.get("edge_voice", "pt-BR-FranciscaNeural")
        self.rate: str = config_tts.get("edge_rate", "-10%")
        self.pitch: str = config_tts.get("edge_pitch", "+0Hz")
        self.output_path: Path = _TEMP_DIR / "tts_output.mp3"
        self._dispositivo_virtual: Optional[int] = None
        self._virtual_checado = False

        _TEMP_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(
            "EdgeTTSSynthesizer configurado: voice=%s, rate=%s, pitch=%s",
            self.voice,
            self.rate,
            self.pitch,
        )

    def _carregar_pipeline(self) -> None:
        """Mantem compatibilidade com o fluxo de warmup."""
        logger.info("Edge TTS nao exige pre-carregamento local.")

    @staticmethod
    def _pos_processar(texto: str) -> str:
        """Limpa o texto para soar melhor na fala."""
        texto_limpo = _EMOJI_RE.sub("", texto or "")

        abreviacoes = {
            r"\bvc\b": "você",
            r"\bvcs\b": "vocês",
            r"\btb\b": "também",
            r"\btbm\b": "também",
            r"\bpq\b": "porque",
            r"\bqdo\b": "quando",
            r"\bp/\b": "para",
            r"\bc/\b": "com",
            r"\bs/\b": "sem",
            r"\bobs:\b": "observação:",
            r"\bex:\b": "exemplo:",
            r"\betc\.\b": "etcetera",
            r"\bmsg\b": "mensagem",
            r"\bmsgs\b": "mensagens",
        }

        for padrao, substituicao in abreviacoes.items():
            texto_limpo = re.sub(
                padrao,
                substituicao,
                texto_limpo,
                flags=re.IGNORECASE,
            )

        texto_limpo = re.sub(r"([,;:])\s*", r"\1 ... ", texto_limpo)
        texto_limpo = re.sub(r"([.!?])\s*", r"\1 ... ", texto_limpo)
        texto_limpo = re.sub(r"\s+", " ", texto_limpo).strip(" .")

        return texto_limpo.strip()

    def _encontrar_dispositivo_virtual(self) -> Optional[int]:
        """Encontra o VB-Audio Virtual Cable pelo nome do dispositivo."""
        if self._virtual_checado:
            return self._dispositivo_virtual

        self._virtual_checado = True

        try:
            dispositivos: Sequence[dict] = sd.query_devices()
        except Exception as exc:
            logger.warning("Nao foi possivel listar dispositivos de audio: %s", exc)
            return None

        for indice, dispositivo in enumerate(dispositivos):
            nome = str(dispositivo.get("name", ""))
            canais_saida = int(dispositivo.get("max_output_channels", 0))
            if "CABLE Input" in nome and canais_saida > 0:
                self._dispositivo_virtual = indice
                logger.info("VB-Audio detectado no device %s (%s)", indice, nome)
                return indice

        logger.warning(
            "VB-Audio Virtual Cable nao encontrado. "
            "A reproducao seguira apenas no dispositivo principal."
        )
        return None

    async def _gerar_audio_async(self, texto: str) -> float:
        """Gera o arquivo de audio MP3 com Edge TTS."""
        import edge_tts

        texto_limpo = self._pos_processar(texto)
        inicio = time.perf_counter()

        communicate = edge_tts.Communicate(
            text=texto_limpo,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch,
        )
        await communicate.save(str(self.output_path))

        return (time.perf_counter() - inicio) * 1000

    def _carregar_audio(self) -> tuple[np.ndarray, int]:
        """Carrega o MP3 gerado para um ndarray compativel com sounddevice."""
        audio, sample_rate = sf.read(str(self.output_path), dtype="float32", always_2d=False)

        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = audio.mean(axis=1)

        return np.asarray(audio, dtype=np.float32), int(sample_rate)

    def _reproduzir_em_dispositivo(
        self,
        audio: np.ndarray,
        sample_rate: int,
        device: Optional[int] = None,
    ) -> None:
        """Reproduz o audio de forma bloqueante em um unico dispositivo."""
        audio_saida = np.asarray(audio, dtype=np.float32)
        if audio_saida.ndim == 1:
            audio_saida = audio_saida.reshape(-1, 1)

        with sd.OutputStream(
            samplerate=sample_rate,
            channels=audio_saida.shape[1],
            dtype="float32",
            device=device,
        ) as stream:
            stream.write(audio_saida)

    async def _reproduzir_dual(self, audio: np.ndarray, sample_rate: int) -> None:
        """Reproduz no dispositivo principal e no VB-Audio em paralelo."""
        device_virtual = self._encontrar_dispositivo_virtual()

        if device_virtual is None:
            await asyncio.to_thread(self._reproduzir_em_dispositivo, audio, sample_rate, None)
            return

        await asyncio.gather(
            asyncio.to_thread(self._reproduzir_em_dispositivo, audio, sample_rate, None),
            asyncio.to_thread(
                self._reproduzir_em_dispositivo,
                audio,
                sample_rate,
                device_virtual,
            ),
        )

    def _limpar_arquivo_temporario(self) -> None:
        """Remove o arquivo temporario do Edge TTS, se existir."""
        try:
            self.output_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.debug("Nao foi possivel remover o audio temporario: %s", exc)

    async def _sintetizar_e_reproduzir_async(self, texto: str) -> float:
        """Fluxo assincrono completo de sintese e reproducao."""
        latencia_ms = await self._gerar_audio_async(texto)
        audio, sample_rate = self._carregar_audio()
        await self._reproduzir_dual(audio, sample_rate)
        return latencia_ms

    def sintetizar_e_reproduzir(self, texto: str) -> float:
        """Sintetiza o texto e retorna a latencia da etapa de geracao."""
        try:
            return asyncio.run(self._sintetizar_e_reproduzir_async(texto))
        finally:
            self._limpar_arquivo_temporario()
