"""
core.vision.capture
===================
Captura de tela rápida usando mss.
Suporta captura única, watch contínuo e compressão automática.
"""

import io
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from core.utils.config_loader import obter_secao

logger = logging.getLogger(__name__)


class ScreenCapture:
    """Captura screenshots da tela com compressão automática.

    Usa mss para captura rápida e PIL para redimensionamento
    e compressão JPEG antes de enviar ao modelo de visão.

    Attributes:
        compression_quality: Qualidade JPEG (1-100).
        max_width: Largura máxima da imagem.
        screenshot_dir: Diretório para salvar screenshots.
    """

    def __init__(self) -> None:
        """Inicializa a captura com parâmetros do config.yaml."""
        config_vision = obter_secao("vision")

        self.compression_quality: int = config_vision.get("compression_quality", 60)
        self.max_width: int = config_vision.get("max_width", 1280)
        self.screenshot_dir: str = config_vision.get(
            "screenshot_dir", "./data/screenshots"
        )
        self.auto_interval: int = config_vision.get("auto_interval_seconds", 30)

        # Thread de watch
        self._watch_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ultimo_screenshot: Optional[Image.Image] = None

        # Garantir diretório
        Path(self.screenshot_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"ScreenCapture configurado: quality={self.compression_quality}, "
            f"max_width={self.max_width}"
        )

    def capture(self) -> Tuple[Image.Image, Path]:
        """Captura um screenshot da tela principal.

        Returns:
            Tupla (imagem_pil, caminho_arquivo).
            A imagem já é comprimida e redimensionada.
        """
        import mss

        with mss.mss() as sct:
            # Capturar monitor principal
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)

            # Converter BGRA → RGB
            img = Image.frombytes(
                "RGB", screenshot.size, screenshot.bgra, "raw", "BGRX"
            )

        # Redimensionar se necessário
        img = self._redimensionar(img)

        # Salvar comprimido
        caminho = self._salvar_comprimido(img)

        self._ultimo_screenshot = img

        logger.info(
            f"Screenshot capturado: {img.size[0]}x{img.size[1]}, "
            f"salvo em {caminho}"
        )

        return img, caminho

    def capture_as_bytes(self) -> Tuple[bytes, Image.Image]:
        """Captura screenshot e retorna como bytes JPEG.

        Ideal para enviar diretamente à API do Gemini.

        Returns:
            Tupla (jpeg_bytes, imagem_pil).
        """
        img, _ = self.capture()
        jpeg_bytes = self._imagem_para_bytes(img)
        return jpeg_bytes, img

    def start_watch(self, interval_seconds: Optional[int] = None) -> None:
        """Inicia loop de captura em thread separada.

        Args:
            interval_seconds: Intervalo entre capturas (padrão do config).
        """
        if self._watch_thread and self._watch_thread.is_alive():
            logger.warning("Watch já está ativo.")
            return

        intervalo = interval_seconds or self.auto_interval
        self._stop_event.clear()

        self._watch_thread = threading.Thread(
            target=self._loop_watch,
            args=(intervalo,),
            daemon=True,
            name="ScreenCapture-Watch",
        )
        self._watch_thread.start()

        logger.info(f"Watch iniciado: captura a cada {intervalo}s")
        print(f"👁️  Watch de tela iniciado (a cada {intervalo}s)")

    def stop_watch(self) -> None:
        """Para o loop de captura (retorno imediato)."""
        self._stop_event.set()
        if self._watch_thread:
            self._watch_thread.join(timeout=5)
            self._watch_thread = None
        logger.info("Watch de tela parado.")

    @property
    def ultimo_screenshot(self) -> Optional[Image.Image]:
        """Retorna o último screenshot capturado."""
        return self._ultimo_screenshot

    def _loop_watch(self, intervalo: int) -> None:
        """Loop interno de captura periódica.

        Usa threading.Event.wait() para parada imediata em vez
        de time.sleep() que bloqueia stop_watch().

        Args:
            intervalo: Segundos entre capturas.
        """
        while not self._stop_event.is_set():
            try:
                self.capture()
            except Exception as e:
                logger.error(f"Erro no watch de tela: {e}")

            self._stop_event.wait(timeout=intervalo)

    def _redimensionar(self, img: Image.Image) -> Image.Image:
        """Redimensiona a imagem mantendo a proporção.

        Args:
            img: Imagem PIL original.

        Returns:
            Imagem redimensionada se necessário, ou original.
        """
        if img.width <= self.max_width:
            return img

        # Calcular nova altura proporcionalmente
        proporcao = self.max_width / img.width
        nova_altura = int(img.height * proporcao)

        img_redimensionada = img.resize(
            (self.max_width, nova_altura),
            Image.Resampling.LANCZOS,
        )

        logger.debug(
            f"Imagem redimensionada: {img.size} → {img_redimensionada.size}"
        )

        return img_redimensionada

    def _salvar_comprimido(self, img: Image.Image) -> Path:
        """Salva a imagem como JPEG comprimido.

        Args:
            img: Imagem PIL a salvar.

        Returns:
            Caminho do arquivo salvo.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        caminho = Path(self.screenshot_dir) / f"screen_{timestamp}.jpg"

        img.save(caminho, "JPEG", quality=self.compression_quality)

        tamanho_kb = caminho.stat().st_size / 1024
        logger.debug(f"Screenshot salvo: {caminho} ({tamanho_kb:.1f} KB)")

        return caminho

    def _imagem_para_bytes(self, img: Image.Image) -> bytes:
        """Converte imagem PIL para bytes JPEG.

        Args:
            img: Imagem PIL.

        Returns:
            Bytes da imagem em formato JPEG.
        """
        from core.vision.utils import imagem_para_bytes
        return imagem_para_bytes(img, quality=self.compression_quality)
