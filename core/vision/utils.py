"""
core.vision.utils
=================
Utilitários compartilhados para o módulo de visão.
"""

import io

from PIL import Image


def imagem_para_bytes(img: Image.Image, quality: int = 60) -> bytes:
    """Converte imagem PIL para bytes JPEG.

    Args:
        img: Imagem PIL.
        quality: Qualidade JPEG (1-100).

    Returns:
        Bytes da imagem em formato JPEG.
    """
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()
