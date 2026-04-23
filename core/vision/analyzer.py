"""
core.vision.analyzer
====================
Análise de imagem com Gemini Vision (multimodal).
Descreve a tela e responde perguntas sobre o conteúdo visual.
"""

import io
import logging
import os
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from core.utils.config_loader import obter_secao

logger = logging.getLogger(__name__)


class VisionAnalyzer:
    """Analisa imagens usando o Gemini Vision (multimodal).

    Integra com a API do Gemini para descrever screenshots
    e responder perguntas sobre o conteúdo visual da tela.

    Attributes:
        model: Modelo Gemini para visão.
        enabled: Se o módulo de visão está habilitado.
    """

    def __init__(self) -> None:
        """Inicializa o analisador com parâmetros do config.yaml."""
        config_vision = obter_secao("vision")

        self.enabled: bool = config_vision.get("enabled", True)
        self.model: str = config_vision.get("model", "gemini-2.0-flash")
        self.compression_quality: int = config_vision.get("compression_quality", 60)

        # API key do Gemini (compartilhada com o módulo LLM)
        self._api_key: str = (
            os.environ.get("GEMINI_API_KEY", "")
            or obter_secao("llm").get("api_key", "")
        )

        # Cliente será inicializado sob demanda
        self._client = None

        logger.info(
            f"VisionAnalyzer configurado: model={self.model}, "
            f"enabled={self.enabled}"
        )

    def _inicializar_client(self) -> Any:
        """Retorna o cliente Google GenAI compartilhado (singleton).

        Returns:
            Cliente genai inicializado.
        """
        if self._client is None:
            from core.utils.genai_client import get_genai_client
            self._client = get_genai_client()
            logger.info("VisionAnalyzer usando cliente GenAI compartilhado.")

        return self._client

    def _imagem_para_bytes(self, image: Image.Image) -> bytes:
        """Converte imagem PIL para bytes JPEG.

        Args:
            image: Imagem PIL.

        Returns:
            Bytes JPEG da imagem.
        """
        from core.vision.utils import imagem_para_bytes
        return imagem_para_bytes(image, quality=self.compression_quality)

    def describe_screen(self, image: Image.Image) -> str:
        """Descreve o conteúdo da tela em 1-2 frases.

        Args:
            image: Imagem PIL do screenshot.

        Returns:
            Descrição curta do que está na tela.
        """
        if not self.enabled:
            return "Módulo de visão desabilitado."

        try:
            client = self._inicializar_client()

            from google.genai import types

            image_bytes = self._imagem_para_bytes(image)

            logger.info("Descrevendo tela com Gemini Vision...")

            response = client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg",
                    ),
                    "Descreva brevemente o que você vê nesta captura de tela "
                    "em 1-2 frases em português do Brasil. "
                    "Seja direto e mencione o aplicativo ou conteúdo principal visível.",
                ],
            )

            descricao = response.text.strip() if response.text else "Não foi possível descrever a tela."

            logger.info(f"Descrição da tela: '{descricao[:80]}'")
            return descricao
        except ValueError as e:
            logger.warning(f"Vision API Key error: {e}")
            return "Visão desabilitada (chave API ausente)."
        except Exception as e:
            logger.error(f"Erro ao processar imagem no Gemini Vision: {e}")
            return "Erro ao analisar captura de tela via Gemini."

    def analyze_for_context(
        self, image: Image.Image, user_question: str
    ) -> str:
        """Responde uma pergunta específica sobre o conteúdo da tela.

        Args:
            image: Imagem PIL do screenshot.
            user_question: Pergunta do usuário sobre a tela.

        Returns:
            Resposta à pergunta baseada no conteúdo visual.
        """
        if not self.enabled:
            return "Módulo de visão desabilitado."

        try:
            client = self._inicializar_client()

            from google.genai import types

            image_bytes = self._imagem_para_bytes(image)

            logger.info(f"Analisando tela para: '{user_question[:60]}'")

            response = client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg",
                    ),
                    f"Olhe esta captura de tela e responda em português do Brasil, "
                    f"de forma concisa (máximo 2-3 frases): {user_question}",
                ],
            )

            resposta = response.text.strip() if response.text else "Não foi possível analisar a tela."

            logger.info(f"Análise contextual: '{resposta[:80]}'")
            return resposta
        except ValueError as e:
            logger.warning(f"Vision API Key error: {e}")
            return "Visão desabilitada (chave API ausente)."
        except Exception as e:
            logger.error(f"Erro ao processar imagem no Gemini Vision: {e}")
            return "Erro ao analisar captura de tela via Gemini."

    def describe_screen_from_path(self, image_path: str) -> str:
        """Descreve a tela a partir de um arquivo de imagem.

        Args:
            image_path: Caminho do arquivo de screenshot.

        Returns:
            Descrição do conteúdo da tela.
        """
        caminho = Path(image_path)
        if not caminho.exists():
            return f"Arquivo não encontrado: {image_path}"

        img = Image.open(caminho)
        return self.describe_screen(img)
