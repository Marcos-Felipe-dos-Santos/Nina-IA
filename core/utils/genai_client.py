"""
core.utils.genai_client
========================
Cliente Google GenAI compartilhado.
Inicializado uma única vez e reutilizado por todos os módulos.
"""

import logging
import os
from typing import Any, Optional

from core.utils.config_loader import obter_secao

logger = logging.getLogger(__name__)

_client: Optional[Any] = None


def get_genai_client() -> Any:
    """Retorna o cliente Google GenAI singleton.

    Inicializa na primeira chamada e reutiliza nas seguintes.

    Returns:
        Cliente genai.Client configurado.

    Raises:
        ValueError: Se a API key não estiver configurada.
    """
    global _client

    if _client is not None:
        return _client

    api_key = (
        os.environ.get("GEMINI_API_KEY", "")
        or obter_secao("llm").get("api_key", "")
    )

    if not api_key:
        raise ValueError(
            "API key do Gemini não configurada. "
            "Defina a variável de ambiente GEMINI_API_KEY ou "
            "preencha 'api_key' em config.yaml na seção 'llm'."
        )

    from google import genai

    _client = genai.Client(api_key=api_key)
    logger.info("Cliente Google GenAI singleton inicializado.")

    return _client
