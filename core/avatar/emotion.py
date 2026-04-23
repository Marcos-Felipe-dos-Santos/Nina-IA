"""
core.avatar.emotion
===================
Detecção de emoções baseada puramente em palavras-chave para alta performance.
Execução em menos de 5ms — nenhuma chamada ao LLM.
"""

import string
from typing import Dict, List


def detect_emotion(text: str, keywords_dict: Dict[str, List[str]]) -> str:
    """Detecta a emoção mais provável no texto por contagem de palavras-chave.

    Case-insensitive, ignora pontuação. Retorna a emoção com mais matches,
    ou "neutro" se nenhuma keyword for detectada.

    Args:
        text: Resposta gerada pela Nina.
        keywords_dict: Mapeamento de emoções para listas de palavras-chave.

    Returns:
        Nome da emoção ('alegria', 'tristeza', etc.) ou 'neutro'.
    """
    if not text or not keywords_dict:
        return "neutro"

    # Preparar texto: lowercase + pontuação → espaços
    text_lower = text.lower()
    for char in string.punctuation:
        text_lower = text_lower.replace(char, " ")

    # Padding para word boundary simples
    text_padded = f" {text_lower} "

    counts: Dict[str, int] = {}
    has_matches = False

    for emotion, keywords in keywords_dict.items():
        count = 0
        for kw in keywords:
            kw_clean = kw.lower()
            for char in string.punctuation:
                kw_clean = kw_clean.replace(char, " ")
            kw_clean = kw_clean.strip()

            if not kw_clean:
                continue

            if f" {kw_clean} " in text_padded:
                count += 1
                has_matches = True

        counts[emotion] = count

    if not has_matches:
        return "neutro"

    return max(counts, key=counts.get)
