"""
tests.test_tts_edge
===================
Testes do provider Edge TTS e do truncamento para voz.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.pipeline import _truncar_para_voz
from core.tts.edge_synthesizer import EdgeTTSSynthesizer


class _CommunicateFake:
    """Stub simples para simular o edge_tts.Communicate."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    async def save(self, caminho: str) -> None:
        Path(caminho).write_bytes(b"fake-mp3")


def test_sintese_frase_curta_cria_arquivo_temporario() -> None:
    """Uma frase curta deve gerar o arquivo MP3 temporario."""
    sintetizador = EdgeTTSSynthesizer()

    with patch("edge_tts.Communicate", _CommunicateFake):
        latencia = asyncio.run(sintetizador._gerar_audio_async("Bom dia, Nina."))

    assert latencia >= 0
    assert sintetizador.output_path.exists()
    sintetizador._limpar_arquivo_temporario()


def test_latencia_de_sintese_fica_abaixo_de_3000ms() -> None:
    """A medicao da sintese mockada deve respeitar o limite pedido."""
    sintetizador = EdgeTTSSynthesizer()

    with patch("edge_tts.Communicate", _CommunicateFake):
        latencia = asyncio.run(sintetizador._gerar_audio_async("Resposta curta."))

    assert latencia < 3000
    sintetizador._limpar_arquivo_temporario()


def test_truncamento_corta_na_ultima_pontuacao() -> None:
    """Textos longos devem ser truncados na ultima pontuacao natural."""
    texto = (
        "Primeira frase curta. Segunda frase ainda aceitavel! "
        "Terceira frase bem longa para ultrapassar o limite e forcar o corte final."
    )

    truncado = _truncar_para_voz(texto, limite=55)

    assert truncado == "Primeira frase curta. Segunda frase ainda aceitavel!"


def test_pos_processamento_remove_emoji_e_normaliza_abreviacoes() -> None:
    """O texto enviado ao Edge deve sair mais natural para voz."""
    processado = EdgeTTSSynthesizer._pos_processar("Oi 😊 vc pode ver isso, pq tb é urgente!")

    assert "😊" not in processado
    assert "você" in processado
    assert "porque" in processado
    assert "também" in processado


def test_detecta_dispositivo_vb_audio() -> None:
    """Deve localizar automaticamente o CABLE Input quando existir."""
    sintetizador = EdgeTTSSynthesizer()
    dispositivos = [
        {"name": "Alto-falantes", "max_output_channels": 2},
        {"name": "CABLE Input (VB-Audio Virtual Cable)", "max_output_channels": 2},
    ]

    with patch("sounddevice.query_devices", return_value=dispositivos):
        indice = sintetizador._encontrar_dispositivo_virtual()

    assert indice == 1


def test_fallback_para_kokoro_se_edge_falhar() -> None:
    """O pipeline deve trocar para Kokoro se o Edge falhar em tempo de execucao."""
    from core.pipeline import NinaPipeline

    pipeline = object.__new__(NinaPipeline)
    pipeline.sintetizador = MagicMock()
    pipeline.sintetizador.sintetizar_e_reproduzir.side_effect = RuntimeError("edge falhou")
    pipeline.tts_provider = "edge"

    kokoro_mock = MagicMock()
    kokoro_mock.sintetizar_e_reproduzir.return_value = 321.0

    with patch("core.pipeline.KokoroSynthesizer", return_value=kokoro_mock):
        latencia = pipeline._executar_tts_com_fallback("Teste de fallback.")

    assert latencia == 321.0
    assert pipeline.tts_provider == "kokoro"
    assert pipeline.sintetizador is kokoro_mock
