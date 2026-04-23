"""
tests/test_avatar.py
====================
Testes para integração VTube Studio e detecção de emoções.
"""

import asyncio
import pytest
from unittest.mock import patch, MagicMock

from core.avatar.vtube import VTubeController
from core.avatar.emotion import detect_emotion
from core.tools.actions import change_expression
from core.utils.config_loader import carregar_config


@pytest.fixture
def avatar_config():
    """Retorna a configuração do avatar."""
    config = carregar_config()
    return config.get("avatar", {})


@pytest.fixture
def keywords(avatar_config):
    """Retorna o dicionário de keywords de emoção."""
    return avatar_config.get("keywords", {})


# ── Testes de detect_emotion ──────────────────────────────────────────


class TestDetectEmotion:
    """Testes para a função detect_emotion."""

    def test_detect_alegria(self, keywords):
        """Detecta alegria em texto com keywords de alegria."""
        texto = "Que maravilhoso! Eu adorei a notícia haha."
        assert detect_emotion(texto, keywords) == "alegria"

    def test_detect_tristeza(self, keywords):
        """Detecta tristeza em texto com keywords de tristeza."""
        texto = "Sinto muito por isso, que pena."
        assert detect_emotion(texto, keywords) == "tristeza"

    def test_detect_surpresa(self, keywords):
        """Detecta surpresa em texto com keywords de surpresa."""
        texto = "Sério? Não acredito, que surpresa!"
        assert detect_emotion(texto, keywords) == "surpresa"

    def test_detect_raiva(self, keywords):
        """Detecta raiva em texto com keywords de raiva."""
        texto = "Isso é um absurdo, completamente ridículo!"
        assert detect_emotion(texto, keywords) == "raiva"

    def test_detect_neutro_sem_keywords(self, keywords):
        """Retorna neutro quando nenhuma keyword bate."""
        texto = "Esta é uma resposta genérica sem emoção específica."
        assert detect_emotion(texto, keywords) == "neutro"

    def test_detect_neutro_texto_vazio(self, keywords):
        """Retorna neutro para texto vazio."""
        assert detect_emotion("", keywords) == "neutro"

    def test_detect_neutro_keywords_vazio(self):
        """Retorna neutro para dicionário de keywords vazio."""
        assert detect_emotion("Texto qualquer", {}) == "neutro"

    def test_case_insensitive(self, keywords):
        """Detecta emoção independente de maiúsculas/minúsculas."""
        assert detect_emotion("NÃO ACREDITO!!!!", keywords) == "surpresa"

    def test_ignora_pontuacao(self, keywords):
        """Detecta emoção ignorando pontuação excessiva."""
        assert detect_emotion("Adorei!!! Maravilhoso!!!", keywords) == "alegria"

    def test_multiplas_emocoes_retorna_maior(self, keywords):
        """Quando há keywords de múltiplas emoções, retorna a com mais matches."""
        texto = "Adorei, maravilhoso, que bom! Mas sinto muito."
        # 3 keywords de alegria vs 1 de tristeza → alegria vence
        assert detect_emotion(texto, keywords) == "alegria"


# ── Testes do VTubeController ─────────────────────────────────────────


class TestVTubeController:
    """Testes para o VTubeController."""

    def test_init_no_crash(self):
        """VTubeController inicializa sem crashar mesmo sem VTube aberto."""
        controller = VTubeController(port=8001)
        assert controller.connected is False
        assert isinstance(controller.expressions, dict)

    @pytest.mark.asyncio
    async def test_connect_vtube_closed(self):
        """Connect não dá crash quando VTube Studio está fechado."""
        controller = VTubeController(port=8001)
        await controller.connect()

        # Sem VTube aberto, deve ficar desconectado sem exceção
        assert controller.connected is False

    @pytest.mark.asyncio
    async def test_trigger_expression_disconnected(self):
        """trigger_expression silenciosamente não faz nada quando desconectado."""
        controller = VTubeController(port=8001)
        # Não deve disparar exceção
        await controller.trigger_expression("alegria")
        assert controller.connected is False


# ── Testes da tool change_expression ──────────────────────────────────


class TestChangeExpressionTool:
    """Testes para a tool change_expression."""

    def test_avatar_disabled_returns_message(self):
        """Com avatar desabilitado, retorna mensagem informativa."""
        config = carregar_config()
        assert config.get("avatar", {}).get("enabled") is False

        result = change_expression("alegria")
        assert result == "Avatar desabilitado no config.yaml"

    def test_avatar_disabled_all_emotions(self):
        """Todas as emoções válidas retornam mensagem de desabilitado."""
        for emotion in ["alegria", "tristeza", "surpresa", "raiva", "neutro"]:
            result = change_expression(emotion)
            assert result == "Avatar desabilitado no config.yaml"


# ── Teste de integração do pipeline ───────────────────────────────────


class TestPipelineIntegration:
    """Testes de integração do pipeline com avatar desabilitado."""

    def test_pipeline_continues_avatar_disabled(self):
        """Pipeline inicializa e funciona com avatar desabilitado."""
        with patch("core.stt.microphone.MicrophoneCapture"), \
             patch("core.stt.transcriber.WhisperTranscriber"), \
             patch("core.llm.client.NinaLLM") as mock_llm, \
             patch("core.tts.synthesizer.KokoroSynthesizer"):

            mock_llm_instance = MagicMock()
            mock_llm_instance.model = "test"
            mock_llm.return_value = mock_llm_instance

            from core.pipeline import NinaPipeline
            pipeline = NinaPipeline()

            assert pipeline.vtube is None
            assert pipeline.avatar_enabled is False
