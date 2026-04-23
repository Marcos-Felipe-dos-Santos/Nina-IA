"""
tests/test_optimizations.py
============================
Testes para as otimizações da Fase 2-4:
- Config cache
- Singleton MemoryManager
- GenAI client compartilhado
- Emotion detection performance
- VTube controller (token persistence, disconnect)
- EventBus (queue limitada, cleanup)
- Pipeline error recovery
- Vision utils
- Warmup sequence
"""

import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

import pytest

from core.utils.config_loader import carregar_config, obter_secao
from core.avatar.emotion import detect_emotion


# ── Config Cache ──────────────────────────────────────────────


class TestConfigCache:
    """Testes para o cache do config.yaml."""

    def test_cache_retorna_mesmo_objeto(self):
        """Chamadas consecutivas retornam o mesmo dicionário (referência)."""
        config1 = carregar_config()
        config2 = carregar_config()
        assert config1 is config2

    def test_obter_secao_usa_cache(self):
        """obter_secao() reutiliza o cache de carregar_config()."""
        secao = obter_secao("stt")
        assert isinstance(secao, dict)
        assert "model_name" in secao

    def test_config_contém_todas_secoes(self):
        """Config carregado contém todas as seções esperadas."""
        config = carregar_config()
        secoes_esperadas = {"stt", "vad", "tts", "llm", "memory", "vision", "avatar", "general"}
        assert secoes_esperadas.issubset(set(config.keys()))

    def test_invalidar_cache_e_recarregar(self):
        """Forçar invalidação do cache permite recarregar do disco."""
        import core.utils.config_loader as loader
        old_cache = loader._cached_config
        loader._cached_config = None

        config = carregar_config()
        assert config is not None

        # Restaurar
        loader._cached_config = old_cache


# ── Singleton MemoryManager ───────────────────────────────────


class TestMemoryManagerSingleton:
    """Testes para o padrão Singleton do MemoryManager."""

    def test_singleton_retorna_mesma_instância(self):
        """get_memory_manager() retorna a mesma instância sempre."""
        from core.memory.manager import get_memory_manager

        m1 = get_memory_manager()
        m2 = get_memory_manager()
        assert m1 is m2

    def test_singleton_tem_atributos_esperados(self):
        """Instância singleton tem todos os atributos básicos."""
        from core.memory.manager import get_memory_manager

        manager = get_memory_manager()
        assert hasattr(manager, "enabled")
        assert hasattr(manager, "persist_directory")
        assert hasattr(manager, "collection_name")


# ── GenAI Client Compartilhado ────────────────────────────────


class TestGenAIClient:
    """Testes para o cliente GenAI singleton."""

    def test_singleton_sem_api_key_raise(self):
        """get_genai_client() levanta ValueError sem API key."""
        import core.utils.genai_client as gc

        old_client = gc._client
        gc._client = None

        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}, clear=False):
            with patch("core.utils.config_loader.obter_secao", return_value={}):
                with pytest.raises(ValueError, match="API key"):
                    gc.get_genai_client()

        gc._client = old_client

    def test_singleton_com_api_key_mocada(self):
        """get_genai_client() cria cliente quando API key existe."""
        import core.utils.genai_client as gc

        old_client = gc._client
        gc._client = None

        mock_client = MagicMock()
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=False):
            with patch("google.genai.Client", return_value=mock_client):
                client = gc.get_genai_client()
                assert client is mock_client

                # Segunda chamada retorna mesmo
                client2 = gc.get_genai_client()
                assert client2 is mock_client

        gc._client = old_client


# ── Emotion Detection Performance ────────────────────────────


class TestEmotionPerformance:
    """Testes de performance para detecção de emoção."""

    @pytest.fixture
    def keywords(self):
        """Keywords de emoção do config."""
        config = carregar_config()
        return config.get("avatar", {}).get("keywords", {})

    def test_performance_sob_5ms(self, keywords):
        """detect_emotion() executa em menos de 5ms."""
        texto = "Que maravilhoso! Eu adorei isso, haha. É realmente incrível!"

        start = time.perf_counter()
        for _ in range(100):
            detect_emotion(texto, keywords)
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100

        assert elapsed_ms < 5.0, f"detect_emotion levou {elapsed_ms:.2f}ms (limite: 5ms)"

    def test_performance_texto_longo(self, keywords):
        """Performance se mantém com texto longo (500+ chars)."""
        texto = " ".join(["Essa é uma resposta longa com várias palavras"] * 50)
        texto += " adorei maravilhoso"

        start = time.perf_counter()
        for _ in range(100):
            detect_emotion(texto, keywords)
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100

        assert elapsed_ms < 10.0, f"detect_emotion (texto longo) levou {elapsed_ms:.2f}ms"

    def test_empate_retorna_primeiro_max(self, keywords):
        """Com empate de keywords, retorna consistentemente."""
        texto = "sinto muito, que surpresa"
        resultado = detect_emotion(texto, keywords)
        assert resultado in ("tristeza", "surpresa")


# ── VTube Controller Polish ───────────────────────────────────


class TestVTubeControllerPolish:
    """Testes para as melhorias do VTubeController."""

    def test_init_sets_global(self):
        """__init__ registra a instância global."""
        from core.avatar.vtube import get_global_vtube, VTubeController

        controller = VTubeController(port=8001)
        assert get_global_vtube() is controller

    def test_expressions_loaded_from_config(self):
        """Expressões são carregadas do config.yaml."""
        from core.avatar.vtube import VTubeController

        controller = VTubeController(port=8001)
        assert "alegria" in controller.expressions
        assert controller.expressions["alegria"] == "Sorrir"

    @pytest.mark.asyncio
    async def test_disconnect_not_connected(self):
        """disconnect() é seguro quando não conectado."""
        from core.avatar.vtube import VTubeController

        controller = VTubeController(port=8001)
        assert controller.connected is False
        await controller.disconnect()  # não deve crashar
        assert controller.connected is False

    @pytest.mark.asyncio
    async def test_disconnect_clears_state(self):
        """disconnect() reseta connected e hotkeys_map."""
        from core.avatar.vtube import VTubeController

        controller = VTubeController(port=8001)
        controller.connected = True
        controller.hotkeys_map = {"Sorrir": "abc123"}

        with patch.object(controller.vts, "close", new_callable=AsyncMock):
            await controller.disconnect()

        assert controller.connected is False
        assert controller.hotkeys_map == {}

    @pytest.mark.asyncio
    async def test_trigger_when_disconnected_noop(self):
        """trigger_expression é no-op quando desconectado."""
        from core.avatar.vtube import VTubeController

        controller = VTubeController(port=8001)
        # Não deve logar warning nem crashar
        await controller.trigger_expression("alegria")

    @pytest.mark.asyncio
    async def test_connect_writes_token_on_new_auth(self):
        """connect() chama write_token após nova autenticação."""
        from core.avatar.vtube import VTubeController

        controller = VTubeController(port=8001)

        mock_vts = controller.vts
        mock_vts.connect = AsyncMock()
        mock_vts.read_token = AsyncMock()
        mock_vts.authentic_token = None
        mock_vts.get_authentic_status = MagicMock(side_effect=[False, True])
        mock_vts.request_authenticate_token = AsyncMock()
        mock_vts.request_authenticate = AsyncMock()
        mock_vts.write_token = AsyncMock()

        mock_req = MagicMock()
        mock_req.requestHotKeyList = MagicMock(return_value={})
        mock_vts.vts_request = mock_req
        mock_vts.request = AsyncMock(return_value={"data": {"availableHotkeys": []}})

        await controller.connect()

        assert controller.connected is True
        mock_vts.write_token.assert_called_once()


# ── EventBus Queue e Cleanup ──────────────────────────────────


class TestEventBusOptimizations:
    """Testes para as otimizações do EventBus."""

    def test_subscribe_creates_bounded_queue(self):
        """subscribe() cria Queue com maxsize=100."""
        from dashboard.events import event_bus

        async def _test():
            queue = event_bus.subscribe()
            assert queue.maxsize == 100

        asyncio.run(_test())

    def test_emit_event_removes_dead_loops(self):
        """_emit_event remove subscribers com loops fechados."""
        from dashboard.events import event_bus

        # Criar um loop e fechá-lo para simular subscriber orfão
        dead_loop = asyncio.new_event_loop()
        dead_queue = asyncio.Queue(maxsize=100)
        event_bus._subscribers.append((dead_queue, dead_loop))
        dead_loop.close()

        initial_count = len(event_bus._subscribers)
        event_bus._emit_event({"type": "test", "data": {}})

        # Subscriber orfão deve ter sido removido
        assert len(event_bus._subscribers) < initial_count

    def test_queue_full_drops_event(self):
        """Quando a queue está cheia, evento é descartado sem exceção."""
        from dashboard.events import event_bus

        async def _test():
            queue = event_bus.subscribe()

            # Encher a queue
            for i in range(100):
                await queue.put({"type": "fill", "data": {"i": i}})

            # Emitir mais um — não deve crashar
            event_bus._emit_event({"type": "overflow", "data": {}})

            # Queue ainda tem 100 items (overflow foi descartado)
            assert queue.qsize() == 100

            # Limpar subscriber de teste
            event_bus._subscribers = [
                s for s in event_bus._subscribers if s[0] is not queue
            ]

        asyncio.run(_test())


# ── Vision Utils ──────────────────────────────────────────────


class TestVisionUtils:
    """Testes para utilitários de visão."""

    def test_imagem_para_bytes(self):
        """imagem_para_bytes converte PIL Image para JPEG bytes."""
        from PIL import Image
        from core.vision.utils import imagem_para_bytes

        img = Image.new("RGB", (100, 100), color="red")
        result = imagem_para_bytes(img, quality=50)

        assert isinstance(result, bytes)
        assert len(result) > 0
        # JPEG magic bytes
        assert result[:2] == b'\xff\xd8'

    def test_imagem_para_bytes_quality(self):
        """Qualidade menor produz arquivo menor."""
        from PIL import Image
        from core.vision.utils import imagem_para_bytes

        img = Image.new("RGB", (200, 200), color="blue")

        bytes_low = imagem_para_bytes(img, quality=10)
        bytes_high = imagem_para_bytes(img, quality=95)

        assert len(bytes_low) < len(bytes_high)


# ── Pipeline Error Recovery ───────────────────────────────────


class TestPipelineErrorRecovery:
    """Testes para o recovery do pipeline em caso de erro."""

    def test_pipeline_init_with_singleton(self):
        """Pipeline usa singleton get_memory_manager."""
        from core.memory.manager import get_memory_manager

        with patch("core.stt.microphone.MicrophoneCapture"), \
             patch("core.stt.transcriber.WhisperTranscriber"), \
             patch("core.llm.client.NinaLLM") as mock_llm, \
             patch("core.tts.synthesizer.KokoroSynthesizer"):

            mock_llm_instance = MagicMock()
            mock_llm_instance.model = "test"
            mock_llm.return_value = mock_llm_instance

            from core.pipeline import NinaPipeline
            pipeline = NinaPipeline()

            # Deve ser a mesma instância singleton
            assert pipeline.memoria is get_memory_manager()


# ── Capture Watch com threading.Event ─────────────────────────


class TestCaptureWatch:
    """Testes para o watch de tela com threading.Event."""

    def test_stop_event_exists(self):
        """ScreenCapture usa threading.Event em vez de bool."""
        import threading
        from core.vision.capture import ScreenCapture

        capture = ScreenCapture()

        assert hasattr(capture, "_stop_event")
        assert isinstance(capture._stop_event, threading.Event)

    def test_stop_watch_immediate(self):
        """stop_watch() seta o Event para parada imediata."""
        from core.vision.capture import ScreenCapture

        capture = ScreenCapture()

        capture._stop_event.clear()
        capture.stop_watch()
        assert capture._stop_event.is_set()
