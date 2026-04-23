"""
core.avatar.vtube
=================
Controlador do VTube Studio via WebSockets (usando pyvts).
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import pyvts

logger = logging.getLogger(__name__)

# Referência global para a instância ativa
_global_vtube_instance: Optional["VTubeController"] = None


def get_global_vtube() -> Optional["VTubeController"]:
    """Retorna a instância global do VTubeController."""
    return _global_vtube_instance


def set_global_vtube(instance: "VTubeController") -> None:
    """Define a instância global."""
    global _global_vtube_instance
    _global_vtube_instance = instance


class VTubeController:
    """Controlador assíncrono para gerenciar expressões do VTube Studio.

    Conecta via WebSocket em ws://localhost:{port}, autentica como plugin
    e dispara hotkeys de expressões faciais. Se o VTube Studio não estiver
    aberto, loga aviso e continua sem quebrar o pipeline.

    Attributes:
        vts: Instância pyvts.vts.
        connected: Se está conectado e autenticado.
        expressions: Mapeamento emoção → nome da hotkey.
        hotkeys_map: Mapeamento nome da hotkey → ID.
    """

    TOKEN_PATH = Path("data") / "vtube_token.json"

    def __init__(self, port: int = 8001, plugin_name: str = "Nina IA") -> None:
        plugin_info = {
            "plugin_name": plugin_name,
            "developer": "Nina",
            "authentication_token_path": str(self.TOKEN_PATH),
        }

        self.TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)

        self.vts = pyvts.vts(plugin_info=plugin_info, port=port)
        self.connected = False
        self.hotkeys_map: Dict[str, str] = {}

        from core.utils.config_loader import carregar_config
        config = carregar_config()
        self.expressions: Dict[str, str] = config.get("avatar", {}).get("expressions", {})

        set_global_vtube(self)

    async def connect(self) -> None:
        """Conecta ao VTube Studio, autenticando com o plugin.

        Fluxo:
        1. Abre WebSocket
        2. Tenta ler token salvo e autenticar
        3. Se falhar, solicita novo token (popup aparece no VTube Studio)
        4. Persiste token para próxima inicialização
        5. Carrega lista de hotkeys disponíveis

        Se o VTube Studio não estiver aberto, loga aviso e continua.
        """
        try:
            await self.vts.connect()

            # Tenta ler token salvo e autenticar
            await self.vts.read_token()

            if self.vts.authentic_token:
                try:
                    await self.vts.request_authenticate()
                except Exception:
                    logger.debug("Token salvo inválido, solicitando novo.")

            # Se ainda não autenticou, solicitar novo token
            if not self.vts.get_authentic_status():
                logger.info("Aprove o plugin Nina IA no popup do VTube Studio")
                print("⚠️  Aprove o plugin Nina IA no popup do VTube Studio")
                await self.vts.request_authenticate_token()
                await self.vts.request_authenticate()

                # Persistir token para não precisar re-aprovar
                try:
                    await self.vts.write_token()
                    logger.info(f"Token VTube salvo em {self.TOKEN_PATH}")
                except Exception as e:
                    logger.warning(f"Não foi possível salvar token: {e}")

            if self.vts.get_authentic_status():
                self.connected = True
                await self._load_hotkeys()
                logger.info(
                    f"VTube Studio conectado! "
                    f"{len(self.hotkeys_map)} hotkeys disponíveis."
                )
            else:
                logger.warning("Não foi possível autenticar com o VTube Studio.")

        except Exception as e:
            logger.warning(
                f"VTube Studio não disponível — avatar desabilitado. "
                f"Erro: {e}"
            )
            self.connected = False

    async def disconnect(self) -> None:
        """Desconecta do VTube Studio de forma limpa.

        Fecha o WebSocket e reseta o estado de conexão.
        Seguro para chamar mesmo se não estiver conectado.
        """
        if not self.connected:
            return

        try:
            await self.vts.close()
            logger.info("Desconectado do VTube Studio.")
        except Exception as e:
            logger.warning(f"Erro ao desconectar do VTube Studio: {e}")
        finally:
            self.connected = False
            self.hotkeys_map.clear()

    async def _load_hotkeys(self) -> None:
        """Carrega a lista de hotkeys disponíveis no modelo atual."""
        try:
            msg = self.vts.vts_request.requestHotKeyList()
            response = await self.vts.request(msg)

            if response and isinstance(response, dict) and "data" in response:
                available = response["data"].get("availableHotkeys", [])
                for hk in available:
                    self.hotkeys_map[hk["name"]] = hk["hotkeyID"]

                logger.info(f"Hotkeys carregadas: {list(self.hotkeys_map.keys())}")
        except Exception as e:
            logger.warning(f"Erro ao buscar hotkeys do VTube: {e}")

    async def trigger_expression(self, emotion: str) -> None:
        """Dispara a hotkey correspondente à emoção no VTube Studio.

        Args:
            emotion: Nome da emoção (alegria, tristeza, surpresa, raiva, neutro).
        """
        if not self.connected:
            return

        hotkey_name = self.expressions.get(emotion)
        if not hotkey_name:
            hotkey_name = self.expressions.get("neutro", "Neutro")

        # Usa o ID se disponível, senão tenta pelo nome
        hotkey_id = self.hotkeys_map.get(hotkey_name, hotkey_name)

        try:
            msg = self.vts.vts_request.requestTriggerHotKey(hotkey_id)
            await self.vts.request(msg)
            logger.debug(f"Expressão '{emotion}' → hotkey '{hotkey_name}' disparada.")
        except Exception as e:
            logger.warning(f"Erro ao disparar expressão '{emotion}': {e}")
