"""
core.tools.executor
===================
Executor de ferramentas integrado com o Gemini function calling nativo.
Configura as tools, as expõe ao LLM e executa automaticamente.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from core.tools.registry import ToolRegistry
from core.tools import actions

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Executor de ferramentas da Nina IA.

    Registra todas as ferramentas disponíveis e fornece
    a lista de funções ao Gemini para function calling automático.

    O Gemini SDK executa as funções automaticamente quando o
    modelo decide que precisa de uma ferramenta, mas este executor
    também suporta execução manual.

    Attributes:
        registry: Registro de ferramentas.
    """

    def __init__(self) -> None:
        """Inicializa o executor e registra todas as tools padrão."""
        self.registry = ToolRegistry()
        self._registrar_tools_padrao()

        logger.info(
            f"ToolExecutor inicializado com {self.registry.count()} tools: "
            f"{self.registry.list_names()}"
        )

    def _registrar_tools_padrao(self) -> None:
        """Registra todas as ferramentas padrão da Nina."""

        self.registry.register(
            name="open_app",
            description="Abre um aplicativo no computador pelo nome",
            function=actions.open_app,
            parameters={
                "app_name": {
                    "type": "string",
                    "description": "Nome do aplicativo (ex: 'vscode', 'chrome', 'calculadora')",
                }
            },
        )

        self.registry.register(
            name="web_search",
            description="Busca na internet usando DuckDuckGo e retorna top 3 resultados",
            function=actions.web_search,
            parameters={
                "query": {
                    "type": "string",
                    "description": "Termo de busca",
                }
            },
        )

        self.registry.register(
            name="create_note",
            description="Cria uma nota de texto salva em arquivo no disco",
            function=actions.create_note,
            parameters={
                "title": {
                    "type": "string",
                    "description": "Título da nota",
                },
                "content": {
                    "type": "string",
                    "description": "Conteúdo da nota",
                },
            },
        )

        self.registry.register(
            name="get_time_date",
            description="Retorna a hora e data atual formatada em português",
            function=actions.get_time_date,
            parameters={},
        )

        self.registry.register(
            name="list_notes",
            description="Lista todas as notas salvas no disco",
            function=actions.list_notes,
            parameters={},
        )

        self.registry.register(
            name="look_at_screen",
            description="Captura e descreve o que está visível na tela do computador agora",
            function=actions.look_at_screen,
            parameters={},
        )

        self.registry.register(
            name="change_expression",
            description="Altera a expressão facial da Nina no VTube Studio",
            function=actions.change_expression,
            parameters={
                "emotion": {
                    "type": "string",
                    "description": "Emoção válida: alegria, tristeza, surpresa, raiva, neutro",
                }
            },
        )

    def get_tools_for_gemini(self) -> List[Callable]:
        """Retorna lista de funções Python para o Gemini function calling.

        O Gemini SDK aceita funções Python diretamente no parâmetro
        'tools' da configuração. Ele usa docstrings e type hints
        para gerar o schema automaticamente.

        Returns:
            Lista de callables para passar ao GenerateContentConfig.
        """
        return self.registry.get_functions_for_gemini()

    def execute(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Executa uma ferramenta manualmente pelo nome.

        Args:
            tool_name: Nome da ferramenta registrada.
            params: Dicionário de parâmetros.

        Returns:
            Resultado da execução como string.

        Raises:
            ValueError: Se a ferramenta não estiver registrada.
        """
        tool = self.registry.get(tool_name)

        if tool is None:
            msg = (
                f"Ferramenta '{tool_name}' não encontrada. "
                f"Disponíveis: {self.registry.list_names()}"
            )
            logger.error(msg)
            raise ValueError(msg)

        logger.info(f"Executando tool '{tool_name}' com params: {params}")
        print(f"🔧 Executando: {tool_name}({params})")

        try:
            resultado = tool.function(**params)
            logger.info(f"Tool '{tool_name}' executada com sucesso.")
            return str(resultado)

        except Exception as e:
            msg = f"Erro ao executar '{tool_name}': {e}"
            logger.error(msg)
            return msg

    def list_tools_summary(self) -> str:
        """Retorna um resumo formatado de todas as tools disponíveis.

        Returns:
            Texto com nome e descrição de cada ferramenta.
        """
        tools = self.registry.list_tools()

        if not tools:
            return "Nenhuma ferramenta registrada."

        linhas = [f"🔧 Ferramentas disponíveis ({len(tools)}):"]
        for tool in tools:
            linhas.append(f"  • {tool.name}: {tool.description}")

        return "\n".join(linhas)
