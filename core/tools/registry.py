"""
core.tools.registry
====================
Registro de ferramentas (tools) disponíveis para a Nina IA.
Cada tool é uma função Python com metadados que o LLM pode chamar.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolInfo:
    """Informações de uma ferramenta registrada.

    Attributes:
        name: Nome único da ferramenta.
        description: Descrição do que a ferramenta faz.
        function: Função Python que implementa a ferramenta.
        parameters: Schema JSON dos parâmetros.
    """

    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters: Dict[str, Any],
    ) -> None:
        self.name = name
        self.description = description
        self.function = function
        self.parameters = parameters

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário para exibição.

        Returns:
            Dicionário com nome, descrição e parâmetros.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Registro centralizado de ferramentas da Nina IA.

    Gerencia todas as ferramentas disponíveis, permitindo
    registro, busca e listagem.

    Uso:
        registry = ToolRegistry()
        registry.register("get_time", "Retorna hora atual", func, {})
        tool = registry.get("get_time")
        resultado = tool.function()
    """

    def __init__(self) -> None:
        """Inicializa o registro vazio."""
        self._tools: Dict[str, ToolInfo] = {}
        logger.info("ToolRegistry inicializado.")

    def register(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Registra uma nova ferramenta.

        Args:
            name: Nome único da ferramenta.
            description: Descrição para o LLM entender quando usar.
            function: Função Python que implementa a ferramenta.
            parameters: Schema JSON dos parâmetros (opcional).
        """
        if name in self._tools:
            logger.warning(f"Tool '{name}' já registrada. Sobrescrevendo.")

        self._tools[name] = ToolInfo(
            name=name,
            description=description,
            function=function,
            parameters=parameters or {},
        )
        logger.info(f"Tool registrada: '{name}' — {description[:60]}")

    def get(self, name: str) -> Optional[ToolInfo]:
        """Retorna uma ferramenta pelo nome.

        Args:
            name: Nome da ferramenta.

        Returns:
            ToolInfo ou None se não encontrada.
        """
        return self._tools.get(name)

    def list_tools(self) -> List[ToolInfo]:
        """Lista todas as ferramentas registradas.

        Returns:
            Lista de ToolInfo.
        """
        return list(self._tools.values())

    def list_names(self) -> List[str]:
        """Lista os nomes de todas as ferramentas registradas.

        Returns:
            Lista de nomes.
        """
        return list(self._tools.keys())

    def get_functions_for_gemini(self) -> List[Callable]:
        """Retorna a lista de funções Python para passar ao Gemini.

        O Gemini SDK aceita funções Python diretamente como tools,
        usando docstrings e type hints para gerar o schema.

        Returns:
            Lista de funções callable.
        """
        return [tool.function for tool in self._tools.values()]

    def count(self) -> int:
        """Retorna o número de ferramentas registradas."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Verifica se uma ferramenta está registrada."""
        return name in self._tools

    def __repr__(self) -> str:
        nomes = ", ".join(self._tools.keys())
        return f"ToolRegistry({self.count()} tools: {nomes})"
