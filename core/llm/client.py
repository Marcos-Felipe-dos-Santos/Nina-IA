"""
core.llm.client
===============
Cliente LLM para comunicação com a API do Google Gemini.
Suporta streaming, histórico de conversa, retry automático e function calling.
"""

import asyncio
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.utils.config_loader import obter_secao

logger = logging.getLogger(__name__)


class NinaLLM:
    """Cliente LLM da Nina usando Google Gemini.

    Gerencia a comunicação com a API Gemini, incluindo:
    - Streaming de tokens em tempo real
    - Histórico de conversa com limite configurável
    - System prompt com personalidade da Nina
    - Retry automático em caso de falhas
    - Function calling nativo (tool use)

    Attributes:
        model: Nome do modelo Gemini.
        temperature: Temperatura de geração (criatividade).
        max_tokens: Limite máximo de tokens na resposta.
        max_history: Número máximo de mensagens no histórico.
        max_retries: Número de tentativas em caso de erro.
        streaming: Se deve usar streaming de resposta.
        system_prompt: Instrução de sistema com a personalidade da Nina.
    """

    def __init__(self) -> None:
        """Inicializa o cliente LLM com parâmetros do config.yaml."""
        config_llm = obter_secao("llm")

        self.provider: str = os.environ.get("LLM_PROVIDER", config_llm.get("provider", "gemini")).lower()
        self.gemini_model: str = config_llm.get("gemini_model", config_llm.get("model", "gemini-1.5-flash"))
        self.ollama_model: str = config_llm.get("ollama_model", "llama3")
        self.ollama_url: str = config_llm.get("ollama_url", "http://localhost:11434")

        self.model = self.gemini_model if self.provider == "gemini" else self.ollama_model

        self.temperature: float = config_llm.get("temperature", 0.7)
        self.max_tokens: int = config_llm.get("max_tokens", 1024)
        self.max_history: int = config_llm.get("max_history", 20)
        self.max_retries: int = config_llm.get("max_retries", 3)
        self.streaming: bool = config_llm.get("streaming", True)
        self.system_prompt: str = config_llm.get("system_prompt", "").strip()

        # API key: prioridade para variável de ambiente
        self._api_key: str = (
            os.environ.get("GEMINI_API_KEY", "")
            or config_llm.get("api_key", "")
        )

        # Histórico de conversa: lista de dicts {"role": ..., "content": ...}
        self._historico: List[Dict[str, str]] = []

        # Tools para function calling (lista de funções Python)
        self._tools: Optional[List[Callable]] = None

        # Cliente Gemini será inicializado sob demanda
        self._client = None
        
        if self.provider == "ollama":
            self._verificar_ollama()

        logger.info(
            f"NinaLLM configurado: provider={self.provider}, model={self.model}, "
            f"temperature={self.temperature}, max_history={self.max_history}, "
            f"streaming={self.streaming}"
        )

    def _verificar_ollama(self) -> None:
        """Verifica se o servidor Ollama está rodando. Faz fallback para Gemini se falhar."""
        import urllib.request
        try:
            logger.info(f"Verificando servidor Ollama em {self.ollama_url}...")
            urllib.request.urlopen(f"{self.ollama_url}/api/tags", timeout=2)
            logger.info("Servidor Ollama detectado com sucesso.")
        except Exception as e:
            logger.warning(
                f"⚠️ Servidor Ollama inativo no endpoint {self.ollama_url} ({e}). "
                "Fazendo fallback automático para o Gemini."
            )
            self.provider = "gemini"
            self.model = self.gemini_model

    def configurar_tools(self, tools: List[Callable]) -> None:
        """Configura as ferramentas disponíveis para function calling.

        Args:
            tools: Lista de funções Python com docstrings e type hints.
                   O Gemini SDK gera o schema automaticamente.
        """
        self._tools = tools
        nomes = [f.__name__ for f in tools]
        logger.info(f"Tools configuradas para function calling: {nomes}")

    def _inicializar_client(self) -> Any:
        """Retorna o cliente Google GenAI compartilhado (singleton).

        Returns:
            Cliente genai inicializado.

        Raises:
            ValueError: Se a API key não estiver configurada.
        """
        if self._client is None:
            from core.utils.genai_client import get_genai_client
            self._client = get_genai_client()
            logger.info("NinaLLM usando cliente GenAI compartilhado.")

        return self._client

    def _construir_contents(self, prompt: str) -> List[Dict[str, Any]]:
        """Constrói a lista de conteúdos para enviar à API.

        Combina o histórico de conversa com a nova mensagem.

        Args:
            prompt: Nova mensagem do usuário.

        Returns:
            Lista de dicts no formato esperado pela API Gemini.
        """
        contents = []

        # Adicionar histórico
        for msg in self._historico:
            contents.append({
                "role": msg["role"],
                "parts": [{"text": msg["content"]}],
            })

        # Adicionar nova mensagem do usuário
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}],
        })

        return contents

    def _gerenciar_historico(self, prompt: str, resposta: str) -> None:
        """Adiciona a troca ao histórico e aplica o limite.

        Args:
            prompt: Mensagem do usuário.
            resposta: Resposta do modelo.
        """
        self._historico.append({"role": "user", "content": prompt})
        self._historico.append({"role": "model", "content": resposta})

        # Aplicar limite de histórico (mantém últimas N mensagens)
        if len(self._historico) > self.max_history:
            excesso = len(self._historico) - self.max_history
            self._historico = self._historico[excesso:]
            logger.debug(
                f"Histórico truncado: removidas {excesso} mensagens antigas."
            )

    def _construir_config(self) -> Any:
        """Constrói a configuração de geração incluindo tools se disponíveis.

        Returns:
            GenerateContentConfig com system prompt, temperatura e tools.
        """
        from google.genai import types

        config_params = {
            "system_instruction": self.system_prompt if self.system_prompt else None,
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }

        # Adicionar tools se configuradas
        if self._tools:
            config_params["tools"] = self._tools

        return types.GenerateContentConfig(**config_params)

    async def ask(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Envia uma mensagem ao Gemini e retorna a resposta.

        Suporta streaming (imprime tokens conforme chegam),
        function calling automático e retry automático.

        Args:
            prompt: Mensagem do usuário.
            history: Histórico externo opcional (sobrescreve o interno
                     apenas para esta chamada).

        Returns:
            Texto completo da resposta do modelo.

        Raises:
            RuntimeError: Se todas as tentativas falharem.
        """
        client = self._inicializar_client() if self.provider == "gemini" else None

        # Usar histórico externo se fornecido
        historico_anterior = None
        if history is not None:
            historico_anterior = self._historico.copy()
            self._historico = history.copy()

        contents = self._construir_contents(prompt) if self.provider == "gemini" else None
        config = self._construir_config() if self.provider == "gemini" else None

        ultimo_erro: Optional[Exception] = None

        for tentativa in range(1, self.max_retries + 1):
            try:
                if self.provider == "ollama":
                    resposta = await self._ask_ollama(prompt, tentativa)
                else:
                    # Com tools, desabilitar streaming para permitir
                    # function calling automático do SDK
                    if self._tools:
                        resposta = await self._ask_com_tools(
                            client, contents, config, tentativa
                        )
                    elif self.streaming:
                        resposta = await self._ask_streaming(
                            client, contents, config, tentativa
                        )
                    else:
                        resposta = await self._ask_normal(
                            client, contents, config, tentativa
                        )

                # Gerenciar histórico
                self._gerenciar_historico(prompt, resposta)

                # Restaurar histórico se era externo
                if historico_anterior is not None:
                    self._historico = historico_anterior

                return resposta

            except Exception as e:
                ultimo_erro = e
                logger.warning(
                    f"Tentativa {tentativa}/{self.max_retries} falhou: {e}"
                )
                if tentativa < self.max_retries:
                    espera = 2 ** (tentativa - 1)
                    logger.info(f"Aguardando {espera}s antes de tentar novamente...")
                    await asyncio.sleep(espera)

        # Restaurar histórico se era externo
        if historico_anterior is not None:
            self._historico = historico_anterior

        raise RuntimeError(
            f"Todas as {self.max_retries} tentativas falharam. "
            f"Último erro: {ultimo_erro}"
        )

    async def _ask_com_tools(
        self,
        client: Any,
        contents: List[Dict[str, Any]],
        config: Any,
        tentativa: int,
    ) -> str:
        """Envia requisição com function calling automático.

        O Gemini SDK executa as funções automaticamente quando
        o modelo decide usar uma ferramenta.

        Args:
            client: Cliente GenAI.
            contents: Conteúdos da conversa.
            config: Configuração de geração (com tools).
            tentativa: Número da tentativa atual.

        Returns:
            Texto completo da resposta final.
        """
        logger.info(
            f"Enviando ao Gemini (com tools, tentativa {tentativa})..."
        )

        response = await client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        # Verificar se houve function call
        texto = ""
        if response.candidates:
            candidate = response.candidates[0]
            # Verificar se há partes com texto na resposta
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        texto += part.text
                    elif hasattr(part, "function_call") and part.function_call:
                        func_call = part.function_call
                        logger.info(
                            f"🔧 Function call detectado: "
                            f"{func_call.name}({func_call.args})"
                        )
                        print(
                            f"🔧 Executando: {func_call.name}"
                            f"({dict(func_call.args) if func_call.args else ''})"
                        )

        # Se não houve texto, tentar response.text
        if not texto and response.text:
            texto = response.text

        texto = texto.strip()
        if texto:
            print(f"🤖 Nina: {texto}")

        return texto

    async def _ask_streaming(
        self,
        client: Any,
        contents: List[Dict[str, Any]],
        config: Any,
        tentativa: int,
    ) -> str:
        """Envia requisição com streaming de tokens.

        Args:
            client: Cliente GenAI.
            contents: Conteúdos da conversa.
            config: Configuração de geração.
            tentativa: Número da tentativa atual.

        Returns:
            Texto completo da resposta.
        """
        logger.info(
            f"Enviando ao Gemini (streaming, tentativa {tentativa})..."
        )
        print("🤖 Nina: ", end="", flush=True)

        texto_completo = ""

        async for chunk in await client.aio.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                print(chunk.text, end="", flush=True)
                texto_completo += chunk.text

        print()  # Nova linha após streaming
        return texto_completo.strip()

    async def _ask_normal(
        self,
        client: Any,
        contents: List[Dict[str, Any]],
        config: Any,
        tentativa: int,
    ) -> str:
        """Envia requisição sem streaming.

        Args:
            client: Cliente GenAI.
            contents: Conteúdos da conversa.
            config: Configuração de geração.
            tentativa: Número da tentativa atual.

        Returns:
            Texto completo da resposta.
        """
        logger.info(
            f"Enviando ao Gemini (sem streaming, tentativa {tentativa})..."
        )

        response = await client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        texto = response.text.strip() if response.text else ""
        print(f"🤖 Nina: {texto}")
        return texto

    async def _ask_ollama(self, prompt: str, tentativa: int) -> str:
        """Envia requisição usando o provedor Ollama local.
        
        Args:
            prompt: Mensagem do usuário.
            tentativa: Número da tentativa atual.
            
        Returns:
            Texto completo da resposta final.
        """
        import ollama
        logger.info(f"Enviando ao Ollama ({self.model}, tentativa {tentativa})...")
        
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
            
        for msg in self._historico:
            # model is 'assistant' in ollama
            role = "assistant" if msg["role"] == "model" else msg["role"]
            messages.append({"role": role, "content": msg["content"]})
            
        messages.append({"role": "user", "content": prompt})

        client = ollama.AsyncClient(host=self.ollama_url)
        texto_completo = ""
        
        if self.streaming:
            print("🤖 Nina: ", end="", flush=True)
            async for chunk in await client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": self.temperature, "num_predict": self.max_tokens},
                stream=True,
            ):
                content = chunk.get("message", {}).get("content", "")
                if content:
                    print(content, end="", flush=True)
                    texto_completo += content
            print()
        else:
            response = await client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": self.temperature, "num_predict": self.max_tokens},
                stream=False,
            )
            texto_completo = response.get("message", {}).get("content", "").strip()
            print(f"🤖 Nina: {texto_completo}")

        return texto_completo.strip()

    def ask_sync(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Versão síncrona de `ask()` para uso fora de contextos async.

        Args:
            prompt: Mensagem do usuário.
            history: Histórico externo opcional.

        Returns:
            Texto completo da resposta do modelo.
        """
        return asyncio.run(self.ask(prompt, history))

    @property
    def historico(self) -> List[Dict[str, str]]:
        """Retorna uma cópia do histórico de conversa atual."""
        return self._historico.copy()

    @property
    def tem_tools(self) -> bool:
        """Verifica se há tools configuradas."""
        return self._tools is not None and len(self._tools) > 0

    def limpar_historico(self) -> None:
        """Limpa o histórico de conversa."""
        self._historico.clear()
        logger.info("Histórico de conversa limpo.")

    def obter_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o estado atual do cliente.

        Returns:
            Dicionário com modelo, histórico e configurações.
        """
        return {
            "model": self.model,
            "provider": self.provider,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_history": self.max_history,
            "historico_atual": len(self._historico),
            "streaming": self.streaming,
            "system_prompt_ativo": bool(self.system_prompt),
            "tools_ativas": len(self._tools) if self._tools else 0,
        }
