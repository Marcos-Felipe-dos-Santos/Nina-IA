"""
core.memory.manager
===================
Gerenciador de memória de longo prazo com ChromaDB e RAG.
Salva conversas resumidas como embeddings e busca por similaridade semântica.
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.utils.config_loader import obter_secao

logger = logging.getLogger(__name__)

# Singleton: instância global única
_instance: Optional["MemoryManager"] = None


def get_memory_manager() -> "MemoryManager":
    """Retorna a instância singleton do MemoryManager.

    Na primeira chamada, cria a instância. Chamadas seguintes
    retornam a mesma instância.

    Returns:
        Instância global do MemoryManager.
    """
    global _instance
    if _instance is None:
        _instance = MemoryManager()
    return _instance


class MemoryManager:
    """Gerencia a memória persistente da Nina com ChromaDB.

    Salva resumos de conversas como embeddings vetoriais e
    permite busca por similaridade semântica (RAG).

    Attributes:
        persist_directory: Caminho do banco ChromaDB em disco.
        collection_name: Nome da coleção de memórias.
        embedding_model: Nome do modelo sentence-transformers.
        n_results: Número padrão de resultados por busca.
    """

    def __init__(self) -> None:
        """Inicializa o gerenciador com parâmetros do config.yaml."""
        config_memory = obter_secao("memory")

        self.enabled: bool = config_memory.get("enabled", True)
        self.persist_directory: str = config_memory.get(
            "persist_directory", "./data/memory"
        )
        self.collection_name: str = config_memory.get(
            "collection_name", "nina_memory"
        )
        self.embedding_model_name: str = config_memory.get(
            "embedding_model", "paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.n_results: int = config_memory.get("n_results", 3)
        self.summary_prompt: str = config_memory.get("summary_prompt", "").strip()

        # Componentes carregados sob demanda
        self._client = None
        self._collection = None
        self._embedding_fn = None

        if self.enabled:
            logger.info(
                f"MemoryManager configurado: "
                f"collection={self.collection_name}, "
                f"embedding_model={self.embedding_model_name}, "
                f"persist={self.persist_directory}"
            )
        else:
            logger.info("MemoryManager desabilitado via config.")

    def _inicializar(self) -> None:
        """Inicializa ChromaDB e embedding function (lazy loading)."""
        if self._collection is not None:
            return

        import chromadb
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )

        # Garantir que o diretório existe
        persist_path = Path(self.persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Inicializando ChromaDB em '{persist_path}'...")

        # Criar cliente persistente
        self._client = chromadb.PersistentClient(path=str(persist_path))

        # Criar embedding function com sentence-transformers
        logger.info(
            f"Carregando modelo de embeddings: {self.embedding_model_name}..."
        )
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name,
            device="cuda",
        )

        # Obter ou criar a coleção
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
        )

        total = self._collection.count()
        logger.info(
            f"ChromaDB inicializado. "
            f"Coleção '{self.collection_name}': {total} memórias."
        )

    def _inicializar_sem_embeddings(self) -> None:
        """Inicializa ChromaDB sem carregar o modelo de embeddings (leve para dashboard)."""
        if self._collection is not None:
            return

        import chromadb

        persist_path = Path(self.persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_path))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name
        )

    def save_conversation(
        self,
        user_msg: str,
        nina_response: str,
        summary: Optional[str] = None,
    ) -> None:
        """Salva uma conversa resumida na memória de longo prazo.

        Se nenhum resumo for fornecido, cria um resumo automático
        concatenando a pergunta e a resposta.

        Args:
            user_msg: Mensagem do usuário.
            nina_response: Resposta da Nina.
            summary: Resumo personalizado (opcional). Se None,
                     gera um resumo simples automaticamente.
        """
        if not self.enabled:
            return

        self._inicializar()

        # Gerar resumo se não fornecido
        if summary is None:
            summary = self._resumir_conversa(user_msg, nina_response)

        # Gerar ID único
        doc_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Metadados da memória
        metadata = {
            "user_msg": user_msg[:500],  # Limitar tamanho
            "nina_response": nina_response[:500],
            "timestamp": datetime.now().isoformat(),
            "type": "conversation",
        }

        # Salvar no ChromaDB
        self._collection.add(
            documents=[summary],
            metadatas=[metadata],
            ids=[doc_id],
        )

        logger.info(f"Memória salva [{doc_id}]: '{summary[:80]}'")

    def search_memories(
        self,
        query: str,
        n_results: Optional[int] = None,
    ) -> List[str]:
        """Busca memórias relevantes por similaridade semântica.

        Args:
            query: Texto de busca.
            n_results: Número de resultados (padrão do config).

        Returns:
            Lista de strings com as memórias mais relevantes.
        """
        if not self.enabled:
            return []

        self._inicializar()

        n = n_results or self.n_results

        # Verificar se há memórias
        total = self._collection.count()
        if total == 0:
            logger.debug("Nenhuma memória salva ainda.")
            return []

        # Limitar n_results ao total disponível
        n = min(n, total)

        logger.info(f"Buscando memórias para: '{query[:60]}' (top {n})...")

        resultados = self._collection.query(
            query_texts=[query],
            n_results=n,
        )

        # Extrair documentos
        documentos = resultados.get("documents", [[]])[0]

        logger.info(f"Encontradas {len(documentos)} memórias relevantes.")

        return documentos

    def search_memories_detailed(
        self,
        query: str,
        n_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Busca memórias com metadados completos.

        Args:
            query: Texto de busca.
            n_results: Número de resultados.

        Returns:
            Lista de dicts com 'document', 'metadata' e 'distance'.
        """
        if not self.enabled:
            return []

        self._inicializar()

        n = n_results or self.n_results
        total = self._collection.count()

        if total == 0:
            return []

        n = min(n, total)

        resultados = self._collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        memorias = []
        docs = resultados.get("documents", [[]])[0]
        metas = resultados.get("metadatas", [[]])[0]
        dists = resultados.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            memorias.append({
                "document": doc,
                "metadata": meta,
                "distance": dist,
            })

        return memorias

    def formatar_memorias_para_prompt(self, query: str) -> str:
        """Busca memórias e formata para injeção no prompt do LLM.

        Args:
            query: Texto de busca (geralmente a mensagem do usuário).

        Returns:
            String formatada para injeção no prompt, ou string vazia
            se não houver memórias relevantes.
        """
        memorias = self.search_memories(query)

        if not memorias:
            return ""

        memorias_formatadas = " | ".join(memorias)
        return f"[Memórias relevantes]: {memorias_formatadas}"

    def count(self) -> int:
        """Retorna o número total de memórias salvas.

        Returns:
            Quantidade de memórias na coleção.
        """
        if not self.enabled:
            return 0

        self._inicializar_sem_embeddings()
        return self._collection.count()

    def _resumir_conversa(self, user_msg: str, nina_response: str) -> str:
        """Gera um resumo simples da conversa.

        Cria um resumo concatenando a pergunta e resposta de forma
        condensada. Para resumos mais inteligentes, use o LLM.

        Args:
            user_msg: Mensagem do usuário.
            nina_response: Resposta da Nina.

        Returns:
            Resumo em 1-2 frases.
        """
        # Resumo simples: truncar e combinar
        user_resumo = user_msg[:200].strip()
        nina_resumo = nina_response[:200].strip()

        return f"Usuário perguntou: {user_resumo} — Nina respondeu: {nina_resumo}"
