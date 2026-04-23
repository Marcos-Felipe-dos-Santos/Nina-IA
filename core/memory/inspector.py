"""
core.memory.inspector
=====================
Utilitário para inspecionar e gerenciar as memórias da Nina.
Permite listar todas as memórias e limpar o banco.
"""

import logging
from typing import Any, Dict, List

from core.memory.manager import get_memory_manager

logger = logging.getLogger(__name__)


def list_all_memories() -> List[Dict[str, Any]]:
    """Lista todas as memórias armazenadas no banco.

    Returns:
        Lista de dicts, cada um com 'id', 'document' e 'metadata'.
    """
    manager = get_memory_manager()
    manager._inicializar()

    total = manager.count()

    if total == 0:
        print("📭 Nenhuma memória armazenada.")
        return []

    # Buscar todos os documentos
    resultados = manager._collection.get(
        include=["documents", "metadatas"],
    )

    memorias = []
    ids = resultados.get("ids", [])
    docs = resultados.get("documents", [])
    metas = resultados.get("metadatas", [])

    print(f"\n📚 Total de memórias: {total}")
    print("=" * 60)

    for i, (doc_id, doc, meta) in enumerate(zip(ids, docs, metas), 1):
        memoria = {
            "id": doc_id,
            "document": doc,
            "metadata": meta,
        }
        memorias.append(memoria)

        timestamp = meta.get("timestamp", "N/A")
        print(f"\n  [{i}] ID: {doc_id}")
        print(f"      📅 Data: {timestamp}")
        print(f"      📝 Resumo: {doc[:120]}")

        user_msg = meta.get("user_msg", "")
        if user_msg:
            print(f"      👤 Usuário: {user_msg[:80]}")

        nina_resp = meta.get("nina_response", "")
        if nina_resp:
            print(f"      🤖 Nina: {nina_resp[:80]}")

    print(f"\n{'=' * 60}")
    return memorias


def clear_memories() -> int:
    """Apaga todas as memórias do banco.

    Returns:
        Número de memórias removidas.
    """
    manager = get_memory_manager()
    manager._inicializar()

    total = manager.count()

    if total == 0:
        print("📭 Nenhuma memória para remover.")
        return 0

    # Obter todos os IDs
    resultados = manager._collection.get()
    ids = resultados.get("ids", [])

    # Remover todos
    if ids:
        manager._collection.delete(ids=ids)

    print(f"🗑️  {total} memória(s) removida(s) com sucesso.")
    logger.info(f"Memórias limpas: {total} removidas.")

    return total


def search_interactive() -> None:
    """Modo interativo para buscar memórias.

    Permite ao usuário digitar queries no terminal e
    ver os resultados de busca semântica.
    """
    manager = get_memory_manager()

    total = manager.count()
    print(f"\n🔍 Busca Interativa de Memórias ({total} memórias no banco)")
    print("   Digite 'sair' para encerrar.\n")

    while True:
        query = input("🔎 Buscar: ").strip()

        if query.lower() in ("sair", "exit", "quit"):
            print("👋 Encerrando busca.")
            break

        if not query:
            continue

        resultados = manager.search_memories_detailed(query)

        if not resultados:
            print("   📭 Nenhuma memória encontrada.\n")
            continue

        print(f"   📚 {len(resultados)} resultado(s):\n")
        for i, mem in enumerate(resultados, 1):
            dist = mem.get("distance", 0)
            doc = mem.get("document", "")
            print(f"   [{i}] (dist: {dist:.4f}) {doc[:120]}")

        print()
