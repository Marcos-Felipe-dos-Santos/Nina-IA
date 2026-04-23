"""
tests/test_fase3.py
===================
Teste isolado da Fase 3 — Memória de longo prazo com RAG.

Execute com:
    python tests/test_fase3.py

Testes realizados:
    1. Salva 3 conversas fictícias no ChromaDB
    2. Busca por query relacionada e verifica relevância
    3. Reinicia o MemoryManager e verifica persistência
    4. Testa pipeline completo com memória injetada
    5. Exibe PASS ou FAIL para cada teste
"""

import os
import shutil
import sys
import traceback

# Adicionar raiz do projeto ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Constantes ─────────────────────────────────────────────
SEPARADOR = "=" * 60
RESULTADO_PASS = "✅ PASS"
RESULTADO_FAIL = "❌ FAIL"

# Diretório temporário para testes (isolado do banco real)
TEST_MEMORY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "memory_test",
)

# Conversas fictícias para teste
CONVERSAS_TESTE = [
    {
        "user": "Meu aniversário é dia 15 de março.",
        "nina": "Anotado! Vou lembrar que seu aniversário é no dia 15 de março.",
    },
    {
        "user": "Eu trabalho como engenheiro de software na empresa TechCorp.",
        "nina": "Entendi! Você é engenheiro de software na TechCorp. Parece um trabalho incrível!",
    },
    {
        "user": "Minha cor favorita é azul e adoro comida japonesa.",
        "nina": "Boa escolha! Azul é uma cor elegante. E comida japonesa é maravilhosa, adoro sushi!",
    },
]


def imprimir_cabecalho() -> None:
    """Imprime o cabeçalho do teste."""
    print(f"\n{SEPARADOR}")
    print("  🧪 TESTE DA FASE 3 — Memória de Longo Prazo (RAG)")
    print(f"{SEPARADOR}\n")


def imprimir_resultado(nome: str, passou: bool, detalhe: str = "") -> None:
    """Imprime o resultado de um teste."""
    status = RESULTADO_PASS if passou else RESULTADO_FAIL
    msg = f"  {status} | {nome}"
    if detalhe:
        msg += f" — {detalhe}"
    print(msg)


def limpar_banco_teste() -> None:
    """Remove o diretório de teste do ChromaDB se existir."""
    if os.path.exists(TEST_MEMORY_DIR):
        shutil.rmtree(TEST_MEMORY_DIR, ignore_errors=True)


def criar_manager_teste():
    """Cria um MemoryManager apontando para o banco de teste.

    Sobrescreve temporariamente o persist_directory para isolar
    os testes do banco de produção.

    Returns:
        MemoryManager configurado para teste.
    """
    from core.memory.manager import MemoryManager

    manager = MemoryManager()
    # Sobrescrever diretório para usar banco de teste isolado
    manager.persist_directory = TEST_MEMORY_DIR
    # Resetar componentes para forçar reinicialização
    manager._client = None
    manager._collection = None
    manager._embedding_fn = None
    return manager


# ─── Teste 1: Salvar 3 conversas ───────────────────────────
def teste_salvar_conversas() -> tuple[bool, str]:
    """Teste 1: Salva 3 conversas fictícias no ChromaDB.

    Returns:
        Tupla (passou, detalhes).
    """
    print("\n💾 Teste 1: Salvando 3 conversas no ChromaDB")

    try:
        manager = criar_manager_teste()

        for i, conv in enumerate(CONVERSAS_TESTE, 1):
            manager.save_conversation(conv["user"], conv["nina"])
            print(f"   ✅ Conversa {i} salva: '{conv['user'][:50]}...'")

        total = manager.count()
        print(f"   📊 Total no banco: {total} memórias")

        passou = total == 3
        return passou, f"{total} memórias salvas"

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, str(e)


# ─── Teste 2: Busca semântica ──────────────────────────────
def teste_busca_semantica() -> tuple[bool, str]:
    """Teste 2: Busca por query relacionada e verifica relevância.

    Returns:
        Tupla (passou, detalhes).
    """
    print("\n🔍 Teste 2: Busca semântica por memórias")

    try:
        manager = criar_manager_teste()

        # Query 1: Buscar sobre aniversário
        query1 = "Quando é o aniversário do usuário?"
        resultados1 = manager.search_memories(query1, n_results=1)

        print(f"   🔎 Query: '{query1}'")
        if resultados1:
            print(f"   📄 Resultado: '{resultados1[0][:100]}'")
            relevante1 = "março" in resultados1[0].lower() or "aniversário" in resultados1[0].lower()
        else:
            print(f"   📭 Nenhum resultado")
            relevante1 = False

        # Query 2: Buscar sobre trabalho
        query2 = "Onde o usuário trabalha?"
        resultados2 = manager.search_memories(query2, n_results=1)

        print(f"\n   🔎 Query: '{query2}'")
        if resultados2:
            print(f"   📄 Resultado: '{resultados2[0][:100]}'")
            relevante2 = "techcorp" in resultados2[0].lower() or "engenheiro" in resultados2[0].lower()
        else:
            print(f"   📭 Nenhum resultado")
            relevante2 = False

        # Query 3: Buscar sobre preferências
        query3 = "Qual é a comida favorita?"
        resultados3 = manager.search_memories(query3, n_results=1)

        print(f"\n   🔎 Query: '{query3}'")
        if resultados3:
            print(f"   📄 Resultado: '{resultados3[0][:100]}'")
            relevante3 = "japonesa" in resultados3[0].lower() or "comida" in resultados3[0].lower()
        else:
            print(f"   📭 Nenhum resultado")
            relevante3 = False

        passou = relevante1 and relevante2 and relevante3
        acertos = sum([relevante1, relevante2, relevante3])
        detalhe = f"{acertos}/3 buscas retornaram resultado relevante"

        return passou, detalhe

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, str(e)


# ─── Teste 3: Persistência entre sessões ───────────────────
def teste_persistencia() -> tuple[bool, str]:
    """Teste 3: Reinicia o MemoryManager e verifica se memórias persistiram.

    Returns:
        Tupla (passou, detalhes).
    """
    print("\n💽 Teste 3: Persistência entre sessões")

    try:
        # Criar NOVO manager (simula reinício da aplicação)
        print("   🔄 Criando novo MemoryManager (simulando reinício)...")
        manager_novo = criar_manager_teste()

        total = manager_novo.count()
        print(f"   📊 Memórias encontradas após reinício: {total}")

        # Verificar se consegue buscar
        resultados = manager_novo.search_memories("aniversário", n_results=1)
        tem_resultado = len(resultados) > 0

        if tem_resultado:
            print(f"   📄 Busca funcional: '{resultados[0][:80]}'")

        passou = total == 3 and tem_resultado
        detalhe = f"{total} memórias persistidas, busca: {'OK' if tem_resultado else 'FALHA'}"

        return passou, detalhe

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, str(e)


# ─── Teste 4: Formatação para prompt ───────────────────────
def teste_formatacao_prompt() -> tuple[bool, str]:
    """Teste 4: Testa formatação de memórias para injeção no prompt do LLM.

    Returns:
        Tupla (passou, detalhes).
    """
    print("\n📝 Teste 4: Formatação de memórias para prompt do LLM")

    try:
        manager = criar_manager_teste()

        # Buscar e formatar
        query = "O que você sabe sobre mim?"
        prompt_formatado = manager.formatar_memorias_para_prompt(query)

        print(f"   🔎 Query: '{query}'")

        if prompt_formatado:
            print(f"   📄 Prompt formatado:")
            # Mostrar o prompt formatado com indentação
            for linha in prompt_formatado.split("\n"):
                print(f"      {linha[:100]}")

            tem_prefixo = prompt_formatado.startswith("[Memórias relevantes]:")
            tem_conteudo = len(prompt_formatado) > 30
            passou = tem_prefixo and tem_conteudo
        else:
            print(f"   📭 Nenhuma memória formatada")
            passou = False

        detalhe = f"Formato correto: {'sim' if passou else 'não'}"
        return passou, detalhe

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, str(e)


# ─── Teste 5: Inspector ────────────────────────────────────
def teste_inspector() -> tuple[bool, str]:
    """Teste 5: Testa list_all_memories e clear_memories.

    Returns:
        Tupla (passou, detalhes).
    """
    print("\n🔎 Teste 5: Memory Inspector (listar e limpar)")

    try:
        # Importar funções do inspector
        from core.memory.inspector import list_all_memories, clear_memories
        from core.memory.manager import MemoryManager

        # Precisamos monkeypatch o manager dentro do inspector
        # Para apontar para o banco de teste
        import core.memory.inspector as inspector_module

        # Salvar original
        original_init = MemoryManager.__init__

        # Monkeypatchar para usar banco de teste
        def patched_init(self):
            original_init(self)
            self.persist_directory = TEST_MEMORY_DIR
            self._client = None
            self._collection = None
            self._embedding_fn = None

        MemoryManager.__init__ = patched_init

        try:
            # Listar
            memorias = list_all_memories()
            total_antes = len(memorias)
            print(f"\n   📊 Memórias listadas: {total_antes}")

            # Limpar
            removidas = clear_memories()
            print(f"   🗑️  Memórias removidas: {removidas}")

            # Verificar se limpou
            manager_verificar = MemoryManager()
            total_depois = manager_verificar.count()
            print(f"   📊 Memórias após limpeza: {total_depois}")

            passou = total_antes == 3 and total_depois == 0
            detalhe = f"Antes: {total_antes}, Removidas: {removidas}, Depois: {total_depois}"

        finally:
            # Restaurar init original
            MemoryManager.__init__ = original_init

        return passou, detalhe

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, str(e)


# ─── Execução principal ────────────────────────────────────
def main() -> None:
    """Executa todos os testes da Fase 3 em sequência."""
    imprimir_cabecalho()

    # Limpar banco de teste antes de começar
    limpar_banco_teste()
    print("   🧹 Banco de teste limpo.\n")

    resultados: dict[str, bool] = {}

    # ── Teste 1: Salvar conversas ──
    passou, detalhe = teste_salvar_conversas()
    resultados["Salvar 3 conversas no ChromaDB"] = passou

    # ── Teste 2: Busca semântica ──
    passou, detalhe = teste_busca_semantica()
    resultados["Busca semântica por similaridade"] = passou

    # ── Teste 3: Persistência ──
    passou, detalhe = teste_persistencia()
    resultados["Persistência entre sessões"] = passou

    # ── Teste 4: Formatação para prompt ──
    passou, detalhe = teste_formatacao_prompt()
    resultados["Formatação para prompt LLM"] = passou

    # ── Teste 5: Inspector ──
    passou, detalhe = teste_inspector()
    resultados["Memory Inspector (listar/limpar)"] = passou

    # Limpar banco de teste ao final
    limpar_banco_teste()
    print("\n   🧹 Banco de teste removido.\n")

    # ── Resumo final ──
    print(f"{SEPARADOR}")
    print("  📊 RESUMO DOS TESTES — FASE 3")
    print(f"{SEPARADOR}")

    for nome, passou in resultados.items():
        imprimir_resultado(nome, passou)

    total = len(resultados)
    aprovados = sum(1 for v in resultados.values() if v)
    print(f"\n  📈 Resultado: {aprovados}/{total} testes aprovados")

    if aprovados == total:
        print(f"\n  🎉 TODOS OS TESTES PASSARAM!")
    else:
        print(f"\n  ⚠️  {total - aprovados} teste(s) falharam.")

    print(f"{SEPARADOR}\n")

    sys.exit(0 if aprovados == total else 1)


if __name__ == "__main__":
    main()
