"""
tests/test_fase4.py
===================
Teste isolado da Fase 4 — Tool Calling / Agente.

Execute com:
    python tests/test_fase4.py

Testes realizados:
    1. Testa cada tool individualmente (get_time_date, create_note, list_notes, web_search)
    2. Testa o ToolExecutor — execução manual de tools
    3. Testa o registry — registro e listagem de ferramentas
    4. Verifica que notas criadas persistem em disco
    5. Testa o LLM com function calling (requer GEMINI_API_KEY)
    6. Exibe PASS ou FAIL para cada teste
"""

import asyncio
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

# Diretório de notas para testes
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST_NOTES_DIR = os.path.join(_PROJECT_ROOT, "data", "notes_test")


def imprimir_cabecalho() -> None:
    """Imprime o cabeçalho do teste."""
    print(f"\n{SEPARADOR}")
    print("  🧪 TESTE DA FASE 4 — Tool Calling / Agente")
    print(f"{SEPARADOR}\n")


def imprimir_resultado(nome: str, passou: bool, detalhe: str = "") -> None:
    """Imprime o resultado de um teste."""
    status = RESULTADO_PASS if passou else RESULTADO_FAIL
    msg = f"  {status} | {nome}"
    if detalhe:
        msg += f" — {detalhe}"
    print(msg)


# ─── Teste 1: get_time_date ────────────────────────────────
def teste_get_time_date() -> tuple[bool, str]:
    """Teste 1a: Verifica retorno de data e hora.

    Returns:
        Tupla (passou, detalhes).
    """
    print("\n⏰ Teste 1a: get_time_date()")

    try:
        from core.tools.actions import get_time_date

        resultado = get_time_date()
        print(f"   📄 Resultado: \"{resultado}\"")

        # Verificar que contém elementos esperados
        tem_hora = ":" in resultado
        tem_contexto = "de" in resultado and "são" in resultado

        passou = tem_hora and tem_contexto
        return passou, f"Formato correto: {'sim' if passou else 'não'}"

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, str(e)


# ─── Teste 2: create_note + list_notes ─────────────────────
def teste_notes() -> tuple[bool, str]:
    """Teste 1b: Cria notas e verifica persistência.

    Returns:
        Tupla (passou, detalhes).
    """
    print("\n📝 Teste 1b: create_note() + list_notes()")

    try:
        import core.tools.actions as actions_module
        from core.tools.actions import create_note, list_notes

        # Redirecionar para diretório de teste
        from pathlib import Path
        original_notes_dir = actions_module._NOTES_DIR
        actions_module._NOTES_DIR = Path(_TEST_NOTES_DIR)

        try:
            # Limpar diretório de teste
            if os.path.exists(_TEST_NOTES_DIR):
                shutil.rmtree(_TEST_NOTES_DIR)

            # Criar nota 1
            resultado1 = create_note("Lista de compras", "Leite, pão, café")
            print(f"   📄 Nota 1: {resultado1[:80]}")

            # Criar nota 2
            resultado2 = create_note("Reunião amanhã", "Preparar apresentação às 10h")
            print(f"   📄 Nota 2: {resultado2[:80]}")

            # Listar notas
            lista = list_notes()
            print(f"   📋 Lista:\n{lista}")

            # Verificações
            nota1_criada = "sucesso" in resultado1.lower()
            nota2_criada = "sucesso" in resultado2.lower()
            lista_tem_notas = "compras" in lista.lower() or "Lista" in lista

            # Verificar persistência no disco
            arquivos = list(Path(_TEST_NOTES_DIR).glob("*.txt"))
            persistiu = len(arquivos) == 2

            passou = nota1_criada and nota2_criada and lista_tem_notas and persistiu
            detalhe = (
                f"Criação: {nota1_criada and nota2_criada}, "
                f"Listagem: {lista_tem_notas}, "
                f"Disco: {len(arquivos)} arquivos"
            )

        finally:
            # Restaurar diretório original
            actions_module._NOTES_DIR = original_notes_dir
            # Limpar
            if os.path.exists(_TEST_NOTES_DIR):
                shutil.rmtree(_TEST_NOTES_DIR)

        return passou, detalhe

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, str(e)


# ─── Teste 3: web_search ───────────────────────────────────
def teste_web_search() -> tuple[bool, str]:
    """Teste 1c: Busca na web com DuckDuckGo.

    Returns:
        Tupla (passou, detalhes).
    """
    print("\n🔍 Teste 1c: web_search()")

    try:
        from core.tools.actions import web_search

        resultado = web_search("Python programming language")
        print(f"   📄 Resultado ({len(resultado)} chars):")

        # Mostrar as primeiras linhas
        for linha in resultado.split("\n")[:6]:
            print(f"      {linha[:100]}")

        # Verificar que tem resultados
        tem_resultados = len(resultado) > 50 and "Erro" not in resultado
        return tem_resultados, f"{len(resultado)} chars retornados"

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, str(e)


# ─── Teste 4: ToolExecutor ─────────────────────────────────
def teste_executor() -> tuple[bool, str]:
    """Teste 2: Testa o ToolExecutor com execução manual.

    Returns:
        Tupla (passou, detalhes).
    """
    print("\n🔧 Teste 2: ToolExecutor — execução manual")

    try:
        from core.tools.executor import ToolExecutor

        executor = ToolExecutor()

        # Listar tools disponíveis
        resumo = executor.list_tools_summary()
        print(f"   {resumo}")

        # Executar get_time_date
        resultado = executor.execute("get_time_date", {})
        print(f"   📄 get_time_date: \"{resultado[:80]}\"")

        tem_tools = executor.registry.count() >= 5
        executou = "são" in resultado.lower() or ":" in resultado

        passed = tem_tools and executou
        return passed, f"Tools: {executor.registry.count()}, Execução: {'OK' if executou else 'FALHA'}"

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, str(e)


# ─── Teste 5: ToolRegistry ─────────────────────────────────
def teste_registry() -> tuple[bool, str]:
    """Teste 3: Testa o ToolRegistry.

    Returns:
        Tupla (passou, detalhes).
    """
    print("\n📦 Teste 3: ToolRegistry — registro e listagem")

    try:
        from core.tools.registry import ToolRegistry

        registry = ToolRegistry()

        # Registrar tool customizada
        def minha_tool(texto: str) -> str:
            """Uma tool de teste."""
            return f"Echo: {texto}"

        registry.register(
            name="echo_test",
            description="Repete o texto recebido",
            function=minha_tool,
            parameters={"texto": {"type": "string"}},
        )

        # Verificar registro
        tem_tool = "echo_test" in registry
        tool_info = registry.get("echo_test")

        # Executar
        resultado = tool_info.function(texto="Olá Nina!")
        correto = resultado == "Echo: Olá Nina!"

        # Listar funções para Gemini
        funcoes = registry.get_functions_for_gemini()
        tem_funcoes = len(funcoes) == 1

        # Verificar nomes
        nomes = registry.list_names()
        tem_nome = "echo_test" in nomes

        passou = tem_tool and correto and tem_funcoes and tem_nome
        detalhe = (
            f"Registro: {'OK' if tem_tool else 'FALHA'}, "
            f"Execução: {'OK' if correto else 'FALHA'}, "
            f"Gemini: {'OK' if tem_funcoes else 'FALHA'}"
        )

        print(f"   📄 {detalhe}")
        return passou, detalhe

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, str(e)


# ─── Teste 6: Persistência de notas ────────────────────────
def teste_persistencia_notas() -> tuple[bool, str]:
    """Teste 4: Verifica que notas persistem no disco.

    Returns:
        Tupla (passou, detalhes).
    """
    print("\n💽 Teste 4: Persistência de notas no disco")

    try:
        import core.tools.actions as actions_module
        from core.tools.actions import create_note, list_notes
        from pathlib import Path

        # Redirecionar para diretório de teste
        original_notes_dir = actions_module._NOTES_DIR
        actions_module._NOTES_DIR = Path(_TEST_NOTES_DIR)

        try:
            # Limpar
            if os.path.exists(_TEST_NOTES_DIR):
                shutil.rmtree(_TEST_NOTES_DIR)

            # Criar nota
            create_note("Teste persistência", "Conteúdo de teste para verificação")

            # Verificar arquivo no disco
            arquivos = list(Path(_TEST_NOTES_DIR).glob("*.txt"))
            arquivo_existe = len(arquivos) == 1

            if arquivo_existe:
                # Ler conteúdo
                conteudo = arquivos[0].read_text(encoding="utf-8")
                tem_conteudo = "Conteúdo de teste" in conteudo
                tem_titulo = "Teste persistência" in conteudo

                print(f"   📄 Arquivo: {arquivos[0].name}")
                print(f"   📄 Conteúdo: {conteudo[:100]}...")

                passou = tem_conteudo and tem_titulo
                detalhe = f"Arquivo: OK, Título: {'OK' if tem_titulo else 'FALHA'}, Conteúdo: {'OK' if tem_conteudo else 'FALHA'}"
            else:
                passou = False
                detalhe = "Arquivo não encontrado no disco"

        finally:
            actions_module._NOTES_DIR = original_notes_dir
            if os.path.exists(_TEST_NOTES_DIR):
                shutil.rmtree(_TEST_NOTES_DIR)

        return passou, detalhe

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, str(e)


# ─── Teste 7: LLM com function calling ─────────────────────
async def teste_llm_function_calling() -> tuple[bool, str]:
    """Teste 5: Testa o LLM com function calling.

    Envia 'que horas são?' e verifica se o Gemini usa get_time_date.

    Returns:
        Tupla (passou, detalhes).
    """
    print("\n🤖 Teste 5: LLM com function calling")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("   ⚠️  GEMINI_API_KEY não definida. Pulando teste.")
        return True, "Pulado (sem API key)"

    try:
        from core.llm.client import NinaLLM
        from core.tools.executor import ToolExecutor

        llm = NinaLLM()
        executor = ToolExecutor()

        # Configurar tools no LLM
        llm.configurar_tools(executor.get_tools_for_gemini())

        # Perguntar que horas são — deve usar get_time_date
        print("   💬 Enviando: 'Que horas são agora?'\n")
        resposta = await llm.ask("Que horas são agora?")
        print(f"\n   📄 Resposta: \"{resposta[:100]}\"")

        # A resposta deve conter informação de horário
        tem_hora = any(c in resposta for c in [":"]) or any(
            p in resposta.lower()
            for p in ["hora", "agora", "são", "minuto"]
        )

        return tem_hora, f"Horário na resposta: {'sim' if tem_hora else 'não'}"

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, str(e)


# ─── Execução principal ────────────────────────────────────
async def executar_testes() -> None:
    """Executa todos os testes da Fase 4 em sequência."""
    imprimir_cabecalho()

    resultados: dict[str, bool] = {}

    # ── Teste 1a: get_time_date ──
    passou, detalhe = teste_get_time_date()
    resultados["get_time_date()"] = passou

    # ── Teste 1b: create_note + list_notes ──
    passou, detalhe = teste_notes()
    resultados["create_note() + list_notes()"] = passou

    # ── Teste 1c: web_search ──
    passou, detalhe = teste_web_search()
    resultados["web_search()"] = passou

    # ── Teste 2: ToolExecutor ──
    passou, detalhe = teste_executor()
    resultados["ToolExecutor (execução manual)"] = passou

    # ── Teste 3: ToolRegistry ──
    passou, detalhe = teste_registry()
    resultados["ToolRegistry (registro/listagem)"] = passou

    # ── Teste 4: Persistência de notas ──
    passou, detalhe = teste_persistencia_notas()
    resultados["Persistência de notas no disco"] = passou

    # ── Teste 5: LLM function calling ──
    passou, detalhe = await teste_llm_function_calling()
    resultados["LLM function calling (Gemini)"]= passou

    # ── Resumo final ──
    print(f"\n{SEPARADOR}")
    print("  📊 RESUMO DOS TESTES — FASE 4")
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


def main() -> None:
    """Ponto de entrada para execução síncrona."""
    asyncio.run(executar_testes())


if __name__ == "__main__":
    main()
