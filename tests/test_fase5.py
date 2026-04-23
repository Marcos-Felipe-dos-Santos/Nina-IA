"""
tests/test_fase5.py
===================
Teste isolado da Fase 5 — Visão de tela (multimodal).

Execute com:
    python tests/test_fase5.py

Testes realizados:
    1. Captura screenshot e verifica se a imagem foi salva
    2. Envia screenshot ao Gemini Vision e verifica resposta (requer GEMINI_API_KEY)
    3. Testa a tool look_at_screen() (requer GEMINI_API_KEY)
    4. Simula modo auto com 2 capturas
    5. Exibe PASS ou FAIL para cada teste
"""

import os
import shutil
import sys
import time
import traceback
from pathlib import Path

# Adicionar raiz do projeto ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Constantes ─────────────────────────────────────────────
SEPARADOR = "=" * 60
RESULTADO_PASS = "✅ PASS"
RESULTADO_FAIL = "❌ FAIL"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST_SCREENSHOTS_DIR = os.path.join(_PROJECT_ROOT, "data", "screenshots_test")


def imprimir_cabecalho() -> None:
    """Imprime o cabeçalho do teste."""
    print(f"\n{SEPARADOR}")
    print("  🧪 TESTE DA FASE 5 — Visão de Tela (Multimodal)")
    print(f"{SEPARADOR}\n")


def imprimir_resultado(nome: str, passou: bool, detalhe: str = "") -> None:
    """Imprime o resultado de um teste."""
    status = RESULTADO_PASS if passou else RESULTADO_FAIL
    msg = f"  {status} | {nome}"
    if detalhe:
        msg += f" — {detalhe}"
    print(msg)


# ─── Teste 1: Captura de screenshot ────────────────────────
def teste_captura_screenshot() -> tuple[bool, str]:
    """Teste 1: Captura screenshot e verifica imagem e arquivo.

    Returns:
        Tupla (passou, detalhes).
    """
    print("\n📸 Teste 1: Captura de screenshot com mss")

    try:
        from core.vision.capture import ScreenCapture

        # Criar captura com diretório de teste
        captura = ScreenCapture()
        original_dir = captura.screenshot_dir
        captura.screenshot_dir = _TEST_SCREENSHOTS_DIR
        Path(_TEST_SCREENSHOTS_DIR).mkdir(parents=True, exist_ok=True)

        try:
            img, caminho = captura.capture()

            # Verificações
            tem_imagem = img is not None
            tem_dimensoes = img.width > 0 and img.height > 0
            arquivo_existe = caminho.exists()
            tamanho_kb = caminho.stat().st_size / 1024 if arquivo_existe else 0

            print(f"   📄 Dimensões: {img.width}x{img.height}")
            print(f"   📄 Arquivo: {caminho}")
            print(f"   📄 Tamanho: {tamanho_kb:.1f} KB")

            # Testar capture_as_bytes
            jpeg_bytes, img2 = captura.capture_as_bytes()
            tem_bytes = len(jpeg_bytes) > 0

            print(f"   📄 Bytes JPEG: {len(jpeg_bytes)} bytes")

            passou = tem_imagem and tem_dimensoes and arquivo_existe and tem_bytes
            detalhe = (
                f"{img.width}x{img.height}, "
                f"{tamanho_kb:.1f} KB, "
                f"bytes: {len(jpeg_bytes)}"
            )

        finally:
            captura.screenshot_dir = original_dir
            if os.path.exists(_TEST_SCREENSHOTS_DIR):
                shutil.rmtree(_TEST_SCREENSHOTS_DIR)

        return passou, detalhe

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, str(e)


# ─── Teste 2: Gemini Vision — descrição ────────────────────
def teste_gemini_vision() -> tuple[bool, str]:
    """Teste 2: Envia screenshot ao Gemini Vision.

    Returns:
        Tupla (passou, detalhes).
    """
    print("\n🧠 Teste 2: Gemini Vision — describe_screen()")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("   ⚠️  GEMINI_API_KEY não definida ou inválida. Pulando teste.")
        return True, "Pulado (sem API key)"

    try:
        from core.vision.capture import ScreenCapture
        from core.vision.analyzer import VisionAnalyzer

        captura = ScreenCapture()
        analisador = VisionAnalyzer()

        # Capturar screenshot
        img, _ = captura.capture()

        # Descrever tela
        print("   💭 Enviando ao Gemini Vision...")
        descricao = analisador.describe_screen(img)

        print(f"   📄 Descrição: \"{descricao[:150]}\"")

        tem_texto = len(descricao) > 10
        é_fallback = "erro" in descricao.lower() or "desabilitada" in descricao.lower()

        passou = tem_texto
        detalhe = f"{len(descricao)} chars retornados (Fallback: {é_fallback})"

        return passou, detalhe

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, str(e)


# ─── Teste 3: Analyze for context ──────────────────────────
def teste_analyze_context() -> tuple[bool, str]:
    """Teste 3: Testa análise contextual com pergunta.

    Returns:
        Tupla (passou, detalhes).
    """
    print("\n🔍 Teste 3: Gemini Vision — analyze_for_context()")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("   ⚠️  GEMINI_API_KEY não definida ou inválida. Pulando teste.")
        return True, "Pulado (sem API key)"

    try:
        from core.vision.capture import ScreenCapture
        from core.vision.analyzer import VisionAnalyzer

        captura = ScreenCapture()
        analisador = VisionAnalyzer()

        # Capturar screenshot
        img, _ = captura.capture()

        # Fazer pergunta sobre a tela
        pergunta = "Qual aplicativo está aberto na tela?"
        print(f"   💬 Pergunta: '{pergunta}'")

        resposta = analisador.analyze_for_context(img, pergunta)

        print(f"   📄 Resposta: \"{resposta[:150]}\"")

        tem_texto = len(resposta) > 10
        é_fallback = "erro" in resposta.lower() or "desabilitada" in resposta.lower()

        passou = tem_texto
        return passou, f"{len(resposta)} chars (Fallback: {é_fallback})"

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, str(e)


# ─── Teste 4: Modo auto (watch) ────────────────────────────
def teste_modo_auto() -> tuple[bool, str]:
    """Teste 4: Simula modo auto com 2 capturas periódicas.

    Returns:
        Tupla (passou, detalhes).
    """
    print("\n👁️  Teste 4: Modo auto — start_watch / stop_watch")

    try:
        from core.vision.capture import ScreenCapture

        captura = ScreenCapture()
        original_dir = captura.screenshot_dir
        captura.screenshot_dir = _TEST_SCREENSHOTS_DIR
        Path(_TEST_SCREENSHOTS_DIR).mkdir(parents=True, exist_ok=True)

        try:
            # Iniciar watch com intervalo curto (2s)
            captura.start_watch(interval_seconds=2)
            print("   ⏱️  Watch iniciado (intervalo: 2s)...")
            print("   ⏳ Aguardando 5 segundos para capturas...")

            time.sleep(5)

            # Parar watch
            captura.stop_watch()
            print("   ⏹️  Watch parado.")

            # Verificar quantidade de screenshots
            arquivos = list(Path(_TEST_SCREENSHOTS_DIR).glob("*.jpg"))
            total_capturas = len(arquivos)

            print(f"   📄 Screenshots capturados: {total_capturas}")

            # Deve ter pelo menos 2 capturas
            passou = total_capturas >= 2
            detalhe = f"{total_capturas} capturas em 5s"

        finally:
            captura.stop_watch()
            captura.screenshot_dir = original_dir
            if os.path.exists(_TEST_SCREENSHOTS_DIR):
                shutil.rmtree(_TEST_SCREENSHOTS_DIR)

        return passou, detalhe

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, str(e)


# ─── Teste 5: Tool look_at_screen ──────────────────────────
def teste_tool_look_at_screen() -> tuple[bool, str]:
    """Teste 5: Testa a tool look_at_screen() no executor.

    Returns:
        Tupla (passou, detalhes).
    """
    print("\n🔧 Teste 5: Tool look_at_screen()")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("   ⚠️  GEMINI_API_KEY não definida ou inválida. Pulando teste.")
        return True, "Pulado (sem API key)"

    try:
        from core.tools.executor import ToolExecutor

        executor = ToolExecutor()

        # Verificar que look_at_screen está registrada
        tem_tool = "look_at_screen" in executor.registry
        print(f"   📄 Tool registrada: {'sim' if tem_tool else 'não'}")

        if not tem_tool:
            return False, "Tool não registrada"

        # Executar
        print("   💭 Executando look_at_screen()...")
        resultado = executor.execute("look_at_screen", {})

        print(f"   📄 Resultado: \"{resultado[:150]}\"")

        tem_texto = len(resultado) > 10
        é_fallback = "erro" in resultado.lower() or "desabilitada" in resultado.lower()

        passou = tem_tool and tem_texto
        return passou, f"Tool OK, {len(resultado)} chars (Fallback: {é_fallback})"

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, str(e)


# ─── Execução principal ────────────────────────────────────
def main() -> None:
    """Executa todos os testes da Fase 5 em sequência."""
    imprimir_cabecalho()

    # Limpar diretório de teste
    if os.path.exists(_TEST_SCREENSHOTS_DIR):
        shutil.rmtree(_TEST_SCREENSHOTS_DIR)

    resultados: dict[str, bool] = {}

    # ── Teste 1: Captura ──
    passou, detalhe = teste_captura_screenshot()
    resultados["Captura de screenshot (mss)"] = passou

    # ── Teste 2: Gemini Vision describe ──
    passou, detalhe = teste_gemini_vision()
    resultados["Gemini Vision — describe_screen()"] = passou

    # ── Teste 3: Analyze for context ──
    passou, detalhe = teste_analyze_context()
    resultados["Gemini Vision — analyze_for_context()"] = passou

    # ── Teste 4: Modo auto ──
    passou, detalhe = teste_modo_auto()
    resultados["Modo auto — watch periódico"] = passou

    # ── Teste 5: Tool ──
    passou, detalhe = teste_tool_look_at_screen()
    resultados["Tool look_at_screen()"] = passou

    # Limpar
    if os.path.exists(_TEST_SCREENSHOTS_DIR):
        shutil.rmtree(_TEST_SCREENSHOTS_DIR)

    # ── Resumo final ──
    print(f"\n{SEPARADOR}")
    print("  📊 RESUMO DOS TESTES — FASE 5")
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
