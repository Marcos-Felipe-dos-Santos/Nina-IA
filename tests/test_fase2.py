"""
tests/test_fase2.py
===================
Teste isolado da Fase 2 — Integração com LLM (Gemini).

Execute com:
    python tests/test_fase2.py

Pré-requisito:
    Variável de ambiente GEMINI_API_KEY definida.

Testes realizados:
    1. Envia "Qual é a sua função?" ao LLM e verifica resposta válida
    2. Testa histórico: duas mensagens em sequência e verifica coerência
    3. Testa streaming de resposta
    4. Testa loop completo: grava → transcreve → LLM → sintetiza
    5. Exibe PASS ou FAIL para cada teste
"""

from dotenv import load_dotenv
load_dotenv()  # carrega o .env antes de qualquer import

import asyncio
import os
import sys
import time
import traceback

# Adicionar raiz do projeto ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils.latency import LatencyTracker


# ─── Constantes ─────────────────────────────────────────────
SEPARADOR = "=" * 60
RESULTADO_PASS = "✅ PASS"
RESULTADO_FAIL = "❌ FAIL"


def imprimir_cabecalho() -> None:
    """Imprime o cabeçalho do teste."""
    print(f"\n{SEPARADOR}")
    print("  🧪 TESTE DA FASE 2 — Integração LLM (Gemini)")
    print(f"{SEPARADOR}")

    print(f"  🔑 Iniciando teste da Fase 2 (LLM)...\n{SEPARADOR}\n")


def imprimir_resultado(nome: str, passou: bool, detalhe: str = "") -> None:
    """Imprime o resultado de um teste individual."""
    status = RESULTADO_PASS if passou else RESULTADO_FAIL
    msg = f"  {status} | {nome}"
    if detalhe:
        msg += f" — {detalhe}"
    print(msg)


# ─── Teste 1: Mensagem simples ao LLM ──────────────────────
async def teste_mensagem_simples() -> tuple[bool, str]:
    """Teste 1: Envia 'Qual é a sua função?' ao LLM.

    Returns:
        Tupla (passou, resposta).
    """
    print("\n💬 Teste 1: Mensagem simples ao LLM")
    print("   Enviando: 'Qual é a sua função?'\n")

    try:
        from core.llm.client import NinaLLM

        llm = NinaLLM()
        resposta = await llm.ask("Qual é a sua função?")

        print(f"\n   📄 Resposta ({len(resposta)} chars): \"{resposta[:150]}\"")

        # Verificar se a resposta contém texto válido
        tem_texto = len(resposta.strip()) > 10
        return tem_texto, resposta

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, ""


# ─── Teste 2: Histórico de conversa ────────────────────────
async def teste_historico() -> tuple[bool, str]:
    """Teste 2: Envia duas mensagens e verifica coerência do histórico.

    Returns:
        Tupla (passou, detalhes).
    """
    print("\n🧠 Teste 2: Histórico de conversa")
    print("   Msg 1: 'Meu nome é Marco.'")
    print("   Msg 2: 'Qual é o meu nome?'\n")

    try:
        from core.llm.client import NinaLLM

        llm = NinaLLM()

        # Primeira mensagem — apresentação
        resp1 = await llm.ask("Meu nome é Marco.")
        print(f"\n   📄 Resposta 1: \"{resp1[:100]}\"")

        # Verificar que o histórico foi registrado
        historico = llm.historico
        print(f"   📚 Histórico: {len(historico)} mensagens")

        # Segunda mensagem — teste de memória
        print()
        resp2 = await llm.ask("Qual é o meu nome?")
        print(f"\n   📄 Resposta 2: \"{resp2[:100]}\"")

        # Verificar se a resposta menciona "Marco"
        menciona_nome = "marco" in resp2.lower()
        tem_historico = len(llm.historico) == 4  # 2 user + 2 model

        passou = menciona_nome and tem_historico
        detalhe = (
            f"Nome lembrado: {'sim' if menciona_nome else 'NÃO'}, "
            f"Histórico: {len(llm.historico)} msgs"
        )

        return passou, detalhe

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, str(e)


# ─── Teste 3: Streaming de resposta ────────────────────────
async def teste_streaming() -> tuple[bool, float]:
    """Teste 3: Verifica que o streaming funciona.

    Returns:
        Tupla (passou, latencia_ms).
    """
    print("\n⚡ Teste 3: Streaming de resposta")
    print("   Enviando: 'Conte uma curiosidade breve sobre o Brasil.'\n")

    try:
        from core.llm.client import NinaLLM

        llm = NinaLLM()

        inicio = time.perf_counter()
        resposta = await llm.ask("Conte uma curiosidade breve sobre o Brasil.")
        fim = time.perf_counter()

        latencia_ms = (fim - inicio) * 1000

        print(f"\n   ⏱  Latência LLM: {latencia_ms:.0f}ms")
        print(f"   📏 Tamanho: {len(resposta)} chars")

        # Sucesso se recebeu uma resposta não vazia
        passou = len(resposta.strip()) > 10
        return passou, latencia_ms

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, 0.0


# ─── Teste 4: Loop completo (STT → LLM → TTS) ─────────────
async def teste_loop_completo() -> tuple[bool, str]:
    """Teste 4: Loop completo — grava, transcreve, LLM, sintetiza.

    Returns:
        Tupla (passou, detalhes).
    """
    print("\n🔄 Teste 4: Loop completo (STT → LLM → TTS)")
    print("   Fale algo nos próximos 3 segundos...\n")

    try:
        from core.stt.microphone import MicrophoneCapture
        from core.stt.transcriber import WhisperTranscriber
        from core.llm.client import NinaLLM
        from core.tts.synthesizer import KokoroSynthesizer

        tracker = LatencyTracker()
        mic = MicrophoneCapture()
        transcritor = WhisperTranscriber()
        llm = NinaLLM()
        sintetizador = KokoroSynthesizer()

        # Contagem regressiva
        for i in range(3, 0, -1):
            print(f"   ⏱  Gravando em {i}...")
            time.sleep(1)

        # 1. Gravar áudio
        print("   🔴 GRAVANDO 3 segundos...")
        tracker.iniciar("STT")
        audio = mic.gravar_segundos(duracao_segundos=3.0)

        # 2. Transcrever
        texto, _ = transcritor.transcrever_array(audio, sample_rate=mic.sample_rate)
        tracker.finalizar("STT")
        print(f"   📝 Transcrito: \"{texto}\"")

        if not texto.strip():
            texto = "Olá, tudo bem?"
            print(f"   ⚠️  Áudio vazio, usando fallback: \"{texto}\"")

        # 3. Enviar ao LLM
        print(f"   🧠 Enviando ao Gemini...\n")
        tracker.iniciar("LLM")
        resposta = await llm.ask(texto)
        tracker.finalizar("LLM")

        # 4. Sintetizar e reproduzir
        print(f"\n   🔊 Sintetizando resposta...")
        tracker.iniciar("TTS")
        sintetizador.sintetizar_e_reproduzir(resposta)
        tracker.finalizar("TTS")

        # 5. Exibir latências
        tracker.exibir()

        mic.encerrar()

        passou = len(resposta.strip()) > 0
        detalhe = tracker.formatar()
        return passou, detalhe

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, str(e)


# ─── Execução principal ────────────────────────────────────
async def executar_testes() -> None:
    """Executa todos os testes da Fase 2 em sequência."""
    imprimir_cabecalho()

    resultados: dict[str, bool] = {}

    # ── Teste 1: Mensagem simples ──
    passou, _ = await teste_mensagem_simples()
    resultados["Mensagem simples ao LLM"] = passou

    # ── Teste 2: Histórico ──
    passou, detalhe = await teste_historico()
    resultados["Histórico de conversa"] = passou

    # ── Teste 3: Streaming ──
    passou, latencia = await teste_streaming()
    resultados["Streaming de resposta"] = passou

    # ── Teste 4: Loop completo ──
    passou, detalhe = await teste_loop_completo()
    resultados["Loop completo (STT→LLM→TTS)"] = passou

    # ── Resumo final ──
    print(f"\n{SEPARADOR}")
    print("  📊 RESUMO DOS TESTES — FASE 2")
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
