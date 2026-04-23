"""
tests/test_fase1.py
===================
Teste isolado da Fase 1 — Pipeline de voz base.

Execute com:
    python test_fase1.py

Testes realizados:
    1. Grava 3 segundos de áudio do microfone
    2. Transcreve o áudio com WhisperX e imprime o resultado
    3. Sintetiza a frase "Olá, eu sou a Nina" com Kokoro e reproduz
    4. Imprime as latências de cada etapa
    5. Exibe PASS ou FAIL para cada teste
"""

import sys
import os
import time
import traceback

# Adicionar raiz do projeto ao path para importações
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils.latency import LatencyTracker


# ─── Constantes ─────────────────────────────────────────────
FRASE_TTS = "Olá, eu sou a Nina"
SEPARADOR = "=" * 60
RESULTADO_PASS = "✅ PASS"
RESULTADO_FAIL = "❌ FAIL"


def imprimir_cabecalho() -> None:
    """Imprime o cabeçalho do teste."""
    print(f"\n{SEPARADOR}")
    print("  🧪 TESTE DA FASE 1 — Pipeline de Voz Base")
    print(f"{SEPARADOR}\n")


def imprimir_resultado(nome: str, passou: bool, detalhe: str = "") -> None:
    """Imprime o resultado de um teste individual.

    Args:
        nome: Nome do teste.
        passou: Se o teste passou ou não.
        detalhe: Informação adicional opcional.
    """
    status = RESULTADO_PASS if passou else RESULTADO_FAIL
    msg = f"  {status} | {nome}"
    if detalhe:
        msg += f" — {detalhe}"
    print(msg)


# ─── Teste 1: Gravação de áudio ────────────────────────────
def teste_gravacao_microfone() -> tuple[bool, any]:
    """Teste 1: Grava 3 segundos de áudio do microfone.

    Returns:
        Tupla (passou, audio_array).
    """
    print("\n🎤 Teste 1: Gravação de 3 segundos do microfone")
    print("   Fale algo nos próximos 3 segundos...\n")

    try:
        from core.stt.microphone import MicrophoneCapture

        mic = MicrophoneCapture()

        # Contagem regressiva
        for i in range(3, 0, -1):
            print(f"   ⏱  Gravando em {i}...")
            time.sleep(1)

        print("   🔴 GRAVANDO...")
        audio = mic.gravar_segundos(duracao_segundos=3.0)
        mic.encerrar()

        if audio is not None and len(audio) > 0:
            duracao = len(audio) / mic.sample_rate
            return True, audio
        else:
            return False, None

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, None


# ─── Teste 2: Transcrição com WhisperX ─────────────────────
def teste_transcricao(audio) -> tuple[bool, str, float]:
    """Teste 2: Transcreve o áudio gravado com WhisperX.

    Args:
        audio: Array numpy float32 do áudio capturado.

    Returns:
        Tupla (passou, texto_transcrito, latencia_ms).
    """
    print("\n📝 Teste 2: Transcrição com WhisperX")

    try:
        from core.stt.transcriber import WhisperTranscriber

        transcritor = WhisperTranscriber()
        texto, latencia_ms = transcritor.transcrever_array(audio)

        print(f"   📄 Texto: \"{texto}\"")
        print(f"   ⏱  Latência STT: {latencia_ms:.0f}ms")

        # Consideramos sucesso se retornou sem erro,
        # mesmo que o texto esteja vazio (silêncio)
        return True, texto, latencia_ms

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, "", 0.0


# ─── Teste 3: Síntese e reprodução com Kokoro TTS ──────────
def teste_sintese_tts() -> tuple[bool, float]:
    """Teste 3: Sintetiza e reproduz frase com Kokoro TTS.

    Returns:
        Tupla (passou, latencia_ms).
    """
    print(f"\n🔊 Teste 3: Síntese TTS — \"{FRASE_TTS}\"")

    try:
        from core.tts.synthesizer import KokoroSynthesizer

        sintetizador = KokoroSynthesizer()

        # Sintetizar
        audio, latencia_ms = sintetizador.sintetizar(FRASE_TTS)

        print(f"   ⏱  Latência TTS: {latencia_ms:.0f}ms")

        if len(audio) == 0:
            print("   ⚠️  Áudio gerado está vazio!")
            return False, latencia_ms

        duracao = len(audio) / sintetizador.sample_rate
        print(f"   🎵 Áudio gerado: {duracao:.2f}s")

        # Reproduzir
        print("   ▶️  Reproduzindo...")
        sintetizador.reproduzir(audio)

        return True, latencia_ms

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, 0.0


# ─── Teste 4: Latency Tracker ──────────────────────────────
def teste_latency_tracker(latencia_stt: float, latencia_tts: float) -> bool:
    """Teste 4: Verifica e exibe o Latency Tracker.

    Args:
        latencia_stt: Latência medida no teste de STT (ms).
        latencia_tts: Latência medida no teste de TTS (ms).

    Returns:
        True se o tracker funcionou corretamente.
    """
    print("\n⏱  Teste 4: Latency Tracker")

    try:
        tracker = LatencyTracker()

        # Simular as medições com os valores reais dos testes
        tracker.iniciar("STT")
        time.sleep(latencia_stt / 1000.0 if latencia_stt > 0 else 0.001)
        tracker.finalizar("STT")

        # LLM não foi testado nesta fase (deve aparecer como "--ms")

        tracker.iniciar("TTS")
        time.sleep(latencia_tts / 1000.0 if latencia_tts > 0 else 0.001)
        tracker.finalizar("TTS")

        # Exibir resultado formatado
        linha = tracker.formatar()
        print(f"   {linha}")

        # Validar que o formato está correto
        tem_stt = "[STT]" in linha
        tem_llm = "[LLM] --ms" in linha  # LLM deve estar vazio
        tem_tts = "[TTS]" in linha
        tem_total = "[TOTAL]" in linha

        return tem_stt and tem_llm and tem_tts and tem_total

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False


# ─── Execução principal ────────────────────────────────────
def main() -> None:
    """Executa todos os testes da Fase 1 em sequência."""
    imprimir_cabecalho()

    resultados: dict[str, bool] = {}
    latencia_stt = 0.0
    latencia_tts = 0.0

    # ── Teste 1: Gravação ──
    passou, audio = teste_gravacao_microfone()
    resultados["Gravação de áudio (3s)"] = passou

    if not passou:
        print("\n⚠️  Teste 1 falhou. Pulando teste de transcrição.")
        resultados["Transcrição WhisperX"] = False
    else:
        # ── Teste 2: Transcrição ──
        passou, texto, latencia_stt = teste_transcricao(audio)
        resultados["Transcrição WhisperX"] = passou

    # ── Teste 3: Síntese TTS ──
    passou, latencia_tts = teste_sintese_tts()
    resultados["Síntese Kokoro TTS"] = passou

    # ── Teste 4: Latency Tracker ──
    passou = teste_latency_tracker(latencia_stt, latencia_tts)
    resultados["Latency Tracker"] = passou

    # ── Resumo final ──
    print(f"\n{SEPARADOR}")
    print("  📊 RESUMO DOS TESTES — FASE 1")
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

    # Retornar código de saída
    sys.exit(0 if aprovados == total else 1)


if __name__ == "__main__":
    main()
