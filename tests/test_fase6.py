"""
tests/test_fase6.py
===================
Teste isolado da Fase 6 — Dashboard web com FastAPI.

Execute com:
    python tests/test_fase6.py

Testes realizados:
    1. Inicia o servidor FastAPI em modo de teste
    2. Testa cada endpoint GET com httpx
    3. Simula eventos via WebSocket e verifica recebimento
    4. Verifica que o histórico é atualizado após conversa simulada
    5. Exibe PASS ou FAIL para cada teste
"""

import asyncio
import json
import os
import sys
import time
import traceback

# Adicionar raiz do projeto ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Constantes ─────────────────────────────────────────────
SEPARADOR = "=" * 60
RESULTADO_PASS = "✅ PASS"
RESULTADO_FAIL = "❌ FAIL"
BASE_URL = "http://127.0.0.1:8765"
WS_URL = "ws://127.0.0.1:8765/ws"


def imprimir_cabecalho() -> None:
    """Imprime o cabeçalho do teste."""
    print(f"\n{SEPARADOR}")
    print("  🧪 TESTE DA FASE 6 — Dashboard Web (FastAPI)")
    print(f"{SEPARADOR}\n")


def imprimir_resultado(nome: str, passou: bool, detalhe: str = "") -> None:
    """Imprime o resultado de um teste."""
    status = RESULTADO_PASS if passou else RESULTADO_FAIL
    msg = f"  {status} | {nome}"
    if detalhe:
        msg += f" — {detalhe}"
    print(msg)


# ─── Server Setup ──────────────────────────────────────────
_server_thread = None
_server_started = False


def start_test_server():
    """Inicia o servidor FastAPI em thread separada para testes."""
    global _server_thread, _server_started

    import threading
    import uvicorn
    from dashboard.api import app

    def _run():
        uvicorn.run("dashboard.api:app", host="127.0.0.1", port=8765, log_level="info")

    _server_thread = threading.Thread(target=_run, daemon=True)
    _server_thread.start()
    _server_started = True

    # Aguardar servidor iniciar
    time.sleep(5)
    print("   🌐 Servidor de teste iniciado em http://127.0.0.1:8765\n")


# ─── Teste 1: Endpoints GET ────────────────────────────────
def teste_endpoints_get() -> tuple[bool, str]:
    """Teste 1: Testa cada endpoint GET com httpx.

    Returns:
        Tupla (passou, detalhes).
    """
    print("\n🌐 Teste 1: Endpoints GET")

    try:
        import requests

        endpoints = [
            ("/status", ["state", "timestamp"]),
            ("/history", ["conversations", "total"]),
            ("/memories", ["memories", "total"]),
            ("/metrics", []),
            ("/tools/log", ["tools", "total"]),
        ]

        resultados = {}

        for path, expected_keys in endpoints:
            url = f"{BASE_URL}{path}"
            response = requests.get(url, timeout=15)

            status_ok = response.status_code == 200
            data = response.json()

            # Verificar que as chaves esperadas existem
            keys_ok = all(k in data for k in expected_keys)

            resultado = status_ok and keys_ok
            resultados[path] = resultado

            status_icon = "✅" if resultado else "❌"
            print(f"   {status_icon} GET {path} → {response.status_code}, keys: {list(data.keys())}")

        todos_ok = all(resultados.values())
        detalhe = f"{sum(resultados.values())}/{len(resultados)} endpoints OK"

        return todos_ok, detalhe

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, str(e)


# ─── Teste 2: Página HTML ──────────────────────────────────
def teste_pagina_html() -> tuple[bool, str]:
    """Teste 2: Verifica que a página HTML é servida.

    Returns:
        Tupla (passou, detalhes).
    """
    print("\n📄 Teste 2: Página HTML (index.html)")

    try:
        import requests

        response = requests.get(f"{BASE_URL}/", timeout=15)

        status_ok = response.status_code == 200
        tem_html = "Nina" in response.text
        tem_css = "var(--bg-primary)" in response.text
        tem_js = "connectWebSocket" in response.text

        passou = status_ok and tem_html and tem_css and tem_js

        print(f"   📄 Status: {response.status_code}")
        print(f"   📄 HTML: {'OK' if tem_html else 'FALHA'}")
        print(f"   📄 CSS: {'OK' if tem_css else 'FALHA'}")
        print(f"   📄 JS: {'OK' if tem_js else 'FALHA'}")
        print(f"   📄 Tamanho: {len(response.text)} bytes")

        return passou, f"{len(response.text)} bytes, todos componentes presentes"

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, str(e)


# ─── Teste 3: EventBus ─────────────────────────────────────
def teste_event_bus() -> tuple[bool, str]:
    """Teste 3: Testa o EventBus — estado, histórico e métricas.

    Returns:
        Tupla (passou, detalhes).
    """
    print("\n📡 Teste 3: EventBus — estado, histórico e métricas")

    try:
        from dashboard.events import NinaState, event_bus

        # Testar mudança de estado
        event_bus.set_state(NinaState.LISTENING)
        estado_ok = event_bus.state == "listening"
        print(f"   📄 Estado listening: {'OK' if estado_ok else 'FALHA'}")

        event_bus.set_state(NinaState.THINKING)
        estado2_ok = event_bus.state == "thinking"
        print(f"   📄 Estado thinking: {'OK' if estado2_ok else 'FALHA'}")

        # Adicionar conversa
        event_bus.add_conversation(
            "Que horas são?",
            "São 14:30 de quarta-feira.",
            {"STT": 320, "LLM": 850, "TTS": 210},
        )

        history = event_bus.get_history()
        tem_historico = len(history) > 0
        print(f"   📄 Histórico: {len(history)} conversas")

        # Verificar métricas
        metrics = event_bus.get_metrics()
        tem_metricas = "STT" in metrics and metrics["STT"]["count"] > 0
        print(f"   📄 Métricas STT: {metrics.get('STT', {})}")

        # Log tool
        event_bus.log_tool("get_time_date", {}, "São 14:30")
        tool_log = event_bus.get_tool_log()
        tem_tool_log = len(tool_log) > 0
        print(f"   📄 Tool log: {len(tool_log)} entradas")

        # Status
        status = event_bus.get_status()
        tem_status = "state" in status
        print(f"   📄 Status: {status}")

        # Voltar ao idle
        event_bus.set_state(NinaState.IDLE)

        passou = estado_ok and estado2_ok and tem_historico and tem_metricas and tem_tool_log and tem_status
        return passou, "Todos componentes do EventBus OK"

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, str(e)


# ─── Teste 4: Histórico via API ────────────────────────────
def teste_historico_api() -> tuple[bool, str]:
    """Teste 4: Verifica que o histórico reflete no endpoint.

    Returns:
        Tupla (passou, detalhes).
    """
    print("\n📜 Teste 4: Histórico atualizado via API")

    try:
        import requests
        from dashboard.events import event_bus

        # Adicionar mais uma conversa
        event_bus.add_conversation(
            "Qual a capital do Brasil?",
            "Brasília é a capital do Brasil.",
            {"STT": 280, "LLM": 600, "TTS": 180},
        )

        # Buscar via API
        response = requests.get(f"{BASE_URL}/history", timeout=15)
        data = response.json()

        conversations = data.get("conversations", [])
        tem_conversa = any(
            "Brasília" in c.get("nina", "") or "capital" in c.get("user", "")
            for c in conversations
        )

        print(f"   📄 Total conversas: {len(conversations)}")
        print(f"   📄 Conversa sobre capital: {'encontrada' if tem_conversa else 'não encontrada'}")

        # Verificar métricas atualizadas
        response_metrics = requests.get(f"{BASE_URL}/metrics", timeout=15)
        metrics = response_metrics.json()
        stt_count = metrics.get("STT", {}).get("count", 0)

        print(f"   📄 Métricas STT amostras: {stt_count}")

        passou = tem_conversa and stt_count >= 2
        return passou, f"{len(conversations)} conversas, {stt_count} amostras"

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, str(e)


# ─── Teste 5: WebSocket ────────────────────────────────────
async def teste_websocket_async() -> tuple[bool, str]:
    """Teste 5: Conecta via WebSocket e recebe eventos.

    Returns:
        Tupla (passou, detalhes).
    """
    print("\n🔌 Teste 5: WebSocket — stream de eventos")

    try:
        import websockets
        from dashboard.events import NinaState, event_bus

        mensagens_recebidas = []

        async with websockets.connect(WS_URL) as ws:
            # Deve receber initial_state primeiro
            msg = await asyncio.wait_for(ws.recv(), timeout=15)
            data = json.loads(msg)
            mensagens_recebidas.append(data)

            tem_initial = data.get("type") == "initial_state"
            print(f"   📄 Initial state: {'OK' if tem_initial else 'FALHA'}")

            # Emitir eventos
            event_bus.set_state(NinaState.LISTENING)

            # Receber evento de state change
            msg2 = await asyncio.wait_for(ws.recv(), timeout=15)
            data2 = json.loads(msg2)
            mensagens_recebidas.append(data2)

            tem_state_change = data2.get("type") == "state_change"
            estado_correto = data2.get("state") == "listening"
            print(f"   📄 State change: {'OK' if tem_state_change else 'FALHA'}")
            print(f"   📄 Estado correto: {'OK' if estado_correto else 'FALHA'}")

            # Emitir conversa
            event_bus.add_conversation("Teste WS", "Resposta WS", {"STT": 100})

            msg3 = await asyncio.wait_for(ws.recv(), timeout=15)
            data3 = json.loads(msg3)

            tem_new_conv = data3.get("type") == "new_conversation"
            print(f"   📄 New conversation: {'OK' if tem_new_conv else 'FALHA'}")

            # Reset
            event_bus.set_state(NinaState.IDLE)

        passou = tem_initial and tem_state_change and estado_correto and tem_new_conv
        detalhe = f"{len(mensagens_recebidas) + 1} mensagens recebidas"

        return passou, detalhe

    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
        traceback.print_exc()
        return False, str(e)


def teste_websocket() -> tuple[bool, str]:
    """Wrapper síncrono para o teste de WebSocket."""
    return asyncio.get_event_loop().run_until_complete(teste_websocket_async())


# ─── Execução principal ────────────────────────────────────
def main() -> None:
    """Executa todos os testes da Fase 6 em sequência."""
    imprimir_cabecalho()

    # Criar event loop antes de tudo
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Iniciar servidor de teste
    start_test_server()

    resultados: dict[str, bool] = {}

    # ── Teste 1: Endpoints GET ──
    passou, detalhe = teste_endpoints_get()
    resultados["Endpoints GET (5 rotas)"] = passou

    # ── Teste 2: Página HTML ──
    passou, detalhe = teste_pagina_html()
    resultados["Página HTML (index.html)"] = passou

    # ── Teste 3: EventBus ──
    passou, detalhe = teste_event_bus()
    resultados["EventBus (estado/histórico/métricas)"] = passou

    # ── Teste 4: Histórico via API ──
    passou, detalhe = teste_historico_api()
    resultados["Histórico atualizado via API"] = passou

    # ── Teste 5: WebSocket ──
    passou, detalhe = teste_websocket()
    resultados["WebSocket (stream de eventos)"] = passou

    # ── Resumo final ──
    print(f"\n{SEPARADOR}")
    print("  📊 RESUMO DOS TESTES — FASE 6")
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
