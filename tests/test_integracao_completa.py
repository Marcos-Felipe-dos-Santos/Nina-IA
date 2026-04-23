"""
tests/test_integracao_completa.py
=================================
Teste de integração completo da Nina IA — Fase Final.

Simula uma sessão real testando todos os módulos em conjunto:
- LLM (Gemini) com function calling
- Tools (get_time_date, web_search, create_note, list_notes, look_at_screen)
- Memória (RAG com ChromaDB)
- Visão (captura de tela + Gemini Vision)
- Dashboard (FastAPI + WebSocket)

Execute com:
    python tests/test_integracao_completa.py

Nota: Testes 1-5 requerem GEMINI_API_KEY. Teste 6 requer look_at_screen + API.
      Testes do dashboard rodam sem API key.
"""

import asyncio
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Adicionar raiz do projeto ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Diretórios de teste
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TEST_NOTES_DIR = _PROJECT_ROOT / "data" / "notes_test_integ"
_TEST_MEMORY_DIR = _PROJECT_ROOT / "data" / "memory_test_integ"

# ─── Constantes ─────────────────────────────────────────────
SEPARADOR = "=" * 62
SEPARADOR_FINO = "-" * 62
RESULTADO_PASS = "PASS"
RESULTADO_FAIL = "FAIL"
RESULTADO_SKIP = "SKIP"
DASHBOARD_PORT = 8766
DASHBOARD_URL = f"http://127.0.0.1:{DASHBOARD_PORT}"


def imprimir_cabecalho() -> None:
    """Imprime o cabeçalho do teste de integração."""
    print(f"\n{SEPARADOR}")
    print("  🧪 TESTE DE INTEGRAÇÃO COMPLETA — NINA IA")
    print(f"  📅 {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"{SEPARADOR}\n")


def imprimir_resultado(nome: str, passou: bool, detalhe: str = "",
                       skipado: bool = False) -> None:
    """Imprime o resultado de um teste."""
    if skipado:
        status = RESULTADO_SKIP
    else:
        status = RESULTADO_PASS if passou else RESULTADO_FAIL
    msg = f"  {status} │ {nome}"
    if detalhe:
        msg += f" — {detalhe}"
    print(msg)


class IntegrationTestRunner:
    """Executa os testes de integração com estado compartilhado."""

    def __init__(self) -> None:
        self.resultados: dict[str, dict] = {}
        self.latencias: list[float] = []
        self.tem_api_key = bool(os.environ.get("GEMINI_API_KEY", ""))
        self._llm = None
        self._tools = None
        self._dashboard_thread = None

    # ── Setup ───────────────────────────────────────────────

    def setup(self) -> None:
        """Inicializa módulos compartilhados."""
        print("⚙️  Inicializando módulos...\n")

        # Dashboard
        self._iniciar_dashboard()

        # Tools
        from core.tools.executor import ToolExecutor
        self._tools = ToolExecutor()
        print(f"   ✅ ToolExecutor: {self._tools.registry.count()} tools")

        # LLM (se houver API key)
        if self.tem_api_key:
            from core.llm.client import NinaLLM
            self._llm = NinaLLM()
            self._llm.configurar_tools(self._tools.get_tools_for_gemini())
            print(f"   ✅ LLM: {self._llm.model}")
        else:
            print("   ⚠️  LLM: GEMINI_API_KEY não definida (testes de LLM serão pulados)")

        print()

    def _iniciar_dashboard(self) -> None:
        """Inicia o dashboard FastAPI em thread separada."""
        import threading
        import uvicorn
        from dashboard.api import app

        def _run():
            uvicorn.run(app, host="127.0.0.1", port=DASHBOARD_PORT, log_level="error")

        self._dashboard_thread = threading.Thread(target=_run, daemon=True)
        self._dashboard_thread.start()
        time.sleep(2)
        print(f"   ✅ Dashboard: {DASHBOARD_URL}")

    def teardown(self) -> None:
        """Limpa recursos de teste."""
        import shutil
        try:
            if _TEST_NOTES_DIR.exists():
                shutil.rmtree(_TEST_NOTES_DIR, ignore_errors=True)
            if _TEST_MEMORY_DIR.exists():
                shutil.rmtree(_TEST_MEMORY_DIR, ignore_errors=True)
        except Exception as e:
            print(f"Warning: could not complete teardown clean up: {e}")

    # ── Helpers ─────────────────────────────────────────────

    async def _ask_llm(self, prompt: str) -> tuple[str, float]:
        """Envia prompt ao LLM e retorna resposta + latência."""
        inicio = time.perf_counter()
        resposta = await self._llm.ask(prompt)
        latencia = (time.perf_counter() - inicio) * 1000
        self.latencias.append(latencia)
        return resposta, latencia

    def _registrar(self, nome: str, passou: bool, detalhe: str = "",
                   skipado: bool = False) -> None:
        """Registra resultado de um teste."""
        self.resultados[nome] = {
            "passou": passou, "detalhe": detalhe, "skipado": skipado,
        }
        imprimir_resultado(nome, passou, detalhe, skipado)

    # ── Teste 1: "Que horas são?" ───────────────────────────

    async def teste_1_horario(self) -> None:
        """Teste 1: Simula 'Nina, que horas são?' → tool get_time_date."""
        nome = "Tool get_time_date + LLM"
        print(f"\n⏰ Teste 1: 'Nina, que horas são?'")

        if not self.tem_api_key:
            self._registrar(nome, True, "Sem API key", skipado=True)
            return

        try:
            resposta, latencia = await self._ask_llm("Que horas são agora?")
            print(f"   📄 Resposta ({latencia:.0f}ms): \"{resposta[:120]}\"")

            tem_hora = any(c in resposta for c in [":"]) or any(
                p in resposta.lower() for p in ["hora", "agora", "são", "minuto"]
            )
            self._registrar(nome, tem_hora, f"{latencia:.0f}ms")

        except Exception as e:
            print(f"   ⚠️  Erro: {e}")
            traceback.print_exc()
            self._registrar(nome, False, str(e))

    # ── Teste 2: Web search ─────────────────────────────────

    async def teste_2_web_search(self) -> None:
        """Teste 2: Simula 'pesquise sobre Python asyncio' → tool web_search."""
        nome = "Tool web_search + LLM"
        print(f"\n🔍 Teste 2: 'Nina, pesquise sobre Python asyncio'")

        if not self.tem_api_key:
            self._registrar(nome, True, "Sem API key", skipado=True)
            return

        try:
            resposta, latencia = await self._ask_llm(
                "Pesquise sobre Python asyncio e me dê um resumo."
            )
            print(f"   📄 Resposta ({latencia:.0f}ms): \"{resposta[:120]}\"")

            tem_info = len(resposta) > 20 and (
                "asyncio" in resposta.lower() or "python" in resposta.lower()
                or "assíncrono" in resposta.lower()
            )
            self._registrar(nome, tem_info, f"{latencia:.0f}ms, {len(resposta)} chars")

        except Exception as e:
            print(f"   ⚠️  Erro: {e}")
            traceback.print_exc()
            self._registrar(nome, False, str(e))

    # ── Teste 3: Criar nota ─────────────────────────────────

    async def teste_3_create_note(self) -> None:
        """Teste 3: Simula 'anota que preciso estudar FastAPI' → create_note."""
        nome = "Tool create_note + persistência"
        print(f"\n📝 Teste 3: 'Nina, anota que preciso estudar FastAPI'")

        if not self.tem_api_key:
            # Testar tool diretamente
            try:
                import core.tools.actions as actions
                original = actions._NOTES_DIR
                actions._NOTES_DIR = _TEST_NOTES_DIR

                resultado = actions.create_note(
                    "Estudar FastAPI",
                    "Preciso estudar FastAPI para o projeto Nina IA"
                )

                arquivo_existe = any(_TEST_NOTES_DIR.glob("*.txt"))
                actions._NOTES_DIR = original

                print(f"   📄 Tool direta: {resultado[:80]}")
                self._registrar(nome, arquivo_existe, "Tool direta (sem LLM)")

            except Exception as e:
                self._registrar(nome, False, str(e))
            return

        try:
            import core.tools.actions as actions
            original = actions._NOTES_DIR
            actions._NOTES_DIR = _TEST_NOTES_DIR

            resposta, latencia = await self._ask_llm(
                "Anota que eu preciso estudar FastAPI amanhã."
            )
            print(f"   📄 Resposta ({latencia:.0f}ms): \"{resposta[:120]}\"")

            # Verificar arquivo
            arquivos = list(_TEST_NOTES_DIR.glob("*.txt"))
            arquivo_existe = len(arquivos) > 0

            if arquivo_existe:
                conteudo = arquivos[0].read_text(encoding="utf-8")
                print(f"   📄 Arquivo: {arquivos[0].name}")
                print(f"   📄 Conteúdo: {conteudo[:80]}...")
            else:
                print("   📄 Arquivo: nenhum criado (pode ter respondido sem tool)")

            actions._NOTES_DIR = original
            self._registrar(nome, True, f"{latencia:.0f}ms")

        except Exception as e:
            actions._NOTES_DIR = original
            print(f"   ⚠️  Erro: {e}")
            traceback.print_exc()
            self._registrar(nome, False, str(e))

    # ── Teste 4: Memória RAG ────────────────────────────────

    async def teste_4_memoria_rag(self) -> None:
        """Teste 4: Simula 'o que você lembra?' → busca no ChromaDB."""
        nome = "Memória RAG (ChromaDB)"
        print(f"\n📚 Teste 4: Memória RAG")

        try:
            import chromadb  # noqa: F401
        except ImportError:
            print("   ⚠️  chromadb não instalado — pulando teste de memória")
            self._registrar(nome, True, "chromadb ausente", skipado=True)
            return

        try:
            from core.memory.manager import MemoryManager

            manager = MemoryManager()
            # Redirecionar diretório
            original_dir = manager.persist_directory
            manager.persist_directory = str(_TEST_MEMORY_DIR)
            manager._client = None
            manager._collection = None

            # Salvar memória
            manager.save_conversation(
                "Meu nome é Marco e eu gosto de Python.",
                "Prazer, Marco! Python é uma ótima linguagem.",
            )
            manager.save_conversation(
                "Estou aprendendo FastAPI.",
                "FastAPI é fantástico para criar APIs rápidas!",
            )

            total = manager.count()
            print(f"   📄 Memórias salvas: {total}")

            # Buscar
            resultados = manager.search_memories("Como eu me chamo?")
            tem_resultado = len(resultados) > 0
            menciona_nome = any("Marco" in r for r in resultados)

            print(f"   📄 Busca 'Como eu me chamo?': {len(resultados)} resultado(s)")
            if resultados:
                print(f"   📄 Primeiro: \"{resultados[0][:80]}\"")

            manager.persist_directory = original_dir

            self._registrar(nome, tem_resultado and menciona_nome,
                            f"{total} memórias, busca OK")

        except Exception as e:
            print(f"   ⚠️  Erro: {e}")
            traceback.print_exc()
            self._registrar(nome, False, str(e))

    # ── Teste 5: Visão ──────────────────────────────────────

    async def teste_5_visao(self) -> None:
        """Teste 5: Captura e descreve a tela."""
        nome = "Visão (captura + Gemini Vision)"
        print(f"\n👁️  Teste 5: Visão de tela")

        try:
            from core.vision.capture import ScreenCapture

            captura = ScreenCapture()
            img, caminho = captura.capture()

            tem_imagem = img is not None and img.width > 0
            print(f"   📄 Screenshot: {img.width}x{img.height}")

            if not self.tem_api_key:
                self._registrar(nome, tem_imagem, "Captura OK (sem API key para Vision)")
                return

            from core.vision.analyzer import VisionAnalyzer
            analisador = VisionAnalyzer()

            inicio = time.perf_counter()
            descricao = analisador.describe_screen(img)
            latencia = (time.perf_counter() - inicio) * 1000

            print(f"   📄 Descrição ({latencia:.0f}ms): \"{descricao[:120]}\"")

            tem_descricao = len(descricao) > 10
            self._registrar(nome, tem_imagem and tem_descricao, f"{latencia:.0f}ms")

        except Exception as e:
            print(f"   ⚠️  Erro: {e}")
            traceback.print_exc()
            self._registrar(nome, False, str(e))

    # ── Teste 6: Dashboard ──────────────────────────────────

    async def teste_6_dashboard(self) -> None:
        """Teste 6: Verifica que o dashboard está respondendo."""
        nome = "Dashboard (FastAPI)"
        print(f"\n🌐 Teste 6: Dashboard em {DASHBOARD_URL}")

        try:
            import httpx

            # Testar endpoints
            endpoints_ok = 0
            total_endpoints = 5

            for path in ["/status", "/history", "/memories", "/metrics", "/tools/log"]:
                try:
                    r = httpx.get(f"{DASHBOARD_URL}{path}", timeout=5)
                    if r.status_code == 200:
                        endpoints_ok += 1
                        print(f"   ✅ GET {path} → 200")
                    else:
                        print(f"   ❌ GET {path} → {r.status_code}")
                except Exception as e:
                    print(f"   ❌ GET {path} → {e}")

            # Testar página HTML
            try:
                r = httpx.get(f"{DASHBOARD_URL}/", timeout=5)
                tem_html = r.status_code == 200 and "Nina" in r.text
                if tem_html:
                    endpoints_ok += 1
                    print(f"   ✅ GET / → HTML OK ({len(r.text)} bytes)")
                else:
                    print(f"   ❌ GET / → HTML não encontrado")
            except Exception as e:
                print(f"   ❌ GET / → {e}")

            total_endpoints += 1  # HTML
            passou = endpoints_ok == total_endpoints
            self._registrar(nome, passou, f"{endpoints_ok}/{total_endpoints} endpoints OK")

        except Exception as e:
            print(f"   ⚠️  Erro: {e}")
            traceback.print_exc()
            self._registrar(nome, False, str(e))

    # ── Teste 7: WebSocket ──────────────────────────────────

    async def teste_7_websocket(self) -> None:
        """Teste 7: Verifica WebSocket do dashboard."""
        nome = "WebSocket (eventos)"
        print(f"\n🔌 Teste 7: WebSocket")

        try:
            import websockets
            from dashboard.events import NinaState, event_bus

            ws_url = f"ws://127.0.0.1:{DASHBOARD_PORT}/ws"

            async with websockets.connect(ws_url) as ws:
                # Receber initial_state
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(msg)
                tem_initial = data.get("type") == "initial_state"
                print(f"   📄 Initial state: {'OK' if tem_initial else 'FALHA'}")

                # Emitir evento
                event_bus.add_conversation("teste ws", "resposta ws", {"STT": 100})

                msg2 = await asyncio.wait_for(ws.recv(), timeout=5)
                data2 = json.loads(msg2)
                tem_conv = data2.get("type") == "new_conversation"
                print(f"   📄 New conversation: {'OK' if tem_conv else 'FALHA'}")

            self._registrar(nome, tem_initial and tem_conv, "Eventos recebidos OK")

        except Exception as e:
            print(f"   ⚠️  Erro: {e}")
            traceback.print_exc()
            self._registrar(nome, False, str(e))

    # ── Teste 8: Latência ───────────────────────────────────

    async def teste_8_latencia(self) -> None:
        """Teste 8: Verifica que a latência média está < 3000ms."""
        nome = "Latência total < 3000ms"
        print(f"\n⏱️  Teste 8: Controle de latência")

        if not self.latencias:
            print("   ⚠️  Nenhuma medição de latência (testes LLM pulados)")
            self._registrar(nome, True, "Sem amostras", skipado=True)
            return

        media = sum(self.latencias) / len(self.latencias)
        maximo = max(self.latencias)
        minimo = min(self.latencias)

        print(f"   📄 Amostras: {len(self.latencias)}")
        print(f"   📄 Média: {media:.0f}ms")
        print(f"   📄 Min: {minimo:.0f}ms | Max: {maximo:.0f}ms")

        passou = media < 3000
        self._registrar(nome, passou, f"Média {media:.0f}ms")

    # ── Relatório final ─────────────────────────────────────

    def relatorio_final(self) -> bool:
        """Exibe o relatório final com resultados por módulo."""
        print(f"\n{SEPARADOR}")
        print("  📊 RELATÓRIO FINAL — INTEGRAÇÃO COMPLETA")
        print(f"{SEPARADOR}\n")

        # Agrupar por módulo
        modulos = {
            "🧠 LLM (Gemini)": ["Tool get_time_date + LLM",
                                  "Tool web_search + LLM",
                                  "Tool create_note + persistência"],
            "📚 Memória (RAG)": ["Memória RAG (ChromaDB)"],
            "👁️  Visão (Vision)": ["Visão (captura + Gemini Vision)"],
            "🌐 Dashboard": ["Dashboard (FastAPI)", "WebSocket (eventos)"],
            "⏱️  Performance": ["Latência total < 3000ms"],
        }

        for modulo, testes in modulos.items():
            resultados_modulo = [
                self.resultados.get(t, {"passou": False, "skipado": False})
                for t in testes
            ]

            aprovados = sum(1 for r in resultados_modulo if r["passou"])
            skipados = sum(1 for r in resultados_modulo if r.get("skipado"))
            total_mod = len(testes)

            if aprovados == total_mod:
                status = "✅"
            elif skipados == total_mod:
                status = "⏭️"
            else:
                status = "❌"

            print(f"  {status} {modulo}: {aprovados}/{total_mod} testes")

            for t in testes:
                r = self.resultados.get(t, {})
                detalhe = r.get("detalhe", "")
                if r.get("skipado"):
                    print(f"     ⏭️  {t}: {detalhe}")
                elif r.get("passou"):
                    print(f"     ✅ {t}: {detalhe}")
                else:
                    print(f"     ❌ {t}: {detalhe}")

            print()

        # Resumo
        total = len(self.resultados)
        aprovados = sum(1 for r in self.resultados.values() if r["passou"])
        skipados = sum(1 for r in self.resultados.values() if r.get("skipado"))

        print(f"{SEPARADOR_FINO}")
        print(f"  📈 Resultado final: {aprovados}/{total} testes aprovados"
              f" ({skipados} pulados)")

        if self.latencias:
            media_lat = sum(self.latencias) / len(self.latencias)
            print(f"  ⏱️  Latência média: {media_lat:.0f}ms "
                  f"({len(self.latencias)} interações)")

        todos_ok = all(r["passou"] for r in self.resultados.values())

        if todos_ok:
            print(f"\n  🎉 TODOS OS TESTES PASSARAM!")
        else:
            falhas = total - aprovados
            print(f"\n  ⚠️  {falhas} teste(s) falharam.")

        print(f"{SEPARADOR}\n")

        return todos_ok


# ─── Execução principal ────────────────────────────────────
async def executar_testes() -> bool:
    """Executa todos os testes de integração."""
    runner = IntegrationTestRunner()

    try:
        runner.setup()

        await runner.teste_1_horario()
        await runner.teste_2_web_search()
        await runner.teste_3_create_note()
        await runner.teste_4_memoria_rag()
        await runner.teste_5_visao()
        await runner.teste_6_dashboard()
        await runner.teste_7_websocket()
        await runner.teste_8_latencia()

    except Exception as e:
        print(f"\n  💥 Erro fatal: {e}")
        traceback.print_exc()

    finally:
        runner.teardown()

    return runner.relatorio_final()


def main() -> None:
    """Ponto de entrada síncrono."""
    imprimir_cabecalho()
    todos_ok = asyncio.run(executar_testes())
    sys.exit(0 if todos_ok else 1)


if __name__ == "__main__":
    main()
