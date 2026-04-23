"""
core.tools.actions
==================
Implementação das ferramentas (tools) que a Nina pode executar.
Cada função é auto-documentada com docstrings e type hints
para que o Gemini entenda quando e como usá-la.
"""

import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from core.utils.config_loader import carregar_config

logger = logging.getLogger(__name__)

# Diretório base para notas
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_NOTES_DIR = _PROJECT_ROOT / "data" / "notes"


def open_app(app_name: str) -> str:
    """Abre um aplicativo no computador pelo nome.

    Args:
        app_name: Nome do aplicativo a abrir, por exemplo:
                  'vscode', 'navegador', 'chrome', 'notepad',
                  'calculadora', 'explorer', 'terminal'.

    Returns:
        Mensagem confirmando se o aplicativo foi aberto ou se houve erro.
    """
    # Mapeamento de nomes comuns para comandos Windows
    app_map = {
        "vscode": "code",
        "vs code": "code",
        "visual studio code": "code",
        "navegador": "start https://www.google.com",
        "browser": "start https://www.google.com",
        "chrome": "start chrome",
        "google chrome": "start chrome",
        "firefox": "start firefox",
        "edge": "start msedge",
        "notepad": "notepad",
        "bloco de notas": "notepad",
        "calculadora": "calc",
        "calculator": "calc",
        "explorer": "explorer",
        "explorador": "explorer",
        "terminal": "wt",
        "cmd": "cmd",
        "powershell": "powershell",
        "spotify": "start spotify:",
        "paint": "mspaint",
    }

    # Normalizar nome
    nome_normalizado = app_name.strip().lower()

    # Buscar no mapeamento
    comando = app_map.get(nome_normalizado)

    if comando is None:
        # Tentar abrir diretamente pelo nome
        comando = nome_normalizado
        logger.info(f"App '{app_name}' não mapeado, tentando diretamente: {comando}")

    try:
        if sys.platform == "win32":
            # No Windows, usar 'start' para apps que precisam
            if comando.startswith("start "):
                os.system(comando)
            else:
                subprocess.Popen(
                    comando,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        else:
            subprocess.Popen(
                comando,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        logger.info(f"Aplicativo '{app_name}' aberto com sucesso.")
        return f"Aplicativo '{app_name}' aberto com sucesso."

    except Exception as e:
        logger.error(f"Erro ao abrir '{app_name}': {e}")
        return f"Erro ao abrir '{app_name}': {e}"


def web_search(query: str) -> str:
    """Busca na internet usando DuckDuckGo e retorna os 3 primeiros resultados.

    Args:
        query: Termo de busca, por exemplo: 'clima em São Paulo hoje'.

    Returns:
        Texto com os 3 primeiros resultados formatados com título, link e resumo.
    """
    try:
        from ddgs import DDGS

        logger.info(f"Buscando na web: '{query}'")

        with DDGS() as ddgs:
            resultados = list(ddgs.text(query, max_results=3))

        if not resultados:
            return f"Nenhum resultado encontrado para: '{query}'"

        texto = f"Resultados para '{query}':\n\n"
        for i, r in enumerate(resultados, 1):
            titulo = r.get("title", "Sem título")
            link = r.get("href", "")
            resumo = r.get("body", "Sem descrição")
            texto += f"{i}. {titulo}\n   {link}\n   {resumo}\n\n"

        logger.info(f"Busca concluída: {len(resultados)} resultados.")
        return texto.strip()

    except Exception as e:
        logger.error(f"Erro na busca web: {e}")
        return f"Erro ao buscar na web: {e}"


def create_note(title: str, content: str) -> str:
    """Cria uma nota de texto salva em arquivo no disco.

    Args:
        title: Título da nota (será usado como nome do arquivo).
        content: Conteúdo da nota.

    Returns:
        Mensagem confirmando a criação da nota com o caminho do arquivo.
    """
    try:
        # Garantir que o diretório existe
        _NOTES_DIR.mkdir(parents=True, exist_ok=True)

        # Sanitizar título para nome de arquivo
        nome_seguro = "".join(
            c if c.isalnum() or c in " _-" else "_" for c in title
        ).strip()
        nome_seguro = nome_seguro.replace(" ", "_")

        # Adicionar timestamp para evitar duplicatas
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_arquivo = f"{nome_seguro}_{timestamp}.txt"

        caminho = _NOTES_DIR / nome_arquivo

        # Escrever conteúdo
        with open(caminho, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n")
            f.write(f"# Criado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"# ---\n\n")
            f.write(content)

        logger.info(f"Nota criada: {caminho}")
        return f"Nota '{title}' criada com sucesso em: {caminho}"

    except Exception as e:
        logger.error(f"Erro ao criar nota: {e}")
        return f"Erro ao criar nota: {e}"


def get_time_date() -> str:
    """Retorna a hora e data atual formatada em português do Brasil.

    Returns:
        String com a data e hora atual, por exemplo:
        'Agora são 14:30 de quarta-feira, 16 de abril de 2025.'
    """
    agora = datetime.now()

    # Nomes dos dias da semana em português
    dias_semana = [
        "segunda-feira", "terça-feira", "quarta-feira",
        "quinta-feira", "sexta-feira", "sábado", "domingo",
    ]

    # Nomes dos meses em português
    meses = [
        "janeiro", "fevereiro", "março", "abril", "maio", "junho",
        "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
    ]

    dia_semana = dias_semana[agora.weekday()]
    mes = meses[agora.month - 1]

    resultado = (
        f"Agora são {agora.strftime('%H:%M')} de {dia_semana}, "
        f"{agora.day} de {mes} de {agora.year}."
    )

    logger.info(f"Data/hora consultada: {resultado}")
    return resultado


def list_notes() -> str:
    """Lista todas as notas salvas no diretório de notas.

    Returns:
        Texto com a lista de notas encontradas, incluindo nome e data,
        ou mensagem informando que não há notas.
    """
    try:
        if not _NOTES_DIR.exists():
            return "Nenhuma nota encontrada. O diretório de notas ainda não foi criado."

        arquivos = sorted(_NOTES_DIR.glob("*.txt"))

        if not arquivos:
            return "Nenhuma nota encontrada."

        texto = f"Notas salvas ({len(arquivos)}):\n\n"
        for i, arq in enumerate(arquivos, 1):
            # Obter data de modificação
            mod_time = datetime.fromtimestamp(arq.stat().st_mtime)
            data_fmt = mod_time.strftime("%d/%m/%Y %H:%M")

            # Ler primeira linha para pegar o título
            try:
                with open(arq, "r", encoding="utf-8") as f:
                    primeira_linha = f.readline().strip().lstrip("# ")
            except Exception:
                primeira_linha = arq.stem

            texto += f"{i}. {primeira_linha} ({data_fmt})\n"
            texto += f"   Arquivo: {arq.name}\n\n"

        logger.info(f"Listadas {len(arquivos)} notas.")
        return texto.strip()

    except Exception as e:
        logger.error(f"Erro ao listar notas: {e}")
        return f"Erro ao listar notas: {e}"


def look_at_screen() -> str:
    """Captura e descreve o que está visível na tela do computador agora.

    Returns:
        Descrição do conteúdo atual da tela em português do Brasil.
    """
    try:
        from core.vision.capture import ScreenCapture
        from core.vision.analyzer import VisionAnalyzer

        logger.info("Capturando e analisando a tela...")

        captura = ScreenCapture()
        analisador = VisionAnalyzer()

        # Capturar screenshot
        img, caminho = captura.capture()
        print(f"📸 Screenshot capturado: {caminho}")

        # Analisar com Gemini Vision
        descricao = analisador.describe_screen(img)

        logger.info(f"Descrição da tela: '{descricao[:80]}'")
        return descricao

    except Exception as e:
        logger.error(f"Erro ao analisar a tela: {e}")
        return f"Erro ao analisar a tela: {e}"


def change_expression(emotion: str) -> str:
    """Altera a expressão facial da Nina no VTube Studio.

    Args:
        emotion: Emoção válida a ser exibida (alegria, tristeza, surpresa, raiva, neutro).

    Returns:
        Mensagem confirmando a mudança ou aviso que o avatar está desabilitado.
    """
    from core.utils.config_loader import carregar_config
    config = carregar_config()
    
    if not config.get("avatar", {}).get("enabled", False):
        return "Avatar desabilitado no config.yaml"

    try:
        from core.avatar.vtube import get_global_vtube
        vtube = get_global_vtube()
        if vtube and vtube.connected:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(vtube.trigger_expression(emotion))
            except RuntimeError:
                asyncio.run(vtube.trigger_expression(emotion))
                
            return f"Expressão '{emotion}' ativada com sucesso no VTube Studio."
        
        return "VTube Studio não conectado, mas emoção anotada."
    except ImportError:
        return "Falha ao carregar componente do VTube Studio."


