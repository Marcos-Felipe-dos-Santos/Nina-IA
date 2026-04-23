"""
core.utils.config_loader
========================
Carrega e fornece acesso ao arquivo config.yaml centralizado.
Garante que nunca haja hardcode de parâmetros no código.

Variáveis sensíveis (API keys) devem ficar no arquivo .env na raiz
do projeto e são carregadas automaticamente via python-dotenv.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


# Caminho raiz do projeto (2 níveis acima de core/utils/)
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH: Path = _PROJECT_ROOT / "config.yaml"

# Carrega variáveis do .env para os.environ (se o arquivo existir)
_ENV_PATH: Path = _PROJECT_ROOT / ".env"
load_dotenv(_ENV_PATH)


# Cache do config para evitar I/O repetido
_cached_config: Optional[Dict[str, Any]] = None
_cached_config_path: Optional[Path] = None


def carregar_config(caminho: Optional[Path] = None) -> Dict[str, Any]:
    """Carrega o arquivo YAML de configuração (com cache em memória).

    Na primeira chamada, lê do disco e armazena em cache.
    Chamadas subsequentes com o mesmo caminho retornam o cache.

    Args:
        caminho: Caminho absoluto ou relativo ao config.yaml.
                 Se None, usa o padrão na raiz do projeto.

    Returns:
        Dicionário com todas as configurações.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        yaml.YAMLError: Se o YAML for inválido.
    """
    global _cached_config, _cached_config_path

    config_path = Path(caminho) if caminho else _DEFAULT_CONFIG_PATH

    if _cached_config is not None and _cached_config_path == config_path:
        return _cached_config

    if not config_path.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    _cached_config = config
    _cached_config_path = config_path

    return config


def obter_secao(secao: str, caminho: Optional[Path] = None) -> Dict[str, Any]:
    """Retorna uma seção específica do config.yaml.

    Args:
        secao: Nome da seção (ex: 'stt', 'tts', 'vad').
        caminho: Caminho opcional ao config.yaml.

    Returns:
        Dicionário da seção solicitada.

    Raises:
        KeyError: Se a seção não existir.
    """
    config = carregar_config(caminho)

    if secao not in config:
        raise KeyError(
            f"Seção '{secao}' não encontrada em config.yaml. "
            f"Seções disponíveis: {list(config.keys())}"
        )

    return config[secao]
