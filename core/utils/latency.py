"""
core.utils.latency
==================
Rastreador de latência para o pipeline de voz da Nina IA.
Mede e exibe no terminal os tempos de cada etapa:
  [STT] 320ms | [LLM] --ms | [TTS] 210ms | [TOTAL] 530ms
"""

import time
from typing import Dict, Optional


class LatencyTracker:
    """Mede a latência de cada etapa do pipeline de voz.

    Uso:
        tracker = LatencyTracker()
        tracker.iniciar("STT")
        # ... processamento STT ...
        tracker.finalizar("STT")
        tracker.exibir()
    """

    # Etapas do pipeline na ordem de exibição
    _ETAPAS_PADRAO = ["STT", "LLM", "TTS"]

    def __init__(self) -> None:
        """Inicializa o rastreador com dicionários vazios."""
        self._inicio: Dict[str, float] = {}
        self._latencias: Dict[str, Optional[float]] = {
            etapa: None for etapa in self._ETAPAS_PADRAO
        }

    def iniciar(self, etapa: str) -> None:
        """Marca o início da medição de uma etapa.

        Args:
            etapa: Nome da etapa (ex: 'STT', 'LLM', 'TTS').
        """
        self._inicio[etapa] = time.perf_counter()

    def finalizar(self, etapa: str) -> float:
        """Marca o fim da medição e calcula a latência em milissegundos.

        Args:
            etapa: Nome da etapa que foi iniciada com `iniciar()`.

        Returns:
            Latência em milissegundos.

        Raises:
            ValueError: Se `iniciar()` não foi chamado para essa etapa.
        """
        if etapa not in self._inicio:
            raise ValueError(
                f"Etapa '{etapa}' não foi iniciada. "
                f"Chame tracker.iniciar('{etapa}') primeiro."
            )

        fim = time.perf_counter()
        latencia_ms = (fim - self._inicio[etapa]) * 1000
        self._latencias[etapa] = latencia_ms
        return latencia_ms

    def obter_latencia(self, etapa: str) -> Optional[float]:
        """Retorna a latência de uma etapa em milissegundos.

        Args:
            etapa: Nome da etapa.

        Returns:
            Latência em ms, ou None se a etapa não foi medida.
        """
        return self._latencias.get(etapa)

    def obter_total(self) -> Optional[float]:
        """Retorna a soma de todas as latências medidas.

        Returns:
            Total em ms, ou None se nenhuma etapa foi medida.
        """
        medidas = [v for v in self._latencias.values() if v is not None]
        return sum(medidas) if medidas else None

    def formatar(self) -> str:
        """Formata a linha de latência para exibição no terminal.

        Returns:
            String formatada, ex:
            [STT] 320ms | [LLM] --ms | [TTS] 210ms | [TOTAL] 530ms
        """
        partes = []
        for etapa in self._ETAPAS_PADRAO:
            latencia = self._latencias.get(etapa)
            if latencia is not None:
                partes.append(f"[{etapa}] {latencia:.0f}ms")
            else:
                partes.append(f"[{etapa}] --ms")

        total = self.obter_total()
        if total is not None:
            partes.append(f"[TOTAL] {total:.0f}ms")
        else:
            partes.append("[TOTAL] --ms")

        return " | ".join(partes)

    def exibir(self) -> None:
        """Imprime a linha de latência formatada no terminal."""
        print(f"\n⏱  {self.formatar()}")

    def resetar(self) -> None:
        """Reseta todas as medições para um novo ciclo."""
        self._inicio.clear()
        self._latencias = {etapa: None for etapa in self._ETAPAS_PADRAO}
