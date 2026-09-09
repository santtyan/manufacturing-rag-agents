"""Record-replay para respostas externas (Ollama, Postgres) durante execucao
de agente (Fase 6 do plano de disciplina experimental do grupo PDC).

Por que este arquivo existe: o checkpointer do LangGraph ja da replay de
ESTADO de graca (qualquer prefixo do grafo pode ser retomado a partir de um
checkpoint) -- mas nao registra as RESPOSTAS EXTERNAS que produziram aquele
estado (o texto que o Ollama devolveu, o resultado que o Postgres devolveu).
Sem isso, reexecutar um agente para testar uma mudanca de prompt tambem
muda as respostas do LLM em si -- impossivel isolar "o que mudou por causa
do prompt novo" de "o que mudou porque o LLM e nao-deterministico ou os
dados do banco mudaram entre as duas execucoes".

Este modulo grava (modo RECORD) as respostas reais de chamar_ollama() e de
execucao de SQL durante uma execucao, indexadas pelo prompt/query exato
(hash), e no modo REPLAY devolve a resposta gravada em vez de fazer a
chamada real -- permitindo comparar duas versoes de PROMPT/logica contra as
MESMAS observacoes externas, isolando o que de fato mudou.

Uso tipico: gravar uma vez com --record, depois rodar variantes do agente
(prompt B, teto de passos diferente, etc) com --replay usando a mesma
fita -- qualquer chamada que bata exatamente com uma ja gravada volta
instantaneamente sem tocar Ollama/Postgres; qualquer chamada NOVA (o
prompt mudou de verdade) fica visivel como "MISS" no replay, o sinal de que
aquele ponto do fluxo realmente mudou de comportamento.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

DIR_FITAS = Path(__file__).resolve().parent.parent / "eval" / "fitas_replay"


def _hash_chamada(*args: Any, **kwargs: Any) -> str:
    """Chave estavel para uma chamada (prompt+modelo+... ou sql), usada para
    casar a mesma chamada entre RECORD e REPLAY."""
    bruto = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.sha256(bruto.encode("utf-8")).hexdigest()[:16]


@dataclass
class Fita:
    """Uma fita de record-replay: mapa hash_da_chamada -> resposta gravada.
    Serializada em JSON simples (nao JSONL, porque e um dicionario indexado
    por chave, nao uma sequencia de eventos como o Trace)."""

    caminho: Path
    modo: str  # "record" ou "replay"
    dados: dict[str, Any] = field(default_factory=dict)
    misses: list[str] = field(default_factory=list)  # chamadas nao encontradas no replay

    @classmethod
    def abrir(cls, nome: str, modo: str) -> "Fita":
        caminho = DIR_FITAS / f"{nome}.json"
        dados = {}
        if modo == "replay":
            if not caminho.exists():
                raise FileNotFoundError(f"Fita de replay nao encontrada: {caminho}. Grave primeiro com modo=record.")
            dados = json.loads(caminho.read_text(encoding="utf-8"))
        return cls(caminho=caminho, modo=modo, dados=dados)

    def salvar(self) -> None:
        if self.modo != "record":
            return
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self.caminho.write_text(json.dumps(self.dados, ensure_ascii=False, indent=2), encoding="utf-8")

    def interceptar(self, chave: str, chamada_real: Callable[[], Any], serializar: Callable[[Any], Any],
                     desserializar: Callable[[Any], Any]) -> Any:
        """Nucleo do record-replay: em modo record, executa a chamada real e
        grava o resultado serializado; em modo replay, devolve o resultado
        gravado sem executar a chamada real -- ou registra um MISS e executa
        a chamada real mesmo assim (fail-open: nunca trava um experimento
        so porque uma chamada nova apareceu, mas o MISS fica visivel no
        relatorio para o pesquisador decidir se isso invalida a comparacao)."""
        if self.modo == "record":
            resultado = chamada_real()
            self.dados[chave] = serializar(resultado)
            return resultado

        # modo replay
        if chave in self.dados:
            return desserializar(self.dados[chave])
        self.misses.append(chave)
        return chamada_real()


def chamar_ollama_com_fita(fita: Fita, prompt: str, modelo: str, **kwargs: Any) -> str:
    """Wrapper de shared.ollama_client.chamar() com record-replay. Serializa
    so o texto de resposta (nao os metadados de custo/timing -- esses nao
    fazem sentido "replayar", contabilizar de novo distorceria o custo real
    de uma execucao de replay, que deveria ser ~0)."""
    from shared.ollama_client import chamar

    chave = _hash_chamada("ollama", prompt=prompt, modelo=modelo, **kwargs)
    return fita.interceptar(
        chave,
        chamada_real=lambda: str(chamar(prompt, modelo=modelo, **kwargs)),
        serializar=lambda r: r,
        desserializar=lambda r: r,
    )


def executar_sql_com_fita(fita: Fita, sql: str, executar_real: Callable[[str], Any]) -> Any:
    """Wrapper de nl_to_sql._executar() com record-replay. Serializa o
    DataFrame como registros JSON-compativeis (to_dict) e desserializa de
    volta via pandas.DataFrame -- suficiente para o proposito de comparar
    comportamento do agente, nao para preservar tipos de coluna exatos."""
    import pandas as pd

    chave = _hash_chamada("sql", sql=sql)
    return fita.interceptar(
        chave,
        chamada_real=lambda: executar_real(sql),
        serializar=lambda df: df.to_dict(orient="records"),
        desserializar=lambda registros: pd.DataFrame(registros),
    )


def relatorio_replay(fita: Fita) -> str:
    """Resumo de quantas chamadas bateram com a fita (replay puro) vs
    quantas foram MISS (executaram de verdade porque nao existiam na fita)
    -- o sinal de que algo realmente mudou de comportamento entre as duas
    execucoes comparadas."""
    total_na_fita = len(fita.dados)
    n_misses = len(fita.misses)
    linhas = [
        f"Fita: {fita.caminho}",
        f"Modo: {fita.modo}",
        f"Chamadas na fita: {total_na_fita}",
    ]
    if fita.modo == "replay":
        linhas.append(f"Misses (executaram de verdade, nao estavam na fita): {n_misses}")
        if n_misses:
            linhas.append("  -> isso significa que o comportamento realmente mudou nesses pontos "
                           "(prompt diferente, tool diferente, ou dado externo diferente) -- "
                           "nao e um erro, e o sinal que o replay existe para revelar.")
    return "\n".join(linhas)
