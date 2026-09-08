"""Esquema de trace para execucoes de agente (Fase 1 do plano de disciplina
experimental do grupo PDC "Harness Engineering").

Por que este arquivo existe: o grupo de pesquisa PDC define que "o ativo e o
trace" - o esquema de como registrar uma execucao vem antes do harness em si
e sobrevive a troca de modelo/framework/equipe. Antes deste modulo, o Harbor
nao produzia trace algum: 13 pontos de chamada ao Ollama descartavam a
resposta com `.get("response")`, jogando fora `eval_count`/`prompt_eval_count`
(tokens) que o proprio Ollama ja retorna.

Decisao de design (pesquisa registrada no plano de implementacao, nao repetida
aqui): nem OpenTelemetry GenAI puro, nem schema livre.

- Os nomes de campo de `Passo` espelham deliberadamente a convencao
  OpenTelemetry GenAI (`gen_ai.request.model`, `gen_ai.usage.input_tokens`,
  etc, aqui como `gen_ai_request_model`, `gen_ai_usage_input_tokens` - sem
  ponto porque dataclass nao aceita ponto em nome de campo) para ficar
  migravel para um exportador OTel real no futuro sem reescrever a captura.
- Mas o survey "From Agent Traces to Trust" (arXiv:2606.04990) mostra que
  OTel puro nao basta para pesquisa em agentes: falta a camada semantica de
  evidencia/proveniencia entre passos. Por isso `Passo` carrega `evidencias`
  (o que foi recuperado/consultado) e `relacoes` (SUPPORT/DERIVE/DEPEND_ON/
  CONTRADICT/INVALIDATE entre passos) - a parte que OTel nao cobre.
- O relatorio de replicacao (Rollout Card, arXiv:2605.12131) fica em modulo
  separado (`shared/rollout_card.py`), porque resolve um problema diferente
  do trace: o trace registra UMA execucao, o Rollout Card resume uma COLECAO
  de execucoes para consumo humano.

Regra permanente do projeto (registrada tambem na skill padrao-react-
raciocinio-acao e na memoria de projeto do grupo PDC): toda replicacao ou
experimento com agente entrega tres artefatos - codigo, trace, relatorio de
replicacao. Nunca menos.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

TIPO_PASSO = Literal["llm", "tool", "retrieval"]
TIPO_RELACAO = Literal["SUPPORT", "DERIVE", "DEPEND_ON", "CONTRADICT", "INVALIDATE"]

DIR_TRACES = Path(__file__).resolve().parent.parent / "eval" / "traces"


@dataclass
class Relacao:
    """Uma relacao de proveniencia entre dois pontos do trace.

    Achado real ao pesquisar (2026-09-08): o survey de Agent Traces mostra que
    e exatamente essa camada semantica - nao os spans em si - que falta em
    instrumentacao OTel generica para permitir auditoria/debug de um agente.
    """

    tipo: TIPO_RELACAO
    de: str  # id do passo/evidencia de origem
    para: str  # id do passo/evidencia de destino
    detalhe: str = ""


@dataclass
class Passo:
    """Um passo dentro de uma execucao de agente (Thought/Action/Observation
    ou uma chamada de tool/retrieval). Nomes de campo `gen_ai_*` espelham a
    convencao OpenTelemetry GenAI (ver docstring do modulo)."""

    indice: int
    tipo: TIPO_PASSO
    thought: str = ""
    action: str = ""
    action_input: str = ""
    observation: str = ""
    duracao_s: float = 0.0

    # Espelha gen_ai.* (OpenTelemetry GenAI semantic conventions)
    gen_ai_operation_name: str = ""
    gen_ai_request_model: str = ""
    gen_ai_response_model: str = ""
    gen_ai_usage_input_tokens: int = 0
    gen_ai_usage_output_tokens: int = 0

    # Camada semantica (nao coberta por OTel puro, ver docstring do modulo)
    evidencias: list[str] = field(default_factory=list)
    relacoes: list[Relacao] = field(default_factory=list)


@dataclass
class Trace:
    """Uma execucao completa de agente, serializavel em JSONL (uma linha por
    execucao). Ver `iniciar_trace()` para o context manager de uso normal."""

    trace_id: str
    tarefa: str
    entrada: str
    timestamp_inicio: str
    duracao_total_s: float = 0.0
    resultado_final: str = ""
    sucesso: bool = False
    modelo: str = ""
    versoes: dict[str, str] = field(default_factory=dict)
    passos: list[Passo] = field(default_factory=list)

    @property
    def custo_total(self) -> dict[str, int]:
        return {
            "input_tokens": sum(p.gen_ai_usage_input_tokens for p in self.passos),
            "output_tokens": sum(p.gen_ai_usage_output_tokens for p in self.passos),
        }

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["custo_total"] = self.custo_total
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def salvar(self, caminho: Path) -> None:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        with open(caminho, "a", encoding="utf-8") as f:
            f.write(self.to_json() + "\n")


@contextmanager
def iniciar_trace(
    tarefa: str,
    entrada: str,
    modelo: str = "",
    versoes: dict[str, str] | None = None,
    arquivo: Path | str | None = None,
) -> Iterator[Trace]:
    """Context manager que acumula passos e serializa o trace no __exit__,
    mesmo se a execucao lancar excecao (fica sucesso=False, mas o trace de
    uma falha real e o dado mais importante para debug - nunca descartar).

    Uso:
        with iniciar_trace("nl_to_sql", pergunta, modelo="qwen2.5:7b") as tr:
            registrar_passo(tr, tipo="llm", action="gerar_sql", ...)
            tr.resultado_final = resultado
            tr.sucesso = True
    """
    inicio = time.monotonic()
    trace = Trace(
        trace_id=str(uuid.uuid4()),
        tarefa=tarefa,
        entrada=entrada,
        timestamp_inicio=datetime.now(timezone.utc).isoformat(),
        modelo=modelo,
        versoes=versoes or {},
    )
    caminho = Path(arquivo) if arquivo else DIR_TRACES / f"{tarefa}.jsonl"
    try:
        yield trace
    finally:
        trace.duracao_total_s = time.monotonic() - inicio
        trace.salvar(caminho)


def registrar_passo(trace: Trace, **kwargs: Any) -> Passo:
    """Cria um Passo, atribui o proximo indice automaticamente e anexa ao
    trace. Repassar campos de Passo como kwargs (thought=, action=, etc)."""
    passo = Passo(indice=len(trace.passos), **kwargs)
    trace.passos.append(passo)
    return passo


def carregar_traces(caminho: Path | str) -> list[dict[str, Any]]:
    """Le um arquivo JSONL de traces de volta em lista de dicts. Usado tanto
    por testes de round-trip quanto pelo gerador de Rollout Card."""
    caminho = Path(caminho)
    if not caminho.exists():
        return []
    linhas = caminho.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(linha) for linha in linhas if linha.strip()]
