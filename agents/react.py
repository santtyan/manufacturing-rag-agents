"""Agente ReAct generico em LangGraph, com trace completo por execucao (Fase 5
do plano de disciplina experimental do grupo PDC "Harness Engineering").

Por que este arquivo existe: a primeira tarefa dada ao grupo de pesquisa
PDC (2026-09-08) e replicar um agente ReAct simples sobre uma tarefa de
benchmark e entregar o trace. O ciclo Thought->Action->Observation (Yao et
al., ICLR 2023, arXiv:2210.03629) e a base teorica ja documentada na skill
pessoal padrao-react-raciocinio-acao; este modulo e a implementacao real.

Decisao LangGraph vs Python puro (registrada no plano): LangGraph, porque o
checkpointer da replay de prefixo de graca (metade do record-replay da Fase
6), a avaliacao recomendada de agente e em nivel de grafo (nao so resposta
final), e paga junto o item 2 do roadmap de migrar-para-langchain
(self-repair -> critic node). O checkpointer NAO substitui o trace de
shared/trace.py -- ele nao registra custo, versao de modelo nem latencia por
passo, so o estado em si.

Duas tarefas usam este nucleo (ver agents/tarefa_hotpotqa.py e
agents/tarefa_sql.py):
- HotpotQA: replicacao literal do paper original (tool de busca Wikipedia).
- NL-to-SQL: aplicacao do mesmo ciclo ao self-repair ja existente do Harbor
  (gerar SQL = Action, executar no Postgres = Observation).

Regra permanente do projeto: toda replicacao de agente entrega TRES
artefatos -- codigo (este modulo), trace (JSONL via shared/trace.py),
relatorio de replicacao (Rollout Card via shared/rollout_card.py). Nunca
menos.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from shared.ollama_client import chamar as chamar_ollama
from shared.trace import Trace, registrar_passo


MAX_PASSOS = 6  # teto de iteracoes -- estudos de agentes mostram loop sem teto como falha comum


@dataclass
class Tool:
    """Uma acao que o agente pode executar. `nome` e o que aparece no prompt
    ReAct como "Action: <nome>[<input>]"; `funcao` recebe o texto do input e
    devolve a observacao como string."""

    nome: str
    descricao: str
    funcao: Callable[[str], str]


class EstadoReAct(TypedDict, total=False):
    tarefa: str
    entrada: str
    modelo: str
    historico: str  # transcript acumulado de Thought/Action/Observation
    resposta_final: str
    finalizado: bool
    n_passos: int
    trace: Trace
    _ultima_resposta: str  # ponte entre o node pensar_agir e o node observar


_PADRAO_ACAO = re.compile(r"Action:\s*(\w+)\[(.*)\]", re.DOTALL)
_PADRAO_FINALIZAR = re.compile(r"Action:\s*finalizar\[(.*)\]", re.DOTALL | re.IGNORECASE)


def _montar_prompt(tools: list[Tool], estado: EstadoReAct) -> str:
    lista_tools = "\n".join(f"- {t.nome}[entrada]: {t.descricao}" for t in tools)
    return f"""Voce e um agente que resolve uma tarefa intercalando Thought (raciocinio) e
Action (acao real sobre o mundo), no padrao ReAct (Yao et al. 2023). NUNCA invente uma
Observation -- ela sempre vem de uma Action real que voce executa.

Ferramentas disponiveis:
{lista_tools}
- finalizar[resposta]: use quando tiver informacao suficiente para responder a tarefa.

Formato OBRIGATORIO de cada passo (uma linha Thought, uma linha Action, nada mais):
Thought: <seu raciocinio sobre o proximo passo>
Action: <nome_da_tool>[<entrada>]

Tarefa: {estado['entrada']}

Historico ate agora:
{estado.get('historico') or '(inicio)'}

Gere APENAS o proximo Thought e a proxima Action (uma linha cada, nada depois)."""


def _no_pensar_agir(tools_por_nome: dict[str, Tool], tools: list[Tool]):
    """Fabrica o node LangGraph que gera Thought+Action via LLM e retorna o
    texto cru -- a extracao/execucao da Action acontece no node seguinte
    (observar), para manter cada node fazendo uma coisa so (mais facil de
    tracear e testar isoladamente)."""

    def node(estado: EstadoReAct) -> dict[str, Any]:
        prompt = _montar_prompt(tools, estado)
        inicio = time.monotonic()
        resposta = chamar_ollama(prompt, modelo=estado["modelo"], temperature=0.1)
        duracao = time.monotonic() - inicio

        trace = estado["trace"]
        passo = registrar_passo(
            trace,
            tipo="llm",
            gen_ai_operation_name="chat",
            gen_ai_request_model=estado["modelo"],
            duracao_s=duracao,
        )
        if hasattr(resposta, "metadados"):
            passo.gen_ai_response_model = resposta.metadados.gen_ai_response_model
            passo.gen_ai_usage_input_tokens = resposta.metadados.gen_ai_usage_input_tokens
            passo.gen_ai_usage_output_tokens = resposta.metadados.gen_ai_usage_output_tokens

        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", resposta, re.DOTALL)
        passo.thought = thought_match.group(1).strip() if thought_match else ""

        return {"historico": estado.get("historico", "") + "\n" + resposta.strip(), "_ultima_resposta": resposta}

    return node


def _no_observar(tools_por_nome: dict[str, Tool]):
    """Extrai a Action da ultima resposta do LLM, executa a tool real (ou
    finaliza), e anexa a Observation real ao historico -- e este passo que
    fundamenta o raciocinio em algo do mundo externo, o ponto central do
    ReAct para reduzir alucinacao (ver padrao-react-raciocinio-acao)."""

    def node(estado: EstadoReAct) -> dict[str, Any]:
        resposta = estado.get("_ultima_resposta", "")
        trace = estado["trace"]
        n_passos = estado.get("n_passos", 0) + 1

        finalizar_match = _PADRAO_FINALIZAR.search(resposta)
        if finalizar_match:
            resultado = finalizar_match.group(1).strip()
            if trace.passos:
                trace.passos[-1].action = "finalizar"
                trace.passos[-1].action_input = resultado
                trace.passos[-1].observation = "(finalizado)"
            return {"resposta_final": resultado, "finalizado": True, "n_passos": n_passos}

        acao_match = _PADRAO_ACAO.search(resposta)
        if not acao_match:
            observacao = "Formato invalido: nenhuma Action reconhecida. Use Action: nome[entrada]."
            if trace.passos:
                trace.passos[-1].observation = observacao
            historico_novo = estado.get("historico", "") + f"\nObservation: {observacao}"
            return {"historico": historico_novo, "n_passos": n_passos}

        nome_tool, entrada_tool = acao_match.group(1), acao_match.group(2).strip()
        tool = tools_por_nome.get(nome_tool)

        inicio = time.monotonic()
        if tool is None:
            observacao = f"Tool '{nome_tool}' nao existe. Tools disponiveis: {list(tools_por_nome)}."
        else:
            try:
                observacao = tool.funcao(entrada_tool)
            except Exception as erro:
                observacao = f"Erro ao executar {nome_tool}: {erro}"
        duracao_tool = time.monotonic() - inicio

        if trace.passos:
            ultimo = trace.passos[-1]
            ultimo.action = nome_tool
            ultimo.action_input = entrada_tool
            ultimo.observation = observacao

        # Passo de tool separado no trace (camada semantica: relaciona o passo
        # de tool como DEPEND_ON do passo de LLM que decidiu a acao).
        registrar_passo(
            trace,
            tipo="tool",
            action=nome_tool,
            action_input=entrada_tool,
            observation=observacao,
            duracao_s=duracao_tool,
        )

        historico_novo = estado.get("historico", "") + f"\nObservation: {observacao}"
        finalizado = n_passos >= MAX_PASSOS
        return {"historico": historico_novo, "n_passos": n_passos, "finalizado": finalizado,
                "resposta_final": estado.get("resposta_final", "") if not finalizado else
                f"[MAX_PASSOS atingido sem finalizar] {observacao}"}

    return node


def _decidir_continuar(estado: EstadoReAct) -> str:
    return END if estado.get("finalizado") else "pensar_agir"


def construir_grafo(tools: list[Tool]):
    """Monta e compila o StateGraph do agente ReAct com as tools dadas."""
    tools_por_nome = {t.nome: t for t in tools}
    grafo = StateGraph(EstadoReAct)
    grafo.add_node("pensar_agir", _no_pensar_agir(tools_por_nome, tools))
    grafo.add_node("observar", _no_observar(tools_por_nome))
    grafo.add_edge(START, "pensar_agir")
    grafo.add_edge("pensar_agir", "observar")
    grafo.add_conditional_edges("observar", _decidir_continuar, {"pensar_agir": "pensar_agir", END: END})
    return grafo.compile()


def rodar_react(
    tarefa: str,
    entrada: str,
    tools: list[Tool],
    modelo: str = "qwen2.5:7b",
    versoes: Optional[dict[str, str]] = None,
    arquivo_trace: Optional[str] = None,
) -> Trace:
    """Ponto de entrada usado pelos scripts de tarefa (tarefa_hotpotqa.py,
    tarefa_sql.py). Roda o grafo ReAct ate finalizar/estourar MAX_PASSOS e
    devolve o Trace completo (ja salvo em JSONL antes de retornar)."""
    from shared.trace import iniciar_trace

    grafo = construir_grafo(tools)

    with iniciar_trace(tarefa, entrada, modelo=modelo, versoes=versoes or {}, arquivo=arquivo_trace) as trace:
        estado_inicial: EstadoReAct = {
            "tarefa": tarefa, "entrada": entrada, "modelo": modelo,
            "historico": "", "n_passos": 0, "finalizado": False, "trace": trace,
        }
        try:
            resultado = grafo.invoke(estado_inicial, config={"recursion_limit": MAX_PASSOS * 3})
            trace.resultado_final = resultado.get("resposta_final", "")
            trace.sucesso = bool(trace.resultado_final) and "MAX_PASSOS atingido" not in trace.resultado_final
        except Exception as erro:
            trace.resultado_final = f"[erro] {erro}"
            trace.sucesso = False

    return trace
