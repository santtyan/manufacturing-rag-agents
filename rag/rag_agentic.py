"""RAG agentic (item 3 da rodada de fechamento da fila de trabalho, 2026-09-09) -- item 3 do
roadmap `roadmap-rag-survey`: retrieval adaptativo real dentro do RAG de texto, via LangGraph
critic node (padrao Self-RAG/FLARE), em vez da chamada unica atual de
`RAGHibrido.buscar()`.

Motivacao: hoje `RAGHibrido.buscar()` (rag/rag_hibrido_langchain.py) e uma unica chamada --
recupera k trechos, reranqueia, retorna, sem nenhuma decisao sobre se o resultado e bom o
suficiente. Se a query do usuario usa vocabulario informal que nao bate com o texto tecnico do
manual (ex. "a maquina ta chacoalhando" em vez de "vibracao elevada"), o retrieval pode vir
fraco e o RAG nunca sabe disso -- so entrega o que achou.

ACHADO REAL (2026-09-09) que mudou o design deste modulo: a primeira versao usava um limiar de
score absoluto do Cross-Encoder (score_min) como criterio de confianca do critic node. Medindo
contra 15 perguntas reais do golden set de rota "rag", o score do Cross-Encoder se mostrou um
PESSIMO preditor de acerto -- casos que ERRAM o documento certo com score alto (6.514, 4.419) e
casos que ACERTAM com score baixo (0.836). Pesquisa confirmou que isso e esperado: o padrao-ouro
(CRAG -- Corrective RAG, Yan et al.) usa um "lightweight retrieval evaluator" que julga a
RELEVANCIA SEMANTICA do texto recuperado (via LLM), nao um score numerico de reranker isolado --
score de Cross-Encoder mede similaridade sintatica/lexical local, nao se o TEXTO de fato responde
a pergunta. Por isso este modulo usa um LLM juiz (mesmo padrao ja validado do DBA-Agent em
nl_to_sql.py::verificar_resultado_responde), nao um limiar de score.

Este modulo adiciona uma camada de decisao: apos o primeiro retrieval (com rerank), um "critic
node" (LLM juiz) avalia se o MELHOR candidato recuperado de fato responde a pergunta. Se sim,
retorna direto (custo extra: 1 chamada de LLM por busca -- barato comparado ao ganho de
confiabilidade, dado que o score do reranker nao serve como proxy). Se nao, reformula a query
via LLM (vocabulario informal -> termo tecnico) e busca de novo, UMA vez (mesmo principio de
teto de tentativas ja aplicado no self-repair do NL-to-SQL e no roteador -- nunca loop sem teto).

Nao reescreve RAGHibrido -- so orquestra chamadas a `buscar()` existente dentro de um grafo,
mesmo padrao ja usado em nl_to_sql_langgraph.py/roteador_langgraph.py (reusar a logica de
dominio existente, mudar so a orquestracao).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_hibrido_langchain import RAGHibrido
from shared.ollama_client import chamar as chamar_ollama

MODELO_REESCRITA = "qwen2.5:7b"
MODELO_JUIZ = "qwen2.5:7b"


class EstadoRAGAgentic(TypedDict, total=False):
    pergunta: str
    pergunta_efetiva: str
    k: int
    candidatos: list
    tentativas: int
    rag: object  # instancia de RAGHibrido, injetada para reusar indice ja carregado


def _no_buscar(estado: EstadoRAGAgentic) -> dict:
    rag: RAGHibrido = estado["rag"]
    candidatos = rag.buscar(estado["pergunta_efetiva"], k=estado.get("k", 3), usar_rerank=True, usar_hybrid=True)
    return {"candidatos": candidatos}


def _texto_responde_pergunta(pergunta: str, texto: str) -> bool:
    """LLM juiz (mesmo padrao do DBA-Agent, nl_to_sql.py::verificar_resultado_responde):
    avalia RELEVANCIA SEMANTICA, nao score numerico -- ver achado real na docstring do modulo
    sobre por que o score do Cross-Encoder nao serve como proxy de acerto."""
    import json
    import re

    prompt = f"""Voce esta avaliando se um trecho de manual tecnico responde a pergunta de um
usuario. Responda em JSON: {{"responde": true ou false}}.

=== PERGUNTA ===
{pergunta}

=== TRECHO RECUPERADO ===
{texto[:800]}

O trecho contem informacao que responde diretamente a pergunta? Responda SOMENTE o JSON."""
    try:
        resposta = str(chamar_ollama(prompt, modelo=MODELO_JUIZ))
        match = re.search(r"\{.*\}", resposta, re.DOTALL)
        corpo = json.loads(match.group(0)) if match else {}
        return bool(corpo.get("responde", True))
    except Exception:
        # Juiz indisponivel/resposta invalida -- nao bloqueia o resultado original (mesma
        # postura de fail-open do DBA-Agent original).
        return True


def _decidir_apos_buscar(estado: EstadoRAGAgentic) -> str:
    candidatos = estado.get("candidatos") or []
    ja_tentou_reformular = estado.get("tentativas", 0) >= 1
    if not candidatos or ja_tentou_reformular:
        return END
    melhor_responde = _texto_responde_pergunta(estado["pergunta"], candidatos[0]["texto"])
    if melhor_responde:
        return END
    return "reformular_query"


def _no_reformular_query(estado: EstadoRAGAgentic) -> dict:
    """Query rewrite via LLM: traduz vocabulario informal para termo tecnico -- mesma ideia do
    item 1 do roadmap roadmap-rag-survey (nunca implementado antes disoladamente; aqui entra
    como parte do critic node, nao como etapa sempre-ligada)."""
    prompt = f"""Reescreva a pergunta abaixo trocando linguagem informal por termos tecnicos de
manutencao industrial, mantendo o mesmo significado. Responda SOMENTE com a pergunta reescrita,
sem explicacao.

Pergunta original: {estado['pergunta']}

Pergunta reescrita:"""
    pergunta_reescrita = str(chamar_ollama(prompt, modelo=MODELO_REESCRITA)).strip()
    return {"pergunta_efetiva": pergunta_reescrita or estado["pergunta"], "tentativas": estado.get("tentativas", 0) + 1}


def construir_grafo_rag_agentic():
    grafo = StateGraph(EstadoRAGAgentic)
    grafo.add_node("buscar", _no_buscar)
    grafo.add_node("reformular_query", _no_reformular_query)

    grafo.add_edge(START, "buscar")
    grafo.add_conditional_edges("buscar", _decidir_apos_buscar, {
        END: END, "reformular_query": "reformular_query",
    })
    grafo.add_edge("reformular_query", "buscar")

    return grafo.compile()


_GRAFO_RAG_AGENTIC = None


def _obter_grafo():
    global _GRAFO_RAG_AGENTIC
    if _GRAFO_RAG_AGENTIC is None:
        _GRAFO_RAG_AGENTIC = construir_grafo_rag_agentic()
    return _GRAFO_RAG_AGENTIC


def buscar_agentic(pergunta, rag: RAGHibrido, k=3):
    """Substituto de RAGHibrido.buscar() com retrieval adaptativo: se o primeiro resultado
    tiver score baixo, reformula a query uma vez e busca de novo. Retorna a mesma lista de
    candidatos que buscar() retornaria (drop-in no formato de saida)."""
    return buscar_agentic_com_estado(pergunta, rag, k)["candidatos"]


def buscar_agentic_com_estado(pergunta, rag: RAGHibrido, k=3) -> dict:
    """Mesma logica de buscar_agentic(), mas devolve o ESTADO FINAL completo do grafo (nao so
    'candidatos') -- usado por eval/avaliar_rag_agentic.py (P3 do plano de 2026-09-14) para
    instrumentar custo agentico real: 'tentativas' (quantas vezes reformulou, 0 = respondeu de
    primeira) e 'pergunta_efetiva' (a query final usada, igual a original se nunca reformulou).
    Sem isso nao ha como medir taxa de re-retrieval sem re-executar o grafo por fora."""
    grafo = _obter_grafo()
    estado_inicial: EstadoRAGAgentic = {
        "pergunta": pergunta, "pergunta_efetiva": pergunta, "k": k, "rag": rag, "tentativas": 0,
    }
    return grafo.invoke(estado_inicial, config={"recursion_limit": 6})
