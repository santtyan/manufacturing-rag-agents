"""Roteador do chat como grafo LangGraph (item 2 da rodada de fechamento da fila de trabalho,
2026-09-09) -- item 3 do roadmap `migrar-para-langchain`, o de maior risco/esforço da tabela.

Por que este arquivo existe (nao editar roteador.py in-place): mesma regra ja seguida na
migracao do self-repair (nl_to_sql_langgraph.py) -- nenhuma migracao e aceita se regredir
eval/rodar_golden.py, entao a nova implementacao precisa existir lado a lado com a antiga ate
ser validada pelo harness.

Nenhum gate foi reescrito ou reinterpretado -- todas as 11 funcoes `pede_*` e a logica de
`rotear_por_keyword()`/`rotear_por_llm()` de dashboard/roteador.py sao REUSADAS diretamente,
so a orquestracao (qual gate roda em que ordem, quando desempatar com o LLM) virou um grafo de
estados explicito em vez da cadeia de `if` em cascata de `rotear_pergunta()`. A ordem de
prioridade dos gates e EXATAMENTE a mesma do original (mapeada com precisao lendo
roteador.py:337-362 e roteador.py:261-293, nao reconstruida de memoria):

1. pede_identificacao_de_pessoa -> "recusa_identificacao_pessoa" (prioridade maxima, adicionado
   2026-09-12 na integracao do dataset 8/OpenPack -- recusa deterministica a pedido de julgar/
   identificar um sujeito especifico, antes de qualquer outra checagem)
2. pede_interpretacao_recall -> "interpretacao_recall" (prioridade maxima, ver comentario
   original sobre a trava que impede o desempate por LLM de reclassificar essa pergunta)
3. pede_confirmacao_alarme_automatico -> "contexto"
4. pede_ranking_assets_duas_empresas -> "contexto"
5. pede_downtime_dataset_errado -> "contexto"
6. pede_matriz_confusao_precalculada -> "contexto"
7. rotear_por_keyword() -- que por sua vez checa, nesta ordem interna:
   7a. pede_cruzamento_categoria_x_periodo_lss -> "nao_respondivel_lss"
   7b. pede_roi_dado_inexistente -> "nao_respondivel_roi"
   7c. pede_cruzamento_ciclo_x_anomalia_cnc -> "nao_respondivel_cnc"
   7d. pede_cruzamento_openpack_x_maquina -> "nao_respondivel_openpack" (2026-09-12)
   7e. pede_planned_vs_unplanned -> "planned_vs_unplanned"
   7f. pede_lss_melhorou_tudo -> "lss_melhorou_tudo"
   7g. palavra-chave SQL ou pede_agregacao_com_filtro ou pede_ranking_categoria -> "sql"
   7h. palavra-chave RAG (sem sinal de dado calculado) -> "rag"
   7i. default -> "contexto"
7. Se o resultado de (6) foi "contexto" E usar_llm=True: desempate via rotear_por_llm()
   (JSON-schema do Ollama) -- se o LLM discordar do keyword, o LLM decide; se falhar, mantem
   o keyword.

Este grafo tem 2 nos porque a decisao real e sequencial-com-curto-circuito (nao ha
ramificacao paralela nem estado compartilhado complexo entre gates) -- um no
"gates_deterministicos" que roda toda a cadeia de prioridade + keyword de uma vez (mesma
funcao pura de decisao que ja existia, sem chamada de LLM), e um no condicional
"desempate_llm" que so executa quando o primeiro no devolve "contexto" e usar_llm=True.
Fragmentar os 11 gates em 11 nos separados do grafo nao mudaria nenhum comportamento -- so
adicionaria overhead de serializacao de estado entre eles sem ganho real (nenhum gate depende
do resultado de outro, sao checagens independentes em ordem de prioridade).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

# Carregamento explicito por CAMINHO DE ARQUIVO (nao "from dashboard.roteador import ...") --
# mesmo problema de ambiguidade de nome ja resolvido em nl_to_sql_langgraph.py: "dashboard" so
# e um pacote importavel quando o root do projeto esta no sys.path (ex.: pytest rodando da
# raiz); quando dashboard/app.py roda via `streamlit run app.py` de dentro da propria pasta
# dashboard/, "dashboard" nunca chega a ser registrado como pacote, e
# "from dashboard.roteador import ..." falha com ModuleNotFoundError (achado real, 2026-09-09).
# Carregar por caminho de arquivo elimina a dependencia de como o sys.path foi montado.
_caminho_roteador = Path(__file__).resolve().parent / "roteador.py"
_spec = importlib.util.spec_from_file_location("_roteador_impl", _caminho_roteador)
_roteador_impl = importlib.util.module_from_spec(_spec)
sys.modules["_roteador_impl"] = _roteador_impl
_spec.loader.exec_module(_roteador_impl)

pede_interpretacao_recall = _roteador_impl.pede_interpretacao_recall
pede_identificacao_de_pessoa = _roteador_impl.pede_identificacao_de_pessoa
pede_confirmacao_alarme_automatico = _roteador_impl.pede_confirmacao_alarme_automatico
pede_ranking_assets_duas_empresas = _roteador_impl.pede_ranking_assets_duas_empresas
pede_downtime_dataset_errado = _roteador_impl.pede_downtime_dataset_errado
pede_matriz_confusao_precalculada = _roteador_impl.pede_matriz_confusao_precalculada
rotear_por_keyword = _roteador_impl.rotear_por_keyword
rotear_por_llm = _roteador_impl.rotear_por_llm


class EstadoRoteador(TypedDict, total=False):
    pergunta: str
    usar_llm: bool
    ollama_url: str
    ollama_model: str
    rota: str


def _no_gates_deterministicos(estado: EstadoRoteador) -> dict:
    """Reproduz fielmente a cascata de prioridade maxima de rotear_pergunta() original
    (roteador.py:345-357) mais rotear_por_keyword() (roteador.py:261-293) -- mesma ordem,
    mesmas funcoes, sem reescrever nenhuma regra."""
    pergunta = estado["pergunta"]
    if pede_identificacao_de_pessoa(pergunta):
        return {"rota": "recusa_identificacao_pessoa"}
    if pede_interpretacao_recall(pergunta):
        return {"rota": "interpretacao_recall"}
    if pede_confirmacao_alarme_automatico(pergunta):
        return {"rota": "contexto"}
    if pede_ranking_assets_duas_empresas(pergunta):
        return {"rota": "contexto"}
    if pede_downtime_dataset_errado(pergunta):
        return {"rota": "contexto"}
    if pede_matriz_confusao_precalculada(pergunta):
        return {"rota": "contexto"}
    return {"rota": rotear_por_keyword(pergunta)}


def _decidir_desempate(estado: EstadoRoteador) -> str:
    if estado["rota"] == "contexto" and estado.get("usar_llm", True):
        return "desempate_llm"
    return END


def _no_desempate_llm(estado: EstadoRoteador) -> dict:
    """Mesmo comportamento de rotear_pergunta() original (roteador.py:361-362): so chega aqui
    quando o keyword caiu em 'contexto' -- o LLM pode reclassificar para rag/sql (sinonimo que
    o keyword nao pegou); se o LLM falhar/nao decidir, mantem 'contexto'."""
    rota_llm = rotear_por_llm(estado["pergunta"], estado["ollama_url"], estado["ollama_model"])
    return {"rota": rota_llm or estado["rota"]}


def construir_grafo_roteador():
    grafo = StateGraph(EstadoRoteador)
    grafo.add_node("gates_deterministicos", _no_gates_deterministicos)
    grafo.add_node("desempate_llm", _no_desempate_llm)

    grafo.add_edge(START, "gates_deterministicos")
    grafo.add_conditional_edges("gates_deterministicos", _decidir_desempate, {
        "desempate_llm": "desempate_llm", END: END,
    })
    grafo.add_edge("desempate_llm", END)

    return grafo.compile()


_GRAFO_ROTEADOR = None


def _obter_grafo():
    global _GRAFO_ROTEADOR
    if _GRAFO_ROTEADOR is None:
        _GRAFO_ROTEADOR = construir_grafo_roteador()
    return _GRAFO_ROTEADOR


def rotear_pergunta_langgraph(
    pergunta,
    usar_llm=True,
    ollama_url="http://localhost:11434/api/generate",
    ollama_model="llama3.2",
):
    """Substituto de rotear_pergunta() (roteador.py) via grafo LangGraph -- mesma assinatura e
    mesmo tipo de retorno (string da rota), drop-in replacement para dashboard/app.py e
    eval/rodar_golden.py. Ver docstring do modulo para a ordem exata de prioridade preservada."""
    grafo = _obter_grafo()
    estado_inicial: EstadoRoteador = {
        "pergunta": pergunta,
        "usar_llm": usar_llm,
        "ollama_url": ollama_url,
        "ollama_model": ollama_model,
    }
    resultado = grafo.invoke(estado_inicial)
    return resultado["rota"]
