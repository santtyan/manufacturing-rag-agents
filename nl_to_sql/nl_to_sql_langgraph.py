"""Self-repair do NL-to-SQL como grafo LangGraph (item 1 da rodada de fechamento da fila de
trabalho, 2026-09-09) -- promove o prototipo de agents/tarefa_sql.py de tarefa de benchmark para
o self-repair de PRODUCAO, substituindo perguntar()/perguntar_com_dba() de nl_to_sql.py.

Por que este arquivo existe (nao editar nl_to_sql.py in-place): a regra inegociavel ja
registrada na skill migrar-para-langchain e que nenhuma migracao e aceita se regredir
eval/rodar_golden.py -- precisa existir um jeito de comparar as duas implementacoes lado a lado
antes de trocar o import de producao. Depois de validado (harness sem regressao), os
consumidores (dashboard/app.py, eval/rodar_golden.py) trocam de perguntar_com_dba() para
perguntar_com_dba_langgraph() -- ver instrucoes no final deste docstring.

Diferenca central em relacao ao self-repair original: aqui o ciclo
gerar->executar->corrigir->reexecutar->verificar DBA->gerar novo->reexecutar->reverificar e
representado como um grafo de estados explicito (StateGraph), nao uma sequencia de
if/try/except aninhados. Isso reusa exatamente os MESMOS prompts e funcoes de
nl_to_sql.py (gerar_sql, corrigir_sql, verificar_resultado_responde, _executar,
validar_sql_seguro) -- a mudanca e estrutural (como o fluxo e orquestrado), nao no que cada
etapa faz ou no texto de cada prompt.

Decisao sobre o teto de tentativas (achado real do protótipo, Rollout Card
eval/traces/sql_rollout_card.md, 2026-09-08): o self-repair original tem teto de 1 tentativa de
correcao de erro + 1 tentativa de correcao de DBA (2 no total). O prototipo ReAct testado com
teto de 6 mostrou o modelo local (qwen2.5:7b) repetindo o MESMO erro de sintaxe varias vezes sem
usar a Observation de erro para se corrigir de verdade -- ou seja, aumentar o teto sozinho, sem
melhorar o prompt de correcao, nao teria ajudado. Por isso este grafo MANTEM o teto de 1+1 do
original (nao adota o teto de 6 do prototipo de benchmark) -- a decisao e sobre estrutura
(grafo vs. if/else), nao sobre "tentar mais vezes".
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from nl_to_sql.nl_to_sql import (
    call_ollama,
    corrigir_sql,
    gerar_sql,
    verificar_resultado_responde,
    _executar,
    _limpar_sql,
    ESQUEMA,
)


class EstadoSQL(TypedDict, total=False):
    pergunta_nl: str
    chamar_llm: object  # Callable[[str], str] -- injetavel, mesmo padrao de nl_to_sql.py
    tentar_corrigir: bool
    usar_dba_agent: bool
    tentar_novo_sql_se_nao_responde: bool

    sql: str
    resultado: object  # pandas.DataFrame
    erro: str | None
    dba_responde: bool
    dba_motivo: str
    finalizado: bool


def _no_gerar_sql(estado: EstadoSQL) -> dict:
    chamar_llm = estado.get("chamar_llm") or call_ollama
    sql = gerar_sql(estado["pergunta_nl"], chamar_llm=chamar_llm)
    return {"sql": sql}


def _no_executar_sql(estado: EstadoSQL) -> dict:
    try:
        resultado = _executar(estado["sql"])
        return {"resultado": resultado, "erro": None}
    except Exception as erro:
        return {"erro": str(erro)}


def _decidir_apos_executar(estado: EstadoSQL) -> str:
    if estado.get("erro") is None:
        return "verificar_dba"
    if not estado.get("tentar_corrigir", True):
        # Sem self-repair habilitado (ablacao) -- propaga o erro como no original (`raise`).
        raise RuntimeError(estado["erro"])
    return "corrigir_sql"


def _no_corrigir_sql(estado: EstadoSQL) -> dict:
    chamar_llm = estado.get("chamar_llm") or call_ollama
    sql_corrigido = corrigir_sql(estado["pergunta_nl"], estado["sql"], estado["erro"], chamar_llm=chamar_llm)
    return {"sql": sql_corrigido}


def _no_reexecutar_sql(estado: EstadoSQL) -> dict:
    # Mesmo comportamento do original: se a correcao tambem falhar, propaga (nao ha 3a tentativa).
    resultado = _executar(estado["sql"])
    return {"resultado": resultado, "erro": None}


def _no_verificar_dba(estado: EstadoSQL) -> dict:
    if not estado.get("usar_dba_agent", True):
        return {"dba_responde": True, "dba_motivo": "DBA-Agent desligado (ablacao).", "finalizado": True}
    dba_responde, dba_motivo = verificar_resultado_responde(
        estado["pergunta_nl"], estado["sql"], estado["resultado"], chamar_llm=estado.get("chamar_llm")
    )
    return {"dba_responde": dba_responde, "dba_motivo": dba_motivo}


def _decidir_apos_dba(estado: EstadoSQL) -> str:
    if estado.get("finalizado"):
        return END
    if estado["dba_responde"] or not estado.get("tentar_novo_sql_se_nao_responde", True):
        return END
    return "gerar_sql_novo"


def _no_gerar_sql_novo(estado: EstadoSQL) -> dict:
    """Equivalente ao bloco de prompt_novo em perguntar_com_dba() original -- pede um SQL
    diferente, informando o motivo do DBA ter rejeitado o anterior."""
    chamar_llm = estado.get("chamar_llm") or call_ollama
    prompt_novo = f"""Voce e um especialista em SQL PostgreSQL. A consulta abaixo RODOU SEM ERRO
mas um DBA revisor apontou que ela NAO responde a pergunta original. Motivo do DBA: {estado['dba_motivo']}
Gere uma NOVA consulta que responda corretamente. Responda SOMENTE com o SQL, sem explicacao.

REGRA CRITICA DE SINTAXE: colunas com maiusculas entre aspas duplas. So SELECT. Prefira 1 tabela.

=== ESQUEMA ===
{ESQUEMA}

=== PERGUNTA ORIGINAL ===
{estado['pergunta_nl']}

=== SQL QUE RODOU MAS NAO RESPONDEU (segundo o DBA) ===
{estado['sql']}

SQL corrigido:"""
    sql_novo = _limpar_sql(chamar_llm(prompt_novo))
    return {"sql": sql_novo}


def _no_reexecutar_e_reverificar(estado: EstadoSQL) -> dict:
    # Mesmo comportamento do original: se a 2a tentativa falhar, mantem o resultado anterior
    # (except: pass) -- aqui representado como nao atualizar resultado/dba_* em caso de excecao.
    try:
        resultado_novo = _executar(estado["sql"])
        dba_responde_novo, dba_motivo_novo = verificar_resultado_responde(
            estado["pergunta_nl"], estado["sql"], resultado_novo, chamar_llm=estado.get("chamar_llm")
        )
        return {"resultado": resultado_novo, "dba_responde": dba_responde_novo, "dba_motivo": dba_motivo_novo}
    except Exception:
        return {}


def construir_grafo_sql():
    grafo = StateGraph(EstadoSQL)
    grafo.add_node("gerar_sql", _no_gerar_sql)
    grafo.add_node("executar_sql", _no_executar_sql)
    grafo.add_node("corrigir_sql", _no_corrigir_sql)
    grafo.add_node("reexecutar_sql", _no_reexecutar_sql)
    grafo.add_node("verificar_dba", _no_verificar_dba)
    grafo.add_node("gerar_sql_novo", _no_gerar_sql_novo)
    grafo.add_node("reexecutar_e_reverificar", _no_reexecutar_e_reverificar)

    grafo.add_edge(START, "gerar_sql")
    grafo.add_edge("gerar_sql", "executar_sql")
    grafo.add_conditional_edges("executar_sql", _decidir_apos_executar, {
        "verificar_dba": "verificar_dba", "corrigir_sql": "corrigir_sql",
    })
    grafo.add_edge("corrigir_sql", "reexecutar_sql")
    grafo.add_edge("reexecutar_sql", "verificar_dba")
    grafo.add_conditional_edges("verificar_dba", _decidir_apos_dba, {
        END: END, "gerar_sql_novo": "gerar_sql_novo",
    })
    grafo.add_edge("gerar_sql_novo", "reexecutar_e_reverificar")
    grafo.add_edge("reexecutar_e_reverificar", END)

    return grafo.compile()


_GRAFO_SQL = None


def _obter_grafo():
    global _GRAFO_SQL
    if _GRAFO_SQL is None:
        _GRAFO_SQL = construir_grafo_sql()
    return _GRAFO_SQL


def perguntar_com_dba_langgraph(
    pergunta_nl,
    tentar_corrigir=True,
    tentar_novo_sql_se_nao_responde=True,
    usar_dba_agent=True,
    chamar_llm=None,
):
    """Substituto de perguntar_com_dba() (nl_to_sql.py) via grafo LangGraph -- mesma assinatura
    e mesmo dict de retorno, para ser um drop-in replacement nos consumidores
    (dashboard/app.py, eval/rodar_golden.py). Ver docstring do modulo para o porque desta
    reescrita ser estrutural (grafo) e nao mudar o comportamento/prompts do self-repair."""
    grafo = _obter_grafo()
    estado_inicial: EstadoSQL = {
        "pergunta_nl": pergunta_nl,
        "chamar_llm": chamar_llm,
        "tentar_corrigir": tentar_corrigir,
        "usar_dba_agent": usar_dba_agent,
        "tentar_novo_sql_se_nao_responde": tentar_novo_sql_se_nao_responde,
    }
    resultado_final = grafo.invoke(estado_inicial, config={"recursion_limit": 10})
    return {
        "sql": resultado_final["sql"],
        "resultado": resultado_final["resultado"],
        "dba_responde": resultado_final.get("dba_responde", True),
        "dba_motivo": resultado_final.get("dba_motivo", ""),
    }
