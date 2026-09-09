"""Tarefa NL-to-SQL como agente ReAct (Fase 5b do plano de disciplina
experimental do PDC) -- a aplicacao do padrao no dominio do Harbor,
complementando a replicacao literal do paper em agents/tarefa_hotpotqa.py.

Diferenca central em relacao ao self-repair original de nl_to_sql.py: aqui o
ciclo gerar->executar->observar erro->corrigir e explicito como Thought/
Action/Observation do agente generico (agents/react.py), nao um `try/except`
de uma unica tentativa escondido dentro de uma funcao. O agente pode tentar
ate MAX_PASSOS vezes (nao so 1 self-repair) e cada tentativa fica registrada
como um passo do trace -- o self-repair original vira, na pratica, um caso
particular deste agente com teto de 1.

A tool "executar_sql" chama validar_sql_seguro() de nl_to_sql.py antes de
rodar no Postgres -- a mesma validacao SELECT-only de producao, nunca
contornada aqui. Isso preserva a regra ja registrada na skill
migrar-para-langchain: nenhum modulo migrado abre mao da validacao
deterministica em troca de "o LLM revisa a propria query".
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.react import Tool, rodar_react
from nl_to_sql.nl_to_sql import ESQUEMA, _executar, _limpar_sql, validar_sql_seguro


def _tool_executar_sql(sql_bruto: str) -> str:
    """Tool exposta ao agente: recebe o texto de SQL gerado pelo proprio
    agente (dentro de Action: executar_sql[<sql>]), valida (SELECT-only,
    reusa validar_sql_seguro de producao) e executa no Postgres real.
    A observacao devolvida ao agente e OU o resultado (amostra), OU o erro
    real do banco -- nunca uma suposicao do agente sobre o que teria
    acontecido, o ponto central do ReAct para fundamentar o raciocinio.

    ACHADO REAL (2026-09-08, primeiro rollout de teste): o LLM as vezes
    escreve a Action como executar_sql["SELECT ..."] (SQL entre aspas
    duplas OU simples, como se fosse uma string), porque o formato ReAct usa
    colchetes para o input e o modelo generaliza colocando aspas por dentro
    tambem -- isso faz o SQL comecar com `"`/`'` em vez de `select`, e
    validar_sql_seguro() rejeita como "nao e SELECT" mesmo sendo uma query
    valida por dentro das aspas. Corrigido stripando aspas envolventes
    (duplas ou simples) ANTES de _limpar_sql(), sem afetar aspas internas
    de valores de coluna.

    ACHADO REAL #2 (mesma rodada de teste): quando o SQL foi rejeitado por
    esse motivo, o modelo repetiu a MESMA query 5 vezes seguidas sem se
    corrigir, mesmo recebendo o erro de volta como Observation -- sinal de
    que o raciocinio nao estava de fato usando a observacao de erro para
    revisar a proxima tentativa (limitacao do modelo local, nao do harness
    -- ver Rollout Card da tarefa sql para a distribuicao real disso)."""
    sql_bruto = sql_bruto.strip()
    if len(sql_bruto) >= 2 and sql_bruto[0] == sql_bruto[-1] and sql_bruto[0] in "\"'":
        sql_bruto = sql_bruto[1:-1]
    sql = _limpar_sql(sql_bruto)
    try:
        df = _executar(sql)
        if df.empty:
            return "Consulta executada sem erro, mas retornou 0 linhas."
        return f"Consulta executada com sucesso. Amostra (ate 5 linhas):\n{df.head(5).to_string()}"
    except Exception as erro:
        return f"ERRO ao executar SQL: {erro}"


def montar_tools_sql() -> list[Tool]:
    return [
        Tool(
            nome="executar_sql",
            descricao=(
                "Executa uma consulta SQL (somente SELECT) no Postgres harbor_manufatura e "
                "retorna o resultado real ou o erro real do banco. Esquema disponivel:\n" + ESQUEMA
            ),
            funcao=_tool_executar_sql,
        ),
    ]


def rodar_tarefa_sql(pergunta: str, modelo: str = "qwen2.5:7b", arquivo_trace: str | None = None):
    """Roda o agente ReAct sobre uma pergunta em portugues que deveria virar
    SQL. Devolve o Trace completo -- ver agents/react.py::rodar_react."""
    tools = montar_tools_sql()
    entrada = (
        f"Traduza a pergunta a seguir para uma consulta SQL PostgreSQL e execute-a para obter "
        f"a resposta. Pergunta: {pergunta}\n\n"
        f"Regra critica de sintaxe: colunas com maiuscula devem ir entre aspas duplas. "
        f"So SELECT e permitido. Prefira consultar 1 tabela por vez."
    )
    return rodar_react(
        tarefa="sql",
        entrada=entrada,
        tools=tools,
        modelo=modelo,
        versoes={"nl_to_sql": "self-repair-as-react"},
        arquivo_trace=arquivo_trace,
    )


if __name__ == "__main__":
    import argparse
    import json as _json

    from shared.rollout_card import salvar_rollout_card

    parser = argparse.ArgumentParser(description="Agente ReAct sobre NL-to-SQL (Fase 5b PDC)")
    parser.add_argument("--n", type=int, default=5, help="numero de perguntas do golden set de rota sql a rodar")
    parser.add_argument("--modelo", default="qwen2.5:7b")
    args = parser.parse_args()

    golden = _json.loads((Path(__file__).resolve().parent.parent / "eval" / "golden_questions.json").read_text(encoding="utf-8"))
    perguntas_sql = [p for p in golden["perguntas"] if p.get("rota_esperada") == "sql"][: args.n]

    if not perguntas_sql:
        print("Nenhuma golden question de rota 'sql' encontrada.")
        raise SystemExit(1)

    print(f"Rodando agente ReAct sobre {len(perguntas_sql)} perguntas de rota SQL (modelo={args.modelo})...\n")
    for pq in perguntas_sql:
        print(f"[{pq['id']}] {pq['pergunta']}")
        trace = rodar_tarefa_sql(pq["pergunta"], modelo=args.modelo)
        print(f"  sucesso={trace.sucesso} passos={len(trace.passos)} custo={trace.custo_total}")
        print(f"  resposta: {trace.resultado_final[:150]}")
        print("-" * 70)

    caminho_card = salvar_rollout_card(
        "sql",
        dependencias={"langgraph": "1.x", "langchain-ollama": "1.x"},
        tools=["executar_sql"],
        prompt_resumo="Traduzir pergunta em portugues para SQL PostgreSQL e executar via ReAct.",
    )
    print(f"\nRollout Card salvo em {caminho_card}")
