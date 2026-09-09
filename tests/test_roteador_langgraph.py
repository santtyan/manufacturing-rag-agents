"""Testes de nivel 1 (nos isolados) do grafo LangGraph do roteador (item 6 da rodada de
fechamento da fila de trabalho, 2026-09-09).

Estes testes usam `usar_llm=False` para exercitar so os gates deterministicos (nenhuma
chamada ao Ollama) -- comportamento 100% reproduzivel. O desempate por LLM (usar_llm=True) e
coberto pela comparacao em lote contra o roteador original (nao aqui, para nao depender de
Ollama no ar durante `pytest` normal).
"""
from dashboard.roteador_langgraph import rotear_pergunta_langgraph
from dashboard.roteador import rotear_pergunta


def test_interpretacao_recall_tem_prioridade_maxima():
    # Precisa de "recall"/"precision" + palavra de direcao (acerta/erra/significa que/na pratica)
    # -- ver pede_interpretacao_recall() em dashboard/roteador.py:208-223.
    pergunta = "um recall de 0.097 significa que o modelo acerta quase 10 em cada 10 falhas?"
    assert rotear_pergunta_langgraph(pergunta, usar_llm=False) == "interpretacao_recall"


def test_gate_roi_dado_inexistente():
    # Precisa de palavra de investimento (roi/investimos/payback) + "lean six sigma"/"lss" --
    # ver pede_roi_dado_inexistente() em dashboard/roteador.py:123-133.
    pergunta = "qual o ROI do Lean Six Sigma, quanto tempo leva pra se pagar o investimento?"
    rota = rotear_pergunta_langgraph(pergunta, usar_llm=False)
    assert rota == "nao_respondivel_roi"


def test_gate_planned_vs_unplanned():
    pergunta = "quais paradas consomem mais minutos, planned ou unplanned?"
    rota = rotear_pergunta_langgraph(pergunta, usar_llm=False)
    assert rota == "planned_vs_unplanned"


def test_palavra_chave_sql_sem_gate_especifico():
    pergunta = "qual a media de temperatura no periodo de janeiro?"
    rota = rotear_pergunta_langgraph(pergunta, usar_llm=False)
    assert rota == "sql"


def test_palavra_chave_rag_manual():
    pergunta = "qual a temperatura maxima de operacao segundo o manual?"
    rota = rotear_pergunta_langgraph(pergunta, usar_llm=False)
    assert rota == "rag"


def test_default_contexto_sem_llm():
    pergunta = "o MTTR caiu, isso e bom ou ruim?"
    rota = rotear_pergunta_langgraph(pergunta, usar_llm=False)
    assert rota == "contexto"


def test_equivalencia_completa_golden_set_sem_llm():
    """Regressao direta: roteador novo e antigo concordam em TODO o golden set quando o
    desempate por LLM esta desligado (caminho 100% deterministico, sem rede)."""
    import json
    from pathlib import Path

    golden = json.loads((Path(__file__).resolve().parent.parent / "eval" / "golden_questions.json").read_text(encoding="utf-8"))
    divergencias = []
    for pq in golden["perguntas"]:
        r_original = rotear_pergunta(pq["pergunta"], usar_llm=False)
        r_novo = rotear_pergunta_langgraph(pq["pergunta"], usar_llm=False)
        if r_original != r_novo:
            divergencias.append((pq["id"], r_original, r_novo))

    assert divergencias == [], f"Divergencias entre roteador original e LangGraph: {divergencias}"
