"""Testes de nivel 1 (nos isolados, LLM mockado) do grafo LangGraph do self-repair
(item 6 da rodada de fechamento da fila de trabalho, 2026-09-09).

Padrao-ouro pesquisado: separar testes deterministicos (mock do LLM, este arquivo) de
avaliacao de qualidade (golden set real, eval/rodar_golden.py -- nao muda, continua sendo o
nivel 3). Estes testes NUNCA chamam o Ollama -- so o Postgres real (para validar que
validar_sql_seguro()/_executar() continuam se comportando como esperado), com o LLM sempre
mockado via `chamar_llm` injetado.

Requer Postgres real no ar (harbor_manufatura) -- os testes de execucao de SQL nao mockam o
banco, so o LLM. Ver skill subir-servicos se os testes falharem por falta de conexao.
"""
import pytest

from nl_to_sql.nl_to_sql_langgraph import perguntar_com_dba_langgraph


def _mock_sequencial(respostas):
    """Fabrica um chamar_llm que devolve as respostas na ordem, uma por chamada."""
    it = iter(respostas)
    return lambda prompt: next(it)


def test_sql_correto_de_primeira_dba_aprova():
    """Caminho feliz: SQL gerado roda sem erro, DBA aprova -- so 2 chamadas ao LLM."""
    chamar_llm = _mock_sequencial([
        'SELECT count(*) FROM oee_downtime_raw;',
        '{"responde": true, "motivo": "ok"}',
    ])
    resultado = perguntar_com_dba_langgraph("quantos registros tem a tabela de paradas?", chamar_llm=chamar_llm)
    assert resultado["dba_responde"] is True
    assert len(resultado["resultado"]) == 1


def test_self_repair_apos_erro_de_sintaxe():
    """SQL gerado tem erro real do Postgres (coluna inexistente) -- self-repair corrige."""
    chamar_llm = _mock_sequencial([
        'SELECT contagem_invalida FROM tabela_que_nao_existe;',
        'SELECT count(*) FROM oee_downtime_raw;',
        '{"responde": true, "motivo": "ok apos correcao"}',
    ])
    resultado = perguntar_com_dba_langgraph("quantos registros tem a tabela de paradas?", chamar_llm=chamar_llm)
    assert resultado["sql"] == "SELECT count(*) FROM oee_downtime_raw;"
    assert resultado["dba_responde"] is True


def test_dba_rejeita_e_gera_sql_novo():
    """SQL roda sem erro mas o DBA rejeita (semanticamente errado) -- gera um segundo SQL."""
    chamar_llm = _mock_sequencial([
        "SELECT sum(mean) FROM oee_agregacao_paradas;",
        '{"responde": false, "motivo": "pergunta pede contagem, nao soma de media"}',
        "SELECT count(*) FROM oee_downtime_raw;",
        '{"responde": true, "motivo": "agora sim"}',
    ])
    resultado = perguntar_com_dba_langgraph("quantos registros tem a tabela de paradas?", chamar_llm=chamar_llm)
    assert resultado["sql"] == "SELECT count(*) FROM oee_downtime_raw;"
    assert resultado["dba_responde"] is True


def test_ablacao_dba_desligado_nao_chama_llm_extra():
    """Flag usar_dba_agent=False pula a segunda opiniao inteira -- so 1 chamada ao LLM."""
    chamadas = []
    def chamar_llm(prompt):
        chamadas.append(prompt)
        return "SELECT count(*) FROM oee_downtime_raw;"

    resultado = perguntar_com_dba_langgraph("quantos registros?", usar_dba_agent=False, chamar_llm=chamar_llm)
    assert resultado["dba_responde"] is True
    assert resultado["dba_motivo"] == "DBA-Agent desligado (ablacao)."
    assert len(chamadas) == 1  # so gerar_sql, sem chamada de DBA


def test_sql_invalido_rejeitado_por_validar_sql_seguro():
    """SQL nao-SELECT (ex: DROP) e rejeitado por validar_sql_seguro(), nunca chega ao Postgres --
    sem tentar_corrigir, o erro deve propagar (mesmo comportamento do original)."""
    chamar_llm = _mock_sequencial(["DROP TABLE oee_downtime_raw;"])
    with pytest.raises(RuntimeError, match="Query rejeitada por seguranca"):
        perguntar_com_dba_langgraph("apague a tabela", tentar_corrigir=False, chamar_llm=chamar_llm)
