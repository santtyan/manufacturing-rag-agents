"""Testes de nivel 1 (nos isolados, LLM mockado) do RAG agentic (item 6, aplicado ao item 3 --
2026-09-09). Ver rag/rag_agentic.py para o achado real: medido contra o golden set, este
protótipo PIOROU o Recall@1 (12/15 -> 11/15) e nao foi promovido para producao. Estes testes
cobrem o comportamento do grafo em si (decisao correta dado o julgamento do LLM), nao uma
alegacao de que o resultado final e melhor -- ver a skill/plano para a decisao de nao adotar.
"""
from unittest.mock import MagicMock

from rag.rag_agentic import buscar_agentic, construir_grafo_rag_agentic, _decidir_apos_buscar


def _rag_fake(sequencia_de_resultados):
    """RAGHibrido fake cujo buscar() devolve a proxima lista de candidatos da sequencia a
    cada chamada -- permite simular 1a busca fraca + 2a busca (apos reformular) melhor/pior."""
    rag = MagicMock()
    it = iter(sequencia_de_resultados)
    rag.buscar.side_effect = lambda *a, **kw: next(it)
    return rag


def test_aceita_direto_quando_juiz_aprova(monkeypatch):
    """Se o LLM juiz aprova o primeiro resultado, nao busca de novo (1 chamada a buscar())."""
    import rag.rag_agentic as mod
    monkeypatch.setattr(mod, "_texto_responde_pergunta", lambda pergunta, texto: True)

    candidatos = [{"texto": "texto relevante", "fonte": "manual_x.md", "id": "c1", "score": 1.0}]
    rag = _rag_fake([candidatos])

    resultado = buscar_agentic("pergunta qualquer", rag=rag, k=1)
    assert resultado == candidatos
    assert rag.buscar.call_count == 1


def test_reformula_uma_vez_quando_juiz_rejeita(monkeypatch):
    """Se o LLM juiz rejeita a primeira busca, reformula e busca de novo -- mas so uma vez
    (teto de tentativas, mesmo principio do self-repair/roteador)."""
    import rag.rag_agentic as mod
    monkeypatch.setattr(mod, "_texto_responde_pergunta", lambda pergunta, texto: False)
    monkeypatch.setattr(mod, "chamar_ollama", lambda prompt, modelo=None: "pergunta reescrita")

    candidatos_1 = [{"texto": "texto fraco", "fonte": "manual_x.md", "id": "c1", "score": 0.5}]
    candidatos_2 = [{"texto": "texto ainda fraco", "fonte": "manual_y.md", "id": "c2", "score": 0.3}]
    rag = _rag_fake([candidatos_1, candidatos_2])

    resultado = buscar_agentic("pergunta qualquer", rag=rag, k=1)
    assert resultado == candidatos_2  # resultado da SEGUNDA busca (apos reformular)
    assert rag.buscar.call_count == 2  # nunca uma terceira tentativa


def test_decidir_apos_buscar_sem_candidatos_finaliza():
    """Corpus vazio/sem candidatos nao trava o grafo -- finaliza direto."""
    from langgraph.graph import END
    estado = {"candidatos": [], "tentativas": 0, "pergunta": "x"}
    assert _decidir_apos_buscar(estado) == END
