"""Testes de nivel 1 (LLM mockado) do Contextual Retrieval (item 4a, 2026-09-09).

Ver rag/rag_hibrido_langchain.py e o plano de implementacao para o resultado real medido:
Recall@5 identico (98%), MRR levemente melhor (0,907 vs 0,899), Precision@5 PIOROU (46,4% vs
50,0%) no golden set completo (50 perguntas) -- nao promovido para producao (flag de ablacao
usar_contexto=False continua o default). Causa provavel: manuais curtos (40-181 linhas),
corpus onde a tecnica tem retorno baixo segundo a propria pesquisa original.

Estes testes cobrem o comportamento da funcao em si (prepend correto, fail-open em erro), nao
uma alegacao de que a tecnica melhora o sistema -- ver a decisao de nao promover no plano.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))

from rag_hibrido_langchain import _contextualizar_chunk
import rag_hibrido_langchain as mod


def test_prepend_contexto_ao_chunk(monkeypatch):
    # patch no ponto de import real (dentro da funcao, via shared.ollama_client)
    import shared.ollama_client as cliente_mod
    monkeypatch.setattr(cliente_mod, "chamar", lambda prompt, modelo=None: "Este trecho fala sobre limites de temperatura.")

    resultado = _contextualizar_chunk("## Limite térmico\nA temperatura máxima é 90°C.", "documento completo aqui", "manual.md")
    assert resultado.startswith("Este trecho fala sobre limites de temperatura.")
    assert "A temperatura máxima é 90°C." in resultado


def test_fail_open_quando_ollama_falha(monkeypatch):
    import shared.ollama_client as cliente_mod

    def levanta_erro(prompt, modelo=None):
        raise ConnectionError("Ollama indisponível")

    monkeypatch.setattr(cliente_mod, "chamar", levanta_erro)

    chunk_original = "## Limite térmico\nA temperatura máxima é 90°C."
    resultado = _contextualizar_chunk(chunk_original, "documento completo", "manual.md")
    assert resultado == chunk_original  # fail-open: chunk original sem contexto, nao trava


def test_contexto_vazio_retorna_chunk_original(monkeypatch):
    import shared.ollama_client as cliente_mod
    monkeypatch.setattr(cliente_mod, "chamar", lambda prompt, modelo=None: "")

    chunk_original = "## Limite térmico\nA temperatura máxima é 90°C."
    resultado = _contextualizar_chunk(chunk_original, "documento completo", "manual.md")
    assert resultado == chunk_original
