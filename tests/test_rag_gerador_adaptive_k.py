"""Testes de eval/rag_gerador.py::adaptive_k (P1b do plano "RAG agentico + chunking",
2026-09-14). Cada teste cobre um achado real da implementacao -- mesmo padrao ja usado no
resto do projeto.

Rodar via: python -m pytest tests/test_rag_gerador_adaptive_k.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from rag_gerador import K_PISO_ADAPTIVE, K_TETO_ADAPTIVE, adaptive_k


class _RAGFake:
    """Stub minimo de RAGHibrido: so precisa responder a buscar() e _carregar_reranker()."""

    def __init__(self, candidatos_rrf, ordem_apos_rerank=None):
        self._candidatos_rrf = candidatos_rrf
        # Se None, rerank preserva a ordem de entrada (score decrescente identico ao RRF) --
        # simplifica os testes que nao precisam testar reordenacao do rerank em si.
        self._ordem_apos_rerank = ordem_apos_rerank

    def buscar(self, pergunta, k, usar_rerank, usar_hybrid, usar_score_rrf=False, k_candidatos=None):
        assert usar_rerank is False and usar_score_rrf is True, (
            "adaptive_k deve buscar o score RRF (usar_score_rrf=True, usar_rerank=False) para "
            "decidir o corte -- nunca o score do Cross-Encoder (achado real: pessimo preditor)"
        )
        return [dict(c) for c in self._candidatos_rrf[:k]]

    def _carregar_reranker(self):
        return _RerankerFake(self._ordem_apos_rerank)


class _RerankerFake:
    def __init__(self, ordem_apos_rerank):
        self._ordem = ordem_apos_rerank

    def score(self, pares):
        if self._ordem is None:
            # Preserva a ordem de entrada: score decrescente artificial.
            return [1.0 - i * 0.01 for i in range(len(pares))]
        # Usa a ordem fornecida para simular o rerank reordenando (achado real: Cross-Encoder
        # as vezes promove um candidato tematicamente parecido, mas errado, para o topo).
        return self._ordem


def _cand(fonte, texto, score):
    return {"fonte": fonte, "texto": texto, "id": f"{fonte}::{texto[:10]}", "score": score}


def test_corta_cedo_quando_ha_gap_forte_logo_no_inicio():
    """Achado real: a maioria das perguntas do golden set tem 1 chunk claramente mais
    relevante que o resto -- adaptive_k deve cortar perto do piso nesse caso, nao ir ate o
    teto (esse e o ganho de custo sobre k=10 fixo)."""
    candidatos = [
        _cand("manual_a.md", "chunk 1", 0.10),
        _cand("manual_a.md", "chunk 2", 0.098),
        _cand("manual_a.md", "chunk 3", 0.095),
        _cand("manual_b.md", "chunk 4", 0.02),  # queda >=40% em relacao ao anterior
        _cand("manual_b.md", "chunk 5", 0.019),
        _cand("manual_c.md", "chunk 6", 0.018),
    ]
    rag = _RAGFake(candidatos)
    resultado = adaptive_k("pergunta qualquer", rag, k_piso=3, k_teto=6)
    assert len(resultado) == 3  # corta no piso, antes do candidato 4 (onde a queda acontece)


def test_vai_ate_o_teto_quando_scores_caem_gradualmente_sem_gap_forte():
    """Achado real (2026-09-14, ao validar contra o golden set): rag-oee-alerta-mtbf tem
    scores quase uniformes -- sem gap >=40% em nenhum ponto, adaptive_k deve ir ate o teto
    (mesmo comportamento de k=10 fixo nesse caso especifico, aceitavel pois e minoria)."""
    candidatos = [_cand("manual_a.md", f"chunk {i}", 0.020 - i * 0.0005) for i in range(10)]
    rag = _RAGFake(candidatos)
    resultado = adaptive_k("pergunta qualquer", rag, k_piso=3, k_teto=10)
    assert len(resultado) == 10


def test_reranqueia_apenas_os_candidatos_cortados_nao_busca_de_novo():
    """Regressao direta de um bug real (2026-09-14): a primeira versao chamava
    rag.buscar(k=k_efetivo, usar_rerank=True) de NOVO apos calcular o corte -- isso nao
    reranqueia o MESMO conjunto de candidatos_rrf, porque o pipeline com rerank tem seu
    proprio k_candidatos default (10) e pode trazer candidatos diferentes do corte just
    calculado, jogando fora o proprio trabalho do Adaptive-k. Confirmado com o golden set
    real: rag-sensores-criticidade-temperatura tinha a secao certa em 1o lugar no RRF, mas
    uma segunda busca com rerank a jogava para fora do top-3. Corrigido reranqueando
    DIRETAMENTE os candidatos_rrf ja obtidos, sem nova chamada a buscar(). Este teste usa um
    RAGFake cujo buscar() SEMPRE retorna os mesmos 6 candidatos (nao simula um pipeline
    diferente) -- o que importa aqui e que o RESULTADO final so contem itens que vieram do
    unico buscar() chamado (k=k_teto, usar_rerank=False), nunca um candidato "novo"."""
    candidatos = [_cand(f"doc{i}.md", f"chunk {i}", 0.10 - i * 0.001) for i in range(6)]
    rag = _RAGFake(candidatos)
    resultado = adaptive_k("pergunta qualquer", rag, k_piso=3, k_teto=6)
    ids_originais = {c["id"] for c in candidatos}
    assert all(c["id"] in ids_originais for c in resultado)


def test_rerank_pode_reordenar_dentro_do_conjunto_cortado():
    """O rerank ainda decide a ORDEM final de quem entra -- Adaptive-k so decide QUANTOS.
    Se o Cross-Encoder pontuar o 2o candidato do RRF mais alto que o 1o, o resultado final
    deve refletir essa reordenacao (mesmo comportamento de RAGHibrido.buscar() normal)."""
    candidatos = [
        _cand("manual_a.md", "chunk 1", 0.10),
        _cand("manual_a.md", "chunk 2", 0.099),
        _cand("manual_a.md", "chunk 3", 0.098),
        _cand("manual_b.md", "chunk 4", 0.02),
    ]
    # Rerank inverte: da nota mais alta ao 2o candidato do RRF (indice 1).
    ordem_rerank = [0.5, 0.9, 0.3]
    rag = _RAGFake(candidatos, ordem_apos_rerank=ordem_rerank)
    resultado = adaptive_k("pergunta qualquer", rag, k_piso=3, k_teto=4)
    assert resultado[0]["texto"] == "chunk 2"


def test_usa_piso_e_teto_default_do_modulo():
    """Confirma que os defaults exportados (K_PISO_ADAPTIVE=3, K_TETO_ADAPTIVE=10) sao os
    mesmos usados pela ablacao real -- evita defaults divergentes entre o codigo de producao
    e o script de avaliacao que gerou os numeros documentados no plano."""
    assert K_PISO_ADAPTIVE == 3
    assert K_TETO_ADAPTIVE == 10


def test_corpus_menor_que_o_piso_nao_quebra():
    """Corpus/candidatos disponiveis menor que k_piso (ex. golden set minusculo em teste) nao
    deve lancar excecao -- retorna o que existir."""
    candidatos = [_cand("manual_a.md", "chunk 1", 0.10), _cand("manual_a.md", "chunk 2", 0.09)]
    rag = _RAGFake(candidatos)
    resultado = adaptive_k("pergunta qualquer", rag, k_piso=3, k_teto=10)
    assert len(resultado) == 2
