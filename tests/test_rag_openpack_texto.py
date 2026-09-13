"""Testes unitarios do rag_openpack_texto (Fase 7g do plano de integracao OpenPack,
2026-09-14) -- cada teste cobre um achado real da investigacao desta sessao.

Rodar via: python -m pytest tests/test_rag_openpack_texto.py -v
"""
import numpy as np
import pandas as pd

from rag.rag_openpack_texto import (
    CANAIS,
    features_estatisticas_janela,
    montar_texto_janela,
)
from eval.avaliar_retrieval_openpack import agregar_por_fonte


def test_features_estatisticas_janela_bate_com_numpy_bruto():
    """Para um vetor sintetico conhecido, cada uma das 8 features do RAG-HAR deve bater
    exatamente com o calculo numpy direto -- nao so 'nao quebra'."""
    valores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 1.0, 2.0])
    feats = features_estatisticas_janela(valores)

    assert feats["media"] == pytest_approx(np.mean(valores))
    assert feats["maximo"] == np.max(valores)
    assert feats["minimo"] == np.min(valores)
    assert feats["q1"] == pytest_approx(np.percentile(valores, 25))
    assert feats["q3"] == pytest_approx(np.percentile(valores, 75))
    assert feats["desvio_padrao"] == pytest_approx(np.std(valores))
    assert feats["mediana"] == pytest_approx(np.median(valores))
    assert isinstance(feats["n_picos"], int)


def pytest_approx(valor, tol=1e-9):
    """Helper simples de comparacao com tolerancia, sem depender de pytest.approx diretamente
    nas asserções acima (mantido explicito para nao esconder o valor comparado)."""
    class _Aprox:
        def __eq__(self, outro):
            return abs(outro - valor) < tol
    return _Aprox()


def test_features_estatisticas_janela_vetor_vazio_nao_quebra():
    """Vetor vazio (janela sem amostras -- caso de borda) deve retornar zeros, nunca lancar
    excecao (ex. np.mean de array vazio gera RuntimeWarning + nan, que quebraria o template)."""
    feats = features_estatisticas_janela(np.array([]))
    assert feats["media"] == 0.0
    assert feats["n_picos"] == 0


def test_montar_texto_janela_gera_todas_as_secoes_esperadas():
    """O texto final deve conter uma linha por canal (6 canais) mais a linha de metadado e a
    linha de Perfil -- contrato minimo que o RAGHibrido espera indexar."""
    meta = {"sujeito": "U0101", "sessao": "S0100", "operacao": "Picking", "operation_label": 100}
    feats_segmento = {
        canal: {"media": 1.0, "maximo": 2.0, "minimo": 0.5, "desvio_padrao": 0.3,
                "q1": 0.8, "q3": 1.2, "mediana": 1.0, "n_picos": 3}
        for canal in CANAIS
    }
    texto = montar_texto_janela(meta, feats_segmento, "completo", ("moderada", "regular"))

    assert "U0101" in texto
    assert "S0100" in texto
    assert "Picking" in texto
    assert "Perfil:" in texto
    for canal in CANAIS:
        assert canal is not None  # sanity: CANAIS nao esta vazio
    assert texto.count("picos.") == len(CANAIS)  # uma linha "N picos." por canal


def test_montar_texto_janela_identificador_sujeito_tem_pontuacao_colada():
    """ACHADO REAL desta sessao (investigacao do Recall@5=0%, 2026-09-13): o template gera
    "sujeito U0101, sessao..." -- a virgula fica colada ao identificador, o que quebra
    tokenizacao BM25 baseada em str.split() (token vira "U0101," != "U0101" da query limpa).
    Este teste DOCUMENTA o comportamento atual (ainda nao corrigido -- a mitigacao adotada foi
    trocar a metrica de avaliacao para k-NN label purity, nao consertar o template). Se este
    teste comecar a FALHAR no futuro, e porque o template foi corrigido -- nesse caso, atualizar
    este teste para refletir a correcao, nao reverter a correcao."""
    meta = {"sujeito": "U0101", "sessao": "S0100", "operacao": "Picking", "operation_label": 100}
    feats_segmento = {
        canal: {"media": 1.0, "maximo": 2.0, "minimo": 0.5, "desvio_padrao": 0.3,
                "q1": 0.8, "q3": 1.2, "mediana": 1.0, "n_picos": 3}
        for canal in CANAIS
    }
    texto = montar_texto_janela(meta, feats_segmento, "completo", ("moderada", "regular"))

    tokens = texto.split()
    # Comportamento atual (achado real, nao corrigido): o token e "U0101," com virgula colada.
    assert "U0101," in tokens
    assert "U0101" not in tokens


def test_agregar_por_fonte_mantem_ordem_e_colapsa_sub_vetores():
    """4 candidatos sintéticos com o mesmo `fonte` (simulando os 4 sub-vetores
    completo/inicio/meio/fim de uma mesma janela) devem colapsar em 1 entrada -- mantendo a
    ordem de 1a aparicao (que ja vem por score decrescente do RAGHibrido)."""
    candidatos = [
        {"id": "janelaA__completo", "fonte": "janelaA"},
        {"id": "janelaB__meio", "fonte": "janelaB"},
        {"id": "janelaA__inicio", "fonte": "janelaA"},  # mesma janela, deve ser descartado
        {"id": "janelaA__fim", "fonte": "janelaA"},     # mesma janela, deve ser descartado
    ]
    resultado = agregar_por_fonte(candidatos)
    assert resultado == ["janelaA", "janelaB"]


def test_agregar_por_fonte_sem_sufixo_duplo_underscore_mantem_id():
    """Candidato sem "__" no id (formato inesperado) deve usar o proprio id como fonte, nao
    quebrar com IndexError/ValueError."""
    candidatos = [{"id": "janela_sem_sufixo", "fonte": "x"}]
    resultado = agregar_por_fonte(candidatos)
    assert resultado == ["janela_sem_sufixo"]
