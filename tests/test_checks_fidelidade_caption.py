"""Testes dos checks deterministicos de fidelidade de caption (item 1, 2026-09-09).

Ver eval/checks_fidelidade_caption.py para o motivo de cada check existir -- cada teste aqui
reproduz um achado real documentado nas legendas do moondream ja indexadas no projeto.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from checks_fidelidade_caption import (
    avaliar_legenda, nao_degenerado, idioma_pt, tipo_grafico_correto,
    menciona_eixos_corretos, contem_ranking, sem_numeros_inventados,
)


def test_nao_degenerado_rejeita_vetor_de_floats():
    """Achado real: moondream retornou '[0.0, 0.13, 0.99, 0.28]' para um boxplot."""
    assert nao_degenerado("[0.0, 0.13, 0.99, 0.28]") is False


def test_nao_degenerado_aceita_texto_normal():
    assert nao_degenerado("Este é um gráfico de barras mostrando anomalias por componente.") is True


def test_idioma_pt_rejeita_ingles():
    """Achado real: 3 de 4 legendas do moondream vieram em inglês apesar do prompt pedir portugues."""
    assert idioma_pt("1. Bar graph with temperature and component.") is False


def test_idioma_pt_aceita_portugues():
    assert idioma_pt("Este é um gráfico de barras por componente.") is True


def test_tipo_grafico_correto_rejeita_tipo_errado():
    """Achado real: moondream disse 'Line graph' para um scatter (grafico_pressao_vazao)."""
    assert tipo_grafico_correto("1. Tipo de gráfico: Line graph", "scatter") is False


def test_tipo_grafico_correto_aceita_tipo_certo_em_portugues_ou_ingles():
    assert tipo_grafico_correto("Este é um gráfico de dispersão", "scatter") is True
    assert tipo_grafico_correto("This is a scatter plot", "scatter") is True


def test_menciona_eixos_rejeita_eixos_ausentes():
    assert menciona_eixos_corretos(
        "Um gráfico qualquer sem nada específico", "Pressão (bar)", "Vazão (L/min)"
    ) is False


def test_menciona_eixos_aceita_parafraseamento():
    assert menciona_eixos_corretos(
        "O eixo mostra a pressão e a vazão do sistema.", "Pressão (bar)", "Vazão (L/min)"
    ) is True


def test_contem_ranking_none_sempre_passa():
    assert contem_ranking("qualquer texto sem comparação", None) is True


def test_contem_ranking_exige_linguagem_comparativa():
    assert contem_ranking("O gráfico mostra componentes.", "decrescente") is False
    assert contem_ranking("O Spindle tem o maior número de anomalias.", "decrescente") is True


def test_sem_numeros_inventados_pega_numero_fabricado():
    """Achado real mais grave: legenda inventou 'frequency 0.6/0.4' para um boxplot de
    Vibration_Level, cuja faixa real e ~0.78-0.81 -- os numeros nao existem no dado de origem."""
    limites = {"Vibration_Level": (0.7854981192907039, 0.8096322378716745)}
    legenda = "the normal class having a frequency of 0.6 and the fault class having 0.4"
    assert sem_numeros_inventados(legenda, limites) is False


def test_sem_numeros_inventados_aceita_numero_plausivel():
    limites = {"Temperature_C": (60.0, 80.0)}
    legenda = "A temperatura média foi de 70 graus."
    assert sem_numeros_inventados(legenda, limites) is True


def test_sem_numeros_inventados_ignora_indices_pequenos():
    """Numeros pequenos tipicos de contexto (classe 1, 2 categorias) nao devem reprovar."""
    limites = {"Temperature_C": (60.0, 80.0)}
    legenda = "A classe 1 e a classe 2 diferem na temperatura."
    assert sem_numeros_inventados(legenda, limites) is True


def test_avaliar_legenda_boa_passa_quase_tudo():
    """Controle positivo: legenda realista e fiel deve ter taxa de aprovacao alta."""
    limites = {}
    legenda = ("Este é um gráfico de barras mostrando o número de leituras anômalas por "
               "componente. O Spindle apresenta o maior número de anomalias de temperatura, "
               "seguido pelos eixos X, Y e Z.")
    r = avaliar_legenda("grafico_anomalias_componentes_cnc", legenda, limites)
    assert r["taxa_aprovacao"] >= 0.8


def test_avaliar_legenda_chave_desconhecida_levanta_erro():
    import pytest
    with pytest.raises(KeyError):
        avaliar_legenda("documento_inexistente", "qualquer legenda", {})
