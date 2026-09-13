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


def test_avaliar_legenda_taxa_100_quando_todos_os_6_checks_passam():
    """Regressao direta de um bug real encontrado em 2026-09-14, presente desde a criacao do
    modulo (commit b70ebda, 2026-09-10): avaliar_legenda() inseria "passou_todos" no dict de
    resultados ANTES de capturar len(resultados) para calcular taxa_aprovacao -- dividindo por
    7 chaves (os 6 checks + a propria "passou_todos") em vez de 6. Uma legenda que passasse os
    6 checks reais ficava com taxa_aprovacao=6/7=85,7%, nunca 100% -- mascarando o resultado
    real do qwen3-vl:4b (documentado como 77,5%/reprovado; recalculado sem o bug, 96,2% sobre
    o MESMO cache de legendas ja existente, o que teria mudado a conclusao de "reprovado" para
    "aprovado" na sessao original). Este teste usa uma legenda desenhada para passar os 6
    checks e trava que taxa_aprovacao seja exatamente 1.0, nao 6/7."""
    legenda_boa = (
        "Este é um gráfico de barras mostrando o número de leituras anômalas por componente "
        "(CNC 5 eixos). O Spindle apresenta o maior número de anomalias, seguido pelos "
        "demais eixos, do maior para o menor."
    )
    r = avaliar_legenda("grafico_anomalias_componentes_cnc", legenda_boa, {})
    assert all(r[chave] for chave in
               ("nao_degenerado", "idioma_pt", "tipo_grafico_correto",
                "menciona_eixos_corretos", "contem_ranking", "sem_numeros_inventados"))
    assert r["passou_todos"] is True
    assert r["taxa_aprovacao"] == 1.0


def test_avaliar_legenda_restringe_limites_plausiveis_as_variaveis_da_propria_imagem():
    """Regressao direta de um bug real encontrado em 2026-09-14, ao validar a via de legenda
    determinística (rag/legendas_deterministicas.py): sem_numeros_inventados() e chamado com
    o dict GLOBAL de limites_plausiveis (todas as variaveis de todos os CSVs), e aceita
    qualquer numero que caiba em QUALQUER faixa do dict -- nao so nas variaveis QUE A IMAGEM EM
    QUESTAO mostra. Isso mascarou uma alucinacao numerica GENUINA e ja documentada do qwen3-vl
    (voltagem ~0,086-0,098 V relatada quando Voltage_V real e ~220,19 V) assim que o dict
    global passou a incluir a faixa de CONTAGEM de anomalias (0..52026, naturalmente ampla) --
    0,086 cabe dentro da margem de 50% dessa faixa emprestada, mesmo sem nenhuma relacao com
    voltagem. avaliar_legenda() agora restringe limites_plausiveis as variaveis_fonte do
    proprio documento antes de chamar sem_numeros_inventados -- este teste prova que uma faixa
    ampla de OUTRA variavel (nao usada por esta imagem) nao mascara mais um numero fora da
    faixa real da variavel que a imagem de fato mostra."""
    limites_globais = {
        "Voltage_V": (220.1865414710485, 220.1987103707684),
        # Faixa deliberadamente ampla de uma variavel SEM RELACAO com a imagem de voltagem --
        # simula a faixa de contagem de anomalias que mascarou o bug real.
        "Spindle_motor_temperature_anomalo": (0.0, 52026.0),
    }
    legenda_com_alucinacao = (
        "A classe Normal apresenta valores mais altos (cerca de 0,098 V), enquanto a classe "
        "Fault tem valores mais baixos (cerca de 0,086 V)."
    )
    r = avaliar_legenda("grafico_voltagem_por_classe", legenda_com_alucinacao, limites_globais)
    assert r["sem_numeros_inventados"] is False
