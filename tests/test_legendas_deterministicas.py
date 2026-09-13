"""Testes de rag/legendas_deterministicas.py -- geracao de legenda determinística de gráfico
técnico (substitui o VLM na trilha RGB do RAG multimodal, 2026-09-14). Cada teste cobre um bug
real encontrado durante a implementação, mesmo padrão de tests/test_checks_fidelidade_caption.py
e tests/test_rag_openpack_texto.py.

Rodar via: python -m pytest tests/test_legendas_deterministicas.py -v
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from legendas_deterministicas import (
    _mapear_eixos_para_colunas,
    _texto_barra_contagem_por_componente,
    _texto_boxplot_ou_barra,
    _texto_linha_temporal,
    _texto_scatter,
    estatisticas_da_variavel,
    montar_texto_grafico,
)
from metadados_imagens_ground_truth import GROUND_TRUTH
from checks_fidelidade_caption import avaliar_legenda


def _df_classe(valores_fault, valores_normal, coluna="Variavel"):
    linhas = [{"Target": "Fault", coluna: v} for v in valores_fault]
    linhas += [{"Target": "Normal", coluna: v} for v in valores_normal]
    return pd.DataFrame(linhas)


def test_estatisticas_da_variavel_com_1_amostra_por_classe_nao_gera_nan():
    """Regressao direta do bug real (2026-09-14): separacao_features_por_classe.csv tem SO 2
    LINHAS (1 media ja pre-calculada por classe, nao dado bruto) -- desvio_padrao com n=1 dava
    NaN com o ddof=1 padrao do pandas. Corrigido para None quando n<=1, nunca NaN."""
    df = _df_classe([69.4], [70.1])
    stats = estatisticas_da_variavel(df, "Variavel", por_classe=True)
    assert stats["Fault"]["n"] == 1
    assert stats["Fault"]["desvio_padrao"] is None
    assert stats["Fault"]["minimo"] == stats["Fault"]["maximo"] == 69.4


def test_estatisticas_da_variavel_com_multiplas_amostras_calcula_desvio_real():
    df = _df_classe([10.0, 20.0, 30.0], [5.0, 5.0])
    stats = estatisticas_da_variavel(df, "Variavel", por_classe=True)
    assert stats["Fault"]["n"] == 3
    assert stats["Fault"]["media"] == 20.0
    assert stats["Fault"]["desvio_padrao"] is not None
    assert stats["Fault"]["desvio_padrao"] > 0


def test_texto_boxplot_com_1_amostra_nao_inventa_faixa():
    """Com n=1, o texto deve dizer so 'valor X', nunca 'variando de X a X' (que sugere
    distribuicao real quando na verdade e um unico ponto)."""
    df = _df_classe([69.4], [70.1])
    gt = {"titulo": "Teste", "tipo": "boxplot", "eixo_x": "Classe", "eixo_y": "Var (u)",
          "ranking": None, "variaveis_fonte": ["Variavel"]}
    texto = _texto_boxplot_ou_barra(gt, df)
    assert "variando de" not in texto
    assert "valor" in texto


def test_texto_barra_contagem_por_componente_usa_todas_variaveis_fonte():
    """Regressao direta do bug real (2026-09-14): grafico_anomalias_componentes_cnc NAO e
    'variavel por classe' (nao tem coluna Target) -- e uma CONTAGEM de 5 colunas booleanas
    diferentes, uma por componente. O template errado (_texto_boxplot_ou_barra) so olhava
    variaveis_fonte[0] e ignorava as outras 4, tratando como se fosse 1 unica serie continua."""
    df = pd.DataFrame({
        "Spindle_motor_temperature_anomalo": [True, False, False],
        "X_Axis_motor_temperature_anomalo": [False, False, False],
    })
    gt = {"titulo": "Anomalias por componente", "tipo": "barra", "eixo_x": "Componente",
          "eixo_y": "Numero de leituras anomalas", "ranking": "decrescente",
          "variaveis_fonte": ["Spindle_motor_temperature_anomalo", "X_Axis_motor_temperature_anomalo"]}
    texto = _texto_barra_contagem_por_componente(gt, df)
    assert "Spindle" in texto
    assert "1 leituras" in texto or "1 leitura" in texto
    assert "X Axis motor: 0 leituras" in texto


def test_texto_barra_contagem_ranking_ascendente_ordena_do_menor_para_o_maior():
    df = pd.DataFrame({
        "A_anomalo": [True, True, True],
        "B_anomalo": [True, False, False],
    })
    gt = {"titulo": "T", "tipo": "barra", "eixo_x": "C", "eixo_y": "N",
          "ranking": "ascendente", "variaveis_fonte": ["A_anomalo", "B_anomalo"]}
    texto = _texto_barra_contagem_por_componente(gt, df)
    assert "menor número de anomalias é B" in texto


def test_mapear_eixos_para_colunas_respeita_ordem_do_ground_truth_nao_da_lista_fonte():
    """Regressao do risco real da familia de distratores 'eixos trocados' do corpus: as
    variaveis_fonte sempre vem na MESMA ordem (ex. Pressure_bar, FlowRate_Lmin), mas o
    eixo_x/eixo_y podem estar TROCADOS entre as duas imagens da familia -- se o mapeamento so
    usasse a ordem da lista, as duas imagens (normal e eixos-trocados) gerariam o MESMO texto,
    quebrando o proposito do distrator."""
    gt_normal = {"eixo_x": "Pressão (bar)", "eixo_y": "Vazão (L/min)",
                 "variaveis_fonte": ["Pressure_bar", "FlowRate_Lmin"]}
    gt_trocado = {"eixo_x": "Vazão (L/min)", "eixo_y": "Pressão (bar)",
                  "variaveis_fonte": ["Pressure_bar", "FlowRate_Lmin"]}
    assert _mapear_eixos_para_colunas(gt_normal) == ("Pressure_bar", "FlowRate_Lmin")
    assert _mapear_eixos_para_colunas(gt_trocado) == ("FlowRate_Lmin", "Pressure_bar")


def test_texto_linha_temporal_sem_anomalia_nao_cita_numero_de_leituras_como_anomalo():
    df = pd.DataFrame({"Col_anomalo": [False] * 100})
    gt = {"titulo": "T", "tipo": "linha_temporal", "eixo_x": "Tempo", "eixo_y": "Anomalia",
          "ranking": None, "variaveis_fonte": ["Col_anomalo"]}
    texto = _texto_linha_temporal(gt, df)
    assert "constante em 0" in texto
    assert "nenhuma anomalia" in texto


def test_texto_linha_temporal_com_anomalia_reporta_contagem_e_fracao_exatas():
    df = pd.DataFrame({"Col_anomalo": [True, True, False, False]})
    gt = {"titulo": "T", "tipo": "linha_temporal", "eixo_x": "Tempo", "eixo_y": "Anomalia",
          "ranking": None, "variaveis_fonte": ["Col_anomalo"]}
    texto = _texto_linha_temporal(gt, df)
    assert "2 de 4 leituras" in texto
    assert "50,00%" in texto or "50,0%" in texto


def test_texto_scatter_com_1_variavel_fonte_usa_indice_nao_quebra():
    """Regressao direta do bug real (2026-09-14): grafico_temperatura_scatter_tempo e scatter
    contra o INDICE (proxy de tempo), nao contra uma 2a variavel -- so tem 1 item em
    variaveis_fonte. O codigo original tentava desempacotar 2 colunas e quebrava com
    IndexError antes desta excecao ser tratada."""
    df = _df_classe([10.0, 20.0], [30.0, 40.0], coluna="Temperature_C")
    gt = {"titulo": "T", "tipo": "scatter", "eixo_x": "Índice da leitura",
          "eixo_y": "Temperatura (°C)", "ranking": None, "variaveis_fonte": ["Temperature_C"]}
    texto = _texto_scatter(gt, df)
    assert "dispersão" in texto
    assert "Fault" in texto and "Normal" in texto


def test_gerador_produz_legenda_100_por_cento_fiel_para_todas_as_26_imagens():
    """Teste de integracao: cada uma das 26 imagens do GROUND_TRUTH real gera uma legenda que
    passa os 6 checks de eval/checks_fidelidade_caption.py -- e o criterio de promocao que
    motivou a troca de arquitetura (VLM nunca atingiu isso; via deterministica deve atingir por
    construcao, ja que o numero vem direto do CSV que o proprio check usa como ground truth)."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
    from checks_fidelidade_caption import _calcular_limites_plausiveis

    dfs = {
        "pipeline2": pd.read_csv(
            Path(r"C:\Projetos\Harbor\outputs\pipeline2_legacy_sensor\separacao_features_por_classe.csv")
        ),
        "pipeline4": pd.read_csv(
            Path(r"C:\Projetos\Harbor\outputs\pipeline4_five_axis_cnc\anomalias_temperatura.csv")
        ),
    }
    limites = _calcular_limites_plausiveis()

    n_passou = 0
    for nome_imagem, gt in GROUND_TRUTH.items():
        texto = montar_texto_grafico(nome_imagem, gt, dfs)
        r = avaliar_legenda(nome_imagem, texto, limites)
        if r["passou_todos"]:
            n_passou += 1
        assert r["sem_numeros_inventados"] is True, (
            f"{nome_imagem}: número inventado detectado na legenda determinística -- "
            f"isso seria um bug de template, nunca deveria acontecer por construção"
        )

    assert n_passou == len(GROUND_TRUTH), (
        f"{n_passou}/{len(GROUND_TRUTH)} passaram todos os checks -- critério de promoção "
        f"exige 100% para a via determinística ser considerada válida"
    )
