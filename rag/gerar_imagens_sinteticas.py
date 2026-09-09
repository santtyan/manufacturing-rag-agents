"""
Gera imagens sinteticas de graficos tecnicos a partir de dados reais dos pipelines do Harbor,
para prototipar o RAG multimodal (skill rag-multimodal, item 1 do checklist).

Grafico gerado PROGRAMATICAMENTE (matplotlib), nao via IA generativa de imagem -- mais barato,
controlavel, e sabe-se exatamente o conteudo de cada grafico para validar o captioning do VLM
depois (ver achado 5 de references/resumo_rag_multimodal_2025_2026.md).

Fonte dos dados: outputs/pipeline2_legacy_sensor/separacao_features_por_classe.csv (valores reais
de sensor por classe Fault/Normal) e outputs/pipeline4_five_axis_cnc/anomalias_temperatura.csv
(serie temporal de anomalia de temperatura por componente do CNC).

EXPANSAO 2026-09-09 (item 1 do plano "Evoluir o RAG multimodal"): o corpus original de 4 imagens
tornava o golden set de avaliacao matematicamente incapaz de detectar qualidade de captioning --
com 4 documentos e k=3, um ranqueador aleatorio ja acerta Recall@3=75% por construcao. Esta
expansao produz DISTRATORES ADVERSARIAIS DELIBERADOS, nao so "mais graficos genericos" --
familias que so se distinguem visualmente, forcando o sinal de retrieval a vir do CAPTION, nao
do nome do arquivo/BM25:

1. Mesmo dado, TIPO DE GRAFICO diferente (boxplot vs. barra vs. scatter da mesma variavel) --
   testa se o VLM le o TIPO certo (achado real: moondream respondeu "Line graph" para um
   scatter).
2. Mesma variavel, EIXOS TROCADOS -- testa se o VLM nao inverte X/Y (achado real: moondream
   leu o titulo do grafico como rotulo do eixo X).
3. Mesmo tipo de grafico, ORDENACAO diferente (barras ascendente vs. descendente) -- testa se
   o VLM captura RANKING, nao so presenca de categorias (achado real: legenda sem nenhuma
   mencao a qual componente tem mais/menos anomalias).
4. Variaveis NOVAS do mesmo CSV (Voltage_V, Current_A, Sound_dB, Humidity_%, Oil_Quality_Index,
   Energy_Consumption_kWh, Production_Rate, Load_Percentage) -- todas colunas REAIS ja
   existentes no pipeline 2, nunca visualizadas antes; quebra o atalho onde "vibracao" so
   aparecia em 1 imagem e "temperatura" em 2, permitindo a rota lexical (BM25) resolver a
   pergunta sem qualquer contribuicao do caption.

Nenhum dado é inventado -- todas as 26 imagens vêm dos mesmos 2 CSVs reais já usados nas 4
imagens originais, só variando tipo de grafico / eixos / ordenacao / variavel.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RAIZ = Path(r"C:\Projetos\Harbor")
SAIDA = RAIZ / "rag" / "manuais_imagens"
SAIDA.mkdir(parents=True, exist_ok=True)

CSV_PIPELINE2 = RAIZ / "outputs" / "pipeline2_legacy_sensor" / "separacao_features_por_classe.csv"
CSV_PIPELINE4 = RAIZ / "outputs" / "pipeline4_five_axis_cnc" / "anomalias_temperatura.csv"

CORES_CLASSE = {"Fault": "#F5642D", "Normal": "#A6186B"}


def _salvar(fig, nome):
    caminho = SAIDA / f"{nome}.png"
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return caminho


# ── As 4 imagens originais (mantidas, sem alteracao de conteudo) ──────────────────────────

def grafico_temperatura_por_classe():
    """Boxplot de temperatura (Fault vs Normal) -- dado real do pipeline 2."""
    df = pd.read_csv(CSV_PIPELINE2)
    fig, ax = plt.subplots(figsize=(7, 5))
    df.boxplot(column="Temperature_C", by="Target", ax=ax)
    ax.set_title("Temperatura do sensor por classe de operação")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Temperatura (°C)")
    plt.suptitle("")
    return _salvar(fig, "grafico_temperatura_por_classe")


def grafico_vibracao_por_classe():
    """Boxplot de vibração (Fault vs Normal) -- dado real do pipeline 2."""
    df = pd.read_csv(CSV_PIPELINE2)
    fig, ax = plt.subplots(figsize=(7, 5))
    df.boxplot(column="Vibration_Level", by="Target", ax=ax)
    ax.set_title("Nível de vibração por classe de operação")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Vibração (nível)")
    plt.suptitle("")
    return _salvar(fig, "grafico_vibracao_por_classe")


def grafico_pressao_vazao():
    """Dispersão pressão x vazão, colorido por classe -- dado real do pipeline 2."""
    df = pd.read_csv(CSV_PIPELINE2)
    fig, ax = plt.subplots(figsize=(7, 5))
    for classe, grupo in df.groupby("Target"):
        ax.scatter(grupo["Pressure_bar"], grupo["FlowRate_Lmin"], label=classe,
                   alpha=0.4, s=12, color=CORES_CLASSE.get(classe, "gray"))
    ax.set_title("Pressão vs. vazão por classe de operação")
    ax.set_xlabel("Pressão (bar)")
    ax.set_ylabel("Vazão (L/min)")
    ax.legend()
    return _salvar(fig, "grafico_pressao_vazao")


def grafico_anomalias_componentes_cnc():
    """Contagem de anomalias de temperatura por componente do CNC de 5 eixos -- dado real do
    pipeline 4, ordenado DECRESCENTE (variante ascendente em grafico_anomalias_cnc_ascendente)."""
    df = pd.read_csv(CSV_PIPELINE4)
    colunas_componentes = [c for c in df.columns if c.endswith("_anomalo")]
    contagens = df[colunas_componentes].sum().sort_values(ascending=False)
    nomes = [c.replace("_temperature_anomalo", "").replace("_anomalo", "").replace("_", " ")
             for c in contagens.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(nomes, contagens.values, color="#5B1A6B")
    ax.set_title("Anomalias de temperatura por componente (CNC 5 eixos)")
    ax.set_xlabel("Componente")
    ax.set_ylabel("Número de leituras anômalas")
    plt.xticks(rotation=30, ha="right")
    return _salvar(fig, "grafico_anomalias_componentes_cnc")


# ── Familia 1: MESMO DADO, TIPO DE GRAFICO diferente ──────────────────────────────────────

def grafico_temperatura_barras():
    """Mesma variavel de grafico_temperatura_por_classe, mas em BARRAS (media por classe) em
    vez de boxplot -- distrator: so o TIPO de grafico muda, testa se o VLM confunde boxplot
    com barra ou identifica corretamente."""
    df = pd.read_csv(CSV_PIPELINE2)
    medias = df.groupby("Target")["Temperature_C"].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(medias.index, medias.values, color=[CORES_CLASSE.get(c, "gray") for c in medias.index])
    ax.set_title("Temperatura média do sensor por classe de operação")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Temperatura média (°C)")
    return _salvar(fig, "grafico_temperatura_barras")


def grafico_temperatura_scatter_tempo():
    """Mesma variavel Temperature_C, mas em SCATTER contra indice (proxy de tempo), colorido
    por classe -- distrator: mesmo dado de origem, tipo de grafico (scatter) diferente de
    boxplot e de barra."""
    df = pd.read_csv(CSV_PIPELINE2).reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    for classe, grupo in df.groupby("Target"):
        ax.scatter(grupo["index"], grupo["Temperature_C"], label=classe,
                   alpha=0.4, s=12, color=CORES_CLASSE.get(classe, "gray"))
    ax.set_title("Temperatura do sensor ao longo das leituras, por classe")
    ax.set_xlabel("Índice da leitura")
    ax.set_ylabel("Temperatura (°C)")
    ax.legend()
    return _salvar(fig, "grafico_temperatura_scatter_tempo")


def grafico_vibracao_barras():
    """Mesma variavel de grafico_vibracao_por_classe, em BARRAS (media) -- distrator de tipo."""
    df = pd.read_csv(CSV_PIPELINE2)
    medias = df.groupby("Target")["Vibration_Level"].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(medias.index, medias.values, color=[CORES_CLASSE.get(c, "gray") for c in medias.index])
    ax.set_title("Vibração média por classe de operação")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Vibração média (nível)")
    return _salvar(fig, "grafico_vibracao_barras")


# ── Familia 2: MESMA VARIAVEL, EIXOS TROCADOS ─────────────────────────────────────────────

def grafico_pressao_vazao_eixos_trocados():
    """Mesmo dado de grafico_pressao_vazao, com X e Y INVERTIDOS -- distrator deliberado para
    testar se o VLM confunde qual eixo e qual variavel (achado real: moondream ja leu o
    TITULO do grafico como rotulo do eixo X numa legenda anterior)."""
    df = pd.read_csv(CSV_PIPELINE2)
    fig, ax = plt.subplots(figsize=(7, 5))
    for classe, grupo in df.groupby("Target"):
        ax.scatter(grupo["FlowRate_Lmin"], grupo["Pressure_bar"], label=classe,
                   alpha=0.4, s=12, color=CORES_CLASSE.get(classe, "gray"))
    ax.set_title("Vazão vs. pressão por classe de operação")
    ax.set_xlabel("Vazão (L/min)")
    ax.set_ylabel("Pressão (bar)")
    ax.legend()
    return _salvar(fig, "grafico_pressao_vazao_eixos_trocados")


def grafico_voltagem_corrente_eixos_trocados():
    """Par de graficos com eixos trocados usando variaveis NOVAS (Voltage_V x Current_A) --
    mesmo principio da familia 2, dado diferente."""
    df = pd.read_csv(CSV_PIPELINE2)
    fig, ax = plt.subplots(figsize=(7, 5))
    for classe, grupo in df.groupby("Target"):
        ax.scatter(grupo["Voltage_V"], grupo["Current_A"], label=classe,
                   alpha=0.4, s=12, color=CORES_CLASSE.get(classe, "gray"))
    ax.set_title("Corrente vs. voltagem por classe de operação")
    ax.set_xlabel("Voltagem (V)")
    ax.set_ylabel("Corrente (A)")
    ax.legend()
    return _salvar(fig, "grafico_voltagem_corrente_eixos_trocados")


def grafico_corrente_voltagem_eixos_trocados():
    """Mesmo par de grafico_voltagem_corrente_eixos_trocados, com X/Y invertidos."""
    df = pd.read_csv(CSV_PIPELINE2)
    fig, ax = plt.subplots(figsize=(7, 5))
    for classe, grupo in df.groupby("Target"):
        ax.scatter(grupo["Current_A"], grupo["Voltage_V"], label=classe,
                   alpha=0.4, s=12, color=CORES_CLASSE.get(classe, "gray"))
    ax.set_title("Voltagem vs. corrente por classe de operação")
    ax.set_xlabel("Corrente (A)")
    ax.set_ylabel("Voltagem (V)")
    ax.legend()
    return _salvar(fig, "grafico_corrente_voltagem_eixos_trocados")


# ── Familia 3: MESMO TIPO DE GRAFICO, ORDENACAO diferente ─────────────────────────────────

def grafico_anomalias_cnc_ascendente():
    """Mesmo dado de grafico_anomalias_componentes_cnc, ordenado ASCENDENTE -- distrator
    deliberado para testar se o VLM captura a ORDEM certa (achado real: legenda original nao
    mencionava nenhum ranking, so "bar graph with temperature and component")."""
    df = pd.read_csv(CSV_PIPELINE4)
    colunas_componentes = [c for c in df.columns if c.endswith("_anomalo")]
    contagens = df[colunas_componentes].sum().sort_values(ascending=True)
    nomes = [c.replace("_temperature_anomalo", "").replace("_anomalo", "").replace("_", " ")
             for c in contagens.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(nomes, contagens.values, color="#5B1A6B")
    ax.set_title("Anomalias de temperatura por componente, do menor para o maior (CNC 5 eixos)")
    ax.set_xlabel("Componente")
    ax.set_ylabel("Número de leituras anômalas")
    plt.xticks(rotation=30, ha="right")
    return _salvar(fig, "grafico_anomalias_cnc_ascendente")


def grafico_producao_por_classe_ordenado():
    """Barras de Production_Rate medio por classe, ordenado decrescente -- variavel nova,
    mesma logica de ranking da familia 3."""
    df = pd.read_csv(CSV_PIPELINE2)
    medias = df.groupby("Target")["Production_Rate"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(medias.index, medias.values, color=[CORES_CLASSE.get(c, "gray") for c in medias.index])
    ax.set_title("Taxa de produção média por classe, do maior para o menor")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Taxa de produção média")
    return _salvar(fig, "grafico_producao_por_classe_ordenado")


# ── Familia 4: variaveis NOVAS do mesmo CSV (nunca visualizadas antes) ────────────────────

def grafico_voltagem_por_classe():
    df = pd.read_csv(CSV_PIPELINE2)
    fig, ax = plt.subplots(figsize=(7, 5))
    df.boxplot(column="Voltage_V", by="Target", ax=ax)
    ax.set_title("Voltagem por classe de operação")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Voltagem (V)")
    plt.suptitle("")
    return _salvar(fig, "grafico_voltagem_por_classe")


def grafico_corrente_por_classe():
    df = pd.read_csv(CSV_PIPELINE2)
    fig, ax = plt.subplots(figsize=(7, 5))
    df.boxplot(column="Current_A", by="Target", ax=ax)
    ax.set_title("Corrente por classe de operação")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Corrente (A)")
    plt.suptitle("")
    return _salvar(fig, "grafico_corrente_por_classe")


def grafico_som_por_classe():
    df = pd.read_csv(CSV_PIPELINE2)
    fig, ax = plt.subplots(figsize=(7, 5))
    df.boxplot(column="Sound_dB", by="Target", ax=ax)
    ax.set_title("Nível de som por classe de operação")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Som (dB)")
    plt.suptitle("")
    return _salvar(fig, "grafico_som_por_classe")


def grafico_umidade_por_classe():
    df = pd.read_csv(CSV_PIPELINE2)
    fig, ax = plt.subplots(figsize=(7, 5))
    df.boxplot(column="Humidity_%", by="Target", ax=ax)
    ax.set_title("Umidade por classe de operação")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Umidade (%)")
    plt.suptitle("")
    return _salvar(fig, "grafico_umidade_por_classe")


def grafico_qualidade_oleo_por_classe():
    df = pd.read_csv(CSV_PIPELINE2)
    fig, ax = plt.subplots(figsize=(7, 5))
    df.boxplot(column="Oil_Quality_Index", by="Target", ax=ax)
    ax.set_title("Índice de qualidade do óleo por classe de operação")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Índice de qualidade do óleo")
    plt.suptitle("")
    return _salvar(fig, "grafico_qualidade_oleo_por_classe")


def grafico_consumo_energia_por_classe():
    df = pd.read_csv(CSV_PIPELINE2)
    fig, ax = plt.subplots(figsize=(7, 5))
    df.boxplot(column="Energy_Consumption_kWh", by="Target", ax=ax)
    ax.set_title("Consumo de energia por classe de operação")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Consumo de energia (kWh)")
    plt.suptitle("")
    return _salvar(fig, "grafico_consumo_energia_por_classe")


def grafico_carga_por_classe():
    df = pd.read_csv(CSV_PIPELINE2)
    fig, ax = plt.subplots(figsize=(7, 5))
    df.boxplot(column="Load_Percentage", by="Target", ax=ax)
    ax.set_title("Percentual de carga por classe de operação")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Carga (%)")
    plt.suptitle("")
    return _salvar(fig, "grafico_carga_por_classe")


def grafico_energia_vs_carga():
    """Scatter de duas variaveis novas, colorido por classe."""
    df = pd.read_csv(CSV_PIPELINE2)
    fig, ax = plt.subplots(figsize=(7, 5))
    for classe, grupo in df.groupby("Target"):
        ax.scatter(grupo["Load_Percentage"], grupo["Energy_Consumption_kWh"], label=classe,
                   alpha=0.4, s=12, color=CORES_CLASSE.get(classe, "gray"))
    ax.set_title("Consumo de energia vs. carga por classe de operação")
    ax.set_xlabel("Carga (%)")
    ax.set_ylabel("Consumo de energia (kWh)")
    ax.legend()
    return _salvar(fig, "grafico_energia_vs_carga")


def grafico_som_vs_vibracao():
    """Scatter de som x vibracao, colorido por classe -- combinacao nova."""
    df = pd.read_csv(CSV_PIPELINE2)
    fig, ax = plt.subplots(figsize=(7, 5))
    for classe, grupo in df.groupby("Target"):
        ax.scatter(grupo["Vibration_Level"], grupo["Sound_dB"], label=classe,
                   alpha=0.4, s=12, color=CORES_CLASSE.get(classe, "gray"))
    ax.set_title("Som vs. vibração por classe de operação")
    ax.set_xlabel("Vibração (nível)")
    ax.set_ylabel("Som (dB)")
    ax.legend()
    return _salvar(fig, "grafico_som_vs_vibracao")


# ── Anomalias CNC por componente individual (series temporais reais) ─────────────────────

def _grafico_serie_componente_cnc(coluna, nome_arquivo, titulo_componente):
    """Serie temporal (indice = tempo) de 0/1 de anomalia para 1 componente do CNC --
    distratores por componente individual, mesmo CSV do pipeline 4."""
    df = pd.read_csv(CSV_PIPELINE4)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.index, df[coluna], color="#5B1A6B", linewidth=0.8)
    ax.set_title(f"Anomalia de temperatura ao longo do tempo — {titulo_componente} (CNC 5 eixos)")
    ax.set_xlabel("Tempo (índice da leitura)")
    ax.set_ylabel("Anomalia (0=normal, 1=anômalo)")
    return _salvar(fig, nome_arquivo)


def grafico_anomalia_spindle_tempo():
    return _grafico_serie_componente_cnc("Spindle_motor_temperature_anomalo",
                                          "grafico_anomalia_spindle_tempo", "Spindle")


def grafico_anomalia_eixo_x_tempo():
    return _grafico_serie_componente_cnc("X_Axis_motor_temperature_anomalo",
                                          "grafico_anomalia_eixo_x_tempo", "Eixo X")


def grafico_anomalia_eixo_y_tempo():
    return _grafico_serie_componente_cnc("Y_Axis_Motor_temperature_anomalo",
                                          "grafico_anomalia_eixo_y_tempo", "Eixo Y")


def grafico_anomalia_eixo_z_tempo():
    return _grafico_serie_componente_cnc("Z_Axis_Motor_temperature_anomalo",
                                          "grafico_anomalia_eixo_z_tempo", "Eixo Z")


def grafico_anomalia_geral_tempo():
    return _grafico_serie_componente_cnc("General_temperature_anomalo",
                                          "grafico_anomalia_geral_tempo", "Geral")


if __name__ == "__main__":
    geradores = [
        # 4 originais
        grafico_temperatura_por_classe,
        grafico_vibracao_por_classe,
        grafico_pressao_vazao,
        grafico_anomalias_componentes_cnc,
        # Familia 1: mesmo dado, tipo de grafico diferente (3 novos)
        grafico_temperatura_barras,
        grafico_temperatura_scatter_tempo,
        grafico_vibracao_barras,
        # Familia 2: mesma variavel, eixos trocados (3 novos)
        grafico_pressao_vazao_eixos_trocados,
        grafico_voltagem_corrente_eixos_trocados,
        grafico_corrente_voltagem_eixos_trocados,
        # Familia 3: mesmo tipo, ordenacao diferente (2 novos)
        grafico_anomalias_cnc_ascendente,
        grafico_producao_por_classe_ordenado,
        # Familia 4: variaveis novas do mesmo CSV (8 novos)
        grafico_voltagem_por_classe,
        grafico_corrente_por_classe,
        grafico_som_por_classe,
        grafico_umidade_por_classe,
        grafico_qualidade_oleo_por_classe,
        grafico_consumo_energia_por_classe,
        grafico_carga_por_classe,
        grafico_energia_vs_carga,
        grafico_som_vs_vibracao,
        # Series temporais por componente CNC (5 novos)
        grafico_anomalia_spindle_tempo,
        grafico_anomalia_eixo_x_tempo,
        grafico_anomalia_eixo_y_tempo,
        grafico_anomalia_eixo_z_tempo,
        grafico_anomalia_geral_tempo,
    ]
    for gerar in geradores:
        caminho = gerar()
        print(f"Gerado: {caminho}")
    print(f"\nTotal: {len(geradores)} imagens.")
