"""
Gera imagens sinteticas de graficos tecnicos a partir de dados reais dos pipelines do Harbor,
para prototipar o RAG multimodal (skill rag-multimodal, item 1 do checklist).

Grafico gerado PROGRAMATICAMENTE (matplotlib), nao via IA generativa de imagem -- mais barato,
controlavel, e sabe-se exatamente o conteudo de cada grafico para validar o captioning do VLM
depois (ver achado 5 de references/resumo_rag_multimodal_2025_2026.md).

Fonte dos dados: outputs/pipeline2_legacy_sensor/separacao_features_por_classe.csv (valores reais
de sensor por classe Fault/Normal) e outputs/pipeline4_five_axis_cnc/anomalias_temperatura.csv
(serie temporal de anomalia de temperatura por componente do CNC).
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RAIZ = Path(r"C:\Projetos\Harbor")
SAIDA = RAIZ / "rag" / "manuais_imagens"
SAIDA.mkdir(parents=True, exist_ok=True)


def grafico_temperatura_por_classe():
    """Boxplot de temperatura (Fault vs Normal) -- dado real do pipeline 2."""
    df = pd.read_csv(RAIZ / "outputs" / "pipeline2_legacy_sensor" / "separacao_features_por_classe.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    df.boxplot(column="Temperature_C", by="Target", ax=ax)
    ax.set_title("Temperatura do sensor por classe de operação")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Temperatura (°C)")
    plt.suptitle("")
    fig.tight_layout()
    caminho = SAIDA / "grafico_temperatura_por_classe.png"
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return caminho


def grafico_vibracao_por_classe():
    """Boxplot de vibração (Fault vs Normal) -- dado real do pipeline 2."""
    df = pd.read_csv(RAIZ / "outputs" / "pipeline2_legacy_sensor" / "separacao_features_por_classe.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    df.boxplot(column="Vibration_Level", by="Target", ax=ax)
    ax.set_title("Nível de vibração por classe de operação")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Vibração (nível)")
    plt.suptitle("")
    fig.tight_layout()
    caminho = SAIDA / "grafico_vibracao_por_classe.png"
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return caminho


def grafico_pressao_vazao():
    """Dispersão pressão x vazão, colorido por classe -- dado real do pipeline 2."""
    df = pd.read_csv(RAIZ / "outputs" / "pipeline2_legacy_sensor" / "separacao_features_por_classe.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    cores = {"Fault": "#F5642D", "Normal": "#A6186B"}
    for classe, grupo in df.groupby("Target"):
        ax.scatter(grupo["Pressure_bar"], grupo["FlowRate_Lmin"], label=classe,
                   alpha=0.4, s=12, color=cores.get(classe, "gray"))
    ax.set_title("Pressão vs. vazão por classe de operação")
    ax.set_xlabel("Pressão (bar)")
    ax.set_ylabel("Vazão (L/min)")
    ax.legend()
    fig.tight_layout()
    caminho = SAIDA / "grafico_pressao_vazao.png"
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return caminho


def grafico_anomalias_componentes_cnc():
    """Contagem de anomalias de temperatura por componente do CNC de 5 eixos -- dado real do
    pipeline 4, com rótulo de componente explícito (Spindle, X/Y/Z Axis) -- caso de uso direto
    do achado de "attribute binding" da pesquisa (imagem com múltiplos componentes rotulados)."""
    df = pd.read_csv(RAIZ / "outputs" / "pipeline4_five_axis_cnc" / "anomalias_temperatura.csv")
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
    fig.tight_layout()
    caminho = SAIDA / "grafico_anomalias_componentes_cnc.png"
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return caminho


if __name__ == "__main__":
    geradores = [
        grafico_temperatura_por_classe,
        grafico_vibracao_por_classe,
        grafico_pressao_vazao,
        grafico_anomalias_componentes_cnc,
    ]
    for gerar in geradores:
        caminho = gerar()
        print(f"Gerado: {caminho}")
