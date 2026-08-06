"""
Pipeline 4 - Five-Axis CNC Milling Dataset (Zenodo, Siemens 840D-SL / Spinner U5-620)
Foco: ciclo de producao por produto (Program_path), tempo de changeover vs producao,
e deteccao de anomalia de temperatura dos motores (proxy simples de degradacao/falha).
Dataset mais complexo (182 colunas, granularidade de segundo) -- pipeline enxuto por ser
o de menor ROI/maior esforco entre os 4, conforme priorizacao do plano.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(r"C:\Users\USER\Downloads\Projeto_HarboR-20260707T002634Z-3-001\Projeto_HarboR\Dataset\Five-Axis CNC Milling Dataset\data_v1_0_1.csv")
OUT = Path(r"C:\Projetos\Harbor\outputs\pipeline4_five_axis_cnc")
OUT.mkdir(parents=True, exist_ok=True)

# Program_status: 1=? 2=changeover/prep 3=producao 5=idle/erro (valores inferidos a partir da distribuicao dos dados)
TEMP_COLS = [
    "Spindle_motor_temperature",
    "X_Axis_motor_temperature",
    "Z_Axis_Motor_temperature",
    "Y_Axis_Motor_temperature",
    "General_temperature",
]


def load_data(nrows=None):
    usecols = [
        "time", "Program_path", "Program_status", "Cycle_time_program",
        "Total_running_time_NC_program", "Net_running_time_program",
    ] + TEMP_COLS
    df = pd.read_csv(DATA_PATH, usecols=usecols, parse_dates=["time"], nrows=nrows, low_memory=False)
    return df


def ciclo_por_produto(df):
    """Tempo de ciclo medio e numero de registros por Program_path (proxy de produto)."""
    return (
        df.groupby("Program_path")
        .agg(
            n_registros=("Cycle_time_program", "count"),
            cycle_time_medio=("Cycle_time_program", "mean"),
            cycle_time_max=("Cycle_time_program", "max"),
            running_time_medio=("Total_running_time_NC_program", "mean"),
        )
        .sort_values("n_registros", ascending=False)
    )


def tempo_changeover_vs_producao(df):
    """Program_status como proxy: valores nao-producao contam como changeover/preparacao."""
    status_counts = df["Program_status"].value_counts().sort_index()
    return status_counts


def detectar_anomalia_temperatura(df, z_threshold=3.0):
    """Z-score sobre temperatura dos motores -- outlier estatistico como proxy de irregularidade."""
    resultado = {}
    anomalias_df = df[["time"]].copy()
    total_anomalias = 0

    for col in TEMP_COLS:
        if col not in df.columns:
            continue
        mean = df[col].mean()
        std = df[col].std()
        if std == 0 or np.isnan(std):
            continue
        z = (df[col] - mean) / std
        anomalias_df[f"{col}_anomalo"] = z.abs() > z_threshold
        n_anomalias = int((z.abs() > z_threshold).sum())
        total_anomalias += n_anomalias
        resultado[col] = {
            "media": round(float(mean), 2),
            "std": round(float(std), 2),
            "n_anomalias": n_anomalias,
        }

    return resultado, anomalias_df, total_anomalias


def main():
    df = load_data()

    ciclo = ciclo_por_produto(df)
    ciclo.to_csv(OUT / "ciclo_por_produto.csv")

    status_dist = tempo_changeover_vs_producao(df)
    status_dist.to_csv(OUT / "distribuicao_program_status.csv", header=["n_registros"])

    resumo_temp, anomalias_df, total_anomalias = detectar_anomalia_temperatura(df)
    anomalias_df.to_csv(OUT / "anomalias_temperatura.csv", index=False)

    resumo_por_componente = pd.DataFrame(
        [{"componente": col, **info} for col, info in resumo_temp.items()]
    ).sort_values("n_anomalias", ascending=False)
    resumo_por_componente.to_csv(OUT / "resumo_anomalias_por_componente.csv", index=False)

    print("=== Ciclo por produto (Program_path) ===")
    print(ciclo)
    print("\n=== Distribuicao de Program_status (proxy changeover vs producao) ===")
    print(status_dist)
    print("\n=== Resumo de anomalias de temperatura (z-score > 3) ===")
    for col, info in resumo_temp.items():
        print(f"{col}: media={info['media']}, std={info['std']}, anomalias={info['n_anomalias']}")
    print(f"\nTotal de leituras anomalas de temperatura: {total_anomalias}")
    print(f"\nOutputs salvos em: {OUT}")


if __name__ == "__main__":
    main()
