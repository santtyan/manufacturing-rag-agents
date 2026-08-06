"""
Pipeline 3 - Discrete Manufacturing Dataset (SME, company_A e company_B)
Foco: serie temporal de estado de maquina + deteccao de mudanca de regime (estilo CUSUM/quant)
sinalizando possivel transicao para estado de alarme antes que aconteca.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\USER\Downloads\Projeto_HarboR-20260707T002634Z-3-001\Projeto_HarboR\Dataset\SME-Manufacturing-Dataset-main")
OUT = Path(r"C:\Projetos\Harbor\outputs\pipeline3_discrete_manufacturing")
OUT.mkdir(parents=True, exist_ok=True)

STATUS_A_LABELS = {0: "idle", 1: "manual", 2: "automatico", 3: "alarme"}


def load_company_a():
    df = pd.read_csv(BASE / "company_A.csv", parse_dates=["ts"])
    df["status_label"] = df["status"].map(STATUS_A_LABELS)
    return df.sort_values(["asset", "ts"]).reset_index(drop=True)


def load_company_b():
    df = pd.read_csv(BASE / "company_B" / "company_B.csv", parse_dates=["ts"])
    return df.sort_values(["asset", "ts"]).reset_index(drop=True)


def duracao_por_estado(df, group_cols, status_col):
    """Tempo total e numero de ocorrencias por estado de maquina."""
    return (
        df.groupby(group_cols + [status_col])
        .size()
        .reset_index(name="n_registros")
        .sort_values("n_registros", ascending=False)
    )


def cusum_regime_change(series, threshold_std=5.0, drift_std=0.5):
    """
    CUSUM classico (tecnica de deteccao de mudanca de regime em series financeiras):
    acumula desvios da media (descontando uma folga/drift) e RESETA a zero sempre que
    sinaliza uma mudanca -- sem isso o acumulado so cresce e satura o sinal (bug corrigido).
    """
    s = series.dropna()
    if len(s) < 5 or s.std() == 0:
        return pd.Series(dtype=float), pd.Series(dtype=bool)

    mean = s.mean()
    std = s.std()
    threshold = threshold_std * std
    drift = drift_std * std

    cusum_pos = np.zeros(len(s))
    cusum_neg = np.zeros(len(s))
    change_points = np.zeros(len(s), dtype=bool)
    deviations = (s - mean).values

    for i in range(1, len(s)):
        cusum_pos[i] = max(0, cusum_pos[i - 1] + deviations[i] - drift)
        cusum_neg[i] = min(0, cusum_neg[i - 1] + deviations[i] + drift)

        if cusum_pos[i] > threshold or cusum_neg[i] < -threshold:
            change_points[i] = True
            cusum_pos[i] = 0
            cusum_neg[i] = 0

    cusum = pd.Series(np.maximum(cusum_pos, -cusum_neg), index=s.index)
    return cusum, pd.Series(change_points, index=s.index)


def analisar_mudanca_regime_power(df, asset_col="asset", power_col="power_avg", n_top_assets=3):
    resultados = []
    top_assets = df[asset_col].value_counts().head(n_top_assets).index

    for asset in top_assets:
        sub = df[df[asset_col] == asset].sort_values("ts")
        cusum, change_points = cusum_regime_change(sub[power_col])
        n_changes = int(change_points.sum())
        resultados.append({
            "asset": asset,
            "n_registros": len(sub),
            "power_avg_medio": round(float(sub[power_col].mean()), 2),
            "pontos_mudanca_regime": n_changes,
        })

    return pd.DataFrame(resultados)


def transicao_para_alarme(df_a):
    """Verifica quais estados costumam preceder o estado de alarme (3) em company_A."""
    df_a = df_a.sort_values(["asset", "ts"]).copy()
    df_a["proximo_status"] = df_a.groupby("asset")["status"].shift(-1)
    pre_alarme = df_a[df_a["proximo_status"] == 3]
    contagem = pre_alarme["status_label"].value_counts()
    return contagem


def main():
    company_a = load_company_a()
    company_b = load_company_b()

    duracao_a = duracao_por_estado(company_a, ["asset"], "status_label")
    duracao_a.to_csv(OUT / "duracao_estados_company_a.csv", index=False)

    duracao_b = duracao_por_estado(company_b, ["asset"], "status")
    duracao_b.to_csv(OUT / "duracao_estados_company_b.csv", index=False)

    regime_a = analisar_mudanca_regime_power(company_a, power_col="power_avg")
    regime_a.to_csv(OUT / "mudanca_regime_company_a.csv", index=False)

    regime_b = analisar_mudanca_regime_power(company_b, power_col="power_avg")
    regime_b.to_csv(OUT / "mudanca_regime_company_b.csv", index=False)

    pre_alarme = transicao_para_alarme(company_a)
    pre_alarme.to_csv(OUT / "estados_antes_do_alarme.csv")

    consumo_por_status_b = company_b.groupby("status")[["power_avg", "power_min", "power_max"]].mean()
    consumo_por_status_b.to_csv(OUT / "consumo_energia_por_status_company_b.csv")

    print("=== Duracao por estado (company_A, top 10) ===")
    print(duracao_a.head(10))
    print("\n=== Mudanca de regime de power_avg (CUSUM, company_A) ===")
    print(regime_a)
    print("\n=== Estados que precedem o alarme (company_A) ===")
    print(pre_alarme)
    print("\n=== Consumo de energia medio por status (company_B) ===")
    print(consumo_por_status_b)
    print(f"\nOutputs salvos em: {OUT}")


if __name__ == "__main__":
    main()
