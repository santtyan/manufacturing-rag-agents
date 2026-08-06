"""
Pipeline 1 - OEE and Downtime Dataset (Heavy Clay Manufacturing, Kakoyiannis Bricks / Chipre)
Foco: agregacao de paradas, MTTR/MTBF, comparacao antes/depois Lean Six Sigma, resumo via Ollama.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = Path(r"C:\Users\USER\Downloads\Projeto_HarboR-20260707T002634Z-3-001\Projeto_HarboR\Dataset\OEE")
OUT = Path(r"C:\Projetos\Harbor\outputs\pipeline1_oee")
OUT.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"


DOWNTIME_COLUMN_MAP = {
    "Date": "Date",
    "Productcode": "Productcode",
    "Product code": "Productcode",
    "StopGroup": "StopGroup",
    "Stop group": "StopGroup",
    "Stop": "Stop",
    "StopType": "StopType",
    "Stop type": "StopType",
    "StopLocation": "StopLocation",
    "Stop Location": "StopLocation",
    "ExtraText": "ExtraText",
    "Extra text": "ExtraText",
    "StopStartTime": "StopStartTime",
    "Stop start time": "StopStartTime",
    "StopEndTime": "StopEndTime",
    "Stop end time": "StopEndTime",
    "StopDuration(min)": "StopDuration(min)",
    "Stop duration (min) (TTR)": "StopDuration(min)",
}


def load_downtime(path):
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(axis=1, how="all")
    df = df.rename(columns=DOWNTIME_COLUMN_MAP)
    df = df.dropna(subset=["StopStartTime"])
    df["StopStartTime"] = pd.to_datetime(df["StopStartTime"])
    df["StopEndTime"] = pd.to_datetime(df["StopEndTime"])
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    df["StopDuration(min)"] = pd.to_numeric(df["StopDuration(min)"], errors="coerce")
    return df


def load_oee(path):
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(axis=1, how="all")
    df["shiftDate"] = pd.to_datetime(df["shiftDate"], format="mixed")
    return df


def compute_mttr_mtbf(downtime_df):
    """MTTR = duracao media de reparo (paradas nao planejadas). MTBF = tempo medio entre inicios de parada."""
    unplanned = downtime_df[downtime_df["StopType"] == "Unplanned"].sort_values("StopStartTime")
    mttr_min = unplanned["StopDuration(min)"].mean()

    gaps = unplanned["StopStartTime"].diff().dropna().dt.total_seconds() / 60
    mtbf_min = gaps.mean()

    return {
        "MTTR_min": round(float(mttr_min), 2),
        "MTBF_min": round(float(mtbf_min), 2),
        "n_unplanned_stops": int(len(unplanned)),
    }


def rolling_anomaly_by_day(downtime_df):
    """Rolling z-score sobre tempo total de parada diario -- estilo quant (deteccao de outlier)."""
    daily = downtime_df.groupby(downtime_df["StopStartTime"].dt.date)["StopDuration(min)"].sum()
    daily = daily.sort_index()
    roll_mean = daily.rolling(window=3, min_periods=1).mean()
    roll_std = daily.rolling(window=3, min_periods=1).std().replace(0, np.nan)
    z = (daily - roll_mean) / roll_std
    result = pd.DataFrame({
        "total_stop_min": daily,
        "rolling_mean": roll_mean,
        "z_score": z,
    })
    result["anomalo"] = result["z_score"].abs() > 2
    return result


def compare_before_after_lss(oee_before, oee_after, downtime_before, downtime_after):
    def summarize(oee_df, downtime_df):
        return {
            "OEE_medio": round(float(oee_df["OEE"].mean()), 4),
            "Availability_media": round(float(oee_df["Availability"].mean()), 4),
            "Performance_media": round(float(oee_df["Performance"].mean()), 4),
            "Quality_media": round(float(oee_df["Quality"].mean()), 4),
            "tempo_parada_total_min": round(float(downtime_df["StopDuration(min)"].sum()), 1),
            **compute_mttr_mtbf(downtime_df),
        }

    before = summarize(oee_before, downtime_before)
    after = summarize(oee_after, downtime_after)
    comparison = pd.DataFrame({"antes_LSS": before, "depois_LSS": after})
    comparison["variacao_%"] = (
        (comparison["depois_LSS"] - comparison["antes_LSS"]) / comparison["antes_LSS"].abs() * 100
    ).round(1)
    return comparison


def call_ollama(prompt, model=OLLAMA_MODEL, timeout=120):
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as exc:
        return f"[Ollama indisponivel: {exc}]"


def salvar_texto_utf8(path, texto):
    with open(path, "w", encoding="utf-8") as f:
        f.write(texto)


def resumir_motivos_parada(downtime_df, top_n=3):
    top_groups = (
        downtime_df.groupby("StopGroup")["StopDuration(min)"]
        .agg(["sum", "count"])
        .sort_values("sum", ascending=False)
        .head(top_n)
    )
    notas = downtime_df[downtime_df["ExtraText"].notna() & (downtime_df["ExtraText"] != "-")]["ExtraText"].tolist()

    tabela_formatada = "\n".join(
        f"- {idx}: {row['sum']:.1f} minutos totais, {int(row['count'])} ocorrencias"
        for idx, row in top_groups.iterrows()
    )

    prompt = f"""Voce e um especialista em manutencao industrial analisando uma linha de producao de tijolos.

Categorias de parada com maior tempo total perdido (dados reais, NAO calcule percentuais, use apenas os minutos e contagens abaixo):
{tabela_formatada}

Notas dos operadores (amostra):
{chr(10).join(notas[:15]) if notas else "(sem notas relevantes)"}

Escreva em portugues, em ate 3 frases curtas: quais motivos de parada sao mais criticos (cite APENAS os minutos e contagens exatos acima) e uma recomendacao pratica de melhoria. PROIBIDO calcular ou mencionar qualquer percentual, fracao ou "equivalente a X horas". Use somente os numeros ja fornecidos, sem nenhuma conta adicional."""

    return call_ollama(prompt)


def main():
    downtime = load_downtime(BASE / "DowntimeDataset.csv")
    downtime_after = load_downtime(BASE / "DowntimeDataset_afterLSS.csv")
    oee = load_oee(BASE / "OEEdataset.csv")
    oee_after = load_oee(BASE / "OEEdataset_afterLSS.csv")

    agg_by_group = (
        downtime.groupby(["StopGroup", "StopType", "StopLocation"])["StopDuration(min)"]
        .agg(["sum", "count", "mean"])
        .sort_values("sum", ascending=False)
    )
    agg_by_group.to_csv(OUT / "agregacao_paradas.csv")

    mttr_mtbf = compute_mttr_mtbf(downtime)
    with open(OUT / "mttr_mtbf.json", "w", encoding="utf-8") as f:
        json.dump(mttr_mtbf, f, indent=2, ensure_ascii=False)

    anomalias = rolling_anomaly_by_day(downtime)
    anomalias.to_csv(OUT / "anomalias_diarias.csv")

    comparacao = compare_before_after_lss(oee, oee_after, downtime, downtime_after)
    comparacao.to_csv(OUT / "comparacao_antes_depois_lss.csv")

    resumo_llm = resumir_motivos_parada(downtime)
    with open(OUT / "resumo_llm.txt", "w", encoding="utf-8") as f:
        f.write(resumo_llm)

    print("=== MTTR / MTBF ===")
    print(mttr_mtbf)
    print("\n=== Top motivos de parada (agregado) ===")
    print(agg_by_group.head(5))
    print("\n=== Dias anomalos (z-score > 2) ===")
    print(anomalias[anomalias["anomalo"]])
    print("\n=== Comparacao antes/depois LSS ===")
    print(comparacao)
    print("\n=== Resumo do LLM ===")
    print(resumo_llm)
    print(f"\nOutputs salvos em: {OUT}")


if __name__ == "__main__":
    main()
