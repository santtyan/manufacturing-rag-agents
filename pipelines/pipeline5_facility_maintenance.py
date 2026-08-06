"""
Pipeline 5 - Facility Maintenance Dataset (ordens de servico de manutencao predial).
Dado Tipo B (historico/chamados em linguagem natural) do projeto Harbor.
"""
from pathlib import Path

import pandas as pd

ORIGEM = Path(r"C:\Users\USER\Downloads\Facility_Maintenance_Dataset.csv")
SAIDA = Path(r"C:\Projetos\Harbor\outputs\pipeline5_facility_maintenance")


def main():
    df = pd.read_csv(ORIGEM)
    df["DATE_REQUESTED"] = pd.to_datetime(df["DATE_REQUESTED"], errors="coerce")
    df["DATE_COMPLETED"] = pd.to_datetime(df["DATE_COMPLETED"], errors="coerce")
    df["dias_para_concluir"] = (df["DATE_COMPLETED"] - df["DATE_REQUESTED"]).dt.days

    top_problemas = df["PROB_TYPE"].value_counts().reset_index()
    top_problemas.columns = ["PROB_TYPE", "count"]
    top_problemas.to_csv(SAIDA / "top_tipos_problema.csv", index=False)

    top_sites = df["SITE_ID"].value_counts().reset_index()
    top_sites.columns = ["SITE_ID", "count"]
    top_sites.to_csv(SAIDA / "top_sites.csv", index=False)

    tempo_por_tipo = (
        df.dropna(subset=["dias_para_concluir"])
        .groupby("PROB_TYPE")["dias_para_concluir"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    tempo_por_tipo.to_csv(SAIDA / "tempo_conclusao_por_tipo.csv", index=False)

    resumo = {
        "n_ordens": len(df),
        "n_tipos_problema": df["PROB_TYPE"].nunique(),
        "n_sites": df["SITE_ID"].nunique(),
        "dias_medio_conclusao": round(df["dias_para_concluir"].mean(), 2),
        "tipo_mais_comum": top_problemas.iloc[0]["PROB_TYPE"],
        "site_mais_ordens": top_sites.iloc[0]["SITE_ID"],
    }
    pd.Series(resumo).to_json(SAIDA / "resumo.json", indent=2)

    print(f"Pipeline 5 concluido. {len(df)} ordens processadas.")
    print(resumo)


if __name__ == "__main__":
    main()
