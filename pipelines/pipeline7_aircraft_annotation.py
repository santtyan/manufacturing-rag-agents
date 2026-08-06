"""
Pipeline 7 - Aircraft Annotation Dataset (pares problema->acao de manutencao de aeronaves).
Dado Tipo B (historico/chamados em linguagem natural) do projeto Harbor -- maior volume (6169
registros), bom candidato para RAG sobre pares problema-acao real.
"""
from pathlib import Path

import pandas as pd

ORIGEM = Path(r"C:\Users\USER\Downloads\Aircraft_Annotation_DataFile.csv")
SAIDA = Path(r"C:\Projetos\Harbor\outputs\pipeline7_aircraft_annotation")


def main():
    df = pd.read_csv(ORIGEM, encoding="utf-8-sig")

    top_problemas = df["PROBLEM"].value_counts().head(20).reset_index()
    top_problemas.columns = ["PROBLEM", "count"]
    top_problemas.to_csv(SAIDA / "top_problemas.csv", index=False)

    # Para os problemas mais comuns, qual a acao mais tomada.
    top5 = top_problemas["PROBLEM"].head(5).tolist()
    acoes_top5 = (
        df[df["PROBLEM"].isin(top5)]
        .groupby(["PROBLEM", "ACTION"])
        .size()
        .reset_index(name="count")
        .sort_values(["PROBLEM", "count"], ascending=[True, False])
    )
    acoes_top5.to_csv(SAIDA / "acoes_por_problema_top5.csv", index=False)

    resumo = {
        "n_registros": len(df),
        "n_problemas_distintos": df["PROBLEM"].nunique(),
        "n_acoes_distintas": df["ACTION"].nunique(),
        "problema_mais_comum": top_problemas.iloc[0]["PROBLEM"],
        "ocorrencias_problema_mais_comum": int(top_problemas.iloc[0]["count"]),
    }
    pd.Series(resumo).to_json(SAIDA / "resumo.json", indent=2)

    print(f"Pipeline 7 concluido. {len(df)} registros processados.")
    print(resumo)


if __name__ == "__main__":
    main()
