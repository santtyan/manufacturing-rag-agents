"""
Pipeline 6 - Labeled Car Dataset (manutencao de frota de veiculos).
Dado Tipo B (historico/chamados em linguagem natural) do projeto Harbor.
"""
from pathlib import Path

import pandas as pd

ORIGEM = Path(r"C:\Users\USER\Downloads\Labeled_Car_DatasetFile.csv")
SAIDA = Path(r"C:\Projetos\Harbor\outputs\pipeline6_labeled_car")


def main():
    df = pd.read_csv(ORIGEM)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    top_motivos = df["Reason"].value_counts().reset_index()
    top_motivos.columns = ["Reason", "count"]
    top_motivos.to_csv(SAIDA / "top_motivos.csv", index=False)

    por_veiculo = df.groupby("Dept").size().reset_index(name="n_chamados")
    por_veiculo = por_veiculo.sort_values("n_chamados", ascending=False)
    por_veiculo.to_csv(SAIDA / "chamados_por_veiculo.csv", index=False)

    resumo = {
        "n_chamados": len(df),
        "n_veiculos": df["Dept"].nunique(),
        "n_motivos_distintos": df["Reason"].nunique(),
        "motivo_mais_comum": top_motivos.iloc[0]["Reason"],
        "pct_motivo_mais_comum": round(100 * top_motivos.iloc[0]["count"] / len(df), 1),
    }
    pd.Series(resumo).to_json(SAIDA / "resumo.json", indent=2)

    print(f"Pipeline 6 concluido. {len(df)} chamados processados.")
    print(resumo)


if __name__ == "__main__":
    main()
