"""
Trilha 1 - Carrega as tabelas ja limpas/calculadas pelos 4 pipelines no Postgres.
Base concreta para NL-to-SQL (Trilha 3) e para consultas diretas.
"""
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

OUTPUTS = Path(r"C:\Projetos\Harbor\outputs")
DATASET_ROOT = Path(r"C:\Users\USER\Downloads\Projeto_HarboR-20260707T002634Z-3-001\Projeto_HarboR\Dataset")

ENGINE_URL = "postgresql+psycopg2://harbor:harbor123@localhost:5432/harbor_manufatura"


def main():
    engine = create_engine(ENGINE_URL)

    tabelas = {
        # Pipeline 1 - OEE/Downtime
        "oee_downtime_raw": DATASET_ROOT / "OEE" / "DowntimeDataset.csv",
        "oee_agregacao_paradas": OUTPUTS / "pipeline1_oee" / "agregacao_paradas.csv",
        "oee_comparacao_lss": OUTPUTS / "pipeline1_oee" / "comparacao_antes_depois_lss.csv",

        # Pipeline 2 - Legacy Sensor Logs
        "sensor_predicoes": OUTPUTS / "pipeline2_legacy_sensor" / "predicoes.csv",
        "sensor_backtest_separacao": OUTPUTS / "pipeline2_legacy_sensor" / "separacao_features_por_classe.csv",

        # Pipeline 3 - Discrete Manufacturing
        "manufacturing_duracao_estados_a": OUTPUTS / "pipeline3_discrete_manufacturing" / "duracao_estados_company_a.csv",
        "manufacturing_regime_a": OUTPUTS / "pipeline3_discrete_manufacturing" / "mudanca_regime_company_a.csv",
        "manufacturing_estados_antes_alarme_a": OUTPUTS / "pipeline3_discrete_manufacturing" / "estados_antes_do_alarme.csv",
        "manufacturing_duracao_estados_b": OUTPUTS / "pipeline3_discrete_manufacturing" / "duracao_estados_company_b.csv",
        "manufacturing_regime_b": OUTPUTS / "pipeline3_discrete_manufacturing" / "mudanca_regime_company_b.csv",
        "manufacturing_consumo_energia_status_b": OUTPUTS / "pipeline3_discrete_manufacturing" / "consumo_energia_por_status_company_b.csv",

        # Pipeline 4 - Five-Axis CNC
        "cnc_ciclo_por_produto": OUTPUTS / "pipeline4_five_axis_cnc" / "ciclo_por_produto.csv",
        "cnc_distribuicao_program_status": OUTPUTS / "pipeline4_five_axis_cnc" / "distribuicao_program_status.csv",
        "cnc_resumo_anomalias_por_componente": OUTPUTS / "pipeline4_five_axis_cnc" / "resumo_anomalias_por_componente.csv",
    }

    for nome_tabela, caminho in tabelas.items():
        try:
            df = pd.read_csv(caminho, low_memory=False)
            df.to_sql(nome_tabela, engine, if_exists="replace", index=False)
            print(f"OK  {nome_tabela}: {len(df)} linhas carregadas de {caminho.name}")
        except Exception as exc:
            print(f"ERRO {nome_tabela}: {exc}")

    print("\nCarregamento concluido.")


if __name__ == "__main__":
    main()
