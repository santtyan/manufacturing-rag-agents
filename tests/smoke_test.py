"""
Smoke tests da entrega Harbor -- sem dependencia de pytest (indisponivel por instabilidade
de rede na maquina). Roda com: python tests/smoke_test.py
Confere que os 4 pipelines geram os outputs esperados e que os servicos respondem.
"""
import json
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(r"C:\Projetos\Harbor")
OUTPUTS = ROOT / "outputs"

falhas = []


def check(descricao, condicao):
    status = "OK  " if condicao else "FALHA"
    print(f"[{status}] {descricao}")
    if not condicao:
        falhas.append(descricao)


def check_arquivo_existe(nome_pipeline, caminho):
    check(f"{nome_pipeline}: {caminho.name} existe", caminho.exists())


def check_json_tem_chaves(nome_pipeline, caminho, chaves):
    if not caminho.exists():
        check(f"{nome_pipeline}: {caminho.name} tem chaves esperadas", False)
        return
    with open(caminho, encoding="utf-8") as f:
        data = json.load(f)
    ok = all(k in data for k in chaves)
    check(f"{nome_pipeline}: {caminho.name} tem chaves {chaves}", ok)


def check_endpoint(nome, url, timeout=5):
    try:
        resp = requests.get(url, timeout=timeout)
        check(f"{nome} responde ({url})", resp.status_code == 200)
    except Exception as exc:
        check(f"{nome} responde ({url}) -- erro: {exc}", False)


def main():
    print("=== Outputs dos 4 pipelines ===")

    out1 = OUTPUTS / "pipeline1_oee"
    check_arquivo_existe("Pipeline 1 (OEE)", out1 / "mttr_mtbf.json")
    check_arquivo_existe("Pipeline 1 (OEE)", out1 / "comparacao_antes_depois_lss.csv")
    check_arquivo_existe("Pipeline 1 (OEE)", out1 / "resumo_llm.txt")
    check_json_tem_chaves("Pipeline 1 (OEE)", out1 / "mttr_mtbf.json", ["MTTR_min", "MTBF_min"])

    out2 = OUTPUTS / "pipeline2_legacy_sensor"
    check_arquivo_existe("Pipeline 2 (Legacy Sensor)", out2 / "backtest_metrics.json")
    check_arquivo_existe("Pipeline 2 (Legacy Sensor)", out2 / "predicoes.csv")
    check_json_tem_chaves(
        "Pipeline 2 (Legacy Sensor)", out2 / "backtest_metrics.json",
        ["precision", "recall", "confusion_matrix"],
    )

    out3 = OUTPUTS / "pipeline3_discrete_manufacturing"
    check_arquivo_existe("Pipeline 3 (Discrete Mfg)", out3 / "duracao_estados_company_a.csv")
    check_arquivo_existe("Pipeline 3 (Discrete Mfg)", out3 / "estados_antes_do_alarme.csv")

    out4 = OUTPUTS / "pipeline4_five_axis_cnc"
    check_arquivo_existe("Pipeline 4 (CNC)", out4 / "ciclo_por_produto.csv")
    check_arquivo_existe("Pipeline 4 (CNC)", out4 / "anomalias_temperatura.csv")

    print("\n=== Servicos ativos ===")
    check_endpoint("Ollama", "http://localhost:11434/api/tags")
    check_endpoint("Streamlit dashboard", "http://localhost:8501")

    print("\n=== Resumo ===")
    if falhas:
        print(f"{len(falhas)} falha(s):")
        for f in falhas:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("Todos os checks passaram.")
        sys.exit(0)


if __name__ == "__main__":
    main()
