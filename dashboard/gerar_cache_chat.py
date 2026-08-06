"""
Pre-gera respostas do LLM para as perguntas de exemplo de cada chatbot especializado.
Evita depender do Ollama respondendo ao vivo durante a demo -- se a pergunta do usuario
bater com uma das perguntas em cache, o dashboard usa a resposta pronta (instantanea).
Rodar antes da reuniao: python gerar_cache_chat.py
"""
import json
from pathlib import Path

import pandas as pd
import requests

OUTPUTS = Path(r"C:\Projetos\Harbor\outputs")
DATASET_ROOT = Path(r"C:\Users\USER\Downloads\Projeto_HarboR-20260707T002634Z-3-001\Projeto_HarboR\Dataset")
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
CACHE_PATH = Path(__file__).parent / "cache_respostas_chat.json"


def call_ollama(prompt, timeout=180):
    try:
        resp = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as exc:
        return f"[Ollama indisponivel: {exc}]"


def montar_prompt(persona, readme_resumo, contexto_agregado, amostra_bruta, pergunta):
    return f"""{persona}

Voce pode explicar livremente o dataset, o contexto e a metodologia usando as informacoes abaixo
(README, resultados do pipeline, amostra de dados). A UNICA restricao e: nao invente metricas ou
numeros especificos (percentuais, medias, contagens) que nao estejam explicitamente escritos nos
"RESULTADOS JA CALCULADOS" ou na "AMOSTRA DE LINHAS BRUTAS" abaixo. Para perguntas conceituais ou
explicativas sobre o dataset, responda normalmente usando o README como base.

=== SOBRE O DATASET (README) ===
{readme_resumo}

=== RESULTADOS JA CALCULADOS PELO PIPELINE ===
{contexto_agregado}

=== AMOSTRA DE LINHAS BRUTAS DO CSV ===
{amostra_bruta}

=== PERGUNTA ===
{pergunta}

Responda em portugues, de forma direta e curta (2-4 frases)."""


def amostra_csv(path, n=8, **kwargs):
    try:
        return pd.read_csv(path, nrows=n, **kwargs).to_string()
    except Exception as exc:
        return f"(amostra indisponivel: {exc})"


def main():
    out1 = OUTPUTS / "pipeline1_oee"
    ds1 = DATASET_ROOT / "OEE"
    mttr_mtbf_txt = (out1 / "mttr_mtbf.json").read_text(encoding="utf-8")
    comparacao = pd.read_csv(out1 / "comparacao_antes_depois_lss.csv", index_col=0)
    agregacao = pd.read_csv(out1 / "agregacao_paradas.csv")

    out2 = OUTPUTS / "pipeline2_legacy_sensor"
    ds2 = DATASET_ROOT / "Legacy Industrial" / "archive"
    metrics_txt = (out2 / "backtest_metrics.json").read_text(encoding="utf-8")
    separacao = pd.read_csv(out2 / "separacao_features_por_classe.csv", index_col=0)
    vereditos = pd.read_csv(out2 / "vereditos_llm.csv")

    out3 = OUTPUTS / "pipeline3_discrete_manufacturing"
    ds3 = DATASET_ROOT / "SME-Manufacturing-Dataset-main"
    duracao_a = pd.read_csv(out3 / "duracao_estados_company_a.csv")
    pre_alarme = pd.read_csv(out3 / "estados_antes_do_alarme.csv", index_col=0)
    regime_a = pd.read_csv(out3 / "mudanca_regime_company_a.csv")
    consumo_b = pd.read_csv(out3 / "consumo_energia_por_status_company_b.csv", index_col=0)

    out4 = OUTPUTS / "pipeline4_five_axis_cnc"
    ds4 = DATASET_ROOT / "Five-Axis CNC Milling Dataset"
    ciclo = pd.read_csv(out4 / "ciclo_por_produto.csv")
    status_dist = pd.read_csv(out4 / "distribuicao_program_status.csv", index_col=0)
    anomalias_temp = pd.read_csv(out4 / "anomalias_temperatura.csv")
    anomaly_cols = [c for c in anomalias_temp.columns if c.endswith("_anomalo")]
    contagem_temp = anomalias_temp[anomaly_cols].sum()

    configs = {
        "oee": dict(
            persona=(
                "Especialista em Lean Six Sigma e OEE (Overall Equipment Effectiveness).\n"
                "Voce analisa uma linha de producao de tijolos (Kakoyiannis Bricks, Chipre, MES Evocon) "
                "e conhece MTTR, MTBF, Availability, Performance, Quality e categorias de parada "
                "(StopGroup, StopType, StopLocation)."
            ),
            contexto_agregado=(
                f"MTTR/MTBF (antes do LSS): {mttr_mtbf_txt}\n\n"
                f"Comparacao antes/depois LSS:\n{comparacao.to_string()}\n\n"
                f"Top motivos de parada:\n{agregacao.head(10).to_string()}"
            ),
            readme_resumo=(
                "Dataset da linha de producao continua de tijolos. Objetivo: analise de variabilidade de OEE, "
                "analise de downtime/modos de falha, avaliacao de MTTR/MTBF, identificacao de falhas criticas "
                "e locais criticos na linha. Dados coletados via MES Evocon. Existe versao antes e depois de "
                "implementacao de Lean Six Sigma (LSS)."
            ),
            amostra_bruta=amostra_csv(ds1 / "DowntimeDataset.csv", n=8, low_memory=False),
            exemplos=[
                "Qual foi o ganho percentual de OEE apos o Lean Six Sigma?",
                "Qual categoria de parada mais afeta a linha?",
                "O que e MTTR e como ele mudou?",
            ],
        ),
        "legacy_sensor": dict(
            persona=(
                "Especialista em manutencao preditiva e deteccao de anomalias em sensores industriais.\n"
                "Voce analisa logs de sensores de equipamentos industriais legados (temperatura, pressao, "
                "vibracao, corrente, etc.) e o desempenho de um modelo Isolation Forest validado contra "
                "o rotulo real Normal/Fault."
            ),
            contexto_agregado=(
                f"Metricas do backtest (Isolation Forest vs Target real): {metrics_txt}\n\n"
                f"Media das features numericas por classe (separabilidade):\n{separacao.to_string()}\n\n"
                f"Amostra de vereditos do LLM (camada 3):\n{vereditos.to_string()}"
            ),
            readme_resumo=(
                "Dados de sensores de equipamentos industriais antigos, incluindo notas do operador, "
                "mensagens de erro e uma coluna de status (Normal ou Fault). Util para analises de falhas. "
                "Colunas: Timestamp, Machine_ID, Temperature_C, Pressure_bar, Vibration_Level, Voltage_V, "
                "Current_A, Sound_dB, FlowRate_Lmin, Humidity_%, Oil_Quality_Index, Energy_Consumption_kWh, "
                "Production_Rate, Load_Percentage, Operator_Notes, Error_Message, Target."
            ),
            amostra_bruta=amostra_csv(ds2 / "industrial_dataset.csv", n=8),
            exemplos=[
                "Por que o recall do Isolation Forest foi tao baixo?",
                "Os sensores numericos conseguem distinguir Fault de Normal?",
                "O que a camada de LLM adiciona ao diagnostico?",
            ],
        ),
        "discrete_mfg": dict(
            persona=(
                "Especialista em analise de series temporais de manufatura discreta e deteccao de mudanca "
                "de regime (tecnica CUSUM, usada em series financeiras).\n"
                "Voce analisa dados de duas empresas anonimas (company_A e company_B) com estados de "
                "maquina (idle, manual, automatico, alarme) e consumo de energia."
            ),
            contexto_agregado=(
                f"Duracao por estado (company_A, top 10):\n{duracao_a.head(10).to_string()}\n\n"
                f"Estados que precedem o alarme:\n{pre_alarme.to_string()}\n\n"
                f"Mudancas de regime detectadas (CUSUM):\n{regime_a.to_string()}\n\n"
                f"Consumo de energia medio por status (company_B):\n{consumo_b.to_string()}"
            ),
            readme_resumo=(
                "Dataset de manufatura discreta de duas empresas anonimas. company_A.csv: timestamp, asset, "
                "items produzidos, status (0=idle, 1=manual, 2=automatico, 3=alarme), power_avg, cycle_time. "
                "company_B.csv: timestamp, asset, status textual (Alarm, Standby, MachineOn, Production, "
                "Loading, Tooling), tempos por estado, power_avg/min/max."
            ),
            amostra_bruta=amostra_csv(ds3 / "company_A.csv", n=8),
            exemplos=[
                "Qual estado de maquina mais precede o alarme?",
                "Quantas mudancas de regime foram detectadas no consumo de energia?",
                "Qual status consome mais energia em media?",
            ],
        ),
        "cnc": dict(
            persona=(
                "Especialista em usinagem CNC de 5 eixos e manutencao de maquinas-ferramenta.\n"
                "Voce analisa dados de um centro de usinagem Siemens 840D-SL / Spinner U5-620, com ciclo "
                "de producao por programa/produto e temperatura dos motores dos eixos."
            ),
            contexto_agregado=(
                f"Ciclo por produto (Program_path):\n{ciclo.to_string()}\n\n"
                f"Distribuicao de Program_status:\n{status_dist.to_string()}\n\n"
                f"Contagem de anomalias de temperatura por motor:\n{contagem_temp.to_string()}"
            ),
            readme_resumo=(
                "Dados de um processo de fresagem CNC de 5 eixos. Tres produtos diferentes foram fabricados, "
                "com matriz de changeover garantindo todas as combinacoes possiveis. Producao repetida 5 vezes "
                "(30 sessoes de manufatura). Dados registrados a partir de um controle Siemens 840D-SL em uma "
                "fresadora de 5 eixos 'Spinner U5-620'."
            ),
            amostra_bruta=amostra_csv(
                ds4 / "data_v1_0_1.csv", n=5,
                usecols=["time", "Program_path", "Program_status", "Cycle_time_program", "Spindle_motor_temperature"],
                low_memory=False,
            ),
            exemplos=[
                "Qual produto tem o maior tempo de ciclo?",
                "Qual motor teve mais leituras anomalas de temperatura?",
                "O que significa o Program_status na producao?",
            ],
        ),
    }

    cache = {}
    for dataset_key, cfg in configs.items():
        cache[dataset_key] = {}
        for pergunta in cfg["exemplos"]:
            print(f"[{dataset_key}] gerando: {pergunta}")
            prompt = montar_prompt(cfg["persona"], cfg["readme_resumo"], cfg["contexto_agregado"], cfg["amostra_bruta"], pergunta)
            resposta = call_ollama(prompt)
            cache[dataset_key][pergunta] = resposta
            print(f"  -> {resposta[:120]}")

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"\nCache salvo em {CACHE_PATH}")


if __name__ == "__main__":
    main()
