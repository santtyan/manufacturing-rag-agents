"""
API REST de diagnostico de falhas - Trilha 2 (C4, C5, C13 do cronograma).
Envolve a arquitetura hibrida de 3 camadas (regra/estatistica -> Isolation Forest -> LLM)
do pipeline 2 (Legacy Sensor Logs) num endpoint FastAPI com Swagger automatico.

Rodar com: uvicorn main:app --reload --port 8000
Swagger em: http://localhost:8000/docs
"""
import json
import os
import secrets
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import requests
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest

DATA_PATH = Path(r"C:\Users\USER\Downloads\Projeto_HarboR-20260707T002634Z-3-001\Projeto_HarboR\Dataset\Legacy Industrial\archive\industrial_dataset.csv")
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

# API key simples via header -- gerada automaticamente se nao houver HARBOR_API_KEY no ambiente.
# /health e / ficam publicos (uteis para checagem de saude sem credencial); /diagnostico e /amostra exigem a chave.
API_KEY = os.environ.get("HARBOR_API_KEY") or secrets.token_urlsafe(24)
if not os.environ.get("HARBOR_API_KEY"):
    print(f"[aviso] HARBOR_API_KEY nao definida -- chave gerada para esta sessao: {API_KEY}")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verificar_api_key(chave: str = Security(api_key_header)):
    if chave != API_KEY:
        raise HTTPException(status_code=401, detail="X-API-Key ausente ou invalida")
    return chave

FEATURE_COLS = [
    "Temperature_C", "Pressure_bar", "Vibration_Level", "Voltage_V", "Current_A",
    "Sound_dB", "FlowRate_Lmin", "Humidity_%", "Oil_Quality_Index",
    "Energy_Consumption_kWh", "Production_Rate", "Load_Percentage",
]

app = FastAPI(
    title="Harbor - API de Diagnostico de Falhas",
    description="Arquitetura hibrida de 3 camadas: regra estatistica (Isolation Forest) + validacao semantica (LLM local via Ollama). Padrao de arquitetura ja validado por Juliano/Vinicius no CERISE.",
    version="1.0.0",
)

_state = {"df": None, "model": None}


class LeituraSensor(BaseModel):
    Machine_ID: int
    Temperature_C: float
    Pressure_bar: float
    Vibration_Level: float
    Voltage_V: float
    Current_A: float
    Sound_dB: float
    FlowRate_Lmin: float
    Humidity_pct: float
    Oil_Quality_Index: float
    Energy_Consumption_kWh: float
    Production_Rate: float
    Load_Percentage: float
    Operator_Notes: Optional[str] = ""
    Error_Message: Optional[str] = ""


VerdictoLLM = Literal["REAL", "FALSO_POSITIVO", "INCONCLUSIVO", "NAO_APLICAVEL"]


class DiagnosticoResponse(BaseModel):
    camada1_regra: str
    anomalia_estatistica: bool
    veredito_llm: VerdictoLLM
    veredito_final: Literal["CRITICO", "ALERTA", "REAL", "FALSO_POSITIVO", "INCONCLUSIVO", "NORMAL"]
    detalhe: str


# Camada 1 -- thresholds fixos, extraidos do manual tecnico (rag/manuais/manual_manutencao_sensores_industriais.md).
# Sao checados ANTES do Isolation Forest: uma violacao critica dispensa a camada estatistica.
def checar_regra_deterministica(leitura: "LeituraSensor"):
    if leitura.Temperature_C > 90 or leitura.Vibration_Level > 2.0:
        return "CRITICO", "Temperature_C > 90C ou Vibration_Level > 2.0 -- parada imediata recomendada (regra do manual)."
    if leitura.Temperature_C > 85:
        return "ALERTA", "Temperature_C > 85C -- risco de superaquecimento (regra do manual)."
    if leitura.Pressure_bar > 5.0 or leitura.Pressure_bar < 2.0:
        return "ALERTA", "Pressure_bar fora da faixa segura (2.5-4.5 bar, regra do manual)."
    if leitura.Vibration_Level > 1.5:
        return "ALERTA", "Vibration_Level > 1.5 -- possivel falha mecanica iminente (regra do manual)."
    return "NORMAL", "Nenhum threshold fixo violado."


def carregar_modelo():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].values
    model = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
    model.fit(X)
    _state["df"] = df
    _state["model"] = model


@app.on_event("startup")
def startup_event():
    carregar_modelo()


def call_ollama(prompt, timeout=120):
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as exc:
        return f"[Ollama indisponivel: {exc}]"


def call_ollama_veredito(prompt, timeout=120) -> VerdictoLLM:
    """Veredito da camada 3 com saida estruturada (Parte 4 padrao-ouro): forca o Ollama a
    retornar {"veredito": "REAL"|"FALSO_POSITIVO"|"INCONCLUSIVO"} via format JSON-schema,
    em vez de texto livre + .strip().upper() (fragil a variacao de pontuacao/idioma do LLM)."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                "format": {
                    "type": "object",
                    "properties": {"veredito": {"type": "string",
                                                 "enum": ["REAL", "FALSO_POSITIVO", "INCONCLUSIVO"]}},
                    "required": ["veredito"],
                },
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        veredito = json.loads(resp.json().get("response", "{}")).get("veredito")
        return veredito if veredito in ("REAL", "FALSO_POSITIVO", "INCONCLUSIVO") else "INCONCLUSIVO"
    except Exception:
        return "INCONCLUSIVO"


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    ollama_ok = True
    try:
        requests.get("http://localhost:11434/api/tags", timeout=3)
    except Exception:
        ollama_ok = False
    return {
        "api": "ok",
        "modelo_isolation_forest": "carregado" if _state["model"] is not None else "nao carregado",
        "ollama": "disponivel" if ollama_ok else "indisponivel",
    }


@app.post("/diagnostico", response_model=DiagnosticoResponse)
def diagnosticar(leitura: LeituraSensor, _: str = Depends(verificar_api_key)):
    """
    Camada 1: regra deterministica (thresholds fixos) -- roda primeiro, sem custo de ML/LLM.
    Camada 2: Isolation Forest classifica a leitura como anomala ou normal (captura combinacoes
              multivariadas que a regra isolada nao pegaria).
    Camada 3: se anomala (por regra ou por ML), o LLM local valida com base nas notas/erros e
              decide REAL/FALSO_POSITIVO.
    """
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Modelo nao carregado ainda")

    camada1_status, camada1_detalhe = checar_regra_deterministica(leitura)

    valores = [
        leitura.Temperature_C, leitura.Pressure_bar, leitura.Vibration_Level,
        leitura.Voltage_V, leitura.Current_A, leitura.Sound_dB, leitura.FlowRate_Lmin,
        leitura.Humidity_pct, leitura.Oil_Quality_Index, leitura.Energy_Consumption_kWh,
        leitura.Production_Rate, leitura.Load_Percentage,
    ]
    X = np.array(valores).reshape(1, -1)
    predicao = _state["model"].predict(X)[0]
    anomalo_ml = predicao == -1
    anomalo = anomalo_ml or camada1_status != "NORMAL"

    if not anomalo:
        return DiagnosticoResponse(
            camada1_regra=camada1_status,
            anomalia_estatistica=False,
            veredito_llm="NAO_APLICAVEL",
            veredito_final="NORMAL",
            detalhe=f"{camada1_detalhe} Isolation Forest tambem nao detectou anomalia.",
        )

    prompt = f"""Voce e um especialista em manutencao industrial. Um algoritmo estatistico (Isolation Forest)
marcou a leitura abaixo como ANOMALA. Analise e decida se e uma falha REAL, um FALSO_POSITIVO ou se e
INCONCLUSIVO com os dados disponiveis.

Maquina: {leitura.Machine_ID}
Temperatura: {leitura.Temperature_C} C
Pressao: {leitura.Pressure_bar} bar
Vibracao: {leitura.Vibration_Level}
Nota do operador: {leitura.Operator_Notes}
Mensagem de erro: {leitura.Error_Message}"""

    veredito = call_ollama_veredito(prompt)

    origem = []
    if camada1_status != "NORMAL":
        origem.append(f"regra ({camada1_status})")
    if anomalo_ml:
        origem.append("Isolation Forest")

    # PRECEDENCIA: quando a regra deterministica (camada 1) ja e definitiva (CRITICO), ela
    # prevalece sobre o veredito do LLM (camada 3) em caso de discordancia. Respaldo: T7/
    # DiagnosticIQ da revisao de literatura -- "LLM e fragil em decisao critica, deve ser
    # combinado com regra deterministica, nao ter a palavra final sozinho". Achado real do
    # Harbor (sessao anterior): leitura com Temperature_C=94 disparou CRITICO na camada 1, mas
    # o LLM respondeu FALSO_POSITIVO -- a camada 1 deveria ter vencido e nao vencia.
    if camada1_status == "CRITICO":
        veredito_final = "CRITICO"
        detalhe = (f"Anomalia sinalizada por: {', '.join(origem)}. {camada1_detalhe} "
                   f"A regra deterministica (camada 1) e CRITICA e tem PRECEDENCIA sobre o "
                   f"veredito da camada 3 (LLM disse: {veredito}) -- decisao final: CRITICO.")
    elif camada1_status == "ALERTA" and veredito == "FALSO_POSITIVO":
        # regra deu alerta mas nao e critica; ainda assim nao aceitamos "falso positivo" cego
        # do LLM quando ha uma regra de alerta ativa -- rebaixa para INCONCLUSIVO em vez de
        # descartar a leitura.
        veredito_final = "INCONCLUSIVO"
        detalhe = (f"Anomalia sinalizada por: {', '.join(origem)}. {camada1_detalhe} "
                   f"A regra deu ALERTA mas o LLM disse FALSO_POSITIVO -- discordancia tratada "
                   f"como INCONCLUSIVO em vez de descartar, ate revisao humana.")
    else:
        veredito_final = veredito
        detalhe = (f"Anomalia sinalizada por: {', '.join(origem)}. {camada1_detalhe} "
                   f"Veredito da camada 3 (LLM): {veredito}.")

    return DiagnosticoResponse(
        camada1_regra=camada1_status,
        anomalia_estatistica=True,
        veredito_llm=veredito,
        veredito_final=veredito_final,
        detalhe=detalhe,
    )


@app.get("/amostra")
def amostra(n: int = 5, _: str = Depends(verificar_api_key)):
    """Retorna uma amostra de leituras reais do dataset, uteis para testar o endpoint /diagnostico."""
    if _state["df"] is None:
        raise HTTPException(status_code=503, detail="Dados nao carregados ainda")
    df = _state["df"].sample(n)
    return df.to_dict(orient="records")
