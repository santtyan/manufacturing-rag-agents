"""
Trilha 4 - Segundo workflow N8N: roda sozinho de tempos em tempos (cron), sem depender de
um webhook manual. Mais fiel ao padrao de automacao continua do cronograma (D3-D9):
"monitoramento continuo" em vez de "responde quando chamado".
"""
import os

import requests

N8N_URL = "http://localhost:5678"
API_KEY = os.environ.get("N8N_API_KEY")
if not API_KEY:
    raise SystemExit(
        "Defina a variavel de ambiente N8N_API_KEY antes de rodar este script.\n"
        "Gere uma chave em: Settings > n8n API > Create an API key (na UI do N8N)."
    )

HARBOR_API_KEY = os.environ.get("HARBOR_API_KEY", "harbor-demo-2026")
HEADERS = {"X-N8N-API-KEY": API_KEY, "Content-Type": "application/json"}

workflow = {
    "name": "Harbor - Monitoramento Continuo (Agendado)",
    "nodes": [
        {
            "parameters": {
                "rule": {
                    "interval": [{"field": "minutes", "minutesInterval": 5}]
                }
            },
            "id": "cron_trigger",
            "name": "A cada 5 minutos",
            "type": "n8n-nodes-base.scheduleTrigger",
            "typeVersion": 1.2,
            "position": [0, 0],
        },
        {
            "parameters": {
                "url": "http://host.docker.internal:8000/amostra?n=1",
                "sendHeaders": True,
                "headerParameters": {
                    "parameters": [{"name": "X-API-Key", "value": HARBOR_API_KEY}]
                },
                "options": {},
            },
            "id": "buscar_amostra",
            "name": "Buscar amostra do sensor",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [220, 0],
        },
        {
            "parameters": {
                "method": "POST",
                "url": "http://host.docker.internal:8000/diagnostico",
                "sendHeaders": True,
                "headerParameters": {
                    "parameters": [{"name": "X-API-Key", "value": HARBOR_API_KEY}]
                },
                "sendBody": True,
                "specifyBody": "keypair",
                "bodyParameters": {
                    "parameters": [
                        {"name": "Machine_ID", "value": "={{ $json.Machine_ID }}"},
                        {"name": "Temperature_C", "value": "={{ $json.Temperature_C }}"},
                        {"name": "Pressure_bar", "value": "={{ $json.Pressure_bar }}"},
                        {"name": "Vibration_Level", "value": "={{ $json.Vibration_Level }}"},
                        {"name": "Voltage_V", "value": "={{ $json.Voltage_V }}"},
                        {"name": "Current_A", "value": "={{ $json.Current_A }}"},
                        {"name": "Sound_dB", "value": "={{ $json.Sound_dB }}"},
                        {"name": "FlowRate_Lmin", "value": "={{ $json.FlowRate_Lmin }}"},
                        {"name": "Humidity_pct", "value": "={{ $json['Humidity_%'] }}"},
                        {"name": "Oil_Quality_Index", "value": "={{ $json.Oil_Quality_Index }}"},
                        {"name": "Energy_Consumption_kWh", "value": "={{ $json.Energy_Consumption_kWh }}"},
                        {"name": "Production_Rate", "value": "={{ $json.Production_Rate }}"},
                        {"name": "Load_Percentage", "value": "={{ $json.Load_Percentage }}"},
                        {"name": "Operator_Notes", "value": "={{ $json.Operator_Notes }}"},
                        {"name": "Error_Message", "value": "={{ $json.Error_Message }}"},
                    ]
                },
                "options": {"bodyContentType": "json"},
            },
            "id": "chamar_diagnostico",
            "name": "Chamar API de diagnostico",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [440, 0],
        },
        {
            "parameters": {
                "conditions": {
                    "options": {"caseSensitive": True, "leftValue": "", "typeValidation": "strict"},
                    "conditions": [
                        {
                            "id": "cond_anomalia",
                            "leftValue": "={{ $json.anomalia_estatistica }}",
                            "rightValue": True,
                            "operator": {"type": "boolean", "operation": "equals"},
                        }
                    ],
                    "combinator": "and",
                },
                "options": {},
            },
            "id": "checar_anomalia",
            "name": "Anomalia detectada?",
            "type": "n8n-nodes-base.if",
            "typeVersion": 2.2,
            "position": [660, 0],
        },
    ],
    "connections": {
        "A cada 5 minutos": {
            "main": [[{"node": "Buscar amostra do sensor", "type": "main", "index": 0}]]
        },
        "Buscar amostra do sensor": {
            "main": [[{"node": "Chamar API de diagnostico", "type": "main", "index": 0}]]
        },
        "Chamar API de diagnostico": {
            "main": [[{"node": "Anomalia detectada?", "type": "main", "index": 0}]]
        },
    },
    "settings": {"executionOrder": "v1"},
}


def deletar_workflows_existentes():
    resp = requests.get(f"{N8N_URL}/api/v1/workflows", headers=HEADERS, timeout=15)
    for wf in resp.json().get("data", []):
        if wf["name"] == workflow["name"]:
            requests.delete(f"{N8N_URL}/api/v1/workflows/{wf['id']}", headers=HEADERS, timeout=15)
            print(f"Workflow antigo removido: {wf['id']}")


def main():
    deletar_workflows_existentes()
    resp = requests.post(f"{N8N_URL}/api/v1/workflows", headers=HEADERS, json=workflow, timeout=30)
    if resp.status_code not in (200, 201):
        print(f"Erro ao criar workflow: {resp.status_code} - {resp.text}")
        return

    data = resp.json()
    workflow_id = data["id"]
    print(f"Workflow criado: {workflow_id}")

    ativar = requests.post(f"{N8N_URL}/api/v1/workflows/{workflow_id}/activate", headers=HEADERS, timeout=30)
    print(f"Ativacao: {ativar.status_code} - {ativar.text[:200]}")
    print("\nEste workflow roda sozinho a cada 5 minutos -- sem necessidade de chamada manual.")
    print("Verifique execucoes em: http://localhost:5678 > Executions")


if __name__ == "__main__":
    main()
