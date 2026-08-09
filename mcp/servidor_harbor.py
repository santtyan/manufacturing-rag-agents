"""
Servidor MCP do projeto Harbor (Eixo 3, C14-C16 do cronograma).

Expõe os recursos industriais do projeto como ferramentas MCP, para que qualquer cliente
compatível (Claude Desktop, MCP Inspector, etc.) possa consultar o sistema Harbor via o
protocolo padrão. Reaproveita a lógica já pronta de RAG neural, NL-to-SQL e diagnóstico.

Rodar (stdio transport, o mais simples):
    python C:\\Projetos\\Harbor\\mcp\\servidor_harbor.py

Testar com o MCP Inspector:
    npx @modelcontextprotocol/inspector python C:\\Projetos\\Harbor\\mcp\\servidor_harbor.py
"""
import sys

import requests
from mcp.server.fastmcp import FastMCP

# Reaproveita os modulos ja implementados do projeto.
sys.path.insert(0, r"C:\Projetos\Harbor\rag")
sys.path.insert(0, r"C:\Projetos\Harbor\nl_to_sql")

API_URL = "http://localhost:8000"
HARBOR_API_KEY = "harbor-demo-2026"

mcp = FastMCP("harbor-manufatura")


@mcp.tool()
def consultar_banco(pergunta: str) -> str:
    """Consulta o banco de manufatura (Postgres) em linguagem natural (NL-to-SQL).
    Traduz a pergunta em SQL, executa (somente SELECT) e retorna a tabela resultante.
    Use para perguntas quantitativas sobre paradas, OEE, sensores, ciclos de maquina."""
    from nl_to_sql import perguntar
    try:
        sql, resultado = perguntar(pergunta)
        return f"SQL gerado:\n{sql}\n\nResultado:\n{resultado.to_string()}"
    except Exception as exc:
        return f"Nao foi possivel responder via SQL: {exc}"


@mcp.tool()
def buscar_manual(pergunta: str) -> str:
    """Busca no manual tecnico de manutencao industrial (RAG hibrido, embeddings E5 + TF-IDF).
    Use para perguntas sobre procedimentos, limiares de seguranca, thresholds de
    temperatura/vibracao, arquitetura de diagnostico. Retorna trechos com citacao de fonte."""
    from rag_hibrido import RAGHibrido
    rag = RAGHibrido()
    rag.indexar()
    docs = rag.buscar(pergunta, k=3)
    if not docs:
        return "Nenhum trecho relevante encontrado no manual tecnico."
    return "\n\n".join(f"[Fonte: {d['fonte']} | score {d['score']}]\n{d['texto']}" for d in docs)


@mcp.tool()
def diagnosticar_leitura(
    Machine_ID: str,
    Temperature_C: float,
    Vibration_Level: float,
    Pressure_bar: float = 5.0,
) -> str:
    """Diagnostica uma leitura de sensor via a arquitetura hibrida de 3 camadas
    (regra deterministica -> Isolation Forest -> veredito LLM), chamando a API do projeto.
    Retorna o veredito de cada camada. Use para avaliar se uma leitura indica falha real."""
    payload = {
        "Machine_ID": Machine_ID,
        "Temperature_C": Temperature_C,
        "Vibration_Level": Vibration_Level,
        "Pressure_bar": Pressure_bar,
        "Voltage_V": 220.0, "Current_A": 10.0, "Sound_dB": 60.0,
        "FlowRate_Lmin": 50.0, "Humidity_pct": 40.0, "Oil_Quality_Index": 0.8,
        "Energy_Consumption_kWh": 100.0, "Production_Rate": 90.0, "Load_Percentage": 70.0,
        "Operator_Notes": "", "Error_Message": "",
    }
    try:
        resp = requests.post(
            f"{API_URL}/diagnostico", json=payload,
            headers={"X-API-Key": HARBOR_API_KEY}, timeout=60,
        )
        resp.raise_for_status()
        return str(resp.json())
    except Exception as exc:
        return f"Nao foi possivel diagnosticar (API offline?): {exc}"


@mcp.resource("harbor://esquema-banco")
def esquema_banco() -> str:
    """Esquema das tabelas do banco de manufatura Harbor (recurso de contexto)."""
    from nl_to_sql import ESQUEMA
    return ESQUEMA


if __name__ == "__main__":
    mcp.run()
