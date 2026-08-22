---
name: subir-servicos
description: Sobe (ou verifica a saúde de) toda a infraestrutura do projeto Harbor — Docker (Postgres+N8N), Ollama, FastAPI e Streamlit. Use quando o usuário pedir para "subir o Harbor", "iniciar os serviços", "testar se tá tudo no ar", ou quando um comando falhar por causa de um serviço fora do ar (conexão recusada na 8000/8501/5432/11434).
---

# Subir e verificar serviços do Harbor

## Fluxo padrão

1. Rode o script principal, que já sobe tudo na ordem certa e reporta saúde no final:

```powershell
powershell -ExecutionPolicy Bypass -File start_all.ps1
```

Ordem: Docker Desktop → containers Postgres+N8N (`infra/docker-compose.yml`) → Ollama (`ollama serve`) → FastAPI (porta 8000) → Streamlit (porta 8501).

2. Leia o resumo final do script (`[OK]`/`[FALHA]` por serviço). Se tudo `[OK]`, está pronto — não precisa investigar mais nada.

3. Se algo falhar, diagnostique o serviço específico antes de reiniciar tudo:

| Serviço | Sintoma | Causa comum |
|---|---|---|
| Docker | `docker ps` falha | Docker Desktop não abriu a tempo (o script já tenta abrir e espera até 60s) |
| Postgres/N8N | Container não sobe | Rodar `docker compose up -d` manualmente dentro de `infra/` e ler o log com `docker logs harbor_postgres` / `docker logs harbor_n8n` |
| Ollama | `11434` não responde | Processo `ollama` não está rodando — `ollama serve` em background |
| FastAPI | `8000/health` falha | Ver o console do uvicorn; erro comum é `HARBOR_API_KEY` não definida (não impede subir, mas gera chave aleatória por sessão — ver aviso no console) |
| Streamlit | `8501` não responde | Ver o console do streamlit; demora ~6s para subir, o script já espera |

## Subir um serviço isolado (sem rodar o script todo)

```powershell
# API (de dentro de api/)
python -m uvicorn main:app --reload --port 8000

# Dashboard (de dentro de dashboard/)
python -m streamlit run app.py --server.port 8501

# Servidor MCP (stdio transport, não abre porta HTTP)
python C:\Projetos\Harbor\mcp\servidor_harbor.py
```

## Checar saúde sem subir nada

```powershell
python tests/smoke_test.py
```

Isso confere outputs dos pipelines E os 4 endpoints de serviço (Ollama/FastAPI/Streamlit/N8N) E o fluxo ponta a ponta de `/diagnostico`. Mais completo que só olhar o resumo do `start_all.ps1`, mas assume que os serviços já estão no ar (não sobe nada sozinho).

## Endereços

- Dashboard: http://localhost:8501
- API Docs (Swagger): http://localhost:8000/docs
- N8N: http://localhost:5678
- Webhook: http://localhost:5678/webhook/diagnostico-automatico

## Atenção

`HARBOR_API_KEY` não definida faz `api/main.py` gerar uma chave aleatória por sessão (impressa no console do uvicorn) — sem ela, `/diagnostico` e `/amostra` retornam 401. Se for testar a API via script separado, defina a variável antes de subir o serviço.
