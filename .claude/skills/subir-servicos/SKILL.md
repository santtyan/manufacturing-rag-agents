---
name: subir-servicos
description: Sobe (ou verifica a saúde de) toda a infraestrutura do projeto Harbor — Docker (Postgres+N8N), Ollama e Streamlit. Use quando o usuário pedir para "subir o Harbor", "iniciar os serviços", "testar se tá tudo no ar", ou quando um comando falhar por causa de um serviço fora do ar (conexão recusada na 8501/5432/11434).
---

# Subir e verificar serviços do Harbor

## Fluxo padrão

1. Rode o script principal, que já sobe tudo na ordem certa e reporta saúde no final:

```powershell
powershell -ExecutionPolicy Bypass -File start_all.ps1
```

Ordem: Docker Desktop → containers Postgres+N8N (`infra/docker-compose.yml`) → Ollama (`ollama serve`) → Streamlit (porta 8501).

2. Leia o resumo final do script (`[OK]`/`[FALHA]` por serviço). Se tudo `[OK]`, está pronto — não precisa investigar mais nada.

3. Se algo falhar, diagnostique o serviço específico antes de reiniciar tudo:

| Serviço | Sintoma | Causa comum |
|---|---|---|
| Docker | `docker ps` falha | Docker Desktop não abriu a tempo (o script já tenta abrir e espera até 60s) |
| Postgres/N8N | Container não sobe | Rodar `docker compose up -d` manualmente dentro de `infra/` e ler o log com `docker logs harbor_postgres` / `docker logs harbor_n8n` |
| Ollama | `11434` não responde | Processo `ollama` não está rodando — `ollama serve` em background |
| Streamlit | `8501` não responde | Ver o console do streamlit; demora ~6s para subir, o script já espera |

## Subir um serviço isolado (sem rodar o script todo)

```powershell
# Dashboard (de dentro de dashboard/)
python -m streamlit run app.py --server.port 8501
```

## Checar saúde sem subir nada

```powershell
python tests/smoke_test.py
```

Isso confere outputs dos pipelines e os endpoints de serviço (Ollama/Streamlit/N8N). Mais completo que só olhar o resumo do `start_all.ps1`, mas assume que os serviços já estão no ar (não sobe nada sozinho).

## Endereços

- Dashboard: http://localhost:8501
- N8N: http://localhost:5678

## Atenção

API FastAPI de diagnóstico (`/diagnostico`, porta 8000) e servidor MCP foram removidos em
2026-09-12 — entregas planejadas para reimplementação futura, ver
`docs/mapeamento_cronograma.md`. Este skill não cobre mais esses dois serviços.
