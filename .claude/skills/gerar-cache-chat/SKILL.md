---
name: gerar-cache-chat
description: Pré-gera e atualiza o cache de respostas do chat (dashboard/cache_respostas_chat.json) para as perguntas de exemplo de cada chatbot especializado, evitando depender do Ollama ao vivo durante demo/reunião. Use antes de qualquer reunião/apresentação que use o dashboard Streamlit, ou depois de mudar outputs de pipeline que alimentam o contexto desses chatbots (mttr_mtbf.json, backtest_metrics.json, duracao_estados_*, ciclo_por_produto.csv, etc).
---

# Gerar cache do chat para demo/reunião

## Quando rodar

- Antes de qualquer reunião/apresentação onde o dashboard Streamlit será demonstrado ao vivo (a docstring do script é literal: "Rodar antes da reuniao").
- Depois de rodar qualquer um dos 4 primeiros pipelines de novo (OEE, Legacy Sensor, Discrete Manufacturing, CNC) e os outputs mudarem — o cache ficaria com números antigos.
- Se o usuário mencionar "internet instável", "Ollama pode cair na demo" ou similar.

## Pré-requisitos

- Ollama rodando (`http://localhost:11434`) — o script chama o LLM ao vivo para gerar cada resposta, só depois é que ela vira "cache". Se não tiver certeza que está no ar, use a skill `subir-servicos` primeiro.
- Outputs dos pipelines 1-4 já existirem em `outputs/pipeline{1,2,3,4}_*/` (o script lê CSVs/JSONs de lá, falha se não existirem).

## Rodar

```powershell
cd dashboard
python gerar_cache_chat.py
```

Gera respostas para os 4 chatbots especializados (`oee`, `legacy_sensor`, `discrete_mfg`, `cnc`), 3 perguntas de exemplo cada, salvando em `dashboard/cache_respostas_chat.json`. Demora alguns minutos (12 chamadas ao Ollama, ~llama3.2 local).

## Depois de rodar

- Confira no console que nenhuma resposta veio como `[Ollama indisponivel: ...]` — se aparecer, o Ollama caiu no meio da geração; suba de novo e rode só a parte afetada (ou rode tudo de novo, é idempotente).
- O dashboard (`dashboard/app.py`) usa esse cache automaticamente quando a pergunta do usuário bate com uma das perguntas de exemplo — não precisa reiniciar o Streamlit para o cache novo valer, mas vale confirmar visualmente no dashboard se o texto mudou como esperado.

## Se quiser adicionar/mudar uma pergunta de exemplo

As perguntas e o contexto de cada chatbot estão no dicionário `configs` dentro de `main()` (`dashboard/gerar_cache_chat.py`). Cada dataset tem `persona`, `contexto_agregado` (resultados já calculados pelo pipeline), `readme_resumo` e `exemplos` (lista de perguntas). Editar `exemplos` e rodar o script de novo é o fluxo correto — não edite `cache_respostas_chat.json` manualmente.
