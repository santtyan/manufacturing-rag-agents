# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## O que é este projeto

Harbor é um projeto de IA aplicada a manutenção industrial (bolsa FUNAPE/CERISE): pipelines de análise sobre 7 datasets industriais, um chatbot que roteia perguntas entre contexto pré-calculado / RAG sobre manuais técnicos / NL-to-SQL sobre um Postgres, uma API de diagnóstico de falhas (arquitetura de 3 camadas: regra determinística → Isolation Forest → LLM), um servidor MCP, e um harness próprio de avaliação (além de benchmarks acadêmicos: NanoBEIR, BIRD-SQL, Spider).

Todo o código, comentários e docstrings estão em português. Os comentários frequentemente documentam um "achado real" (bug encontrado ao vivo, com data) que motivou a regra atual — leia-os antes de tocar em roteamento/gates, eles carregam contexto que não está em nenhum outro lugar.

## Subir o ambiente

```powershell
powershell -ExecutionPolicy Bypass -File start_all.ps1
```

Sobe, nessa ordem: Docker Desktop → containers Postgres+N8N (`infra/docker-compose.yml`) → Ollama (`ollama serve`) → FastAPI (porta 8000) → Streamlit (porta 8501). Verifica saúde de cada serviço no final.

Serviços manuais, se preferir subir peça por peça:
```powershell
# API (de dentro de api/)
python -m uvicorn main:app --reload --port 8000

# Dashboard (de dentro de dashboard/)
python -m streamlit run app.py --server.port 8501

# Servidor MCP (stdio transport)
python C:\Projetos\Harbor\mcp\servidor_harbor.py
```

`HARBOR_API_KEY` não definida faz `api/main.py` gerar uma chave aleatória por sessão e imprimi-la no console — sem isso, `/diagnostico` e `/amostra` retornam 401. Defina a variável de ambiente antes de subir a API se for testar via script.

## Testes e avaliação

```powershell
# Smoke test (sem pytest — indisponível por instabilidade de rede na máquina)
python tests/smoke_test.py

# Harness de roteamento + faithfulness sobre golden_questions.json
python eval/rodar_golden.py

# Consistência (mesma pergunta repetida, mede variação de resposta)
python eval/rodar_consistencia.py

# Benchmarks acadêmicos de retrieval / NL-to-SQL
python eval/avaliar_retrieval_nanobeir.py
python eval/avaliar_bird_sql.py [N_PERGUNTAS] [OLLAMA_MODEL]
python eval/avaliar_spider_sql.py [N_PERGUNTAS] [OLLAMA_MODEL]
```

`eval/rodar_golden.py` importa `dashboard/roteador.py` e `eval/rag_gerador.py` — **nunca duplique lógica de roteamento ou geração RAG dentro de `eval/`**; isso já causou uma regressão fantasma de 67,9%→52% (ver histórico em `dashboard/roteador.py`). Se adicionar um gate novo de roteamento, ele deve viver só em `dashboard/roteador.py`.

## Arquitetura

### Módulos compartilhados (fonte única de verdade)

- **`dashboard/roteador.py`** — decide se uma pergunta do chat vai para `contexto` (dados já calculados pelos pipelines), `rag` (manuais técnicos) ou `sql` (NL-to-SQL sobre o Postgres), mais uma série de "answerability gates" determinísticos (`pede_*`) que interceptam ANTES do LLM perguntas que ele historicamente alucinava (cruzamentos impossíveis no schema, ROI inventado, direção de métrica invertida, etc). Roteamento híbrido: keyword primeiro (rápido), LLM (Ollama, saída JSON-schema) só como desempate quando a keyword cai em `contexto` por default. Importado por `dashboard/app.py` e `eval/rodar_golden.py` — era duplicado manualmente em 3 lugares até 2026-08-07 (ver comentário no topo do arquivo).
- **`rag/rag_hibrido.py`** (classe `RAGHibrido`) — motor de busca sobre `rag/manuais/*.md`: embeddings E5 (`intfloat/multilingual-e5-small`, com prefixos `passage:`/`query:` obrigatórios) + busca lexical (TF-IDF por default, BM25 opcional via `RAGHibrido(lexico="bm25")` — ver skill `comparar-tfidf-bm25`) + rerank Cross-Encoder (`ms-marco-MiniLM-L-6-v2`). Chunking por seção Markdown (`## `), não por caractere fixo — ver docstring de `chunk_texto()` para a ressalva de que isso não é uma "melhor prática universal", é ajustado ao corpus atual. Persistido em ChromaDB (`rag/chroma_db/`). Score do BM25 é normalizado min-max por query antes de entrar no sort combinado — não é cosseno como TF-IDF/E5, e sem normalizar degradava Recall@5 no modo sem rerank (achado real, 2026-08-22).
- **`nl_to_sql/nl_to_sql.py`** — traduz pergunta em português para SQL via Ollama, valida que só é `SELECT`, executa no Postgres (`harbor_manufatura`), com self-repair e um "DBA-Agent" de segunda opinião. `ESQUEMA` no topo do arquivo documenta cada tabela por dataset de origem — **nunca cruzar tabelas de datasets diferentes** (ex.: company_A e company_B do dataset 3 usam schemas de status incompatíveis).
- **`eval/rag_gerador.py`** — geração de resposta RAG compartilhada entre `dashboard/app.py` e o harness, pelo mesmo motivo do roteador.

### Serviços

- **`api/main.py`** (FastAPI, porta 8000) — endpoint `/diagnostico`: 3 camadas em sequência — regra determinística (thresholds fixos do manual) → Isolation Forest (scikit-learn) → veredito do LLM local (Ollama, saída estruturada). A regra determinística tem **precedência** sobre o LLM quando `CRITICO` (achado real: LLM discordava de uma leitura obviamente crítica). Autenticação via header `X-API-Key`.
- **`dashboard/app.py`** (Streamlit, porta 8501) — chat principal, consome `dashboard/roteador.py` + `rag/rag_hibrido.py` + `nl_to_sql/nl_to_sql.py`. Contém `st.set_page_config()` em nível de módulo, por isso não é importável fora do Streamlit — é o motivo de `roteador.py` e `rag_gerador.py` terem sido extraídos como módulos separados.
- **`mcp/servidor_harbor.py`** — expõe `consultar_banco`, `buscar_manual`, `diagnosticar_leitura` como ferramentas MCP (FastMCP, stdio transport), reaproveitando os mesmos módulos acima.
- **`infra/docker-compose.yml`** — Postgres 16 (`harbor_manufatura`) + N8N.

### Pipelines (`pipelines/`)

7 pipelines, um por dataset, cada um gravando outputs em `outputs/pipelineN_*/` (CSVs e JSONs consumidos pelo dashboard como "contexto" pré-calculado — a rota mais confiável do roteador, porque não depende do LLM calcular nada). Datasets: OEE/Downtime, Legacy Sensor Logs, Discrete Manufacturing (2 empresas anônimas), Five-Axis CNC Milling, Facility Maintenance, Labeled Car, Aircraft Annotation.

### Golden set e gates

`eval/golden_questions.json` é o conjunto de perguntas de referência do harness. Ao adicionar um gate novo em `roteador.py` para corrigir uma alucinação encontrada manualmente, adicione também a pergunta que expôs o bug ao golden set — é assim que o harness evita regressão silenciosa.

## Skills do projeto (`.claude/skills/`)

Fluxos recorrentes já empacotados como skills — usar em vez de reimprovisar o procedimento do zero:

- **`subir-servicos`** — sobe/diagnostica toda a infraestrutura (`start_all.ps1` + diagnóstico por serviço).
- **`rodar-harness`** — roda o harness de roteamento/faithfulness e os benchmarks acadêmicos, interpreta resultados.
- **`adicionar-gate-roteamento`** — fluxo completo para quando o chat aluciona: gate determinístico em `roteador.py` + golden set.
- **`gerar-cache-chat`** — pré-gera `dashboard/cache_respostas_chat.json` antes de reunião/demo.
- **`adicionar-manual-rag`** — adiciona manual novo em `rag/manuais/`, reindexa, valida Recall@k/MRR.
- **`atualizar-slide-resposta`** — cria/atualiza resposta preparada em `slides/*.md` com números atuais.
- **`comparar-tfidf-bm25`** — roda o benchmark que compara os dois algoritmos lexicais de `rag_hibrido.py`.
- **`roadmap-slm-multiagente`** — guia vivo do fit do Harbor com o Projeto 1 do PDC (multiagentes confiáveis + SLMs em português); lista priorizada de itens implementáveis.
