# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## O que é este projeto

Harbor é um projeto de IA aplicada a manutenção industrial (bolsa FUNAPE/CERISE): pipelines de análise sobre 8 datasets industriais, um chatbot que roteia perguntas entre contexto pré-calculado / RAG sobre manuais técnicos / NL-to-SQL sobre um Postgres, e um harness próprio de avaliação (além de benchmarks acadêmicos: NanoBEIR, BIRD-SQL, Spider e, desde 2026-09-14, o benchmark oficial do OpenPack via `openpack-torch`). Uma API de diagnóstico de falhas (arquitetura de 3 camadas: regra determinística → Isolation Forest → LLM) e um servidor MCP existiram no projeto e foram removidos em 2026-09-12 — entregas planejadas para reimplementação futura, ver `docs/mapeamento_cronograma.md`.

Todo o código, comentários e docstrings estão em português. Os comentários frequentemente documentam um "achado real" (bug encontrado ao vivo, com data) que motivou a regra atual — leia-os antes de tocar em roteamento/gates, eles carregam contexto que não está em nenhum outro lugar.

## Subir o ambiente

```powershell
powershell -ExecutionPolicy Bypass -File start_all.ps1
```

Sobe, nessa ordem: Docker Desktop → container Postgres (`infra/docker-compose.yml`) → Ollama (`ollama serve`) → Streamlit (porta 8501). Verifica saúde de cada serviço no final.

Serviços manuais, se preferir subir peça por peça:
```powershell
# Dashboard (de dentro de dashboard/)
python -m streamlit run app.py --server.port 8501
```

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

# OpenPack (RAG-HAR training-free): retrieval por categoria (k-NN label purity, não Recall@k
# — ver nota na skill rodar-harness) e classificação F1-macro vs. benchmark oficial
python eval/avaliar_retrieval_openpack.py
python eval/avaliar_classificacao_openpack.py [--k 5]

# RAG multimodal de gráfico técnico (geração determinística, sem VLM — ver skill rag-multimodal)
python rag/legendas_deterministicas.py
python eval/checks_fidelidade_caption.py rag/legendas_deterministicas.json
python eval/avaliar_rag_multimodal.py --cache-legendas rag/legendas_deterministicas.json
```

`eval/rodar_golden.py` importa `dashboard/roteador.py` e `eval/rag_gerador.py` — **nunca duplique lógica de roteamento ou geração RAG dentro de `eval/`**; isso já causou uma regressão fantasma de 67,9%→52% (ver histórico em `dashboard/roteador.py`). Se adicionar um gate novo de roteamento, ele deve viver só em `dashboard/roteador.py`.

## Arquitetura

### Módulos compartilhados (fonte única de verdade)

- **`dashboard/roteador.py`** — decide se uma pergunta do chat vai para `contexto` (dados já calculados pelos pipelines), `rag` (manuais técnicos) ou `sql` (NL-to-SQL sobre o Postgres), mais uma série de "answerability gates" determinísticos (`pede_*`) que interceptam ANTES do LLM perguntas que ele historicamente alucinava (cruzamentos impossíveis no schema, ROI inventado, direção de métrica invertida, etc). Roteamento híbrido: keyword primeiro (rápido), LLM (Ollama, saída JSON-schema) só como desempate quando a keyword cai em `contexto` por default. Importado por `dashboard/app.py` e `eval/rodar_golden.py` — era duplicado manualmente em 3 lugares até 2026-08-07 (ver comentário no topo do arquivo).
- **`rag/rag_hibrido_langchain.py`** (classe `RAGHibrido`) — motor de busca de PRODUÇÃO desde 2026-09-08 (substituiu `rag/rag_hibrido.py`, que fica intacto e é usado só pelos benchmarks de comparação por enquanto): `EnsembleRetriever` (E5 denso + `BM25Retriever`, fusão RRF) + rerank Cross-Encoder (`ms-marco-MiniLM-L-6-v2`). Chunking por seção Markdown herdado de `rag_hibrido.py::chunk_texto()`. Persistido em ChromaDB (`rag/chroma_db_langchain/`). **Achado real (2026-09-09)**: com `usar_rerank=False`, `buscar()` sempre devolvia `score=0.0` fixo (o `EnsembleRetriever` do LangChain não expõe score de RRF) — invalidava qualquer calibração de limiar de score rio abaixo (ex. o benchmark multimodal 2x2). Corrigido com o parâmetro `usar_score_rrf` (default `False`, flag de ablação): quando `True`, recalcula o RRF explicitamente e expõe o score fundido real. O parâmetro `score_min` da assinatura de `buscar()` nunca é usado no corpo da função (parâmetro morto, documentado, não removido).
- **`nl_to_sql/nl_to_sql.py`** — traduz pergunta em português para SQL via Ollama, valida que só é `SELECT`, executa no Postgres (`harbor_manufatura`), com self-repair e um "DBA-Agent" de segunda opinião. `ESQUEMA` no topo do arquivo documenta cada tabela por dataset de origem — **nunca cruzar tabelas de datasets diferentes** (ex.: company_A e company_B do dataset 3 usam schemas de status incompatíveis).
- **`eval/rag_gerador.py`** — geração de resposta RAG compartilhada entre `dashboard/app.py` e o harness, pelo mesmo motivo do roteador.

### Migrações LangGraph em andamento (coexistindo com o original)

Seguem a regra fixada em `migrar-para-langchain`: nenhuma migração troca o import de produção antes de validada pelo harness (`eval/rodar_golden.py`) sem regressão — por isso existem lado a lado com o módulo original, não o substituem ainda.

- **`dashboard/roteador_langgraph.py`** — o roteamento de `dashboard/roteador.py` como grafo de estados explícito. Reusa as 11 funções `pede_*` e a lógica de `rotear_por_keyword()`/`rotear_por_llm()` originais sem reescrever nenhum gate; só a orquestração vira grafo.
- **`nl_to_sql/nl_to_sql_langgraph.py`** — o self-repair de `nl_to_sql.py` (gerar→executar→corrigir→reexecutar→verificar DBA→...) como `StateGraph`, reusando os mesmos prompts e funções (`gerar_sql`, `corrigir_sql`, `verificar_resultado_responde`, `validar_sql_seguro`).
- **`rag/rag_agentic.py`** — retrieval adaptativo (padrão Self-RAG/FLARE) sobre `RAGHibrido.buscar()` (`rag/rag_hibrido_langchain.py`), via LangGraph critic node com LLM juiz (não score de Cross-Encoder — achado real 2026-09-09: score de reranker é péssimo preditor de acerto). Resultado negativo (2026-09-09, n=15 pequeno demais — reexecução com n=49 pendente em `eval/avaliar_rag_agentic.py`, script pronto, não executado): piorou Recall@1. Investigação 2026-09-14 achou a causa raiz real — não é falta de reformulação de query, é o retrieval/rerank ordenando mal quando duas seções do mesmo manual são tematicamente próximas (ver `eval/avaliar_retrieval.py`, métrica de chunk-level nova). A correção que resolveu os casos genuínos foi mais simples que qualquer técnica agêntica: ver `eval/rag_gerador.py::adaptive_k()`.
- **`eval/rag_gerador.py::adaptive_k()`** (2026-09-14) — corta o nº de chunks passados ao gerador pelo gap de score RRF (Adaptive-k, EMNLP 2025, arXiv:2506.08479), em vez de um k fixo. `rag_responder(..., usar_adaptive_k=True)` liga; default `False` preserva produção (k=3 fixo) até a medição formal final confirmar a promoção — **medição interrompida por desligamento de máquina em 2026-09-14, retomar antes de promover**.

### Serviços

- **`dashboard/app.py`** (Streamlit, porta 8501) — chat principal, consome `dashboard/roteador.py` + `rag/rag_hibrido_langchain.py` + `nl_to_sql/nl_to_sql.py`. Contém `st.set_page_config()` em nível de módulo, por isso não é importável fora do Streamlit — é o motivo de `roteador.py` e `rag_gerador.py` terem sido extraídos como módulos separados. 8 abas de chat especializado (`chat_especializado()`, mesmo padrão em todas): 4 datasets originais + OpenPack (Fase 7f) + Gráficos Técnicos/RAG multimodal (2026-09-14) + 2 utilitárias (reprocessar pipeline, avaliação). As duas últimas abas de dataset usam sub-roteador hierárquico dentro da rota `rag` (`rag_openpack_responder_ou_manual`, `rag_multimodal_responder_ou_manual`) para escolher entre o corpus de manuais e um corpus próprio — padrão corpus-aware routing (UniversalRAG/RAGRouter).
- **`infra/docker-compose.yml`** — Postgres 16 (`harbor_manufatura`).

**Removidos** (entregas planejadas para reimplementação futura, ver `docs/mapeamento_cronograma.md`): API FastAPI de diagnóstico (`/diagnostico`, 3 camadas — regra determinística → Isolation Forest → veredito LLM, com precedência da regra sobre o LLM quando `CRITICO`) e servidor MCP (`consultar_banco`, `buscar_manual`, `diagnosticar_leitura` via FastMCP stdio), ambos em 2026-09-12; N8N (container + workflows de automação), na sessão seguinte. Os módulos compartilhados que API/MCP consumiam (`dashboard/roteador.py`, `rag/rag_hibrido_langchain.py`, `nl_to_sql/nl_to_sql.py`) continuam em produção via `dashboard/app.py`, sem impacto.

### Pipelines (`pipelines/`)

8 pipelines, um por dataset, cada um gravando outputs em `outputs/pipelineN_*/`. **Só os pipelines 1-4** (OEE/Downtime, Legacy Sensor Logs, Discrete Manufacturing, Five-Axis CNC Milling) têm outputs consumidos pelo dashboard como "contexto" pré-calculado (a rota mais confiável do roteador, porque não depende do LLM calcular nada) e aparecem no dict de reprocessamento de `dashboard/app.py` (aba "Reprocessar pipeline"). Os pipelines 5-7 (Facility Maintenance, Labeled Car, Aircraft Annotation) processam seus datasets e gravam outputs, mas **não alimentam o chat/dashboard ainda** — ficaram fora do escopo do roteador/chatbot (achado da auditoria de scripts órfãos, 2026-09-08). Se algum dia forem integrados ao chat, atualizar também o dict `pipelines_disponiveis` em `dashboard/app.py`.

O **pipeline8** (`pipeline8_openpack.py`, integrado em 2026-09-13/14) tem a mesma ressalva editorial: processa o dataset OpenPack (HAR de operações de embalagem via sensores IMU vestíveis, licença CC BY-NC-SA 4.0 — ver `docs/openpack_licenca_e_atribuicao.md`) e alimenta as rotas `contexto` e `sql` do roteador (com 2 gates novos, `pede_cruzamento_openpack_x_maquina` e `pede_identificacao_de_pessoa`, em `dashboard/roteador.py`) e a rota `rag` via corpus texto determinístico (`rag/rag_openpack_texto.py`, arquitetura RAG-HAR/arXiv:2512.08984 — features estatísticas de sensor convertidas em texto por template, sem VLM). Tem aba própria no Streamlit (Fase 7f, 2026-09-14): "5. OpenPack (Operações de Embalagem)" em `dashboard/app.py`, com sub-roteador hierárquico entre o corpus de manuais e o corpus de janelas IMU dentro da rota `rag`. F1-macro de classificação (protocolo k-NN training-free) validado em dois protocolos contra o benchmark oficial `openpack-torch`/split "Pilot Challenge": 0,9217 (amostra estratificada) e 0,9114 (teste completo, 2.591 janelas) — supera os 3 baselines supervisionados (UNet=0,3451, ST-GCN=0,7024, DeepConvLSTM=0,7081). Ver `eval/resultados_classificacao_openpack_completo.json`.

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
- **`atualizar-slide-migracao-langchain`** — gera/atualiza slides Beamer da migração para LangChain/LangGraph, comparando Python puro x LangChain com números reais por módulo migrado.
- **`comparar-tfidf-bm25`** — roda o benchmark que compara os dois algoritmos lexicais de `rag_hibrido.py`.
- **`migrar-para-langchain`** — guia vivo para migrar módulos do Harbor (roteador, RAG híbrido, NL-to-SQL, diagnóstico em camadas) para LangChain/LangGraph, em ordem de risco crescente.
- **`rag-multimodal`** — guia e checklist de implementação para RAG multimodal (texto + imagem). Em produção desde 2026-09-14 via **geração determinística** (`rag/legendas_deterministicas.py`), não VLM — alucinação numérica em VLMs pequenos lendo gráfico é estrutural (arXiv:2312.10160), não um gap de "modelo certo ainda não testado"; para gráfico gerado a partir de dado tabular conhecido, template sobre os dados de origem bate qualquer VLM em fidelidade (100% vs. 93,6%) e custo (segundos vs. 412s/imagem). Ver skill para o histórico completo e a correção de um bug de avaliação que mascarava o resultado real dos VLMs testados.
- **`roadmap-rag-survey`** — mapeia o RAG do Harbor contra a taxonomia Naive/Advanced/Modular RAG do survey de Gao et al. (arXiv:2312.10997), lista priorizada de técnicas a implementar.
- **`roadmap-slm-multiagente`** — guia vivo do fit do Harbor com o Projeto 1 do PDC (multiagentes confiáveis + SLMs em português); lista priorizada de itens implementáveis.
- **`refatorar-organizar-repositorio`** — audita dívida técnica estrutural (duplicação, arquivos órfãos, nomenclatura, organização de pastas) e aplica refatoração com comportamento preservado em mudanças pequenas e reversíveis; skill genérica, não específica do Harbor.
- **`auditar-scripts-orfaos`** — classifica scripts Python standalone (`if __name__ == "__main__":`) em útil/órfão/arquivado intencionalmente/setup one-shot, cruzando evidência de chamada real com sinais de auto-abandono no docstring; skill genérica, não específica do Harbor.
- **`metricas-avaliacao-ia-industrial`** — checklist de métricas padrão (Recuperação, LLM, Ferramentas, Industrial, Sistema) do grupo CERISE, com o mapa do que já é medido no Harbor vs. lacuna; usar sempre que possível ao desenhar ou reportar qualquer avaliação nova.
- **`slide-progresso-sessao`** — gera slide Beamer panorâmico de progresso num período (conquistas mensuráveis, decisões com o porquê, métricas com contexto, riscos/pendências com indicador de status, próximos passos), em contraste com `atualizar-slide-migracao-langchain`/`slides-beamer-decisao-tecnica` que cobrem uma decisão técnica isolada.
- **`revisar-funcoes-orfas`** — revisa um arquivo/pasta indicado pelo usuário linha por linha em busca de funções/métodos sem nenhuma chamada, cuidando de despacho por dicionário, métodos homônimos entre classes e underscore importado cross-file; nunca remove sozinha, exige confirmação por função; skill genérica, não específica do Harbor.
