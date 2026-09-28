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

- **`dashboard/roteador.py`** — decide se uma pergunta do chat vai para `contexto` (dados já calculados pelos pipelines), `rag` (manuais técnicos) ou `sql` (NL-to-SQL sobre o Postgres), mais uma série de "answerability gates" determinísticos (`pede_*`) que interceptam ANTES do LLM perguntas que ele historicamente alucinava (cruzamentos impossíveis no schema, ROI inventado, direção de métrica invertida, etc). Roteamento híbrido: keyword primeiro (rápido), LLM (Ollama, saída JSON-schema) só como desempate quando a keyword cai em `contexto` por default. Importado por `dashboard/app.py` e `eval/rodar_golden.py` — era duplicado manualmente em 3 lugares até 2026-08-07 (ver comentário no topo do arquivo). **Achado sobre `llama3.2:3b` (seletor "Rápido", `app.py:274`) — 2026-09-25**: modelo baixado e testado pela primeira vez (disco em 98%/8,3GB livres na máquina — achado real que se soma ao fallback silencioso já documentado em `app.py:264-265`, "disco cheio" não é hipotético). Contra 20 perguntas do golden set via `rotear_por_llm()`: **1/20 (5%) de falha de JSON schema** e **8/20 (40%) de erro de classificação** — praticamente igual ao `llama3.2:1b` (fallback silencioso: 0/20 falhas de formato, 9/20=45% de erro de classificação, testado antes). **O modelo maior não resolve o problema de classificação e ainda introduz falha de formato que o menor não tinha.** Mitigado na prática pela arquitetura híbrida (LLM só decide quando o keyword cai em `contexto` por default); se precisar reduzir esse erro, trocar de família de modelo (ex. `qwen2.5:7b`, já usado no seletor "Qualidade"), não de tamanho dentro de `llama3.2`.
- **`rag/rag_hibrido_langchain.py`** (classe `RAGHibrido`) — motor de busca de PRODUÇÃO desde 2026-09-08 (substituiu `rag/rag_hibrido.py`, que fica intacto e é usado só pelos benchmarks de comparação por enquanto): `EnsembleRetriever` (E5 denso + `BM25Retriever`, fusão RRF) + rerank Cross-Encoder (`ms-marco-MiniLM-L-6-v2`). Chunking por seção Markdown herdado de `rag_hibrido.py::chunk_texto()`. Persistido em ChromaDB (`rag/chroma_db_langchain/`). **Achado real (2026-09-09)**: com `usar_rerank=False`, `buscar()` sempre devolvia `score=0.0` fixo (o `EnsembleRetriever` do LangChain não expõe score de RRF) — invalidava qualquer calibração de limiar de score rio abaixo (ex. o benchmark multimodal 2x2). Corrigido com o parâmetro `usar_score_rrf` (default `False`, flag de ablação): quando `True`, recalcula o RRF explicitamente e expõe o score fundido real. O parâmetro `score_min` da assinatura de `buscar()` nunca é usado no corpo da função (parâmetro morto, documentado, não removido).
- **`nl_to_sql/nl_to_sql.py`** — traduz pergunta em português para SQL via Ollama, valida que só é `SELECT`, executa no Postgres (`harbor_manufatura`), com self-repair e um "DBA-Agent" de segunda opinião. `ESQUEMA` no topo do arquivo documenta cada tabela por dataset de origem — **nunca cruzar tabelas de datasets diferentes** (ex.: company_A e company_B do dataset 3 usam schemas de status incompatíveis).
- **`eval/rag_gerador.py`** — geração de resposta RAG compartilhada entre `dashboard/app.py` e o harness, pelo mesmo motivo do roteador.

### Migrações LangGraph

Seguem a regra fixada em `migrar-para-langchain`: nenhuma migração troca o import de produção antes de validada pelo harness (`eval/rodar_golden.py`) sem regressão.

- **`dashboard/roteador_langgraph.py`** — **em produção desde 2026-09-09** (`dashboard/app.py:521` e `eval/rodar_golden.py:35` importam `rotear_pergunta_langgraph`, validado sem divergência nas 67 perguntas do golden set da época). O roteamento de `dashboard/roteador.py` como grafo de estados explícito. Reusa as **14** funções `pede_*` (contagem em 2026-09-25 — 11 no texto original desta seção, desatualizado desde os 2 gates do OpenPack em 2026-09-12) e a lógica de `rotear_por_keyword()`/`rotear_por_llm()` originais sem reescrever nenhum gate; só a orquestração é grafo. `roteador.py` continua existindo e é importado diretamente por `dashboard/app.py` só para constantes (ex. `PALAVRAS_CHAVE_OPENPACK`), não mais para o roteamento principal.
- **`nl_to_sql/nl_to_sql_langgraph.py`** — **em produção desde 2026-09-09**. O self-repair de `nl_to_sql.py` (gerar→executar→corrigir→reexecutar→verificar DBA→...) como `StateGraph`, reusando os mesmos prompts e funções (`gerar_sql`, `corrigir_sql`, `verificar_resultado_responde`, `validar_sql_seguro`). `eval/rodar_golden.py` importa `perguntar_com_dba_langgraph as perguntar_com_dba`.
- **`rag/rag_agentic.py`** — **DECISÃO FINAL (2026-09-25): encerrado, não promovido a produção.** Retrieval adaptativo (padrão Self-RAG/FLARE) sobre `RAGHibrido.buscar()` (`rag/rag_hibrido_langchain.py`), via LangGraph critic node com LLM juiz (não score de Cross-Encoder — achado real 2026-09-09: score de reranker é péssimo preditor de acerto). Resultado negativo original (2026-09-09, n=15): piorou Recall@1 (12/15 → 11/15). **Reexecução formal com n=49** (`eval/avaliar_rag_agentic.py`, 2026-09-25) confirmou a regressão com amostra adequada — piorou em **todas** as métricas: Recall@k documento 93,9%→91,8% (-2,0pp), MRR 0,857→0,827 (-0,031), Recall@k chunk 90,0%→85,0% (-5,0pp), e custou **23,09x mais** (81,6s → 1885,2s; 18/49 perguntas reformularam query). Resultado persistido em `eval/resultados_rag_agentic.json`. A causa raiz real (investigação 2026-09-14) segue válida: não é falta de reformulação de query, é o retrieval/rerank ordenando mal quando duas seções do mesmo manual são tematicamente próximas (ver `eval/avaliar_retrieval.py`, métrica de chunk-level). A correção que resolveu os casos genuínos foi mais simples que qualquer técnica agêntica: ver `eval/rag_gerador.py::adaptive_k()`. **Não reabrir esta frente sem uma hipótese nova** — duas medições independentes (n=15 e n=49) convergem na mesma direção.
- **`eval/rag_gerador.py::adaptive_k()`** (2026-09-14) — corta o nº de chunks passados ao gerador pelo gap de score RRF (Adaptive-k, EMNLP 2025, arXiv:2506.08479), em vez de um k fixo. **PROMOVIDO a default em 2026-09-25**: `rag_responder(..., usar_adaptive_k=True)` agora é o default (antes `False`); nenhum consumidor (`dashboard/app.py`, `eval/rodar_golden.py`) passa o parâmetro explicitamente, então ambos herdam o novo comportamento. **Veredito de variância (N=3 execuções, `python eval/avaliar_variancia_adaptive_k.py --n 3`)**: faithfulness médio = **0,8077 ± 0,067** (min 0,769, max 0,885; CV=0,0829, acima do alvo estrito <0,05 da skill `metricas-avaliacao-ia-industrial`, mas o **pior caso das 3 execuções empata com o melhor k fixo (0,769), nunca fica abaixo** — "PROMOVER COM RESSALVA", ver `eval/resultados_variancia_adaptive_k.json`). O bug do timeout silencioso do `subprocess.run` que impedia essa medição (ver `[[bug_timeout_silencioso_subprocess_2026-09-16]]`) permaneceu corrigido e não recorreu nas 3 execuções (2881s/1554s/1225s, todas dentro do timeout de 5400s). **Reportar sempre com o intervalo/desvio, nunca só a média**, dado o CV acima do alvo.

### Serviços

- **`dashboard/app.py`** (Streamlit, porta 8501) — chat principal, consome `dashboard/roteador.py` + `rag/rag_hibrido_langchain.py` + `nl_to_sql/nl_to_sql.py`. Contém `st.set_page_config()` em nível de módulo, por isso não é importável fora do Streamlit — é o motivo de `roteador.py` e `rag_gerador.py` terem sido extraídos como módulos separados. 8 abas de chat especializado (`chat_especializado()`, mesmo padrão em todas): 4 datasets originais + OpenPack (Fase 7f) + Gráficos Técnicos/RAG multimodal (2026-09-14) + 2 utilitárias (reprocessar pipeline, avaliação). As duas últimas abas de dataset usam sub-roteador hierárquico dentro da rota `rag` (`rag_openpack_responder_ou_manual`, `rag_multimodal_responder_ou_manual`) para escolher entre o corpus de manuais e um corpus próprio — padrão corpus-aware routing (UniversalRAG/RAGRouter).
- **`infra/docker-compose.yml`** — Postgres 16 (`harbor_manufatura`).

**Removidos** (entregas planejadas para reimplementação futura, ver `docs/mapeamento_cronograma.md`): API FastAPI de diagnóstico (`/diagnostico`, 3 camadas — regra determinística → Isolation Forest → veredito LLM, com precedência da regra sobre o LLM quando `CRITICO`) e servidor MCP (`consultar_banco`, `buscar_manual`, `diagnosticar_leitura` via FastMCP stdio), ambos em 2026-09-12; N8N (container + workflows de automação), na sessão seguinte. Os módulos compartilhados que API/MCP consumiam (`dashboard/roteador.py`, `rag/rag_hibrido_langchain.py`, `nl_to_sql/nl_to_sql.py`) continuam em produção via `dashboard/app.py`, sem impacto.

### Pipelines (`pipelines/`)

8 pipelines, um por dataset, cada um gravando outputs em `outputs/pipelineN_*/`. **Só os pipelines 1-4** (OEE/Downtime, Legacy Sensor Logs, Discrete Manufacturing, Five-Axis CNC Milling) têm outputs consumidos pelo dashboard como "contexto" pré-calculado (a rota mais confiável do roteador, porque não depende do LLM calcular nada) e aparecem no dict de reprocessamento de `dashboard/app.py` (aba "Reprocessar pipeline"). Os pipelines 5-7 (Facility Maintenance, Labeled Car, Aircraft Annotation) processam seus datasets e gravam outputs, mas **não alimentam o chat/dashboard ainda** — ficaram fora do escopo do roteador/chatbot (achado da auditoria de scripts órfãos, 2026-09-08). Se algum dia forem integrados ao chat, atualizar também o dict `pipelines_disponiveis` em `dashboard/app.py`.

O **pipeline8** (`pipeline8_openpack.py`, integrado em 2026-09-13/14) tem a mesma ressalva editorial: processa o dataset OpenPack (HAR de operações de embalagem via sensores IMU vestíveis, licença CC BY-NC-SA 4.0 — ver `docs/openpack_licenca_e_atribuicao.md`) e alimenta as rotas `contexto` e `sql` do roteador (com 2 gates novos, `pede_cruzamento_openpack_x_maquina` e `pede_identificacao_de_pessoa`, em `dashboard/roteador.py`) e a rota `rag` via corpus texto determinístico (`rag/rag_openpack_texto.py`, arquitetura RAG-HAR/arXiv:2512.08984 — features estatísticas de sensor convertidas em texto por template, sem VLM). Tem aba própria no Streamlit (Fase 7f, 2026-09-14): "5. OpenPack (Operações de Embalagem)" em `dashboard/app.py`, com sub-roteador hierárquico entre o corpus de manuais e o corpus de janelas IMU dentro da rota `rag`.

**CORRIGIDO EM 2026-09-15 — bug de vazamento de rótulo**: até 2026-09-14, o texto indexado citava sujeito/sessão/operação/classe na primeira linha — vazamento que fazia o k-NN de avaliação casar rótulo com rótulo (BM25 lexical) em vez de sinal de sensor, inflando o F1-macro artificialmente. Corrigido em `rag/rag_openpack_texto.py::montar_texto_janela()` (identificação agora só em metadado, nunca no texto embeddado); `tests/test_rag_openpack_texto.py` atualizado para testar a ausência do vazamento, não mais protegê-lo. Validado por 3 checagens (ausência léxica 0/4000, rótulo embaralhado F1=0,0182 < acaso 0,10, LOSO sem F1=1,0000 exato) — ver `[[bug_vazamento_rotulo_openpack_2026-09-15]]`.

**Números REAIS pós-correção** (split oficial "Pilot Challenge", `openpack-torch`): F1-macro = **0,0750** (amostra pequena, 26 janelas de teste, alta variância) e **0,1664** (teste completo, 2.591 janelas, sem amostragem — número de referência) — ambos muito abaixo dos 3 baselines supervisionados: UNet=0,3451 e DeepConvLSTM=0,7081, **ambos sobre a modalidade IMU** (mesma modalidade do Harbor), e ST-GCN=0,7024 **sobre keypoints/pose** (modalidade diferente — achado da sessão 2026-09-28, corrigindo uma imprecisão anterior deste documento que listava os 3 como se fossem comparáveis sem distinção de modalidade). O ST-GCN é o único que cai no split de submissão oficial (0,7024→0,6106) — o próprio paper PerCom atribui isso a oclusão de punho por caixa, relevante para qualquer futura tentativa de usar pose neste dataset. **LOSO** (Leave-One-Subject-Out, protocolo padrão-ouro em HAR, 21 sujeitos): F1-macro médio = **0,1626 ± 0,0686** (min 0,0311 em U0107, max 0,3270 em U0101) — consistente com o teste completo, também abaixo dos baselines IMU. O método training-free não supera redes supervisionadas neste dataset; isso é resultado honesto, não fracasso. Ver `eval/resultados_classificacao_openpack.json`, `eval/resultados_classificacao_openpack_completo.json` e `eval/resultados_loso_openpack.json`.

**Hipótese confirmada (2026-09-28) — índice numérico substitui índice textual no protocolo de
avaliação.** `RAG-HAR+` (arXiv:2607.26631, continuação direta do RAG-HAR) reporta que retrieval por
vetor numérico supera retrieval por embedding de texto para casar features de sensor. Testado sob o
MESMO split oficial e MESMO protocolo LOSO (`eval/indice_numerico_openpack.py`, kNN euclidiano +
voto ponderado por 1/distância sobre as mesmas 48 features de `rag/rag_openpack_texto.py`, nada mais
mudou): F1-macro salta de **0,0750→0,3905** (split "Pilot Challenge", 26 janelas) e de
**0,1626±0,0686→0,4314±0,0634** (LOSO, 21 sujeitos) — o índice numérico agora **supera o baseline
UNet (0,3451)**, o único dos 3 na mesma modalidade IMU. Teste de significância (mesmo protocolo de
`avaliar_significancia_openpack.py`): Wilcoxon vs. UNet **não significativo** (p=0,9999 — o Harbor
não fica sistematicamente abaixo), vs. ST-GCN/DeepConvLSTM ainda p<0,001 (redes supervisionadas com
GPU seguem à frente, esperado). IC 95% via bootstrap: [0,4047; 0,4580]. **O gargalo nunca foi a
modalidade IMU — era a representação textual do índice.** Ver
`eval/resultados_loso_indice_numerico_openpack.json` e
`eval/resultados_significancia_indice_numerico_openpack.json`. 4 baterias de sanidade (rótulo
embaralhado → F1 ≈ acaso) confirmam ausência de vazamento nesse novo caminho
(`tests/test_sanidade_harnesses.py`). Trilha RGB/pose (planejada em
`docs/openpack_licenca_e_atribuicao.md`, seção "OpenPack RGB") fica registrada como próximo passo,
não mais como prioridade — a migração de modalidade só se justifica se este número novo (índice
numérico sobre IMU) ficar insuficiente para o objetivo do projeto.

**Índice numérico integrado ao chat de produção (2026-09-29)**: duas vias — resumo agregado
(rota `contexto`, sem gate novo) e classificação ao vivo de uma janela real via gate novo
`pede_classificacao_janela_openpack` (replicado em `roteador.py` e `roteador_langgraph.py`,
resposta 100% em Python sem LLM na predição). Ambas em produção, cobrem perguntas diferentes.
Também cruzado `annotation/openpack-outliers/` (1792 linhas de irregularidade anotada por
humano) contra as mesmas 48 features via Mann-Whitney U + Bonferroni — **nenhuma feature
significativa em nenhuma categoria testável** (Additional/Incident/Struggling, 6-14 janelas
cada) — resultado negativo honesto, amostra pequena e eventos curtos dentro da janela de 4s
diluem o sinal. Ver detalhe completo em `docs/openpack_licenca_e_atribuicao.md`.

**Suporte estatístico formal (2026-09-25)**: `eval/avaliar_significancia_openpack.py` roda Wilcoxon signed-rank de uma amostra (H0: mediana do F1-macro por sujeito = valor do baseline; teste one-sided) e bootstrap (10.000 reamostragens) sobre os 21 F1-macro por sujeito do LOSO. **p<0,001 contra os 3 baselines** (UNet, ST-GCN, DeepConvLSTM) — a diferença visual (0,16 vs. 0,70+) tem suporte estatístico, não é só leitura do número bruto. IC 95% da média via bootstrap: [0,1343, 0,1923] — nem o limite superior chega perto do menor baseline. Ver `eval/resultados_significancia_openpack.json`.

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
- **`varrer-residuo-de-ia`** — varre o repositório em busca de resíduo de sessão de agente (scratch versionado, diretório duplicado detectado por hash, cache versionado por engano, untracked sem decisão, stub/placeholder, docstring de estado morto), classificando cada achado em certeza ALTA/MÉDIA/BAIXA com a evidência da busca. Desvia do padrão-ouro público (`agent-sh/deslop`, que auto-corrige certeza alta): aqui **nunca remove sozinha**, a certeza governa a força da recomendação, não a autonomia. Catálogo de falsos positivos em `references/padroes_residuo_ia.md` — dois são críticos neste repo: a palavra portuguesa "todo" (~18 hits, nenhum é marcador de tarefa; usar sempre `TODO:` com dois-pontos) e `self.placeholder` do Streamlit em `dashboard/app.py:209`. Delega para `auditar-scripts-orfaos`/`revisar-funcoes-orfas`/`refatorar-organizar-repositorio` quando o achado é dívida estrutural, não resíduo; skill genérica, não específica do Harbor.
