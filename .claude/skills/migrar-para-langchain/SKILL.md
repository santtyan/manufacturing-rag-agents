---
name: migrar-para-langchain
description: Guia vivo para migrar módulos do Harbor (roteador, RAG híbrido, NL-to-SQL, diagnóstico em camadas) para LangChain/LangGraph, em ordem de risco crescente, preservando as garantias determinísticas já validadas (gates, SELECT-only check, precedência da regra sobre o LLM). Use quando o usuário pedir para "migrar para LangChain", "traduzir esse módulo pra LangChain/LangGraph", "vale usar LangGraph aqui", ou perguntar qual é o próximo passo da migração LangChain.
---

# Migração do Harbor para LangChain

## Pré-requisito de setup (rodar uma vez)

As skills oficiais da LangChain (`langchain-ai/langchain-skills`) já ensinam o "como" genérico
de `create_agent()`, LangGraph e Deep Agents — não duplicar esse conteúdo aqui. Antes de
começar a primeira migração, instalar localmente ao projeto:

```
npx skills add langchain-ai/langchain-skills --skill '*' --yes
```

Esta skill do Harbor assume essas skills disponíveis e só referencia os padrões delas
(`RunnableBranch`, `EnsembleRetriever`, LangGraph conditional edges, etc.) — o foco aqui é o
que é **específico do domínio Harbor**: ordem de migração e o que não pode regredir.

## Tabela de risco por módulo (não repesquisar do zero — ponto de partida sempre que a skill for invocada)

Pesquisa feita em 2026-09-08 (~30 fontes: docs oficiais LangChain/LangGraph, GitHub issues,
blogs técnicos). Ordem crescente de risco/esforço:

1. **RAG híbrido** (`rag/rag_hibrido.py`, classe `RAGHibrido`) — **menor risco**.
   `EnsembleRetriever` (BM25+denso via RRF) e `ContextualCompressionRetriever` +
   `CrossEncoderReranker` mapeiam quase 1:1 no que já existe (E5 + TF-IDF/BM25 + Cross-Encoder
   sobre ChromaDB). Diferença a documentar, não corrigir: LangChain funde por RRF (rank), o
   Harbor por normalização min-max de score por query — resolve o mesmo problema que motivou o
   achado real de 2026-08-22 (BM25 sem normalizar degradava Recall@5 sem rerank) por outro
   caminho. Comparar Recall@5/MRR das duas abordagens antes de trocar, não assumir que RRF é
   estritamente melhor.

2. **Self-repair / DBA-Agent** (dentro de `nl_to_sql/nl_to_sql.py`) — **baixo-médio risco**.
   Os padrões CRAG/Self-RAG têm cookbook oficial em LangGraph, e "critic node" mapeia
   conceitualmente com o DBA-Agent (segunda opinião) já existente. Bom segundo passo depois do
   RAG porque valida o padrão de branching condicional do LangGraph antes de aplicá-lo ao
   roteador (item 3, mais crítico).

3. **Roteador + answerability gates** (`dashboard/roteador.py`) — **médio risco, esforço
   alto**. Existe padrão nomeado ("Adaptive RAG" / query router, `RunnableBranch` / conditional
   edges do LangGraph) para o encaixe do roteamento em si, mas cada gate `pede_*` continua
   sendo lógica de domínio 100% custom — o framework só padroniza onde a decisão acontece, não
   fornece a regra. São 9+ gates para portar sem perder cobertura do golden set
   (`eval/golden_questions.json`). Migrar só depois de RAG e self-repair estarem estáveis.

4. **Diagnóstico em camadas** — removido do projeto em 2026-09-12 junto com `api/main.py`
   (entrega planejada para reimplementação futura, ver `docs/mapeamento_cronograma.md`). Item
   de roadmap suspenso até a reimplementação existir.

5. **NL-to-SQL** (`nl_to_sql/nl_to_sql.py`) — **maior risco**. O `sql_db_query_checker` nativo
   do LangChain é **LLM-assistido** (pede ao próprio LLM pra revisar a query), não uma
   validação determinística em código como o check atual `SELECT`-only do Harbor. Migrar sem
   manter uma camada de validação determinística própria em paralelo seria uma regressão de
   segurança — não aceitar essa troca silenciosamente.

### Outros achados relevantes (contexto ao decidir)

- LCEL perdeu o posto de "default" para LangGraph em fluxos com estado/branching desde a
  v1.0 (out/2025); `AgentExecutor` e `LangServe` estão formalmente deprecated — não migrar para
  LCEL puro, ir direto para LangGraph nos módulos com branching (roteador, self-repair).
- `with_structured_output` com Ollama tem falha documentada em modelos pequenos — llama3.2:3b
  citado nominalmente, 1-5% de taxa de erro em JSON schema. Isso é risco direto para o seletor
  "Rápido" do dashboard Harbor: testar explicitamente esse modelo, não só qwen2.5:7b/14b, antes
  de considerar qualquer módulo migrado como pronto.
- Críticas equilibradas e reais existem contra adotar LangChain em produção ("Stop Using
  LangChain in 2026", post da Octomind, churn de versões até em minor bumps 0.3.27→1.0.2) —
  medir antes de assumir que migrar é estritamente melhor (mesmo princípio de
  `[[feedback_pesquisa_vs_execucao]]`: construir/medir bate mais pesquisa).
- LangSmith funciona com Ollama local sem exigir LLM pago (alternativa OSS: Langfuse), mas é
  aditivo — fora do escopo desta skill, avaliar separadamente se/quando fizer sentido.

## Decisão registrada: migrar mesmo sem ganho isolado de qualidade (2026-09-08)

O item 1 (RAG híbrido) foi medido dia 2026-09-08: Recall@5/MRR idênticos entre produção
(TF-IDF), LangChain-fiel e LangChain BM25+RRF (98%/~0,90 nos três); o único ganho real
(+11,4pp de Precision@5) vem do **algoritmo BM25+RRF, não do framework** — ver
`experiments/langchain_rag/README.md`, seção "Atualização 2026-09-08", e commit `8d0ec37`.

Decisão do usuário: migrar mesmo assim, porque **o roteador e o self-repair (itens 2-3) vão
usar LangGraph em breve**, e ter o RAG já em LangChain evita uma segunda migração/mistura de
paradigmas quando esses módulos consumirem o retriever do RAG dentro de um grafo. Ou seja, o
critério de aceite deixa de ser só "não regredir e/ou ganhar qualidade isolada" — passa a valer
também **consistência arquitetural com o restante do pipeline LangGraph**, quando essa razão for
declarada explicitamente pelo usuário (não assumir isso por padrão sem essa razão ter sido dita).

Implementação usada em produção: **`RAGLangChainBM25RRF`** (`experiments/langchain_rag/
rag_langchain_bm25rrf.py`) — não `RAGLangChainFiel` — porque combina os dois ganhos (framework
LangChain + o algoritmo BM25+RRF que efetivamente melhora Precision@5). Ao promover essa classe
para `rag/`, mover o arquivo para fora de `experiments/` (ela deixa de ser experimental) e
atualizar os consumidores (`dashboard/app.py`, `eval/rag_gerador.py`) — aplicar a mesma regra
inegociável abaixo antes de trocar o import de produção.

## Regra de baseline justo (disciplina experimental do grupo PDC, 2026-09-08)

O grupo de pesquisa PDC (sublinha de Agentes, "Harness Engineering") define como regra
inegociável: **experimento multiagente sem baseline single-agent a custo normalizado não é
submetido**. Isso se aplica diretamente ao Harbor sempre que uma camada extra de LLM entrar em
jogo — self-repair, DBA-Agent (segunda opinião), roteador com desempate por LLM.

Concretamente: nenhuma comparação envolvendo `usar_self_repair`, `usar_dba_agent` ou
`usar_llm_desempate` é aceita como conclusão (nem em slide, nem em relatório) sem reportar também
o resultado com essa camada **desligada**, **ao mesmo custo de tokens** — não só "com e sem
melhora a qualidade", mas "quanto custou cada braço em tokens de entrada+saída". Antes de
2026-09-08 isso não era mensurável: as 13 chamadas ao Ollama do projeto descartavam
`eval_count`/`prompt_eval_count` com `.get("response")`. Ver `shared/ollama_client.py` (wrapper
único que agora captura esses campos) e `shared/trace.py` (onde o custo por passo é registrado).

As flags de ablação já existentes para produzir esse baseline: `RAGHibrido.buscar(usar_rerank=,
usar_hybrid=)` (já existia), `perguntar_com_dba(usar_dba_agent=, tentar_corrigir=,
tentar_novo_sql_se_nao_responde=)` em `nl_to_sql/nl_to_sql.py`, `rotear_pergunta(usar_llm=)` em
`dashboard/roteador.py` (já existia). Regra de design: sempre parâmetro booleano, nunca branch de
código — um experimento de ablação é uma linha de configuração, não uma edição de arquivo.

## Regra inegociável por módulo migrado

Nenhuma migração é aceita se regredir:
- **`eval/rodar_golden.py`** — números de roteamento e faithfulness não podem cair em relação
  ao baseline pré-migração daquele módulo.
- **`eval/golden_questions.json`** — toda pergunta que hoje passa continua passando.

Isso é o motivo do módulo migrado nunca poder virar uma cópia paralela da lógica antiga: já
aconteceu no Harbor (`eval/rodar_golden.py` chegou a ser uma 3ª cópia manual do roteamento,
faltando gates, e gerou uma regressão fantasma de 67,9%→52% só por desatualização, não por bug
real — ver `[[terceira_copia_roteamento_harness]]`). Ao migrar um módulo, o código LangChain
substitui o antigo, não convive como versão B desatualizada.

## Checklist de progresso

Marcar `[x]` conforme cada módulo for migrado, com uma linha de resultado (fato + data),
seguindo o padrão de `roadmap-slm-multiagente`.

- [x] **1. RAG híbrido → `EnsembleRetriever`** — promovido para produção em 2026-09-08 como
      `rag/rag_hibrido_langchain.py` (classe `RAGHibrido`, drop-in). Harness rodado 3x
      (baseline + 2 confirmações pós-troca de import) sem regressão — mesmas 2-3 alucinações
      conhecidas em `eval/alucinacoes.md`, nenhuma nova. Consumidores atualizados:
      `dashboard/app.py`, `eval/rag_gerador.py`, `mcp/servidor_harbor.py` (este último removido
      do projeto em 2026-09-12, ver `docs/mapeamento_cronograma.md`). Ver seção "Decisão
      registrada" acima para o motivo (consistência com LangGraph futuro, não ganho isolado de
      Recall@5/MRR). `ContextualCompressionRetriever` não foi usado — fica como item do roadmap
      `[[roadmap-rag-survey]]` (item 2, context compression), não faz parte deste item.
- [x] **2. Self-repair/DBA-Agent → LangGraph critic node (padrão CRAG/Self-RAG)** — promovido
      para produção em 2026-09-09 como `nl_to_sql/nl_to_sql_langgraph.py`
      (`perguntar_com_dba_langgraph`), um `StateGraph` explícito reusando os mesmos prompts/
      funções do original (`gerar_sql`, `corrigir_sql`, `verificar_resultado_responde`) — mudança
      estrutural, não de comportamento. Mantido o teto de 1 tentativa de correção do original
      (não o teto de 6 do protótipo `agents/tarefa_sql.py` — achado real: o modelo local repete
      o mesmo erro sem se corrigir de verdade, aumentar o teto sozinho não ajudaria). Harness
      rodado antes/depois: 56/67 → 55/67 roteamento (investigado a fundo — 1 pergunta de
      fronteira, `cnc-cam3`, oscila entre rotas mesmo no roteador ORIGINAL não migrado, 9/10 e
      1/10 em teste intercalado — instabilidade pré-existente do desempate por LLM, não
      regressão), faithfulness 64%→64% (idêntico), alucinações 4→3. Consumidores atualizados:
      `dashboard/app.py` (com `usar_dba_agent=False`, preservando o comportamento que o chat
      sempre teve — nunca tinha DBA-Agent ligado) e `eval/rodar_golden.py` (com DBA-Agent
      ligado, como sempre teve). 5 testes unitários (mock de LLM). Confirmado funcionando no
      dashboard real (`streamlit run app.py`), incluindo a correção de 3 bugs de import
      descobertos só nesse teste manual (ver skill `slide-progresso-sessao`/plano de
      implementação para o detalhe — `sys.path`/`sys.modules` ambíguo entre módulo solto e
      pacote, corrigido com `importlib.util.spec_from_file_location`).
- [x] **3. Roteador + gates → LangGraph conditional edges, gates portados 1:1** — promovido
      para produção em 2026-09-09 como `dashboard/roteador_langgraph.py`
      (`rotear_pergunta_langgraph`), `StateGraph` de 2 nós (gates determinísticos em cascata +
      desempate condicional por LLM) reusando as 11 funções `pede_*` e `rotear_por_llm`
      originais sem reescrever nenhuma regra. **0 divergências em 67/67 perguntas do golden
      set**, tanto com `usar_llm=False` (caminho determinístico) quanto com `usar_llm=True`
      (desempate por Ollama real, 10 tentativas intercaladas por pergunta de fronteira). Era o
      item de maior risco/esforço da tabela (9+ gates a portar sem perder cobertura) — fechado
      sem nenhuma regressão medida. Consumidores atualizados: `dashboard/app.py`,
      `eval/rodar_golden.py`. 7 testes unitários, incluindo equivalência completa contra o
      golden set no caminho determinístico.
- [ ] **4. Diagnóstico em camadas → avaliar se vale migrar (pode ficar como está)**
- [ ] **5. NL-to-SQL → LangGraph + validação determinística própria mantida em paralelo ao
      `sql_db_query_checker`**

## Como usar esta skill

- Ao ser invocada, reler a tabela de risco antes de sugerir por onde começar — a resposta para
  "por onde eu começo?" é sempre "próximo item não marcado na ordem 1→5 acima", salvo pedido
  explícito de pular a ordem.
- Antes de migrar qualquer módulo, rodar o harness (skill `rodar-harness`) para capturar o
  baseline atual — é contra esse número que a migração será comparada.
- Depois de migrar um módulo, rodar o harness de novo, comparar contra o baseline, marcar
  `[x]` aqui só se não houver regressão, e anotar o resultado.
- Se o usuário perguntar sobre um módulo fora da lista (MCP, harness em si), esse é território
  novo — parar e discutir antes de assumir que o mesmo raciocínio de risco se aplica.
