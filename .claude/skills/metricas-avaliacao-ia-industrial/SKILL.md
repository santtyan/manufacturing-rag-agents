---
name: metricas-avaliacao-ia-industrial
description: Checklist de métricas padrão (recuperação, LLM, ferramentas, industrial, sistema) para avaliar qualquer parte do Harbor, com o vocabulário fixado pelo grupo CERISE — usar ao desenhar um harness/benchmark novo, ao reportar resultado de uma avaliação, ou quando o usuário perguntar "que métrica usar aqui" ou "estamos medindo isso direito". Mantém o mapa do que já é medido no projeto vs. o que ainda é lacuna.
---

# Métricas de avaliação de IA industrial (vocabulário CERISE)

## A tabela de referência (nunca reinventar categoria nova sem checar aqui primeiro)

| Categoria | Métricas | Quando usar |
|---|---|---|
| **Recuperação** | Recall@k, Precision@k, MRR, nDCG | Qualquer avaliação de retrieval — texto (RAG) ou imagem (RAG multimodal) |
| **LLM** | Answer Correctness, Faithfulness, Hallucination Rate, Answer Relevancy, BERTScore | Avaliação da RESPOSTA gerada, não do retrieval que a alimentou |
| **Ferramentas** | SQL Accuracy, Cypher Accuracy, Tool Selection Accuracy | Quando o sistema usa tool-calling (NL-to-SQL, agente com tools) |
| **Industrial** | Diagnosis Accuracy, Root Cause Accuracy, Manual Compliance, Unsafe Recommendation Rate | Específico do domínio: diagnóstico de falha, recomendação de manutenção |
| **Sistema** | Latência, Custo, Tokens, GPU, Tempo de resposta | Qualquer execução — sempre medir junto com a métrica de qualidade, nunca isolado (ver regra de baseline justo em `migrar-para-langchain`/`replicacao-experimento-agente`) |

## Estado real do Harbor por categoria (atualizar conforme cada item for fechado)

Legenda: ✅ medido | 🔶 parcial | ❌ lacuna

### Recuperação
- ✅ Recall@k/Precision@k/MRR: `eval/avaliar_retrieval.py` (RAG texto), `eval/avaliar_retrieval_nanobeir.py` (benchmark acadêmico), `eval/avaliar_rag_multimodal.py` e `eval/avaliar_benchmark_multimodal_2x2.py` (RAG multimodal).
- ✅ nDCG (multimodal, desde 2026-09-09): `eval/avaliar_benchmark_multimodal_2x2.py`.
- ❌ nDCG no RAG de texto — `eval/avaliar_retrieval.py` ainda só tem Recall/Precision/MRR. Se
  for adicionar, reusar a função `ndcg_at_k` já escrita em `avaliar_benchmark_multimodal_2x2.py`
  em vez de reimplementar.
- ✅ Recall@5/Precision@5/MRR medidos também para duas técnicas avançadas de RAG nesta rodada
  (2026-09-09): Agentic RAG (`rag/rag_agentic.py`, resultado negativo, não promovido) e
  Contextual Retrieval (`rag_hibrido_langchain.py::indexar(usar_contexto=True)`, resultado
  misto, não promovido) — ver `roadmap-rag-survey` itens 3 e 5 para os números completos.

### LLM
- ✅ Faithfulness: métrica central de `eval/rodar_golden.py` (números esperados citados na resposta).
- ✅ Hallucination Rate: `dashboard/log_alucinacoes.jsonl` + `eval/alucinacoes.md`.
- 🔶 Answer Correctness: parecido com faithfulness, não formalizado com esse nome.
- ❌ Answer Relevancy, BERTScore — nunca implementados (BERTScore já sugerido pela equipe CERISE
  antes, ver memória `avaliacao_llm_cerise`).

### Ferramentas
- 🔶 SQL Accuracy: BIRD-SQL/Spider (`eval/avaliar_bird_sql.py`, `eval/avaliar_spider_sql.py`)
  medem execução correta, não sob o nome formal "SQL Accuracy". Também medido indiretamente via
  `tests/test_nl_to_sql_langgraph.py` (5 testes cobrindo sucesso de primeira, self-repair,
  rejeição do DBA-Agent, ablação) e a comparação DeepSeek vs. qwen2.5:7b (2026-09-09): sucesso
  2/2 vs 1/2 numa amostra de 2 perguntas — ainda não é "SQL Accuracy" formal sobre golden set
  inteiro, mas mais evidência que antes.
- ❌ Cypher Accuracy — não aplicável, Harbor não usa grafo/Neo4j.
- 🔶 Tool Selection Accuracy — "rota_ok" do roteamento (`eval/rodar_golden.py`) segue medindo
  roteamento entre 3 fontes, não seleção de tool dentro de um agente. Porém
  `tests/test_roteador_langgraph.py::test_equivalencia_completa_golden_set_sem_llm` (2026-09-09)
  é o primeiro teste determinístico real de decisão de rota no projeto — ainda não é Tool
  Selection Accuracy formal de agente (`agents/react.py` continua sem essa métrica), mas é a
  peça mais próxima disso hoje.

### Industrial
- ❌ Diagnosis Accuracy, Root Cause Accuracy, Manual Compliance, Unsafe Recommendation Rate —
  nenhuma medida hoje. A API de diagnóstico em 3 camadas (`api/main.py`) que motivava esta
  categoria foi removida do projeto em 2026-09-12 — entrega planejada para reimplementação
  futura, ver `docs/mapeamento_cronograma.md`. Quando reimplementada, retomar a nota de que
  Unsafe Recommendation Rate é conceitualmente o que a precedência da regra determinística sobre
  o LLM tentava evitar (achado real: "LLM discordava de uma leitura obviamente crítica"), mas
  nunca foi quantificado como taxa sobre um golden set.

### Sistema
- ✅ Custo/Tokens (desde 2026-09-08): `shared/ollama_client.py`, captura
  `gen_ai_usage_input_tokens`/`gen_ai_usage_output_tokens` em toda chamada Ollama. Aplicado
  nesta rodada (2026-09-09) para comparar `qwen2.5:7b` vs. `deepseek-r1:7b` (tokens de entrada/
  saída, duração total) — primeiro uso real desse dado para decisão de modelo, não só captura.
- 🔶 Latência/Tempo de resposta: `shared/trace.py` tem `duracao_s` por passo e
  `duracao_total_s` por execução, mas só instrumentado nos agentes (`agents/`), não no
  dashboard/chat de produção real (`dashboard/app.py` ainda não grava latência em lugar
  nenhum) — lacuna confirmada ainda aberta em 2026-09-09, mesmo depois de self-repair e
  roteador terem sido migrados e promovidos para produção no dashboard.
- ❌ GPU — não medido; a própria máquina de desenvolvimento não tem GPU CUDA disponível
  (`torch.cuda.is_available() == False`, confirmado 2026-09-09), então localmente nem faria
  sentido medir isso ainda — só relevante se/quando o projeto rodar em hardware com GPU.

## Métricas adicionais já usadas no projeto, fora da tabela CERISE

- **Distribuição de modos de falha** (categorização, não número único) — usada na replicação
  ReAct sobre HotpotQA (`agents/tarefa_hotpotqa.py`), inspirada no paper original (56%
  alucinação em CoT puro vs. 23% busca errada em ReAct).
- **Taxa de sucesso de execução de agente** (`Trace.sucesso`, `shared/trace.py`) — genérica,
  aplicável a qualquer tarefa de agente, não específica do domínio industrial como
  Diagnosis/Root Cause Accuracy seriam.
- **Custo adaptativo de reranking via top-p** (`avaliar_benchmark_multimodal_2x2.py`) — não é
  métrica de qualidade, é controle operacional de custo, adjacente a "Sistema > Custo" mas não
  nomeada na tabela CERISE.

## Como usar esta skill

1. Antes de desenhar uma avaliação nova, ache a linha certa da tabela de referência acima e use
   o nome dela — não invente terminologia nova para algo que já tem nome padrão.
2. Ao reportar resultado (slide, memória, PR, skill), cite a categoria CERISE junto com o
   número, para manter consistência de vocabulário entre sessões e com o grupo CERISE.
3. Ao fechar uma lacuna (❌ virando ✅), atualizar a tabela "Estado real do Harbor" acima com a
   data e o arquivo que passou a medir aquilo — mesmo padrão de checklist já usado em
   `roadmap-rag-survey`/`roadmap-slm-multiagente`.
4. Métrica de qualidade sozinha nunca é conclusão válida — sempre reportar junto com custo
   (tokens/latência) da categoria Sistema, regra já fixada em `migrar-para-langchain` e
   `replicacao-experimento-agente` (repo pessoal): nenhuma comparação é aceita sem o custo
   normalizado do outro lado.
5. As lacunas atuais (Answer Relevancy, BERTScore, Tool Selection Accuracy, Diagnosis/Root Cause
   Accuracy, Manual Compliance, Unsafe Recommendation Rate) são a fila natural de próximos itens
   de harness quando o usuário pedir para expandir avaliação.

## Ver também

`rodar-harness` — como rodar o harness de roteamento/faithfulness já existente.
`roadmap-rag-survey` — técnicas de RAG de texto, algumas medidas pelas métricas de Recuperação daqui.
`rag-multimodal` — RAG multimodal, onde nDCG foi implementado pela primeira vez no projeto.
`migrar-para-langchain` — regra de baseline justo (custo normalizado) referenciada no item 4 acima.
