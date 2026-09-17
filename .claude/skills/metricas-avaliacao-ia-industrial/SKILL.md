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
- ✅ nDCG no RAG de texto (fechado em 2026-09-15, item 7.6/6.3 do plano): `eval/avaliar_retrieval.py`
  agora importa e reusa `ndcg_at_k` de `avaliar_benchmark_multimodal_2x2.py` (nenhuma
  reimplementação) — relevância binária por posição (1 se fonte==alvo). Resultado:
  nDCG@5 médio = 0,891 (coerente com MRR = 0,898 do mesmo conjunto de 49 perguntas).
- ✅ Recall@5/Precision@5/MRR medidos também para duas técnicas avançadas de RAG nesta rodada
  (2026-09-09): Agentic RAG (`rag/rag_agentic.py`, resultado negativo inicial com n=15 — **n
  pequeno demais, ver correção abaixo**) e Contextual Retrieval
  (`rag_hibrido_langchain.py::indexar(usar_contexto=True)`, resultado misto, não promovido) —
  ver `roadmap-rag-survey` itens 3 e 5 para os números completos.
- ✅ **Recall/MRR em nível de CHUNK, não só de documento** (novo, 2026-09-14,
  `eval/avaliar_retrieval.py::secao_esperada()`/`secao_do_chunk()`) — achado que motivou: o
  Agentic RAG piorou porque "documento certo, chunk errado" é um modo de falha invisível na
  métrica de documento (Recall@5=98% mesmo quando a seção certa não está no top-k). Formalizado
  como [Seven Failure Points, FP2](https://arxiv.org/abs/2401.05856). Extraído do campo `nota`
  do golden set (41/49 perguntas já citam a seção esperada, sem precisar reanotar) — **cuidado**:
  5 notas citam 2 seções (armadilha + resposta certa), a extração ingênua pela 1ª ocorrência dá
  falso positivo, ver `SECOES_ESPERADAS_AMBIGUAS` no código. Resultado real: gap doc-level vs.
  chunk-level é ~zero sem rerank (98,0%/97,5%), mas real com rerank (98,0%/95,0%, 2 casos
  genuínos) — o rerank, não o retrieval bruto nem o chunking, introduz o erro.
- 🔶 **k-NN label purity** (nova, 2026-09-13, `eval/avaliar_retrieval_openpack.py`) — variante de
  Recall/Precision para corpus de retrieval *class-level* (várias instâncias legítimas por
  classe, ex. séries de sensor segmentadas em janelas), onde Recall@k clássico mede a métrica
  errada (deu 0% mesmo com retrieval funcionando, porque nenhuma janela específica é "a resposta
  certa"). Mede a fração dos k vizinhos que compartilham o rótulo/classe da janela de origem,
  contra o acaso esperado (1/n_classes). Resultado: pureza=12,9% (acaso=10%), **reconfirmado
  idêntico em 2026-09-15 sobre o corpus corrigido** (ver bug de vazamento abaixo) — este script
  já era imune por construção, porque a query vem de `golden_questions_openpack.json` (perguntas
  em linguagem natural), não do texto indexado. Formalizado como 🔶 porque cobre só o OpenPack
  até agora — ainda não é usado em nenhum outro corpus do Harbor que precise dela (todos os
  outros são instance-level de fato).
- ✅ **F1-macro de classificação OpenPack** (`eval/avaliar_classificacao_openpack.py`,
  `eval/avaliar_loso_openpack.py`) — **adicionada à tabela nesta rodada (2026-09-15); nunca
  tinha entrado formalmente**, apesar de ser o número mais citado do projeto. Split oficial
  "Pilot Challenge": F1-macro = 0,0750 (amostra pequena) e 0,1664 (teste completo, 2.591
  janelas, referência). LOSO (21 sujeitos, protocolo padrão-ouro): F1-macro médio =
  0,1626 ± 0,0686, consistente com o teste completo. Todos abaixo dos 3 baselines
  supervisionados — ver bug de vazamento abaixo para o porquê dos números terem mudado dos
  0,9217/0,9114 publicados até 2026-09-14.
- ✅ **Proveniência numérica das legendas** (`eval/teste_provenencia_numerica_legendas.py`,
  novo em 2026-09-15, item 5.5 do plano): diferente dos checks de fidelidade abaixo (que
  validam por REGRA heurística), este verifica por **invariante** — para cada número no texto
  de cada legenda, confirma que ele bate com um valor de fato recalculado a partir do CSV de
  origem via `estatisticas_da_variavel()`, independente do texto já gerado. Resultado: 78/78
  números com proveniência confirmada (0 falhas), sobre as 26 legendas do corpus. Transforma a
  tese "fidelidade 100% por construção" (skill `rag-multimodal`, item 6b) de afirmação em
  invariante verificado — material forte para um manuscrito, não só um número reportado.
- ✅ Fidelidade de caption multimodal (`eval/checks_fidelidade_caption.py`, 6 checks
  determinísticos): **achado importante (2026-09-14) sobre validar a própria métrica** — um bug
  de divisão (`len(dict)` capturado depois de inserir uma chave extra no mesmo dict, dividindo
  por 7 em vez de 6) subestimava `taxa_aprovacao` de QUALQUER legenda que passasse todos os
  checks reais, mascarando por 4 dias que o `qwen3-vl:4b` já batia perto do critério de promoção
  (documentado como 77,5%/reprovado; real, 93,6%, ainda reprovado mas por margem bem menor). Um
  segundo bug relacionado (`sem_numeros_inventados` aceitando número de qualquer faixa do dict
  global, não só das variáveis da imagem em questão) mascarava uma alucinação numérica genuína
  já catalogada. **Lição generalizável**: um harness/check que nunca é exercitado por um caso
  que deveria passar 100% (aqui, só apareceu ao validar uma via determinística nova) pode
  esconder bugs de aritmética básica por muito tempo — vale ter pelo menos 1 caso de controle
  positivo perfeito em qualquer suíte de avaliação nova.
- ❌➜✅ **BUG CRÍTICO DE AVALIAÇÃO corrigido em 2026-09-15 — vazamento de rótulo no OpenPack**:
  até então, o texto indexado usado como corpus e como QUERY de avaliação continha o nome da
  operação (`rag/rag_openpack_texto.py`) — o k-NN de classificação casava rótulo com rótulo por
  BM25 lexical, não sinal de sensor, inflando F1-macro de ~0,08-0,16 reais para 0,9217/0,9114
  publicados. Descoberto rodando LOSO pela primeira vez: 3 sujeitos seguidos deram F1=1,0000
  exato (perfeição repetida = sintoma clássico de vazamento). **Agravante**: um teste unitário
  (`tests/test_rag_openpack_texto.py`) afirmava a presença do rótulo no texto como comportamento
  ESPERADO — protegia o bug ativamente, não só deixava passar.
  **Lição generalizável, complementando a lição acima sobre fidelidade de caption**: aquela
  lição cobre o CONTROLE POSITIVO (ter 1 caso que deve passar 100%); esta cobre o **CONTROLE
  NEGATIVO**, que faltava — um teste de sanidade tipo *data randomization test* (embaralhar
  rótulos e verificar que a métrica desaba para o acaso; padrão desde Zhang et al. 2016,
  formalizado em Sanity Checks for Saliency Maps, NeurIPS 2018) teria pego este bug no primeiro
  dia. **Regra nova**: todo harness de classificação/retrieval deve ter os dois controles, não
  só o positivo. Ver `[[bug_vazamento_rotulo_openpack_2026-09-15]]` para evidência completa.

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
- 🔶 **F1-macro de classificação** (nova, 2026-09-14, `eval/avaliar_classificacao_openpack.py`) —
  mais próxima em espírito de Diagnosis Accuracy (classificar uma condição/operação a partir de
  sensor) do que de qualquer métrica de Recuperação, mas usa retrieval (k-NN training-free, sem
  treino) como classificador. Validado contra o benchmark oficial `openpack-torch`/split "Pilot
  Challenge" em dois protocolos: 0,9217 (amostra estratificada) e 0,9114 (teste completo, 2.591
  janelas) — ambos superam UNet=0,3451, ST-GCN=0,7024, DeepConvLSTM=0,7081 (supervisionados).
  🔶 porque é o único caso do Harbor hoje de "classificação via retrieval" — não generaliza
  ainda para os outros datasets.

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
- 🔶 **Hardware + tokens/s, primeira medição formal (2026-09-16, atividade do grupo PDC)** —
  `llama3.2:1b` via Ollama 0.34.1, Intel i5-1235U (12ª ger., sem GPU dedicada), 23,7 GB RAM,
  arquivo do modelo 1,32 GB. **10,15–15,42 tokens/s** em geração pura (3 execuções, mesmo
  prompt — variação real de ~50%, reportar como faixa, não número único). **Achado**: com
  prompt longo (~1.700 tokens), o gargalo em CPU sem GPU é **ingestão do contexto de entrada**
  (`prompt_eval_duration`), não a geração da resposta — tempo total 50,6s, mas
  `eval_duration` (geração) só 1,3s. Formalizado como 🔶 porque cobre só 1 modelo/1 máquina;
  ver `[[atividade_pdc_rodar_modelo_local_2026-09-16]]`.
- 🔶 Custo agêntico (nº de iterações, taxa de re-retrieval, latência extra) — exigido pelo SoK
  de agentic RAG ([arXiv:2603.07379](https://arxiv.org/pdf/2603.07379): "output-only metrics
  are insufficient for agentic RAG"). `eval/avaliar_rag_agentic.py` (2026-09-14) instrumenta
  isso via `shared.trace`, mas **nunca foi executado** — script pronto, execução pendente (ver
  plano `quero-utilizar-esse-dataset-quizzical-harbor.md`). Sem esse número, a comparação
  Agentic RAG vs. baseline continua incompleta mesmo com a qualidade já medida.
- 🔶 **Variância entre execuções** (nova, 2026-09-16, item 8 do plano de teste/benchmark) —
  para qualquer sistema não-determinístico (LLM com temperature>0), uma métrica de qualidade
  medida em **1 execução só** não é suficiente para decidir promoção: literatura de 2026 exige
  N=3-10 execuções, reportar média±desvio, e só considerar a diferença real se sobreviver ao
  desvio. `eval/avaliar_variancia_adaptive_k.py` implementa isso para a decisão de promover
  Adaptive-k (ganho original de 88,5% vs 76,9% do melhor k fixo vinha de 1 execução só — ainda
  não confirmado se sobrevive a N=3, ver CLAUDE.md). Formalizado como 🔶 porque cobre só essa
  decisão; generalizar para outras comparações de configuração no futuro.
  **Achado colateral importante**: um bug de timeout (`subprocess.run(timeout=2400)` sem
  `try/except`) no script de N execuções foi mal-diagnosticado por 5-6h como "Ollama travando"
  — ver `[[bug_timeout_silencioso_subprocess_2026-09-16]]`. Lição para qualquer harness que
  chama subprocessos longos: sempre capturar `TimeoutExpired` e logar no stdout, nunca deixar
  o traceback só no stderr sem ninguém checando.

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
`rag-multimodal` — RAG multimodal, onde nDCG foi implementado pela primeira vez no projeto, e onde
o princípio "conversão para texto perde sinal discriminativo" está documentado (relevante para
decidir template determinístico vs. VLM/caption em qualquer corpus novo, texto ou imagem).
`migrar-para-langchain` — regra de baseline justo (custo normalizado) referenciada no item 4 acima.
