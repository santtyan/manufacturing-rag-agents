---
name: roadmap-rag-survey
description: Guia vivo baseado no survey "Retrieval-Augmented Generation for Large Language Models" (Gao et al., arXiv:2312.10997) — mapeia rag_hibrido.py contra a taxonomia Naive/Advanced/Modular RAG do artigo, aponta gaps técnicos, e mantém uma lista priorizada de técnicas do survey que valeria implementar no Harbor. Use quando o usuário perguntar "que estágio de RAG o Harbor está" (Naive/Advanced/Modular), pedir para avaliar o RAG contra a literatura, avançar em algum item do roadmap de técnicas de RAG, ou citar/embasar uma decisão de design do RAG com essa referência.
---

# Roadmap Harbor ↔ survey de RAG (Gao et al., 2312.10997)

## A referência (contexto do artigo)

"Retrieval-Augmented Generation for Large Language Models: A Survey" (Gao, Xiong, Gao, Jia,
Pan, Bi, Dai, Sun, Wang, Wang — Tongji/Fudan, arXiv:2312.10997, revisado mar/2024). Survey de
~100 estudos de RAG, organizado em 3 partes: **Retrieval**, **Generation**, **Augmentation**.
Referência completa em `references/resumo_survey_rag.md` (terminologia, tabelas-chave,
citações) — reler antes de usar como fonte em slides/relatórios, não citar de memória.

**Complemento 2025-2026**: `references/resumo_advanced_rag_2025_2026.md` traz pesquisa mais
recente especificamente sobre Advanced RAG (query rewrite, hybrid fusion, reranking, context
compression), com achados que o survey de 2023 não cobre — reler junto com o resumo principal
antes de decidir o que implementar.

**Regra fixada pelo usuário (2026-09-08)**: toda implementação de técnica desta skill usa
componentes LangChain/LangGraph — nunca Python puro. Ver mapeamento técnica→componente na
seção "Itens implementáveis" abaixo.

### As 3 gerações de RAG do artigo (usar para responder "em que estágio o Harbor está")

- **Naive RAG**: indexação → retrieval → geração, pipeline fixo ("Retrieve-Read").
- **Advanced RAG**: adiciona otimização pré-retrieval (indexação melhor, reescrita de query)
  e pós-retrieval (rerank, compressão de contexto) — ainda uma cadeia linear.
- **Modular RAG**: módulos substituíveis/reconfiguráveis, retrieval iterativo/recursivo/
  adaptativo, roteamento entre fontes, memória, mais integração com fine-tuning.

**Avaliação do Harbor contra isso**: `rag/rag_hibrido.py` (`RAGHibrido`) já está em **Advanced
RAG** — tem otimização de indexação (chunking por seção Markdown, ver docstring de
`chunk_texto()`), hybrid search na etapa de retrieval (E5 denso + TF-IDF/BM25 lexical, seção
"Mix/hybrid Retrieval" do survey), e pós-retrieval com rerank Cross-Encoder (seção "Reranking").
O sistema mais amplo do Harbor (`dashboard/roteador.py`) já tem uma peça de **Modular RAG**
(Query Routing, seção III-C-3 do survey — roteia entre contexto/RAG/SQL) mesmo sem o RAG em si
ser modular internamente. Não é 100% Modular RAG: falta retrieval iterativo/recursivo/adaptativo
dentro do próprio `RAGHibrido` (ver tabela abaixo).

## Tabela de gap: técnicas do survey vs. Harbor hoje

| Técnica do survey | Seção do artigo | Harbor tem? | Onde / observação |
|---|---|---|---|
| Chunking por estrutura (não char fixo) | III-B-1 Indexing | ✅ | `chunk_texto()`, chunking por `## ` Markdown |
| Metadata attachment | III-B-2 | 🔶 parcial | `file_name`/`chunk_index` no Chroma, mas sem timestamp/filtro por metadata na busca |
| Hierarchical/Knowledge Graph index | III-B-3 | ❌ | corpus pequeno (5-9 manuais), provavelmente não justifica ainda |
| Query expansion/rewrite (multi-query, HyDE) | III-C-1/2 | ❌ | pergunta do usuário vai direto para embedding, sem reescrita |
| Query routing | III-C-3 | ✅ (fora do RAG) | `dashboard/roteador.py`, mas roteia contexto/RAG/SQL, não fontes dentro do RAG |
| Hybrid retrieval (sparse+dense) | III-D-1 | ✅, já migrado | `RAGLangChainBM25RRF` (`rag/rag_hibrido_langchain.py`) — `EnsembleRetriever` (RRF), promovido a produção 2026-09-08. **Achado 2025-2026 confirma a escolha**: RRF usa só posição no ranking, elimina a classe de bug (score não-normalizado) que a normalização manual antiga tinha |
| Fine-tuning do embedding model | III-D-2 | ❌ | E5 usado pronto, sem fine-tuning no domínio industrial em português |
| Reranking | IV-A-1 | ✅, mas não medido | Cross-Encoder `ms-marco-MiniLM-L-6-v2` — **achado 2025-2026**: esse modelo específico degrada -0,3% a -3,1% fora de domínio técnico; nunca foi medido isoladamente se ajuda no corpus do Harbor |
| Context compression/seleção | IV-A-2 | ❌ | os k chunks retornados vão inteiros pro prompt, sem compressão |
| Fine-tuning do LLM gerador | IV-B | ❌ | Ollama com modelos prontos (llama3.2/qwen2.5), sem fine-tuning — mesmo gap já registrado em `[[roadmap-slm-multiagente]]` item 4 |
| Retrieval iterativo | V-A | ❌ | `buscar()` é sempre 1 chamada só |
| Retrieval recursivo | V-B | ❌ | sem decomposição de pergunta complexa em sub-perguntas |
| Retrieval adaptativo (decidir SE precisa buscar) | V-C | 🔶 parcial | os answerability gates de `roteador.py` decidem SE vale chamar RAG/SQL, mas é a nível de roteamento geral, não "o RAG decide se precisa buscar de novo" (padrão FLARE/Self-RAG do survey) |

## Itens implementáveis, em ordem de esforço/retorno

Marcar `[x]` conforme for implementado, com resultado (fato + data), mesmo padrão de
`[[roadmap-slm-multiagente]]`. **Todo item usa LangChain/LangGraph** — nunca Python puro (regra
fixada pelo usuário, 2026-09-08).

### Fazer primeiro — mensurável agora, sem nova infraestrutura

- [ ] **0. Medir se o rerank Cross-Encoder atual ajuda ou atrapalha** — achado 2025-2026: modelos
      genéricos tipo `ms-marco-MiniLM` (o que o Harbor usa) degradam -0,3% a -3,1% em domínio
      técnico fora da distribuição de treino. `RAGLangChainBM25RRF.buscar()` já aceita
      `usar_rerank=True/False` — rodar o golden set de retrieval (`eval/avaliar_retrieval.py`)
      nos dois modos e comparar Recall@5/MRR. Não decidir manter/remover o rerank sem essa
      medição — é o item de maior retorno por menor esforço desta lista, porque não exige
      escrever nenhum código novo, só rodar o que já existe com o parâmetro trocado.

### Alto retorno, baixo esforço

- [ ] **1. Query rewrite antes do embedding, via LCEL** — um `RunnableSequence` (`ChatOllama`
      + `PromptTemplate`) que reescreve a pergunta do usuário antes de embedar, corrigindo
      vocabulário informal→técnico (ex.: "esquentando demais" → "temperatura acima do limite").
      Já existe prova de que sinônimos importam (`rag_hibrido.py` linha 417-421). **Ressalva
      2025-2026, avaliar com cuidado antes de adotar**: evidência é mista — reescrita pode
      piorar recall ("query drift"), e HyDE especificamente arrisca alucinar terminologia
      técnica errada com um LLM local pequeno (3B-14B). Testar Recall@5/MRR do golden set
      antes/depois é obrigatório aqui, mais do que em qualquer outro item — não adotar cego.
- [ ] **2. Context compression via `ContextualCompressionRetriever`** (IV-A-2) — usar
      `EmbeddingsFilter` (mais barato, sem chamada de LLM extra) ou `LLMChainFilter` antes de
      montar o prompt em `eval/rag_gerador.py`. **Ressalva 2025-2026**: nenhuma fonte encontrada
      endossa essa técnica especificamente para corpus pequeno (5-9 documentos, chunks curtos)
      — os casos de uso citados na literatura são de 100+ documentos recuperados. Provavelmente
      baixo retorno aqui; medir antes de investir tempo, expectativa é de pouco ganho.

### Esforço médio

- [ ] **3. Retrieval adaptativo real dentro do RAG, via LangGraph critic node** (V-C, padrão
      Self-RAG/FLARE) — um grafo LangGraph onde um nó avalia o score do primeiro resultado e
      decide se refina a query e busca de novo. Mesmo padrão do item 2 da skill
      `[[migrar-para-langchain]]` (self-repair/DBA-Agent) — implementar UMA VEZ, servindo os
      dois roadmaps, não duplicar. LangGraph tem cookbook nativo de CRAG/Self-RAG.
- [ ] **4. Multi-query (query expansion) via `EnsembleRetriever` com múltiplas queries** (III-C-1)
      — gerar 2-3 variações da pergunta via `ChatOllama`, rodar cada uma contra o
      `RAGLangChainBM25RRF`, fundir com RRF (mesmo mecanismo já usado para sparse+dense, agora
      aplicado a variações de query).

### Baixo retorno hoje / avaliar depois

- **5. Fine-tuning do embedding E5 no domínio industrial em português** — mesmo gap do item 4
  de `[[roadmap-slm-multiagente]]`; exige dataset supervisionado e GPU, maior esforço.
- **6. Hierarchical/Knowledge Graph index** — corpus atual (5-9 manuais curtos) provavelmente
  não justifica a complexidade; reavaliar se o corpus crescer significativamente.
- **7. Retrieval recursivo (decompor pergunta complexa em sub-perguntas)** — só relevante se o
  golden set expor perguntas multi-hop que o RAG atual erra; verificar isso primeiro com o
  harness antes de implementar.
- **8. Metadata auto-retrieval** (III-B-2, LlamaIndex `VectorIndexAutoRetriever` — fora do
  ecossistema LangChain, avaliar se há equivalente LangChain antes de considerar) — achado
  2025-2026: sem evidência de ganho em corpus pequeno; risco de o LLM inferir filtro errado e
  excluir o manual certo. Baixa prioridade.
- **9. ColBERT/late-interaction ou LLM-as-reranker (RankLLM) como substituto do Cross-Encoder**
  — só avaliar SE o item 0 (medir o rerank atual) confirmar que `ms-marco-MiniLM` está
  prejudicando no corpus do Harbor. ColBERT exige reindexação multi-vetor (mudança de
  arquitetura); LLM-as-reranker via Ollama é citado como não-ideal por latência sem serving
  otimizado (vLLM/SGLang), que o Harbor não roda.
- **10. Avaliar adoção de RAGAS como camada complementar ao harness próprio** (Seção VI-D do
  survey, ver `references/resumo_survey_rag.md`) — já sugerido pela equipe CERISE antes
  (`[[avaliacao_llm_cerise]]`), nunca adotado. RAGAS/ARES são LLM-as-judge (delegam o
  julgamento a outro LLM, introduzindo sua própria fonte de erro) vs. o harness próprio do
  Harbor, que é determinístico — não é substituto automático, é complementar: RAGAS escalaria
  mais perguntas mais rápido, o harness próprio continua sendo o mais auditável/controlável.
  Funciona com Ollama local, mesmo achado já registrado para LangSmith em
  `[[migrar-para-langchain]]`. Baixa prioridade — só avaliar se o golden set de 67 perguntas
  virar um gargalo de escala.

## Como usar esta skill

- Se o usuário perguntar "em que estágio de RAG o Harbor está" ou pedir para comparar com a
  literatura, responder com a seção "3 gerações" acima, não repesquisar do zero.
- Se perguntar "o que podemos aplicar agora", a resposta é: **item 0 primeiro** (medir o rerank,
  zero código novo, maior retorno/menor esforço), depois 1→4 na ordem listada — não repesquisar
  nem redesenhar a priorização do zero.
- Se pedir uma citação/resumo do survey original para slides/relatório, usar
  `references/resumo_survey_rag.md`; se for sobre achados 2025-2026 (query rewrite, RRF,
  reranking fora de domínio), usar `references/resumo_advanced_rag_2025_2026.md` — não citar
  nenhum dos dois de memória, reler antes.
- Ao avançar em qualquer item, usar sempre o componente LangChain/LangGraph indicado (nunca
  Python puro), marcar `[x]` aqui e medir o impacto real via `eval/avaliar_retrieval.py` /
  `eval/rodar_golden.py` (skill `rodar-harness`) antes/depois — mesma disciplina de medição do
  resto do projeto, não aceitar "deveria melhorar" sem medir.
- Itens 3, 5 e 9 se sobrepõem com outras skills (`migrar-para-langchain`, `roadmap-slm-multiagente`)
  — não duplicar o trabalho, linkar e decidir onde implementar primeiro.
