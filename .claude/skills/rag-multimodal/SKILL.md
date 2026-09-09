---
name: rag-multimodal
description: Guia e checklist de implementação para RAG multimodal (texto + imagem) no Harbor — diagramas técnicos e gráficos de sensor, via caption-then-embed com VLM local (Qwen2.5-VL/Ollama) entrando no pipeline LangChain já existente. Use quando o usuário pedir para "adicionar imagem ao RAG", "RAG multimodal", "indexar diagramas/gráficos", ou avançar em algum item do checklist de implementação multimodal.
---

# RAG multimodal (texto + imagem) — Harbor

## Pesquisa e decisão de arquitetura

Pesquisa feita em 2026-09-08 (agente `general-purpose`, ~30 fontes 2025-2026) — resumo completo
em `references/resumo_rag_multimodal_2025_2026.md`, reler antes de citar ou revisar a decisão.

**Decisão de arquitetura**: **caption-then-embed**, não embeddings compartilhados (CLIP/SigLIP).
Um VLM (vision-language model) gera uma legenda textual de cada imagem; a legenda vira um
`Document` de texto comum e entra no pipeline LangChain já existente
(`rag/rag_hibrido_langchain.py`, `EnsembleRetriever`/RRF), sem trocar vectorstore nem depender
de `langchain_experimental` (pacote instável).

**Por quê, resumido** (detalhe completo na referência):
- CLIP é comprovadamente fraco em ler texto/rótulo embutido em imagem (achado 2025-2026,
  "modality gap" e falha de "attribute binding") — exatamente o problema de um diagrama técnico
  com múltiplos componentes rotulados.
- `langchain_experimental.open_clip` + `Chroma.add_images()` é experimental — arriscaria a
  estabilidade que o RAG acabou de ganhar ao migrar para `langchain-chroma` de produção
  (ver `[[migrar-para-langchain]]`).
- Caption-then-embed é a via mais madura em produção e a que menos briga com a arquitetura
  atual — o objeto indexado continua sendo texto.
- ColPali/ColQwen2 (page-as-image) e GraphRAG-sobre-estrutura (ex. DEXPI para P&ID) ficam como
  vias secundárias de roadmap — mais pesadas/menos maduras em LangChain hoje, e a segunda só
  faz sentido se houver diagrama em formato estruturado formal, que o Harbor não tem (dados são
  sintéticos, sem padrão DEXPI).

**Modelo VLM**: `qwen2.5-vl` via Ollama — sensivelmente melhor que LLaVA no mesmo porte para
conteúdo visual estruturado (gráficos, tabelas, OCR), segundo a pesquisa. **Risco de
compatibilidade conhecido**: versões recentes do Ollama podem não conectar corretamente o
projetor multimodal (`mmproj`) de alguns modelos Qwen mais novos — testar
`ollama pull qwen2.5-vl` (não `qwen3-vl`) e confirmar que a via de imagem realmente funciona
ANTES de comprometer a arquitetura. Se falhar, fallback documentado: `llava` ou `moondream`
(mais leve, <4GB, mas entendimento mais limitado de cena complexa).

## Regra: tudo via LangChain

Mesma regra fixada pelo usuário para Advanced RAG (ver `[[roadmap-rag-survey]]`) — captioning
via `ChatOllama` multimodal (ou chamada direta ao VLM se `ChatOllama` não suportar imagem de
forma estável na versão instalada, documentar a exceção se acontecer), indexação via o mesmo
`RAGLangChainBM25RRF`/`EnsembleRetriever` já em produção.

## Checklist de implementação

Escopo desta sessão é um **smoke test end-to-end**, não o roadmap inteiro — confirmar que o
pipeline funciona com poucas imagens antes de expandir.

- [x] **1. Gerar imagens sintéticas de teste** (2026-09-08) — 4 gráficos técnicos via
      `matplotlib`, a partir de dados reais de `outputs/pipeline2_legacy_sensor/` (temperatura,
      vibração, pressão/vazão por classe Fault/Normal) e `outputs/pipeline4_five_axis_cnc/`
      (anomalias por componente do CNC). Script: `rag/gerar_imagens_sinteticas.py`. Salvo em
      `rag/manuais_imagens/`.
- [x] **2. Confirmar VLM funcional** (2026-09-08) — `qwen2.5-vl`/`qwen2.5vl` **FALHOU**:
      `ollama._types.ResponseError: llama-server startup failed ... Failed to load CLIP model`
      — confirma exatamente o risco de compatibilidade já documentado (mmproj não carrega).
      Bloqueio adicional real: disco C: estava 100% cheio (0 bytes livres), impedindo baixar
      qualquer modelo alternativo até remover o `qwen2.5vl` quebrado (liberou 6GB). Fallback
      usado: **`moondream`** (~1.7GB), funcional mas com captioning pobre/inconsistente (ver
      item 3).
- [x] **3. Gerar legendas e indexar** (2026-09-08) — `rag/rag_multimodal_langchain.py`.
      **ACHADO REAL #2, importante para qualquer código futuro que combine Ollama+LangChain
      com sentence-transformers/torch**: importar `langchain_ollama.ChatOllama` e depois
      `HuggingFaceEmbeddings`/`sentence-transformers` no MESMO processo Python causa
      **segmentation fault** (exit 139) nesta máquina — reproduzido isolado, mesmo sem chamada
      de rede, só com os dois imports presentes. Solução aplicada: captioning (`--captionar`) e
      indexação/busca rodam como **dois processos Python separados**, com todos os imports
      relevantes movidos para dentro das funções (nunca no topo do módulo) — ver docstring do
      arquivo. Indexado numa coleção separada (`chroma_db_multimodal`), não a de produção.
- [x] **4. Smoke test manual** (2026-09-08) — 3 perguntas de teste rodadas sem rerank
      (`usar_rerank=False`). Resultado: 2 de 3 perguntas acharam a imagem certa em 1º lugar
      (vibração e pressão/vazão); a pergunta sobre anomalias de componente do CNC NÃO achou a
      imagem certa em nenhuma das duas posições retornadas — o gráfico de anomalias nunca
      apareceu no top-2 de nenhuma pergunta. **Causa provável: qualidade da legenda do
      `moondream`**, que só descreveu de forma genérica ("bar graph", "line graph") sem
      capturar o conteúdo semântico real do gráfico (qual componente teve mais anomalias) —
      confere exatamente o achado 6.4 da pesquisa ("captioning perde detalhe técnico
      preciso"), mais agravado ainda por ser um VLM pequeno (<2B) em vez do Qwen2.5-VL 7B
      recomendado. **Conclusão do smoke test**: pipeline funciona end-to-end (sem crash,
      retrieval plugado no `RAGHibrido` de produção), mas a qualidade do captioning com
      `moondream` é insuficiente para uso real — não promover para produção nem expandir sem
      antes resolver o item 2 com um VLM maior.

## Estado da arte 2026 e critério de escolha de arquitetura (pesquisa atualizada, 2026-09-08)

Três arquiteturas competem hoje e **não há vencedor universal** — atualiza a decisão acima com
o resultado de uma segunda rodada de pesquisa:

- **caption-then-embed** (a que o Harbor usa): mais simples, adequada a corpora pouco
  complexos visualmente. Continua a escolha certa para o Harbor enquanto o corpus for pequeno
  e o custo de indexação (1 vetor de texto por imagem) importar mais que recall máximo.
- **Embeddings visuais unificados** (Cohere Embed 4, voyage-multimodal-3.5): agora
  competitivos com ColPali na maioria dos corpora empresariais, a uma fração do custo de
  armazenamento — via de meio-termo se caption-then-embed não bastar mas ColPali for caro
  demais para o volume de imagens.
- **Late interaction / page-as-image** (ColPali, ICLR 2025, ColQwen2.5, ColNomic): SOTA em
  ViDoRe e em UNIDOC-BENCH para layout difícil e documentos escaneados, ao custo de ~1.000
  vetores de patch por página — não cabe no `ChromaDB` de produção sem uma coleção dedicada
  com esquema diferente. Achado relevante: RAG textual + rerank ColPali **nem sempre** bate
  ColPali sozinho — em documentos escaneados, erro de OCR no ramo textual derruba o híbrido
  abaixo do ColPali puro.

**Critério prático de escolha**: late interaction quando recall é o gargalo e as queries são
visualmente difíceis (ex. localizar um número específico dentro de uma tabela complexa);
caption-and-index quando não são. **Regra do Harbor**: não trocar de arquitetura antes de medir
o baseline atual — ver item 5 abaixo, já executado.

## Item 5 executado: harness formal de retrieval multimodal (2026-09-08)

Promovido de smoke test (3 perguntas manuais, stdout) para módulo medido:

- `eval/golden_questions_multimodal.json` — 7 perguntas cobrindo as 4 imagens de
  `rag/manuais_imagens/`, mesmo formato de 1-documento-relevante de
  `eval/avaliar_retrieval.py`.
- `eval/avaliar_rag_multimodal.py` — Recall@k/Precision@k/MRR sobre o cache de legendas já
  gerado (`rag/legendas_cache.json`), reusando `RAGLangChainBM25RRF`. **Não chama Ollama** —
  roda no mesmo processo sem risco do segfault do achado #2 (só toca sentence-transformers).

**Resultado do baseline** (k=3, sem rerank, `moondream` como VLM): **Recall@3 = 86%, MRR =
0,524, Precision@3 = 29%**. Corpus de só 4 imagens torna o número frágil estatisticamente (uma
troca de posição muda o resultado em pontos percentuais grandes), mas é a primeira medição que
existe — antes disso a avaliação era "funcionou/não funcionou" em 3 perguntas manuais. As
legendas do `moondream` seguem qualitativamente pobres (uma legenda saiu como
`"[0.0, 0.13, 0.99, 0.28]"`, coordenadas cruas sem conteúdo semântico) — o número de retrieval
"aceitável" esconde uma camada de captioning que não está de fato descrevendo o gráfico.

**Não decidir arquitetura nova a partir deste número isolado** — ele serve de baseline para
comparar contra qualquer mudança futura (troca de VLM, ou arquitetura), não como veredito de
que caption-then-embed "funciona bem" no geral.

## Late interaction (ColModernVBERT) instalado e validado (2026-09-09)

Benchmark padrão-ouro 2026 é **ViDoRe** (v3, ACL 2026) e, para RAG multimodal ponta-a-ponta,
**UNIDOC-BENCH** (arXiv:2510.03663) — compara sob protocolo unificado 4 paradigmas (texto-só,
imagem-só, fusão texto-imagem, embedding multimodal conjunto); achado central: fusão
texto-imagem supera abordagens unimodais isoladas.

**Restrição real de ambiente**: ViDoRe v3 completo (26.000+ páginas) e até o subset menor de
domínio compatível (`vidore_v3_industrial`, 5.244 páginas/2,68GB) são grandes demais para o
disco disponível (13-22GB livres). A máquina também não tem GPU CUDA
(`torch.cuda.is_available() == False`), tornando ColQwen2.5 (~3B parâmetros, a variante SOTA de
ColPali) inviável em CPU. Escolhida a alternativa leve: **ColModernVBERT**
(`ModernVBERT/colmodernvbert`, 250M parâmetros, ~10x menor que ColPali, só 0,6 pontos de nDCG@5
abaixo no ViDoRe agregado), via `sentence-transformers[image]>=6.0`.

**Pegadinhas de ambiente novas, encontradas ao instalar (2026-09-09)**:
1. **Ordem de import de `pandas` vs `sentence_transformers` importa nesta máquina Windows** —
   `from sentence_transformers import MultiVectorEncoder` (ou `SentenceTransformer`) como
   PRIMEIRO import pesado do processo causa access violation dentro de `pyarrow/__init__.py`
   (conflito de DLL nativa entre a versão de `pyarrow` que `sentence_transformers` carrega
   internamente via `pandas.compat.pyarrow` e alguma inicialização de estado). Reproduzido
   determinístico 3x. **Fix**: sempre `import pandas` (import puro, sem uso) ANTES de qualquer
   import de `sentence_transformers` num processo novo. A produção (`rag/rag_hibrido.py`) nunca
   bateu nisso porque outros módulos do projeto já importam `pandas` antes dela no processo real
   do dashboard/harness — só apareceu num processo Python isolado rodando só o import.
2. **Dependências de versão não documentadas no pip install básico**: `ModernVBERT/colmodernvbert`
   exige `transformers>=5.15` (erro explícito e claro se a versão for menor — instalado
   `5.16.1`) e o pacote `peft` (LoRA adapter do modelo, erro igualmente claro faltando).
   `pip install -U "sentence-transformers[image]>=6.0.0"` sozinho não traz nenhum dos dois.
3. **Custo de indexação bem maior que captioning**: embedar 1 imagem (960x600) levou ~31s em CPU
   (vs. segundos para gerar 1 legenda via `moondream`) — carregamento do modelo em si levou
   ~47s (one-time por processo). Nada crítico para um corpus pequeno, mas o `EmbeddingsFilter`/
   indexação em lote de um corpus real precisa contabilizar esse custo — ColModernVBERT ainda é
   ~10x mais rápido de indexar que ColPali/ColQwen2.5 completo, mas não é gratuito.

Smoke test confirmado: `model.encode_document([imagem])` + `model.encode_query([pergunta])` +
`model.similarity(...)` funcionam ponta a ponta sobre uma imagem real de
`rag/manuais_imagens/`, com score de similaridade calculado sem erro.

## Benchmark 2 estágios (retrieval + rerank) executado (2026-09-09)

`eval/avaliar_benchmark_multimodal_2x2.py` — desenho de produção real (paper HEAVEN,
arXiv:2510.22215): retrieval barato (caption-then-embed/E5) filtra candidatos sobre todo o
corpus, ColModernVBERT só rerankeia o top-k/top-p já filtrado (nunca o corpus inteiro — inviável
em CPU, ~36s/imagem). Duas estratégias de seleção de candidatos comparadas, análogas a top-k/
top-p de sampling de geração de texto mas aplicadas a scores de similaridade de retrieval:
top-k fixo (sempre 5 candidatos) e "top-p" (corte por massa cumulativa de score normalizado,
limiar 0,85).

**Resultado real** (ViDoRe subset, 2 queries de teste; corpus Harbor, 4 imagens/golden set completo):

| Corpus | Estratégia | nDCG@5 | Custo | Candidatos rerankeados |
|---|---|---|---|---|
| ViDoRe (60 pág.) | só retrieval | 1.000 | 31,6s | — |
| ViDoRe | rerank top-k=5 | 0.815 | 302s | 10 |
| ViDoRe | rerank top-p=0.85 | 1.000 | 100s | 2 |
| Harbor (4 img.) | só retrieval | 0.823 | 9,2s | — |
| Harbor | rerank top-k=5 | **1.000** | 831,5s | 28 |
| Harbor | rerank top-p=0.85 | **0.286** | 175,9s | 7 |

**ACHADO REAL importante — top-p não é universalmente melhor, e piorou drasticamente no corpus
pequeno**: no ViDoRe, top-p empatou com "sem rerank" (ambos 1.000) e foi 3x mais barato que
top-k. No corpus Harbor, top-p **piorou** o resultado (0.286, pior que nem rerankear) enquanto
top-k acertou tudo (1.000).

**CAUSA RAIZ REAL, investigada e corrigida em 2026-09-09** (a hipótese anterior — "limiar mal
calibrado por causa de scores concentrados" — estava errada): o script testou `TOP_P_LIMIAR`
0,7, 0,85 e 0,99 no corpus Harbor e obteve **resultado idêntico nos três** (sempre 7 candidatos
rerankeados). Isso não é questão de calibração — é que `estagio1_retrieval()` chama
`rag.buscar(..., usar_rerank=False, ...)`, e sem rerank `RAGHibrido.buscar()` **não popula
score real** (todos os candidatos vêm com `score=0.0`, confirmado diretamente). `selecionar_top_p`
sempre cai no caso "sem sinal, retorna só 1 candidato" — a variável que o limiar deveria
controlar nunca é exercida, porque não há massa de score para cortar. **Não decidir estratégia
de produção a partir deste número** — o item de continuidade correto agora é dar ao 1º estágio
um sinal de score real (ex. `usar_rerank=True` também no estágio 1, ou expor o score RRF do
`EnsembleRetriever`) antes de re-tentar qualquer calibração de `TOP_P_LIMIAR`.

## Itens de roadmap (não bloqueiam a entrega desta sessão)

- [ ] **2b. Resolver VLM de qualidade suficiente** — prioridade real, mais evidente agora com
      número: as legendas do `moondream` incluem uma saída sem sentido semântico
      (`"[0.0, 0.13, 0.99, 0.28]"`). Tentar `qwen2.5-vl`/Qwen2.5-VL novamente quando houver
      mais espaço em disco, ou `llava` (~4GB).
- [ ] **6. Integrar ao dashboard/chat de produção** — aguardando o dataset real do usuário
      chegar (imagens sintéticas atuais são só prova de conceito); reavaliar Recall@k contra
      esse dataset antes de decidir arquitetura final, seguindo o critério acima.
- [ ] **7. ColPali/ColQwen2 como via secundária** — decisão adiada explicitamente até o dataset
      real chegar (ver critério de escolha acima): usar quando recall for o gargalo medido E as
      queries forem visualmente difíceis, não antes.
- [ ] **8. GraphRAG-sobre-estrutura (ex. DEXPI)** — só relevante se um dia houver diagrama real
      em formato estruturado formal (P&ID), não para os dados sintéticos atuais.

## Como usar esta skill

- Ao ser invocada, reler a decisão de arquitetura acima antes de sugerir abordagem — não
  redecidir CLIP vs. captioning do zero a cada vez.
- Seguir o checklist 1→4 em ordem na primeira implementação; os itens 5-8 são roadmap, não
  bloqueiam entrega.
- Se o item 2 (VLM funcional) falhar com `qwen2.5-vl`, documentar o fallback usado nesta seção
  do arquivo antes de prosseguir — é informação que a próxima invocação da skill precisa saber
  para não repetir a mesma tentativa falha.
- Medir antes de apontar produção para o pipeline multimodal — mesma disciplina do resto do
  projeto (`[[migrar-para-langchain]]`, `[[roadmap-rag-survey]]`).
