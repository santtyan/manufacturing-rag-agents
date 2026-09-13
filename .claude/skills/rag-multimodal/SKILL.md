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

## Item 1 do plano "Evoluir o RAG multimodal" executado: golden set expandido + checks de fidelidade (2026-09-09)

**Motivação**: o golden set de 7 perguntas/4 imagens usado até aqui é matematicamente incapaz
de detectar qualidade de captioning — com corpus de 4 documentos e k=3, um ranqueador aleatório
já acerta Recall@3=75% por construção (o baseline de 86% medido estava a 1 pergunta de
distância do acaso puro), e Precision@3 estava hard-capped em 33% (1 relevante em 3
devolvidos). Nenhuma comparação de VLM seria interpretável nesse desenho.

**Corpus expandido**: `rag/gerar_imagens_sinteticas.py` de 4 para 26 imagens, com 4 famílias de
distratores adversariais deliberados (mesmo dado/tipo diferente, eixos trocados, ordenação
diferente, variáveis novas do mesmo CSV) — desenhados para forçar o sinal a vir do caption, não
do BM25/nome de arquivo. Ground truth factual de cada imagem em
`rag/metadados_imagens_ground_truth.py` (tipo, título, eixos, ranking esperado, variáveis
fonte). `eval/golden_questions_multimodal.json` de 7 para 29 perguntas, incluindo pares
discriminativos deliberados (ex. ranking ascendente vs. descendente do mesmo gráfico CNC,
boxplot vs. barra da mesma variável, eixos trocados voltagem/corrente).

**Checks determinísticos de fidelidade de caption** (novo módulo `eval/checks_fidelidade_caption.py`,
sem LLM-judge — o ground truth é conhecido de antemão porque as imagens são geradas
programaticamente): `nao_degenerado`, `idioma_pt`, `tipo_grafico_correto`,
`menciona_eixos_corretos`, `contem_ranking`, `sem_numeros_inventados`. Critério de promoção de
um VLM novo, definido antes de rodar (não depois de ver o número): **≥90% de taxa média de
aprovação E 100% em `sem_numeros_inventados`**. 15 testes unitários em
`tests/test_checks_fidelidade_caption.py`, cada um mapeado a um achado real nas legendas do
moondream (inclui 2 bugs reais corrigidos no próprio código dos checks durante o
desenvolvimento: a regra "ignorar números entre 0-1" mascarava exatamente os números
inventados que o check deveria pegar; a margem de folga fixa `±1` era desproporcional para
variáveis de amplitude real pequena — corrigida para margem proporcional à amplitude).

**Resultado real, moondream no corpus de 26 imagens**: taxa média de fidelidade **42,3%**, **0
de 26** legendas passando todos os checks, **19 de 26** sem números inventados. Bem abaixo do
critério de promoção (≥90%/100%) — o moondream está definitivamente reprovado como VLM de
produção, não é um julgamento qualitativo, é um número.

**Baseline honesto de retrieval no corpus expandido** (`eval/avaliar_rag_multimodal.py`, sem
mudança de código — só corpus/golden set maiores): **Recall@3 caiu de 86% para 38%,
Precision@3 de 29% para 13%, MRR de 0,524 para 0,253**. Essa queda é **esperada e correta**, não
regressão — o número de 86% era artefato estatístico de um corpus pequeno demais para o k
usado; 38% é a primeira medição de retrieval multimodal do Harbor que de fato significa algo.

**Conclusão**: o item 1 (pré-requisito do plano) está completo. Trocar o VLM agora (item 3 do
plano, `granite3.2-vision:2b`/`qwen3-vl:4b`) já produz uma comparação interpretável contra este
baseline (moondream 42,3%/38%/13%/0,253), ao contrário de antes.

## Item 3 do plano: candidato `granite3.2-vision:2b` testado e REPROVADO (2026-09-09)

Smoke test de 1 imagem antes de investir no harness completo (regra do plano, evita repetir o
custo do qwen2.5vl que falhou em 2026-09-08). Resultado real:

- **Tempo**: 282,1s (quase 5 minutos) para UMA legenda em CPU — inviabiliza rodar as 26 imagens
  do corpus expandido (>2h só neste modelo).
- **Qualidade**: legenda retornada foi `"Line 1 trends horizontal at ylabel 0.00."` — sem tipo
  de gráfico reconhecível, sem título, sem eixos, sem conteúdo semântico. Reprovaria de
  imediato pelo critério de promoção (`tipo_grafico_correto`, `menciona_eixos_corretos`,
  `idioma_pt` todos falhariam).

**Não prosseguir com `granite3.2-vision:2b`** — reprovado tanto por custo quanto por qualidade,
sem precisar rodar o harness completo. Próximo candidato do plano: `qwen3-vl:4b`.

## Item 3 do plano: candidato `qwen3-vl:4b` testado — qualidade muito superior, custo proibitivo (2026-09-09)

**Superado pelo resultado do corpus completo em 2026-09-10, ver seção "Item 3 do plano:
CONCLUÍDO" mais abaixo** — a decisão de custo/benefício aqui foi tomada (rodar mesmo assim), e
o resultado real do harness completo (77,5% de fidelidade, reprovado no critério de promoção)
está documentado lá. Seção mantida como registro histórico do smoke test que embasou a decisão.

Mesmo smoke test de 1 imagem. Resultado real:

- **Qualidade**: legenda em português correto, identifica o tipo de gráfico (linha), os dois
  eixos com o texto certo (tempo/índice da leitura, anomalia 0=normal/1=anômalo), e é
  factualmente precisa — a série realmente é constante em 0 no arquivo de teste, e a legenda
  descreve exatamente isso ("não há categorias ou componentes com valores mais altos ou mais
  baixos, já que a anomalia permanece estável"). Primeira legenda de todo o processo que
  passaria os 5 checks de fidelidade sem ressalva.
- **Tempo**: **412,2s (quase 7 minutos) por imagem em CPU** — pior que o granite (282s) apesar
  da qualidade muito maior. Rodar as 26 imagens do corpus expandido levaria **~3 horas** só de
  captioning.

**Decisão**: qualidade aprovaria pelo critério de promoção (a confirmar com o harness
completo), mas o custo de ~3h para indexar 26 imagens é proibitivo para qualquer ciclo de
iteração nesta máquina (sem GPU CUDA). **Não rodar o harness completo (26 imagens) com
`qwen3-vl:4b` sem antes decidir explicitamente que vale a espera** — ao contrário do
`granite3.2-vision:2b` (reprovado por completo), aqui a decisão é de custo/benefício, não de
qualidade. Se decidir seguir, rodar em background (`nohup ... &`, como já feito com o
moondream) e não bloquear a sessão interativa esperando.

## Item 2 do plano executado: score RRF real atrás de flag (2026-09-09)

Causa raiz já documentada acima (bug do benchmark 2x2) corrigida em `rag_hibrido_langchain.py::buscar()`:
novo parâmetro `usar_score_rrf` (default `False`, flag de ablação, mesmo padrão de
`usar_contexto`). Quando `True` e `usar_hybrid=True`, o RRF é recalculado explicitamente
(retriever denso e BM25Retriever chamados em separado, fundidos por `1/(60+rank)`, mesma
constante `k=60` do `EnsembleRetriever`), expondo o score fundido real em vez do `score: 0.0`
literal que sempre existiu quando `usar_rerank=False`.

**Confirmado sem regressão** contra o RAG de texto de produção: com a flag desligada
(comportamento inalterado), `eval/avaliar_retrieval.py` deu Recall@5=98%/Precision@5=38%/
MRR=0,898 — idêntico ao baseline conhecido. Teste ad-hoc direto na classe LangChain confirmou
que ligar a flag **não muda o ranking** (mesmo Recall/Precision/MRR), mas passa a expor **52
scores distintos** em vez de 1 valor fixo — o sinal que o benchmark 2x2/calibração de top-p
precisava e não tinha.

Achado registrado também: `score_min` (parâmetro da assinatura de `buscar()`) nunca é usado no
corpo da função — documentado na docstring como parâmetro morto, mantido por compatibilidade
de interface (nenhum consumidor real passa esse argumento hoje).

**Próximo passo (item 4 do plano)**: re-rodar a calibração de `TOP_P_LIMIAR` do benchmark 2x2
com `usar_score_rrf=True` no 1º estágio — agora um experimento válido, porque o limiar
finalmente tem massa de score real para cortar.

## Item 3 do plano: CONCLUÍDO — qwen3-vl:4b captionou o corpus completo, reprovado no critério de promoção (2026-09-10)

Terceira tentativa foi a que completou (as 2 primeiras caíram por queda de sessão/desligamento,
ver seção anterior) — só depois de `gerar_legendas()` em `rag/rag_multimodal_langchain.py` ganhar
um parâmetro `caminho_checkpoint` que grava a legenda em disco após CADA imagem e pula as já
processadas ao retomar. Sem isso, qualquer queda no meio das ~3h perdia tudo.

4 das 26 imagens deram resposta VAZIA e silenciosa do Ollama na primeira rodada completa (sem
erro/traceback) — intermitente, não reproduzível: todas completaram em 1-4 tentativas de
reprocessamento isolado (`grafico_anomalia_spindle_tempo` precisou de 4). Cache final:
`rag/legendas_cache_qwen3-vl_4b.json`, 26/26 preenchidas.

**Resultado dos checks de fidelidade** (`eval/checks_fidelidade_caption.py`), depois de também
corrigir **2 bugs de falso positivo** encontrados no próprio check `sem_numeros_inventados`
(commit `b70ebda`):
1. Regex `\d+\.?\d*` não reconhecia vírgula decimal pt-BR — "0,099" virava dois números falsos
   "0" e "099". Fix: `-?\d+(?:[.,]\d+)?` + `.replace(",", ".")`.
2. "5" de "CNC 5 eixos" (nome do equipamento, repetido no título de todo o corpus deste
   dataset) contava como número de dado inventado. Fix: lista de índices ignorados ampliada de
   `(0,1,2,3)` para `(0,1,2,3,4,5)`.

| Estágio | Taxa média | Passou tudo | Sem números inventados |
|---|---|---|---|
| moondream (baseline) | 42,3% | — | 0/26 |
| qwen3-vl bruto (com vazios, bug regex) | 66,5% | 6/26 | 13/26 |
| qwen3-vl corpus completo sem vazios | 76,4% | 11/26 | 15/26 |
| qwen3-vl com os 2 bugs do check corrigidos | **77,5%** | **13/26** | **17/26** |

**RESULTADO FINAL: reprovado pelo critério de promoção** (≥90% de taxa média E 100% sem números
inventados) — mas melhoria real e substancial confirmada sobre o moondream (42,3%→77,5%),
número já confiável (não inflado por bug de metodologia). Restam pelo menos 2 casos de
alucinação numérica GENUÍNA confirmada contra o CSV de origem: voltagem reportada em
~0,086-0,099 V quando o valor real de `Voltage_V` é ~220,19-220,20 V (3 ordens de grandeza de
erro) — vale investigar se é troca de rótulo de eixo ou confusão de escala do VLM, caso o item
3 seja retomado com outro candidato de VLM no futuro.

**Conclusão prática**: nenhum VLM local testado até agora atinge o critério formal — consistente
com a literatura de VLMs pequenos alucinando valores numéricos específicos em gráficos técnicos.
Não relançar captioning de novo salvo se aparecer um VLM candidato novo.

## Item 4 do plano: recalibração de top-p RETOMADA, mas travada em custo de rerank em CPU (2026-09-10/11)

Com o cache do qwen3-vl pronto, `eval/avaliar_benchmark_multimodal_2x2.py` ganhou o argumento
`--cache-legendas` (commit `3d9b9bc`) para recalibrar `TOP_P_LIMIAR` com legendas de um VLM
diferente do padrão, sem precisar sobrescrever `rag/legendas_cache.json` manualmente.

**Achado real de custo**: rodando `--so-harbor --cache-legendas rag/legendas_cache_qwen3-vl_4b.json
--top-p-limiar 0.7`, o corpus Harbor agora tem 26 imagens (não mais 4 como o nome do corpus no
código ainda sugere) — o rerank `ColModernVBERT` em CPU pura custou **48,76s/imagem** (145
imagens rerankeadas = 7070,6s ≈ 2h só para top-k=5). A sessão achou que o processo tinha travado
(log sem output por 2h20min) e matou-o por engano — na verdade estava processando de verdade
(CPU real sendo consumida) e terminou sozinho com exit code 0 no instante exato do kill. Ver
[[feedback_confirmar_cpu_antes_matar_processo]] para a regra geral extraída disso.

Resultado parcial obtido antes do kill (rodada anterior, corpus de 4 imagens, referência):
nDCG@5=1,0/Recall@5=100%/MRR=0,929 para top-k=5 (948,2s) — mas **não é o número do corpus de 26
imagens com legendas do qwen3-vl**, que não chegou a ser salvo (`eval/resultados_benchmark_multimodal_2x2.json`
continua com o resultado antigo de 4 imagens).

**Pesquisa de estado da arte feita** (ver [[gargalo_rerank_multimodal_cpu_2026-09-10]] para o
detalhe completo): ONNX/`optimum-onnx` daria ~3,23x de speedup em CPU mas já **descartado**
(exige `transformers<4.58`, incompatível com `sentence-transformers 6.x` de produção — não
reabrir essa decisão). `FlashRank` é candidato a reranker ONNX-nativo que não colide com essa
restrição, mas suporte a modelos ColBERT-style multimodal não confirmado — checar antes de
investir tempo. O próprio paper do ColModernVBERT promete ~7x speedup vs. modelos parecidos em
CPU, o que sugere que 48,76s/imagem pode não ser o teto físico do modelo.

**Próximo passo real, ao retomar**:
1. Profiling isolado de 1 imagem (cronometrar load/encode_document/encode_query/similarity em
   separado) antes de rodar tudo de novo — descobrir se há gordura para cortar.
2. Se não houver bug óbvio: aceitar o custo, mas rodar em background com `flush=True` explícito
   nos prints (ou `python -u`) em vez de confiar no buffering padrão de `nohup > log.txt` — foi
   isso que tornou impossível diferenciar "travado" de "processando" nesta rodada.
3. Adicionar um `--n-queries-harbor` (hoje só existe `--n-queries-vidore`) para calibrar em
   subset pequeno antes de rodar a bateria completa dos 3 limiares.
4. Avaliar FlashRank como reranker alternativo SE o profiling confirmar que o custo está mesmo
   no rerank em si.

## Item 4 do plano: CONCLUÍDO — recalibração completa, `top-p=0,7` vencedor (2026-09-11)

**Novo bug de ambiente encontrado e corrigido antes de conseguir rodar**: `import pandas` antes
de `sentence_transformers` (achado de 2026-09-09, seção acima) **parou de bastar sozinho** —
`import sentence_transformers` voltou a causar access violation (torch × `pyarrow.dataset`,
puxado internamente por `sentence_transformers.base` → `datasets` → `pyarrow.dataset`). Isolado
por bisseção de imports: o crash exige `torch` + `sklearn` + `pyarrow.dataset` + `datasets` juntos
na mesma ordem que `sentence_transformers` usa internamente; **`import datasets` explícito ANTES
de `import sentence_transformers`** força a ordem de carregamento de DLL que evita o crash — sem
isso nem o profiling isolado rodava. Aplicado em `eval/avaliar_benchmark_multimodal_2x2.py` (logo
após o `import pandas` já existente).

**Profiling isolado (1 imagem, 3 rodadas) confirmou onde o custo realmente está**: load do modelo
8,6s (one-time); `encode_document` **~14,5s consistente entre rodadas** (praticamente 100% do
custo por imagem, sem warm-up nem gordura óbvia); `encode_query` 0,14s; `similarity` 0,01s — os
48,76s/imagem calculados na sessão anterior não eram o custo real do encode em si, mas overhead
acumulado numa rodada mais longa. Não foi necessário trocar de reranker nem investir em
FlashRank/ONNX — o custo medido isoladamente já era razoável, só faltava rodar sem interrupção.

**Recalibração dos 3 limiares** (corpus Harbor 26 imagens, legendas `qwen3-vl:4b`, score RRF real
via `usar_score_rrf=True`):

| Estratégia | nDCG@5 | Recall@5 | MRR | Candidatos rerankeados | Custo |
|---|---|---|---|---|---|
| sem rerank | 0,864 | 93% | 0,679 | — | 37,6-62,2s |
| top-k=5 (fixo) | 0,867 | 93% | 0,693 | 145 | ~3000-3560s |
| **top-p=0,7** | **0,889** | **97%** | **0,702** | 145 | ~3015s |
| top-p=0,85 | 0,802 | 86% | 0,661 | 235 | ~6734s |
| top-p=0,99 | 0,780 | 86% | 0,647 | 290 | ~7055s |

**Achado real, ao contrário da intuição de "mais candidatos = mais chance de achar o certo"**:
limiares de top-p mais permissivos (0,85/0,99) selecionaram bem mais candidatos por pergunta
(235/290 vs. 145) e **pioraram todas as 3 métricas ao mesmo tempo que custaram mais** — o rerank
multimodal precisa de um conjunto de candidatos já filtrado com confiança pelo 1º estágio; incluir
candidatos de score baixo introduz ruído que o ColModernVBERT não consegue corrigir sozinho.
`TOP_P_LIMIAR` do script atualizado de `0,85` (chute inicial nunca validado) para **`0,7`**
(vencedor real, mais barato ainda por cima). Resultados completos em `eval/logs_recalibracao/`.

**Conclusão prática**: com score real e limiar correto, o rerank em 2 estágios (top-p=0,7) supera
tanto "sem rerank" quanto "top-k fixo" em todas as métricas — primeira evidência de que o desenho
de 2 estágios (HEAVEN) realmente compensa o custo do rerank multimodal neste corpus, não só em
teoria. **Mas essa comparação foi feita com o MESMO VLM (moondream) nos dois lados** — o item 5
testa se o rerank compensa TROCAR de VLM, pergunta diferente e mais importante para a decisão de
arquitetura.

## Item 5 do plano: CONCLUÍDO — rerank NÃO compensa captioning ruim (2026-09-11)

**Pergunta**: o rerank multimodal (ColModernVBERT, top-p=0,7 calibrado no item 4) compensa um
VLM de captioning ruim, ou a qualidade do VLM importa mais que o rerank? Script novo
`eval/avaliar_rerank_compensa_captioning.py` (reusa `estagio1_retrieval`/`estagio2_rerank` do
benchmark 2x2, sem duplicar lógica), comparando 2 cenários sobre o mesmo corpus/golden set:

| Cenário | nDCG@5 | Recall@5 | MRR | Custo |
|---|---|---|---|---|
| A) moondream (caption ruim) + rerank top-p=0,7 | 0,436 | 48% | 0,362 | 4779,4s (189 img. rerankeadas) |
| **B) qwen3-vl (caption bom) SEM rerank** | **0,864** | **93%** | **0,679** | **11,8s** |

**VEREDITO CLARO: o rerank NÃO compensa captioning ruim.** (B) supera (A) em TODAS as métricas
por margem larga (quase o dobro de nDCG/Recall) E custa ~400x menos (11,8s vs. 4779,4s) — não é
resultado marginal nem ambíguo. A hipótese do desenho HEAVEN (retrieval barato + rerank caro
recupera qualidade de captioning ruim) **não se sustenta neste corpus**: o rerank multimodal
opera sobre os candidatos que o 1º estágio (embedding da legenda) já selecionou — se a legenda
em si não descreve o conteúdo real da imagem (moondream: "bar graph", "line graph" genéricos,
às vezes texto degenerado), nenhum rerank visual recupera informação que nunca entrou no texto
indexado. O rerank reordena o que já foi selecionado; não resgata candidatos que o retrieval
textual nunca trouxe para perto do top-k por causa de uma legenda ruim.

**Implicação para a arquitetura de produção**: o esforço deve ir para achar/aprovar um VLM de
qualidade suficiente (item 3, ainda sem candidato aprovado — qwen3-vl chegou perto, 77,5% de
fidelidade, mas reprovado pelo critério de ≥90%), não para otimizar o rerank multimodal. O
rerank ColModernVBERT como está calibrado hoje (top-p=0,7) é valioso como ganho incremental
SOBRE um bom captioning (ver item 4: rerank supera "sem rerank" quando o VLM já é o mesmo dos
dois lados), mas não é substituto para resolver a qualidade da legenda em si. Não promover o
rerank multimodal para produção como forma de "economizar" na escolha do VLM.

**Ressalva de validade**: comparação com apenas 1 VLM ruim (moondream) e 1 VLM bom (qwen3-vl,
ainda reprovado formalmente) — não é uma prova formal de que NENHUM rerank compensa NENHUM VLM
ruim, é evidência forte neste corpus/estas condições. Resultados completos em
`eval/resultados_item5_rerank_vs_captioning.json`.

## Item 6 PARCIALMENTE DESTRAVADO — trilha IMU (OpenPack) integrada e validada (2026-09-13/14)

O dataset real do usuário chegou: **OpenPack** (Yoshimura et al., PerCom 2024, HAR de operações
de embalagem via sensores IMU vestíveis, licença CC BY-NC-SA 4.0 — ver
`docs/openpack_licenca_e_atribuicao.md`). Mas a arquitetura aplicada **não é** caption-then-embed
com VLM — é a trilha determinística do RAG-HAR (arXiv:2512.08984): features estatísticas do
sinal (`media, maximo, minimo, q1, q3, desvio_padrao, mediana, n_picos`) convertidas em texto por
template fixo (`rag/rag_openpack_texto.py`), sem nenhum VLM/captioning. É o **oposto** do gargalo
de custo desta skill (412s/imagem, nenhum VLM aprovado) — aqui o "captioner" é `numpy`: custo
~zero e fidelidade 100% por construção, porque o texto é derivado deterministicamente do dado,
não interpretado por um modelo.

**Achado central, reforça a decisão de arquitetura desta skill de um ângulo novo**: mesmo com
fidelidade perfeita do texto, o **retrieval por instância** (BM25/E5 tentando achar "a janela
certa") falhou (Recall@5=0%) — não por bug, mas porque séries de sensor segmentadas em janelas
são um problema de retrieval **class-level**, não instance-level (nenhuma janela específica é "a
resposta certa"; o que importa é achar vizinhos da mesma classe). Corrigido trocando a métrica
para k-NN label purity (12,9%, ver skill `metricas-avaliacao-ia-industrial`) e, mais importante,
implementando o protocolo de **classificação** k-NN do próprio RAG-HAR: **F1-macro = 0,9217 e
0,9114** (dois protocolos) contra o benchmark oficial `openpack-torch`, superando os 3 baselines
supervisionados (UNet=0,3451, ST-GCN=0,7024, DeepConvLSTM=0,7081) — ver skill `rodar-harness`
para os comandos e `eval/resultados_classificacao_openpack_completo.json` para o resultado
completo.

**Princípio geral, confirmado de forma independente 3 vezes (varredura de ~20 referências de
GitHub/papers enviadas pelo usuário em 2026-09-13, nenhuma diretamente aplicável ao domínio
HAR/sensor, mas 2 achados indiretos relevantes)**: **conversão para texto perde sinal
discriminativo**, sempre que o texto for uma *interpretação* (VLM/caption) em vez de uma
*derivação determinística* (features estatísticas via numpy/pandas). Confirmado (1)
empiricamente no próprio Harbor duas vezes — aqui (retrieval por instância falha mesmo com
templates fiéis) e no item 5 acima (rerank não compensa captioning ruim); (2) MAVIS
(arXiv:2511.12142) — LVLMs têm "text dominance" e groundedness mais fraca em documentos de
imagem; (3) M4-RAG (arXiv:2512.05959) — "naive text-based retrieval... converting image to text
introduces noise". **Regra de design**: sempre que o dado de origem permitir representação
textual determinística (sensor, série temporal, dado tabular), preferir isso a VLM/caption —
mesmo que o retrieval final precise de uma métrica diferente (class-level) do que se esperaria
inicialmente. Referência de vocabulário de métricas: `Multimodal-RAG-Survey` (Abootorabi et al.,
ACL 2025 Findings, arXiv:2502.08826) — usar para nomear categorias de avaliação, não como
fundamento de arquitetura (é majoritariamente sobre imagem/vídeo/áudio via CLIP/VLM).

**O que continua bloqueado**: a trilha RGB (imagens sintéticas de gráfico, câmera do OpenPack)
segue sem VLM aprovado (qwen3-vl:4b reprovado, 77,5% < critério de 90%) — item 6 permanece
parcialmente aberto para essa trilha especificamente. E `dashboard/app.py` **ainda não tem aba de
chat para o OpenPack** — os CSVs de `outputs/pipeline8_openpack/` e o corpus RAG existem e estão
validados, mas não há UI consumindo isso ainda (pendência aberta, não implícita).

## Itens de roadmap (não bloqueiam a entrega desta sessão)

- [ ] **2b. Resolver VLM de qualidade suficiente** — prioridade real, mais evidente agora com
      número: as legendas do `moondream` incluem uma saída sem sentido semântico
      (`"[0.0, 0.13, 0.99, 0.28]"`). Tentar `qwen2.5-vl`/Qwen2.5-VL novamente quando houver
      mais espaço em disco, ou `llava` (~4GB).
- [x] **6 (parcial). Integrar ao dashboard/chat de produção — trilha IMU/OpenPack** — ver seção
      acima. Restam: aba de chat no Streamlit (pendência aberta) e a trilha RGB (segue bloqueada
      por falta de VLM aprovado).
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
