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

## Itens de roadmap (não bloqueiam a entrega desta sessão)

- [ ] **2b. Resolver VLM de qualidade suficiente** — prioridade real após o smoke test:
      `moondream` não é suficiente (achado item 4 acima). Tentar `qwen2.5-vl`/Qwen2.5-VL
      novamente quando houver mais espaço em disco (a falha de carregamento do projetor pode
      ou não estar relacionada ao disco cheio — não foi isolado; testar de novo com disco
      livre antes de descartar o modelo definitivamente) ou `llava` (~4GB, não testado ainda
      por falta de espaço em disco na tentativa desta sessão).
- [ ] **5. Harness formal para retrieval multimodal** — golden set de perguntas sobre as
      imagens, medindo Recall@k/MRR como já se faz para texto (`eval/avaliar_retrieval.py`).
- [ ] **6. Integrar ao dashboard/chat de produção** — só depois do item 5 confirmar que o
      retrieval multimodal não regride nada; seguir a mesma regra inegociável de
      `[[migrar-para-langchain]]` (não apontar produção para algo não medido).
- [ ] **7. ColPali/ColQwen2 como via secundária** — avaliar se compensar quando o volume de
      diagramas crescer o suficiente para a legenda perder informação demais na prática
      (mensurável, não decidir por achismo).
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
