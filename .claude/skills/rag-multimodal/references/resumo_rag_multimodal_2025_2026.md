# Resumo estruturado — RAG multimodal (imagem), estado da arte 2025-2026

Pesquisa feita em 2026-09-08 (agente de pesquisa, ~30 fontes), aplicada ao contexto do Harbor:
RAG 100% local (Ollama, sentence-transformers, ChromaDB), já migrado para LangChain
(`langchain-chroma`, `EnsembleRetriever` RRF), corpus alvo é diagrama técnico/gráfico de sensor
sintético, não foto natural. Reler antes de citar em slides/relatório.

## 1. Duas arquiteturas dominantes, sem vencedor único

- **Embeddings compartilhados** (CLIP/SigLIP/JinaCLIP): imagem e texto no mesmo espaço vetorial,
  busca direta por similaridade, sem texto intermediário.
- **Caption-then-embed**: VLM gera legenda textual da imagem, indexada com o pipeline de texto
  já existente.
- Terceira via emergente: embeddings multimodais unificados de alta capacidade (Cohere Embed 4,
  Voyage-multimodal-3 — APIs pagas, incompatíveis com requisito 100% local) e **page-as-image /
  late-interaction** (ColPali, ColQwen2 — ver seção 4).

**Trade-off**: caption-then-embed é simples de encaixar num pipeline de RAG textual maduro, mas
é lossy — detalhe que a legenda não capturar (valor numérico, rótulo de componente) some da
base pesquisável. Embeddings diretos preservam mais sinal visual bruto, mas são fracos em texto
embutido na imagem (ver seção 6) e custam mais para indexar/manter.
[BigData Boutique, "Multimodal RAG in 2026"](https://bigdataboutique.com/blog/multimodal-rag-retrieval-over-images-pdfs-and-text),
[Zilliz, "Multimodal RAG locally with CLIP and Llama3"](https://zilliz.com/blog/multimodal-RAG-with-CLIP-Llama3-and-milvus).

## 2. Modelos VLM locais viáveis via Ollama (sem GPU dedicada de produção)

| Faixa de hardware | Modelo recomendado 2026 | Observação |
|---|---|---|
| Sem GPU / <4GB VRAM | Moondream 2 (1.9B) | única opção prática nessa faixa; entendimento limitado |
| 6-8GB | MiniCPM-V 4.5 (8B) / LLaVA 1.6 7B | OCR de documento, Q&A de imagem |
| 8-16GB | **Llama 3.2 Vision 11B** | VLM local mais forte para documento/foto nessa faixa |
| 40GB+ | Qwen2.5-VL 72B / LLaVA 34B | fora do alcance do Harbor |

**Achado mais específico para o Harbor**: Qwen2.5-VL 7B produz resultados sensivelmente
melhores que LLaVA no mesmo tamanho para conteúdo visual estruturado (gráficos, tabelas,
screenshots, OCR), porque o treino incluiu muito mais desse tipo de conteúdo.
[InsiderLLM](https://insiderllm.com/guides/vision-models-locally/),
[PromptQuorum 2026](https://www.promptquorum.com/power-local-llm/local-vision-models-llava-ollama-2026).

**Alerta de compatibilidade**: em jul/2026, Ollama carregava o texto de alguns modelos Qwen mais
novos mas não conectava corretamente o arquivo `mmproj` (projetor multimodal) — visão falha
nesses casos via Ollama, exigindo llama.cpp/LM Studio. Testar `ollama pull qwen2.5-vl` (não
`qwen3-vl`) e confirmar que a via de imagem funciona antes de comprometer arquitetura.
[ContraCollective](https://contracollective.com/blog/local-vision-llm-apple-silicon-mlx-qwen-vl-moondream-2026).

## 3. Integração com LangChain

LangChain **não tem** vectorstore multimodal de produção maduro. `langchain_experimental.
open_clip.OpenCLIPEmbeddings` + `Chroma.add_images(uris=...)` existe mas está no pacote
**experimental** — API menos estável, relevante porque o Harbor acabou de migrar para
`langchain-chroma` de produção.
[Chroma docs — OpenCLIP](https://docs.trychroma.com/integrations/embedding-models/open-clip).

Caption-then-embed em LangChain é um passo de pré-processamento: VLM gera descrição textual →
vira `Document` de texto comum → `MultiVectorRetriever` mantém o vetor da legenda (para busca)
e o docstore com a imagem original (bytes, para devolver ao LLM gerador na resposta). Não exige
embedding multimodal dedicado — "usa embeddings de texto comuns para indexar resumos de imagem
como qualquer outro texto".
[LangChain Benchmarks — multi-modal eval](https://langchain-ai.github.io/langchain-benchmarks/notebooks/retrieval/multi_modal_benchmarking/multi_modal_eval.html).

## 4. Diagramas técnicos/PDFs mistos

**ColPali** (ColQwen2.5, ColSmolVLM, ColInternVL): screenshot de cada página, codifica como
grade de patches via VLM, busca via late interaction (mecanismo ColBERT). Evita parsing de
layout tradicional. Adoção crescente em Qdrant/Vespa como feature de produção.
[arXiv 2407.01449](https://arxiv.org/abs/2407.01449), [Vespa blog](https://blog.vespa.ai/retrieval-with-vision-language-models-colpali/).

**Diagramas de engenharia especificamente (P&ID)**: paper ChatP&ID (2026) converte P&IDs no
padrão DEXPI em **grafos de conhecimento estruturados**, faz GraphRAG sobre o grafo — não sobre
a imagem crua. Resultado: +18% acurácia, -85% custo de tokens vs. ingestão direta de imagem;
variante ContextRAG atinge 91% de acurácia a US$0,004/consulta.
[arXiv 2603.22528](https://arxiv.org/abs/2603.22528), [AIChE Journal](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.70540).

**Aplicação ao Harbor**: quando há estrutura formal conhecida (DEXPI), extrair estrutura supera
tratar a imagem como pixel bruto. O Harbor não terá diagramas DEXPI (dados sintéticos, sem
padrão formal) — essa via fica de roadmap, não ponto de partida. ColPali é a alternativa
realista quando não há estrutura formal disponível.

## 5. Geração de dados sintéticos de imagem técnica

Não existe dataset público padrão-ouro de "diagramas de manutenção industrial". Trabalhos
relacionados usam diffusion para gerar imagem industrial com anotação pixel-level
([arXiv 2505.03623](https://arxiv.org/pdf/2505.03623)) ou VLM para completar categorias de
defeito sub-representadas ([arXiv 2605.26533](https://arxiv.org/pdf/2605.26533)) — mais focados
em treinar detecção do que em prototipar RAG.

**Caminho mais barato e controlável para o Harbor**: gráficos gerados **programaticamente**
(matplotlib/plotly) a partir das séries numéricas que os pipelines já produzem
(`outputs/pipelineN_*/`) — não geração de imagem via IA generativa (que introduz ruído/
aleatoriedade desnecessária para um teste controlado, e cujo conteúdo exato não se sabe de
antemão para validar o captioning depois).

## 6. Armadilhas confirmadas na literatura

1. **CLIP não lê texto embutido em imagem de forma confiável** — "CLIP não codifica
   precisamente a informação de ortografia do texto renderizado", problemático para diagrama
   com rótulos. [arXiv 2411.05195](https://arxiv.org/html/2411.05195v1).
2. **"Modality gap" estrutural** — embeddings de texto e imagem no CLIP ocupam regiões
   amplamente disjuntas do espaço vetorial; alinhamento fraco para descrição básica, atributos,
   relação espacial e negação simultaneamente. [arXiv 2511.04247](https://arxiv.org/pdf/2511.04247).
3. **Attribute binding falha** — CLIP não amarra bem atributos a objetos específicos; risco real
   de recuperar a imagem errada por similaridade genérica quando há múltiplos componentes
   rotulados (ex.: "sensor de temperatura no motor B" vs "sensor de pressão no motor A").
4. **Captioning perde detalhe técnico preciso** — confirmado indiretamente pelo ChatP&ID:
   ingestão direta de imagem (mesmo VLM de ponta comercial) perde 18% de acurácia frente a
   representação estruturada — legenda solta provavelmente perde ainda mais nuance numérica/
   posicional que uma estrutura de grafo.
5. **Custo de VLM local em lote** — nenhuma fonte quantifica isso especificamente para Ollama,
   mas captioning em lote é ordens de magnitude mais lento que embeddings CLIP puros (forward
   pass simples vs. geração autoregressiva de texto) — medir tempo de indexação antes de
   escalar além do smoke test.

## Recomendação de caminho (não decide implementação em detalhe, aponta direção)

**Caption-then-embed com Qwen2.5-VL via Ollama** é a via mais realista: menor atrito
arquitetural (sem trocar vectorstore, sem sair de `langchain-chroma` estável), maior
maturidade em produção, resultado mensurável com o harness que já existe. CLIP/ColPali ficam
como via secundária de roadmap — o primeiro porque é documentadamente fraco no tipo de
conteúdo-alvo (texto/rótulo em diagrama), o segundo porque é mais pesado/menos maduro em
LangChain hoje, mais adequado se o volume de diagramas crescer o suficiente para a legenda
perder informação demais na prática (mensurável, não assumido).
