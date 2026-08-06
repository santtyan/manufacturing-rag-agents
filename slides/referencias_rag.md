# Referências — Pipeline RAG do Harbor

Bibliografia e material de apoio consultado no desenvolvimento e validação do pipeline RAG
do projeto Harbor (sessão de trabalho 2026-07-11). Organizado por categoria, na ordem em
que embasaram decisões técnicas concretas do sistema.

---

## Papers acadêmicos

### 1. Lewis et al. (2020) — o paper que cunhou "RAG"
**Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe
Kiela — Facebook AI Research / UCL / NYU
NeurIPS 2020 · arXiv:2005.11401

**O que embasa no Harbor:**
- Justificativa acadêmica original (2020) para **hybrid search**: ablação BM25 vs. dense
  retrieval mostrou que nenhum método vence sempre — depende do tipo de conteúdo
  (entity-centric favorece BM25, texto semântico favorece dense). Confirma a decisão de
  combinar E5 (denso) + TF-IDF (esparso) em `rag/rag_hibrido.py`.
- Confirma que **RAG reduz alucinação, mas não a elimina** — já reconhecido em 2020, com
  exemplos de erro documentados. Justifica a necessidade da verificação pós-geração
  (`eval/verificacao.py`), não como opcional.
- Propriedade de **índice "hot-swappable"**: trocar o índice (Wikipedia 2016→2018)
  atualiza o conhecimento do modelo sem re-treinar. Mesma propriedade usada ao atualizar
  os manuais do Harbor com dados do Zenodo/Kaggle e reindexar, sem tocar no LLM.
- Distinção entre **RAG "canônico"** (retriever+gerador treinados fim-a-fim, documento
  como variável latente marginalizada) e o **RAG "moderno"** que o Harbor usa (componentes
  fixos, conectados só via prompt em inferência) — simplificação universal da indústria
  com LLMs prontos, não uma limitação específica do projeto.

Citado no diagrama publicado: `slides/imagens/pipeline_rag_harbor.html`

---

### 2. Trivedi et al. (2023) — IRCoT, retrieval multi-hop
**Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive
Multi-Step Questions**
Harsh Trivedi, Niranjan Balasubramanian (Stony Brook University), Tushar Khot, Ashish
Sabharwal (Allen Institute for AI)
ACL 2023 · arXiv:2212.10509

**O que embasa no Harbor:**
- Prova experimental (até 50% menos erros factuais) de que retrieval de UM passo é
  insuficiente para perguntas multi-hop (que exigem encadear fatos entre buscas).
- **Distinção importante** documentada na memória do projeto: o caso investigado nesta
  sessão (`discrete-mfg-status-codes`, chunk específico perdendo no rerank) é um problema
  de qualidade de chunking DENTRO de um documento — IRCoT não resolveria isso. O que IRCoT
  resolveria é diferente: perguntas que exigem encadear SQL→RAG (ex: "procedimento de
  manutenção da máquina com mais paradas por FAILURE?"), que ainda não existem no golden
  set do Harbor.
- Reforça o item **"Agentic RAG"** do roadmap com evidência acadêmica forte, não só
  analogia de framework — o "one-step retrieve-and-read" que o paper chama de insuficiente
  é exatamente a arquitetura de roteamento fixo do Harbor hoje.

---

## Benchmark acadêmico padrão-ouro (dado externo, não domínio do Harbor)

### 2.5. Thakur et al. (2021) — BEIR, benchmark padrão-ouro de retrieval
**BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval
Models**
Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, Iryna Gurevych
NeurIPS 2021 (Datasets and Benchmarks Track)

**O que embasa no Harbor:** o golden set próprio (`eval/golden_questions.json`) não é
comparável externamente — nenhum benchmark público existe para "perguntas sobre
manutenção industrial + datasets do Harbor" (ver `distincao_benchmark_vs_harness_proprio.md`).
Para obter um número comparável com a literatura, o MESMO motor de retrieval do Harbor
(embeddings E5 + Cross-Encoder rerank, sem a metade TF-IDF do hybrid) foi rodado contra
o subset **NanoSciFact** do NanoBEIR (versão "nano" oficial do BEIR, feita pela
Sentence-Transformers para avaliação rápida — `sentence-transformers/NanoBEIR-en` no
HuggingFace Hub) via `eval/avaliar_retrieval_nanobeir.py`.

**Resultados obtidos (2026-07-11), 4 subsets, pipeline hybrid completo (E5+TF-IDF+rerank):**

| Subset | Documentos | Queries | Recall@5 | MRR |
|---|---|---|---|---|
| NanoSciFact (só denso, primeira execução) | 2.919 | 50 | 76,0% | 0,663 |
| NanoFEVER | 4.996 | 50 | 98,0% | 0,890 |
| NanoHotpotQA | 5.090 | 50 | 98,0% | 0,955 |
| NanoDBPedia | 6.045 | 50 | 94,0% | 0,879 |

Número na faixa esperada para um pipeline hybrid+rerank com embedding multilingual
genérico (E5), sem fine-tuning específico a nenhum dos domínios testados — comparação
informal com o MTEB leaderboard (huggingface.co/spaces/mteb/leaderboard), não um
resultado oficial de submissão (hardware/config diferem). Contraste importante para os
slides: Recall@5=100% no golden set próprio (corpus pequeno, perguntas escritas
conhecendo o conteúdo) vs. 76-98% no benchmark público (corpus maior, mais distratores,
domínio genérico) — a diferença é esperada e mostra o motor de retrieval funcionando bem
mesmo fora do domínio para o qual foi ajustado (chunking por seção, hybrid search).

**Nota sobre NanoSciFact vs. os demais**: o número de NanoSciFact (76%) foi medido só com
a metade densa (E5+rerank, sem TF-IDF), antes do script ser estendido para o hybrid
completo — não é diretamente comparável aos outros 3 subsets, que já rodaram com
TF-IDF+E5+rerank juntos. O salto para 94-98% nos subsets seguintes reflete tanto o hybrid
completo quanto a natureza dos datasets (FEVER/HotpotQA/DBPedia têm mais overlap lexical
exato entre pergunta e documento relevante, o que favorece a metade TF-IDF do hybrid,
diferente de SciFact que é mais parafrástico/científico).

**Nota sobre NanoHotpotQA (multi-hop)**: 98%/0,955 é surpreendentemente alto para um
dataset desenhado para exigir raciocínio multi-hop (ver Trivedi et al. 2023, seção 2,
acima) com retrieval de um único passo. Explicação provável: a versão "nano" (50 queries)
é uma amostra pequena e pode não preservar a dificuldade multi-hop proporcionalmente à
versão completa do HotpotQA — não deve ser lido como "retrieval de um passo resolve
multi-hop", e sim como uma característica do subset reduzido. Não investigado a fundo;
registrado aqui como ressalva para não superinterpretar o número nos slides.

Índice do benchmark fica isolado em `rag/chroma_db_benchmark/`, nunca misturado com o
índice de produção (`rag/chroma_db/`).

---

## Documentação oficial de frameworks/provedores

### 3. OpenAI Agents SDK — Tools
Doc oficial de Custom Tools do Claude Platform (Managed Agents API), Anthropic, 2026.

**O que embasa:** confirmou que `mcp/servidor_harbor.py` já segue as boas práticas
recomendadas (descrições ricas de 3-4 frases, namespacing por domínio nos nomes das
tools) sem precisar de mudança.

### 4. LangChain — RAG with Deep Agents
Doc oficial "RAG patterns for Deep Agents" (padrão "retrieve, offload, delegate"),
LangChain, 2026.

**O que embasa:**
- Confirma por que **Agentic RAG/Deep Agents não são prioridade agora**: o padrão
  "offload+delegate" existe porque corpora grandes (782 chunks, >100k tokens) não cabem
  no contexto do orquestrador. O Harbor tem 38 chunks — cabe inteiro sem problema.
- Confirma que a defesa de prompt injection do Harbor (`rag_gerador.py::montar_prompt_rag`)
  já é prática padrão, mas com limitação reconhecida — a doc diz textualmente que nenhuma
  estratégia de prompt/delimitador previne totalmente prompt injection indireto.
  Comentário de limitação adicionado ao código citando essa fonte.
- `RubricMiddleware` (grader sub-agent que REVISA a resposta até passar) é mais maduro que
  a verificação atual do Harbor (que só anexa aviso ⚠️ pós-hoc) — mesma ideia do item
  "Guardrails em paralelo, fail-fast" do roadmap.

### 5. LangChain — overview / create_agent
Doc oficial "LangChain overview", LangChain, 2026.

**O que embasa:** terminologia "Agent = Model + Harness" aplicada à arquitetura do Harbor
(modelo=Ollama; harness=roteamento+tools MCP+prompts) — vocabulário útil para descrever o
sistema na apresentação. Ver memória dedicada `vocabulario_agent_harness.md`.

---

## Blog posts técnicos (chunking e RAG)

### 6. Pinecone — Chunking Strategies for LLM Applications
Roie Schwaber-Cohen, Arjun Patel — Pinecone, jun/2025

**O que embasa:**
- Confirma, com fonte de peso, que "content-aware chunking" (dividir por estrutura do
  documento, não por caractere cru) é prática recomendada — valida a reescrita de
  `chunk_texto()` em `rag/rag_hibrido.py`.
- Levou à correção aplicada no `readme_discrete_manufacturing.md`: um bloco que misturava
  3 schemas CSV diferentes foi dividido em 3 sub-seções, resolvendo o roteamento para o
  documento certo (mas não 100% o chunk específico vencer no rerank — limitação
  documentada).
- Menciona "contextual chunking" (técnica da Anthropic: LLM gera descrição por chunk antes
  de embedar) como solução mais robusta para ambiguidade entre chunks parecidos — registrada
  como opção futura pontual, não implementada (custo de 1 chamada LLM por chunk na indexação).

### 7. NVIDIA — Finding the Best Chunking Strategy for Accurate AI Responses
Steve Han — NVIDIA Developer Blog, jun/2025

**O que embasa — ressalva importante, não confirmação:**
- Estudo empírico (5 datasets, PDFs longos) achou que **page-level chunking venceu
  section-level chunking** na maioria dos casos — o OPOSTO da intuição que guiou a decisão
  do Harbor.
- Usado para adicionar uma ressalva honesta ao código (`rag_hibrido.py::chunk_texto()`,
  docstring) e à memória: chunking por seção é a escolha certa **para este corpus
  específico** (documentos `.md` curtos, sem páginas físicas, validado empiricamente com
  Recall@5=100%/MRR=1.0) — não uma "melhor prática universal". Também confirma que mesmo
  dentro da mesma categoria de documento, a estratégia ótima varia entre datasets — não
  existe chunking universal, só chunking testado no próprio dado.

---

## Datasets — páginas de origem consultadas

Consultadas para extrair fatos de proveniência/metadados que faltavam nos manuais do RAG
(`rag/manuais/*.md`), incorporados via reindexação:

### 8. Zenodo — OEE and Downtime datasets in Heavy Clay production line
Panos Ntoas (University of Cyprus), DOI: 10.5281/zenodo.17855209, dez/2025

**Fato incorporado:** nota do operador (`ExtraText`) mantida em grego (idioma original, não
traduzida, para evitar erro de tradução) — não estava documentado no README anterior.
Também: DOI, período de coleta (2025-09-22 a 2025-12-18), supervisor.

### 9. Kaggle — Legacy Industrial Equipment Sensor Logs
Autor: Colabsss · Licença CC0 (domínio público)

**Fato incorporado:** 2.500 registros, 17 colunas, coletados ao longo de 2024.

### 10. GitHub — HumanCenteredTechnology/SME-Manufacturing-Dataset
Repositório do dataset de manufatura discreta (discrete manufacturing), autor principal
Daniele Atzeni, citação acadêmica: Atzeni et al. (2023), *Sensors* 23(13):6078, MDPI.

**Fato incorporado:** proveniência do repositório (organização mantenedora, autor) — não
estava no README anterior, só implícito no BibTeX.

---

## Como usar este documento

- Cada seção liga a fonte externa a uma decisão/correção **concreta e verificável** no
  código do Harbor — não é lista de leitura genérica.
- Para a apresentação: seções 1 e 2 (papers acadêmicos) dão embasamento teórico forte;
  seções 6-7 mostram o processo real de iteração empírica (inclusive um caso onde uma
  fonte contradisse a intuição inicial, tratado com honestidade em vez de descartado).
- Detalhes de implementação de cada correção estão nas memórias do projeto:
  `rag_chunking_e_roadmap_2026-07-11.md`, `harness_rag_extensao_2026-07-11.md`,
  `mcp_tools_design_validado.md`, `vocabulario_agent_harness.md`.
