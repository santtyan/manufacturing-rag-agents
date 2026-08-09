# Experimento: RAG artesanal vs. LangChain

Experimento paralelo, isolado do pipeline de produção (`rag/rag_hibrido.py`), comparando a
implementação artesanal do Harbor contra duas versões equivalentes construídas com
componentes LangChain. Não afeta `dashboard/app.py`, `eval/rag_gerador.py` nem nenhum
arquivo fora desta pasta.

## Por que duas versões, não uma

O componente idiomático do LangChain para hybrid search (`EnsembleRetriever` +
`BM25Retriever`) usa BM25 + RRF ponderado — diferente do TF-IDF + união/dedupe manual que a
produção usa hoje. Comparar a produção direto contra essa versão misturaria duas variáveis
(framework diferente **e** algoritmo diferente), inviabilizando a conclusão sobre "o que o
LangChain trouxe". Por isso:

- **Versão A** (`rag_langchain_fiel.py`) replica os **mesmos algoritmos** da produção
  (E5 + ChromaDB, TF-IDF + união/dedupe, Cross-Encoder), só trocando a implementação
  artesanal por componentes LangChain onde eles existem. Isola o efeito do framework.
- **Versão B** (`rag_langchain_bm25rrf.py`) usa o padrão **idiomático** do LangChain
  (BM25Retriever + EnsembleRetriever/RRF), mostrando o que o padrão recomendado pela
  própria documentação entrega — mas mistura framework + algoritmo, não deve ser lido
  como "efeito puro do LangChain".

Chunking (`chunk_texto()` de `rag/rag_hibrido.py`, por seção Markdown) e modelos (E5,
`cross-encoder/ms-marco-MiniLM-L-6-v2`) são idênticos nas 3 implementações — só a camada
de busca/fusão/orquestração muda.

## Resultado (2026-08-06, retrieval isolado sem rerank aplicado à métrica final — ver nota)

50 golden questions de rota RAG (`eval/golden_questions.json`), métrica idêntica a
`eval/avaliar_retrieval.py` (Recall@5, Precision@5, MRR). A coluna "Δ vs. produção" isola
o cenário prático "e se a gente trocasse hoje" (produção vs. BM25+RRF, misturando
framework e algoritmo de propósito) — a leitura por variável isolada vem logo abaixo.

| Métrica | Produção (`rag_hibrido.py`) | Versão A — LangChain fiel (TF-IDF + união) | Versão B — LangChain BM25+RRF | Δ vs. produção (B) |
|---|---|---|---|---|
| Recall@5 | 98,0% | 98,0% | 98,0% | igual |
| Precision@5 | 38,8% | 38,8% | **50,0%** | **+11,2pp** |
| MRR | 0,900 | 0,900 | 0,899 | -0,001 (irrelevante) |
| Indexação | 20,0s | 11,1s | 10,8s | ~2x mais rápido |
| Linhas de código | 317 | 170 | 127 | **-60%** |

BM25+RRF recupera a mesma cobertura (Recall@5 idêntico) com bem menos ruído nos top-5
(Precision@5 +11,2pp) — ou seja, os 5 chunks retornados são mais frequentemente todos
relevantes, não só "pelo menos um relevante". MRR estável confirma que a posição do
primeiro acerto não piorou. Como a seção seguinte mostra, esse ganho é atribuível ao
algoritmo (BM25+RRF), não ao framework — replicável em `rag_hibrido.py` sem adotar
LangChain.

## Leitura dos resultados (decomposição por variável)

1. **Efeito do framework, isolado (produção vs. Versão A)**: paridade quase perfeita em
   qualidade de retrieval (mesmas 3 métricas). O framework por si só não melhora nem piora
   o resultado — o ganho é em **linhas de código** (170 vs. 317, -46%), porque LangChain
   já resolve a integração com ChromaDB e o carregamento do embedding com prefixo E5 nativo
   (`HuggingFaceEmbeddings(encode_kwargs={"prompt": ...})`), código que a produção escreve
   à mão.

2. **Efeito do algoritmo (Versão A vs. Versão B)**: trocar TF-IDF+união por BM25+RRF
   ganhou **+11,2 pontos de Precision@5** (38,8% → 50,0%) mantendo Recall@5 e MRR estáveis
   — ou seja, retorna a mesma cobertura com menos ruído nos top-5. Isso é consistente com o
   feedback já registrado do Pedro (trocar TF-IDF por BM25) e com a pesquisa de estado da
   arte que apontou hybrid BM25+RRF como padrão de fato em 2025-2026. Ver
   `estado_arte_rag_agentes_2026-08-06` na memória do projeto.

3. **Tempo de indexação**: as duas versões LangChain indexam ~2x mais rápido que a
   produção (~11s vs. 20s) neste corpus pequeno (9 documentos). Não investigado a fundo —
   possivelmente reuso de sessão/cache do modelo entre chamadas do LangChain — não é o
   achado central deste experimento, citado só por completude.

## Limitações conhecidas

- Corpus pequeno (9 documentos, 50 perguntas) — resultados não devem ser generalizados
  para um corpus muito maior sem novo teste.
- Métrica medida é retrieval isolado (`usar_rerank=False` no cálculo de Recall/Precision,
  igual ao default de `eval/avaliar_retrieval.py`) — o rerank Cross-Encoder roda nas 3
  implementações mas seu efeito na métrica final não foi isolado neste experimento
  (para isso, rodar `python avaliar.py --rerank`).
- `CrossEncoderReranker` da versão instalada do LangChain (`langchain_classic`) descarta o
  score de relevância no retorno — foi necessário chamar `model.score()` diretamente para
  recuperar o score (ver comentário em `rag_langchain_fiel.py::buscar()`). Se o LangChain
  mudar essa API no futuro, o código pode quebrar silenciosamente sem esse cuidado.
- Comparação de linhas de código é aproximada (conta linhas do arquivo inteiro, incluindo
  comentários/docstrings) — não é uma métrica normalizada de complexidade ciclomática.

## Como rodar de novo

```
pip install -r experiments/langchain_rag/requirements.txt
python experiments/langchain_rag/avaliar.py           # retrieval isolado
python experiments/langchain_rag/avaliar.py --rerank  # pipeline completo (retrieval+rerank)
```

## Conclusão para o Harbor

Não recomendamos migrar a produção para LangChain (ver memória do projeto,
`estado_arte_rag_agentes_2026-08-06` e conversa que motivou este experimento) — o ganho de
qualidade observado aqui vem do **algoritmo** (BM25+RRF), não do framework, e esse algoritmo
pode ser adotado diretamente em `rag/rag_hibrido.py` (trocando `TfidfVectorizer` por
`rank_bm25.BM25Okapi` e a união/dedupe por uma fusão RRF) sem precisar da dependência
LangChain inteira. Este experimento serve como evidência de portfólio (comparação
framework vs. artesanal) e como validação de que vale a pena aplicar hybrid BM25+RRF na
produção — que já era o item de maior ROI identificado antes deste experimento.
