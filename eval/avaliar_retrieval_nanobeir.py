"""
Avaliacao do motor de retrieval do Harbor (E5 hybrid + rerank) contra um subset do
NanoBEIR -- versao "nano" oficial do benchmark academico padrao-ouro BEIR (Thakur et al.
2021, NeurIPS), feita pela Sentence-Transformers para avaliacao rapida sem baixar o
corpus completo. Fonte: sentence-transformers/NanoBEIR-en no HuggingFace Hub.

DIFERENCA em relacao a eval/avaliar_retrieval.py: aquele mede o retrieval contra o
GOLDEN SET PROPRIO do Harbor (perguntas escritas a mao sobre os manuais do dominio
industrial) -- nao e comparavel externamente, porque nao existe benchmark publico para
"perguntas sobre manutencao industrial + datasets Kaggle/Zenodo do Harbor" (ver memoria
distincao_benchmark_vs_harness_proprio.md). Este script mede o MESMO motor de retrieval
(mesma classe de embeddings, mesmos prefixos E5, mesmo hybrid search) mas contra um
corpus academico publico e generico (ex: SciFact, artigos cientificos) -- da um numero
comparavel com o MTEB leaderboard (BM25 e modelos densos), validando a qualidade GERAL
do motor de retrieval, nao a qualidade das respostas sobre o dominio do Harbor.

IMPORTANTE: o indice deste benchmark fica em rag/chroma_db_benchmark/, SEPARADO do
indice de producao (rag/chroma_db/) -- nunca misturar os dois.

Uso: python eval/avaliar_retrieval_nanobeir.py [nome_do_subset]
     (default: NanoSciFact -- corpus pequeno de resumos cientificos, ~3k docs, 50 queries)
"""
import statistics
import sys
from pathlib import Path

CHROMA_DIR_BENCHMARK = Path(r"C:\Projetos\Harbor\rag\chroma_db_benchmark")
MODELO_EMBEDDING = "intfloat/multilingual-e5-small"
MODELO_RERANK = "cross-encoder/ms-marco-MiniLM-L-6-v2"
K = 5


def carregar_subset_nanobeir(nome_subset):
    from datasets import load_dataset
    queries = load_dataset("sentence-transformers/NanoBEIR-en", "queries", split=nome_subset)
    corpus = load_dataset("sentence-transformers/NanoBEIR-en", "corpus", split=nome_subset)
    qrels = load_dataset("sentence-transformers/NanoBEIR-en", "qrels", split=nome_subset)
    return queries, corpus, qrels


def indexar_corpus_benchmark(corpus, nome_subset, forcar=False):
    """Indexa o corpus do NanoBEIR num ChromaDB SEPARADO do indice de producao, usando o
    mesmo modelo/prefixos E5 do motor de retrieval real (rag/rag_hibrido.py), mas sem
    reaproveitar a classe RAGHibrido diretamente -- ela esta acoplada a MANUAIS_DIR
    (le .md do disco), o que nao serve para um corpus vindo de um dataset HuggingFace."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    CHROMA_DIR_BENCHMARK.mkdir(parents=True, exist_ok=True)
    cliente = chromadb.PersistentClient(path=str(CHROMA_DIR_BENCHMARK))
    colecao_nome = f"nanobeir_{nome_subset.lower()}"

    nomes_existentes = [c.name for c in cliente.list_collections()]
    if colecao_nome in nomes_existentes and not forcar:
        return cliente.get_collection(colecao_nome)
    if colecao_nome in nomes_existentes:
        cliente.delete_collection(colecao_nome)

    colecao = cliente.create_collection(colecao_nome, metadata={"hnsw:space": "cosine"})
    modelo = SentenceTransformer(MODELO_EMBEDDING)

    ids = [str(row["_id"]) for row in corpus]
    textos = [row["text"] for row in corpus]
    # Prefixo E5 obrigatorio para documentos -- mesmo padrao de rag_hibrido.py::_embed_passages.
    embeddings = modelo.encode([f"passage: {t}" for t in textos], normalize_embeddings=True).tolist()

    # ChromaDB aceita lotes de ate 5461 itens por chamada (limite do backend) -- corpus dos
    # subsets maiores do NanoBEIR pode passar disso, entao adiciona em lotes.
    LOTE = 5000
    for i in range(0, len(ids), LOTE):
        colecao.add(
            ids=ids[i:i + LOTE],
            documents=textos[i:i + LOTE],
            embeddings=embeddings[i:i + LOTE],
        )
    return colecao


def construir_tfidf(corpus):
    """Indice lexical (TF-IDF) sobre o MESMO corpus do benchmark -- mesma tecnica de
    rag_hibrido.py::_construir_tfidf, reproduzida aqui porque aquela funcao esta acoplada
    ao estado interno de RAGHibrido (self._tfidf), nao reaproveitavel diretamente."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    ids = [str(row["_id"]) for row in corpus]
    textos = [row["text"] for row in corpus]
    vectorizer = TfidfVectorizer(stop_words=None, max_features=2000)
    matriz = vectorizer.fit_transform(textos)
    return vectorizer, matriz, textos, ids


def buscar_tfidf(tfidf_state, pergunta, k):
    """Metade lexical do hybrid search -- mesma logica de rag_hibrido.py::_buscar_tfidf."""
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    vectorizer, matriz, textos, ids = tfidf_state
    vetor_pergunta = vectorizer.transform([pergunta])
    scores = cosine_similarity(vetor_pergunta, matriz)[0]
    top_idx = np.argsort(scores)[::-1][:k]
    return [ids[i] for i in top_idx if scores[i] > 0]


def buscar(colecao, modelo, reranker, pergunta, k=K, k_candidatos=10, tfidf_state=None, usar_rerank=True):
    """Retrieval denso (E5) [+ lexical TF-IDF se tfidf_state for passado] [+ rerank
    Cross-Encoder se usar_rerank] -- replica RAGHibrido.buscar() (rag/rag_hibrido.py) com
    hybrid=True quando tfidf_state e fornecido, medindo o pipeline de producao completo em
    vez de so a metade densa. usar_rerank=False mede o retrieval isolado, para comparar
    contra o mesmo modo do harness proprio (eval/avaliar_retrieval.py)."""
    emb_query = modelo.encode([f"query: {pergunta}"], normalize_embeddings=True).tolist()
    res = colecao.query(query_embeddings=emb_query, n_results=k_candidatos)
    docs = res["documents"][0]
    ids = res["ids"][0]

    candidatos_id = list(ids)
    candidatos_texto = list(docs)
    candidatos_score = [1.0 - d for d in res["distances"][0]]

    if tfidf_state is not None:
        vistos = set(candidatos_id)
        for doc_id in buscar_tfidf(tfidf_state, pergunta, k=k_candidatos):
            if doc_id not in vistos:
                idx = tfidf_state[3].index(doc_id)
                candidatos_id.append(doc_id)
                candidatos_texto.append(tfidf_state[2][idx])
                candidatos_score.append(0.0)  # score TF-IDF nao comparavel ao score E5; so usado sem rerank
                vistos.add(doc_id)

    if not usar_rerank:
        ranking = list(zip(candidatos_id, candidatos_score))[:k]
        return [doc_id for doc_id, _ in ranking]

    pares = [[pergunta, t] for t in candidatos_texto]
    scores = reranker.predict(pares)
    ranking = sorted(zip(candidatos_id, scores), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ranking[:k]]


def avaliar_subset(nome_subset=None, usar_hybrid=True, usar_rerank=True):
    nome_subset = nome_subset or "NanoSciFact"
    partes_modo = []
    partes_modo.append("hybrid E5+TF-IDF" if usar_hybrid else "denso E5 apenas")
    partes_modo.append("+rerank" if usar_rerank else " sem rerank (retrieval isolado)")
    modo = "".join(partes_modo)
    print(f"Carregando subset NanoBEIR: {nome_subset} -- modo: {modo}...")
    queries, corpus, qrels = carregar_subset_nanobeir(nome_subset)
    print(f"  {len(queries)} queries, {len(corpus)} documentos, {len(qrels)} qrels (pares relevantes)")

    # Mapa query_id -> set de corpus_id relevantes (podem existir varios documentos
    # relevantes por query nos datasets BEIR, diferente do golden set proprio do Harbor
    # que tem so 1 documento relevante por pergunta).
    relevantes_por_query = {}
    for row in qrels:
        relevantes_por_query.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))

    print(f"\nIndexando corpus no ChromaDB de benchmark (separado da producao)...")
    colecao = indexar_corpus_benchmark(corpus, nome_subset)

    tfidf_state = None
    if usar_hybrid:
        print("Construindo indice TF-IDF (metade lexical do hybrid)...")
        tfidf_state = construir_tfidf(corpus)

    from sentence_transformers import SentenceTransformer, CrossEncoder
    modelo = SentenceTransformer(MODELO_EMBEDDING)
    reranker = CrossEncoder(MODELO_RERANK)

    print(f"\nAvaliando retrieval (k={K})...\n")
    recalls, mrrs = [], []
    for row in queries:
        query_id = str(row["_id"])
        alvo = relevantes_por_query.get(query_id)
        if not alvo:
            continue  # query sem qrel associado no subset -- pula

        recuperados = buscar(colecao, modelo, reranker, row["text"], k=K, tfidf_state=tfidf_state, usar_rerank=usar_rerank)
        acertou = any(doc_id in alvo for doc_id in recuperados)
        recalls.append(1.0 if acertou else 0.0)

        rr = 0.0
        for i, doc_id in enumerate(recuperados, start=1):
            if doc_id in alvo:
                rr = 1.0 / i
                break
        mrrs.append(rr)

    recall_medio = statistics.mean(recalls) if recalls else 0.0
    mrr_medio = statistics.mean(mrrs) if mrrs else 0.0

    print("=" * 60)
    print(f"NanoBEIR subset     : {nome_subset} ({modo})")
    print(f"Queries avaliadas   : {len(recalls)}")
    print(f"Recall@{K} medio     : {recall_medio*100:.1f}%")
    print(f"MRR                 : {mrr_medio:.3f}")
    print("=" * 60)
    print(
        "\nNota: comparacao INFORMAL com o MTEB leaderboard (huggingface.co/spaces/mteb/"
        "leaderboard) para o mesmo subset -- hardware/config diferem do leaderboard oficial, "
        "isto NAO e um resultado oficial de submissao, e uma checagem de faixa esperada."
    )
    return {"subset": nome_subset, "usar_hybrid": usar_hybrid, "usar_rerank": usar_rerank,
            "n_queries": len(recalls), "recall_at_k": recall_medio, "mrr": mrr_medio}


if __name__ == "__main__":
    # Uso: python avaliar_retrieval_nanobeir.py [subset1,subset2,...] [--denso] [--sem-rerank]
    # --denso desativa a metade TF-IDF do hybrid (default: hybrid completo, igual producao)
    # --sem-rerank mede o retrieval isolado (mesmo modo default de eval/avaliar_retrieval.py,
    # comparavel diretamente contra o harness proprio do Harbor)
    args = sys.argv[1:]
    usar_hybrid = "--denso" not in args
    usar_rerank = "--sem-rerank" not in args
    args = [a for a in args if a not in ("--denso", "--sem-rerank")]
    subsets = args[0].split(",") if args else [None]

    resultados = [avaliar_subset(s, usar_hybrid=usar_hybrid, usar_rerank=usar_rerank) for s in subsets]

    if len(resultados) > 1:
        print("\n" + "#" * 60)
        print("RESUMO COMPARATIVO")
        print("#" * 60)
        for r in resultados:
            print(f"{r['subset']:20} Recall@{K}={r['recall_at_k']*100:5.1f}%  MRR={r['mrr']:.3f}  "
                  f"({r['n_queries']} queries)")
