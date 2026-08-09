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

REESCRITO (2026-08-07): antes reimplementava _embed_passages/_construir_tfidf/
_buscar_tfidf/buscar() em paralelo a rag/rag_hibrido.py::RAGHibrido -- mesma logica,
codigo duplicado, risco de divergencia silenciosa (mesmo problema ja documentado para o
roteamento em terceira_copia_roteamento_harness). RAGHibrido agora aceita chroma_dir/
colecao parametrizados (indice separado da producao) e documentos_customizados (corpus do
benchmark, sem chunking) -- este script chama a classe real, sem reimplementar nada.
Tambem nunca persistia resultado em CSV -- so imprimia no stdout; agora salva
eval/resultados_retrieval_nanobeir_<subset>.csv, igual ao padrao de avaliar_retrieval.py.

IMPORTANTE: o indice deste benchmark fica em rag/chroma_db_benchmark/, SEPARADO do
indice de producao (rag/chroma_db/) -- nunca misturar os dois.

Uso: python eval/avaliar_retrieval_nanobeir.py [nome_do_subset] [--denso] [--sem-rerank]
     (default: NanoSciFact -- corpus pequeno de resumos cientificos, ~3k docs, 50 queries)
     --denso desativa a metade TF-IDF do hybrid (default: hybrid completo, igual producao)
     --sem-rerank mede o retrieval isolado, comparavel diretamente contra
     eval/avaliar_retrieval.py (que tambem usa usar_rerank=False por default)
"""
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Projetos\Harbor\rag")
from rag_hibrido import RAGHibrido  # noqa: E402

EVAL_DIR = Path(__file__).parent
CHROMA_DIR_BENCHMARK = Path(r"C:\Projetos\Harbor\rag\chroma_db_benchmark")
K = 5


def carregar_subset_nanobeir(nome_subset):
    from datasets import load_dataset
    queries = load_dataset("sentence-transformers/NanoBEIR-en", "queries", split=nome_subset)
    corpus = load_dataset("sentence-transformers/NanoBEIR-en", "corpus", split=nome_subset)
    qrels = load_dataset("sentence-transformers/NanoBEIR-en", "qrels", split=nome_subset)
    return queries, corpus, qrels


def avaliar_subset(nome_subset=None, usar_hybrid=True, usar_rerank=True):
    nome_subset = nome_subset or "NanoSciFact"
    modo = ("hybrid E5+TF-IDF" if usar_hybrid else "denso E5 apenas") + \
           ("+rerank" if usar_rerank else " sem rerank (retrieval isolado)")
    print(f"Carregando subset NanoBEIR: {nome_subset} -- modo: {modo}...")
    queries, corpus, qrels = carregar_subset_nanobeir(nome_subset)
    print(f"  {len(queries)} queries, {len(corpus)} documentos, {len(qrels)} qrels (pares relevantes)")

    # Mapa query_id -> set de corpus_id relevantes (podem existir varios documentos
    # relevantes por query nos datasets BEIR, diferente do golden set proprio do Harbor
    # que tem so 1 documento relevante por pergunta).
    relevantes_por_query = {}
    for row in qrels:
        relevantes_por_query.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))

    documentos_customizados = [
        {"id": str(row["_id"]), "texto": row["text"], "fonte": str(row["_id"])}
        for row in corpus
    ]

    print(f"\nIndexando corpus no ChromaDB de benchmark (separado da producao) via RAGHibrido real...")
    rag = RAGHibrido(chroma_dir=CHROMA_DIR_BENCHMARK, colecao=f"nanobeir_{nome_subset.lower()}")
    n = rag.indexar(forcar=True, documentos_customizados=documentos_customizados)
    print(f"  {n} documentos indexados.")

    print(f"\nAvaliando retrieval (k={K})...\n")
    resultados = []
    for row in queries:
        query_id = str(row["_id"])
        alvo = relevantes_por_query.get(query_id)
        if not alvo:
            continue  # query sem qrel associado no subset -- pula

        candidatos = rag.buscar(row["text"], k=K, usar_rerank=usar_rerank, usar_hybrid=usar_hybrid, k_candidatos=10)
        recuperados = [c["id"] for c in candidatos]

        acertou = any(doc_id in alvo for doc_id in recuperados)
        recall = 1.0 if acertou else 0.0

        rr = 0.0
        for i, doc_id in enumerate(recuperados, start=1):
            if doc_id in alvo:
                rr = 1.0 / i
                break

        resultados.append({"query_id": query_id, "recall_at_k": recall, "reciprocal_rank": round(rr, 3),
                            "ids_recuperados": ";".join(recuperados)})

    recall_medio = statistics.mean(r["recall_at_k"] for r in resultados) if resultados else 0.0
    mrr_medio = statistics.mean(r["reciprocal_rank"] for r in resultados) if resultados else 0.0

    sufixo_modo = "" if (usar_hybrid and usar_rerank) else \
        ("_denso" if not usar_hybrid else "") + ("_sem_rerank" if not usar_rerank else "")
    resultados_path = EVAL_DIR / f"resultados_retrieval_nanobeir_{nome_subset.lower()}{sufixo_modo}.csv"
    with resultados_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "recall_at_k", "reciprocal_rank", "ids_recuperados"])
        for r in resultados:
            w.writerow([r["query_id"], r["recall_at_k"], r["reciprocal_rank"], r["ids_recuperados"]])

    print("=" * 60)
    print(f"NanoBEIR subset     : {nome_subset} ({modo})")
    print(f"Queries avaliadas   : {len(resultados)}")
    print(f"Recall@{K} medio     : {recall_medio*100:.1f}%")
    print(f"MRR                 : {mrr_medio:.3f}")
    print(f"Resultados salvos   : {resultados_path}")
    print("=" * 60)
    print(
        "\nNota: comparacao INFORMAL com o MTEB leaderboard (huggingface.co/spaces/mteb/"
        "leaderboard) para o mesmo subset -- hardware/config diferem do leaderboard oficial, "
        "isto NAO e um resultado oficial de submissao, e uma checagem de faixa esperada."
    )
    return {"subset": nome_subset, "usar_hybrid": usar_hybrid, "usar_rerank": usar_rerank,
            "n_queries": len(resultados), "recall_at_k": recall_medio, "mrr": mrr_medio}


if __name__ == "__main__":
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
