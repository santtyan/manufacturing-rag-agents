"""
Avaliacao ISOLADA de retrieval do RAG multimodal (Recall@k, Precision@k, MRR) -- Fase 4 do
plano de disciplina experimental do grupo PDC.

Motivacao: rag/rag_multimodal_langchain.py era so um smoke test (docstring do proprio
arquivo), sem nenhuma metrica -- so 3 perguntas manuais impressas no stdout. A pesquisa de
estado da arte 2026 (skill rag-multimodal) mostra tres arquiteturas competindo (caption-then-
embed, embeddings visuais unificados, late interaction/ColPali) sem vencedor universal: a
escolha certa depende de o recall ser o gargalo e as queries serem visualmente dificeis ou
nao. Sem medir o que ja existe, trocar de arquitetura seria fe, nao medicao -- mesmo
principio ja aplicado na comparacao TF-IDF vs BM25 do RAG de texto (skill
comparar-tfidf-bm25).

Mesmo formato de golden set de 1-documento-relevante de eval/avaliar_retrieval.py (nao o
formato multi-relevante do NanoBEIR/BEIR) -- aqui "documento" e uma imagem (identificada pelo
id/stem do arquivo), nao um chunk de manual .md.

Usa o cache de legendas ja gerado (rag/legendas_cache.json, produzido por
`python rag/rag_multimodal_langchain.py --captionar`) -- este script NAO chama Ollama, so
sentence-transformers/ChromaDB via RAGHibrido, entao pode rodar no mesmo processo sem o risco
de segfault ja documentado (ACHADO REAL #2 em rag_multimodal_langchain.py).

Uso: python eval/avaliar_rag_multimodal.py [--rerank]
     Pre-requisito: python rag/rag_multimodal_langchain.py --captionar (gera o cache)
"""
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Projetos\Harbor\rag")
sys.path.insert(0, str(Path(__file__).parent))

from rag_hibrido_langchain import RAGHibrido

EVAL_DIR = Path(__file__).parent
GOLDEN = EVAL_DIR / "golden_questions_multimodal.json"
CACHE_LEGENDAS = Path(r"C:\Projetos\Harbor\rag\legendas_cache.json")
CHROMA_DIR = Path(r"C:\Projetos\Harbor\rag\chroma_db_multimodal")
COLECAO = "imagens_harbor_multimodal_v1"

K = 3  # corpus tem so 4 imagens -- k=5 do texto nao faz sentido aqui, k=3 ja cobre 75% do corpus


def avaliar_pergunta(rag, pergunta_obj, k=K, usar_rerank=False):
    """Mesma logica de eval/avaliar_retrieval.py::avaliar_pergunta, adaptada para "documento"
    = imagem (campo documento_relevante) em vez de arquivo .md."""
    alvo = pergunta_obj["documento_relevante"]
    candidatos = rag.buscar(pergunta_obj["pergunta"], k=k, usar_rerank=usar_rerank, usar_hybrid=True, k_candidatos=max(k, 4))
    ids = [c.get("id") or c.get("fonte", "").rsplit(".", 1)[0] for c in candidatos]

    relevantes_no_topk = sum(1 for i in ids if i == alvo)
    recall_at_k = 1.0 if relevantes_no_topk > 0 else 0.0
    precision_at_k = relevantes_no_topk / len(ids) if ids else 0.0

    rr = 0.0
    for i, doc_id in enumerate(ids, start=1):
        if doc_id == alvo:
            rr = 1.0 / i
            break

    return {
        "id": pergunta_obj["id"],
        "documento_esperado": alvo,
        "documentos_recuperados": ids,
        "recall_at_k": recall_at_k,
        "precision_at_k": round(precision_at_k, 3),
        "reciprocal_rank": round(rr, 3),
    }


def main():
    if not CACHE_LEGENDAS.exists():
        print(f"Cache de legendas nao encontrado em {CACHE_LEGENDAS}.")
        print("Rode primeiro: python rag/rag_multimodal_langchain.py --captionar")
        return

    usar_rerank = "--rerank" in sys.argv
    sufixo = "_rerank" if usar_rerank else ""
    resultados_path = EVAL_DIR / f"resultados_rag_multimodal{sufixo}.csv"

    docs = json.loads(CACHE_LEGENDAS.read_text(encoding="utf-8"))
    dados = json.loads(GOLDEN.read_text(encoding="utf-8"))
    perguntas = dados["perguntas"]

    modo = "retrieval + rerank (pipeline completo)" if usar_rerank else "retrieval isolado (sem rerank)"
    print(f"Indexando {len(docs)} legendas de imagem (caption-then-embed) e avaliando {modo} "
          f"em {len(perguntas)} perguntas (k={K})...\n")

    rag = RAGHibrido(chroma_dir=CHROMA_DIR, colecao=COLECAO)
    rag.indexar(forcar=True, documentos_customizados=docs)

    resultados = [avaliar_pergunta(rag, pq, k=K, usar_rerank=usar_rerank) for pq in perguntas]
    for r in resultados:
        status = "OK " if r["recall_at_k"] == 1.0 else "MISS"
        print(f"[{r['id']:30}] {status} esperado={r['documento_esperado']:35} "
              f"precision@{K}={r['precision_at_k']:.2f} RR={r['reciprocal_rank']:.2f}")

    with resultados_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "documento_esperado", "recall_at_k", "precision_at_k", "reciprocal_rank", "documentos_recuperados"])
        for r in resultados:
            w.writerow([r["id"], r["documento_esperado"], r["recall_at_k"], r["precision_at_k"],
                        r["reciprocal_rank"], ";".join(r["documentos_recuperados"])])

    recall_medio = statistics.mean(r["recall_at_k"] for r in resultados)
    precision_media = statistics.mean(r["precision_at_k"] for r in resultados)
    mrr = statistics.mean(r["reciprocal_rank"] for r in resultados)

    print("\n" + "=" * 60)
    print(f"Modo               : {modo}")
    print(f"Arquitetura        : caption-then-embed (VLM: ver MODELO_VLM em rag_multimodal_langchain.py)")
    print(f"Perguntas avaliadas: {len(resultados)}")
    print(f"Recall@{K} medio    : {recall_medio*100:.0f}%")
    print(f"Precision@{K} media : {precision_media*100:.0f}%")
    print(f"MRR                : {mrr:.3f}")
    print(f"Resultados salvos  : {resultados_path}")
    print("\nEste numero e o BASELINE do caption-then-embed atual -- ver Fase 4 do plano de "
          "disciplina experimental PDC: a escolha de trocar de arquitetura (ColPali/embeddings "
          "unificados) so acontece depois do dataset real chegar E deste numero existir.")


if __name__ == "__main__":
    main()
