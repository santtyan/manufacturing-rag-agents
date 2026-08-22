"""
Compara os dois algoritmos de busca lexical disponiveis em rag/rag_hibrido.py::RAGHibrido
(TF-IDF, producao atual, vs BM25, alternativa sugerida pelo Pedro -- ver memoria
feedback_pedro_bm25_limiar_alucinacao.md) no MESMO golden set usado por
eval/avaliar_retrieval.py, para decidir se vale trocar o default de producao.

Mede tres cenarios por algoritmo:
  1. Lexical isolado (usar_hybrid=True enganaria a comparacao -- aqui usamos so a metade
     lexical, sem E5, para isolar o efeito puro de TF-IDF vs BM25)
  2. Hybrid sem rerank (E5 + lexical, como eval/avaliar_retrieval.py sem --rerank)
  3. Hybrid com rerank (pipeline completo, como eval/avaliar_retrieval.py --rerank)
  4. Latencia media da busca lexical isolada (ms/query) -- BM25 e TF-IDF tem custo
     computacional diferente, vale registrar junto com a qualidade.

Usa um indice ChromaDB separado por algoritmo (colecao "manuais_harbor_bench_tfidf" /
"manuais_harbor_bench_bm25"), para nao colidir com o indice de producao (rag/chroma_db/)
nem com o de benchmark do NanoBEIR (rag/chroma_db_benchmark/) -- mesmo cuidado de isolamento
documentado em avaliar_retrieval_nanobeir.py.

Uso: python eval/comparar_tfidf_bm25.py
"""
import statistics
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, r"C:\Projetos\Harbor\rag")
sys.path.insert(0, str(Path(__file__).parent))

from rag_hibrido import RAGHibrido  # noqa: E402
from avaliar_retrieval import arquivo_esperado, K  # noqa: E402

EVAL_DIR = Path(__file__).parent
GOLDEN = EVAL_DIR / "golden_questions.json"
CHROMA_DIR_BENCH = Path(r"C:\Projetos\Harbor\rag\chroma_db_benchmark_lexico")


def carregar_perguntas_rag():
    dados = json.loads(GOLDEN.read_text(encoding="utf-8"))
    return [p for p in dados["perguntas"] if p.get("rota_esperada") == "rag"]


def metricas(resultados):
    if not resultados:
        return {"recall": 0.0, "precision": 0.0, "mrr": 0.0}
    return {
        "recall": statistics.mean(r["recall_at_k"] for r in resultados),
        "precision": statistics.mean(r["precision_at_k"] for r in resultados),
        "mrr": statistics.mean(r["reciprocal_rank"] for r in resultados),
    }


def avaliar_lexical_isolado(rag, perguntas, k=K):
    """Mede SO a metade lexical (sem E5), chamando _buscar_lexical diretamente -- isola o
    efeito de TF-IDF vs BM25 sem a busca semantica "carregar" o resultado."""
    resultados = []
    tempos = []
    for pq in perguntas:
        alvo = arquivo_esperado(pq)
        if alvo is None:
            continue
        t0 = time.perf_counter()
        candidatos = rag._buscar_lexical(pq["pergunta"], k=k)
        tempos.append(time.perf_counter() - t0)
        fontes = [c["fonte"] for c in candidatos]
        relevantes = sum(1 for f in fontes if f == alvo)
        rr = 0.0
        for i, f in enumerate(fontes, start=1):
            if f == alvo:
                rr = 1.0 / i
                break
        resultados.append({
            "recall_at_k": 1.0 if relevantes > 0 else 0.0,
            "precision_at_k": relevantes / len(fontes) if fontes else 0.0,
            "reciprocal_rank": rr,
        })
    return resultados, tempos


def avaliar_pipeline(rag, perguntas, k=K, usar_rerank=False):
    """Mede o pipeline hibrido completo (E5+lexical[, +rerank]), igual
    eval/avaliar_retrieval.py::avaliar_pergunta -- reaproveitado via import, sem duplicar."""
    from avaliar_retrieval import avaliar_pergunta
    resultados = []
    for pq in perguntas:
        r = avaliar_pergunta(rag, pq, k=k, usar_rerank=usar_rerank)
        if r is not None:
            resultados.append(r)
    return resultados


def rodar_para_lexico(lexico, perguntas):
    print(f"\n{'='*70}\nLEXICO = {lexico.upper()}\n{'='*70}")
    rag = RAGHibrido(chroma_dir=CHROMA_DIR_BENCH, colecao=f"manuais_harbor_bench_{lexico}", lexico=lexico)
    rag.indexar(forcar=True)

    lex_resultados, tempos = avaliar_lexical_isolado(rag, perguntas)
    lex_m = metricas(lex_resultados)
    latencia_media_ms = statistics.mean(tempos) * 1000 if tempos else 0.0
    print(f"[lexical isolado]      Recall@{K}={lex_m['recall']*100:.0f}%  "
          f"Precision@{K}={lex_m['precision']*100:.0f}%  MRR={lex_m['mrr']:.3f}  "
          f"latencia_media={latencia_media_ms:.2f}ms")

    hyb_resultados = avaliar_pipeline(rag, perguntas, usar_rerank=False)
    hyb_m = metricas(hyb_resultados)
    print(f"[hybrid sem rerank]    Recall@{K}={hyb_m['recall']*100:.0f}%  "
          f"Precision@{K}={hyb_m['precision']*100:.0f}%  MRR={hyb_m['mrr']:.3f}")

    rer_resultados = avaliar_pipeline(rag, perguntas, usar_rerank=True)
    rer_m = metricas(rer_resultados)
    print(f"[hybrid com rerank]    Recall@{K}={rer_m['recall']*100:.0f}%  "
          f"Precision@{K}={rer_m['precision']*100:.0f}%  MRR={rer_m['mrr']:.3f}")

    return {
        "lexical_isolado": lex_m, "latencia_media_ms": latencia_media_ms,
        "hybrid_sem_rerank": hyb_m, "hybrid_com_rerank": rer_m,
    }


def main():
    perguntas = carregar_perguntas_rag()
    if not perguntas:
        print("Nenhuma golden question de rota 'rag' encontrada.")
        return
    print(f"Comparando TF-IDF vs BM25 em {len(perguntas)} perguntas (k={K})...")

    resultado_tfidf = rodar_para_lexico("tfidf", perguntas)
    resultado_bm25 = rodar_para_lexico("bm25", perguntas)

    print(f"\n{'='*70}\nRESUMO COMPARATIVO\n{'='*70}")
    print(f"{'Cenario':<22} {'TF-IDF (R/P/MRR)':<28} {'BM25 (R/P/MRR)':<28}")
    for cenario, label in [("lexical_isolado", "Lexical isolado"),
                            ("hybrid_sem_rerank", "Hybrid sem rerank"),
                            ("hybrid_com_rerank", "Hybrid com rerank")]:
        t, b = resultado_tfidf[cenario], resultado_bm25[cenario]
        t_str = f"{t['recall']*100:.0f}%/{t['precision']*100:.0f}%/{t['mrr']:.3f}"
        b_str = f"{b['recall']*100:.0f}%/{b['precision']*100:.0f}%/{b['mrr']:.3f}"
        print(f"{label:<22} {t_str:<28} {b_str:<28}")
    print(f"{'Latencia lexical':<22} {resultado_tfidf['latencia_media_ms']:.2f}ms"
          f"{'':<22} {resultado_bm25['latencia_media_ms']:.2f}ms")

    resultados_path = EVAL_DIR / "resultados_comparacao_tfidf_bm25.json"
    resultados_path.write_text(
        json.dumps({"tfidf": resultado_tfidf, "bm25": resultado_bm25}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResultados salvos em {resultados_path}")


if __name__ == "__main__":
    main()
