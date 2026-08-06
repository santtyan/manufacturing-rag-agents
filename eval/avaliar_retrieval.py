"""
Avaliacao ISOLADA de retrieval (Recall@k, Precision@k, MRR) -- distinto de avaliar a
resposta final (rodar_golden.py). Mede se o RAG hibrido (rag/rag_hibrido.py, renomeado de
rag_neural.py em 2026-07-15) recupera o chunk certo, independente do que o LLM faz depois
com ele.

Motivacao: eval/resultados_consistencia.csv mostrou uma pergunta variando de 0% a 100%
de faithfulness entre execucoes identicas. Sem medir retrieval isoladamente e impossivel
saber se a causa e retrieval inconsistente (chunk certo as vezes nao entra no top-k) ou
so a geracao do LLM (chunk entra sempre, LLM que erra ao usar). Como RAGHibrido.buscar()
nao tem nenhuma fonte de aleatoriedade (embeddings e cross-encoder sao deterministicos),
a hipotese e que o retrieval e estavel e a variancia vem da geracao -- este script
confirma ou refuta essa hipotese com dados.

Golden set: (pergunta, arquivo-fonte esperado) derivado das golden questions de rota
"rag" em golden_questions.json (campo "fonte"). Como o corpus tem so 5 documentos e a
maioria das perguntas usa so 1 manual, o "documento certo" (nao chunk_id especifico) e
o alvo de relevancia -- suficiente para medir Recall@k/Precision@k/MRR na escala do
corpus atual sem exigir anotacao manual de chunk-a-chunk.

Uso: python eval/avaliar_retrieval.py [--rerank]
     (default: usar_rerank=False, mede o RETRIEVAL isolado. Com --rerank, mede o pipeline
     completo retrieval+rerank de ponta a ponta -- ver avaliar_pergunta() para o porque de
     medir os dois separadamente.)
"""
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Projetos\Harbor\rag")
sys.path.insert(0, str(Path(__file__).parent))

from rag_hibrido import RAGHibrido

EVAL_DIR = Path(__file__).parent
GOLDEN = EVAL_DIR / "golden_questions.json"

K = 5  # top-k avaliado (k_candidatos default do RAGHibrido.buscar ja e 10 nos bastidores)


def arquivo_esperado(pergunta_obj):
    """Extrai o nome do arquivo .md esperado do campo 'fonte' (ex:
    'rag/manuais/manual_manutencao_sensores_industriais.md' -> 'manual_...md')."""
    fonte = pergunta_obj.get("fonte", "")
    if not fonte.endswith(".md"):
        return None
    return fonte.rsplit("/", 1)[-1]


def avaliar_pergunta(rag, pergunta_obj, k=K, usar_rerank=False):
    """Roda buscar() e calcula recall@k/precision@k/rr (reciprocal rank) contra o arquivo
    esperado.

    usar_rerank=False (default): mede a qualidade do RETRIEVAL (hybrid E5+TF-IDF)
    isoladamente. O rerank e uma etapa de POS-processamento que roda por cima do retrieval;
    avaliar as duas juntas misturaria as duas fontes de erro. usar_rerank=True mede o
    pipeline completo (retrieval+rerank) de ponta a ponta -- a metrica que falta comparar
    contra o NanoBEIR, que ja e medido com rerank (ver eval/avaliar_retrieval_nanobeir.py)."""
    alvo = arquivo_esperado(pergunta_obj)
    if alvo is None:
        return None

    candidatos = rag.buscar(pergunta_obj["pergunta"], k=k, usar_rerank=usar_rerank, usar_hybrid=True, k_candidatos=max(k, 10))
    fontes = [c["fonte"] for c in candidatos]

    relevantes_no_topk = sum(1 for f in fontes if f == alvo)
    recall_at_k = 1.0 if relevantes_no_topk > 0 else 0.0  # 1 doc relevante por pergunta -> recall e binario
    precision_at_k = relevantes_no_topk / len(fontes) if fontes else 0.0

    rr = 0.0
    for i, f in enumerate(fontes, start=1):
        if f == alvo:
            rr = 1.0 / i
            break

    return {
        "id": pergunta_obj["id"],
        "arquivo_esperado": alvo,
        "fontes_recuperadas": fontes,
        "recall_at_k": recall_at_k,
        "precision_at_k": round(precision_at_k, 3),
        "reciprocal_rank": round(rr, 3),
    }


def main():
    usar_rerank = "--rerank" in sys.argv
    sufixo = "_rerank" if usar_rerank else ""
    resultados_path = EVAL_DIR / f"resultados_retrieval{sufixo}.csv"

    dados = json.loads(GOLDEN.read_text(encoding="utf-8"))
    perguntas_rag = [p for p in dados["perguntas"] if p.get("rota_esperada") == "rag"]

    if not perguntas_rag:
        print("Nenhuma golden question de rota 'rag' encontrada.")
        return

    modo = "retrieval + rerank (pipeline completo)" if usar_rerank else "retrieval isolado (sem rerank)"
    print(f"Indexando RAG hibrido (forcar=True, para pegar manuais novos) e avaliando {modo} "
          f"em {len(perguntas_rag)} perguntas (k={K})...\n")
    rag = RAGHibrido()
    rag.indexar(forcar=True)  # forcar=True: sem isso, reaproveita indice antigo em disco e
    # ignora .md adicionados depois da ultima indexacao (achado real, 2026-07-16 -- as 12
    # perguntas dos 4 manuais novos deram 100% MISS ate essa correcao).

    resultados = []
    for pq in perguntas_rag:
        r = avaliar_pergunta(rag, pq, k=K, usar_rerank=usar_rerank)
        if r is None:
            print(f"[{pq['id']:26}] sem arquivo-fonte .md no golden set -- pulando")
            continue
        resultados.append(r)
        status = "OK " if r["recall_at_k"] == 1.0 else "MISS"
        print(f"[{r['id']:26}] {status} esperado={r['arquivo_esperado']:45} "
              f"precision@{K}={r['precision_at_k']:.2f} RR={r['reciprocal_rank']:.2f}")

    if not resultados:
        print("\nNenhum resultado avaliavel.")
        return

    with resultados_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "arquivo_esperado", "recall_at_k", "precision_at_k", "reciprocal_rank", "fontes_recuperadas"])
        for r in resultados:
            w.writerow([r["id"], r["arquivo_esperado"], r["recall_at_k"], r["precision_at_k"],
                        r["reciprocal_rank"], ";".join(r["fontes_recuperadas"])])

    recall_medio = statistics.mean(r["recall_at_k"] for r in resultados)
    precision_media = statistics.mean(r["precision_at_k"] for r in resultados)
    mrr = statistics.mean(r["reciprocal_rank"] for r in resultados)

    print("\n" + "=" * 60)
    print(f"Modo               : {modo}")
    print(f"Perguntas avaliadas: {len(resultados)}")
    print(f"Recall@{K} medio    : {recall_medio*100:.0f}%")
    print(f"Precision@{K} media : {precision_media*100:.0f}%")
    print(f"MRR                : {mrr:.3f}")
    print(f"Resultados salvos  : {resultados_path}")


if __name__ == "__main__":
    main()
