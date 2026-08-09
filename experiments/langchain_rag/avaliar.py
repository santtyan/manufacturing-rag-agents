"""
Avaliacao comparativa: producao (RAGHibrido) vs. Versao A (LangChain fiel) vs. Versao B
(LangChain BM25+RRF), todas contra o mesmo golden set (eval/golden_questions.json) e a
mesma metrica (Recall@5, Precision@5, MRR) de eval/avaliar_retrieval.py.

Isolamento: este script NAO importa nem modifica eval/avaliar_retrieval.py -- duplica
o loop de avaliacao (~15 linhas) para nao criar nenhuma dependencia de
experiments/langchain_rag/ sobre codigo de eval/. A logica de calculo de metrica e
identica de proposito (mesma formula de recall/precision/RR), para a comparacao ser
valida.

Uso: python experiments/langchain_rag/avaliar.py
"""
import csv
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(r"C:\Projetos\Harbor\rag")))
sys.path.insert(0, str(Path(__file__).parent))

from rag_hibrido import RAGHibrido  # noqa: E402
from rag_langchain_fiel import RAGLangChainFiel  # noqa: E402
from rag_langchain_bm25rrf import RAGLangChainBM25RRF  # noqa: E402

EXP_DIR = Path(__file__).parent
GOLDEN = Path(r"C:\Projetos\Harbor\eval\golden_questions.json")
K = 5


def arquivo_esperado(pergunta_obj):
    """Identica a eval/avaliar_retrieval.py::arquivo_esperado -- mesma extracao do
    campo 'fonte' do golden set, para garantir alvo de relevancia identico."""
    fonte = pergunta_obj.get("fonte", "")
    if not fonte.endswith(".md"):
        return None
    return fonte.rsplit("/", 1)[-1]


def avaliar_pergunta(rag, pergunta_obj, k=K, usar_rerank=False):
    """Identica a eval/avaliar_retrieval.py::avaliar_pergunta -- mesma formula de
    recall@k (binario), precision@k e reciprocal rank."""
    alvo = arquivo_esperado(pergunta_obj)
    if alvo is None:
        return None

    candidatos = rag.buscar(pergunta_obj["pergunta"], k=k, usar_rerank=usar_rerank, usar_hybrid=True, k_candidatos=max(k, 10))
    fontes = [c["fonte"] for c in candidatos]

    relevantes_no_topk = sum(1 for f in fontes if f == alvo)
    recall_at_k = 1.0 if relevantes_no_topk > 0 else 0.0
    precision_at_k = relevantes_no_topk / len(fontes) if fontes else 0.0

    rr = 0.0
    for i, f in enumerate(fontes, start=1):
        if f == alvo:
            rr = 1.0 / i
            break

    return {"id": pergunta_obj["id"], "recall_at_k": recall_at_k, "precision_at_k": round(precision_at_k, 3), "reciprocal_rank": round(rr, 3)}


def avaliar_implementacao(nome, rag, perguntas_rag, usar_rerank):
    t0 = time.time()
    rag.indexar(forcar=True)
    tempo_indexacao = time.time() - t0

    resultados = []
    for pq in perguntas_rag:
        r = avaliar_pergunta(rag, pq, k=K, usar_rerank=usar_rerank)
        if r is not None:
            resultados.append(r)

    if not resultados:
        return None

    return {
        "implementacao": nome,
        "recall_at_5": round(statistics.mean(r["recall_at_k"] for r in resultados) * 100, 1),
        "precision_at_5": round(statistics.mean(r["precision_at_k"] for r in resultados) * 100, 1),
        "mrr": round(statistics.mean(r["reciprocal_rank"] for r in resultados), 3),
        "tempo_indexacao_s": round(tempo_indexacao, 2),
        "n_perguntas": len(resultados),
    }


def contar_linhas(caminho):
    return len(Path(caminho).read_text(encoding="utf-8").splitlines())


def main():
    usar_rerank = "--rerank" in sys.argv
    modo = "retrieval + rerank" if usar_rerank else "retrieval isolado"

    dados = json.loads(GOLDEN.read_text(encoding="utf-8"))
    perguntas_rag = [p for p in dados["perguntas"] if p.get("rota_esperada") == "rag"]
    print(f"Avaliando {modo} em {len(perguntas_rag)} perguntas, 3 implementacoes...\n")

    implementacoes = [
        ("producao_rag_hibrido", RAGHibrido(), r"C:\Projetos\Harbor\rag\rag_hibrido.py"),
        ("langchain_fiel_tfidf_uniao", RAGLangChainFiel(), r"C:\Projetos\Harbor\experiments\langchain_rag\rag_langchain_fiel.py"),
        ("langchain_bm25_rrf", RAGLangChainBM25RRF(), r"C:\Projetos\Harbor\experiments\langchain_rag\rag_langchain_bm25rrf.py"),
    ]

    linhas = []
    for nome, rag, arquivo in implementacoes:
        print(f"--- {nome} ---")
        r = avaliar_implementacao(nome, rag, perguntas_rag, usar_rerank)
        if r is None:
            print("  sem resultados avaliaveis, pulando\n")
            continue
        r["linhas_codigo"] = contar_linhas(arquivo)
        linhas.append(r)
        print(f"  Recall@5: {r['recall_at_5']}%  Precision@5: {r['precision_at_5']}%  "
              f"MRR: {r['mrr']}  indexacao: {r['tempo_indexacao_s']}s  LOC: {r['linhas_codigo']}\n")

    saida = EXP_DIR / "resultados_comparativos.csv"
    with saida.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["implementacao", "recall_at_5", "precision_at_5", "mrr", "tempo_indexacao_s", "linhas_codigo", "n_perguntas"])
        for r in linhas:
            w.writerow([r["implementacao"], r["recall_at_5"], r["precision_at_5"], r["mrr"],
                        r["tempo_indexacao_s"], r["linhas_codigo"], r["n_perguntas"]])

    print(f"Resultados salvos em {saida}")


if __name__ == "__main__":
    main()
