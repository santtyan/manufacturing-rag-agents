"""
Item 5 do plano "Evoluir o RAG multimodal": experimento decisivo -- o rerank multimodal
(ColModernVBERT, top-p=0,7, calibrado no item 4) compensa um captioning ruim?

Compara dois cenarios sobre o mesmo corpus/golden set do Harbor (26 imagens, 29 perguntas):
  A) caption RUIM (moondream, rag/legendas_cache.json) + rerank ColModernVBERT (top-p=0,7)
  B) caption BOM (qwen3-vl:4b, rag/legendas_cache_qwen3-vl_4b.json) SEM rerank

Se (A) >= (B) em nDCG@5/Recall@5, o desenho de 2 estagios (retrieval barato + rerank caro)
compensa a escolha do VLM de captioning -- resultado forte para a arquitetura de producao,
independente de qual VLM acabar sendo usado. Se (A) < (B), a qualidade do captioning importa
mais que o rerank, e o esforco futuro deve ir para achar um VLM melhor, nao para o rerank.

Reusa as mesmas funcoes de eval/avaliar_benchmark_multimodal_2x2.py (estagio1_retrieval,
estagio2_rerank, calcular_metricas, resumir) -- nao duplica logica de retrieval/rerank.

Uso: python eval/avaliar_rerank_compensa_captioning.py
     Pre-requisito: rag/legendas_cache.json (moondream) e rag/legendas_cache_qwen3-vl_4b.json
     (qwen3-vl) ja existentes, eval/golden_questions_multimodal.json ja existente.
"""
import json
import sys
from pathlib import Path

import pandas  # ACHADO REAL: import antes de sentence_transformers evita access violation
import datasets  # ACHADO REAL (2026-09-11): import antes de sentence_transformers evita access
# violation por conflito de DLL torch x pyarrow.dataset -- ver skill rag-multimodal.

EVAL_DIR = Path(__file__).resolve().parent
HARBOR_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))

from avaliar_benchmark_multimodal_2x2 import (
    estagio1_retrieval, estagio2_rerank, calcular_metricas, resumir, K_FINAL,
)

IMAGENS_HARBOR = HARBOR_ROOT / "rag" / "manuais_imagens"
GOLDEN_HARBOR = EVAL_DIR / "golden_questions_multimodal.json"
CACHE_MOONDREAM = HARBOR_ROOT / "rag" / "legendas_cache.json"
CACHE_QWEN3VL = HARBOR_ROOT / "rag" / "legendas_cache_qwen3-vl_4b.json"


def carregar_corpus(caminho_cache_legendas):
    corpus_docs = json.loads(caminho_cache_legendas.read_text(encoding="utf-8"))
    golden = json.loads(GOLDEN_HARBOR.read_text(encoding="utf-8"))["perguntas"]
    imagens_por_id = {d["id"]: IMAGENS_HARBOR / d["fonte"] for d in corpus_docs}
    perguntas = [pq["pergunta"] for pq in golden]
    alvos = [pq["documento_relevante"] for pq in golden]
    return corpus_docs, imagens_por_id, perguntas, alvos


def main():
    resultados = []

    print("=" * 74)
    print("ITEM 5 -- REKANK COMPENSA CAPTIONING RUIM?")
    print("=" * 74)

    # Cenario A: caption ruim (moondream) + rerank top-p=0,7
    print("\n--- Cenario A: caption RUIM (moondream) + rerank ColModernVBERT (top-p=0,7) ---")
    corpus_docs, imagens_por_id, perguntas, alvos = carregar_corpus(CACHE_MOONDREAM)
    candidatos_por_pergunta, tempo_idx = estagio1_retrieval(
        corpus_docs, perguntas, HARBOR_ROOT / "rag" / "chroma_db_multimodal", "imagens_harbor_multimodal_v1_moondream")
    rankings_rerank, tempo_rerank, n_cand = estagio2_rerank(
        candidatos_por_pergunta, imagens_por_id, perguntas, estrategia="top_p", top_p_limiar=0.7)
    metricas_a = calcular_metricas(rankings_rerank, alvos)
    resultados.append(resumir("A) moondream + rerank top-p=0.7", metricas_a, tempo_idx + tempo_rerank, n_cand,
                               unidade="imagem(ns) rerankeada(s)"))

    # Cenario B: caption bom (qwen3-vl) sem rerank
    print("\n--- Cenario B: caption BOM (qwen3-vl:4b) SEM rerank ---")
    corpus_docs_b, imagens_por_id_b, perguntas_b, alvos_b = carregar_corpus(CACHE_QWEN3VL)
    candidatos_por_pergunta_b, tempo_idx_b = estagio1_retrieval(
        corpus_docs_b, perguntas_b, HARBOR_ROOT / "rag" / "chroma_db_multimodal", "imagens_harbor_multimodal_v1_qwen3vl")
    rankings_sem_rerank = [[doc_id for doc_id, _ in c] for c in candidatos_por_pergunta_b]
    metricas_b = calcular_metricas(rankings_sem_rerank, alvos_b)
    resultados.append(resumir("B) qwen3-vl SEM rerank", metricas_b, tempo_idx_b, len(corpus_docs_b)))

    print("\n" + "=" * 74)
    print("RESUMO COMPARATIVO -- ITEM 5")
    print("=" * 74)
    for r in resultados:
        print(f"{r['nome']:40} nDCG@{K_FINAL}={r['ndcg']:.3f}  Recall@{K_FINAL}={r['recall']*100:.0f}%  "
              f"MRR={r['mrr']:.3f}  custo={r['tempo_s']}s")

    a, b = resultados[0], resultados[1]
    print("\nVEREDITO:")
    if a["ndcg"] >= b["ndcg"] and a["recall"] >= b["recall"]:
        print("  Rerank COMPENSA captioning ruim -- (A) igualou ou superou (B) em nDCG e Recall.")
        print("  Desenho de 2 estagios e defensavel mesmo sem VLM aprovado.")
    elif a["ndcg"] < b["ndcg"] and a["recall"] < b["recall"]:
        print("  Rerank NAO compensa captioning ruim -- (B) superou (A) em nDCG e Recall.")
        print("  Qualidade do VLM de captioning importa mais que o rerank multimodal.")
    else:
        print("  Resultado misto -- (A) e (B) trocam de posicao entre nDCG e Recall. Ver numeros acima.")

    saida = EVAL_DIR / "resultados_item5_rerank_vs_captioning.json"
    saida.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultados salvos em {saida}")


if __name__ == "__main__":
    main()
