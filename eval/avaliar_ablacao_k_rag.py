"""
P1 do plano "RAG agentico + chunking" (2026-09-14): ablacao de k (numero de chunks passados ao
gerador) no RAG de texto, sobre o golden set real de 49 perguntas de rota "rag".

Motivacao: P0 (eval/avaliar_retrieval.py) mediu que o gap doc-level vs. chunk-level, com rerank,
e real mas pequeno (+3,0pp, 2 casos genuinos) -- e que aumentar k_candidatos (antes do corte
final) NAO resolve os 2 casos (a secao certa nao aparece nem estendendo o corte). O proximo
passo, mais barato que qualquer tecnica agentica, e medir se aumentar k NO GERADOR (quantos
chunks o LLM recebe no prompt, nao quantos o retrieval avalia) melhora a resposta final --
mesmo quando o chunk certo nao vem em 1o lugar, ele pode vir em 2o/3o e ainda assim ajudar o
LLM se o prompt incluir mais chunks.

ACHADO REAL do proprio projeto que orienta esta ablacao (2026-09-09): score de Cross-Encoder e
pessimo preditor de acerto (casos que erram com score alto, casos que acertam com score baixo).
Por isso esta ablacao varia k SEM nenhum filtro de score -- aumentar k e confiar no proprio LLM
para triangular entre os chunks recebidos, nao tentar filtrar por confianca do reranker (que ja
provou nao ser confiavel).

Mede FAITHFULNESS (reusa eval/rodar_golden.py::avaliar(), mesma logica do harness principal --
nunca duplicar) e CUSTO/LATENCIA juntos (regra 4 da skill metricas-avaliacao-ia-industrial:
metrica de qualidade sozinha nunca e conclusao valida), para cada valor de k testado.

Uso: python eval/avaliar_ablacao_k_rag.py [--k 3,5,10]  (default: 3,5,10)
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag_gerador
from rodar_golden import GOLDEN, avaliar, call_ollama

EVAL_DIR = Path(__file__).resolve().parent
RESULTADOS = EVAL_DIR / "resultados_ablacao_k_rag.json"


def rodar_k(rag, perguntas_rag, k, usar_adaptive_k=False):
    """Roda rag_gerador.rag_responder() (fonte unica compartilhada com producao/harness) para
    cada pergunta com o k dado (ou Adaptive-k, se usar_adaptive_k=True), mede faithfulness (via
    rodar_golden.avaliar) e tempo por pergunta. Retorna lista de resultados por pergunta."""
    resultados = []
    for pq in perguntas_rag:
        t0 = time.time()
        resposta, docs, contexto = rag_gerador.rag_responder(
            pq["pergunta"], rag, call_ollama, k=k,
            buscar_fallback=rag_gerador.buscar_fallback_tfidf,
            usar_adaptive_k=usar_adaptive_k,
        )
        duracao_s = time.time() - t0
        r = avaliar(pq, resposta, contexto)
        r["duracao_s"] = round(duracao_s, 2)
        r["k"] = len(docs) if usar_adaptive_k else k  # k EFETIVO usado nesta pergunta
        resultados.append(r)
    return resultados


def agregar(resultados):
    n = len(resultados)
    com_ff = [r for r in resultados if r["faithfulness"] is not None]
    ff_medio = sum(r["faithfulness"] for r in com_ff) / len(com_ff) if com_ff else None
    alucinacoes = sum(1 for r in resultados if r["alucinou"])
    duracao_media = sum(r["duracao_s"] for r in resultados) / n if n else 0.0
    duracao_total = sum(r["duracao_s"] for r in resultados)
    return {
        "n_perguntas": n,
        "faithfulness_medio": round(ff_medio, 3) if ff_medio is not None else None,
        "n_com_faithfulness": len(com_ff),
        "alucinacoes": alucinacoes,
        "duracao_media_s": round(duracao_media, 2),
        "duracao_total_s": round(duracao_total, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=str, default="3,5,10",
                         help="valores de k a testar, separados por virgula (default: 3,5,10)")
    parser.add_argument("--adaptive", action="store_true",
                         help="tambem roda Adaptive-k (P1b do plano, EMNLP 2025 arXiv:2506.08479) "
                              "-- corta pelo gap de score RRF em vez de k fixo")
    args = parser.parse_args()
    valores_k = [int(k.strip()) for k in args.k.split(",")]

    dados = json.loads(GOLDEN.read_text(encoding="utf-8"))
    perguntas_rag = [p for p in dados["perguntas"] if p.get("rota_esperada") == "rag"]
    print(f"Ablacao de k sobre {len(perguntas_rag)} perguntas de rota 'rag', "
          f"k in {valores_k} (sem filtro de score -- ver docstring)...\n")

    rag = rag_gerador.carregar_rag_hibrido()
    if rag is None:
        print("RAG hibrido indisponivel -- abortando (fallback TF-IDF nao suporta k arbitrario "
              "de forma comparavel).")
        return

    chaves = [str(k) for k in valores_k] + (["adaptive"] if args.adaptive else [])
    resultado_final = {"k_testados": valores_k, "adaptive_testado": args.adaptive, "por_k": {}}
    for k in valores_k:
        print(f"--- k={k} ---")
        resultados = rodar_k(rag, perguntas_rag, k)
        agregados = agregar(resultados)
        resultado_final["por_k"][str(k)] = {
            "agregados": agregados,
            "por_pergunta": [
                {"id": r["id"], "faithfulness": r["faithfulness"], "alucinou": r["alucinou"],
                 "duracao_s": r["duracao_s"]}
                for r in resultados
            ],
        }
        ff = agregados["faithfulness_medio"]
        ff_str = f"{ff*100:.1f}%" if ff is not None else "-"
        print(f"Faithfulness medio : {ff_str} (sobre {agregados['n_com_faithfulness']} perguntas)")
        print(f"Alucinacoes        : {agregados['alucinacoes']}/{agregados['n_perguntas']}")
        print(f"Duracao media      : {agregados['duracao_media_s']:.2f}s/pergunta")
        print(f"Duracao total      : {agregados['duracao_total_s']:.1f}s\n")

    if args.adaptive:
        print("--- adaptive-k (EMNLP 2025, arXiv:2506.08479) ---")
        resultados = rodar_k(rag, perguntas_rag, k=None, usar_adaptive_k=True)
        agregados = agregar(resultados)
        agregados["k_efetivo_medio"] = round(sum(r["k"] for r in resultados) / len(resultados), 2)
        resultado_final["por_k"]["adaptive"] = {
            "agregados": agregados,
            "por_pergunta": [
                {"id": r["id"], "faithfulness": r["faithfulness"], "alucinou": r["alucinou"],
                 "duracao_s": r["duracao_s"], "k_efetivo": r["k"]}
                for r in resultados
            ],
        }
        ff = agregados["faithfulness_medio"]
        ff_str = f"{ff*100:.1f}%" if ff is not None else "-"
        print(f"Faithfulness medio : {ff_str} (sobre {agregados['n_com_faithfulness']} perguntas)")
        print(f"Alucinacoes        : {agregados['alucinacoes']}/{agregados['n_perguntas']}")
        print(f"k efetivo medio    : {agregados['k_efetivo_medio']}")
        print(f"Duracao media      : {agregados['duracao_media_s']:.2f}s/pergunta")
        print(f"Duracao total      : {agregados['duracao_total_s']:.1f}s\n")

    RESULTADOS.write_text(json.dumps(resultado_final, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 70)
    print("RESUMO -- qualidade e custo lado a lado (regra 4 da skill "
          "metricas-avaliacao-ia-industrial)")
    print("=" * 70)
    print(f"{'k':>10} | {'faithfulness':>12} | {'alucinacoes':>11} | {'s/pergunta':>10} | {'total s':>8}")
    for k in valores_k:
        ag = resultado_final["por_k"][str(k)]["agregados"]
        ff = ag["faithfulness_medio"]
        ff_str = f"{ff*100:.1f}%" if ff is not None else "-"
        print(f"{k:>10} | {ff_str:>12} | {ag['alucinacoes']:>4}/{ag['n_perguntas']:<6} | "
              f"{ag['duracao_media_s']:>9.2f}s | {ag['duracao_total_s']:>7.1f}s")
    if args.adaptive:
        ag = resultado_final["por_k"]["adaptive"]["agregados"]
        ff = ag["faithfulness_medio"]
        ff_str = f"{ff*100:.1f}%" if ff is not None else "-"
        rotulo = f"adaptive(~{ag['k_efetivo_medio']})"
        print(f"{rotulo:>10} | {ff_str:>12} | {ag['alucinacoes']:>4}/{ag['n_perguntas']:<6} | "
              f"{ag['duracao_media_s']:>9.2f}s | {ag['duracao_total_s']:>7.1f}s")
    print(f"\nResultados salvos em {RESULTADOS}")


if __name__ == "__main__":
    main()
