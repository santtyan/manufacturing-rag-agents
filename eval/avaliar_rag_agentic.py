"""
P3 do plano "RAG agentico + chunking" (2026-09-14): avaliacao reproduzivel do RAG agentico
(rag/rag_agentic.py), com custo/latencia e trace -- substituindo a medicao ad-hoc original
(n=15, "12/15 -> 11/15") que nao era reproduzivel (nenhum script versionado, nenhum trace).

ACHADO REAL que motiva esta reexecucao: n=15 e pequeno demais -- 12/15 vs 11/15 e UMA pergunta
de diferenca, dentro do ruido estatistico. Pesquisa de estado da arte (ACL 2026 Industry Track,
arXiv:2601.07711, "Is Agentic RAG worth it?") mostra que agentic RAG e sistematicamente ate 3,6x
mais caro e pior que rerank puro em selecionar documentos relevantes -- exatamente o padrao que
o Harbor ja tinha, so que sem numero robusto nem custo medido. Este script fecha as duas lacunas
rodando sobre as 49 perguntas de rota "rag" (nao 15), com trace via shared.trace e metricas de
custo agentico (numero de iteracoes, taxa de re-retrieval, latencia extra) exigidas pelo SoK de
agentic RAG (arXiv:2603.07379: "output-only metrics are insufficient for agentic RAG").

Compara lado a lado, na MESMA pergunta: RAGHibrido.buscar() puro (baseline) vs.
buscar_agentic() (rag/rag_agentic.py) -- Recall@k/MRR de documento E de chunk (reusa as funcoes
de eval/avaliar_retrieval.py, nunca duplicar), mais o custo agentico extra.

Uso: python eval/avaliar_rag_agentic.py [--k 3]
     Pre-requisito: Ollama rodando (qwen2.5:7b, usado pelo juiz e pela reescrita de query).
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))

from shared.trace import iniciar_trace, registrar_passo

from avaliar_retrieval import arquivo_esperado, secao_do_chunk, secao_esperada
from rag_agentic import buscar_agentic_com_estado
from rag_hibrido_langchain import RAGHibrido

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN = EVAL_DIR / "golden_questions.json"
RESULTADOS = EVAL_DIR / "resultados_rag_agentic.json"
TRACE_ARQUIVO = EVAL_DIR / "traces" / "rag_agentic.jsonl"

K_DEFAULT = 3


def _metricas_de_candidatos(candidatos, alvo, sec_esperada):
    """Recall@k/RR de documento e de chunk para uma lista de candidatos ja ordenada -- mesma
    logica de eval/avaliar_retrieval.py::avaliar_pergunta(), fatorada aqui para ser aplicada
    tanto ao baseline quanto ao agentic sem duplicar o calculo em si."""
    fontes = [c["fonte"] for c in candidatos]
    recall_doc = 1.0 if alvo in fontes else 0.0
    rr_doc = 0.0
    for i, f in enumerate(fontes, start=1):
        if f == alvo:
            rr_doc = 1.0 / i
            break

    recall_chunk, rr_chunk = None, None
    if sec_esperada is not None:
        secoes = [secao_do_chunk(c["texto"]) if f == alvo else None for c, f in zip(candidatos, fontes)]
        recall_chunk = 1.0 if sec_esperada in secoes else 0.0
        rr_chunk = 0.0
        for i, s in enumerate(secoes, start=1):
            if s == sec_esperada:
                rr_chunk = 1.0 / i
                break
    return recall_doc, rr_doc, recall_chunk, rr_chunk


def avaliar_pergunta(rag, pergunta_obj, k, trace):
    alvo = arquivo_esperado(pergunta_obj)
    if alvo is None:
        return None
    sec_esperada = secao_esperada(pergunta_obj)

    # Baseline: RAGHibrido.buscar() puro, mesma chamada de producao (usar_rerank=True default).
    t0 = time.time()
    candidatos_base = rag.buscar(pergunta_obj["pergunta"], k=k, usar_rerank=True, usar_hybrid=True)
    duracao_base_s = time.time() - t0
    recall_doc_base, rr_doc_base, recall_chunk_base, rr_chunk_base = _metricas_de_candidatos(
        candidatos_base, alvo, sec_esperada
    )

    # Agentic: buscar_agentic_com_estado() -- mesmo k, expoe tentativas/pergunta_efetiva para
    # medir o custo agentico real (nao so o resultado final).
    t0 = time.time()
    estado_final = buscar_agentic_com_estado(pergunta_obj["pergunta"], rag, k=k)
    duracao_agentic_s = time.time() - t0
    candidatos_agentic = estado_final["candidatos"]
    tentativas = estado_final.get("tentativas", 0)
    reformulou = tentativas > 0
    recall_doc_ag, rr_doc_ag, recall_chunk_ag, rr_chunk_ag = _metricas_de_candidatos(
        candidatos_agentic, alvo, sec_esperada
    )

    registrar_passo(
        trace, tipo="retrieval", action="comparar_baseline_vs_agentic",
        action_input=pergunta_obj["id"],
        observation=f"reformulou={reformulou} recall_doc_base={recall_doc_base} recall_doc_agentic={recall_doc_ag}",
        duracao_s=round(duracao_base_s + duracao_agentic_s, 4),
    )

    return {
        "id": pergunta_obj["id"],
        "arquivo_esperado": alvo,
        "secao_esperada": sec_esperada,
        "reformulou": reformulou,
        "tentativas": tentativas,
        "recall_doc_base": recall_doc_base, "rr_doc_base": round(rr_doc_base, 3),
        "recall_chunk_base": recall_chunk_base,
        "rr_chunk_base": round(rr_chunk_base, 3) if rr_chunk_base is not None else None,
        "recall_doc_agentic": recall_doc_ag, "rr_doc_agentic": round(rr_doc_ag, 3),
        "recall_chunk_agentic": recall_chunk_ag,
        "rr_chunk_agentic": round(rr_chunk_ag, 3) if rr_chunk_ag is not None else None,
        "duracao_base_s": round(duracao_base_s, 3),
        "duracao_agentic_s": round(duracao_agentic_s, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=K_DEFAULT)
    args = parser.parse_args()

    dados = json.loads(GOLDEN.read_text(encoding="utf-8"))
    perguntas_rag = [p for p in dados["perguntas"] if p.get("rota_esperada") == "rag"]
    print(f"Avaliando RAG agentico vs. baseline sobre {len(perguntas_rag)} perguntas "
          f"(n=49, nao mais n=15 -- ver docstring), k={args.k}...\n")

    rag = RAGHibrido()
    rag.indexar(forcar=False)

    resultados = []
    with iniciar_trace(
        "rag_agentic", entrada=f"n={len(perguntas_rag)},k={args.k}",
        versoes={"golden_set": "eval/golden_questions.json"}, arquivo=TRACE_ARQUIVO,
    ) as trace:
        for pq in perguntas_rag:
            r = avaliar_pergunta(rag, pq, args.k, trace)
            if r is None:
                continue
            resultados.append(r)
            marca_base = "OK" if r["recall_doc_base"] == 1.0 else "MISS"
            marca_ag = "OK" if r["recall_doc_agentic"] == 1.0 else "MISS"
            reform = " [REFORMULOU]" if r["reformulou"] else ""
            print(f"[{r['id']:38}] base={marca_base:4} agentic={marca_ag:4}{reform}")
        trace.resultado_final = f"{len(resultados)} perguntas avaliadas"
        trace.sucesso = len(resultados) > 0

    if not resultados:
        print("\nNenhum resultado avaliavel.")
        return

    n = len(resultados)
    n_reformulou = sum(1 for r in resultados if r["reformulou"])
    recall_doc_base = statistics.mean(r["recall_doc_base"] for r in resultados)
    recall_doc_ag = statistics.mean(r["recall_doc_agentic"] for r in resultados)
    mrr_doc_base = statistics.mean(r["rr_doc_base"] for r in resultados)
    mrr_doc_ag = statistics.mean(r["rr_doc_agentic"] for r in resultados)

    com_chunk = [r for r in resultados if r["recall_chunk_base"] is not None]
    recall_chunk_base = statistics.mean(r["recall_chunk_base"] for r in com_chunk) if com_chunk else None
    recall_chunk_ag = statistics.mean(r["recall_chunk_agentic"] for r in com_chunk) if com_chunk else None

    duracao_total_base = sum(r["duracao_base_s"] for r in resultados)
    duracao_total_ag = sum(r["duracao_agentic_s"] for r in resultados)
    custo_relativo = duracao_total_ag / duracao_total_base if duracao_total_base else float("inf")

    resumo = {
        "n_perguntas": n,
        "k": args.k,
        "n_reformulou": n_reformulou,
        "taxa_re_retrieval": round(n_reformulou / n, 3),
        "recall_doc_base": round(recall_doc_base, 3), "recall_doc_agentic": round(recall_doc_ag, 3),
        "mrr_doc_base": round(mrr_doc_base, 3), "mrr_doc_agentic": round(mrr_doc_ag, 3),
        "recall_chunk_base": round(recall_chunk_base, 3) if recall_chunk_base is not None else None,
        "recall_chunk_agentic": round(recall_chunk_ag, 3) if recall_chunk_ag is not None else None,
        "duracao_total_base_s": round(duracao_total_base, 1),
        "duracao_total_agentic_s": round(duracao_total_ag, 1),
        "custo_relativo_agentic_vs_base": round(custo_relativo, 2),
        "resultados_por_pergunta": resultados,
    }
    RESULTADOS.write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"Perguntas avaliadas    : {n} (n=49, nao mais o n=15 da medicao original)")
    print(f"Taxa de re-retrieval   : {n_reformulou}/{n} ({resumo['taxa_re_retrieval']*100:.1f}%)")
    print()
    print(f"{'Metrica':30} {'Baseline':>12} {'Agentic':>12} {'Delta':>10}")
    print(f"{'Recall@k (documento)':30} {recall_doc_base*100:>11.1f}% {recall_doc_ag*100:>11.1f}% "
          f"{(recall_doc_ag-recall_doc_base)*100:>+9.1f}pp")
    print(f"{'MRR (documento)':30} {mrr_doc_base:>12.3f} {mrr_doc_ag:>12.3f} {mrr_doc_ag-mrr_doc_base:>+10.3f}")
    if com_chunk:
        print(f"{'Recall@k (chunk, ' + str(len(com_chunk)) + '/' + str(n) + ')':30} "
              f"{recall_chunk_base*100:>11.1f}% {recall_chunk_ag*100:>11.1f}% "
              f"{(recall_chunk_ag-recall_chunk_base)*100:>+9.1f}pp")
    print()
    print(f"Duracao total baseline : {duracao_total_base:.1f}s")
    print(f"Duracao total agentic  : {duracao_total_ag:.1f}s")
    print(f"Custo relativo         : {custo_relativo:.2f}x "
          f"({'MAIS CARO' if custo_relativo > 1 else 'mais barato'})")
    print(f"\nTrace salvo em         : {TRACE_ARQUIVO}")
    print(f"Resultados salvos em   : {RESULTADOS}")


if __name__ == "__main__":
    main()
