"""
Avaliacao do corpus OpenPack (rag/corpus_openpack_janelas.json) com a metrica correta para a
arquitetura RAG-HAR (arXiv:2512.08984) -- k-NN LABEL PURITY, nao Recall@k de documento unico.

ACHADO REAL (2026-09-12): a primeira tentativa de avaliacao usou eval/avaliar_benchmark_multimodal_2x2.py
com Recall@5/nDCG@5 (metrica de "encontrar O documento certo" -- instance-level retrieval),
resultando em Recall@5=0% mesmo com E5 denso PURO (sem BM25/hybrid). Investigacao confirmou que
NAO e bug de codigo: o RAG-HAR nunca faz instance-level retrieval. Releitura do paper (Fig. 2,
Secao III-D) confirma que ele recupera K VIZINHOS ESTATISTICAMENTE PARECIDOS via ANN para dar
contexto a um LLM que CLASSIFICA a amostra de teste numa das N classes de atividade conhecidas --
a metrica real reportada e F1-Score de classificacao. A literatura formal de retrieval (Oxford5k/
INSTRE vs. benchmarks de categoria) distingue exatamente isso: "instance-level retrieval" (achar
o documento exato) vs. "class-level retrieval" (achar qualquer documento da mesma classe
semantica) -- o corpus OpenPack, com ~100 janelas quase identicas em texto por classe, so faz
sentido avaliado como class-level. Metrica padrao da literatura de retrieval-augmented
classification: k-NN LABEL PURITY -- fracao dos k vizinhos recuperados que compartilham o
rotulo (aqui, operacao) da janela de origem da pergunta.

Reusa o golden set ja existente (eval/golden_questions_openpack.json) sem reescrever as
perguntas: cada `documento_relevante` (janela_id) tem uma `operacao` conhecida via
outputs/pipeline8_openpack/janelas_amostradas.csv -- a pergunta continua sendo "descreva este
perfil de movimento", so a metrica de sucesso muda de "achou a janela X" para "achou vizinhos da
mesma operacao que a janela X".

Uso: python eval/avaliar_retrieval_openpack.py [--k 5]
     Pre-requisito: rag/corpus_openpack_janelas.json e rag/chroma_db_openpack/ ja indexados
     (rag/rag_openpack_texto.py), outputs/pipeline8_openpack/janelas_amostradas.csv existente.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import datasets  # noqa: F401 (import de protecao contra access violation torch x pyarrow -- ver skill rag-multimodal)

EVAL_DIR = Path(__file__).resolve().parent
HARBOR_ROOT = EVAL_DIR.parent

CORPUS = HARBOR_ROOT / "rag" / "corpus_openpack_janelas.json"
GOLDEN = EVAL_DIR / "golden_questions_openpack.json"
JANELAS_AMOSTRADAS = HARBOR_ROOT / "outputs" / "pipeline8_openpack" / "janelas_amostradas.csv"
CHROMA_DIR = HARBOR_ROOT / "rag" / "chroma_db_openpack"
COLECAO = "openpack_janelas_v1"

K_DEFAULT = 5


def agregar_por_fonte(candidatos):
    """Colapsa os 4 sub-vetores (completo/inicio/meio/fim) numa entrada por janela_id, mantendo
    a ordem de 1a aparicao (ja por score decrescente) -- mesma logica de
    eval/avaliar_benchmark_multimodal_2x2.py::_agregar_por_fonte(), duplicada aqui por ser um
    script standalone sem depender do benchmark multimodal (que tem logica de rerank irrelevante
    para este corpus sem imagem)."""
    vistos = []
    for c in candidatos:
        doc_id = c.get("id") or c.get("fonte", "")
        fonte = doc_id.rsplit("__", 1)[0] if "__" in doc_id else doc_id
        if fonte not in vistos:
            vistos.append(fonte)
    return vistos


def knn_label_purity(rag, golden, operacao_por_janela, k=K_DEFAULT, usar_hybrid=True):
    """Para cada pergunta com alvo conhecido, recupera os k vizinhos mais proximos (excluindo a
    propria janela-alvo do calculo de pureza, ja que ela e trivialmente 'da mesma classe' consigo
    mesma) e mede a fracao que pertence a MESMA operacao da janela-alvo. Retorna lista de purezas
    (uma por pergunta) e a lista de perguntas nao-respondiveis corretamente abstidas."""
    purezas = []
    abstencoes_corretas = 0
    n_nao_respondiveis = 0

    for pq in golden:
        alvo = pq["documento_relevante"]
        candidatos = rag.buscar(pq["pergunta"], k=k + 4, usar_rerank=False,
                                 usar_hybrid=usar_hybrid, usar_score_rrf=True)
        vizinhos = agregar_por_fonte(candidatos)

        if alvo is None:
            # Pergunta nao-respondivel: nao ha operacao esperada -- o "acerto" aqui seria o
            # sistema de producao nao afirmar nada com confianca (fora do escopo desta metrica
            # de retrieval puro, que sempre retorna algo). Contabiliza como referencia, nao
            # como purity.
            n_nao_respondiveis += 1
            continue

        operacao_alvo = operacao_por_janela.get(alvo)
        if operacao_alvo is None:
            continue

        vizinhos_excluindo_alvo = [v for v in vizinhos if v != alvo][:k]
        if not vizinhos_excluindo_alvo:
            purezas.append(0.0)
            continue

        acertos = sum(1 for v in vizinhos_excluindo_alvo if operacao_por_janela.get(v) == operacao_alvo)
        purezas.append(acertos / len(vizinhos_excluindo_alvo))

    return purezas, n_nao_respondiveis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=K_DEFAULT,
                         help=f"numero de vizinhos considerados na pureza (default {K_DEFAULT})")
    parser.add_argument("--sem-hybrid", action="store_true",
                         help="usa so E5 denso, sem BM25/RRF (comparacao de ablacao)")
    args = parser.parse_args()

    sys.path.insert(0, str(HARBOR_ROOT / "rag"))
    from rag_hibrido_langchain import RAGHibrido

    corpus_docs = json.loads(CORPUS.read_text(encoding="utf-8"))
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))["perguntas"]
    df_janelas = pd.read_csv(JANELAS_AMOSTRADAS)
    operacao_por_janela = dict(zip(df_janelas["janela_id"], df_janelas["operacao"]))

    rag = RAGHibrido(chroma_dir=CHROMA_DIR, colecao=COLECAO)
    rag.indexar(forcar=False, documentos_customizados=corpus_docs)

    purezas, n_nao_resp = knn_label_purity(
        rag, golden, operacao_por_janela, k=args.k, usar_hybrid=not args.sem_hybrid
    )

    media = sum(purezas) / len(purezas) if purezas else 0.0
    perfeitas = sum(1 for p in purezas if p == 1.0)
    zeradas = sum(1 for p in purezas if p == 0.0)

    print("=" * 70)
    print(f"k-NN LABEL PURITY -- corpus OpenPack (k={args.k}, "
          f"{'hibrido E5+BM25/RRF' if not args.sem_hybrid else 'so E5 denso'})")
    print("=" * 70)
    print(f"Perguntas avaliadas       : {len(purezas)} (de {len(golden)} totais, "
          f"{n_nao_resp} nao-respondiveis excluidas)")
    print(f"Pureza media              : {media*100:.1f}%")
    print(f"Purezas perfeitas (=100%) : {perfeitas}/{len(purezas)}")
    print(f"Purezas zeradas (=0%)     : {zeradas}/{len(purezas)}")
    print(f"\nAcaso esperado (10 classes balanceadas, ~100 janelas/classe): ~{100/10:.1f}%")
    print("Pureza acima do acaso indica que o retrieval discrimina por operacao "
          "(class-level), mesmo sem achar a janela exata (instance-level).")

    saida = {
        "k": args.k,
        "usar_hybrid": not args.sem_hybrid,
        "n_perguntas_avaliadas": len(purezas),
        "n_nao_respondiveis": n_nao_resp,
        "pureza_media": round(media, 3),
        "n_purezas_perfeitas": perfeitas,
        "n_purezas_zeradas": zeradas,
        "acaso_esperado": round(1 / 10, 3),
        "purezas_por_pergunta": [
            {"id": pq["id"], "pureza": round(p, 3)}
            for pq, p in zip([g for g in golden if g["documento_relevante"] is not None], purezas)
        ],
    }
    caminho_saida = EVAL_DIR / "resultados_retrieval_openpack.json"
    caminho_saida.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados salvos em {caminho_saida}")


if __name__ == "__main__":
    main()
