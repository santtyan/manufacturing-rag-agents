"""
Benchmark de RAG multimodal em dois estagios (retrieval + rerank multimodal) sobre dois corpora
(subset real do ViDoRe e corpus proprio do Harbor) -- padrao-ouro/estado da arte 2026 aplicado,
com metodologia oficial (nDCG@k) em vez de so Recall@k simples.

Motivacao e redesenho (2026-09-09): a primeira versao deste script rodava late interaction
(ColModernVBERT) sobre o CORPUS INTEIRO como paradigma independente -- 36s/imagem em CPU,
inviavel para 60 paginas sem GPU/ONNX (ONNX bloqueado por incompatibilidade real de dependencias,
ver skill rag-multimodal). Pesquisa mais profunda revelou que essa nunca foi a forma de producao
de usar late interaction: o paper HEAVEN ("Hybrid-Vector Retrieval for Visually Rich Documents",
arXiv:2510.22215) mostra que sistemas reais usam DOIS ESTAGIOS -- um encoder single-vector barato
(o que ja temos: caption-then-embed com E5) filtra candidatos sobre TODO o corpus, e o modelo
multi-vetor caro (ColModernVBERT) so faz RERANK dos top-k/top-p candidatos ja filtrados, nunca do
corpus inteiro. E o mesmo padrao ja usado no RAG de texto do Harbor (usar_rerank=True/False com
Cross-Encoder) -- aqui aplicado ao rerank multimodal.

Duas estrategias de selecao de candidatos para o 2o estagio, comparadas:
- top-k fixo: sempre os k=5 melhores do 1o estagio entram no rerank multimodal.
- "top-p" (corte por massa cumulativa de score, analogo ao nucleus sampling de geracao de
  texto aplicado a scores de similaridade em vez de probabilidade de token): inclui candidatos
  enquanto a soma normalizada dos scores nao ultrapassar um limiar -- adapta o numero de
  candidatos ao qua confiante o 1o estagio esta (poucos candidatos quando ha 1 resultado claro,
  mais candidatos quando os scores estao proximos/empatados).

O padrao-ouro de avaliacao e ViDoRe (v3, ACL 2026) e, para RAG ponta-a-ponta, UNIDOC-BENCH
(arXiv:2510.03663) -- comparacao sob protocolo unificado (mesmo pool de candidatos, mesmas
queries, mesmas metricas) e o que diferencia um benchmark real de uma medicao solta.

Restricao de ambiente (ver skill rag-multimodal): ViDoRe v3 completo e ate o subset industrial
menor sao grandes demais para o disco disponivel; sem GPU CUDA, ColQwen2.5 (~3B) e inviavel em
CPU; ONNX via optimum-onnx exige transformers<4.58, incompativel com sentence-transformers 6.x
que a stack do Harbor usa -- nao vale o risco de rebaixar dependencias criticas de producao.
Usamos: subset real do ViDoRe (60 paginas, eval/vidore_subset/, via streaming de
vidore/tabfquad_test_subsampled) e ColModernVBERT (250M parametros, ~10x menor que ColPali) como
reranker, nao como indexador do corpus inteiro -- o desenho em dois estagios torna isso viavel em
CPU puro sem nenhuma otimizacao adicional.

Uso: python eval/avaliar_benchmark_multimodal_2x2.py
     Pre-requisito: eval/vidore_subset/ ja baixado e rag/legendas_cache.json ja gerado.
"""
import json
import math
import statistics
import sys
import time
from pathlib import Path

import pandas  # ACHADO REAL: import antes de sentence_transformers evita access violation (ver skill rag-multimodal)

EVAL_DIR = Path(__file__).resolve().parent
HARBOR_ROOT = EVAL_DIR.parent
VIDORE_SUBSET = EVAL_DIR / "vidore_subset"
VIDORE_METADADOS = VIDORE_SUBSET / "metadados.json"
VIDORE_IMAGENS = VIDORE_SUBSET / "imagens"

CACHE_LEGENDAS_HARBOR = HARBOR_ROOT / "rag" / "legendas_cache.json"
IMAGENS_HARBOR = HARBOR_ROOT / "rag" / "manuais_imagens"
GOLDEN_HARBOR = EVAL_DIR / "golden_questions_multimodal.json"

K_FINAL = 5  # k avaliado no ranking final (nDCG@5/Recall@5) -- k=10 do ViDoRe oficial nao faz
# sentido com corpus de 60 ou 4 documentos (V1/V2 do ViDoRe tambem usam k=5; k=10 so entrou na V3)
K_CANDIDATOS_1O_ESTAGIO = 10  # quantos candidatos o 1o estagio (caption-then-embed) recupera,
# antes de qualquer corte para o 2o estagio (rerank multimodal)
TOP_K_RERANK = 5  # variante top-k fixo: quantos desses candidatos vao para o rerank caro
TOP_P_LIMIAR = 0.85  # variante top-p: inclui candidatos ate a massa cumulativa de score
# normalizado atingir este limiar -- adapta o numero de candidatos a confianca do 1o estagio


def ndcg_at_k(relevancias_ordenadas, k=K_FINAL):
    relevancias_ordenadas = relevancias_ordenadas[:k]
    dcg = sum(rel / (i + 1 if i == 0 else math.log2(i + 1)) for i, rel in enumerate(relevancias_ordenadas))
    ideal = sorted(relevancias_ordenadas, reverse=True)
    idcg = sum(rel / (i + 1 if i == 0 else math.log2(i + 1)) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(relevancias_ordenadas, k=K_FINAL):
    return 1.0 if sum(relevancias_ordenadas[:k]) > 0 else 0.0


def reciprocal_rank(relevancias_ordenadas):
    for i, rel in enumerate(relevancias_ordenadas, start=1):
        if rel > 0:
            return 1.0 / i
    return 0.0


def selecionar_top_p(candidatos_com_score, limiar=None):
    """Corte por massa cumulativa de score normalizado -- analogo de nucleus sampling (top-p de
    geracao de texto) aplicado a scores de similaridade de retrieval em vez de probabilidade de
    token. `candidatos_com_score`: lista de (id, score) ja ordenada por score decrescente, com
    score >= 0 (scores negativos sao clipados a 0 antes de normalizar).

    limiar (calibracao, item 4 da rodada de fechamento da fila de trabalho, 2026-09-09): default
    None usa TOP_P_LIMIAR do modulo, mas pode ser sobrescrito por chamada -- necessario porque o
    achado real do benchmark 2x2 mostrou que o limiar fixo 0,85 funciona bem no ViDoRe (corpus
    maior) mas piora MUITO no corpus Harbor (so 4 imagens, scores concentrados)."""
    if limiar is None:
        limiar = TOP_P_LIMIAR
    scores = [max(s, 0.0) for _, s in candidatos_com_score]
    total = sum(scores)
    if total == 0:
        return candidatos_com_score[:1]  # sem sinal nenhum, ao menos 1 candidato segue
    acumulado = 0.0
    selecionados = []
    for item, score in zip(candidatos_com_score, scores):
        selecionados.append(item)
        acumulado += score / total
        if acumulado >= limiar:
            break
    return selecionados


# ── Estagio 1: retrieval barato (caption-then-embed, producao atual) ────────────────────────
def estagio1_retrieval(corpus_docs, perguntas, chroma_dir, colecao, k_candidatos=K_CANDIDATOS_1O_ESTAGIO):
    """Retorna, por pergunta, a lista ordenada de (doc_id, score) recuperada pelo RAG hibrido
    (E5 + TF-IDF/BM25, sem rerank) -- o "encoder single-vector barato" do desenho HEAVEN.

    usar_score_rrf=True (item 2 do plano "Evoluir o RAG multimodal", 2026-09-09): ACHADO REAL
    corrigido -- sem essa flag, buscar() sempre devolvia score=0.0 aqui (EnsembleRetriever/RRF
    do LangChain nao expoe score quando usar_rerank=False), fazendo selecionar_top_p() sempre
    cair no caso "sem sinal, 1 candidato" independente de TOP_P_LIMIAR. Ver rag_hibrido_langchain.py
    para a implementacao do RRF explicito."""
    sys.path.insert(0, str(HARBOR_ROOT / "rag"))
    from rag_hibrido_langchain import RAGHibrido

    rag = RAGHibrido(chroma_dir=chroma_dir, colecao=colecao)
    t0 = time.time()
    rag.indexar(forcar=True, documentos_customizados=corpus_docs)
    tempo_indexacao = time.time() - t0

    resultados_por_pergunta = []
    for pergunta in perguntas:
        candidatos = rag.buscar(pergunta, k=k_candidatos, usar_rerank=False, usar_hybrid=True,
                                 k_candidatos=k_candidatos, usar_score_rrf=True)
        pares = [(c.get("id") or c.get("fonte", "").rsplit(".", 1)[0], c.get("score", 0.0)) for c in candidatos]
        resultados_por_pergunta.append(pares)
    return resultados_por_pergunta, tempo_indexacao


# ── Estagio 2: rerank caro (ColModernVBERT), so sobre os candidatos filtrados ───────────────
def estagio2_rerank(candidatos_estagio1, imagens_por_id, perguntas, estrategia="top_k", top_p_limiar=None):
    """candidatos_estagio1: lista paralela a `perguntas`, cada item e a lista de (doc_id, score)
    do estagio 1 para aquela pergunta. imagens_por_id: dict doc_id -> Path da imagem.

    Reranqueia SO os candidatos selecionados por `estrategia` (top_k ou top_p) via
    ColModernVBERT -- nunca o corpus inteiro. E o proprio desenho que torna late interaction
    viavel em CPU sem GPU/ONNX, seguindo o padrao HEAVEN (retrieval barato filtra, rerank caro
    so opera no que sobrou)."""
    from sentence_transformers import MultiVectorEncoder
    from PIL import Image

    modelo = MultiVectorEncoder("ModernVBERT/colmodernvbert")

    tempo_total_rerank = 0.0
    n_candidatos_total = 0
    rankings_finais = []
    n_candidatos_por_pergunta = []  # item 4 do plano (2026-09-09): log explicito por pergunta --
    # a ausencia desse log foi o que deixou o bug de score=0.0 passar despercebido atraves de
    # 3 valores de TOP_P_LIMIAR testados sem ninguem notar que o numero de candidatos nunca mudava.

    for candidatos, pergunta in zip(candidatos_estagio1, perguntas):
        if estrategia == "top_k":
            selecionados = candidatos[:TOP_K_RERANK]
        elif estrategia == "top_p":
            selecionados = selecionar_top_p(candidatos, limiar=top_p_limiar)
        else:
            raise ValueError(f"estrategia desconhecida: {estrategia}")

        ids_selecionados = [doc_id for doc_id, _ in selecionados]
        n_candidatos_total += len(ids_selecionados)
        n_candidatos_por_pergunta.append(len(ids_selecionados))

        t0 = time.time()
        imagens = [Image.open(imagens_por_id[doc_id]).convert("RGB") for doc_id in ids_selecionados]
        doc_embeddings = modelo.encode_document(imagens)
        query_embedding = modelo.encode_query([pergunta])
        scores = modelo.similarity(query_embedding, doc_embeddings)[0].tolist()
        tempo_total_rerank += time.time() - t0

        ranking = sorted(zip(ids_selecionados, scores), key=lambda x: -x[1])
        rankings_finais.append([doc_id for doc_id, _ in ranking])

    if estrategia == "top_p":
        print(f"  n_candidatos_selecionados por pergunta (top_p, limiar={top_p_limiar or TOP_P_LIMIAR}): "
              f"{n_candidatos_por_pergunta} (media={statistics.mean(n_candidatos_por_pergunta):.1f})")

    return rankings_finais, tempo_total_rerank, n_candidatos_total


def calcular_metricas(rankings, alvos):
    metricas = []
    for ranking, alvo in zip(rankings, alvos):
        relevancias = [1 if doc_id == alvo else 0 for doc_id in ranking]
        metricas.append({
            "ndcg": ndcg_at_k(relevancias), "recall": recall_at_k(relevancias), "rr": reciprocal_rank(relevancias),
        })
    return metricas


def resumir(nome, metricas, tempo_s, n_unidades, unidade="documento(s)"):
    ndcg = statistics.mean(m["ndcg"] for m in metricas)
    recall = statistics.mean(m["recall"] for m in metricas)
    mrr = statistics.mean(m["rr"] for m in metricas)
    print(f"\n[{nome}]")
    print(f"  nDCG@{K_FINAL}   : {ndcg:.3f}")
    print(f"  Recall@{K_FINAL} : {recall*100:.0f}%")
    print(f"  MRR       : {mrr:.3f}")
    print(f"  Custo     : {tempo_s:.1f}s para {n_unidades} {unidade} ({tempo_s/max(n_unidades,1):.2f}s/{unidade.rstrip('(s)')})")
    return {"nome": nome, "ndcg": round(ndcg, 3), "recall": round(recall, 3), "mrr": round(mrr, 3),
            "tempo_s": round(tempo_s, 1), "n_unidades": n_unidades}


def carregar_corpus_vidore():
    if not VIDORE_METADADOS.exists():
        return None
    metadados = json.loads(VIDORE_METADADOS.read_text(encoding="utf-8"))
    # Corpus de captions do proprio GPT4 (ja vem no dataset) -- mede a ARQUITETURA de fusao
    # retrieval+rerank, nao a qualidade especifica de um VLM local de captioning.
    corpus_docs = [{"id": m["id"], "texto": m["caption_gpt4"], "fonte": m["image_filename"]} for m in metadados]
    imagens_por_id = {m["id"]: VIDORE_IMAGENS / m["image_filename"] for m in metadados}
    perguntas = [m["query"] for m in metadados]
    alvos = [m["id"] for m in metadados]  # cada query so tem 1 documento relevante (o proprio par)
    return corpus_docs, imagens_por_id, perguntas, alvos


def carregar_corpus_harbor():
    if not CACHE_LEGENDAS_HARBOR.exists() or not GOLDEN_HARBOR.exists():
        return None
    corpus_docs = json.loads(CACHE_LEGENDAS_HARBOR.read_text(encoding="utf-8"))
    golden = json.loads(GOLDEN_HARBOR.read_text(encoding="utf-8"))["perguntas"]
    imagens_por_id = {d["id"]: IMAGENS_HARBOR / d["fonte"] for d in corpus_docs}
    perguntas = [pq["pergunta"] for pq in golden]
    alvos = [pq["documento_relevante"] for pq in golden]
    return corpus_docs, imagens_por_id, perguntas, alvos


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-queries-vidore", type=int, default=None,
                         help="limitar quantas queries do ViDoRe avaliar (custo do rerank multimodal "
                              "escala linear com isso, ~36s por imagem rerankeada em CPU); default: todas")
    parser.add_argument("--top-p-limiar", type=float, default=None,
                         help="sobrescreve TOP_P_LIMIAR (0,85 default) -- calibracao por tamanho de "
                              "corpus, ver achado real na skill rag-multimodal")
    parser.add_argument("--so-harbor", action="store_true",
                         help="pula o corpus ViDoRe (caro, ~36s/imagem) -- so roda o corpus proprio "
                              "do Harbor (4 imagens, rapido), util para calibrar top_p_limiar")
    args = parser.parse_args()

    resultados_finais = []

    print("=" * 74)
    print("BENCHMARK RAG MULTIMODAL EM 2 ESTAGIOS -- retrieval (E5) + rerank (ColModernVBERT)")
    print("=" * 74)

    corpus_vidore = carregar_corpus_vidore()
    if corpus_vidore is not None and args.n_queries_vidore is not None:
        corpus_docs, imagens_por_id, perguntas, alvos = corpus_vidore
        n = args.n_queries_vidore
        corpus_vidore = (corpus_docs, imagens_por_id, perguntas[:n], alvos[:n])

    corpora = [("Corpus Harbor (4 img.)", carregar_corpus_harbor(), HARBOR_ROOT / "rag" / "chroma_db_multimodal", "imagens_harbor_multimodal_v1")]
    if not args.so_harbor:
        corpora.insert(0, ("ViDoRe subset (60 pag.)", corpus_vidore, EVAL_DIR / "chroma_db_vidore_subset", "vidore_subset_v1"))

    for nome_corpus, dados_corpus, chroma_dir, colecao in corpora:
        if dados_corpus is None:
            print(f"\n[{nome_corpus}] pulado -- dados nao encontrados (ver docstring do script).")
            continue
        corpus_docs, imagens_por_id, perguntas, alvos = dados_corpus
        n_docs = len(corpus_docs)
        print(f"\n--- Corpus: {nome_corpus} ({n_docs} documentos, {len(perguntas)} queries) ---")

        # Estagio 1 sozinho (baseline sem rerank multimodal -- e o que ja media a Fase 4 antiga)
        candidatos_por_pergunta, tempo_indexacao = estagio1_retrieval(corpus_docs, perguntas, chroma_dir, colecao)
        rankings_estagio1 = [[doc_id for doc_id, _ in c] for c in candidatos_por_pergunta]
        m0 = calcular_metricas(rankings_estagio1, alvos)
        resultados_finais.append(resumir(f"{nome_corpus} | so retrieval (sem rerank)", m0, tempo_indexacao, n_docs))

        # Estagio 1+2 com top-k fixo
        rankings_topk, tempo_topk, n_cand_topk = estagio2_rerank(candidatos_por_pergunta, imagens_por_id, perguntas, estrategia="top_k")
        m1 = calcular_metricas(rankings_topk, alvos)
        resultados_finais.append(resumir(f"{nome_corpus} | rerank top-k={TOP_K_RERANK}", m1, tempo_topk, n_cand_topk, unidade="imagem(ns) rerankeada(s)"))

        # Estagio 1+2 com top-p (corte por massa cumulativa de score)
        limiar_efetivo = args.top_p_limiar if args.top_p_limiar is not None else TOP_P_LIMIAR
        rankings_topp, tempo_topp, n_cand_topp = estagio2_rerank(candidatos_por_pergunta, imagens_por_id, perguntas, estrategia="top_p", top_p_limiar=args.top_p_limiar)
        m2 = calcular_metricas(rankings_topp, alvos)
        resultados_finais.append(resumir(f"{nome_corpus} | rerank top-p={limiar_efetivo}", m2, tempo_topp, n_cand_topp, unidade="imagem(ns) rerankeada(s)"))

    print("\n" + "=" * 74)
    print("RESUMO COMPARATIVO")
    print("=" * 74)
    for r in resultados_finais:
        print(f"{r['nome']:48} nDCG@{K_FINAL}={r['ndcg']:.3f}  Recall@{K_FINAL}={r['recall']*100:.0f}%  "
              f"MRR={r['mrr']:.3f}  custo={r['tempo_s']}s")

    saida = EVAL_DIR / "resultados_benchmark_multimodal_2x2.json"
    saida.write_text(json.dumps(resultados_finais, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultados salvos em {saida}")
    print("\nRegra do projeto: decisao de arquitetura so depois deste relatorio existir -- "
          "nao trocar antes de medir (ver skill rag-multimodal).")


if __name__ == "__main__":
    main()
