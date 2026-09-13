"""
Fase 7a do plano de integracao OpenPack: classificacao F1-macro comparavel ao benchmark
OFICIAL do dataset (openpack-torch, split "Pilot Challenge") -- fecha a lacuna real de nao
termos testado contra nenhum benchmark padrao-ouro do dominio HAR (a Fase 4c media retrieval
via k-NN label purity, um proxy, nao a metrica de classificacao que o dataset oficialmente usa).

ACHADO REAL (2026-09-13): o RAG-HAR (arXiv:2512.08984) nunca foi validado no OpenPack pelos
proprios autores -- ele usa 6 outros benchmarks (HHAR, PAMAP2, MHEALTH, GOTOV, SKODA, USC-HAD).
O benchmark oficial do OpenPack (openpack-torch) reporta F1-macro de classificacao supervisionada
(UNet=0.3451, ST-GCN=0.7024, DeepConvLSTM=0.7081, split "Pilot Challenge"). Este script aplica
o protocolo k-NN classifier do proprio RAG-HAR (retrieval de vizinhos + votacao majoritaria,
sem nenhum treino) sobre o MESMO split oficial, tornando a comparacao legitima.

Split oficial confirmado em openpack_toolkit/configs/datasets/splits.py::PILOT_CHALLENGE_SPLIT:
  train = U0102(S0100-S0500) + U0103(S0100-S0500) + U0105(S0100-S0500)
  val   = U0106(S0200, S0400)
  test  = U0106(S0100, S0300, S0500)
Os 4 sujeitos estao presentes na amostra de 1.000 janelas ja indexada (rag/corpus_openpack_janelas.json) --
nao precisa reindexar nada. train = base de retrieval; test = janelas classificadas.

Uso: python eval/avaliar_classificacao_openpack.py [--k 5]
     Pre-requisito: rag/chroma_db_openpack/ ja indexado (rag/rag_openpack_texto.py),
     outputs/pipeline8_openpack/janelas_amostradas.csv existente.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import datasets  # noqa: F401 (import de protecao contra access violation torch x pyarrow -- ver skill rag-multimodal)
from sklearn.metrics import f1_score

EVAL_DIR = Path(__file__).resolve().parent
HARBOR_ROOT = EVAL_DIR.parent

JANELAS_AMOSTRADAS = HARBOR_ROOT / "outputs" / "pipeline8_openpack" / "janelas_amostradas.csv"
CHROMA_DIR = HARBOR_ROOT / "rag" / "chroma_db_openpack"
COLECAO = "openpack_janelas_v1"

K_DEFAULT = 5

# Split oficial "Pilot Challenge" -- ver docstring do modulo.
SPLIT_TREINO = [
    ("U0102", s) for s in ("S0100", "S0200", "S0300", "S0400", "S0500")
] + [
    ("U0103", s) for s in ("S0100", "S0200", "S0300", "S0400", "S0500")
] + [
    ("U0105", s) for s in ("S0100", "S0200", "S0300", "S0400", "S0500")
]
SPLIT_TESTE = [("U0106", s) for s in ("S0100", "S0300", "S0500")]

# Baselines oficiais publicados em openpack-torch (README, split "Pilot Challenge", test set).
BASELINES_OFICIAIS = {
    "UNet": 0.3451,
    "ST-GCN": 0.7024,
    "DeepConvLSTM": 0.7081,
}


def agregar_por_fonte(candidatos):
    """Colapsa os 4 sub-vetores (completo/inicio/meio/fim) numa entrada por janela_id, mantendo
    a ordem de 1a aparicao (ja por score decrescente) -- mesma logica de
    eval/avaliar_retrieval_openpack.py::agregar_por_fonte(), duplicada aqui por ser um script
    standalone independente."""
    vistos = []
    for c in candidatos:
        doc_id = c.get("id") or c.get("fonte", "")
        fonte = doc_id.rsplit("__", 1)[0] if "__" in doc_id else doc_id
        if fonte not in vistos:
            vistos.append(fonte)
    return vistos


def classificar_por_vizinhos(rag, pergunta, operacao_por_janela, janelas_treino_ids, k=K_DEFAULT):
    """k-NN classifier training-free (protocolo RAG-HAR, sem a etapa de prompt-optimization):
    recupera vizinhos, restringe aos que pertencem ao conjunto de TREINO (nunca usa o proprio
    conjunto de teste como base de retrieval -- vazamento de dado), agrega por fonte, e decide
    a operacao por VOTACAO MAJORITARIA entre os k primeiros vizinhos de treino."""
    candidatos = rag.buscar(pergunta, k=k + 20, usar_rerank=False, usar_hybrid=True, usar_score_rrf=True)
    vizinhos = agregar_por_fonte(candidatos)
    vizinhos_treino = [v for v in vizinhos if v in janelas_treino_ids][:k]

    if not vizinhos_treino:
        return None  # sem vizinho de treino disponivel -- classificacao indefinida

    votos = {}
    for v in vizinhos_treino:
        op = operacao_por_janela.get(v)
        if op is not None:
            votos[op] = votos.get(op, 0) + 1
    if not votos:
        return None
    return max(votos.items(), key=lambda item: item[1])[0]


def montar_texto_pergunta_para_janela(janela_id, df_janelas):
    """Para classificar uma janela de TESTE, a 'pergunta' de retrieval e o proprio perfil
    morfologico dela -- reusa o texto ja gerado em rag/corpus_openpack_janelas.json (segmento
    completo) em vez de gerar uma pergunta em linguagem natural nova, ja que o protocolo
    k-NN do RAG-HAR usa o vetor da propria amostra de teste como query, nao uma pergunta."""
    corpus = json.loads((HARBOR_ROOT / "rag" / "corpus_openpack_janelas.json").read_text(encoding="utf-8"))
    for doc in corpus:
        if doc["id"] == f"{janela_id}__completo":
            return doc["texto"]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=K_DEFAULT,
                         help=f"numero de vizinhos de treino considerados na votacao (default {K_DEFAULT})")
    args = parser.parse_args()

    sys.path.insert(0, str(HARBOR_ROOT / "rag"))
    from rag_hibrido_langchain import RAGHibrido

    df_janelas = pd.read_csv(JANELAS_AMOSTRADAS)
    operacao_por_janela = dict(zip(df_janelas["janela_id"], df_janelas["operacao"]))

    janelas_treino = df_janelas[
        df_janelas[["sujeito", "sessao"]].apply(tuple, axis=1).isin(SPLIT_TREINO)
    ]
    janelas_teste = df_janelas[
        df_janelas[["sujeito", "sessao"]].apply(tuple, axis=1).isin(SPLIT_TESTE)
    ]
    janelas_treino_ids = set(janelas_treino["janela_id"])

    print(f"Split oficial 'Pilot Challenge': {len(janelas_treino)} janelas de treino "
          f"(U0102/U0103/U0105), {len(janelas_teste)} janelas de teste (U0106).")

    if len(janelas_teste) == 0 or len(janelas_treino) == 0:
        raise RuntimeError(
            "Split oficial sem janelas suficientes na amostra atual -- "
            "verificar se outputs/pipeline8_openpack/janelas_amostradas.csv mudou."
        )

    corpus_docs = json.loads((HARBOR_ROOT / "rag" / "corpus_openpack_janelas.json").read_text(encoding="utf-8"))

    rag = RAGHibrido(chroma_dir=CHROMA_DIR, colecao=COLECAO)
    t0_indexacao = time.time()
    rag.indexar(forcar=False, documentos_customizados=corpus_docs)
    tempo_indexacao_s = time.time() - t0_indexacao

    y_true, y_pred = [], []
    tempos_busca = []
    n_sem_vizinho_treino = 0

    for _, janela in janelas_teste.iterrows():
        texto_query = montar_texto_pergunta_para_janela(janela["janela_id"], df_janelas)
        if texto_query is None:
            continue

        t0_busca = time.time()
        predicao = classificar_por_vizinhos(rag, texto_query, operacao_por_janela, janelas_treino_ids, k=args.k)
        tempos_busca.append(time.time() - t0_busca)

        if predicao is None:
            n_sem_vizinho_treino += 1
            continue

        y_true.append(janela["operacao"])
        y_pred.append(predicao)

    if not y_true:
        raise RuntimeError("Nenhuma janela de teste classificada -- verificar retrieval/split.")

    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    tempo_medio_busca_s = sum(tempos_busca) / len(tempos_busca) if tempos_busca else 0.0
    tempo_total_s = tempo_indexacao_s + sum(tempos_busca)

    print("=" * 74)
    print(f"CLASSIFICACAO F1-MACRO -- split oficial 'Pilot Challenge' (k={args.k})")
    print("=" * 74)
    print(f"Janelas de teste classificadas : {len(y_true)}/{len(janelas_teste)} "
          f"({n_sem_vizinho_treino} sem vizinho de treino disponivel)")
    print(f"F1-macro (Harbor/RAG-HAR)      : {f1_macro:.4f}")
    print()
    print("Comparacao com baselines oficiais (openpack-torch, mesmo split, treinados/supervisionados):")
    for nome, f1_oficial in BASELINES_OFICIAIS.items():
        diff = f1_macro - f1_oficial
        print(f"  {nome:15} F1={f1_oficial:.4f}  (Harbor {'supera' if diff > 0 else 'fica abaixo'} em {abs(diff):.4f})")
    print()
    print(f"Tempo de indexacao      : {tempo_indexacao_s:.1f}s ({len(corpus_docs)} documentos)")
    print(f"Tempo medio de busca    : {tempo_medio_busca_s:.3f}s/janela")
    print(f"Tempo total             : {tempo_total_s:.1f}s")
    print()
    print("IMPORTANTE: RAG-HAR e training-free -- os baselines oficiais SAO treinados/supervisionados.")
    print("O valor da comparacao e situar o Harbor no mapa, nao garantir superioridade.")

    resultado = {
        "k": args.k,
        "split": "pilot-challenge",
        "n_janelas_treino": int(len(janelas_treino)),
        "n_janelas_teste": int(len(janelas_teste)),
        "n_classificadas": len(y_true),
        "n_sem_vizinho_treino": n_sem_vizinho_treino,
        "f1_macro": round(float(f1_macro), 4),
        "baselines_oficiais": BASELINES_OFICIAIS,
        "training_free": True,
        "tempo_indexacao_s": round(tempo_indexacao_s, 1),
        "tempo_medio_busca_s": round(tempo_medio_busca_s, 3),
        "tempo_total_s": round(tempo_total_s, 1),
        "n_documentos_indexados": len(corpus_docs),
    }
    caminho_saida = EVAL_DIR / "resultados_classificacao_openpack.json"
    caminho_saida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados salvos em {caminho_saida}")


if __name__ == "__main__":
    main()
