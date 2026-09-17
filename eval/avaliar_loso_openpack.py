"""
LOSO (Leave-One-Subject-Out) sobre o OpenPack: protocolo padrao-ouro em HAR para medir
generalizacao do classificador k-NN training-free (RAG-HAR, arXiv:2512.08984) para SUJEITO NOVO
-- nao apenas para janela nova do mesmo sujeito (que e o que o split oficial "Pilot Challenge" ja
mede em eval/avaliar_classificacao_openpack.py). Complementar, nao substituto: os dois protocolos
respondem perguntas diferentes.

MOTIVACAO (2026-09-15): apos confirmar que o dataset OpenPack pre-processado tem 21 sujeitos com
78.667 janelas ja processadas em disco -- e que o split oficial usa so 4 sujeitos (U0102, U0103,
U0105, U0106) -- LOSO e o experimento de maior retorno cientifico disponivel sem baixar nada:
responde a critica mais previsivel de qualquer revisor ("a amostra foi escolhida a favor?").

Reusa (nao duplica) classificar_por_vizinhos(), agregar_por_fonte() e
montar_texto_pergunta_para_janela() de avaliar_classificacao_openpack.py -- a mesma regra do
CLAUDE.md que proibe duplicar logica de roteamento/geracao RAG se aplica aqui: fonte unica de
verdade para o protocolo de classificacao k-NN.

Protocolo: para cada um dos 21 sujeitos presentes na amostra de outputs/pipeline8_openpack/
janelas_amostradas.csv, usa as janelas desse sujeito como TESTE e as janelas de todos os OUTROS
20 sujeitos como base de retrieval (TREINO). Reporta F1-macro por sujeito, media e desvio padrao.

Uso: python eval/avaliar_loso_openpack.py [--k 5]
     Pre-requisito: rag/chroma_db_openpack/ ja indexado (rag/rag_openpack_texto.py),
     outputs/pipeline8_openpack/janelas_amostradas.csv existente.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.trace import iniciar_trace, registrar_passo

import pandas as pd
import datasets  # noqa: F401 (import de protecao contra access violation torch x pyarrow -- ver skill rag-multimodal)
from sklearn.metrics import f1_score

from avaliar_classificacao_openpack import (
    K_DEFAULT,
    BASELINES_OFICIAIS,
    classificar_por_vizinhos,
    montar_texto_pergunta_para_janela,
    JANELAS_AMOSTRADAS,
    CHROMA_DIR,
    COLECAO,
)

EVAL_DIR = Path(__file__).resolve().parent
HARBOR_ROOT = EVAL_DIR.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=K_DEFAULT,
                         help=f"numero de vizinhos de treino considerados na votacao (default {K_DEFAULT})")
    args = parser.parse_args()

    sys.path.insert(0, str(HARBOR_ROOT / "rag"))
    from rag_hibrido_langchain import RAGHibrido

    df_janelas = pd.read_csv(JANELAS_AMOSTRADAS)
    operacao_por_janela = dict(zip(df_janelas["janela_id"], df_janelas["operacao"]))
    sujeitos = sorted(df_janelas["sujeito"].unique())

    print(f"LOSO sobre {len(sujeitos)} sujeitos, {len(df_janelas)} janelas amostradas totais.")
    print(f"Sujeitos: {sujeitos}")
    print()

    corpus_docs = json.loads((HARBOR_ROOT / "rag" / "corpus_openpack_janelas.json").read_text(encoding="utf-8"))

    rag = RAGHibrido(chroma_dir=CHROMA_DIR, colecao=COLECAO)
    t0_indexacao = time.time()
    rag.indexar(forcar=False, documentos_customizados=corpus_docs)
    tempo_indexacao_s = time.time() - t0_indexacao
    print(f"Indice carregado em {tempo_indexacao_s:.1f}s ({len(corpus_docs)} documentos).\n")

    resultados_por_sujeito = {}
    t0_total = time.time()

    with iniciar_trace("openpack_loso", entrada=f"k={args.k},n_sujeitos={len(sujeitos)}",
                        versoes={"corpus": "rag/corpus_openpack_janelas.json"}) as trace:
        for i, sujeito_teste in enumerate(sujeitos, start=1):
            janelas_teste = df_janelas[df_janelas["sujeito"] == sujeito_teste]
            janelas_treino_ids = set(df_janelas.loc[df_janelas["sujeito"] != sujeito_teste, "janela_id"])

            y_true, y_pred = [], []
            n_sem_vizinho = 0
            t0_sujeito = time.time()

            for _, janela in janelas_teste.iterrows():
                texto_query = montar_texto_pergunta_para_janela(janela["janela_id"], df_janelas)
                if texto_query is None:
                    continue
                predicao = classificar_por_vizinhos(
                    rag, texto_query, operacao_por_janela, janelas_treino_ids, k=args.k
                )
                if predicao is None:
                    n_sem_vizinho += 1
                    continue
                y_true.append(janela["operacao"])
                y_pred.append(predicao)

            duracao_sujeito = time.time() - t0_sujeito
            f1_sujeito = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else None

            registrar_passo(
                trace, tipo="retrieval", action="loso_sujeito",
                action_input=sujeito_teste,
                observation=f"f1={f1_sujeito}" if f1_sujeito is not None else "sem_dados",
                duracao_s=round(duracao_sujeito, 2),
            )

            resultados_por_sujeito[sujeito_teste] = {
                "n_janelas_teste": int(len(janelas_teste)),
                "n_classificadas": len(y_true),
                "n_sem_vizinho_treino": n_sem_vizinho,
                "f1_macro": round(float(f1_sujeito), 4) if f1_sujeito is not None else None,
                "duracao_s": round(duracao_sujeito, 1),
            }

            status = f"F1={f1_sujeito:.4f}" if f1_sujeito is not None else "SEM DADOS CLASSIFICAVEIS"
            print(f"[{i:2d}/{len(sujeitos)}] {sujeito_teste}: {status} "
                  f"({len(y_true)}/{len(janelas_teste)} janelas, {duracao_sujeito:.1f}s)")

        f1s_validos = [r["f1_macro"] for r in resultados_por_sujeito.values() if r["f1_macro"] is not None]
        trace.resultado_final = f"{len(f1s_validos)}/{len(sujeitos)} sujeitos classificados"
        trace.sucesso = len(f1s_validos) > 0

    tempo_total_s = time.time() - t0_total

    if not f1s_validos:
        raise RuntimeError("Nenhum sujeito produziu F1-macro valido -- verificar retrieval/corpus.")

    f1_media = sum(f1s_validos) / len(f1s_validos)
    f1_desvio = (sum((f - f1_media) ** 2 for f in f1s_validos) / len(f1s_validos)) ** 0.5
    f1_min_sujeito = min(resultados_por_sujeito.items(), key=lambda kv: kv[1]["f1_macro"] if kv[1]["f1_macro"] is not None else 1.0)
    f1_max_sujeito = max(resultados_por_sujeito.items(), key=lambda kv: kv[1]["f1_macro"] if kv[1]["f1_macro"] is not None else 0.0)

    print()
    print("=" * 74)
    print(f"LOSO -- F1-MACRO POR SUJEITO (k={args.k}, protocolo training-free RAG-HAR)")
    print("=" * 74)
    print(f"Sujeitos classificados : {len(f1s_validos)}/{len(sujeitos)}")
    print(f"F1-macro medio          : {f1_media:.4f}")
    print(f"F1-macro desvio padrao  : {f1_desvio:.4f}")
    print(f"Sujeito mais dificil    : {f1_min_sujeito[0]} (F1={f1_min_sujeito[1]['f1_macro']:.4f})")
    print(f"Sujeito mais facil      : {f1_max_sujeito[0]} (F1={f1_max_sujeito[1]['f1_macro']:.4f})")
    print()
    print("Comparacao com baselines oficiais (split diferente, treinados/supervisionados -- referencia, nao comparacao direta):")
    for nome, f1_oficial in BASELINES_OFICIAIS.items():
        diff = f1_media - f1_oficial
        print(f"  {nome:15} F1={f1_oficial:.4f}  (Harbor LOSO {'supera' if diff > 0 else 'fica abaixo'} em {abs(diff):.4f})")
    print()
    print(f"Tempo total: {tempo_total_s:.1f}s ({tempo_total_s/60:.1f}min)")
    print()
    print("IMPORTANTE: LOSO mede generalizacao para SUJEITO NOVO (protocolo padrao-ouro em HAR).")
    print("O split oficial 'Pilot Challenge' mede generalizacao para SESSAO nova do MESMO conjunto")
    print("de sujeitos vistos -- os dois numeros respondem perguntas diferentes, nao sao substitutos.")

    resultado = {
        "protocolo": "LOSO (Leave-One-Subject-Out)",
        "k": args.k,
        "n_sujeitos": len(sujeitos),
        "n_sujeitos_classificados": len(f1s_validos),
        "n_janelas_totais": int(len(df_janelas)),
        "f1_macro_medio": round(float(f1_media), 4),
        "f1_macro_desvio": round(float(f1_desvio), 4),
        "f1_macro_min": {"sujeito": f1_min_sujeito[0], "f1": f1_min_sujeito[1]["f1_macro"]},
        "f1_macro_max": {"sujeito": f1_max_sujeito[0], "f1": f1_max_sujeito[1]["f1_macro"]},
        "resultados_por_sujeito": resultados_por_sujeito,
        "baselines_oficiais": BASELINES_OFICIAIS,
        "training_free": True,
        "tempo_indexacao_s": round(tempo_indexacao_s, 1),
        "tempo_total_s": round(tempo_total_s, 1),
        "n_documentos_indexados": len(corpus_docs),
    }
    caminho_saida = EVAL_DIR / "resultados_loso_openpack.json"
    caminho_saida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados salvos em {caminho_saida}")


if __name__ == "__main__":
    main()
