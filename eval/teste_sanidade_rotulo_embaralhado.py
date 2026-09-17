"""
Checagem de aceitacao #2 do item 4a do plano (bug de vazamento de rotulo no OpenPack,
2026-09-15): data randomization test / permutation test (Zhang et al. 2016; Sanity Checks for
Saliency Maps, NeurIPS 2018) -- embaralha os rotulos de TREINO e verifica que o F1-macro desaba
para perto do acaso (~1/n_classes). Se o F1 continuar alto com rotulos embaralhados, ainda ha
vazamento por algum caminho que a correcao do texto nao cobriu.

Reusa classificar_por_vizinhos() de avaliar_classificacao_openpack.py sem duplicar logica --
so a etapa de embaralhamento do dict operacao_por_janela (usado tanto para votar quanto para
avaliar) e nova.

Uso: python eval/teste_sanidade_rotulo_embaralhado.py [--k 5] [--seed 42]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.trace import iniciar_trace, registrar_passo

import pandas as pd
import datasets  # noqa: F401 (import de protecao contra access violation torch x pyarrow)
from sklearn.metrics import f1_score

from avaliar_classificacao_openpack import (
    K_DEFAULT, SPLIT_TREINO, SPLIT_TESTE,
    classificar_por_vizinhos, montar_texto_pergunta_para_janela,
    JANELAS_AMOSTRADAS, CHROMA_DIR, COLECAO,
)

EVAL_DIR = Path(__file__).resolve().parent
HARBOR_ROOT = EVAL_DIR.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=K_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sys.path.insert(0, str(HARBOR_ROOT / "rag"))
    from rag_hibrido_langchain import RAGHibrido

    df_janelas = pd.read_csv(JANELAS_AMOSTRADAS)
    operacao_por_janela_real = dict(zip(df_janelas["janela_id"], df_janelas["operacao"]))
    n_classes = df_janelas["operacao"].nunique()
    acaso_esperado = 1.0 / n_classes

    # Embaralha os rotulos entre as janelas -- mantem o conjunto de rotulos identico (mesma
    # distribuicao de classes), so quebra a associacao janela<->rotulo. Isso e o "keeping the
    # feature matrix fixed" do permutation test: as features de sensor (implicitas no texto de
    # cada janela) continuam as mesmas, so o rotulo usado para votar/avaliar muda de dono.
    random.seed(args.seed)
    janela_ids = list(operacao_por_janela_real.keys())
    rotulos_embaralhados = list(operacao_por_janela_real.values())
    random.shuffle(rotulos_embaralhados)
    operacao_por_janela_embaralhado = dict(zip(janela_ids, rotulos_embaralhados))

    janelas_treino = df_janelas[
        df_janelas[["sujeito", "sessao"]].apply(tuple, axis=1).isin(SPLIT_TREINO)
    ]
    janelas_teste = df_janelas[
        df_janelas[["sujeito", "sessao"]].apply(tuple, axis=1).isin(SPLIT_TESTE)
    ]
    janelas_treino_ids = set(janelas_treino["janela_id"])

    corpus_docs = json.loads((HARBOR_ROOT / "rag" / "corpus_openpack_janelas.json").read_text(encoding="utf-8"))

    rag = RAGHibrido(chroma_dir=CHROMA_DIR, colecao=COLECAO)
    rag.indexar(forcar=False, documentos_customizados=corpus_docs)

    print(f"Teste de sanidade: rotulo embaralhado (seed={args.seed}) sobre split oficial.")
    print(f"n_classes={n_classes}, acaso esperado (macro, classes balanceadas) ~= {acaso_esperado:.4f}")
    print(f"Janelas treino={len(janelas_treino)}, teste={len(janelas_teste)}")

    y_true, y_pred = [], []
    n_sem_vizinho = 0
    t0 = time.time()

    with iniciar_trace("openpack_sanidade_rotulo_embaralhado", entrada=f"seed={args.seed},k={args.k}") as trace:
        for _, janela in janelas_teste.iterrows():
            texto_query = montar_texto_pergunta_para_janela(janela["janela_id"], df_janelas)
            if texto_query is None:
                continue
            predicao = classificar_por_vizinhos(
                rag, texto_query, operacao_por_janela_embaralhado, janelas_treino_ids, k=args.k
            )
            if predicao is None:
                n_sem_vizinho += 1
                continue
            y_true.append(operacao_por_janela_embaralhado[janela["janela_id"]])
            y_pred.append(predicao)
        trace.resultado_final = f"{len(y_true)} classificadas com rotulo embaralhado"
        trace.sucesso = len(y_true) > 0

    duracao = time.time() - t0
    f1_embaralhado = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else None

    print()
    print("=" * 74)
    print("RESULTADO -- teste de sanidade (rotulo embaralhado)")
    print("=" * 74)
    print(f"F1-macro com rotulo embaralhado : {f1_embaralhado:.4f}" if f1_embaralhado is not None else "SEM DADOS")
    print(f"Acaso esperado                  : ~{acaso_esperado:.4f}")
    print(f"Duracao                         : {duracao:.1f}s")
    print()
    if f1_embaralhado is not None and f1_embaralhado > acaso_esperado * 2:
        print("REPROVADO: F1 com rotulo embaralhado esta bem acima do acaso -- AINDA HA "
              "VAZAMENTO por algum caminho alem do texto indexado. Nao promover o corpus "
              "corrigido ate investigar.")
    elif f1_embaralhado is not None:
        print("APROVADO: F1 com rotulo embaralhado esta perto do acaso, como esperado -- "
              "nao ha evidencia de vazamento residual por este teste.")

    resultado = {
        "teste": "sanidade_rotulo_embaralhado",
        "seed": args.seed,
        "k": args.k,
        "n_classes": int(n_classes),
        "acaso_esperado": round(acaso_esperado, 4),
        "f1_macro_embaralhado": round(float(f1_embaralhado), 4) if f1_embaralhado is not None else None,
        "n_classificadas": len(y_true),
        "n_sem_vizinho_treino": n_sem_vizinho,
        "duracao_s": round(duracao, 1),
    }
    caminho_saida = EVAL_DIR / "resultado_sanidade_rotulo_embaralhado.json"
    caminho_saida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultado salvo em {caminho_saida}")


if __name__ == "__main__":
    main()
