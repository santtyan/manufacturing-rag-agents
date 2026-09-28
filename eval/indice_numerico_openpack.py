"""
Fase 1 do plano "OpenPack: corrigir o indice antes de trocar de modalidade" (2026-09-28).

Testa a hipotese levantada por RAG-HAR+ (arXiv:2607.26631): "numerical vector retrieval is more
suitable than text-embedding retrieval for sensor-feature matching". O indice de producao hoje
(rag/rag_openpack_texto.py -> RAGHibrido) casa JANELAS DE SENSOR por texto->embedding E5. Este
modulo substitui SO essa etapa por kNN euclidiano sobre o vetor numerico de features -- mantendo
tudo o resto identico (mesmas 8 features x 6 canais de rag_openpack_texto.py, mesmo split oficial,
mesma metrica F1-macro) para que a comparacao com eval/avaliar_classificacao_openpack.py e
eval/avaliar_loso_openpack.py seja valida.

NAO reimplementa a extracao de features -- reusa features_todos_canais() de
rag/rag_openpack_texto.py (fonte unica), so muda o que acontece DEPOIS: em vez de
montar_texto_janela() + indexacao textual, achata as 8x6=48 features do segmento "completo" num
vetor numpy, normaliza por z-score (media/desvio SO do conjunto de treino, nunca do teste -- mesmo
cuidado que a apresentacao de Carlos Daniel, CERISE 25/08/2026, aplica) e classifica por distancia
euclidiana + voto majoritario ponderado pelo inverso da distancia.

Uso: python eval/indice_numerico_openpack.py [--k 5] [--loso]
     Pre-requisito: outputs/pipeline8_openpack/janelas_amostradas.csv existente (mesmo de
     avaliar_classificacao_openpack.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

HARBOR_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HARBOR_ROOT))
sys.path.insert(0, str(HARBOR_ROOT / "pipelines"))
sys.path.insert(0, str(HARBOR_ROOT / "rag"))

from pipeline8_openpack import carregar_sessao  # noqa: E402
from rag_openpack_texto import CANAIS, NOMES_FEATURES, features_todos_canais  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
JANELAS_AMOSTRADAS = HARBOR_ROOT / "outputs" / "pipeline8_openpack" / "janelas_amostradas.csv"

# Mesmos baselines oficiais citados em eval/avaliar_classificacao_openpack.py -- ver correcao de
# modalidade em CLAUDE.md (2026-09-28): UNet/DeepConvLSTM sao IMU, ST-GCN e keypoints/pose.
BASELINES_OFICIAIS = {"UNet": 0.3451, "ST-GCN": 0.7024, "DeepConvLSTM": 0.7081}

SPLIT_TREINO = [
    ("U0102", s) for s in ("S0100", "S0200", "S0300", "S0400", "S0500")
] + [
    ("U0103", s) for s in ("S0100", "S0200", "S0300", "S0400", "S0500")
] + [
    ("U0105", s) for s in ("S0100", "S0200", "S0300", "S0400", "S0500")
]
SPLIT_TESTE = [("U0106", s) for s in ("S0100", "S0300", "S0500")]


def vetor_da_janela(bloco: pd.DataFrame) -> np.ndarray:
    """Achata as 8 features x 6 canais (features_todos_canais(), reusada de rag_openpack_texto.py)
    num vetor numpy de 48 dimensoes, ordem fixa (CANAIS x NOMES_FEATURES) -- a mesma ordem sempre,
    para que a distancia euclidiana entre dois vetores compare o mesmo indice a mesma feature."""
    feats = features_todos_canais(bloco)
    return np.array([feats[canal][nome] for canal in CANAIS for nome in NOMES_FEATURES], dtype=float)


def construir_matriz(df_janelas: pd.DataFrame, sinais_por_sessao: dict) -> tuple[np.ndarray, list[str], list[str]]:
    """Constroi a matriz (n_janelas, 48) na mesma ordem de df_janelas. Retorna tambem a lista de
    operacao (rotulo) e janela_id paralela, para reconstruir y_true/y_pred depois do kNN."""
    vetores, operacoes, ids = [], [], []
    for _, janela in df_janelas.iterrows():
        df_sinais = sinais_por_sessao[(janela["sujeito"], janela["sessao"])]
        bloco = df_sinais.iloc[janela["indice_inicio"]: janela["indice_fim"]].reset_index(drop=True)
        vetores.append(vetor_da_janela(bloco))
        operacoes.append(janela["operacao"])
        ids.append(janela["janela_id"])
    return np.vstack(vetores), operacoes, ids


def classificar_knn(
    vetores_treino: np.ndarray, operacoes_treino: list[str],
    vetores_teste: np.ndarray, k: int = 5,
) -> list[str | None]:
    """kNN euclidiano + voto majoritario ponderado pelo inverso da distancia (par a par com
    classificar_por_vizinhos() de avaliar_classificacao_openpack.py, que usa RAGHibrido.buscar()
    + voto majoritario simples entre os k primeiros -- aqui o peso 1/distancia desempata casos
    onde o vizinho mais proximo esta bem mais perto que os demais, informacao que o retrieval
    textual via RRF nao expõe de forma comparavel)."""
    predicoes = []
    for v_teste in vetores_teste:
        distancias = np.linalg.norm(vetores_treino - v_teste, axis=1)
        idx_k = np.argsort(distancias)[:k]
        if len(idx_k) == 0:
            predicoes.append(None)
            continue
        votos: dict[str, float] = {}
        for idx in idx_k:
            peso = 1.0 / (distancias[idx] + 1e-9)
            op = operacoes_treino[idx]
            votos[op] = votos.get(op, 0.0) + peso
        predicoes.append(max(votos.items(), key=lambda item: item[1])[0])
    return predicoes


def carregar_sinais(df_janelas: pd.DataFrame) -> dict:
    pares_sessao = df_janelas[["sujeito", "sessao"]].drop_duplicates().itertuples(index=False)
    sinais = {}
    for sujeito, sessao in pares_sessao:
        sinais[(sujeito, sessao)] = carregar_sessao(sujeito, sessao, usar_sample=False)
    return sinais


def avaliar_split_oficial(k: int) -> dict:
    """Reproduz exatamente o split 'Pilot Challenge' de avaliar_classificacao_openpack.py, trocando
    so retrieval textual por kNN numerico -- unica variavel alterada na comparacao."""
    df_janelas = pd.read_csv(JANELAS_AMOSTRADAS)
    janelas_treino = df_janelas[df_janelas[["sujeito", "sessao"]].apply(tuple, axis=1).isin(SPLIT_TREINO)]
    janelas_teste = df_janelas[df_janelas[["sujeito", "sessao"]].apply(tuple, axis=1).isin(SPLIT_TESTE)]

    if len(janelas_teste) == 0 or len(janelas_treino) == 0:
        raise RuntimeError(
            "Split oficial sem janelas suficientes na amostra atual -- "
            "verificar se outputs/pipeline8_openpack/janelas_amostradas.csv mudou."
        )

    print(f"Split oficial 'Pilot Challenge': {len(janelas_treino)} janelas de treino "
          f"(U0102/U0103/U0105), {len(janelas_teste)} janelas de teste (U0106).")

    sinais = carregar_sinais(pd.concat([janelas_treino, janelas_teste]))

    t0 = time.time()
    X_treino, y_treino, _ = construir_matriz(janelas_treino, sinais)
    X_teste, y_teste, _ = construir_matriz(janelas_teste, sinais)
    tempo_extracao_s = time.time() - t0

    # Normalizacao z-score com media/desvio SO do treino -- nunca do teste (mesmo cuidado da
    # apresentacao Carlos Daniel/CERISE: "impede que o teste influencie a preparacao dos dados").
    media_treino = X_treino.mean(axis=0)
    desvio_treino = X_treino.std(axis=0)
    desvio_treino[desvio_treino == 0] = 1.0  # evita divisao por zero em feature constante
    X_treino_norm = (X_treino - media_treino) / desvio_treino
    X_teste_norm = (X_teste - media_treino) / desvio_treino

    t0 = time.time()
    y_pred = classificar_knn(X_treino_norm, y_treino, X_teste_norm, k=k)
    tempo_classificacao_s = time.time() - t0

    y_true_validos = [yt for yt, yp in zip(y_teste, y_pred) if yp is not None]
    y_pred_validos = [yp for yp in y_pred if yp is not None]

    f1_macro = f1_score(y_true_validos, y_pred_validos, average="macro", zero_division=0)

    print("=" * 74)
    print(f"INDICE NUMERICO (kNN euclidiano) -- split oficial 'Pilot Challenge' (k={k})")
    print("=" * 74)
    print(f"Janelas de teste classificadas : {len(y_true_validos)}/{len(janelas_teste)}")
    print(f"F1-macro (Harbor/indice numerico) : {f1_macro:.4f}")
    print()
    print("Comparacao com baselines oficiais (openpack-torch, mesmo split, treinados/supervisionados):")
    for nome, f1_oficial in BASELINES_OFICIAIS.items():
        diff = f1_macro - f1_oficial
        print(f"  {nome:15} F1={f1_oficial:.4f}  (Harbor {'supera' if diff > 0 else 'fica abaixo'} em {abs(diff):.4f})")
    print()
    print(f"Tempo de extracao de features : {tempo_extracao_s:.1f}s")
    print(f"Tempo de classificacao (kNN)  : {tempo_classificacao_s:.3f}s")

    resultado = {
        "k": k,
        "split": "pilot-challenge",
        "metodo": "indice_numerico_knn",
        "n_janelas_treino": int(len(janelas_treino)),
        "n_janelas_teste": int(len(janelas_teste)),
        "n_classificadas": len(y_true_validos),
        "f1_macro": round(float(f1_macro), 4),
        "baselines_oficiais": BASELINES_OFICIAIS,
        "training_free": True,
        "tempo_extracao_features_s": round(tempo_extracao_s, 1),
        "tempo_classificacao_s": round(tempo_classificacao_s, 3),
        "n_features_por_janela": X_treino.shape[1],
    }
    caminho_saida = EVAL_DIR / "resultados_indice_numerico_openpack.json"
    caminho_saida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados salvos em {caminho_saida}")
    return resultado


def avaliar_loso(k: int) -> dict:
    """Leave-One-Subject-Out sobre TODOS os sujeitos presentes em janelas_amostradas.csv -- par a
    par com eval/avaliar_loso_openpack.py, trocando so o metodo de classificacao (retrieval
    textual -> kNN numerico) para a comparacao valer sob o mesmo protocolo cross-subject."""
    df_janelas = pd.read_csv(JANELAS_AMOSTRADAS)
    sujeitos = sorted(df_janelas["sujeito"].unique())
    print(f"LOSO (indice numerico) sobre {len(sujeitos)} sujeitos.")

    sinais = carregar_sinais(df_janelas)

    resultados_por_sujeito = {}
    for sujeito_teste in sujeitos:
        janelas_treino = df_janelas[df_janelas["sujeito"] != sujeito_teste]
        janelas_teste = df_janelas[df_janelas["sujeito"] == sujeito_teste]
        if len(janelas_teste) == 0 or len(janelas_treino) == 0:
            continue

        X_treino, y_treino, _ = construir_matriz(janelas_treino, sinais)
        X_teste, y_teste, _ = construir_matriz(janelas_teste, sinais)

        media_treino = X_treino.mean(axis=0)
        desvio_treino = X_treino.std(axis=0)
        desvio_treino[desvio_treino == 0] = 1.0
        X_treino_norm = (X_treino - media_treino) / desvio_treino
        X_teste_norm = (X_teste - media_treino) / desvio_treino

        y_pred = classificar_knn(X_treino_norm, y_treino, X_teste_norm, k=k)
        y_true_validos = [yt for yt, yp in zip(y_teste, y_pred) if yp is not None]
        y_pred_validos = [yp for yp in y_pred if yp is not None]
        f1 = f1_score(y_true_validos, y_pred_validos, average="macro", zero_division=0) if y_true_validos else 0.0
        resultados_por_sujeito[sujeito_teste] = round(float(f1), 4)
        print(f"  {sujeito_teste}: F1-macro={f1:.4f}  ({len(janelas_teste)} janelas de teste)")

    valores = list(resultados_por_sujeito.values())
    media = float(np.mean(valores)) if valores else 0.0
    desvio = float(np.std(valores)) if valores else 0.0

    print("=" * 74)
    print(f"LOSO (indice numerico) -- F1-macro medio: {media:.4f} +/- {desvio:.4f}")
    print(f"  min={min(valores):.4f}  max={max(valores):.4f}" if valores else "  sem resultados")
    print("=" * 74)

    resultado = {
        "k": k,
        "metodo": "indice_numerico_knn_loso",
        "f1_macro_por_sujeito": resultados_por_sujeito,
        "f1_macro_medio": round(media, 4),
        "f1_macro_desvio": round(desvio, 4),
        "n_sujeitos": len(resultados_por_sujeito),
        "baselines_oficiais": BASELINES_OFICIAIS,
    }
    caminho_saida = EVAL_DIR / "resultados_loso_indice_numerico_openpack.json"
    caminho_saida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados salvos em {caminho_saida}")
    return resultado


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5,
                         help="numero de vizinhos de treino considerados na votacao (default 5)")
    parser.add_argument("--loso", action="store_true",
                         help="roda Leave-One-Subject-Out em vez do split oficial Pilot Challenge")
    args = parser.parse_args()

    if args.loso:
        avaliar_loso(k=args.k)
    else:
        avaliar_split_oficial(k=args.k)


if __name__ == "__main__":
    main()
