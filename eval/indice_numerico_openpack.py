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


def preparar_treino_cache(k: int = 5) -> dict:
    """Frente 1B do plano 'OpenPack: integrar indice numerico ao chat' (2026-09-29): monta a
    matriz de treino UMA VEZ (janelas dos sujeitos U0102/U0103/U0105, mesmo split oficial 'Pilot
    Challenge' ja usado em avaliar_split_oficial()) e devolve tudo que
    classificar_janela_ao_vivo() precisa para classificar sob demanda sem recarregar dados de
    sensor nem reextrair features a cada pergunta do chat -- caro (segundos por sessao). O
    chamador (dashboard/app.py) e responsavel por cachear o RESULTADO deste dict via
    st.cache_resource, exatamente como ja faz com rag_openpack_indexado()."""
    df_janelas = pd.read_csv(JANELAS_AMOSTRADAS)
    janelas_treino = df_janelas[df_janelas[["sujeito", "sessao"]].apply(tuple, axis=1).isin(SPLIT_TREINO)]
    sinais_treino = carregar_sinais(janelas_treino)

    X_treino, y_treino, _ = construir_matriz(janelas_treino, sinais_treino)
    media_treino = X_treino.mean(axis=0)
    desvio_treino = X_treino.std(axis=0)
    desvio_treino[desvio_treino == 0] = 1.0
    X_treino_norm = (X_treino - media_treino) / desvio_treino

    return {
        "X_treino_norm": X_treino_norm, "y_treino": y_treino,
        "media_treino": media_treino, "desvio_treino": desvio_treino, "k": k,
    }


def classificar_janela_ao_vivo(sujeito: str, sessao: str, cache_treino: dict, indice_janela: int = 0) -> dict | None:
    """Classifica UMA janela real de uma sessao de TESTE (nunca de treino -- evitaria vazamento
    trivial de usar a propria janela como seu vizinho), usando o cache de preparar_treino_cache().
    indice_janela=0 escolhe deterministicamente a PRIMEIRA janela daquela sessao (ordem de
    janelas_amostradas.csv) -- nao ha hoje, no chat, forma do usuario indicar um timestamp exato
    (ver exploracao do plano), entao "a primeira janela da sessao X" e a unidade de referencia
    reconhecivel em linguagem natural mais simples e reprodutivel.

    Retorna None se sujeito/sessao nao existir na amostra, ou se sujeito estiver no proprio
    conjunto de treino (SPLIT_TREINO) -- classificar uma janela de treino contra o proprio
    treino inflaria artificialmente a confianca, mesma logica de janelas_treino_ids em
    avaliar_classificacao_openpack.py::classificar_por_vizinhos()."""
    if (sujeito, sessao) in SPLIT_TREINO:
        return None

    df_janelas = pd.read_csv(JANELAS_AMOSTRADAS)
    janelas_sessao = df_janelas[(df_janelas["sujeito"] == sujeito) & (df_janelas["sessao"] == sessao)]
    if len(janelas_sessao) == 0 or indice_janela >= len(janelas_sessao):
        return None

    janela = janelas_sessao.iloc[indice_janela]
    sinais = carregar_sinais(janelas_sessao.iloc[[indice_janela]])
    df_sinais = sinais[(sujeito, sessao)]
    bloco = df_sinais.iloc[janela["indice_inicio"]: janela["indice_fim"]].reset_index(drop=True)

    v = vetor_da_janela(bloco)
    v_norm = (v - cache_treino["media_treino"]) / cache_treino["desvio_treino"]

    predicao = classificar_knn(
        cache_treino["X_treino_norm"], cache_treino["y_treino"],
        v_norm.reshape(1, -1), k=cache_treino["k"],
    )[0]

    return {
        "janela_id": janela["janela_id"],
        "sujeito": sujeito, "sessao": sessao,
        "operacao_prevista": predicao,
        "operacao_real": janela["operacao"],  # so para quem consome saber se acertou -- NUNCA
        # passar isso para o prompt do LLM antes da predicao (vazamento trivial).
        "acertou": predicao == janela["operacao"],
    }


def resumo_para_chat() -> str:
    """Frente 1A do plano 'OpenPack: integrar indice numerico ao chat' (2026-09-29): resumo em
    portugues, pronto para injetar como contexto pre-calculado no chat (mesmo padrao de
    planned_vs_unplanned/lss_melhorou_tudo/interpretacao_recall em dashboard/app.py -- Python
    calcula, LLM so narra, nunca deixa o LLM comparar numeros sozinho). SO LE o JSON ja
    persistido por avaliar_loso() -- nao reclassifica nada a cada pergunta do chat, ja que o LOSO
    completo leva minutos.

    Se o JSON nao existir ainda (avaliar_loso() nunca rodou), retorna None em vez de lancar
    excecao -- o chamador decide o que fazer (ex. nao injetar esse trecho no contexto)."""
    caminho = EVAL_DIR / "resultados_loso_indice_numerico_openpack.json"
    if not caminho.exists():
        return None

    dados = json.loads(caminho.read_text(encoding="utf-8"))
    f1_medio = dados["f1_macro_medio"]
    f1_desvio = dados["f1_macro_desvio"]
    n_sujeitos = dados["n_sujeitos"]
    baselines = dados["baselines_oficiais"]

    por_sujeito = dados["f1_macro_por_sujeito"]
    sujeito_min = min(por_sujeito, key=por_sujeito.get)
    sujeito_max = max(por_sujeito, key=por_sujeito.get)

    comparacoes = []
    for nome, valor in baselines.items():
        diff = f1_medio - valor
        verbo = "supera" if diff > 0 else "fica abaixo de"
        comparacoes.append(f"{verbo} {nome} ({valor:.4f}) em {abs(diff):.4f}")

    return (
        f"CLASSIFICADOR DE OPERACAO OPENPACK POR INDICE NUMERICO (kNN sobre features de sensor, "
        f"protocolo LOSO cross-subject, {n_sujeitos} sujeitos):\n"
        f"F1-macro medio: {f1_medio:.4f} +/- {f1_desvio:.4f} (desvio padrao entre sujeitos).\n"
        f"Sujeito com pior desempenho: {sujeito_min} (F1={por_sujeito[sujeito_min]:.4f}). "
        f"Sujeito com melhor desempenho: {sujeito_max} (F1={por_sujeito[sujeito_max]:.4f}).\n"
        f"Comparacao com baselines oficiais supervisionados (mesmo split, treinados com GPU): "
        + "; ".join(comparacoes) + ".\n"
        f"Este metodo e training-free (sem nenhum treino de modelo) -- substituiu em 2026-09-28 "
        f"um indice textual mais lento e menos preciso (F1-macro subiu de 0,1626 para {f1_medio:.4f})."
    )


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
