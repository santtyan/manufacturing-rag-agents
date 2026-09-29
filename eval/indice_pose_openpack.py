"""
Fase 2 do plano "OpenPack Fase 2: classificador de pose sobre keypoints 3D" (2026-09-29).

Constroi o equivalente ao indice numerico de IMU (eval/indice_numerico_openpack.py), mas sobre
keypoints 3D do Kinect (Azure Kinect Body Tracking, 32 juntas x posicao XYZ + quaternion +
confianca, ver `kinect/3d-kpt/single-ffill-flip-fixed/*.csv` dentro dos zips do Zenodo em
C:\\Users\\USER\\Desktop\\OpenPack\\zenodo\\U*.zip). Decisao explicita do usuario (2026-09-29) de
medir pose mesmo com o indice IMU ja superando o baseline UNet -- objetivo e medir com o mesmo
rigor, nao perseguir SOTA.

ACHADO REAL (2026-09-29): o split oficial "Pilot Challenge" (treino U0102/U0103/U0105, teste
U0106) NAO tem overlap com os 6 sujeitos que tem keypoints (U0201, U0206-U0210) -- nao da pra
comparar pose com o indice IMU sob o MESMO split. Comparacao aqui e sob o MESMO PROTOCOLO (LOSO),
com um universo de sujeitos MENOR (6, nao 21) -- reportar sempre essa diferenca de N, nunca
apresentar os dois lado a lado como equivalentes.

MODULO SEPARADO de indice_numerico_openpack.py de proposito -- fontes de dado e dominio
diferentes (sensor IMU vestivel vs. camera de profundidade), acoplar os dois no mesmo arquivo
arriscaria regressao no indice IMU ja em producao no chat (dashboard/app.py).

Reusa sem modificacao: classificar_knn() e o padrao de normalizacao z-score treino-only de
indice_numerico_openpack.py; features_estatisticas_janela() de rag_openpack_texto.py (100%
generica sobre qualquer serie 1D, ja confirmado); o padrao de leitura zip+unixtime de
avaliar_outliers_openpack.py::carregar_outliers_sessao().

NAO usa quaternion (decisao ja tomada e documentada no plano anterior): bug de SDK do Azure
Kinect nas juntas do tronco superior, e as duas levas de coleta do OpenPack divergem nessa
modalidade (ver [[openpack_duas_levas_coleta_quaternion_2026-09-15]]) -- usar so posicao XYZ
normalizada + confianca por junta.

Uso: python eval/indice_pose_openpack.py [--k 5]
     Pre-requisito: outputs/pipeline8_openpack/janelas_amostradas.csv e os 6 zips do Zenodo em
     C:\\Users\\USER\\Desktop\\OpenPack\\zenodo\\U*.zip.
"""
import argparse
import json
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

HARBOR_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HARBOR_ROOT))
sys.path.insert(0, str(HARBOR_ROOT / "rag"))
sys.path.insert(0, str(HARBOR_ROOT / "eval"))

from rag_openpack_texto import features_estatisticas_janela, NOMES_FEATURES  # noqa: E402
from indice_numerico_openpack import classificar_knn  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
JANELAS_AMOSTRADAS = HARBOR_ROOT / "outputs" / "pipeline8_openpack" / "janelas_amostradas.csv"
ZIPS_DIR = Path(r"C:\Users\USER\Desktop\OpenPack\zenodo")

SUJEITOS_COM_VIDEO = ["U0201", "U0206", "U0207", "U0208", "U0209", "U0210"]

# Baselines oficiais -- mesmos citados em indice_numerico_openpack.py. ST-GCN e o unico que E'
# sobre keypoints/pose (nao IMU) -- ver correcao de modalidade no CLAUDE.md, 2026-09-28. Nao
# comparavel diretamente aqui: e rede supervisionada com GPU, o metodo abaixo e training-free.
BASELINES_OFICIAIS = {"UNet": 0.3451, "ST-GCN": 0.7024, "DeepConvLSTM": 0.7081}

# Esquema oficial Azure Kinect Body Tracking SDK (K4ABT_JOINT_*, k4abttypes.h) -- 32 juntas,
# indice 0-31. Fonte: https://learn.microsoft.com/en-us/previous-versions/azure/kinect-dk/body-joints
JUNTA_PELVIS = "J00"       # PELVIS -- origem da normalizacao (centro do corpo)
JUNTA_HIP_L = "J18"        # HIP_LEFT
JUNTA_HIP_R = "J22"        # HIP_RIGHT
JUNTA_OMBRO_L = "J05"      # SHOULDER_LEFT -- usado com HIP_L para escala ossea
JUNTA_OMBRO_R = "J12"      # SHOULDER_RIGHT

# Juntas relevantes para embalagem -- nao as 32 completas (ruido demais, dedos/olhos nao
# discriminam operacao de embalagem). Nomes seguem o enum oficial K4ABT_JOINT_*.
JUNTAS_RELEVANTES = {
    "punho_esquerdo": "J07",     # WRIST_LEFT
    "punho_direito": "J14",      # WRIST_RIGHT
    "cotovelo_esquerdo": "J06",  # ELBOW_LEFT
    "cotovelo_direito": "J13",   # ELBOW_RIGHT
    "ombro_esquerdo": "J05",     # SHOULDER_LEFT
    "ombro_direito": "J12",      # SHOULDER_RIGHT
    "quadril_esquerdo": "J18",   # HIP_LEFT
    "quadril_direito": "J22",    # HIP_RIGHT
    "joelho_esquerdo": "J19",    # KNEE_LEFT
    "joelho_direito": "J23",     # KNEE_RIGHT
}

CONF_MINIMA = 1.0  # escala K4ABT_JOINT_CONFIDENCE_*: 0=none, 1=low, 2=high. Salvaguarda para
# descartar frame com confianca "none" (0) na junta -- oclusao total. ACHADO REAL (2026-09-29):
# CONF==0 nao ocorre em nenhuma sessao verificada desta variante do dataset ("single-ffill-flip-
# fixed" -- provavelmente ja faz forward-fill de frames perdidos), entao este filtro na pratica
# nunca descarta nada aqui; mantido como salvaguarda caso outra sessao/sujeito tenha o caso.
# O sinal real de degradacao de rastreamento e a oscilacao entre confianca baixa (1) e alta (2),
# medida por taxa_oclusao_junta() abaixo (CONF<=1), nao por este filtro.


def carregar_keypoints_sessao(sujeito: str, sessao: str) -> pd.DataFrame | None:
    """Le kinect/3d-kpt/single-ffill-flip-fixed/{sessao}.csv de dentro do zip do Zenodo daquele
    sujeito -- mesmo padrao de avaliar_outliers_openpack.py::carregar_outliers_sessao(). Retorna
    None se o zip ou o CSV nao existir/estiver vazio (ex. U0206/S0200, ausente no dataset)."""
    caminho_zip = ZIPS_DIR / f"{sujeito}.zip"
    if not caminho_zip.exists():
        return None
    with zipfile.ZipFile(caminho_zip) as z:
        caminho_interno = f"kinect/3d-kpt/single-ffill-flip-fixed/{sessao}.csv"
        if caminho_interno not in z.namelist():
            return None
        conteudo = z.read(caminho_interno)
    if not conteudo.strip():
        return None
    df = pd.read_csv(BytesIO(conteudo))
    if len(df) == 0:
        return None
    return df


def recortar_pose_da_janela(df_kpts: pd.DataFrame, t_inicio_ms: int, t_fim_ms: int) -> pd.DataFrame:
    """Filtra os frames de pose cujo unixtime cai dentro de [t_inicio_ms, t_fim_ms] -- mesmo
    epoch/unidade (ms) que t_inicio_ms/t_fim_ms de janelas_amostradas.csv, confirmado
    empiricamente (ver docstring do modulo). ~60 frames por janela de 4s a 15Hz. Nao precisa de
    logica de sincronizacao nova, e filtro de intervalo direto."""
    return df_kpts[(df_kpts["unixtime"] >= t_inicio_ms) & (df_kpts["unixtime"] <= t_fim_ms)]


def normalizar_pose(bloco: pd.DataFrame) -> pd.DataFrame:
    """Normalizacao OBRIGATORIA por frame (nao opcional -- ver plano):
    1. Centraliza no quadril: subtrai a posicao do ponto medio de HIP_LEFT/HIP_RIGHT (J18/J22)
       de todas as juntas, por frame -- sem isso, sob LOSO, a posicao absoluta mede onde a
       pessoa estava na cena (variavel por sessao/camera), nao a acao.
    2. Escala por comprimento osseo (distancia ombro-quadril medio, mais estavel que altura
       total a variacao de postura entre frames) -- sem isso, features de amplitude medem
       biotipo do sujeito, confundindo o LOSO exatamente como ja documentado para o caso IMU.

    Retorna uma COPIA do bloco com colunas J{XX}_P0/P1/P2 substituidas pelos valores
    normalizados -- colunas _O*/_CONF nao sao tocadas aqui."""
    bloco = bloco.copy()

    hip_l = bloco[[f"{JUNTA_HIP_L}_P0", f"{JUNTA_HIP_L}_P1", f"{JUNTA_HIP_L}_P2"]].to_numpy()
    hip_r = bloco[[f"{JUNTA_HIP_R}_P0", f"{JUNTA_HIP_R}_P1", f"{JUNTA_HIP_R}_P2"]].to_numpy()
    centro_quadril = (hip_l + hip_r) / 2  # (n_frames, 3)

    ombro_l = bloco[[f"{JUNTA_OMBRO_L}_P0", f"{JUNTA_OMBRO_L}_P1", f"{JUNTA_OMBRO_L}_P2"]].to_numpy()
    ombro_r = bloco[[f"{JUNTA_OMBRO_R}_P0", f"{JUNTA_OMBRO_R}_P1", f"{JUNTA_OMBRO_R}_P2"]].to_numpy()
    centro_ombro = (ombro_l + ombro_r) / 2
    comprimento_osseo = np.linalg.norm(centro_ombro - centro_quadril, axis=1)  # (n_frames,)
    comprimento_osseo[comprimento_osseo < 1e-6] = 1.0  # evita divisao por zero (frame degenerado)

    for junta in [f"J{i:02d}" for i in range(32)]:
        cols = [f"{junta}_P0", f"{junta}_P1", f"{junta}_P2"]
        if not all(c in bloco.columns for c in cols):
            continue
        pos = bloco[cols].to_numpy()
        pos_normalizada = (pos - centro_quadril) / comprimento_osseo[:, None]
        bloco[cols] = pos_normalizada

    return bloco


def vetor_pose_da_janela(bloco: pd.DataFrame) -> np.ndarray | None:
    """Equivalente a indice_numerico_openpack.py::vetor_da_janela(), mas sobre trajetoria de
    pose. Para cada junta relevante (JUNTAS_RELEVANTES), calcula o deslocamento normalizado
    (norma da posicao apos normalizar_pose()) ao longo da janela e aplica
    features_estatisticas_janela() -- reusada literalmente de rag_openpack_texto.py, ja
    confirmada 100% generica sobre qualquer serie 1D.

    Descarta frames com confianca "none" (CONF < CONF_MINIMA) na junta antes de calcular a
    serie -- oclusao total. Retorna None se, apos filtrar, sobrarem menos de 3 frames validos
    (janela essencialmente sem dado confiavel para aquela junta)."""
    if len(bloco) < 3:
        return None

    bloco_norm = normalizar_pose(bloco)

    vetor = []
    for nome_junta, codigo in JUNTAS_RELEVANTES.items():
        conf = bloco[f"{codigo}_CONF"].to_numpy()
        validos = conf >= CONF_MINIMA
        if validos.sum() < 3:
            # sem dado confiavel o suficiente nesta junta -- preenche com zeros (nao descarta a
            # janela inteira, so esta junta fica sem sinal, mesmo espirito de features_estatisticas_janela
            # com valores.size==0 em rag_openpack_texto.py)
            feats = {nome: 0.0 for nome in NOMES_FEATURES}
        else:
            pos = bloco_norm.loc[validos, [f"{codigo}_P0", f"{codigo}_P1", f"{codigo}_P2"]].to_numpy()
            deslocamento = np.linalg.norm(pos, axis=1)  # distancia ao centro do quadril, normalizada
            feats = features_estatisticas_janela(deslocamento)
        vetor.extend(feats[nome] for nome in NOMES_FEATURES)

    return np.array(vetor, dtype=float)


def taxa_oclusao_junta(bloco: pd.DataFrame, codigo_junta: str) -> float:
    """Fracao de frames com confianca BAIXA (CONF<=1, escala K4ABT 0=none/1=low/2=high) para
    uma junta -- usado para reportar oclusao por classe de operacao (mesma causa documentada
    para a queda do ST-GCN oficial no split de submissao, 0,7024->0,6106, atribuida pelo paper
    PerCom a oclusao de punho por caixa).

    ACHADO REAL (2026-09-29): CONF==0 ('none') nao ocorre em nenhuma das sessoes verificadas
    deste dataset (variante 'single-ffill-flip-fixed' -- o proprio nome sugere forward-fill de
    frames perdidos, o que explica a ausencia de confianca zero). O criterio original
    (conf==0) sempre dava 0,000 -- nao media o fenomeno real, que e a oscilacao entre baixa (1)
    e alta (2) confianca. Corrigido para CONF<=1, que de fato varia entre sessoes/categorias."""
    conf = bloco[f"{codigo_junta}_CONF"].to_numpy()
    return float((conf <= 1).mean()) if len(conf) else 0.0


def construir_matriz(df_janelas: pd.DataFrame, keypoints_por_sessao: dict) -> tuple[np.ndarray, list[str], list[str], list[float]]:
    """Equivalente a indice_numerico_openpack.py::construir_matriz(), mas recorta por
    unixtime (recortar_pose_da_janela) em vez de indice posicional -- o stream de pose nao
    compartilha indexacao com o stream de IMU (frequencias diferentes: ~33Hz vs 15Hz).
    Retorna tambem a taxa media de oclusao de punho por janela (media dos 2 punhos), para o
    relatorio de oclusao por classe."""
    vetores, operacoes, ids, oclusoes = [], [], [], []
    for _, janela in df_janelas.iterrows():
        df_kpts = keypoints_por_sessao.get((janela["sujeito"], janela["sessao"]))
        if df_kpts is None:
            continue
        bloco = recortar_pose_da_janela(df_kpts, janela["t_inicio_ms"], janela["t_fim_ms"])
        vetor = vetor_pose_da_janela(bloco)
        if vetor is None:
            continue
        vetores.append(vetor)
        operacoes.append(janela["operacao"])
        ids.append(janela["janela_id"])
        oclusao_l = taxa_oclusao_junta(bloco, JUNTAS_RELEVANTES["punho_esquerdo"])
        oclusao_r = taxa_oclusao_junta(bloco, JUNTAS_RELEVANTES["punho_direito"])
        oclusoes.append((oclusao_l + oclusao_r) / 2)
    if not vetores:
        return np.empty((0, len(JUNTAS_RELEVANTES) * len(NOMES_FEATURES))), [], [], []
    return np.vstack(vetores), operacoes, ids, oclusoes


def carregar_keypoints_dos_sujeitos(df_janelas: pd.DataFrame) -> dict:
    pares_sessao = df_janelas[["sujeito", "sessao"]].drop_duplicates().itertuples(index=False)
    keypoints = {}
    for sujeito, sessao in pares_sessao:
        if sujeito not in SUJEITOS_COM_VIDEO:
            continue
        df_kpts = carregar_keypoints_sessao(sujeito, sessao)
        if df_kpts is not None:
            keypoints[(sujeito, sessao)] = df_kpts
    return keypoints


def avaliar_loso(k: int = 5) -> dict:
    """LOSO restrito aos 6 sujeitos com video/keypoints -- NAO o split oficial 'Pilot Challenge'
    (que nao tem overlap com esses sujeitos, ver docstring do modulo). Mesma estrutura de
    indice_numerico_openpack.py::avaliar_loso(), trocando so a fonte de features."""
    df_janelas = pd.read_csv(JANELAS_AMOSTRADAS)
    df_janelas = df_janelas[df_janelas["sujeito"].isin(SUJEITOS_COM_VIDEO)]
    print(f"LOSO (indice de pose) sobre {len(SUJEITOS_COM_VIDEO)} sujeitos com video "
          f"({len(df_janelas)} janelas totais na amostra).")

    keypoints_por_sessao = carregar_keypoints_dos_sujeitos(df_janelas)
    print(f"Keypoints carregados para {len(keypoints_por_sessao)} pares (sujeito, sessao) "
          f"de {df_janelas[['sujeito','sessao']].drop_duplicates().shape[0]} esperados.")

    resultados_por_sujeito = {}
    oclusoes_por_categoria: dict[str, list[float]] = {}

    for sujeito_teste in SUJEITOS_COM_VIDEO:
        janelas_treino = df_janelas[df_janelas["sujeito"] != sujeito_teste]
        janelas_teste = df_janelas[df_janelas["sujeito"] == sujeito_teste]
        if len(janelas_teste) == 0 or len(janelas_treino) == 0:
            continue

        X_treino, y_treino, _, _ = construir_matriz(janelas_treino, keypoints_por_sessao)
        X_teste, y_teste, _, oclusoes_teste = construir_matriz(janelas_teste, keypoints_por_sessao)

        if len(X_treino) == 0 or len(X_teste) == 0:
            print(f"  {sujeito_teste}: SKIP (sem janelas classificaveis apos filtro de keypoints)")
            continue

        for op, ocl in zip(y_teste, oclusoes_teste):
            oclusoes_por_categoria.setdefault(op, []).append(ocl)

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
        print(f"  {sujeito_teste}: F1-macro={f1:.4f}  ({len(y_true_validos)}/{len(janelas_teste)} janelas classificadas)")

    valores = list(resultados_por_sujeito.values())
    media = float(np.mean(valores)) if valores else 0.0
    desvio = float(np.std(valores)) if valores else 0.0

    oclusao_media_por_categoria = {
        cat: round(float(np.mean(vals)), 4) for cat, vals in oclusoes_por_categoria.items()
    }

    print("=" * 74)
    print(f"LOSO (indice de pose, N={len(resultados_por_sujeito)} sujeitos) -- "
          f"F1-macro medio: {media:.4f} +/- {desvio:.4f}")
    if valores:
        print(f"  min={min(valores):.4f}  max={max(valores):.4f}")
    print()
    print("Comparacao (protocolos DIFERENTES -- N=6 sujeitos aqui vs. N=21 no indice IMU, "
          "nao apresentar como equivalente):")
    print(f"  Indice numerico IMU (LOSO, N=21): F1-macro=0,4314 +/- 0,0634 (supera UNet)")
    print(f"  Indice de pose      (LOSO, N={len(resultados_por_sujeito)}):  F1-macro={media:.4f} +/- {desvio:.4f}")
    print()
    print("Taxa media de oclusao de punho por categoria (fracao de frames com confianca baixa, CONF<=1):")
    for cat, tx in sorted(oclusao_media_por_categoria.items(), key=lambda x: -x[1]):
        print(f"  {cat:25} {tx:.3f}")
    print("=" * 74)

    resultado = {
        "k": k,
        "metodo": "indice_pose_knn_loso",
        "sujeitos_avaliados": SUJEITOS_COM_VIDEO,
        "f1_macro_por_sujeito": resultados_por_sujeito,
        "f1_macro_medio": round(media, 4),
        "f1_macro_desvio": round(desvio, 4),
        "n_sujeitos": len(resultados_por_sujeito),
        "oclusao_media_punho_por_categoria": oclusao_media_por_categoria,
        "baselines_oficiais": BASELINES_OFICIAIS,
        "comparacao_indice_imu": {
            "f1_macro_medio": 0.4314, "f1_macro_desvio": 0.0634, "n_sujeitos": 21,
            "nota": "protocolo LOSO igual, N de sujeitos diferente -- nao comparavel como se fosse o mesmo experimento",
        },
        "ressalva_metodologica": (
            "Training-free (kNN), sem GPU, sem treino de modelo -- nao comparavel diretamente "
            "ao ST-GCN oficial (rede supervisionada com GPU, F1=0,7024 test / 0,6106 submission)."
        ),
    }
    caminho_saida = EVAL_DIR / "resultados_loso_indice_pose_openpack.json"
    caminho_saida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados salvos em {caminho_saida}")
    return resultado


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5,
                         help="numero de vizinhos de treino considerados na votacao (default 5)")
    args = parser.parse_args()
    avaliar_loso(k=args.k)


if __name__ == "__main__":
    main()
