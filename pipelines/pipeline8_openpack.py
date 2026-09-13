"""
Pipeline 8 - OpenPack (reconhecimento de operacoes de trabalho em linha de embalagem logistica)
Fase 1 do plano de integracao: viabilidade de parsing contra o sample data (U0209/S0500),
sem depender do ZIP pre-processado de 515MB do Zenodo (Fase 3).

ACHADO REAL (2026-09-12): a documentacao do OpenPack anuncia 30Hz para o CSV pre-processado
(imuWithOperationLabel), mas o sample data do GitHub esta no formato RAW por sensor -- 4 arquivos
separados (atr01..atr04, sem label) + anotacao por INTERVALO (start/end), nao por-amostra. A
frequencia real medida no raw e ~33,3Hz (periodo mediano de 30ms), nao 30Hz. carregar_sessao()
aceita as duas fontes (raw multi-arquivo com merge por intervalo, OU o CSV pre-processado unico)
e validar_sessao() usa uma faixa [28,35] Hz para cobrir ambas sem falso-negativo.

Uso: python pipelines/pipeline8_openpack.py --sample
     (roda so contra o sample_U0209, sem exigir o ZIP completo baixado)
     python pipelines/pipeline8_openpack.py
     (Fase 3+, requer C:\\Users\\USER\\Downloads\\OpenPack\\imu_preprocessado\\ populado)
"""
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_DIR = Path(r"C:\Users\USER\Downloads\OpenPack\sample_U0209")
DATA_PATH = Path(r"C:\Users\USER\Downloads\OpenPack\imu_preprocessado\imuWithOperationLabel")
OUT = Path(r"C:\Projetos\Harbor\outputs\pipeline8_openpack")
OUT.mkdir(parents=True, exist_ok=True)

FREQ_HZ_ESPERADA = 30  # doc do dataset pre-processado; raw real medido ~33,3Hz (ver docstring)
FREQ_HZ_MIN, FREQ_HZ_MAX = 28, 35
JANELA_SEGUNDOS = 4
PASSO_SEGUNDOS = 2
PUREZA_MINIMA = 0.7

SENSORES_ATR = ["atr01", "atr02", "atr03", "atr04"]
CANAIS_RAW = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
              "quat_w", "quat_x", "quat_y", "quat_z"]
# 4 sensores x 10 canais = 40 colunas de sinal, geradas por produto cartesiano (nao hardcoded
# uma a uma) -- nome final e "{sensor}_{canal}", ex. "atr01_acc_x".
COLUNAS_SINAL = [f"{sensor}_{canal}" for sensor, canal in product(SENSORES_ATR, CANAIS_RAW)]

CLASSES_OPERACAO = {
    100: "Picking", 200: "Relocate Item Label", 300: "Assemble Box",
    400: "Insert Items", 500: "Close Box", 600: "Attach Box Label",
    700: "Scan Label", 800: "Attach Shipping Label", 900: "Put on Back Table",
    1000: "Fill out Order", 8100: "Null",
}
# Ordem oficial de OPENPACK_OPERATIONS no toolkit (openpack_toolkit/configs/datasets/annotations.py) --
# ACHADO REAL (2026-09-12): o CSV pre-processado do Zenodo NAO usa os IDs de classe (100, 200, ...)
# na coluna "operation" -- usa o INDICE POSICIONAL 0-10 dentro desta tupla (confirmado contra o
# codigo-fonte do openpack-toolkit, ActSet.__call__(cls_idx) indexa por posicao). Index 10 = Null.
# Suposicao inicial errada (assumir que "operation" ja vinha com o ID) teria mapeado toda a
# distribuicao de classes de forma incorreta silenciosamente -- confirmado antes de rodar em escala.
INDICE_PARA_ID_OPERACAO = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 8100]


# Tolerancia para o merge_asof entre sensores -- cobre jitter de clock entre sensores IMU
# fisicamente distintos sem exigir timestamp identico. Metade do periodo nominal a 30Hz.
TOLERANCIA_MERGE_SENSORES_MS = 15


def _carregar_sessao_raw(pasta_sessao: Path, pasta_anotacao: Path, sujeito: str, sessao: str) -> pd.DataFrame:
    """Formato do sample data do GitHub: 4 CSVs de sensor (unixtime + 10 canais, sem label)
    mais um CSV de anotacao por INTERVALO (start/end em vez de label por-amostra). O merge por
    intervalo e necessario porque nao ha coluna de label direta no sensor -- diferente do CSV
    pre-processado do Zenodo, que ja vem com o label por linha.

    ACHADO REAL (revisao cega dupla, 2026-09-12): a primeira versao usava pd.concat(join="inner")
    pelo indice unixtime para juntar os 4 sensores -- funciona no sample U0209 porque seus 4
    arquivos tem timestamps identicos linha a linha (confirmado por inspecao), mas e uma suposicao
    fragil: sensores IMU fisicamente distintos podem ter jitter de clock entre si, e join exato
    por inteiro descartaria silenciosamente qualquer timestamp nao coincidente nos 4 -- na pior
    hipotese, um DataFrame vazio sem nenhum erro. Corrigido para merge_asof com tolerancia,
    que tambem reporta quantas amostras foram descartadas por falta de par proximo o suficiente."""
    partes = []
    for sensor in SENSORES_ATR:
        caminho = pasta_sessao / f"{sensor}_{sessao}.csv"
        df_sensor = pd.read_csv(caminho)
        df_sensor = df_sensor.rename(columns={c: f"{sensor}_{c}" for c in CANAIS_RAW})
        df_sensor = df_sensor.sort_values("unixtime")
        partes.append(df_sensor)

    n_antes_merge = [len(p) for p in partes]
    df = partes[0].rename(columns={"unixtime": "timestamp_ms"})
    for outro in partes[1:]:
        df = pd.merge_asof(df, outro.rename(columns={"unixtime": "timestamp_ms"}),
                            on="timestamp_ms", direction="nearest",
                            tolerance=TOLERANCIA_MERGE_SENSORES_MS)
    n_apos_merge_sensores = len(df)
    linhas_com_gap = df[[c for c in df.columns if c != "timestamp_ms"]].isna().any(axis=1).sum()
    df = df.dropna()  # linhas sem par dentro da tolerancia em algum sensor

    anotacao = pd.read_csv(pasta_anotacao / f"operations_{sessao}.csv")
    anotacao["start_ms"] = pd.to_datetime(anotacao["start"]).astype("int64") // 10**6
    anotacao["end_ms"] = pd.to_datetime(anotacao["end"]).astype("int64") // 10**6
    anotacao = anotacao.sort_values("start_ms")

    # ACHADO REAL (revisao cega dupla, 2026-09-12): merge_asof(direction="backward") assume
    # implicitamente que os intervalos de anotacao NAO se sobrepoem -- se dois intervalos
    # colidirem, o start mais recente sempre vence, mascarando qual operacao realmente cobre o
    # timestamp. Checagem explicita abaixo, reportada em validar_sessao() (nunca corrigida
    # silenciosamente: overlap em anotacao humana e informacao, nao ruido a descartar).
    intervalos_ordenados = anotacao.sort_values("start_ms")
    overlaps = (intervalos_ordenados["start_ms"].shift(-1) < intervalos_ordenados["end_ms"]).sum()

    # merge_asof por intervalo: para cada timestamp de sensor, acha a operacao cujo start <= t.
    # Confere depois que t tambem esta dentro do end (senao operation_label vira NaN -- vira Null).
    df = df.sort_values("timestamp_ms")
    df = pd.merge_asof(df, anotacao[["start_ms", "end_ms", "id"]],
                        left_on="timestamp_ms", right_on="start_ms", direction="backward")
    dentro_do_intervalo = df["timestamp_ms"] <= df["end_ms"]
    df["operation_label"] = np.where(dentro_do_intervalo, df["id"], 8100)
    df["operation_label"] = df["operation_label"].fillna(8100).astype(int)
    df = df.drop(columns=["start_ms", "end_ms", "id"])

    df.attrs["diagnostico_merge"] = {
        "n_linhas_por_sensor": dict(zip(SENSORES_ATR, n_antes_merge)),
        "n_linhas_apos_merge_sensores": n_apos_merge_sensores,
        "n_linhas_descartadas_sem_par": int(linhas_com_gap),
        "n_intervalos_anotacao_sobrepostos": int(overlaps),
    }

    df["sujeito"] = sujeito
    df["sessao"] = sessao
    return df


def _carregar_sessao_preprocessada(caminho_csv: Path) -> pd.DataFrame:
    """Formato do ZIP pre-processado do Zenodo (Fase 3): TEM header ("unixtime,operation,
    atr01/acc_x,..."), diferente da suposicao inicial de header=None -- confirmado por inspecao
    direta do CSV real apos o download. A coluna "operation" traz o INDICE POSICIONAL 0-10 em
    OPENPACK_OPERATIONS (ver INDICE_PARA_ID_OPERACAO acima), nao o ID de classe (100, 200, ...)
    -- corrigido para converter antes de mapear para nome, senao toda classificacao sai errada
    silenciosamente (achado real, 2026-09-12, confirmado contra o codigo-fonte do openpack-toolkit
    antes de indexar o corpus completo)."""
    df = pd.read_csv(caminho_csv)
    colunas_sinal_originais = [c for c in df.columns if c not in ("unixtime", "operation")]
    if len(colunas_sinal_originais) != len(COLUNAS_SINAL):
        raise ValueError(
            f"{caminho_csv.name}: esperado {len(COLUNAS_SINAL)} colunas de sinal, "
            f"encontrado {len(colunas_sinal_originais)} -- arquivo fora do formato "
            f"pre-processado esperado, nao truncar nomes de coluna silenciosamente."
        )
    # Renomeia "atr01/acc_x" -> "atr01_acc_x" (formato interno usado em toda a base de codigo).
    df = df.rename(columns={c: c.replace("/", "_") for c in colunas_sinal_originais})
    df = df.rename(columns={"unixtime": "timestamp_ms"})
    df["operation_label"] = df["operation"].map(lambda idx: INDICE_PARA_ID_OPERACAO[int(idx)])
    df = df.drop(columns=["operation"])

    nome = caminho_csv.stem  # esperado "{sujeito}-{sessao}"
    partes_nome = nome.split("-")
    df["sujeito"] = partes_nome[0] if partes_nome else nome
    df["sessao"] = partes_nome[1] if len(partes_nome) > 1 else "desconhecida"
    return df


def carregar_sessao(sujeito: str, sessao: str, usar_sample: bool = False) -> pd.DataFrame:
    """Ponto de entrada unico: decide a fonte (raw multi-arquivo do sample, ou CSV pre-processado
    do Zenodo) e devolve sempre o mesmo schema de saida: timestamp_ms, operation_label,
    40 colunas de sinal (COLUNAS_SINAL), sujeito, sessao, operacao."""
    if usar_sample:
        df = _carregar_sessao_raw(SAMPLE_DIR / "atr", SAMPLE_DIR / "annotation", sujeito, sessao)
    else:
        caminho = DATA_PATH / f"{sujeito}-{sessao}.csv"
        df = _carregar_sessao_preprocessada(caminho)

    df["operacao"] = df["operation_label"].map(CLASSES_OPERACAO).fillna("Desconhecida")
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    return df


def validar_sessao(df: pd.DataFrame) -> dict:
    """Checks de sanidade que SAO o entregavel da Fase 1 -- confirmam que o parsing (merge por
    intervalo no caso raw, ou leitura direta no pre-processado) produziu um contrato de dados
    correto antes de qualquer uso rio abaixo (segmentacao, indexacao no RAG, tabela SQL).

    Expoe tambem df.attrs["diagnostico_merge"] (quando presente -- so no caminho raw) para que
    perda silenciosa de amostras no merge entre sensores ou overlap de anotacao nao fiquem
    escondidos atras de um "freq_hz_ok: false" sem causa raiz (achado de revisao cega, 2026-09-12)."""
    deltas_ms = df["timestamp_ms"].diff().dropna()
    freq_hz_medida = 1000 / deltas_ms.median() if len(deltas_ms) else 0.0

    labels_presentes = set(df["operation_label"].unique())
    labels_desconhecidos = sorted(labels_presentes - set(CLASSES_OPERACAO.keys()))

    resumo = {
        "n_linhas": int(len(df)),
        "n_colunas_sinal": sum(1 for c in COLUNAS_SINAL if c in df.columns),
        "freq_hz_medida": round(float(freq_hz_medida), 2),
        "freq_hz_ok": bool(FREQ_HZ_MIN <= freq_hz_medida <= FREQ_HZ_MAX),
        "labels_desconhecidos": labels_desconhecidos,
        "classes_presentes": sorted(df["operacao"].unique().tolist()),
        "pct_null_8100": round(float((df["operation_label"] == 8100).mean() * 100), 2),
        "intervalo_temporal_s": round(
            float((df["timestamp_ms"].iloc[-1] - df["timestamp_ms"].iloc[0]) / 1000), 1
        ) if len(df) else 0.0,
        "sujeitos": sorted(df["sujeito"].unique().tolist()),
        "sessoes": sorted(df["sessao"].unique().tolist()),
        "diagnostico_merge": df.attrs.get("diagnostico_merge"),
    }
    resumo["valido"] = resumo["freq_hz_ok"] and not resumo["labels_desconhecidos"]
    return resumo


def segmentar_janelas(df: pd.DataFrame, freq_hz: float,
                       janela_s: float = JANELA_SEGUNDOS, passo_s: float = PASSO_SEGUNDOS,
                       pureza_minima: float = PUREZA_MINIMA) -> pd.DataFrame:
    """Janela deslizante por INDICE (nao por tempo) -- o sinal e regular dentro de uma sessao,
    entao indice e mais barato e determinstico que reamostrar por timestamp. Label da janela e a
    MODA de operation_label; descarta janela de transicao (pureza < pureza_minima) e janela
    puramente Null (8100), que nao representa nenhuma operacao real."""
    tam_janela = max(1, round(janela_s * freq_hz))
    tam_passo = max(1, round(passo_s * freq_hz))

    ultimo_inicio_coberto = max(0, ((len(df) - tam_janela) // tam_passo) * tam_passo + tam_janela) if len(df) >= tam_janela else 0
    n_amostras_descartadas_no_final = len(df) - ultimo_inicio_coberto

    linhas = []
    for inicio in range(0, len(df) - tam_janela + 1, tam_passo):
        bloco = df.iloc[inicio: inicio + tam_janela]
        contagem = bloco["operation_label"].value_counts()
        label_moda = contagem.index[0]
        pureza = contagem.iloc[0] / len(bloco)

        if label_moda == 8100 or pureza < pureza_minima:
            continue

        linhas.append({
            "janela_id": f"{bloco['sujeito'].iloc[0]}_{bloco['sessao'].iloc[0]}_{inicio:07d}",
            "sujeito": bloco["sujeito"].iloc[0],
            "sessao": bloco["sessao"].iloc[0],
            "indice_inicio": inicio,
            "indice_fim": inicio + tam_janela,
            "t_inicio_ms": int(bloco["timestamp_ms"].iloc[0]),
            "t_fim_ms": int(bloco["timestamp_ms"].iloc[-1]),
            "operation_label": int(label_moda),
            "operacao": CLASSES_OPERACAO.get(label_moda, "Desconhecida"),
            "pureza_label": round(float(pureza), 3),
        })

    resultado = pd.DataFrame(linhas, columns=[
        "janela_id", "sujeito", "sessao", "indice_inicio", "indice_fim",
        "t_inicio_ms", "t_fim_ms", "operation_label", "operacao", "pureza_label",
    ])
    resultado.attrs["n_amostras_descartadas_no_final"] = n_amostras_descartadas_no_final
    resultado.attrs["n_amostras_totais"] = len(df)
    return resultado


N_JANELAS_POR_CLASSE = 100
MIN_SUJEITOS_POR_CLASSE = 6
SEED_AMOSTRAGEM = 42


def _alocar_maior_resto(n_total: int, n_grupos: int) -> list[int]:
    """Metodo do maior resto (Hamilton/Hare-Niemeyer -- mesmo algoritmo de apportionment
    eleitoral, padrao-ouro para distribuir uma cota inteira entre grupos sem vies de
    truncamento): aloca o piso (n_total // n_grupos) a cada grupo, depois distribui o resto
    (n_total % n_grupos) aos grupos com maior parte fracionaria. Corrige o truncamento de
    divisao inteira que, com 21 sujeitos e cota 100, geraria so 84 (4/sujeito) em vez de
    perto de 100 -- achado real ao rodar a Fase 3 em escala pela 1a vez, 2026-09-12."""
    piso = n_total // n_grupos
    resto = n_total % n_grupos
    alocacao = [piso] * n_grupos
    for i in range(resto):
        alocacao[i] += 1
    return alocacao


def amostrar_estratificado(df_janelas: pd.DataFrame, n_por_classe: int = N_JANELAS_POR_CLASSE,
                            min_sujeitos: int = MIN_SUJEITOS_POR_CLASSE,
                            seed: int = SEED_AMOSTRAGEM) -> pd.DataFrame:
    """Corpus completo (~96 mil janelas) e inviavel para indexar (Chroma+BM25 em memoria
    passariam de 2GB, ciclo de avaliacao impraticavel -- ver Fase 3 do plano). Amostra
    ~100 janelas por classe real (exclui 8100/Null), com estratificacao secundaria por sujeito
    para nao deixar o corpus dominado pelo "estilo" de um so operador. Usa o metodo do maior
    resto para alocar a cota entre sujeitos sem o vies de truncamento de divisao inteira simples."""
    partes = []
    for operacao, grupo in df_janelas[df_janelas["operacao"] != "Null"].groupby("operacao"):
        sujeitos_disponiveis = sorted(grupo["sujeito"].unique())
        if len(sujeitos_disponiveis) >= min_sujeitos:
            alocacao = _alocar_maior_resto(n_por_classe, len(sujeitos_disponiveis))
            sub_amostras = []
            for sujeito, cota in zip(sujeitos_disponiveis, alocacao):
                disponivel = grupo[grupo["sujeito"] == sujeito]
                sub_amostras.append(disponivel.sample(n=min(cota, len(disponivel)), random_state=seed))
            amostrado = pd.concat(sub_amostras, ignore_index=True)
        else:
            amostrado = grupo.sample(n=min(n_por_classe, len(grupo)), random_state=seed)
        partes.append(amostrado)

    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=df_janelas.columns)


def processar_todas_sessoes(pasta: Path) -> tuple[pd.DataFrame, dict]:
    """Fase 3: carrega e segmenta TODAS as sessoes do ZIP pre-processado, uma de cada vez
    (nunca concatenando os sinais brutos de 102 sessoes em memoria de uma vez -- so as janelas
    resultantes, que sao muito menores)."""
    arquivos = sorted(pasta.glob("*.csv"))
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {pasta} -- baixe o ZIP pre-processado do Zenodo "
            f"(Fase 3 do plano) ou rode com --sample para testar so o sample U0209."
        )

    todas_janelas = []
    sessoes_invalidas = []
    for caminho in arquivos:
        sujeito, sessao = caminho.stem.split("-")
        df = carregar_sessao(sujeito, sessao, usar_sample=False)
        resumo_sessao = validar_sessao(df)
        if not resumo_sessao["valido"]:
            sessoes_invalidas.append({"arquivo": caminho.name, "motivo": resumo_sessao})
            continue
        janelas = segmentar_janelas(df, freq_hz=resumo_sessao["freq_hz_medida"])
        todas_janelas.append(janelas)

    pct_invalidas = len(sessoes_invalidas) / len(arquivos) * 100
    diagnostico = {
        "n_sessoes_processadas": len(arquivos),
        "n_sessoes_validas": len(arquivos) - len(sessoes_invalidas),
        "n_sessoes_invalidas": len(sessoes_invalidas),
        "pct_sessoes_invalidas": round(pct_invalidas, 1),
        "sessoes_invalidas": sessoes_invalidas,
    }
    if pct_invalidas > 10:
        raise RuntimeError(
            f"[ABORTADO] {pct_invalidas:.1f}% das sessoes reprovaram validar_sessao() "
            f"(> 10%, criterio de parada da Fase 3) -- dado possivelmente corrompido no "
            f"download. Detalhe: {json.dumps(sessoes_invalidas, ensure_ascii=False)[:500]}"
        )

    df_janelas = pd.concat(todas_janelas, ignore_index=True) if todas_janelas else pd.DataFrame()
    return df_janelas, diagnostico


def gerar_outputs_agregados(df_janelas: pd.DataFrame, saida: Path):
    """Outputs de outputs/pipeline8_openpack/ que alimentam as rotas `contexto` e `sql` do chat
    (Fase 4a/4b do plano) -- analogos aos outputs dos pipelines 1-4."""
    duracao = df_janelas.assign(
        duracao_s=(df_janelas["t_fim_ms"] - df_janelas["t_inicio_ms"]) / 1000
    ).groupby("operacao").agg(
        n_janelas=("janela_id", "count"),
        duracao_total_s=("duracao_s", "sum"),
        duracao_media_s=("duracao_s", "mean"),
        duracao_mediana_s=("duracao_s", "median"),
    ).reset_index()
    duracao.to_csv(saida / "duracao_por_operacao.csv", index=False)

    ordenado = df_janelas.sort_values(["sujeito", "sessao", "t_inicio_ms"])
    ordenado["operacao_seguinte"] = ordenado.groupby(["sujeito", "sessao"])["operacao"].shift(-1)
    transicoes = ordenado.dropna(subset=["operacao_seguinte"]).groupby(
        ["operacao", "operacao_seguinte"]
    ).size().reset_index(name="n_ocorrencias").rename(columns={"operacao": "operacao_anterior"})
    transicoes = transicoes.sort_values("n_ocorrencias", ascending=False)
    transicoes.to_csv(saida / "transicoes_operacao.csv", index=False)

    variabilidade = duracao_por_sujeito = df_janelas.assign(
        duracao_s=(df_janelas["t_fim_ms"] - df_janelas["t_inicio_ms"]) / 1000
    ).groupby(["sujeito", "operacao"]).agg(
        duracao_media_s=("duracao_s", "mean"),
        desvio_s=("duracao_s", "std"),
    ).reset_index()
    variabilidade.to_csv(saida / "variabilidade_por_sujeito.csv", index=False)

    print(f"Outputs agregados gravados em {saida}")


def gerar_features_e_anomalias(df_amostrado: pd.DataFrame, saida: Path):
    """features_por_operacao.csv e anomalias_ciclo.csv -- rodado sobre a AMOSTRA estratificada
    (1000 janelas), nao o corpus completo (78 mil janelas), porque exige recarregar sinal bruto
    por sessao (custoso) e a amostra ja e balanceada por classe/sujeito (ver amostrar_estratificado()).
    Reusa features_todos_canais/CANAIS de rag/rag_openpack_texto.py -- mesma tecnica de
    rolling/z-score do pipeline2 (add_quant_features), aqui como z-score de DURACAO DE CICLO
    dentro da propria operacao, nao de sensor de maquina (ver NOTA Dataset 8 do ESQUEMA)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "rag"))
    from rag_openpack_texto import CANAIS, features_todos_canais

    pares_sessao = df_amostrado[["sujeito", "sessao"]].drop_duplicates().itertuples(index=False)
    sinais_por_sessao = {(s, sess): carregar_sessao(s, sess, usar_sample=False) for s, sess in pares_sessao}

    linhas_features = []
    for _, janela in df_amostrado.iterrows():
        df_sinais = sinais_por_sessao[(janela["sujeito"], janela["sessao"])]
        bloco = df_sinais.iloc[janela["indice_inicio"]: janela["indice_fim"]]
        feats = features_todos_canais(bloco)
        linha = {"janela_id": janela["janela_id"], "operacao": janela["operacao"]}
        for canal in CANAIS:
            linha[f"{canal}_media"] = feats[canal]["media"]
        linhas_features.append(linha)
    df_features = pd.DataFrame(linhas_features)

    colunas_media = [f"{c}_media" for c in CANAIS]
    features_por_operacao = df_features.groupby("operacao")[colunas_media].mean().reset_index()
    features_por_operacao.to_csv(saida / "features_por_operacao.csv", index=False)

    # Anomalia de duracao de CICLO (z-score dentro da propria operacao) -- mesma tecnica quant
    # do pipeline2 (rolling z-score), aplicada a duracao de janela em vez de valor de sensor.
    duracoes = df_amostrado.assign(duracao_s=(df_amostrado["t_fim_ms"] - df_amostrado["t_inicio_ms"]) / 1000)
    duracoes["duracao_zscore"] = duracoes.groupby("operacao")["duracao_s"].transform(
        lambda s: (s - s.mean()) / s.std() if s.std() > 0 else 0.0
    )
    anomalias_ciclo = duracoes[duracoes["duracao_zscore"].abs() > 2][
        ["janela_id", "sujeito", "sessao", "operacao", "duracao_s", "duracao_zscore"]
    ].sort_values("duracao_zscore", ascending=False)
    anomalias_ciclo.to_csv(saida / "anomalias_ciclo.csv", index=False)

    print(f"features_por_operacao.csv ({len(features_por_operacao)} linhas) e "
          f"anomalias_ciclo.csv ({len(anomalias_ciclo)} linhas) gravados em {saida}")


def main(usar_sample: bool):
    if usar_sample:
        df = carregar_sessao("U0209", "S0500", usar_sample=True)
        resumo = validar_sessao(df)
        print(json.dumps(resumo, indent=2, ensure_ascii=False))

        if not resumo["valido"]:
            print("\n[ABORTADO] validar_sessao() reprovou o parsing -- corrigir antes de prosseguir.")
            (OUT / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
            return

        janelas = segmentar_janelas(df, freq_hz=resumo["freq_hz_medida"])
        resumo["n_janelas_validas"] = int(len(janelas))
        resumo["distribuicao_operacao"] = (
            janelas["operacao"].value_counts().to_dict() if len(janelas) else {}
        )
        resumo["n_amostras_descartadas_no_final_segmentacao"] = janelas.attrs.get("n_amostras_descartadas_no_final", 0)

        (OUT / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
        janelas.to_csv(OUT / "janelas_sample.csv", index=False)
        print(f"\n{len(janelas)} janelas validas (pureza >= {PUREZA_MINIMA}) gravadas em "
              f"{OUT / 'janelas_sample.csv'}")
        return

    # Fase 3: todas as sessoes do ZIP pre-processado
    print(f"Processando todas as sessoes de {DATA_PATH}...")
    df_janelas_completo, diagnostico = processar_todas_sessoes(DATA_PATH)
    print(f"{len(df_janelas_completo)} janelas segmentadas de "
          f"{diagnostico['n_sessoes_validas']}/{diagnostico['n_sessoes_processadas']} sessoes validas.")

    df_amostrado = amostrar_estratificado(df_janelas_completo)
    print(f"{len(df_amostrado)} janelas apos amostragem estratificada "
          f"({N_JANELAS_POR_CLASSE}/classe, min {MIN_SUJEITOS_POR_CLASSE} sujeitos/classe).")

    resumo = {
        "diagnostico_sessoes": diagnostico,
        "n_janelas_totais_antes_amostragem": int(len(df_janelas_completo)),
        "n_janelas_amostradas": int(len(df_amostrado)),
        "distribuicao_operacao_amostrada": df_amostrado["operacao"].value_counts().to_dict(),
        "n_sujeitos_por_operacao_amostrada": df_amostrado.groupby("operacao")["sujeito"].nunique().to_dict(),
    }
    (OUT / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    df_amostrado.to_csv(OUT / "janelas_amostradas.csv", index=False)
    print(f"\n{len(df_amostrado)} janelas amostradas gravadas em {OUT / 'janelas_amostradas.csv'}")

    gerar_outputs_agregados(df_janelas_completo, OUT)
    gerar_features_e_anomalias(df_amostrado, OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true",
                         help="Roda contra o sample U0209/S0500 (Fase 1), sem exigir o ZIP completo")
    args = parser.parse_args()
    main(usar_sample=args.sample)
