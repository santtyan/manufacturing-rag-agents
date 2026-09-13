"""Testes unitarios do pipeline8_openpack (Fase 7g do plano de integracao OpenPack,
2026-09-14) -- cada teste cobre um bug REAL encontrado e corrigido durante a integracao,
nao cobertura generica (mesmo padrao de tests/test_checks_fidelidade_caption.py).

Rodar via: python -m pytest tests/test_pipeline8_openpack.py -v
"""
import numpy as np
import pandas as pd
import pytest

from pipelines.pipeline8_openpack import (
    _alocar_maior_resto,
    _carregar_sessao_preprocessada,
    _carregar_sessao_raw,
    CANAIS_RAW,
    COLUNAS_SINAL,
    SENSORES_ATR,
    segmentar_janelas,
    validar_sessao,
)


def _escrever_csv_preprocessado(caminho, n_colunas_sinal, nomes_formato_real=True):
    """Gera um CSV sintetico no formato pre-processado real (com header, "unixtime,operation,
    atr01/acc_x,...") -- usado para testar o contrato de 42 colunas de sinal. Se
    nomes_formato_real=False, usa nomes genericos (col0, col1...) so para testar a CONTAGEM de
    colunas, nao o mapeamento de nome."""
    if nomes_formato_real:
        nomes_sinal = [f"{sensor}/{canal}" for sensor in SENSORES_ATR for canal in CANAIS_RAW][:n_colunas_sinal]
    else:
        nomes_sinal = [f"col{i}" for i in range(n_colunas_sinal)]
    colunas = ["unixtime", "operation"] + nomes_sinal
    linhas = [[1000 + i * 33, 0] + [0.0] * n_colunas_sinal for i in range(5)]
    df = pd.DataFrame(linhas, columns=colunas)
    df.to_csv(caminho, index=False)


def test_carregar_sessao_preprocessada_rejeita_numero_errado_de_colunas(tmp_path):
    """Regressao direta do bug real (Fase 3): CSV com numero de colunas diferente de 42 deve
    levantar ValueError, nunca truncar nomes de coluna silenciosamente (achado real: a versao
    original fazia colunas[:df.shape[1]], que renomearia sensores errados sem aviso)."""
    caminho = tmp_path / "U9999-S9999.csv"
    _escrever_csv_preprocessado(caminho, n_colunas_sinal=len(COLUNAS_SINAL) - 2, nomes_formato_real=False)

    with pytest.raises(ValueError, match="esperado"):
        _carregar_sessao_preprocessada(caminho)


def test_carregar_sessao_preprocessada_aceita_numero_correto_de_colunas(tmp_path):
    """Contraprova do teste acima -- com o numero certo de colunas no formato real
    ("atr01/acc_x"), nao deve levantar erro, e o mapeamento indice->ID de operacao deve
    funcionar (regressao do achado real: coluna "operation" traz indice 0-10, nao o ID)."""
    caminho = tmp_path / "U0101-S0100.csv"
    _escrever_csv_preprocessado(caminho, n_colunas_sinal=len(COLUNAS_SINAL))

    df = _carregar_sessao_preprocessada(caminho)
    assert "timestamp_ms" in df.columns
    assert "operation_label" in df.columns
    assert set(COLUNAS_SINAL).issubset(set(df.columns))
    assert df["sujeito"].iloc[0] == "U0101"
    assert df["sessao"].iloc[0] == "S0100"
    # operation=0 no CSV sintetico deve virar operation_label=100 (Picking), nao ficar 0
    assert df["operation_label"].iloc[0] == 100


def test_validar_sessao_detecta_frequencia_fora_da_faixa():
    """DataFrame sintetico com timestamps espacados fora de [28,35] Hz (aqui, 10 Hz) deve dar
    valido=False -- checagem de sanidade que e o entregavel da Fase 1."""
    n = 20
    timestamps = [i * 100 for i in range(n)]  # 100ms = 10 Hz, fora da faixa [28,35]
    df = pd.DataFrame({
        "timestamp_ms": timestamps,
        "operation_label": [100] * n,
        "sujeito": ["U0101"] * n,
        "sessao": ["S0100"] * n,
        "operacao": ["Picking"] * n,
    })
    resumo = validar_sessao(df)
    assert resumo["freq_hz_ok"] is False
    assert resumo["valido"] is False


def test_segmentar_janelas_dataframe_vazio_nao_quebra():
    """Regressao direta do bug real corrigido na revisao cega dupla (Fase 1): DataFrame vazio
    deve retornar DataFrame com as colunas esperadas (incluindo 'operacao'), nunca lancar
    KeyError a jusante quando main() tenta acessar janelas['operacao'].value_counts()."""
    vazio = pd.DataFrame(columns=["timestamp_ms", "operation_label", "sujeito", "sessao", "operacao"])
    resultado = segmentar_janelas(vazio, freq_hz=30)

    assert len(resultado) == 0
    assert "operacao" in resultado.columns
    # Nao deve lancar excecao -- e a asserção real do bug corrigido:
    resultado["operacao"].value_counts().to_dict()


def test_segmentar_janelas_sessao_mais_curta_que_janela_nao_quebra():
    """Sessao com menos amostras que o tamanho de uma janela (ex. sample truncado) deve
    retornar zero janelas sem excecao -- mesma familia do bug de DataFrame vazio."""
    n = 5  # bem menor que uma janela de 4s a 30Hz (~120 amostras)
    df = pd.DataFrame({
        "timestamp_ms": [i * 33 for i in range(n)],
        "operation_label": [100] * n,
        "sujeito": ["U0101"] * n,
        "sessao": ["S0100"] * n,
        "operacao": ["Picking"] * n,
    })
    resultado = segmentar_janelas(df, freq_hz=30)
    assert len(resultado) == 0
    assert "operacao" in resultado.columns


def test_alocar_maior_resto_soma_exata():
    """Regressao direta do bug real de amostragem truncada (Fase 3): dividir 100 entre 21
    grupos por divisao inteira simples dava 84 (4 x 21); o metodo do maior resto
    (Hamilton/Hare-Niemeyer) deve somar exatamente 100."""
    alocacao = _alocar_maior_resto(100, 21)
    assert sum(alocacao) == 100
    assert len(alocacao) == 21
    # Nenhum grupo deve ficar com menos que o piso nem mais que o piso+1
    piso = 100 // 21
    assert all(v in (piso, piso + 1) for v in alocacao)


def test_alocar_maior_resto_divisao_exata():
    """Caso trivial sem resto -- todos os grupos devem receber a mesma cota."""
    alocacao = _alocar_maior_resto(100, 10)
    assert sum(alocacao) == 100
    assert all(v == 10 for v in alocacao)


def _escrever_sensor_raw(caminho, sensor, timestamps):
    """Gera um CSV de sensor raw sintetico (unixtime + 10 canais)."""
    n = len(timestamps)
    dados = {"unixtime": timestamps}
    for canal in CANAIS_RAW:
        dados[canal] = [0.0] * n
    pd.DataFrame(dados).to_csv(caminho, index=False)


def _escrever_anotacao_raw(caminho, intervalos):
    """Gera um CSV de anotacao raw sintetico. intervalos: lista de (id, start_ms, end_ms)."""
    linhas = []
    for id_op, start_ms, end_ms in intervalos:
        linhas.append({
            "id": id_op,
            "start": pd.Timestamp(start_ms, unit="ms").strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "end": pd.Timestamp(end_ms, unit="ms").strftime("%Y-%m-%dT%H:%M:%S.%f"),
        })
    pd.DataFrame(linhas).to_csv(caminho, index=False)


def test_merge_sensores_reporta_perda_quando_ha_desalinhamento(tmp_path):
    """Regressao direta do achado real da revisao cega dupla (2026-09-12): a primeira versao
    usava pd.concat(join="inner") por timestamp exato, que descartaria silenciosamente qualquer
    linha nao coincidente nos 4 sensores. Com merge_asof(tolerance=15ms), timestamps com jitter
    pequeno (dentro da tolerancia) devem ser preservados, e o diagnostico deve reportar zero
    perda quando o jitter esta dentro da tolerancia -- e reportar perda > 0 quando o
    desalinhamento excede a tolerancia."""
    pasta_sessao = tmp_path / "atr"
    pasta_anotacao = tmp_path / "annotation"
    pasta_sessao.mkdir()
    pasta_anotacao.mkdir()

    base = [1000 + i * 33 for i in range(20)]  # ~30Hz
    # atr01 com timestamps exatos; atr02/03/04 com jitter de 5ms (dentro da tolerancia de 15ms)
    _escrever_sensor_raw(pasta_sessao / "atr01_S0500.csv", "atr01", base)
    for sensor in ("atr02", "atr03", "atr04"):
        _escrever_sensor_raw(pasta_sessao / f"{sensor}_S0500.csv", sensor, [t + 5 for t in base])

    _escrever_anotacao_raw(pasta_anotacao / "operations_S0500.csv", [(100, base[0], base[-1] + 100)])

    df = _carregar_sessao_raw(pasta_sessao, pasta_anotacao, "U0101", "S0500")

    diagnostico = df.attrs["diagnostico_merge"]
    # Com jitter de 5ms (< tolerancia de 15ms), nao deve haver perda por desalinhamento --
    # e o diagnostico deve EXISTIR e ser consultavel (o bug original nem reportava isso).
    assert "n_linhas_descartadas_sem_par" in diagnostico
    assert diagnostico["n_linhas_descartadas_sem_par"] == 0
    assert len(df) == len(base)


def test_merge_sensores_com_desalinhamento_grande_reporta_perda(tmp_path):
    """Contraprova: quando o jitter excede a tolerancia (15ms), o merge_asof(nearest) ainda
    encontra ALGUM par (nearest sempre acha o mais proximo, mesmo fora da janela ideal) --
    mas com tolerance=15ms explicito, pares fora da tolerancia viram NaN e sao contados."""
    pasta_sessao = tmp_path / "atr"
    pasta_anotacao = tmp_path / "annotation"
    pasta_sessao.mkdir()
    pasta_anotacao.mkdir()

    base = [1000 + i * 33 for i in range(20)]
    _escrever_sensor_raw(pasta_sessao / "atr01_S0500.csv", "atr01", base)
    # atr02 com desalinhamento de 50ms -- muito maior que a tolerancia de 15ms
    for sensor in ("atr02", "atr03", "atr04"):
        _escrever_sensor_raw(pasta_sessao / f"{sensor}_S0500.csv", sensor, [t + 50 for t in base])

    _escrever_anotacao_raw(pasta_anotacao / "operations_S0500.csv", [(100, base[0], base[-1] + 100)])

    df = _carregar_sessao_raw(pasta_sessao, pasta_anotacao, "U0101", "S0500")
    diagnostico = df.attrs["diagnostico_merge"]
    assert diagnostico["n_linhas_descartadas_sem_par"] > 0


def test_overlap_de_anotacao_e_detectado(tmp_path):
    """Regressao do segundo achado da revisao cega dupla: merge_asof(direction='backward')
    mascara overlap de intervalos de anotacao sem avisar. O diagnostico deve contar
    n_intervalos_anotacao_sobrepostos > 0 quando ha overlap real."""
    pasta_sessao = tmp_path / "atr"
    pasta_anotacao = tmp_path / "annotation"
    pasta_sessao.mkdir()
    pasta_anotacao.mkdir()

    base = [1000 + i * 33 for i in range(20)]
    for sensor in SENSORES_ATR:
        _escrever_sensor_raw(pasta_sessao / f"{sensor}_S0500.csv", sensor, base)

    # Dois intervalos que se sobrepoem deliberadamente (operacao 100 termina APOS operacao 200 comecar)
    _escrever_anotacao_raw(pasta_anotacao / "operations_S0500.csv", [
        (100, base[0], base[10] + 100),
        (200, base[5], base[-1] + 100),
    ])

    df = _carregar_sessao_raw(pasta_sessao, pasta_anotacao, "U0101", "S0500")
    diagnostico = df.attrs["diagnostico_merge"]
    assert diagnostico["n_intervalos_anotacao_sobrepostos"] > 0
