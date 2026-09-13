"""
Fase 2 do plano de integracao OpenPack: representacao de janelas de sensores IMU como texto,
seguindo o padrao RAG-HAR (arXiv:2512.08984, mai/2026) -- HAR training-free via retrieval, sem
embedar serie temporal bruta (comprovadamente "ineffective" para modelos de embedding de texto).

Ao contrario do RAG multimodal de imagem (rag/rag_multimodal_langchain.py), aqui NAO ha VLM: o
"captioner" e pandas/numpy determinístico, sem custo de inferencia e sem risco de alucinacao
numerica -- nao ha ChatOllama envolvido, entao nao ha o segfault especifico de ChatOllama +
sentence-transformers no mesmo processo (esse, sim, exigiria dois processos separados).

ACHADO REAL (2026-09-12, ao rodar buscar() pela 1a vez neste modulo): mesmo sem Ollama, importar
`sentence_transformers` (via RAGHibrido) SEM primeiro importar `pandas` e `datasets` ainda causa
o access violation torch x pyarrow.dataset ja documentado na skill rag-multimodal -- confirmado
por reproducao isolada (exit 139 sem os imports de protecao, exit 0 com eles). A suposicao
original desta docstring ("sem VLM = sem risco de segfault") estava incompleta: o risco de
access violation e uma restricao de ambiente independente de Ollama, dispara em qualquer import
de sentence_transformers neste processo Python especifico. Por isso os imports de protecao
abaixo (pandas, datasets) sao obrigatorios ANTES de qualquer import que puxe sentence_transformers
(inclusive indiretamente via rag_hibrido_langchain.RAGHibrido).

Contrato de saida identico ao de rag/legendas_cache*.json: lista de {"id","texto","fonte"},
pronta para RAGHibrido.indexar(documentos_customizados=...) sem nenhuma mudanca no RAG.

Uso: python rag/rag_openpack_texto.py --sample   (Fase 2, sobre outputs/pipeline8_openpack/janelas_sample.csv)
     python rag/rag_openpack_texto.py             (Fase 3+, sobre janelas_amostradas.csv)
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import datasets  # noqa: F401 (import de protecao contra access violation torch x pyarrow -- ver docstring acima)
import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, r"C:\Projetos\Harbor\pipelines")
from pipeline8_openpack import (  # noqa: E402 (import apos sys.path, padrao do projeto -- ver eval/rag_gerador.py)
    CLASSES_OPERACAO, SAMPLE_DIR, SENSORES_ATR, carregar_sessao,
)

OUT_PIPELINE = Path(r"C:\Projetos\Harbor\outputs\pipeline8_openpack")
CORPUS_SAIDA = Path(r"C:\Projetos\Harbor\rag\corpus_openpack_janelas.json")
CHROMA_DIR = Path(r"C:\Projetos\Harbor\rag\chroma_db_openpack")
COLECAO = "openpack_janelas_v1"

# As 8 features estatisticas do RAG-HAR (Secao III-B3 do paper): media, max, min, Q1, Q3,
# desvio padrao, mediana e numero de picos -- capturam tendencia central, variabilidade, forma
# da distribuicao e frequencia de eventos notaveis, sem exigir nenhum modelo de embedding de
# serie temporal (que a literatura considera "ineffective" para HAR training-free).
NOMES_FEATURES = ["media", "maximo", "minimo", "q1", "q3", "desvio_padrao", "mediana", "n_picos"]

# 6 canais derivados fisicamente interpretaveis, em vez das 40 colunas cruas -- 40 colunas x 8
# features explodiriam para 320 numeros por janela, ilegivel como texto. Cada canal e uma
# magnitude (norma do vetor 3D), que e invariante a rotacao do sensor no pulso -- mais estavel
# entre sujeitos que os eixos x/y/z crus, que dependem de como cada sujeito usa o sensor.
CANAIS = {
    "aceleracao_punho_esquerdo": ("atr01", ["acc_x", "acc_y", "acc_z"], "G"),
    "aceleracao_punho_direito": ("atr02", ["acc_x", "acc_y", "acc_z"], "G"),
    "rotacao_punho_esquerdo": ("atr01", ["gyro_x", "gyro_y", "gyro_z"], "dps"),
    "rotacao_punho_direito": ("atr02", ["gyro_x", "gyro_y", "gyro_z"], "dps"),
    "aceleracao_tronco": ("atr03_atr04_media", ["acc_x", "acc_y", "acc_z"], "G"),
    "inclinacao_quaternion": ("atr01", ["quat_w"], "rad"),
}

SEGMENTOS = ["completo", "inicio", "meio", "fim"]


def _magnitude(df: pd.DataFrame, sensor: str, eixos: list[str]) -> np.ndarray:
    """Norma do vetor 3D (ou valor unico se so 1 eixo, ex. quat_w -> angulo aproximado)."""
    if sensor == "atr03_atr04_media":
        v1 = df[[f"atr03_{e}" for e in eixos]].to_numpy()
        v2 = df[[f"atr04_{e}" for e in eixos]].to_numpy()
        vetor = (v1 + v2) / 2
    else:
        vetor = df[[f"{sensor}_{e}" for e in eixos]].to_numpy()

    if vetor.shape[1] == 1:
        # inclinacao_quaternion: quat_w = cos(theta/2) -- 2*arccos(w) aproxima o angulo de
        # rotacao total do sensor em relacao a referencia, mais interpretavel que w cru.
        return 2 * np.arccos(np.clip(vetor[:, 0], -1.0, 1.0))
    return np.linalg.norm(vetor, axis=1)


def features_estatisticas_janela(valores: np.ndarray) -> dict:
    """As 8 features do RAG-HAR para um vetor 1D de magnitude ja calculada."""
    if len(valores) == 0:
        return {nome: 0.0 for nome in NOMES_FEATURES}

    desvio = float(np.std(valores))
    prominencia_min = max(desvio * 0.5, 1e-6)
    picos, _ = find_peaks(valores, prominence=prominencia_min)

    return {
        "media": float(np.mean(valores)),
        "maximo": float(np.max(valores)),
        "minimo": float(np.min(valores)),
        "q1": float(np.percentile(valores, 25)),
        "q3": float(np.percentile(valores, 75)),
        "desvio_padrao": desvio,
        "mediana": float(np.median(valores)),
        "n_picos": int(len(picos)),
    }


def features_todos_canais(bloco: pd.DataFrame) -> dict[str, dict]:
    """Aplica features_estatisticas_janela aos 6 canais derivados para um bloco (janela ou
    sub-janela) de amostras cruas."""
    resultado = {}
    for nome_canal, (sensor, eixos, _unidade) in CANAIS.items():
        magnitude = _magnitude(bloco, sensor, eixos)
        resultado[nome_canal] = features_estatisticas_janela(magnitude)
    return resultado


def _discretizar_perfil(feats_completo: dict, limiares_intensidade: tuple, limiares_regularidade: tuple) -> tuple[str, str]:
    """Discretizacao qualitativa deterministica (pd.qcut sobre o corpus, nao LLM) -- o E5 casa
    melhor 'movimento intenso e irregular' com uma pergunta em portugues do que casa 'desvio
    0,84 G'. limiares_* sao os tercis (33/66 percentil) calculados sobre o CORPUS INTEIRO antes
    de processar janela a janela, para que a classificacao seja relativa ao dataset, nao a um
    valor absoluto arbitrario."""
    desvio_medio = np.mean([feats_completo[c]["desvio_padrao"] for c in CANAIS])
    media_valores = np.mean([feats_completo[c]["media"] for c in CANAIS if feats_completo[c]["media"] != 0])
    cv = desvio_medio / media_valores if media_valores else 0.0

    t1_int, t2_int = limiares_intensidade
    if desvio_medio <= t1_int:
        intensidade = "baixa"
    elif desvio_medio <= t2_int:
        intensidade = "moderada"
    else:
        intensidade = "alta"

    t1_reg, t2_reg = limiares_regularidade
    regularidade = "regular" if cv <= t1_reg else "irregular"

    return intensidade, regularidade


def montar_texto_janela(meta: dict, feats_segmento: dict, segmento: str, perfil: tuple[str, str]) -> str:
    """Template fixo em portugues -- o contrato de texto que entra no RAGHibrido. Uma linha por
    canal com as 8 features, fechando com a linha 'Perfil:' de discretizacao qualitativa."""
    linhas = [
        f"Janela de sensores vestiveis de {JANELA_SEGUNDOS_DESCRICAO}s (segmento {segmento}) "
        f"do sujeito {meta['sujeito']}, sessao {meta['sessao']}, operacao \"{meta['operacao']}\" "
        f"(classe {meta['operation_label']}) na linha de embalagem."
    ]

    rotulos_pt = {
        "aceleracao_punho_esquerdo": "Aceleracao do punho esquerdo",
        "aceleracao_punho_direito": "Aceleracao do punho direito",
        "rotacao_punho_esquerdo": "Rotacao do punho esquerdo",
        "rotacao_punho_direito": "Rotacao do punho direito",
        "aceleracao_tronco": "Aceleracao do tronco",
        "inclinacao_quaternion": "Inclinacao estimada",
    }
    for canal, (_, _, unidade) in CANAIS.items():
        f = feats_segmento[canal]
        rotulo = rotulos_pt[canal]
        linhas.append(
            f"{rotulo}: media {f['media']:.3f} {unidade}, maxima {f['maximo']:.3f} {unidade}, "
            f"minima {f['minimo']:.3f} {unidade}, desvio {f['desvio_padrao']:.3f} {unidade}, "
            f"quartis {f['q1']:.3f}-{f['q3']:.3f} {unidade}, mediana {f['mediana']:.3f} {unidade}, "
            f"{f['n_picos']} picos."
        )

    intensidade, regularidade = perfil
    n_picos_total = sum(feats_segmento[c]["n_picos"] for c in CANAIS)
    linhas.append(
        f"Perfil: movimento {intensidade} e {regularidade}, com {n_picos_total} picos de "
        f"aceleracao/rotacao ao todo neste segmento."
    )
    return "\n".join(linhas)


JANELA_SEGUNDOS_DESCRICAO = 4  # sincronizado com pipelines.pipeline8_openpack.JANELA_SEGUNDOS


def gerar_corpus_rag(sinais_por_sessao: dict, df_janelas: pd.DataFrame,
                      caminho_saida: Path = CORPUS_SAIDA, multi_vetor: bool = True) -> list[dict]:
    """Gera o corpus multi-vetor (4 documentos por janela: completo/inicio/meio/fim, padrao
    RAG-HAR) a partir das janelas ja segmentadas por pipeline8_openpack.segmentar_janelas().

    sinais_por_sessao: dict {(sujeito, sessao): df_sinais}, onde cada df_sinais e o DataFrame
    completo de carregar_sessao() para aquela sessao (amostra a amostra, 40 colunas de sinal) --
    necessario porque, a partir da Fase 3, df_janelas mistura janelas de VARIAS sessoes/sujeitos
    (amostragem estratificada), entao os indices_inicio/fim de cada janela so fazem sentido
    dentro do df_sinais da SUA PROPRIA sessao, nao de um unico DataFrame global.
    """
    if len(df_janelas) == 0:
        caminho_saida.write_text("[]", encoding="utf-8")
        return []

    def _bloco_da_janela(janela):
        df_sinais = sinais_por_sessao[(janela["sujeito"], janela["sessao"])]
        return df_sinais.iloc[janela["indice_inicio"]: janela["indice_fim"]]

    # Limiares de discretizacao calculados sobre o CORPUS (todas as janelas), nao por janela --
    # ver docstring de _discretizar_perfil.
    desvios_medios, cvs = [], []
    for _, janela in df_janelas.iterrows():
        bloco = _bloco_da_janela(janela)
        feats = features_todos_canais(bloco)
        desvio_medio = np.mean([feats[c]["desvio_padrao"] for c in CANAIS])
        media_valores = np.mean([feats[c]["media"] for c in CANAIS if feats[c]["media"] != 0])
        desvios_medios.append(desvio_medio)
        cvs.append(desvio_medio / media_valores if media_valores else 0.0)

    limiares_intensidade = tuple(np.percentile(desvios_medios, [33, 66])) if desvios_medios else (0.0, 0.0)
    limiares_regularidade = tuple(np.percentile(cvs, [50, 50])) if cvs else (0.0, 0.0)

    documentos = []
    for _, janela in df_janelas.iterrows():
        bloco = _bloco_da_janela(janela).reset_index(drop=True)
        n = len(bloco)
        meta = {
            "sujeito": janela["sujeito"], "sessao": janela["sessao"],
            "operacao": janela["operacao"], "operation_label": janela["operation_label"],
        }

        sub_blocos = {"completo": bloco}
        if multi_vetor and n >= 3:
            terco = n // 3
            sub_blocos["inicio"] = bloco.iloc[:terco]
            sub_blocos["meio"] = bloco.iloc[terco: 2 * terco]
            sub_blocos["fim"] = bloco.iloc[2 * terco:]

        for segmento, sub_bloco in sub_blocos.items():
            feats_segmento = features_todos_canais(sub_bloco)
            perfil = _discretizar_perfil(feats_segmento, limiares_intensidade, limiares_regularidade)
            texto = montar_texto_janela(meta, feats_segmento, segmento, perfil)
            documentos.append({
                "id": f"{janela['janela_id']}__{segmento}",
                "texto": texto,
                "fonte": janela["janela_id"],
            })

    caminho_saida.write_text(json.dumps(documentos, indent=2, ensure_ascii=False), encoding="utf-8")
    return documentos


def indexar_janelas(documentos_customizados: list[dict], forcar: bool = False):
    """Indexa o corpus de janelas no RAGHibrido de producao, coleção separada -- mesmo padrao de
    rag_multimodal_langchain.py::indexar_imagens(). Diferente de la, aqui nao ha risco de
    segfault (sem ChatOllama envolvido), entao roda no mesmo processo sem problema."""
    sys.path.insert(0, r"C:\Projetos\Harbor\rag")
    from rag_hibrido_langchain import RAGHibrido

    rag = RAGHibrido(chroma_dir=CHROMA_DIR, colecao=COLECAO)
    rag.indexar(forcar=forcar, documentos_customizados=documentos_customizados)
    return rag


def main(usar_sample: bool):
    if usar_sample:
        caminho_janelas = OUT_PIPELINE / "janelas_sample.csv"
        df_janelas = pd.read_csv(caminho_janelas)
        sinais_por_sessao = {("U0209", "S0500"): carregar_sessao("U0209", "S0500", usar_sample=True)}
    else:
        caminho_janelas = OUT_PIPELINE / "janelas_amostradas.csv"
        df_janelas = pd.read_csv(caminho_janelas)
        # Carrega so as sessoes que de fato aparecem nas 1000 janelas amostradas (nao as 102
        # inteiras) -- evita reprocessar ~5,8M linhas quando so uma fracao entra no corpus.
        pares_sessao = df_janelas[["sujeito", "sessao"]].drop_duplicates().itertuples(index=False)
        sinais_por_sessao = {}
        for sujeito, sessao in pares_sessao:
            sinais_por_sessao[(sujeito, sessao)] = carregar_sessao(sujeito, sessao, usar_sample=False)
        print(f"Sinais carregados para {len(sinais_por_sessao)} sessoes distintas.")

    print(f"Gerando corpus a partir de {len(df_janelas)} janelas...")

    documentos = gerar_corpus_rag(sinais_por_sessao, df_janelas, caminho_saida=CORPUS_SAIDA)
    print(f"{len(documentos)} documentos gerados (multi-vetor: {len(df_janelas)} janelas x ~4) "
          f"em {CORPUS_SAIDA}")

    print("Indexando no RAGHibrido...")
    indexar_janelas(documentos, forcar=True)
    print(f"Indexado em {CHROMA_DIR} / colecao '{COLECAO}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true",
                         help="Roda contra janelas_sample.csv (Fase 2, sample U0209) em vez "
                              "de janelas_amostradas.csv (Fase 3+, corpus completo de 1000 janelas)")
    args = parser.parse_args()
    main(usar_sample=args.sample)
