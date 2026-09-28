"""
Frente 2 do plano "OpenPack: integrar indice numerico ao chat + cruzar outliers com o sinal IMU"
(2026-09-29).

PERGUNTA CIENTIFICA: o sinal IMU "enxerga" as irregularidades que um humano anotou em
annotation/openpack-outliers/? Se as janelas que coincidem com um evento de categoria
Struggling/Incident/Confused/etc. forem estatisticamente diferentes das janelas "normais" (sem
outlier sobreposto), isso valida o outlier como sinal detectavel por sensor -- nao e obvio a
priori: Confused (hesitacao) pode nao ter assinatura de movimento distinta, enquanto Struggling
(dificuldade fisica) provavelmente tem.

METODO: para cada uma das 6 sessoes com video/keypoints ja baixadas (Zenodo, ver
docs/openpack_licenca_e_atribuicao.md), marca cada janela IMU como tendo ou nao outlier
sobreposto (por sobreposicao temporal via unixtime, nao contencao integral -- eventos de outlier
sao curtos, mediana 1,25s, dentro de janelas de 4s). Compara a distribuicao das mesmas 48
features (reusa vetor_da_janela() de indice_numerico_openpack.py, NAO duplica extracao) entre
janelas com outlier de cada categoria vs. janelas sem nenhum outlier, via teste de Mann-Whitney U
(nao-parametrico, mesmo espirito do Wilcoxon ja usado no projeto para o LOSO), com correcao de
Bonferroni para as 48 comparacoes por categoria.

RESSALVAS DECLARADAS (nao descobrir depois -- ver docstring do plano):
- Amostra desbalanceada por categoria (Investigation=62 eventos totais no dataset completo,
  Additional=722) -- poder estatistico desigual entre categorias.
- Eventos sao curtos (mediana 1,25s) dentro de janelas de 4s -- uma janela pode conter so uma
  fracao do evento, diluindo o sinal. Isso e limitacao do metodo, nao ajustado aqui (mudar o
  tamanho da janela mudaria o corpus de producao, fora de escopo desta frente).
- So associacao estatistica, nunca causalidade.

Uso: python eval/avaliar_outliers_openpack.py
     Pre-requisito: outputs/pipeline8_openpack/janelas_amostradas.csv (mesmo de
     eval/indice_numerico_openpack.py) e os 6 zips do Zenodo ja baixados em
     C:\\Users\\USER\\Desktop\\OpenPack\\zenodo\\U*.zip (ver docs/openpack_licenca_e_atribuicao.md).
"""
import json
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HARBOR_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HARBOR_ROOT))
sys.path.insert(0, str(HARBOR_ROOT / "pipelines"))
sys.path.insert(0, str(HARBOR_ROOT / "eval"))

sys.path.insert(0, str(HARBOR_ROOT / "rag"))

from pipeline8_openpack import carregar_sessao  # noqa: E402
from indice_numerico_openpack import vetor_da_janela, NOMES_FEATURES, CANAIS  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
JANELAS_AMOSTRADAS = HARBOR_ROOT / "outputs" / "pipeline8_openpack" / "janelas_amostradas.csv"
ZIPS_DIR = Path(r"C:\Users\USER\Desktop\OpenPack\zenodo")

# As 6 sessoes com video/keypoints ja baixadas nesta sessao (ver docs/openpack_licenca_e_atribuicao.md)
SUJEITOS_COM_VIDEO = ["U0201", "U0206", "U0207", "U0208", "U0209", "U0210"]
SESSOES = ["S0100", "S0200", "S0300", "S0400", "S0500"]

# Nomes de features achatadas, mesma ordem de vetor_da_janela() (CANAIS x NOMES_FEATURES) --
# usado so para nomear as colunas no relatorio de saida, nao para extrair nada.
NOMES_VETOR = [f"{canal}__{feat}" for canal in CANAIS for feat in NOMES_FEATURES]

ALPHA = 0.05


def carregar_outliers_sessao(sujeito: str, sessao: str) -> pd.DataFrame | None:
    """Le annotation/openpack-outliers/{sessao}.csv de dentro do zip do Zenodo daquele sujeito.
    Retorna None se o zip nao existir (nao baixado) ou o CSV nao existir para essa sessao/for
    vazio (ex. U0201/S0100.csv, ver inventario da sessao anterior)."""
    caminho_zip = ZIPS_DIR / f"{sujeito}.zip"
    if not caminho_zip.exists():
        return None
    with zipfile.ZipFile(caminho_zip) as z:
        caminho_interno = f"annotation/openpack-outliers/{sessao}.csv"
        if caminho_interno not in z.namelist():
            return None
        conteudo = z.read(caminho_interno)
    if not conteudo.strip():
        return None
    df = pd.read_csv(BytesIO(conteudo))
    if len(df) == 0:
        return None
    df["start_ms"] = pd.to_datetime(df["start"], format="ISO8601").astype("int64") // 10**6
    df["end_ms"] = pd.to_datetime(df["end"], format="ISO8601").astype("int64") // 10**6
    return df


def marcar_janelas_com_outlier(df_janelas_sessao: pd.DataFrame, df_sinais: pd.DataFrame,
                                df_outliers: pd.DataFrame) -> pd.DataFrame:
    """Para cada janela IMU da sessao, marca 'categoria_outlier' = a categoria do PRIMEIRO evento
    de outlier cujo intervalo [start_ms, end_ms] SOBREPOE o intervalo de tempo da janela (nao
    precisa conter integralmente -- eventos sao curtos, mediana 1,25s, dentro de janelas de 4s).
    None se nenhum outlier sobrepoe a janela ('janela normal', grupo de controle)."""
    df_janelas_sessao = df_janelas_sessao.copy()
    categorias = []
    for _, janela in df_janelas_sessao.iterrows():
        t_inicio = df_sinais["timestamp_ms"].iloc[janela["indice_inicio"]]
        t_fim = df_sinais["timestamp_ms"].iloc[janela["indice_fim"] - 1]
        # Sobreposicao de intervalos: [t_inicio, t_fim] x [start_ms, end_ms]
        sobrepoe = (df_outliers["start_ms"] <= t_fim) & (df_outliers["end_ms"] >= t_inicio)
        eventos = df_outliers[sobrepoe]
        categorias.append(eventos["category"].iloc[0] if len(eventos) > 0 else None)
    df_janelas_sessao["categoria_outlier"] = categorias
    return df_janelas_sessao


def coletar_amostra() -> pd.DataFrame:
    """Constroi o dataframe completo: uma linha por janela das 6 sessoes com video, com as 48
    features (colunas F0..F47, nomeadas em NOMES_VETOR) + categoria_outlier (None = janela
    normal). So processa janelas ja presentes em janelas_amostradas.csv (nao reprocessa o
    dataset inteiro -- reusa a amostragem estratificada que o pipeline8 ja fez)."""
    df_janelas = pd.read_csv(JANELAS_AMOSTRADAS)
    linhas = []
    for sujeito in SUJEITOS_COM_VIDEO:
        for sessao in SESSOES:
            df_outliers = carregar_outliers_sessao(sujeito, sessao)
            if df_outliers is None:
                continue
            df_janelas_sessao = df_janelas[(df_janelas["sujeito"] == sujeito) & (df_janelas["sessao"] == sessao)]
            if len(df_janelas_sessao) == 0:
                continue
            df_sinais = carregar_sessao(sujeito, sessao, usar_sample=False)
            df_marcado = marcar_janelas_com_outlier(df_janelas_sessao, df_sinais, df_outliers)
            for _, janela in df_marcado.iterrows():
                bloco = df_sinais.iloc[janela["indice_inicio"]: janela["indice_fim"]].reset_index(drop=True)
                vetor = vetor_da_janela(bloco)
                linha = dict(zip(NOMES_VETOR, vetor))
                linha["sujeito"] = sujeito
                linha["sessao"] = sessao
                linha["janela_id"] = janela["janela_id"]
                linha["categoria_outlier"] = janela["categoria_outlier"]
                linhas.append(linha)
    return pd.DataFrame(linhas)


def testar_categoria(df: pd.DataFrame, categoria: str) -> dict:
    """Mann-Whitney U por feature: grupo A = janelas com outlier desta categoria sobreposto,
    grupo B = janelas SEM nenhum outlier sobreposto (categoria_outlier is None). Correcao de
    Bonferroni sobre as 48 features testadas (alpha=0.05 / 48)."""
    grupo_a = df[df["categoria_outlier"] == categoria]
    grupo_b = df[df["categoria_outlier"].isna()]

    if len(grupo_a) < 3 or len(grupo_b) < 3:
        return {"categoria": categoria, "n_com_outlier": len(grupo_a), "n_normal": len(grupo_b),
                "aviso": "amostra insuficiente (< 3 janelas em algum grupo), teste nao aplicado"}

    alpha_corrigido = ALPHA / len(NOMES_VETOR)
    features_significativas = []
    for feat in NOMES_VETOR:
        a = grupo_a[feat].to_numpy()
        b = grupo_b[feat].to_numpy()
        try:
            estatistica, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            continue  # amostra degenerada (ex. todos os valores identicos)
        if p < alpha_corrigido:
            features_significativas.append({"feature": feat, "p_valor": float(p),
                                              "mediana_com_outlier": float(np.median(a)),
                                              "mediana_normal": float(np.median(b))})

    return {
        "categoria": categoria,
        "n_com_outlier": len(grupo_a),
        "n_normal": len(grupo_b),
        "alpha_corrigido_bonferroni": alpha_corrigido,
        "n_features_testadas": len(NOMES_VETOR),
        "n_features_significativas": len(features_significativas),
        "features_significativas": sorted(features_significativas, key=lambda x: x["p_valor"]),
    }


def main():
    print("Coletando amostra (features + categoria de outlier sobreposto) das 6 sessoes com video...")
    df = coletar_amostra()

    if len(df) == 0:
        print("ERRO: nenhuma janela coletada -- verificar se os zips do Zenodo estao em "
              f"{ZIPS_DIR} e se janelas_amostradas.csv cobre as sessoes com video.")
        sys.exit(1)

    n_normal = df["categoria_outlier"].isna().sum()
    categorias = sorted(df["categoria_outlier"].dropna().unique())
    print(f"{len(df)} janelas coletadas: {n_normal} sem outlier sobreposto, "
          f"{len(df) - n_normal} com outlier em {len(categorias)} categorias: {categorias}")
    print()

    resultados = {}
    for categoria in categorias:
        r = testar_categoria(df, categoria)
        resultados[categoria] = r
        if "aviso" in r:
            print(f"{categoria:15} n_outlier={r['n_com_outlier']:4} -- {r['aviso']}")
        else:
            print(f"{categoria:15} n_outlier={r['n_com_outlier']:4}  n_normal={r['n_normal']:4}  "
                  f"features significativas (Bonferroni): {r['n_features_significativas']}/{r['n_features_testadas']}")
            for f in r["features_significativas"][:5]:
                print(f"    {f['feature']:40} p={f['p_valor']:.2e}  "
                      f"mediana outlier={f['mediana_com_outlier']:.3f}  mediana normal={f['mediana_normal']:.3f}")

    print()
    print("RESSALVAS: amostra desbalanceada por categoria; eventos curtos (mediana 1,25s) dentro "
          "de janelas de 4s podem diluir o sinal; resultado e associacao estatistica, nao causalidade.")

    saida = {
        "n_janelas_total": len(df),
        "n_janelas_normais": int(n_normal),
        "categorias_testadas": categorias,
        "alpha_nominal": ALPHA,
        "resultados_por_categoria": resultados,
    }
    caminho_saida = EVAL_DIR / "resultados_outliers_openpack.json"
    caminho_saida.write_text(json.dumps(saida, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nResultados salvos em {caminho_saida}")


if __name__ == "__main__":
    main()
