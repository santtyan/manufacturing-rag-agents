"""
Legenda determinística de gráfico técnico, substituindo o VLM (moondream/qwen3-vl) na trilha
RGB do RAG multimodal. Mesmo princípio já validado em rag/rag_openpack_texto.py (RAG-HAR): o
texto vem de derivação determinística sobre o dado de origem (pandas/numpy), não de
interpretação de modelo -- custo ~zero e fidelidade 100% por construção, em vez de arriscar
alucinação numérica.

ACHADO REAL que motivou esta troca de arquitetura (2026-09-09/10): nenhum VLM local testado
(moondream 42,3%, granite3.2-vision:2b reprovado no smoke test, qwen3-vl:4b 77,5%) atingiu o
critério de promoção (>=90% de fidelidade E 100% sem números inventados,
eval/checks_fidelidade_caption.py). Pesquisa de estado da arte (2026-09-14) confirmou que isso
NÃO é "ainda não achamos o VLM certo" -- é limitação estrutural da classe: arXiv:2312.10160 mede
82,06% de erro factual em legendas de LVLM (GPT-4V incluído, 81,27%), com "Value Errors" como
tipo dominante; mesmo com tabela ground-truth injetada, a factualidade só chega a ~30% (o
gargalo é a INTERPRETAÇÃO por modelo, não a extração). Precedente revisado por pares: MatplotAlt
(Computer Graphics Forum 2025, arXiv:2503.20089) gera alt-text de figura matplotlib por template
sobre os dados de origem -- é exatamente esta técnica.

NOTA DE METODOLOGIA IMPORTANTE (não remover, contraria uma decisão anterior registrada em
rag/metadados_imagens_ground_truth.py): aquela docstring diz que usar o ground truth como
entrada do VLM "seria trapaça" -- e está CORRETA nesse contexto, porque avaliar um VLM que
deveria LER a imagem com o gabarito na mão invalidaria a avaliação. Mas este módulo não avalia
nenhum VLM -- ele SUBSTITUI o VLM por geração determinística. Usar os dados de origem aqui é o
método, não trapaça (é exatamente o que MatplotAlt faz). As imagens não são "lidas" por ninguém
neste módulo: o texto é derivado diretamente do DataFrame que gerou o próprio gráfico.

Contrato de saída idêntico ao de rag/legendas_cache*.json e rag/corpus_openpack_janelas.json:
lista de {"id","texto"}, pronta para RAGHibrido.indexar(documentos_customizados=...).

Uso: python rag/legendas_deterministicas.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import datasets  # noqa: F401 (import de protecao contra access violation torch x pyarrow -- ver skill rag-multimodal)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metadados_imagens_ground_truth import GROUND_TRUTH

RAIZ = Path(r"C:\Projetos\Harbor")
CSV_PIPELINE2 = RAIZ / "outputs" / "pipeline2_legacy_sensor" / "separacao_features_por_classe.csv"
CSV_PIPELINE4 = RAIZ / "outputs" / "pipeline4_five_axis_cnc" / "anomalias_temperatura.csv"
CORPUS_SAIDA = RAIZ / "rag" / "legendas_deterministicas.json"

# Cada imagem em GROUND_TRUTH vem de um dos 2 CSVs -- distinguido pelo prefixo das variaveis
# fonte, ja que nenhuma imagem mistura os dois CSVs (confirmado em gerar_imagens_sinteticas.py).
COLUNAS_PIPELINE4 = {
    "Spindle_motor_temperature_anomalo", "X_Axis_motor_temperature_anomalo",
    "Y_Axis_Motor_temperature_anomalo", "Z_Axis_Motor_temperature_anomalo",
    "General_temperature_anomalo",
}


def _carregar_dataframes() -> dict[str, pd.DataFrame]:
    return {
        "pipeline2": pd.read_csv(CSV_PIPELINE2),
        "pipeline4": pd.read_csv(CSV_PIPELINE4),
    }


def _df_de(nome_imagem: str, ground_truth: dict, dfs: dict) -> pd.DataFrame:
    """Decide qual dos 2 CSVs de origem esta imagem usa, pela intersecao de variaveis_fonte."""
    if set(ground_truth["variaveis_fonte"]) & COLUNAS_PIPELINE4:
        return dfs["pipeline4"]
    return dfs["pipeline2"]


def _fmt(valor: float) -> str:
    """Formata numero em pt-BR (virgula decimal) -- mesmo padrao textual usado em
    rag_openpack_texto.py, para casar com o check idioma_pt e ficar legivel em portugues.

    ACHADO REAL (2026-09-14): 2 casas decimais fixas perdem precisao para variaveis cuja
    diferenca entre classes e minuscula (ex. Oil_Quality_Index: Fault=0,855025 vs
    Normal=0,850770 -- arredondar as 2 para 0,86/0,85 e honesto por si so, mas o texto exibia
    so 1 dos 2 valores exatos, "0,86", que fica fora da margem de tolerancia do check
    sem_numeros_inventados calculada sobre a amplitude REAL nao-arredondada entre as classes).
    4 casas decimais para |valor| < 10 preserva a precisao suficiente para bater com a faixa
    real sem soar artificial (a mesma convencao ja usada em rag_openpack_texto.py para
    features pequenas nao existe -- decisao nova, mas mesma filosofia: nunca exibir um numero
    arredondado que fique fora da faixa do valor exato que ele deveria representar)."""
    casas = 4 if abs(valor) < 10 else 2
    return f"{valor:.{casas}f}".replace(".", ",")


def estatisticas_da_variavel(df: pd.DataFrame, coluna: str, por_classe: bool = True) -> dict:
    """Media/minimo/maximo/desvio da coluna, direto do DataFrame -- sem nenhuma interpretacao,
    so agregacao. Quando por_classe=True (boxplot/barra), agrupa por 'Target' (Fault/Normal);
    quando False (scatter/linha temporal), calcula sobre a serie inteira.

    ACHADO REAL (2026-09-14): separacao_features_por_classe.csv tem SO 2 LINHAS (1 media ja
    pre-calculada por classe, nao dado bruto por leitura) -- min/max/desvio com n=1 dao
    resultado degenerado (min=max=media, desvio=NaN). std(ddof=0) evita o NaN do ddof=1 padrao
    do pandas com 1 amostra, mas o numero ainda seria sempre 0 -- entao min/max/desvio so
    aparecem no texto quando n>1 de fato (ver _texto_boxplot_ou_barra)."""
    if por_classe and "Target" in df.columns:
        resultado = {}
        for classe, grupo in df.groupby("Target"):
            serie = grupo[coluna]
            resultado[classe] = {
                "media": float(serie.mean()),
                "minimo": float(serie.min()),
                "maximo": float(serie.max()),
                "desvio_padrao": float(serie.std(ddof=0)) if len(serie) > 1 else None,
                "n": len(serie),
            }
        return resultado
    serie = df[coluna]
    return {
        "geral": {
            "media": float(serie.mean()),
            "minimo": float(serie.min()),
            "maximo": float(serie.max()),
            "desvio_padrao": float(serie.std(ddof=0)) if len(serie) > 1 else None,
            "n": len(serie),
        }
    }


def _texto_barra_contagem_por_componente(gt: dict, df: pd.DataFrame) -> str:
    """As 2 imagens de 'anomalias por componente' (grafico_anomalias_componentes_cnc e a
    variante _ascendente) NAO sao 'variavel por classe' -- sao 5 colunas booleanas diferentes
    (1 por componente do CNC), sem coluna Target. O grafico soma cada coluna e ordena as
    barras -- o texto replica exatamente esse calculo (groupby por CLASSE nao se aplica aqui,
    ver _df_de/COLUNAS_PIPELINE4)."""
    contagens = df[gt["variaveis_fonte"]].sum()
    nomes = {c: c.replace("_temperature_anomalo", "").replace("_anomalo", "").replace("_", " ")
             for c in contagens.index}
    ascendente = gt["ranking"] == "ascendente"
    ordenado = contagens.sort_values(ascending=ascendente)

    linhas = [f'Título: "{gt["titulo"]}".',
              "Tipo de gráfico: barra.",
              f'Eixo X: {gt["eixo_x"]}. Eixo Y: {gt["eixo_y"]}.']
    for coluna, valor in ordenado.items():
        linhas.append(f"{nomes[coluna]}: {int(valor)} leituras anômalas.")
    extremo_txt = "menor" if ascendente else "maior"
    linhas.append(
        f"O componente com {extremo_txt} número de anomalias é {nomes[ordenado.index[0]]} "
        f"({int(ordenado.iloc[0])}), contra {nomes[ordenado.index[-1]]} "
        f"({int(ordenado.iloc[-1])})."
    )
    return " ".join(linhas)


def _texto_boxplot_ou_barra(gt: dict, df: pd.DataFrame) -> str:
    """Boxplot e barra do gerador sao ambos 'variavel por classe' -- boxplot mostra a
    distribuicao completa (media/min/max/desvio), barra so a media; o texto cobre os dois com
    a mesma riqueza estatistica, ja que o dado de origem e o mesmo groupby("Target").

    Excecao: graficos de 'contagem por componente' (5 colunas booleanas, sem Target) usam
    _texto_barra_contagem_por_componente em vez deste -- ver despacho em montar_texto_grafico."""
    coluna = gt["variaveis_fonte"][0]
    stats = estatisticas_da_variavel(df, coluna, por_classe=True)
    linhas = [f'Título: "{gt["titulo"]}".',
              f'Tipo de gráfico: {gt["tipo"]}.',
              f'Eixo X: {gt["eixo_x"]}. Eixo Y: {gt["eixo_y"]}.']
    for classe, s in stats.items():
        if s["n"] > 1:
            linhas.append(
                f"Classe {classe}: média {_fmt(s['media'])}, "
                f"variando de {_fmt(s['minimo'])} a {_fmt(s['maximo'])}, "
                f"desvio padrão {_fmt(s['desvio_padrao'])}."
            )
        else:
            linhas.append(f"Classe {classe}: valor {_fmt(s['media'])}.")
    if gt["ranking"] and len(stats) >= 2:
        ordenado = sorted(stats.items(), key=lambda kv: kv[1]["media"], reverse=True)
        linhas.append(
            f"A classe com maior média é {ordenado[0][0]} "
            f"({_fmt(ordenado[0][1]['media'])}), acima de {ordenado[-1][0]} "
            f"({_fmt(ordenado[-1][1]['media'])})."
        )
    return " ".join(linhas)


def _texto_scatter(gt: dict, df: pd.DataFrame) -> str:
    """Scatter e 'variavel X vs variavel Y' por classe -- as 2 variaveis_fonte, na ordem em que
    aparecem no eixo_x/eixo_y (a familia de distratores de eixos trocados depende dessa ordem
    estar certa, e ela vem direto do ground truth, nao de inferencia).

    Excecao: grafico_temperatura_scatter_tempo tem so 1 variavel_fonte (scatter contra INDICE,
    nao contra uma 2a variavel real) -- mesmo formato de dado da linha temporal, so que sem
    booleano de anomalia, entao usa estatistica por classe em vez de contagem de anomalia."""
    if len(gt["variaveis_fonte"]) == 1:
        coluna = gt["variaveis_fonte"][0]
        stats = estatisticas_da_variavel(df, coluna, por_classe=True)
        linhas = [f'Título: "{gt["titulo"]}".',
                  "Tipo de gráfico: dispersão (scatter) ao longo do índice das leituras.",
                  f'Eixo X: {gt["eixo_x"]}. Eixo Y: {gt["eixo_y"]}.']
        for classe, s in stats.items():
            if s["n"] > 1:
                linhas.append(
                    f"Classe {classe}: média {_fmt(s['media'])}, "
                    f"variando de {_fmt(s['minimo'])} a {_fmt(s['maximo'])}."
                )
            else:
                linhas.append(f"Classe {classe}: valor {_fmt(s['media'])}.")
        return " ".join(linhas)

    col_x_nome, col_y_nome = _mapear_eixos_para_colunas(gt)
    linhas = [f'Título: "{gt["titulo"]}".',
              "Tipo de gráfico: dispersão (scatter).",
              f'Eixo X: {gt["eixo_x"]}. Eixo Y: {gt["eixo_y"]}.']
    if "Target" in df.columns:
        for classe, grupo in df.groupby("Target"):
            linhas.append(
                f"Classe {classe}: {gt['eixo_x'].split(' (')[0]} médio "
                f"{_fmt(grupo[col_x_nome].mean())}, {gt['eixo_y'].split(' (')[0]} médio "
                f"{_fmt(grupo[col_y_nome].mean())}."
            )
    return " ".join(linhas)


def _mapear_eixos_para_colunas(gt: dict) -> tuple[str, str]:
    """As variaveis_fonte do ground truth nao vem necessariamente na ordem eixo_x/eixo_y (ex.
    "Pressure_bar","FlowRate_Lmin" sempre nessa ordem, mesmo quando o eixo_x real e vazao) --
    decide pela UNIDADE compartilhada entre o sufixo da variavel e o texto entre parenteses do
    rotulo do eixo (ex. "Pressure_bar" <-> "Pressão (bar)", "FlowRate_Lmin" <-> "Vazão (L/min)").

    ACHADO REAL (2026-09-14): a primeira versao usava o PREFIXO da variavel (antes do "_"), nao
    a unidade -- funcionava por coincidencia so quando o nome da variavel e do rotulo
    compartilhavam uma palavra (ex. "Voltage_V" -> "voltagem"), mas falhava silenciosamente
    para "Pressure_bar"/"FlowRate_Lmin" (nomes em ingles, rotulos em portugues, sem overlap de
    prefixo) -- o fallback (ordem da lista) sempre disparava, entao a familia inteira de
    distratores 'eixos trocados' (o proposito EXPLICITO dessas imagens) gerava o MESMO texto
    para as duas variantes, nunca capturando a troca. Unidade e o sinal robusto: esta sempre
    presente tanto no sufixo da variavel quanto entre parenteses no rotulo, em qualquer idioma."""
    variaveis = gt["variaveis_fonte"]
    eixo_x_lower, eixo_y_lower = gt["eixo_x"].lower(), gt["eixo_y"].lower()

    def unidade_de(nome_variavel: str) -> str:
        # Sufixo apos o ultimo "_" -- ex. "Pressure_bar" -> "bar", "FlowRate_Lmin" -> "lmin",
        # "Load_Percentage" -> "percentage" (tratado a parte, ve abaixo).
        return nome_variavel.rsplit("_", 1)[-1].lower()

    def rotulo_contem_unidade(rotulo_lower: str, unidade: str) -> bool:
        if unidade == "percentage":
            return "%" in rotulo_lower
        return unidade in rotulo_lower.replace("/", "")

    def rotulo_contem_radical_do_nome(rotulo_lower: str, nome_variavel: str) -> bool:
        # Fallback para quando a UNIDADE nao tem correspondencia direta entre idiomas (ex.
        # "Vibration_Level" -> "level" nao aparece em "vibração (nível)"). Usa os 5 primeiros
        # caracteres do nome da variavel (antes do "_") -- cobre "vibrat"/"vibra" e afins sem
        # exigir tradução exata, e e curto o suficiente para nao depender de acentuacao.
        radical = nome_variavel.split("_")[0].lower()[:5]
        return radical in rotulo_lower

    unidades = {v: unidade_de(v) for v in variaveis}

    def encontrar(rotulo_lower: str, excluir: str = None) -> str:
        candidatos = [v for v in variaveis if v != excluir]
        return (
            next((v for v in candidatos if rotulo_contem_unidade(rotulo_lower, unidades[v])), None)
            or next((v for v in candidatos if rotulo_contem_radical_do_nome(rotulo_lower, v)), None)
            or candidatos[0]
        )

    col_x = encontrar(eixo_x_lower)
    col_y = encontrar(eixo_y_lower, excluir=col_x) if len(variaveis) > 1 else col_x
    return col_x, col_y


def _texto_linha_temporal(gt: dict, df: pd.DataFrame) -> str:
    """Linha temporal do gerador e sempre uma serie booleana 0/1 de anomalia -- o dado
    determinante nao e media/desvio (sempre proximos de 0, pouco informativo), e sim a
    CONTAGEM e FRACAO de leituras marcadas como anomalas, e se ha alguma anomalia registrada."""
    coluna = gt["variaveis_fonte"][0]
    serie = df[coluna]
    n_anomalo = int(serie.sum())
    n_total = len(serie)
    fracao = (n_anomalo / n_total * 100) if n_total else 0.0
    linhas = [f'Título: "{gt["titulo"]}".',
              "Tipo de gráfico: linha temporal.",
              f'Eixo X: {gt["eixo_x"]}. Eixo Y: {gt["eixo_y"]}.']
    if n_anomalo == 0:
        linhas.append(
            f"A série permanece constante em 0 (normal) ao longo de todas as "
            f"{n_total} leituras -- nenhuma anomalia registrada neste componente."
        )
    else:
        linhas.append(
            f"{n_anomalo} de {n_total} leituras ({_fmt(fracao)}%) foram marcadas como "
            f"anômalas (valor 1) ao longo do tempo."
        )
    return " ".join(linhas)


def montar_texto_grafico(nome_imagem: str, ground_truth: dict, dfs: dict) -> str:
    """Despacha para o template certo por tipo de grafico -- unica funcao publica que decide
    QUAL template usar, mantendo cada _texto_* com uma responsabilidade so."""
    df = _df_de(nome_imagem, ground_truth, dfs)
    tipo = ground_truth["tipo"]
    if tipo == "barra" and len(ground_truth["variaveis_fonte"]) > 1:
        return _texto_barra_contagem_por_componente(ground_truth, df)
    if tipo in ("boxplot", "barra"):
        return _texto_boxplot_ou_barra(ground_truth, df)
    if tipo == "scatter":
        return _texto_scatter(ground_truth, df)
    if tipo == "linha_temporal":
        return _texto_linha_temporal(ground_truth, df)
    raise ValueError(f"Tipo de grafico desconhecido: {tipo!r} (imagem {nome_imagem})")


def gerar_corpus_legendas(caminho_saida: Path = CORPUS_SAIDA) -> list[dict]:
    """Gera a legenda determinística de cada imagem em GROUND_TRUTH e grava no schema
    {"id","texto"} esperado por RAGHibrido.indexar(documentos_customizados=...)."""
    dfs = _carregar_dataframes()
    documentos = [
        {"id": nome_imagem, "texto": montar_texto_grafico(nome_imagem, gt, dfs)}
        for nome_imagem, gt in GROUND_TRUTH.items()
    ]
    caminho_saida.write_text(json.dumps(documentos, indent=2, ensure_ascii=False), encoding="utf-8")
    return documentos


def main():
    documentos = gerar_corpus_legendas()
    print(f"Geradas {len(documentos)} legendas determinísticas -> {CORPUS_SAIDA}")
    for doc in documentos[:3]:
        print(f"\n[{doc['id']}]\n{doc['texto']}")


if __name__ == "__main__":
    main()
