"""
Teste de proveniencia numerica das legendas deterministicas (item 5.5 do plano de 2026-09-15).

MOTIVACAO: os checks de fidelidade (eval/checks_fidelidade_caption.py) validam por REGRA
heuristica (ex. "nao tem numero fora do intervalo esperado"). Este teste valida por
PROVENIENCIA: para cada numero que aparece no texto de uma legenda, confirma que ele bate
(dentro de tolerancia de arredondamento) com um valor DE FATO calculado a partir do CSV de
origem via rag.legendas_deterministicas.estatisticas_da_variavel() -- a mesma funcao que gerou
o numero, reaplicada de forma independente do texto.

Isso transforma a tese central do item 6b da skill rag-multimodal ("fidelidade 100% por
construcao") de afirmacao em invariante VERIFICADO: nao "medimos 100% nesta rodada", mas "aqui
esta o verificador que confirma que todo numero da legenda existe no dado de origem".

Diferente de tests/test_legendas_deterministicas.py (que testa a FUNCAO de geracao com dados
sinteticos controlados), este teste roda sobre o CORPUS REAL gerado
(rag/legendas_deterministicas.json) contra os CSVs reais -- e roda em segundos, sem VLM/Ollama.

Uso: python eval/teste_provenencia_numerica_legendas.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))

import pandas as pd
import datasets  # noqa: F401 (import de protecao contra access violation torch x pyarrow)

from legendas_deterministicas import (
    CSV_PIPELINE2, CSV_PIPELINE4, COLUNAS_PIPELINE4, estatisticas_da_variavel,
)
from metadados_imagens_ground_truth import GROUND_TRUTH

RAIZ = Path(r"C:\Projetos\Harbor")
CORPUS_LEGENDAS = RAIZ / "rag" / "legendas_deterministicas.json"

# Padrao de numero no texto: aceita formato brasileiro (virgula decimal) e negativos.
# Ex.: "69,40", "-3,21", "100". Nao casa numeros dentro de palavras (datas/codigos), que nao
# ocorrem no formato das legendas (confirmado por inspecao do corpus real).
PADRAO_NUMERO = re.compile(r"-?\d+,\d+|-?\d+")

# Frases-modelo ESTRUTURAIS do template -- nao sao estatisticas derivadas do CSV, sao texto fixo
# (nome do dominio, legenda de codificacao binaria). Removidas do texto ANTES de extrair
# numeros, para nao precisar de uma lista solta de "numeros permitidos em qualquer lugar" (que
# mascararia um numero inventado que por acaso fosse 0/1/5). Confirmado por inspecao do corpus
# real: "CNC 5 eixos" e o nome do equipamento (titulo fixo em GROUND_TRUTH); "(0=normal,
# 1=anomalo)" e a legenda da codificacao booleana, sempre com este texto exato em
# _texto_linha_temporal(). Se o template mudar essas frases, atualizar aqui tambem.
FRASES_ESTRUTURAIS_A_IGNORAR = [
    "CNC 5 eixos",
    "(0=normal, 1=anômalo)",  # anômalo, ja no encoding usado pelo corpus json
    "(valor 1)",  # segunda mencao ao mesmo "1" fixo da codificacao booleana, fim da frase
]


def _numeros_do_texto(texto: str) -> list[float]:
    """Extrai numeros do texto (convertendo virgula decimal para ponto), apos remover as
    frases-modelo estruturais que carregam numeros fixos nao derivados do CSV (ver
    FRASES_ESTRUTURAIS_A_IGNORAR)."""
    for frase in FRASES_ESTRUTURAIS_A_IGNORAR:
        texto = texto.replace(frase, "")
    return [float(m.replace(",", ".")) for m in PADRAO_NUMERO.findall(texto)]


def _valores_de_referencia(gt: dict, dfs: dict) -> set[float]:
    """Recalcula, a partir do CSV de origem, TODOS os valores numericos que poderiam
    legitimamente aparecer no texto desta legenda -- cobre os 4 templates de
    rag/legendas_deterministicas.py (_texto_boxplot_ou_barra, _texto_barra_contagem_por_
    componente, _texto_scatter, _texto_linha_temporal), reaplicando o MESMO calculo de cada um
    de forma independente do texto ja gerado."""
    nome_df = "pipeline4" if set(gt["variaveis_fonte"]) & COLUNAS_PIPELINE4 else "pipeline2"
    df = dfs[nome_df]

    valores = set()
    n_colunas = len(gt["variaveis_fonte"])
    todas_binarias = all(
        c in df.columns and df[c].dropna().isin([0, 1]).all() for c in gt["variaveis_fonte"]
    )

    if n_colunas > 1 and todas_binarias:
        # _texto_barra_contagem_por_componente: soma de cada coluna binaria + o par
        # (maior, menor) explicitamente citado na frase final -- ja cobertos pelas somas
        # individuais, que incluem min e max do conjunto.
        contagens = df[gt["variaveis_fonte"]].sum()
        for v in contagens:
            valores.add(float(v))
    elif n_colunas == 1 and todas_binarias:
        # _texto_linha_temporal: contagem de anomalias, total de leituras, e a fracao percentual
        # (ou o proprio total, quando n_anomalo==0 -- "todas as N leituras").
        coluna = gt["variaveis_fonte"][0]
        serie = df[coluna]
        n_anomalo = int(serie.sum())
        n_total = len(serie)
        valores.add(float(n_anomalo))
        valores.add(float(n_total))
        if n_anomalo > 0:
            fracao = round(n_anomalo / n_total * 100, 2) if n_total else 0.0
            valores.add(fracao)
    else:
        # _texto_boxplot_ou_barra / _texto_scatter: estatisticas media/min/max/desvio, por
        # classe (boxplot/barra/scatter-por-classe) e geral (serie inteira) -- gera as duas
        # visoes, ja que nao sabemos de antemao qual o template usou para este grafico.
        for coluna in gt["variaveis_fonte"]:
            if coluna not in df.columns:
                continue
            for por_classe in (True, False):
                stats = estatisticas_da_variavel(df, coluna, por_classe=por_classe)
                for grupo_stats in stats.values():
                    for chave in ("media", "minimo", "maximo", "desvio_padrao"):
                        v = grupo_stats.get(chave)
                        if v is not None:
                            valores.add(round(v, 2))

    return valores


def main():
    dfs = {"pipeline2": pd.read_csv(CSV_PIPELINE2), "pipeline4": pd.read_csv(CSV_PIPELINE4)}
    legendas = json.loads(CORPUS_LEGENDAS.read_text(encoding="utf-8"))

    total_numeros = 0
    numeros_sem_proveniencia = []
    legendas_com_falha = set()

    for legenda in legendas:
        nome = legenda["id"]
        gt = GROUND_TRUTH.get(nome)
        if gt is None:
            continue

        numeros_no_texto = _numeros_do_texto(legenda["texto"])
        valores_ref = _valores_de_referencia(gt, dfs)

        for n in numeros_no_texto:
            total_numeros += 1
            # Tolerancia de 0.015 -- cobre arredondamento de exibicao (_fmt usa 2 casas; a
            # comparacao aqui e feita apos round(...,2) em ambos os lados).
            bate = any(abs(n - v) < 0.015 for v in valores_ref)
            if not bate:
                numeros_sem_proveniencia.append((nome, n))
                legendas_com_falha.add(nome)

    print(f"Legendas verificadas   : {len(legendas)}")
    print(f"Numeros totais checados: {total_numeros}")
    print(f"Numeros SEM proveniencia no CSV de origem: {len(numeros_sem_proveniencia)}")
    if numeros_sem_proveniencia:
        print(f"Legendas afetadas ({len(legendas_com_falha)}):")
        for nome, n in numeros_sem_proveniencia[:20]:
            print(f"  - {nome}: numero {n} nao encontrado nas estatisticas do CSV de origem")
    else:
        print("\nTODOS os numeros das legendas tem proveniencia confirmada no CSV de origem.")
        print("A tese 'fidelidade 100% por construcao' (skill rag-multimodal, item 6b) esta "
              "verificada por invariante, nao so por check heuristico.")

    if numeros_sem_proveniencia:
        sys.exit(1)


if __name__ == "__main__":
    main()
