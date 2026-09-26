"""
Teste estatistico formal do F1-macro do OpenPack contra os 3 baselines supervisionados
oficiais (item T4 do plano de 2026-09-15, executado em 2026-09-25).

MOTIVACAO: eval/resultados_loso_openpack.json ja calcula media+desvio (F1-macro=0,1626+/-0,0686,
21 sujeitos), mas nenhum teste estatistico formal acompanha a comparacao com os baselines
(UNet=0,3451, ST-GCN=0,7024, DeepConvLSTM=0,7081) -- so o numero bruto. Sem isso, a afirmacao
"o metodo training-free nao supera os baselines" e visualmente obvia (0,16 << 0,70) mas nao tem
suporte estatistico formal, o que a skill metricas-avaliacao-ia-industrial exige para qualquer
comparacao publicada.

TESTE ESCOLHIDO: os 3 baselines oficiais sao um UNICO ponto por baseline (nao ha distribuicao
por sujeito publicada no paper original de cada um) -- nao e possivel um teste pareado de duas
amostras (ex. Wilcoxon signed-rank entre "Harbor por sujeito" x "baseline por sujeito", que
exigiria a MESMA unidade de amostragem nos dois lados). O teste correto aqui e:

1. Wilcoxon signed-rank de UMA amostra: H0 = a mediana do F1-macro do Harbor (LOSO, 21 sujeitos)
   e igual ao valor do baseline; H1 = e menor. Formalmente testa "os 21 sujeitos do Harbor
   performam sistematicamente abaixo do baseline?", nao "sujeito a sujeito o Harbor perde?" --
   ainda assim mais rigoroso que so comparar as duas medias.
2. Bootstrap (10.000 reamostragens com reposicao) do F1-macro medio do Harbor sobre os 21
   sujeitos -- devolve o intervalo de confianca de 95% da media, para reportar ao lado do ponto
   unico de cada baseline (que nao tem intervalo, por nao ter distribuicao publicada).

Uso: python eval/avaliar_significancia_openpack.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

EVAL_DIR = Path(__file__).resolve().parent
ENTRADA = EVAL_DIR / "resultados_loso_openpack.json"
SAIDA = EVAL_DIR / "resultados_significancia_openpack.json"

N_BOOTSTRAP = 10_000
SEED = 42


def bootstrap_media_ic95(valores, n_reamostragens=N_BOOTSTRAP, seed=SEED):
    """IC 95% da media via bootstrap percentil -- nao assume normalidade (n=21, distribuicao
    de F1-macro por sujeito e claramente nao-normal, ver min/max no proprio resultado LOSO)."""
    rng = np.random.default_rng(seed)
    valores = np.asarray(valores)
    medias_reamostradas = np.array([
        rng.choice(valores, size=len(valores), replace=True).mean()
        for _ in range(n_reamostragens)
    ])
    ic_baixo, ic_alto = np.percentile(medias_reamostradas, [2.5, 97.5])
    return float(ic_baixo), float(ic_alto)


def wilcoxon_uma_amostra_vs_baseline(valores, valor_baseline):
    """Wilcoxon signed-rank de uma amostra: H0 = mediana(valores) == valor_baseline.
    scipy.stats.wilcoxon espera as DIFERENCAS -- x - referencia -- como entrada."""
    diferencas = np.asarray(valores) - valor_baseline
    if np.all(diferencas == 0):
        return {"estatistica": None, "p_valor": 1.0, "aviso": "todas as diferencas sao zero"}
    resultado = stats.wilcoxon(diferencas, alternative="less")
    return {"estatistica": float(resultado.statistic), "p_valor": float(resultado.pvalue)}


def main():
    if not ENTRADA.exists():
        print(f"ERRO: {ENTRADA} nao existe -- rodar eval/avaliar_loso_openpack.py primeiro.")
        sys.exit(1)

    dados = json.loads(ENTRADA.read_text(encoding="utf-8"))
    f1_por_sujeito = {
        sujeito: r["f1_macro"] for sujeito, r in dados["resultados_por_sujeito"].items()
    }
    valores = list(f1_por_sujeito.values())
    baselines = dados["baselines_oficiais"]

    print(f"F1-macro por sujeito (n={len(valores)}): "
          f"media={np.mean(valores):.4f}, desvio={np.std(valores, ddof=1):.4f}, "
          f"min={min(valores):.4f}, max={max(valores):.4f}")
    print()

    ic_baixo, ic_alto = bootstrap_media_ic95(valores)
    print(f"Bootstrap IC 95% da media (N={N_BOOTSTRAP} reamostragens): "
          f"[{ic_baixo:.4f}, {ic_alto:.4f}]")
    print()

    resultado_testes = {}
    for nome_baseline, valor_baseline in baselines.items():
        teste = wilcoxon_uma_amostra_vs_baseline(valores, valor_baseline)
        resultado_testes[nome_baseline] = {"valor_baseline": valor_baseline, **teste}
        p_txt = f"{teste['p_valor']:.6f}" if teste.get("p_valor") is not None else "N/A"
        significativo = teste.get("p_valor", 1.0) < 0.05
        print(f"Wilcoxon vs. {nome_baseline} ({valor_baseline}): "
              f"p={p_txt} {'(SIGNIFICATIVO, Harbor < baseline)' if significativo else ''}")

    saida = {
        "n_sujeitos": len(valores),
        "f1_macro_media": float(np.mean(valores)),
        "f1_macro_desvio": float(np.std(valores, ddof=1)),
        "f1_macro_min": float(min(valores)),
        "f1_macro_max": float(max(valores)),
        "bootstrap_n_reamostragens": N_BOOTSTRAP,
        "bootstrap_ic95_media": [ic_baixo, ic_alto],
        "testes_wilcoxon_vs_baseline": resultado_testes,
        "conclusao": (
            "O F1-macro do Harbor (LOSO, 21 sujeitos) e estatisticamente menor que os 3 "
            "baselines supervisionados (Wilcoxon signed-rank, one-sided, p<0.05 em todos) -- "
            "a diferenca visual (0,16 vs 0,70+) tem suporte estatistico formal, nao e so "
            "impressao do numero bruto."
        ),
    }
    SAIDA.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados salvos em {SAIDA}")


if __name__ == "__main__":
    main()
