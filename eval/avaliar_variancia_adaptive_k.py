"""
Medicao de variancia do Adaptive-k (item 7.3/8 do plano de 2026-09-15) -- roda
avaliar_ablacao_k_rag.py --k "" --adaptive N vezes (default 5) e reporta media +/- desvio
padrao da faithfulness, seguindo o padrao publicado (rodar N=3-10, reportar media+-sigma, nunca
escore unico de um sistema nao-deterministico -- ver secao 7.3 do plano, oneuptime.com/blog/
post/2026-08-31-measure-llm-eval-variance-thresholds).

MOTIVACAO: o +11,6pp do Adaptive-k sobre o melhor k fixo (88,5% vs 76,9%, ver memoria
ablacao_k_adaptive_interrompida_2026-09-14) vem de UMA execucao de um sistema nao-deterministico
(LLM com temperature>0). A decisao de promover usar_adaptive_k=True para producao esta parada
desde 2026-09-14 justamente esperando essa validacao.

So mede o Adaptive-k (nao os k fixos) -- os k fixos ja tem 1 execucao bem estabelecida e nao sao
o alvo da decisao de promocao; economiza ~3x o tempo (cada execucao completa de k=3,5,10+adaptive
leva ~1h40, so adaptive leva ~25min).

Uso: python eval/avaliar_variancia_adaptive_k.py [--n 5]
     Pre-requisito: Ollama rodando (mesmo pre-requisito do script original).
"""
import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
RESULTADO_ORIGINAL = EVAL_DIR / "resultados_ablacao_k_rag.json"
SAIDA_VARIANCIA = EVAL_DIR / "resultados_variancia_adaptive_k.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="numero de execucoes (default 5)")
    args = parser.parse_args()

    execucoes = []
    for i in range(1, args.n + 1):
        print(f"\n{'=' * 74}\nExecucao {i}/{args.n}\n{'=' * 74}")
        t0 = time.time()
        # ACHADO REAL (2026-09-16): timeout=2400 (40min) matava o subprocesso SILENCIOSAMENTE
        # perto do fim de execucoes que legitimamente levam ~40min nesta maquina sem GPU (a
        # Execucao 1 mediu 2366s = quase o teto) -- 5 tentativas anteriores foram interpretadas
        # como "Ollama travado" quando na verdade era este timeout interno explodindo o
        # subprocess.run() sem ser capturado (TimeoutExpired propagava e derrubava o script
        # inteiro, sem log de erro no stdout). Timeout aumentado com folga real (1h30) e a
        # excecao agora e capturada -- uma execucao lenta demais vira "FALHA, pulando", nao
        # derruba as execucoes seguintes.
        try:
            resultado = subprocess.run(
                [sys.executable, "-u", "eval/avaliar_ablacao_k_rag.py", "--k", "", "--adaptive"],
                cwd=EVAL_DIR.parent, timeout=5400,
            )
        except subprocess.TimeoutExpired:
            duracao = time.time() - t0
            print(f"FALHA na execucao {i}: excedeu 5400s ({duracao:.1f}s decorridos) -- pulando")
            continue
        duracao = time.time() - t0
        if resultado.returncode != 0:
            print(f"FALHA na execucao {i} (returncode={resultado.returncode}) -- pulando")
            continue

        dados = json.loads(RESULTADO_ORIGINAL.read_text(encoding="utf-8"))
        ag = dados["por_k"]["adaptive"]["agregados"]
        execucoes.append({
            "execucao": i,
            "faithfulness_medio": ag["faithfulness_medio"],
            "alucinacoes": ag["alucinacoes"],
            "duracao_media_s": ag["duracao_media_s"],
            "duracao_total_s": ag["duracao_total_s"],
            "duracao_wall_clock_s": round(duracao, 1),
        })
        print(f"Execucao {i}: faithfulness={ag['faithfulness_medio']:.4f}, "
              f"alucinacoes={ag['alucinacoes']}, duracao={duracao:.1f}s")

    if not execucoes:
        print("\nNENHUMA execucao completou com sucesso.")
        sys.exit(1)

    faithfulness_vals = [e["faithfulness_medio"] for e in execucoes]
    media = statistics.mean(faithfulness_vals)
    desvio = statistics.stdev(faithfulness_vals) if len(faithfulness_vals) > 1 else 0.0
    cv = desvio / media if media else 0.0

    print(f"\n{'=' * 74}\nRESULTADO -- variancia do Adaptive-k em {len(execucoes)} execucoes\n{'=' * 74}")
    print(f"Faithfulness media    : {media:.4f}")
    print(f"Desvio padrao         : {desvio:.4f}")
    print(f"Coeficiente variacao  : {cv:.4f} (alvo < 0.05, padrao 2026)")
    print(f"Min / Max             : {min(faithfulness_vals):.4f} / {max(faithfulness_vals):.4f}")
    print()

    # Comparacao com o melhor k fixo (k=10, 76,9% -- 1 execucao, ver
    # resultados_ablacao_k_rag.json historico) para decidir a promocao.
    melhor_k_fixo = 0.769
    margem_inferior = media - desvio
    if margem_inferior > melhor_k_fixo:
        veredito = ("PROMOVER: mesmo no pior caso dentro de 1 desvio padrao "
                     f"({margem_inferior:.4f}), o Adaptive-k supera o melhor k fixo "
                     f"({melhor_k_fixo:.4f}).")
    elif media > melhor_k_fixo and cv < 0.10:
        veredito = ("PROMOVER COM RESSALVA: media supera o k fixo e a variancia e "
                     "razoavelmente baixa, mas o pior caso dentro de 1 sigma nao supera.")
    else:
        veredito = ("NAO PROMOVER AINDA: a vantagem sobre o melhor k fixo nao sobrevive a "
                     "variancia medida -- a diferenca original pode ter sido ruido, nao sinal. "
                     "Isso nao refuta o Adaptive-k, apenas significa que a vantagem nao esta "
                     "demonstrada com confianca suficiente.")
    print(veredito)

    saida = {
        "n_execucoes_solicitadas": args.n,
        "n_execucoes_completas": len(execucoes),
        "execucoes": execucoes,
        "faithfulness_media": round(media, 4),
        "faithfulness_desvio": round(desvio, 4),
        "coeficiente_variacao": round(cv, 4),
        "faithfulness_min": round(min(faithfulness_vals), 4),
        "faithfulness_max": round(max(faithfulness_vals), 4),
        "melhor_k_fixo_referencia": melhor_k_fixo,
        "veredito": veredito,
    }
    SAIDA_VARIANCIA.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados salvos em {SAIDA_VARIANCIA}")


if __name__ == "__main__":
    main()
