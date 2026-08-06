"""
Answer Consistency (FAB-Bench, benchmark de RAG para manufatura) — mede se o chatbot da a
MESMA qualidade de resposta quando a pergunta e repetida sem mudar nada. Alucinacao em LLM e
probabilistica, nao deterministica (achado desta sessao: a mesma pergunta variou de 100% a 0%
de faithfulness entre execucoes) -- rodar 1 vez so nunca prova nada.

Reusa golden_questions.json e as funcoes de rodar_golden.py (responder/avaliar). Roda cada
pergunta N_REPETICOES vezes e calcula: media, desvio padrao, e taxa de flip (quanto a resposta
muda de "acertou" pra "errou" entre execucoes identicas).

Uso: python eval/rodar_consistencia.py (Ollama precisa estar no ar; leva ~N x o tempo do
rodar_golden.py normal, entao roda so as perguntas de rota "contexto" e "rag" -- sql ainda
nao tem geracao avaliada por este harness).
"""
import csv
import json
import statistics
from pathlib import Path

from rodar_golden import GOLDEN, avaliar, responder

EVAL_DIR = Path(__file__).parent
RESULTADOS_CONSISTENCIA = EVAL_DIR / "resultados_consistencia.csv"
N_REPETICOES = 3


def rodar_consistencia(perguntas, n_repeticoes=N_REPETICOES):
    """Roda cada pergunta n_repeticoes vezes, retorna lista de {id, execucoes: [...], stats}."""
    resultados_por_pergunta = []

    for pq in perguntas:
        if pq.get("rota_esperada") not in ("contexto", "rag"):
            continue  # sql ainda nao tem geracao avaliada nesta versao do harness

        print(f"\n[{pq['id']}] rodando {n_repeticoes}x: {pq['pergunta']}")
        execucoes = []
        for i in range(n_repeticoes):
            resposta, contexto = responder(pq)
            r = avaliar(pq, resposta, contexto)
            execucoes.append(r)
            ff = "-" if r["faithfulness"] is None else f"{r['faithfulness']*100:.0f}%"
            alu = " [ALUCINOU]" if r["alucinou"] else ""
            print(f"  rodada {i+1}: faithfulness={ff}{alu}")

        faithfulness_vals = [e["faithfulness"] for e in execucoes if e["faithfulness"] is not None]
        alucinacoes_vals = [e["alucinou"] for e in execucoes]

        media = statistics.mean(faithfulness_vals) if faithfulness_vals else None
        desvio = statistics.pstdev(faithfulness_vals) if len(faithfulness_vals) > 1 else 0.0
        # "flip": a resposta mudou de comportamento (alucinou numa rodada e nao noutra, ou
        # faithfulness variou mais que 30 pontos percentuais entre a melhor e a pior rodada)
        flip_alucinacao = len(set(alucinacoes_vals)) > 1
        flip_faithfulness = (max(faithfulness_vals) - min(faithfulness_vals) > 0.3) if len(faithfulness_vals) > 1 else False

        resultados_por_pergunta.append({
            "id": pq["id"],
            "execucoes": execucoes,
            "faithfulness_media": media,
            "faithfulness_desvio": desvio,
            "flip_alucinacao": flip_alucinacao,
            "flip_faithfulness": flip_faithfulness,
            "inconsistente": flip_alucinacao or flip_faithfulness,
        })

    return resultados_por_pergunta


def main():
    dados = json.loads(GOLDEN.read_text(encoding="utf-8"))
    perguntas = dados["perguntas"]

    print(f"Answer Consistency: rodando cada pergunta de rota 'contexto' {N_REPETICOES}x...")
    resultados = rodar_consistencia(perguntas)

    with RESULTADOS_CONSISTENCIA.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "faithfulness_media", "faithfulness_desvio", "flip_alucinacao",
                    "flip_faithfulness", "inconsistente"])
        for r in resultados:
            w.writerow([r["id"], r["faithfulness_media"], round(r["faithfulness_desvio"], 3),
                        r["flip_alucinacao"], r["flip_faithfulness"], r["inconsistente"]])

    print("\n" + "=" * 70)
    print("RESUMO DE ANSWER CONSISTENCY (FAB-Bench)")
    print("=" * 70)
    n_inconsistentes = sum(1 for r in resultados if r["inconsistente"])
    for r in resultados:
        ff_media = "-" if r["faithfulness_media"] is None else f"{r['faithfulness_media']*100:.0f}%"
        status = "INCONSISTENTE" if r["inconsistente"] else "estavel"
        print(f"[{r['id']:26}] faithfulness_media={ff_media:5} desvio={r['faithfulness_desvio']:.2f} -> {status}")

    print(f"\n{n_inconsistentes}/{len(resultados)} perguntas mostraram inconsistencia entre rodadas.")
    print(f"Resultados salvos em: {RESULTADOS_CONSISTENCIA}")


if __name__ == "__main__":
    main()
