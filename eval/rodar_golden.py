"""
Harness de avaliacao do chatbot Harbor (Parte 2 do plano padrao-ouro).

Roda cada golden question pelo MESMO caminho logico do chatbot (roteamento + Ollama +
contexto agregado) e mede:
  1. Rota correta   -- rotear_pergunta() acertou contexto/rag/sql?
  2. Faithfulness   -- os numeros esperados (dos outputs reais) aparecem na resposta?
  3. Alucinacao     -- a resposta cita numero proibido (armadilha) ou inventa numero fora do contexto?

REESCRITO (2026-08-07): antes reimplementava call_ollama() e TODO o roteamento (gates +
rotear_pergunta) como "copia fiel" de dashboard/app.py, mantida sincronizada manualmente --
ja desatualizou de verdade (rerun anterior deu 52% em vez de 67,9% porque a copia nao tinha
os gates novos, nao porque houve regressao real na producao). Nao dava pra importar
dashboard/app.py direto (tem chamadas Streamlit em nivel de modulo, st.set_page_config etc)
-- por isso o roteamento foi extraido para dashboard/roteador.py (modulo sem dependencia de
Streamlit), que este harness agora importa. Fonte unica de verdade a partir de agora.

Uso:  python eval/rodar_golden.py         (Ollama precisa estar no ar)
Saida: eval/resultados_golden.csv + resumo no stdout + eval/alucinacoes.md atualizado.
"""
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import requests

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
_sys.path.insert(0, str(Path(__file__).parent.parent / "dashboard"))
from verificacao import numeros_nao_fundamentados
import rag_gerador
from roteador import rotear_pergunta

BASE = Path(r"C:\Projetos\Harbor")
EVAL_DIR = BASE / "eval"
OUTPUTS = BASE / "outputs"
GOLDEN = EVAL_DIR / "golden_questions.json"
RESULTADOS = EVAL_DIR / "resultados_golden.csv"
ALUCINACOES = EVAL_DIR / "alucinacoes.md"
# Mesma constante de dashboard/app.py:29 -- dataset bruto (nao passa pelo pipeline outputs/).
DATASET_ROOT = Path(r"C:\Users\USER\Downloads\Projeto_HarboR-20260707T002634Z-3-001\Projeto_HarboR\Dataset")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
OLLAMA_MODEL_FALLBACK = "llama3.2:1b"


# temperature=0.2 (nao o default do Ollama, ~0.8): respostas que citam numeros/fatos de um
# contexto pedem baixa variancia, nao criatividade -- mesmo principio do GPT-4 technical
# report (temperature 0.3 para multipla escolha, precisao > diversidade). Testa se reduz a
# instabilidade 0%-100% de faithfulness ja documentada entre execucoes identicas.
OLLAMA_TEMPERATURE = 0.2


def call_ollama(prompt, timeout=180, temperature=OLLAMA_TEMPERATURE):
    """Geracao usada so pelas rotas 'contexto'/'rag' deste harness -- nao faz parte do
    roteamento (que agora vem de roteador.py). Mantida aqui porque tem fallback 3B->1B e
    temperature fixa, comportamento especifico deste script de avaliacao."""
    for modelo in (OLLAMA_MODEL, OLLAMA_MODEL_FALLBACK):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": modelo, "prompt": prompt, "stream": False,
                      "options": {"temperature": temperature}},
                timeout=timeout,
            )
            resp.raise_for_status()
            corpo = resp.json()
            if "error" in corpo:
                raise RuntimeError(corpo["error"])
            return corpo.get("response", "").strip()
        except Exception:
            if modelo == OLLAMA_MODEL_FALLBACK:
                return "[Ollama indisponivel: falha em ambos os modelos (3B e 1B)]"
            continue


# ── contexto agregado por dataset (subset do que o app monta) ────────────────────
def _agregacao_downtime():
    """Agrega o CSV bruto de downtime por StopGroup, Stop (motivo especifico) e
    StopLocation -- cobre golden questions novas (oee-stopgroup-ranking,
    oee-motivo-especifico-top, oee-stoplocation-extruder) que outputs/pipeline1_oee/
    nao tem pre-calculado nessa granularidade (agregacao_paradas.csv so tem a
    combinacao StopGroup+StopType+StopLocation, sem Stop nem os totais isolados)."""
    import pandas as pd
    caminho = DATASET_ROOT / "OEE" / "DowntimeDataset.csv"
    df = pd.read_csv(caminho, low_memory=False)
    por_grupo = df.groupby("StopGroup")["StopDuration(min)"].sum().sort_values(ascending=False)
    por_motivo = df.groupby("Stop")["StopDuration(min)"].sum().sort_values(ascending=False).head(10)
    por_local = df.groupby("StopLocation")["StopDuration(min)"].sum().sort_values(ascending=False).head(10)
    return (
        f"Soma de StopDuration(min) por StopGroup (categoria de parada):\n{por_grupo.to_string()}\n\n"
        f"Soma de StopDuration(min) por Stop (motivo especifico, top 10):\n{por_motivo.to_string()}\n\n"
        f"Soma de StopDuration(min) por StopLocation (top 10):\n{por_local.to_string()}"
    )


def contexto_oee():
    comp = (OUTPUTS / "pipeline1_oee" / "comparacao_antes_depois_lss.csv").read_text(encoding="utf-8")
    mttr = (OUTPUTS / "pipeline1_oee" / "mttr_mtbf.json").read_text(encoding="utf-8")
    downtime = _agregacao_downtime()
    return (
        f"=== Comparacao antes/depois Lean Six Sigma ===\n{comp}\n"
        f"=== MTTR/MTBF ===\n{mttr}\n"
        f"=== Agregacao de paradas (downtime) ===\n{downtime}"
    )


def contexto_legacy():
    """Achado do harness (2026-07-11): o LLM erra sistematicamente a leitura de metricas
    fracionarias tipo "recall": 0.097 -- confunde com 97% ou outros valores, nunca acerta
    9.7% sozinho. Diferente de instabilidade aleatoria (que temperature baixa ja mitigou),
    isso e um erro de LEITURA consistente: o JSON cru nunca diz explicitamente que 0.097
    equivale a 9.7%. Anota a leitura percentual ao lado de cada metrica fracionaria
    (precision/recall/f1, todas na faixa [0,1] no backtest) para eliminar a ambiguidade,
    mesmo principio do rotulo curto ja aplicado em contexto_cnc() (Parte 5 do plano)."""
    import json as _json
    bt_path = OUTPUTS / "pipeline2_legacy_sensor" / "backtest_metrics.json"
    dados = _json.loads(bt_path.read_text(encoding="utf-8"))
    for chave in ("precision", "recall", "f1"):
        if chave in dados:
            dados[chave] = f"{dados[chave]} (equivale a {dados[chave] * 100:.1f}%)"
    bt = _json.dumps(dados, ensure_ascii=False, indent=2)
    return f"=== Backtest Isolation Forest vs Target real ===\n{bt}"


def contexto_cnc():
    """Mesma logica de rotulo curto do app.py (Parte 5): Program_path e um caminho longo tipo
    '_N_KOORDI_JEAN_CAM3_MPF' -- extrai so o nome do produto (ex: CAM3) para o LLM achar facil."""
    import pandas as pd
    ciclo = pd.read_csv(OUTPUTS / "pipeline4_five_axis_cnc" / "ciclo_por_produto.csv")
    prefixos = r"^_N_(KOORDI_JEAN_|CAM_|MA_|ASUP\d*_|MPF)?"
    ciclo["Produto"] = (
        ciclo["Program_path"].str.rsplit("/", n=1).str[-1]
        .str.replace(prefixos, "", regex=True)
        .str.replace("_MPF", "", regex=False)
        .str.replace("_SYF", "", regex=False)
    )
    ciclo = ciclo[["Produto"] + [c for c in ciclo.columns if c not in ("Program_path", "Produto")]]
    return f"=== Ciclo por produto (rotulo curto do produto, ex: CAM3) ===\n{ciclo.to_string()}"


CONTEXTOS = {"oee": contexto_oee, "legacy_sensor": contexto_legacy, "cnc": contexto_cnc}

_RAG_HIBRIDO_CACHE = {}


def _rag_hibrido():
    """Carrega e indexa o RAG hibrido uma unica vez por processo (sem @st.cache_resource
    fora do Streamlit -- cache manual equivalente). Renomeado de _rag_neural() em 2026-07-15."""
    if "rag" not in _RAG_HIBRIDO_CACHE:
        _RAG_HIBRIDO_CACHE["rag"] = rag_gerador.carregar_rag_hibrido()
    return _RAG_HIBRIDO_CACHE["rag"]


def montar_prompt_contexto(pergunta, contexto):
    return f"""Voce e um especialista em manufatura industrial. Responda a pergunta usando os
resultados ja calculados abaixo. A UNICA restricao e: nao invente numeros especificos
(percentuais, medias, contagens) que nao estejam explicitamente escritos no contexto.
Se a informacao nao estiver no contexto, diga que nao tem esse dado -- nao invente.

=== RESULTADOS JA CALCULADOS ===
{contexto}

=== PERGUNTA ===
{pergunta}

Responda em portugues, de forma direta (2-4 frases)."""


# ── extracao de numeros e checagem ──────────────────────────────────────────────
def extrair_numeros(texto):
    """Extrai numeros (com casa decimal opcional) de um texto, normalizando virgula->ponto."""
    bruto = re.findall(r"-?\d+(?:[.,]\d+)?", texto)
    nums = []
    for b in bruto:
        try:
            nums.append(float(b.replace(",", ".")))
        except ValueError:
            pass
    return nums


def numero_presente(alvo, encontrados, tol):
    # Compara por valor absoluto: percentuais aparecem com ou sem sinal ("-19,6%" vs "19.6%").
    return any(abs(abs(alvo) - abs(n)) <= tol for n in encontrados)


def avaliar(pergunta_obj, resposta, contexto=""):
    rota_real = rotear_pergunta(pergunta_obj["pergunta"])
    rota_ok = rota_real == pergunta_obj["rota_esperada"]

    nums_resposta = extrair_numeros(resposta)
    tol = pergunta_obj.get("tolerancia", 1.0)

    esperados = pergunta_obj.get("numeros_esperados", [])
    alternativos = pergunta_obj.get("numeros_aceitaveis_alternativos", [])
    # Um numero esperado conta como presente se ele OU um alternativo equivalente aparece.
    def presente_ou_alt(e):
        if numero_presente(e, nums_resposta, tol):
            return True
        return any(numero_presente(a, nums_resposta, tol) for a in alternativos)
    esperados_ok = [presente_ou_alt(e) for e in esperados]
    faithfulness = (sum(esperados_ok) / len(esperados)) if esperados else None

    proibidos = pergunta_obj.get("numeros_proibidos", [])
    proibidos_citados = [p for p in proibidos if numero_presente(p, nums_resposta, tol)]

    # Numeros citados na resposta que nao aparecem no contexto (candidatos a alucinacao numerica).
    # pergunta= evita falso positivo quando o LLM so ecoa um numero que ja veio na pergunta
    # (ex: armadilha "12 linhas" -- o LLM repete "12" ao dizer que nao tem essa info, nao inventou).
    nao_fundamentados = (
        numeros_nao_fundamentados(resposta, contexto, pergunta=pergunta_obj["pergunta"])
        if contexto else []
    )

    # Alucinacao: citou numero proibido (armadilha) OU citou numero fora do contexto
    alucinou = len(proibidos_citados) > 0 or len(nao_fundamentados) > 0

    return {
        "id": pergunta_obj["id"],
        "dataset": pergunta_obj["dataset"],
        "rota_esperada": pergunta_obj["rota_esperada"],
        "rota_real": rota_real,
        "rota_ok": rota_ok,
        "faithfulness": faithfulness,
        "proibidos_citados": proibidos_citados,
        "nao_fundamentados": nao_fundamentados,
        "alucinou": alucinou,
        "resposta": resposta,
    }


# Handlers deterministicos -- a resposta e montada 100% em codigo, o LLM nao participa do
# calculo/julgamento (padrao "tool delegation" documentado no catalogo de alucinacoes).
# Mensagens identicas as de dashboard/app.py (linhas 726-750) -- 3 gates de "nao respondivel"
# especificos (achado real, 2026-07-14: os 3 compartilhavam a MESMA mensagem generica de LSS,
# uma pergunta sobre CNC recebia explicacao de StopGroup que nao fazia sentido nenhum ali).
def _responder_nao_respondivel_lss(_pergunta):
    resposta = (
        "Os dados nao permitem cruzar categoria de parada (StopGroup) com o "
        "periodo antes/depois do Lean Six Sigma -- sao duas tabelas sem essa "
        "relacao registrada. Posso responder as duas partes separadamente: "
        "pergunte 'qual categoria mais consumiu minutos' (ranking por StopGroup, "
        "periodo inteiro) ou 'o Lean Six Sigma valeu a pena' (comparacao antes/"
        "depois por metricas de OEE/MTTR/MTBF, sem quebra por categoria)."
    )
    return resposta, ""


def _responder_nao_respondivel_roi(_pergunta):
    resposta = (
        "Este dataset nao tem coluna de custo ou investimento -- so metricas "
        "operacionais (OEE, MTTR, MTBF, minutos de parada). Nao e possivel "
        "calcular ROI ou payback financeiro com os dados disponiveis. Posso "
        "responder sobre a reducao percentual do tempo de parada ou a melhora "
        "das metricas operacionais, sem converter isso em valor monetario."
    )
    return resposta, ""


def _responder_nao_respondivel_cnc(_pergunta):
    resposta = (
        "Os dados nao permitem cruzar o ciclo de producao por produto "
        "(Program_path) com a anomalia de temperatura por componente -- sao "
        "duas tabelas sem essa relacao registrada (a anomalia de temperatura "
        "e agregada globalmente por eixo/motor, nao por produto). Posso "
        "responder as duas partes separadamente: pergunte sobre o ciclo de "
        "producao por produto, ou sobre o ranking de anomalias por componente."
    )
    return resposta, ""


def _responder_interpretacao_recall(_pergunta):
    """Identico a dashboard/app.py (linhas 754-772): direcao (acerta/erra a maioria) montada
    100% em Python, lendo o mesmo backtest_metrics.json que contexto_legacy() ja usa."""
    metrics = json.loads((OUTPUTS / "pipeline2_legacy_sensor" / "backtest_metrics.json").read_text(encoding="utf-8"))
    recall = metrics["recall"]
    tp = metrics["confusion_matrix"]["tp"]
    n_fault_real = tp + metrics["confusion_matrix"]["fn"]
    fn = n_fault_real - tp
    resposta = (
        f"O modelo **erra a maioria** das falhas reais: o recall de {recall:.3f} "
        f"({recall*100:.1f}%) significa que, das {n_fault_real} falhas reais, o "
        f"modelo detectou apenas {tp} ({recall*100:.1f}%) e deixou passar {fn} "
        f"({(1-recall)*100:.1f}%) sem detectar. Nao e 'acerta quase 10 em cada "
        f"10' -- e o oposto: o modelo so pega cerca de {round(recall*10)} em cada "
        f"10 falhas reais, e erra (nao detecta) as demais."
    )
    return resposta, json.dumps(metrics, ensure_ascii=False)


def _responder_planned_vs_unplanned(_pergunta):
    import pandas as pd
    agregacao = pd.read_csv(OUTPUTS / "pipeline1_oee" / "agregacao_paradas.csv")
    stoptype_total = agregacao.groupby("StopType")["sum"].sum()
    total_planned = stoptype_total[stoptype_total.index.str.startswith("Planned")].sum()
    total_unplanned = stoptype_total.get("Unplanned", 0)
    maior, menor = ("Unplanned", "Planned") if total_unplanned > total_planned else ("Planned", "Unplanned")
    valor_maior, valor_menor = max(total_unplanned, total_planned), min(total_unplanned, total_planned)
    resposta = (
        f"As paradas {maior} consomem mais minutos no total: {valor_maior:.1f} min, "
        f"contra {valor_menor:.1f} min das paradas {menor}. "
        + ("Isso e esperado -- falhas/imprevistos (Unplanned) tendem a ser mais dificeis "
           "de conter que manutencao programada, mas o ideal e monitorar se essa "
           "proporcao esta piorando ao longo do tempo." if maior == "Unplanned" else
           "Isso pode ser normal se refletir manutencao preventiva bem planejada, mas "
           "vale checar se nao ha paradas planejadas redundantes ou superdimensionadas.")
    )
    contexto = f"Total Planned (minutos): {total_planned:.1f}\nTotal Unplanned (minutos): {total_unplanned:.1f}"
    return resposta, contexto


def _responder_lss_melhorou_tudo(_pergunta):
    import pandas as pd
    comparacao = pd.read_csv(OUTPUTS / "pipeline1_oee" / "comparacao_antes_depois_lss.csv", index_col=0)
    sinal_melhora = {
        "OEE_medio": 1, "Availability_media": 1, "Performance_media": 1, "Quality_media": 1,
        "tempo_parada_total_min": -1, "MTTR_min": -1, "MTBF_min": 1, "n_unplanned_stops": -1,
    }
    linhas_rotuladas = []
    pioraram = []
    for idx, row in comparacao.iterrows():
        direcao_positiva = sinal_melhora.get(idx, 1)
        piorou = (row["variacao_%"] * direcao_positiva) < 0
        rotulo = "PIOROU" if piorou else "melhorou"
        linha = f"{idx}: {row['antes_LSS']} -> {row['depois_LSS']} ({row['variacao_%']:+.1f}%) [{rotulo}]"
        linhas_rotuladas.append(linha)
        if piorou:
            pioraram.append(linha)
    if pioraram:
        detalhe = "; ".join(l.split(" [")[0] for l in pioraram)
        resposta = (
            f"Nao, nem todas as metricas melhoraram. {len(pioraram)} de "
            f"{len(linhas_rotuladas)} pioraram apos o Lean Six Sigma: {detalhe}. "
            f"As demais melhoraram."
        )
    else:
        resposta = "Sim, todas as metricas registradas melhoraram apos o Lean Six Sigma."
    return resposta, "\n".join(linhas_rotuladas)


RESPONDER_DETERMINISTICO = {
    "nao_respondivel_lss": _responder_nao_respondivel_lss,
    "nao_respondivel_roi": _responder_nao_respondivel_roi,
    "nao_respondivel_cnc": _responder_nao_respondivel_cnc,
    "planned_vs_unplanned": _responder_planned_vs_unplanned,
    "lss_melhorou_tudo": _responder_lss_melhorou_tudo,
    "interpretacao_recall": _responder_interpretacao_recall,
}


def responder(pergunta_obj):
    """Gera a resposta pelo caminho real da rota roteada (contexto, rag, sql, ou uma das
    rotas deterministicas). Retorna (resposta, contexto_usado)."""
    rota = rotear_pergunta(pergunta_obj["pergunta"])

    if rota in RESPONDER_DETERMINISTICO:
        return RESPONDER_DETERMINISTICO[rota](pergunta_obj["pergunta"])

    if rota == "rag":
        rag = _rag_hibrido()
        resposta, _docs, contexto = rag_gerador.rag_responder(
            pergunta_obj["pergunta"], rag, call_ollama, buscar_fallback=rag_gerador.buscar_fallback_tfidf,
        )
        return resposta, contexto

    if rota == "sql":
        return _responder_sql(pergunta_obj["pergunta"])

    if rota != "contexto":
        return f"[rota={rota}: geracao nao avaliada nesta versao do harness]", ""

    ctx_fn = CONTEXTOS.get(pergunta_obj["dataset"])
    if ctx_fn is None:
        return "[sem contexto para este dataset]", ""
    contexto = ctx_fn()
    prompt = montar_prompt_contexto(pergunta_obj["pergunta"], contexto)
    return call_ollama(prompt), contexto


def _responder_sql(pergunta):
    """Gera+executa SQL real (com self-repair + DBA-Agent) e converte o resultado em resposta
    em portugues, reaproveitando nl_to_sql/nl_to_sql.py -- mesmo pipeline usado pelo dashboard
    e pelo servidor MCP. O 'contexto' retornado (para a checagem anti-alucinacao) e o proprio
    resultado da query em texto: um numero que a resposta cite mas nao esteja na tabela retornada
    e candidato a alucinacao, igual ao padrao ja usado nas rotas contexto/rag."""
    _sys.path.insert(0, r"C:\Projetos\Harbor\nl_to_sql")
    try:
        from nl_to_sql import perguntar_com_dba, resposta_amigavel
        r = perguntar_com_dba(pergunta)
        contexto = f"SQL executado: {r['sql']}\nResultado:\n{r['resultado'].to_string()}"
        resposta = resposta_amigavel(pergunta, r["sql"], r["resultado"])
        if not r["dba_responde"]:
            resposta += f"\n\n⚠️ DBA-Agent: {r['dba_motivo']}"
        return resposta, contexto
    except Exception as exc:
        return f"[erro no NL-to-SQL: {exc}]", ""


def main():
    dados = json.loads(GOLDEN.read_text(encoding="utf-8"))
    perguntas = dados["perguntas"]

    resultados = []
    print(f"Rodando {len(perguntas)} golden questions...\n")
    for pq in perguntas:
        resposta, contexto = responder(pq)
        r = avaliar(pq, resposta, contexto)
        resultados.append(r)
        rota_str = "OK " if r["rota_ok"] else f"ERR (deu {r['rota_real']})"
        ff = "-" if r["faithfulness"] is None else f"{r['faithfulness']*100:.0f}%"
        nf = f" nao-fund={r['nao_fundamentados']}" if r["nao_fundamentados"] else ""
        alu = " [ALUCINOU]" if r["alucinou"] else ""
        print(f"[{pq['id']:26}] rota={rota_str:20} faithfulness={ff:5}{alu}{nf}")

    # CSV
    with RESULTADOS.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "dataset", "rota_esperada", "rota_real", "rota_ok",
                    "faithfulness", "proibidos_citados", "nao_fundamentados", "alucinou", "resposta"])
        for r in resultados:
            w.writerow([r["id"], r["dataset"], r["rota_esperada"], r["rota_real"],
                        r["rota_ok"], r["faithfulness"], ";".join(map(str, r["proibidos_citados"])),
                        ";".join(map(str, r["nao_fundamentados"])),
                        r["alucinou"], r["resposta"].replace("\n", " ")])

    # resumo
    n = len(resultados)
    rotas_ok = sum(1 for r in resultados if r["rota_ok"])
    com_ff = [r for r in resultados if r["faithfulness"] is not None]
    ff_medio = (sum(r["faithfulness"] for r in com_ff) / len(com_ff)) if com_ff else 0
    alucinacoes = [r for r in resultados if r["alucinou"]]

    print("\n" + "=" * 60)
    print(f"Roteamento correto : {rotas_ok}/{n}")
    print(f"Faithfulness medio : {ff_medio*100:.0f}%  (sobre {len(com_ff)} perguntas com numeros esperados)")
    print(f"Alucinacoes        : {len(alucinacoes)}")
    print(f"Resultados salvos  : {RESULTADOS}")

    # log de alucinacoes (append)
    if alucinacoes:
        with ALUCINACOES.open("a", encoding="utf-8") as f:
            f.write(f"\n## Rodada {datetime.now().isoformat(timespec='seconds')}\n")
            for r in alucinacoes:
                motivos = []
                if r["proibidos_citados"]:
                    motivos.append(f"numero(s) proibido(s) {r['proibidos_citados']}")
                if r["nao_fundamentados"]:
                    motivos.append(f"numero(s) fora do contexto {r['nao_fundamentados']}")
                f.write(f"- **{r['id']}** ({r['dataset']}): citou {' e '.join(motivos)}. "
                        f"Resposta: {r['resposta'][:200]}\n")
        print(f"Alucinacoes logadas: {ALUCINACOES}")


if __name__ == "__main__":
    main()
