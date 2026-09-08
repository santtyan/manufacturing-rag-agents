"""
Harness de avaliacao do chatbot Harbor (Parte 2 do plano padrao-ouro).

Roda cada golden question pelo MESMO caminho logico do chatbot (roteamento + Ollama +
contexto agregado) e mede:
  1. Rota correta   -- rotear_pergunta() acertou contexto/rag/sql?
  2. Faithfulness   -- os numeros esperados (dos outputs reais) aparecem na resposta?
  3. Alucinacao     -- a resposta cita numero proibido (armadilha) ou inventa numero fora do contexto?

Nao importa dashboard/app.py (que puxa streamlit); replica a logica essencial de forma leve
para rodar standalone. call_ollama e copia fiel do app.py (so troca o modelo principal para
qwen2.5:7b); rotear_pergunta vem de dashboard/roteador.py (mesma fonte unica de verdade usada
por eval/rodar_golden.py -- ver docstring de roteador.py para o problema real que motivou a
extracao: uma copia manual do roteamento aqui ja tinha ficado desatualizada sem os gates novos,
achado real na auditoria de 2026-09-08).

Uso:  python eval/rodar_golden_qwen.py     (Ollama precisa estar no ar)
Saida: eval/resultados_golden_qwen.csv + resumo no stdout + eval/alucinacoes_qwen.md atualizado.
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
RESULTADOS = EVAL_DIR / "resultados_golden_qwen.csv"
ALUCINACOES = EVAL_DIR / "alucinacoes_qwen.md"
# Mesma constante de dashboard/app.py:29 -- dataset bruto (nao passa pelo pipeline outputs/).
DATASET_ROOT = Path(r"C:\Users\USER\Downloads\Projeto_HarboR-20260707T002634Z-3-001\Projeto_HarboR\Dataset")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_MODEL_FALLBACK = "llama3.2:1b"


# ── copia fiel de call_ollama (dashboard/app.py:191) ────────────────────────────
# temperature=0.2 (nao o default do Ollama, ~0.8): respostas que citam numeros/fatos de um
# contexto pedem baixa variancia, nao criatividade -- mesmo principio do GPT-4 technical
# report (temperature 0.3 para multipla escolha, precisao > diversidade). Testa se reduz a
# instabilidade 0%-100% de faithfulness ja documentada entre execucoes identicas.
OLLAMA_TEMPERATURE = 0.2


def call_ollama(prompt, timeout=180, temperature=OLLAMA_TEMPERATURE):
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
    rota_real = rotear_pergunta(pergunta_obj["pergunta"], ollama_model=OLLAMA_MODEL)
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


def responder(pergunta_obj):
    """Gera a resposta pelo caminho real da rota roteada (contexto, rag ou sql).
    Retorna (resposta, contexto_usado)."""
    rota = rotear_pergunta(pergunta_obj["pergunta"], ollama_model=OLLAMA_MODEL)

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
