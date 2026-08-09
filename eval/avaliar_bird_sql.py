"""
Avaliacao do NL-to-SQL do Harbor contra um subset do BIRD-SQL (Li et al. 2023, NeurIPS) --
o benchmark academico padrao-ouro para text-to-SQL sobre bancos "sujos"/realistas, citado
como referencia junto de Spider nos artigos de estado da arte (ver memoria
"referencias_rag_padrao_ouro.md" e pesquisa 2026-07-12 sobre benchmarks padrao-ouro).

LIMITACAO METODOLOGICA IMPORTANTE (leia antes de citar o numero em qualquer lugar):
O BIRD oficial mede "execution accuracy" -- roda a query gerada contra o banco SQLite REAL
(populado com dados) e compara o RESULTADO com o gold. Os arquivos .sqlite oficiais somam
~30-40GB (nao baixados aqui, fora de escopo por ROI). Este script usa o dataset
`xu3kev/BIRD-SQL-data-train` do HuggingFace, que traz o SCHEMA (DDL) + pergunta + SQL gold,
mas SEM os dados populados -- entao mede uma metrica MAIS FRACA: "correspondencia estrutural"
(compara o SQL gerado com o SQL gold normalizado, sem executar nenhum dos dois).
Isso NAO e diretamente comparavel ao numero oficial do estado da arte (~82% no BIRD test set,
via execution accuracy) -- e uma aproximacao honesta, nao um benchmark formal replicado.

NAO usa self-repair nem DBA-Agent (perguntar_com_dba, que a producao real tem): os dois
dependem de EXECUTAR a query contra um banco real e observar erro/resultado -- sem os dados
populados (ver paragrafo acima), nao ha erro do Postgres para reenviar ao LLM nem resultado
para o DBA-Agent avaliar. gerar_sql_com_schema() (nl_to_sql/nl_to_sql.py) e o maximo de reuso
possivel dado essa restricao real: mesmo prompt/logica de geracao da producao, schema
dinamico do BIRD no lugar do ESQUEMA fixo do Harbor.

REESCRITO (2026-08-07): antes reimplementava gerar_sql_bird()/call_ollama() em paralelo a
nl_to_sql/nl_to_sql.py -- mesmo algoritmo (prompt LLM->SQL via Ollama), codigo duplicado.
Agora chama nl_to_sql.gerar_sql_com_schema() real.

Uso: python eval/avaliar_bird_sql.py [n_perguntas] [modelo_ollama]
     (default: 25 perguntas, modelo llama3.2)
     ex: python eval/avaliar_bird_sql.py 100 qwen2.5:7b
Saida: eval/resultados_bird_sql.csv (nome inclui o modelo se != llama3.2) + resumo no stdout.
"""
import csv
import re
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR.parent / "nl_to_sql"))
from nl_to_sql import gerar_sql_com_schema  # noqa: E402
import nl_to_sql  # noqa: E402 -- para sobrescrever OLLAMA_MODEL usado internamente por call_ollama()

N_PERGUNTAS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
OLLAMA_MODEL = sys.argv[2] if len(sys.argv) > 2 else "llama3.2"
nl_to_sql.OLLAMA_MODEL = OLLAMA_MODEL  # nl_to_sql.call_ollama le o modulo-level OLLAMA_MODEL
# Nome de arquivo distinto por modelo, para nao sobrescrever resultados de rodadas anteriores
# com outro modelo (mesmo padrao de rodar_golden.py / rodar_golden_qwen.py).
_sufixo_modelo = "" if OLLAMA_MODEL == "llama3.2" else f"_{OLLAMA_MODEL.replace(':', '').replace('.', '')}"
RESULTADOS = EVAL_DIR / f"resultados_bird_sql{_sufixo_modelo}.csv"


def gerar_sql_bird(pergunta, evidence, schema):
    """Wrapper fino sobre nl_to_sql.gerar_sql_com_schema(): o BIRD tem um campo 'evidence'
    (dica extra do dataset) que nao existe no formato de pergunta do Harbor -- concatenado
    a pergunta antes de chamar a funcao real, para nao precisar de um parametro novo em
    gerar_sql_com_schema() so para este benchmark."""
    pergunta_com_evidence = f"{pergunta}\n\nDica adicional: {evidence}" if evidence else pergunta
    return gerar_sql_com_schema(pergunta_com_evidence, esquema=schema, usar_few_shot=False)


def normalizar_sql(sql):
    """Normalizacao leve para correspondencia estrutural: minusculo, espacos colapsados,
    remove ponto-e-virgula final, remove aspas de identificador (` e "). Nao e um parser SQL
    de verdade -- e uma aproximacao textual, suficiente para medir similaridade grosseira
    sem executar a query."""
    s = sql.lower().strip().rstrip(";")
    s = re.sub(r"[`\"]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def extrair_tabelas_colunas(sql):
    """Extrai o conjunto de tokens 'palavra' do SQL normalizado (proxy para tabelas/colunas
    citadas), para medir sobreposicao com o gold mesmo quando a ordem/sintaxe difere."""
    s = normalizar_sql(sql)
    tokens = re.findall(r"[a-z_][a-z0-9_]*", s)
    stopwords = {
        "select", "from", "where", "and", "or", "as", "on", "join", "inner", "left", "right",
        "outer", "group", "by", "order", "having", "limit", "distinct", "count", "sum", "avg",
        "max", "min", "case", "when", "then", "else", "end", "not", "in", "is", "null", "like",
        "between", "asc", "desc", "cast", "double", "integer", "text", "real",
    }
    return {t for t in tokens if t not in stopwords}


def comparar(sql_gerado, sql_gold):
    """Duas metricas de correspondencia estrutural (nao execucao real):
    1. exato_normalizado: SQL gerado == gold apos normalizacao textual leve (raro bater).
    2. jaccard_tokens: sobreposicao de tabelas/colunas/valores citados (mais informativo --
       mede se o modelo 'entendeu' quais dados usar, mesmo com sintaxe diferente)."""
    norm_gerado = normalizar_sql(sql_gerado)
    norm_gold = normalizar_sql(sql_gold)
    exato = norm_gerado == norm_gold

    tok_gerado = extrair_tabelas_colunas(sql_gerado)
    tok_gold = extrair_tabelas_colunas(sql_gold)
    if not tok_gold:
        jaccard = None
    else:
        inter = len(tok_gerado & tok_gold)
        uniao = len(tok_gerado | tok_gold)
        jaccard = inter / uniao if uniao else 0.0

    return exato, jaccard


def main():
    from datasets import load_dataset
    print(f"Carregando BIRD-SQL (xu3kev/BIRD-SQL-data-train), {N_PERGUNTAS} perguntas...")
    ds = load_dataset("xu3kev/BIRD-SQL-data-train", split=f"train[:{N_PERGUNTAS}]")

    resultados = []
    print(f"\nRodando {len(ds)} perguntas do BIRD contra o NL-to-SQL do Harbor ({OLLAMA_MODEL})...\n")
    for i, row in enumerate(ds):
        pergunta = row["question"]
        evidence = row.get("evidence", "") or ""
        schema = row["schema"]
        sql_gold = row["SQL"]
        db_id = row["db_id"]

        try:
            sql_gerado = gerar_sql_bird(pergunta, evidence, schema)
            erro = None
        except Exception as exc:
            sql_gerado = ""
            erro = str(exc)

        exato, jaccard = comparar(sql_gerado, sql_gold) if not erro else (False, 0.0)

        resultados.append({
            "idx": i, "db_id": db_id, "pergunta": pergunta[:120],
            "sql_gerado": sql_gerado[:200], "sql_gold": sql_gold[:200],
            "exato_normalizado": exato, "jaccard_tokens": round(jaccard, 3) if jaccard is not None else "",
            "erro": erro or "",
        })
        status = "EXATO" if exato else f"jaccard={jaccard:.2f}" if jaccard is not None else "ERRO"
        print(f"[{i+1:3}/{len(ds)}] db={db_id:20} {status}")

    with RESULTADOS.open("w", newline="", encoding="utf-8") as f:
        campos = ["idx", "db_id", "pergunta", "sql_gerado", "sql_gold",
                  "exato_normalizado", "jaccard_tokens", "erro"]
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(resultados)

    n = len(resultados)
    n_exato = sum(1 for r in resultados if r["exato_normalizado"])
    jaccards = [r["jaccard_tokens"] for r in resultados if r["jaccard_tokens"] != ""]
    jaccard_medio = sum(jaccards) / len(jaccards) if jaccards else 0.0
    n_erro = sum(1 for r in resultados if r["erro"])

    print(f"\n=== RESUMO ({n} perguntas do BIRD-SQL, modelo {OLLAMA_MODEL}) ===")
    print(f"Correspondencia EXATA (normalizada, texto): {n_exato}/{n} ({n_exato/n*100:.1f}%)")
    print(f"Jaccard medio de tabelas/colunas citadas:    {jaccard_medio*100:.1f}%")
    print(f"Erros de geracao:                            {n_erro}/{n}")
    print(f"\nNOTA: esta e uma metrica de correspondencia estrutural, NAO execution accuracy")
    print(f"(o padrao oficial do BIRD). Nao comparar diretamente com o numero de estado da arte")
    print(f"(~82%) sem essa ressalva -- ver docstring do arquivo.")
    print(f"\nResultados salvos em {RESULTADOS}")


if __name__ == "__main__":
    main()
