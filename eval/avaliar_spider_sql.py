"""
Avaliacao do NL-to-SQL do Harbor contra um subset do Spider (Yu et al. 2018, EMNLP) --
o benchmark academico "irmao" do BIRD para text-to-SQL, referencia canonica desde 2018,
ainda citado como padrao junto do BIRD/BIRD-2.0 nos artigos de estado da arte (ver
pesquisa 2026-07-12 sobre benchmarks padrao-ouro e "eval/avaliar_bird_sql.py").

Reusa a logica de comparacao/normalizacao de eval/avaliar_bird_sql.py (nao duplica) --
so muda a fonte de dados (2 datasets do HuggingFace: perguntas+SQL gold de um lado,
schema de outro, unidos por db_id) e o formato do schema (Spider usa uma lista de
colunas "tabela : coluna (tipo)", nao DDL CREATE TABLE como o BIRD).

MESMA LIMITACAO METODOLOGICA do BIRD: sem os bancos SQLite reais (Spider tambem exige
banco populado para execution accuracy oficial), a metrica aqui e correspondencia
estrutural de texto, NAO execution accuracy -- nao comparavel diretamente ao estado da
arte publicado (~85% no Spider dev set com sistemas frontier).

REESCRITO (2026-08-07): antes reimplementava gerar_sql_spider() em paralelo a
nl_to_sql/nl_to_sql.py (mesmo problema do BIRD antes da reescrita -- ver
eval/avaliar_bird_sql.py). Agora chama nl_to_sql.gerar_sql_com_schema() real; mesma
ressalva do BIRD sobre nao usar self-repair/DBA-Agent (sem banco real para executar).

Uso: python eval/avaliar_spider_sql.py [n_perguntas] [modelo_ollama]
     (default: 25 perguntas, modelo llama3.2)
Saida: eval/resultados_spider_sql[_modelo].csv + resumo no stdout.
"""
import csv
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).parent
# Reusa normalizar_sql/extrair_tabelas_colunas/comparar de avaliar_bird_sql.py -- uma unica
# implementacao dessas funcoes, compartilhada entre BIRD e Spider (legitimo: sao especificas
# de comparacao estrutural de benchmark, nao logica de producao).
sys.path.insert(0, str(EVAL_DIR))
import avaliar_bird_sql as bird_utils

sys.path.insert(0, str(EVAL_DIR.parent / "nl_to_sql"))
from nl_to_sql import gerar_sql_com_schema  # noqa: E402
import nl_to_sql  # noqa: E402

N_PERGUNTAS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
OLLAMA_MODEL = sys.argv[2] if len(sys.argv) > 2 else "llama3.2"
nl_to_sql.OLLAMA_MODEL = OLLAMA_MODEL  # nl_to_sql.call_ollama le o modulo-level OLLAMA_MODEL
_sufixo_modelo = "" if OLLAMA_MODEL == "llama3.2" else f"_{OLLAMA_MODEL.replace(':', '')}"
RESULTADOS = EVAL_DIR / f"resultados_spider_sql{_sufixo_modelo}.csv"


def gerar_sql_spider(pergunta, schema_txt):
    """Wrapper fino sobre nl_to_sql.gerar_sql_com_schema() -- Spider nao tem campo
    'evidence' (diferente do BIRD), entao chama direto sem concatenar dica extra."""
    return gerar_sql_com_schema(pergunta, esquema=schema_txt, usar_few_shot=False)


def main():
    from datasets import load_dataset
    print(f"Carregando Spider (xlangai/spider) + schemas (richardr1126/spider-schema)...")
    perguntas_ds = load_dataset("xlangai/spider", split=f"train[:{N_PERGUNTAS * 3}]")  # sobra p/ filtrar db_id sem schema
    schemas_ds = load_dataset("richardr1126/spider-schema", split="train")
    schema_por_db = {row["db_id"]: row["Schema (values (type))"] for row in schemas_ds}

    linhas = []
    for row in perguntas_ds:
        if row["db_id"] in schema_por_db:
            linhas.append(row)
        if len(linhas) >= N_PERGUNTAS:
            break
    print(f"{len(linhas)} perguntas com schema disponivel (de {len(perguntas_ds)} candidatas).")

    resultados = []
    print(f"\nRodando {len(linhas)} perguntas do Spider contra o NL-to-SQL do Harbor ({OLLAMA_MODEL})...\n")
    for i, row in enumerate(linhas):
        pergunta = row["question"]
        sql_gold = row["query"]
        db_id = row["db_id"]
        schema_txt = schema_por_db[db_id]

        try:
            sql_gerado = gerar_sql_spider(pergunta, schema_txt)
            erro = None
        except Exception as exc:
            sql_gerado = ""
            erro = str(exc)

        exato, jaccard = bird_utils.comparar(sql_gerado, sql_gold) if not erro else (False, 0.0)

        resultados.append({
            "idx": i, "db_id": db_id, "pergunta": pergunta[:120],
            "sql_gerado": sql_gerado[:200], "sql_gold": sql_gold[:200],
            "exato_normalizado": exato, "jaccard_tokens": round(jaccard, 3) if jaccard is not None else "",
            "erro": erro or "",
        })
        status = "EXATO" if exato else f"jaccard={jaccard:.2f}" if jaccard is not None else "ERRO"
        print(f"[{i+1:3}/{len(linhas)}] db={db_id:25} {status}")

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

    print(f"\n=== RESUMO ({n} perguntas do Spider, modelo {OLLAMA_MODEL}) ===")
    print(f"Correspondencia EXATA (normalizada, texto): {n_exato}/{n} ({n_exato/n*100:.1f}%)")
    print(f"Jaccard medio de tabelas/colunas citadas:    {jaccard_medio*100:.1f}%")
    print(f"Erros de geracao:                            {n_erro}/{n}")
    print(f"\nNOTA: correspondencia estrutural, NAO execution accuracy oficial (~85% estado da arte).")
    print(f"Resultados salvos em {RESULTADOS}")


if __name__ == "__main__":
    main()
