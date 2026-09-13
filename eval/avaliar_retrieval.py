"""
Avaliacao ISOLADA de retrieval (Recall@k, Precision@k, MRR) -- distinto de avaliar a
resposta final (rodar_golden.py). Mede se o RAG hibrido (rag/rag_hibrido.py, renomeado de
rag_neural.py em 2026-07-15) recupera o chunk certo, independente do que o LLM faz depois
com ele.

Motivacao: eval/resultados_consistencia.csv mostrou uma pergunta variando de 0% a 100%
de faithfulness entre execucoes identicas. Sem medir retrieval isoladamente e impossivel
saber se a causa e retrieval inconsistente (chunk certo as vezes nao entra no top-k) ou
so a geracao do LLM (chunk entra sempre, LLM que erra ao usar). Como RAGHibrido.buscar()
nao tem nenhuma fonte de aleatoriedade (embeddings e cross-encoder sao deterministicos),
a hipotese e que o retrieval e estavel e a variancia vem da geracao -- este script
confirma ou refuta essa hipotese com dados.

Golden set: (pergunta, arquivo-fonte esperado) derivado das golden questions de rota
"rag" em golden_questions.json (campo "fonte"). Como o corpus tem so 5 documentos e a
maioria das perguntas usa so 1 manual, o "documento certo" (nao chunk_id especifico) e
o alvo de relevancia -- suficiente para medir Recall@k/Precision@k/MRR na escala do
corpus atual sem exigir anotacao manual de chunk-a-chunk.

ACHADO REAL (2026-09-14, investigacao do resultado negativo do Agentic RAG): o recall acima
mede so acerto de DOCUMENTO ("achou o arquivo certo"), nao de CHUNK ("achou a SECAO certa
dentro do arquivo"). O Agentic RAG (rag/rag_agentic.py) piorou porque a causa raiz diagnosticada
foi "documento certo, mas a resposta esta em outra secao do mesmo arquivo" -- um modo de falha
INVISIVEL na metrica de documento. Pesquisa de estado da arte (Seven Failure Points When
Engineering a RAG System, Barnett et al., arXiv:2401.05856) formaliza esse caso como FP2
(Missed the Top Ranked Documents), com a nuance de existir so no nivel de chunk.

Para medir isso SEM reanotar o golden set: 41 das 49 perguntas de rota "rag" ja tem o numero da
secao esperada escrito no proprio campo "nota" (ex. "Manual (secao 5): ..."), porque quem
escreveu as perguntas ja verificava contra o manual ao criar cada uma. `secao_esperada()` extrai
esse numero por regex; `secao_do_chunk()` extrai o mesmo numero do cabeculho "## N. Titulo" que
cada chunk retornado ja carrega (rag_hibrido.py::chunk_texto() preserva o cabecalho no chunk).
Perguntas sem numero de secao na nota (armadilhas que descrevem a secao ERRADA esperada, ou
perguntas sem nota estruturada) ficam de fora do chunk-level -- continuam contribuindo para o
doc-level normalmente.

Uso: python eval/avaliar_retrieval.py [--rerank]
     (default: usar_rerank=False, mede o RETRIEVAL isolado. Com --rerank, mede o pipeline
     completo retrieval+rerank de ponta a ponta -- ver avaliar_pergunta() para o porque de
     medir os dois separadamente.)
"""
import csv
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Projetos\Harbor\rag")
sys.path.insert(0, str(Path(__file__).parent))

from rag_hibrido import RAGHibrido

EVAL_DIR = Path(__file__).parent
GOLDEN = EVAL_DIR / "golden_questions.json"

K = 5  # top-k avaliado (k_candidatos default do RAGHibrido.buscar ja e 10 nos bastidores)

REGEX_SECAO_NOTA = re.compile(r"secao\s+(\d+)", re.IGNORECASE)
REGEX_SECAO_CHUNK = re.compile(r"^##\s+(\d+)\.", re.MULTILINE)

# ACHADO REAL (2026-09-14): 5 perguntas citam 2+ secoes na mesma nota -- a PRIMEIRA ocorrencia
# de "secao N" nem sempre e a resposta certa. Em notas que comecam com "ARMADILHA: <descricao
# livre> (secao X), mas <resposta certa> (secao Y)", a secao certa e a SEGUNDA citada (o texto
# livre descreve a armadilha primeiro, so depois a resposta). Em notas que comecam com "Manual
# (secao N):" ou "ARMADILHA (secao N):", a secao certa e a PRIMEIRA (o numero abre a nota,
# descrevendo A PROPRIA secao correta; qualquer secao citada depois e so contexto/contraste).
# Verificado manualmente contra o conteudo real de cada secao do manual e o campo
# numeros_esperados de cada pergunta -- ver tests/test_avaliar_retrieval.py para os 5 casos.
SECOES_ESPERADAS_AMBIGUAS = {
    "rag-criticidade-inventada": 8,               # "ARMADILHA: ... (secao 2), mas ... (secao 8)"
    "rag-sensores-criticidade-temperatura": 8,     # "ARMADILHA (secao 8): ..." -- 1a citacao
    "rag-legacy-error-message-critico": 6,         # "Manual (secao 6): ..." -- 1a citacao
    "rag-legacy-peso-camada3": 5,                  # "Manual (secao 5): ..." -- 1a citacao
    "rag-legacy-alerta-sem-mensagem": 6,           # "Manual (secao 6): ..." -- 1a citacao
}


def arquivo_esperado(pergunta_obj):
    """Extrai o nome do arquivo .md esperado do campo 'fonte' (ex:
    'rag/manuais/manual_manutencao_sensores_industriais.md' -> 'manual_...md')."""
    fonte = pergunta_obj.get("fonte", "")
    if not fonte.endswith(".md"):
        return None
    return fonte.rsplit("/", 1)[-1]


def secao_esperada(pergunta_obj):
    """Extrai o numero da secao esperada do campo 'nota', se presente (ex. "Manual (secao 5):
    ..." -> 5). Retorna None se a nota nao cita secao explicitamente -- essas perguntas nao
    entram na metrica de chunk-level (ver docstring do modulo).

    Perguntas com 2+ secoes citadas na mesma nota (armadilhas que descrevem a secao ERRADA
    antes da certa) usam SECOES_ESPERADAS_AMBIGUAS, verificado manualmente -- a extracao pela
    PRIMEIRA ocorrencia de 'secao N' erraria nesses casos (ver comentario da constante)."""
    pid = pergunta_obj.get("id")
    if pid in SECOES_ESPERADAS_AMBIGUAS:
        return SECOES_ESPERADAS_AMBIGUAS[pid]
    nota = pergunta_obj.get("nota", "")
    m = REGEX_SECAO_NOTA.search(nota)
    return int(m.group(1)) if m else None


def secao_do_chunk(texto_chunk):
    """Extrai o numero da secao do cabecalho '## N. Titulo' que cada chunk carrega (preservado
    por rag_hibrido.py::chunk_texto()). Retorna None se o chunk nao comecar com esse padrao
    (ex. veio do fallback _chunk_por_linhas sem cabecalho proprio)."""
    m = REGEX_SECAO_CHUNK.search(texto_chunk)
    return int(m.group(1)) if m else None


def avaliar_pergunta(rag, pergunta_obj, k=K, usar_rerank=False):
    """Roda buscar() e calcula recall@k/precision@k/rr (reciprocal rank) contra o arquivo
    esperado (doc-level) E contra a secao esperada quando disponivel (chunk-level).

    usar_rerank=False (default): mede a qualidade do RETRIEVAL (hybrid E5+TF-IDF)
    isoladamente. O rerank e uma etapa de POS-processamento que roda por cima do retrieval;
    avaliar as duas juntas misturaria as duas fontes de erro. usar_rerank=True mede o
    pipeline completo (retrieval+rerank) de ponta a ponta -- a metrica que falta comparar
    contra o NanoBEIR, que ja e medido com rerank (ver eval/avaliar_retrieval_nanobeir.py)."""
    alvo = arquivo_esperado(pergunta_obj)
    if alvo is None:
        return None

    candidatos = rag.buscar(pergunta_obj["pergunta"], k=k, usar_rerank=usar_rerank, usar_hybrid=True, k_candidatos=max(k, 10))
    fontes = [c["fonte"] for c in candidatos]

    relevantes_no_topk = sum(1 for f in fontes if f == alvo)
    recall_at_k = 1.0 if relevantes_no_topk > 0 else 0.0  # 1 doc relevante por pergunta -> recall e binario
    precision_at_k = relevantes_no_topk / len(fontes) if fontes else 0.0

    rr = 0.0
    for i, f in enumerate(fontes, start=1):
        if f == alvo:
            rr = 1.0 / i
            break

    # Chunk-level: so calculado quando a nota cita a secao esperada explicitamente. Um chunk
    # so conta como acerto de CHUNK se, alem de vir do arquivo certo, tambem carregar a secao
    # certa -- e possivel um candidato ter fonte==alvo mas secao != secao_esperada (mesmo modo
    # de falha diagnosticado no Agentic RAG).
    sec_esperada = secao_esperada(pergunta_obj)
    recall_chunk_at_k = None
    rr_chunk = None
    if sec_esperada is not None:
        secoes_dos_candidatos = [
            secao_do_chunk(c["texto"]) if fonte == alvo else None
            for c, fonte in zip(candidatos, fontes)
        ]
        relevantes_chunk_no_topk = sum(1 for s in secoes_dos_candidatos if s == sec_esperada)
        recall_chunk_at_k = 1.0 if relevantes_chunk_no_topk > 0 else 0.0
        rr_chunk = 0.0
        for i, s in enumerate(secoes_dos_candidatos, start=1):
            if s == sec_esperada:
                rr_chunk = 1.0 / i
                break

    return {
        "id": pergunta_obj["id"],
        "arquivo_esperado": alvo,
        "fontes_recuperadas": fontes,
        "recall_at_k": recall_at_k,
        "precision_at_k": round(precision_at_k, 3),
        "reciprocal_rank": round(rr, 3),
        "secao_esperada": sec_esperada,
        "recall_chunk_at_k": recall_chunk_at_k,
        "reciprocal_rank_chunk": round(rr_chunk, 3) if rr_chunk is not None else None,
    }


def main():
    usar_rerank = "--rerank" in sys.argv
    sufixo = "_rerank" if usar_rerank else ""
    resultados_path = EVAL_DIR / f"resultados_retrieval{sufixo}.csv"

    dados = json.loads(GOLDEN.read_text(encoding="utf-8"))
    perguntas_rag = [p for p in dados["perguntas"] if p.get("rota_esperada") == "rag"]

    if not perguntas_rag:
        print("Nenhuma golden question de rota 'rag' encontrada.")
        return

    modo = "retrieval + rerank (pipeline completo)" if usar_rerank else "retrieval isolado (sem rerank)"
    print(f"Indexando RAG hibrido (forcar=True, para pegar manuais novos) e avaliando {modo} "
          f"em {len(perguntas_rag)} perguntas (k={K})...\n")
    rag = RAGHibrido()
    rag.indexar(forcar=True)  # forcar=True: sem isso, reaproveita indice antigo em disco e
    # ignora .md adicionados depois da ultima indexacao (achado real, 2026-07-16 -- as 12
    # perguntas dos 4 manuais novos deram 100% MISS ate essa correcao).

    resultados = []
    for pq in perguntas_rag:
        r = avaliar_pergunta(rag, pq, k=K, usar_rerank=usar_rerank)
        if r is None:
            print(f"[{pq['id']:26}] sem arquivo-fonte .md no golden set -- pulando")
            continue
        resultados.append(r)
        status = "OK " if r["recall_at_k"] == 1.0 else "MISS"
        chunk_status = ""
        if r["recall_chunk_at_k"] is not None:
            chunk_status = " chunk=OK" if r["recall_chunk_at_k"] == 1.0 else " chunk=MISS"
        print(f"[{r['id']:26}] {status} esperado={r['arquivo_esperado']:45} "
              f"precision@{K}={r['precision_at_k']:.2f} RR={r['reciprocal_rank']:.2f}{chunk_status}")

    if not resultados:
        print("\nNenhum resultado avaliavel.")
        return

    with resultados_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "arquivo_esperado", "recall_at_k", "precision_at_k", "reciprocal_rank",
                     "secao_esperada", "recall_chunk_at_k", "reciprocal_rank_chunk", "fontes_recuperadas"])
        for r in resultados:
            w.writerow([r["id"], r["arquivo_esperado"], r["recall_at_k"], r["precision_at_k"],
                        r["reciprocal_rank"], r["secao_esperada"], r["recall_chunk_at_k"],
                        r["reciprocal_rank_chunk"], ";".join(r["fontes_recuperadas"])])

    recall_medio = statistics.mean(r["recall_at_k"] for r in resultados)
    precision_media = statistics.mean(r["precision_at_k"] for r in resultados)
    mrr = statistics.mean(r["reciprocal_rank"] for r in resultados)

    # Chunk-level: so sobre o subconjunto com secao_esperada conhecida (ver docstring do
    # modulo -- perguntas sem numero de secao explicito na nota nao entram aqui).
    com_chunk = [r for r in resultados if r["recall_chunk_at_k"] is not None]
    recall_chunk_medio = statistics.mean(r["recall_chunk_at_k"] for r in com_chunk) if com_chunk else None
    mrr_chunk = statistics.mean(r["reciprocal_rank_chunk"] for r in com_chunk) if com_chunk else None

    print("\n" + "=" * 60)
    print(f"Modo               : {modo}")
    print(f"Perguntas avaliadas: {len(resultados)}")
    print(f"Recall@{K} medio (DOCUMENTO) : {recall_medio*100:.1f}%")
    print(f"Precision@{K} media          : {precision_media*100:.0f}%")
    print(f"MRR (documento)              : {mrr:.3f}")
    if com_chunk:
        print(f"\nChunk-level (subconjunto com secao esperada conhecida na nota, {len(com_chunk)}/{len(resultados)} perguntas):")
        print(f"Recall@{K} medio (CHUNK)     : {recall_chunk_medio*100:.1f}%")
        print(f"MRR (chunk)                  : {mrr_chunk:.3f}")
        gap = recall_medio - recall_chunk_medio
        print(f"\nGap doc-level - chunk-level  : {gap*100:+.1f} pontos percentuais")
        if gap > 0.01:
            print("Gap positivo confirma o diagnostico: ha perguntas onde o ARQUIVO certo e "
                  "recuperado mas a SECAO certa nao vem no top-k -- modo de falha invisivel na "
                  "metrica de documento (ver docstring do modulo, achado do Agentic RAG).")
        else:
            print("Gap ~zero: NAO ha evidencia de 'documento certo, secao errada' neste "
                  "subconjunto -- reconsiderar a causa raiz do resultado negativo do Agentic RAG.")
    print(f"\nResultados salvos  : {resultados_path}")


if __name__ == "__main__":
    main()
