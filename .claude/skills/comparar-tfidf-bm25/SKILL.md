---
name: comparar-tfidf-bm25
description: Roda o benchmark que compara os dois algoritmos de busca lexical disponíveis em rag/rag_hibrido.py (TF-IDF, produção atual, vs BM25, alternativa via rank_bm25) no golden set de retrieval. Use quando o usuário pedir para "comparar TF-IDF e BM25", "testar o BM25", "ver se vale trocar o índice lexical do RAG", ou depois de adicionar manuais novos ao RAG (para conferir se a comparação ainda vale com o corpus maior).
---

# Comparar TF-IDF vs BM25 no retrieval do Harbor

## Contexto

`rag/rag_hibrido.py::RAGHibrido` aceita `lexico="tfidf"` (default, produção) ou `lexico="bm25"` desde 2026-08. BM25 foi sugestão do Pedro (ver memória `feedback_pedro_bm25_limiar_alucinacao.md`) — tende a se sair melhor que TF-IDF+cosseno em corpora pequenos porque normaliza por tamanho de documento e satura a contribuição de termo repetido.

**Cuidado conhecido:** o score bruto do BM25 (`rank_bm25.BM25Okapi.get_scores`) não é limitado a [0,1] como TF-IDF/E5 — pode passar de 7 no corpus atual. `_buscar_bm25()` já normaliza isso via min-max por query antes de retornar; se você alterar essa função, reconfirme que a normalização continua lá, senão o modo `usar_hybrid=True, usar_rerank=False` degrada (achado real: Recall@5 caiu de 98%→86% sem a normalização).

## Rodar o benchmark

```powershell
python eval/comparar_tfidf_bm25.py
```

Mede, para os dois algoritmos, no mesmo golden set de `eval/golden_questions.json` (perguntas de rota `"rag"`):

1. **Lexical isolado** — só a metade BM25/TF-IDF, sem E5, isola o efeito puro do algoritmo.
2. **Hybrid sem rerank** — E5 + lexical, como `eval/avaliar_retrieval.py` sem `--rerank`.
3. **Hybrid com rerank** — pipeline completo, o modo real de produção (`buscar()` default).
4. **Latência média da busca lexical isolada** (ms/query).

Usa um índice ChromaDB isolado (`rag/chroma_db_benchmark_lexico/`), separado do índice de produção (`rag/chroma_db/`) e do benchmark NanoBEIR (`rag/chroma_db_benchmark/`) — nunca misturar os três.

Resultado salvo em `eval/resultados_comparacao_tfidf_bm25.json`.

## Como interpretar

- O cenário que importa para decidir se vale trocar o default de produção é **hybrid com rerank** — é o que `dashboard/app.py` e o MCP realmente usam. Diferenças nos outros dois cenários são informativas mas não decisivas sozinhas.
- Baseline conhecido (corpus de 9 manuais, 49 perguntas RAG, 2026-08-22): TF-IDF e BM25 empatam no hybrid com rerank (98%/44%/0.866 vs 98%/43%/0.866) — BM25 ganha levemente no lexical isolado (98% vs 94% recall) e tem ~15% menos latência, mas a diferença desaparece depois do Cross-Encoder. Não foi trocado o default por esse motivo. Se o corpus crescer bastante (mais manuais, corpus real da HarboR), vale rodar de novo — BM25 tende a se destacar mais em corpora maiores.
- Números completos e a decisão da última rodada estão documentados em `slides/proximo_slide.md` (mantenha esse arquivo atualizado se rodar de novo com resultado diferente).

## Trocar o default de produção (se o resultado justificar)

Só troque `COLECAO` (força reindexação) e o `lexico=` default em `RAGHibrido.__init__` depois de confirmar com o usuário — é uma mudança de comportamento de produção, não decida sozinho só com base num benchmark.
