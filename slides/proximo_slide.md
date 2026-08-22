# Resposta preparada — "Vocês testaram outras opções de busca lexical no RAG?"

**Resposta curta (30s):**
"Sim. O RAG híbrido usa TF-IDF como metade lexical desde o início, mas implementamos e testamos
BM25 (Okapi, biblioteca rank_bm25) como alternativa — foi uma sugestão de um colega do CERISE
(Pedro). No pipeline completo com rerank, que é o que roda em produção, os dois empatam:
Recall@5 de 98% para ambos, MRR praticamente igual (0.866). BM25 sozinho (sem o rerank) é
levemente melhor — 98% contra 94% de recall — e ~15% mais rápido, mas essa vantagem desaparece
depois que o Cross-Encoder reordena os resultados. Por isso mantivemos TF-IDF como default:
não há ganho real que justifique reindexar o corpus de produção agora."

**Por que isso é bom, não ruim, de mostrar:**
- Mostra que a escolha de algoritmo não foi "porque veio por padrão" — foi medida e comparada
  empiricamente antes de decidir manter.
- No processo de implementar o BM25, encontramos e corrigimos um bug real: o score bruto do
  BM25 não é normalizado como o do TF-IDF/E5 (podia passar de 7, contra uma escala 0-1 dos
  outros dois). Sem corrigir isso, o modo híbrido sem rerank caía de 98% para 86% de Recall@5
  — é exatamente o tipo de erro sutil de integração que o harness existe para pegar antes de
  virar produção.

**Se perguntarem "e se o corpus de manuais crescer bastante?":**
"A expectativa é que BM25 se destaque mais conforme o corpus cresce — a vantagem dele
(normalização por tamanho de documento, saturação de termo repetido) importa mais quando há
mais documentos competindo pelo mesmo termo. O benchmark fica pronto para rodar de novo
(eval/comparar_tfidf_bm25.py) a qualquer momento, inclusive quando entrarem os manuais reais
da HarboR."

**Números para ter na ponta da língua (corpus de 9 manuais, 49 perguntas RAG, 2026-08-22):**

| Cenário | TF-IDF | BM25 |
|---|---|---|
| Lexical isolado (Recall/Precision@5/MRR) | 94% / 46% / 0.854 | 98% / 45% / 0.872 |
| Hybrid sem rerank | 98% / 38% / 0.898 | 98% / 39% / 0.878 |
| **Hybrid com rerank (produção)** | **98% / 44% / 0.866** | **98% / 43% / 0.866** |
| Latência média da busca lexical | 0.92ms | 0.78ms |

- Decisão: manter TF-IDF como default de produção — diferença no cenário real (hybrid+rerank)
  é desprezível.
- BM25 fica disponível como opção (`RAGHibrido(lexico="bm25")`), documentada e testável a
  qualquer momento via `python eval/comparar_tfidf_bm25.py` — não foi descartado, só não
  justificou trocar agora.
