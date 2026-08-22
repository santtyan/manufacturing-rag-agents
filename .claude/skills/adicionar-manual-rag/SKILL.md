---
name: adicionar-manual-rag
description: Adiciona um manual técnico novo ao RAG do Harbor (rag/manuais/), reindexa o ChromaDB e valida com Recall@k/MRR no golden set de retrieval. Use quando o usuário pedir para "adicionar um manual", "incluir esse documento no RAG", "o chat não sabe responder sobre X" (quando X é conteúdo que deveria vir de um manual), ou depois de editar um manual .md existente.
---

# Adicionar manual novo ao RAG e validar retrieval

## Passo 1 — Escrever o manual no formato certo

Salve o arquivo `.md` em `rag/manuais/`. O chunking (`rag/rag_hibrido.py::chunk_texto()`) corta por seção Markdown `## N. Titulo` — **o documento precisa seguir essa convenção de cabeçalho**, senão cai inteiro no fallback por linhas e perde granularidade (mesmo problema que motivou o chunking atual, documentado no código). Cada `##` deve estar sozinho no início da linha.

Boas práticas de conteúdo (para o retrieval funcionar bem):
- Uma ideia/regra por seção — o Cross-Encoder de rerank precisa de granularidade para discriminar.
- Se uma seção tiver uma tabela (schema, thresholds), mantenha a tabela inteira dentro de uma seção — não deixe cabeçalho e tabela em blocos separados.
- Evite seções gigantes (>900 chars); se inevitável, o fallback por linha ainda preserva linhas inteiras, mas prefira quebrar em subsecões.

## Passo 2 — Reindexar

O `RAGHibrido` só reindexa automaticamente se a coleção (`COLECAO` em `rag/rag_hibrido.py`, hoje `"manuais_harbor_v2"`) não existir ainda em `rag/chroma_db/`. Como ela já existe, force a reindexação:

```powershell
cd rag
python -c "from rag_hibrido import RAGHibrido; r = RAGHibrido(); print(r.indexar(forcar=True), 'chunks indexados')"
```

Ou simplesmente rode o módulo direto (`python rag_hibrido.py`), que já chama `indexar(forcar=True)` no `__main__` e imprime alguns testes de busca com perguntas parafraseadas.

**Se você mudou o algoritmo de chunking** (não só adicionou um manual), renomeie `COLECAO` para uma versão nova (ex.: `manuais_harbor_v3`) antes de reindexar — é o padrão já usado no projeto (`v1`→`v2`) para não deixar chunks antigos e novos misturados no mesmo índice.

## Passo 3 — Adicionar perguntas ao golden set de retrieval

Edite `eval/golden_questions.json`: adicione pelo menos 1-2 perguntas novas com rota `"rag"` e o campo `"fonte"` apontando para o nome do arquivo `.md` novo — é esse campo que `eval/avaliar_retrieval.py` usa como alvo de relevância (documento certo, não chunk específico).

## Passo 4 — Validar retrieval isolado

```powershell
python eval/avaliar_retrieval.py
```

Mede Recall@5/Precision@5/MRR do retrieval isolado (sem rerank) contra o golden set. Rode com `--rerank` para medir o pipeline completo (retrieval + Cross-Encoder):

```powershell
python eval/avaliar_retrieval.py --rerank
```

Confira que as perguntas novas (sobre o manual adicionado) recuperam o documento certo. Se Recall@5 vier baixo especificamente para elas, revise a granularidade das seções do manual (passo 1) antes de mexer em parâmetros do RAG.

## Não confundir com

`eval/avaliar_retrieval_nanobeir.py` mede o mesmo motor de retrieval contra um benchmark acadêmico público (NanoBEIR/SciFact) — serve para validar a qualidade geral do motor, não a cobertura de um manual específico do domínio Harbor. Use `avaliar_retrieval.py` (golden set próprio) para este fluxo.

## Não fazer

- Não misture o índice de produção (`rag/chroma_db/`) com o índice de benchmark (`rag/chroma_db_benchmark/`) — são intencionalmente separados.
- Não edite chunks diretamente no ChromaDB — a fonte de verdade é sempre o `.md` em `rag/manuais/`, reindexado do zero.
