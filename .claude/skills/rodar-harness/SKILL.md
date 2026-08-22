---
name: rodar-harness
description: Roda o harness de avaliação do Harbor (roteamento + faithfulness sobre golden_questions.json) e/ou os benchmarks acadêmicos (NanoBEIR, BIRD-SQL, Spider), e interpreta os resultados. Use quando o usuário pedir para "rodar o harness", "avaliar o roteamento/RAG/NL-to-SQL", "medir faithfulness", ou depois de mudar dashboard/roteador.py, rag/rag_hibrido.py, nl_to_sql/nl_to_sql.py ou eval/rag_gerador.py.
---

# Rodar o harness de avaliação do Harbor

## Pré-requisito

Os serviços precisam estar no ar (Ollama pelo menos; Postgres se for testar SQL). Se não tiver certeza, use a skill `subir-servicos` primeiro.

## Harness principal (roteamento + faithfulness)

```powershell
python eval/rodar_golden.py
```

Roda cada pergunta de `eval/golden_questions.json` através de `dashboard/roteador.py` (mesmo módulo usado em produção — não há cópia duplicada aqui) e `eval/rag_gerador.py`, e reporta:
- Acerto de roteamento (rota esperada vs. rota real)
- Faithfulness da resposta gerada (checagem de alucinação)

Leia o resumo final. Se um número cair muito em relação ao histórico conhecido, a causa mais provável é desatualização do próprio harness ou dos módulos importados — **não assuma regressão real sem comparar com uma run anterior**. Já aconteceu de um "52% vs 67,9%" ser puramente causado por uma cópia desatualizada de `roteador.py` (histórico documentado no topo do próprio arquivo).

## Consistência (mesma pergunta repetida)

```powershell
python eval/rodar_consistencia.py
```

Mede o quanto a resposta varia para a mesma pergunta em execuções repetidas — útil depois de mudar temperature/modelo do Ollama.

## Benchmarks acadêmicos

```powershell
# Retrieval (RAG) contra NanoBEIR
python eval/avaliar_retrieval_nanobeir.py

# NL-to-SQL contra BIRD-SQL
python eval/avaliar_bird_sql.py [N_PERGUNTAS] [OLLAMA_MODEL]

# NL-to-SQL contra Spider
python eval/avaliar_spider_sql.py [N_PERGUNTAS] [OLLAMA_MODEL]
```

`N_PERGUNTAS` default 25, `OLLAMA_MODEL` default `llama3.2`. Esses benchmarks chamam o código de produção real (`rag/rag_hibrido.py`, `nl_to_sql/nl_to_sql.py`), não uma reimplementação — resultado é comparável ao comportamento real do dashboard.

## Depois de rodar

- Resuma: roteamento (N/total), faithfulness (%), e qualquer pergunta que mudou de resultado em relação à última run conhecida.
- Se encontrar uma alucinação nova (resposta errada não coberta por gate existente), isso é candidato a virar um gate novo — ver skill `adicionar-gate-roteamento`.
- Não publique números novos em slides/documentação sem confirmar que vieram de uma run com os módulos sincronizados (roteador.py e rag_gerador.py atualizados, não cópias antigas).
