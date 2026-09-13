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

## OpenPack (HAR via RAG-HAR training-free, integrado em 2026-09-13/14)

```powershell
# Retrieval por CATEGORIA sobre o corpus IMU (nao Recall@k -- ver nota abaixo)
python eval/avaliar_retrieval_openpack.py

# Classificacao F1-macro vs. benchmark oficial openpack-torch (split "Pilot Challenge")
python eval/avaliar_classificacao_openpack.py [--k 5]
```

**Achado real (2026-09-13), aplica-se a qualquer corpus de instância quase-única (sensor/série
temporal, não documento único)**: Recall@k/MRR clássico pressupõe retrieval *instance-level*
(existe 1 documento "certo" e o resto é irrelevante — ex. NanoBEIR, manuais RAG). Séries de
sensor segmentadas em janelas são retrieval *class-level*: nenhuma janela específica é "a
resposta certa", o que importa é se os vizinhos recuperados são da mesma classe/operação. Usar
Recall@k aqui mede a métrica errada (deu 0% mesmo com retrieval funcionando) — a métrica correta
é **k-NN label purity** (fração dos k vizinhos que compartilham o rótulo da janela de origem;
acaso esperado = 1/n_classes). Resultado medido: pureza=12,9% (acaso=10%, corpus de 4.000 docs).

`avaliar_classificacao_openpack.py` aplica o protocolo de classificação real do RAG-HAR
(vizinhos → votação majoritária, sem treino) sobre o split oficial "Pilot Challenge" do
`openpack-torch` — é o número comparável ao benchmark do dataset, não um proxy. F1-macro
validado em dois protocolos: 0,9217 (amostra estratificada, 26 janelas de teste) e 0,9114
(teste completo sem amostragem, 2.591 janelas) — ambos superam os 3 baselines supervisionados
(UNet=0,3451, ST-GCN=0,7024, DeepConvLSTM=0,7081). Trace da run instrumentado em
`eval/traces/openpack_classificacao.jsonl` via `shared.trace`.

**Aba de chat no Streamlit** (Fase 7f, 2026-09-14): `dashboard/app.py` tem a aba "5. OpenPack
(Operações de Embalagem)", com sub-roteador hierárquico corpus-aware dentro da rota `rag`
(`rag_openpack_responder_ou_manual`, decide entre o corpus de manuais e o corpus de janelas
IMU usando `PALAVRAS_CHAVE_OPENPACK` do roteador). Rodar `python eval/rodar_golden.py` depois
de qualquer mudança nessa aba ou em `eval/rag_gerador.py::carregar_rag_hibrido()` — as duas
perguntas `openpack-*` do golden set exercitam especificamente os 2 gates dessa aba.

## Depois de rodar

- Resuma: roteamento (N/total), faithfulness (%), e qualquer pergunta que mudou de resultado em relação à última run conhecida.
- Se encontrar uma alucinação nova (resposta errada não coberta por gate existente), isso é candidato a virar um gate novo — ver skill `adicionar-gate-roteamento`.
- Não publique números novos em slides/documentação sem confirmar que vieram de uma run com os módulos sincronizados (roteador.py e rag_gerador.py atualizados, não cópias antigas).
