# Rollout Card - hotpotqa

Gerado a partir de `C:\Projetos\Harbor\eval\traces\hotpotqa.jsonl` (2 execucoes).

## Modelo e versoes

- Modelo(s): qwen2.5:7b (2x)
- Dependencias:
  - langgraph: 1.x
  - datasets: HF hotpotqa/hotpot_qa distractor
- Tools disponiveis ao agente: search, lookup

## Prompt (resumo)

Responder pergunta multi-hop usando search/lookup sobre a Wikipedia (replicacao ReAct, Yao et al. 2023).

## Resultado

- Execucoes: 2
- Taxa de sucesso: 50.0% (1/2)
- Passos por execucao (media): 8.5
- Duracao por execucao (media): 238.71s

## Custo (tokens, somado sobre todas as execucoes)

- Input: 7565
- Output: 481
- Total: 8046

## O que quebrou (execucoes sem sucesso)

- outro: 1

## Reprodutibilidade

- Trace bruto (JSONL) disponivel em `C:\Projetos\Harbor\eval\traces\hotpotqa.jsonl` - cada linha e uma execucao completa, com thought/action/observation por passo, custo e modelo usados naquele passo especifico.
- Regra do projeto: nenhuma conclusao sobre este resultado e valida sem o baseline comparavel a custo de tokens normalizado (ver skill `migrar-para-langchain`, secao de baseline justo).
