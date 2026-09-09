# Rollout Card - sql

Gerado a partir de `C:\Projetos\Harbor\eval\traces\sql.jsonl` (3 execucoes).

## Modelo e versoes

- Modelo(s): qwen2.5:7b (3x)
- Dependencias:
  - langgraph: 1.6.2
  - langchain-ollama: 1.1.0
- Tools disponiveis ao agente: executar_sql

## Prompt (resumo)

Traduzir pergunta em portugues para SQL PostgreSQL e executar via ReAct (ate 6 iteracoes).

## Resultado

- Execucoes: 3
- Taxa de sucesso: 33.3% (1/3)
- Passos por execucao (media): 9.0
- Duracao por execucao (media): 235.35s

## Custo (tokens, somado sobre todas as execucoes)

- Input: 27163
- Output: 1342
- Total: 28505

## O que quebrou (execucoes sem sucesso)

- acao_invalida_ou_erro_execucao: 2

## Reprodutibilidade

- Trace bruto (JSONL) disponivel em `C:\Projetos\Harbor\eval\traces\sql.jsonl` - cada linha e uma execucao completa, com thought/action/observation por passo, custo e modelo usados naquele passo especifico.
- Regra do projeto: nenhuma conclusao sobre este resultado e valida sem o baseline comparavel a custo de tokens normalizado (ver skill `migrar-para-langchain`, secao de baseline justo).
