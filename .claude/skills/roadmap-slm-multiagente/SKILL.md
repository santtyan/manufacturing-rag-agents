---
name: roadmap-slm-multiagente
description: Guia vivo para avaliar e evoluir o fit do Harbor com o "Projeto 1 — Sistemas Multiagentes Confiáveis e SLMs em Arquiteturas Computacionais Abertas" (continuação do PDC 2025–2026, sublinha 1B). Use quando o usuário perguntar sobre esse projeto/edital, pedir para avançar em algum item do roadmap SLM/multiagente, ou revisitar o que já foi implementado vs. o que falta para uma proposta/candidatura nessa linha.
---

# Roadmap Harbor ↔ Projeto 1 (Multiagentes Confiáveis + SLMs)

## O que é o Projeto 1 (contexto do edital)

Continuação do eixo "Padrões de Design para Agentes Inteligentes Proativos" do PDC 2025–2026,
expandido em sistemas multiagentes híbridos LLM-SLM: **sub-agentes especializados em SLMs
rodam localmente sobre hardware aberto; LLMs frontier são acionados só quando justificável**.
Base tecnológica de plataforma: fork controlado do OpenClaw, estendido para uso corporativo.

**Sublinha 1B** (a relevante para o Harbor): desenvolvimento de SLMs especializados em
português para domínios verticais + avaliação sistemática de desempenho em arquiteturas
computacionais abertas e compactas (SoC integrado, memória unificada, fator de forma
reduzido, alta eficiência energética).

### O que é OpenClaw (pesquisado em 2026-08-22, não confundir com LangChain)

OpenClaw **não é** uma biblioteca/SDK que se importa em código para orquestrar chamadas de
LLM (isso é o papel do LangChain, mais parecido com o que o Harbor já faz manualmente com
`requests`+Ollama). OpenClaw é uma **plataforma de agente autônomo já pronta**: roda como
processo com CLI/IDE integration, lê/escreve arquivos, executa shell, escreve e testa código
sozinha — a analogia mais precisa é "OpenClaw está para agentes autônomos assim como o Claude
Code está", não "OpenClaw está para agentes assim como LangChain está para chains". Arquitetura
em 3 camadas: cognitive layer (inferência LLM, providers unificados para Claude/GPT/Gemini/
Llama/DeepSeek), execution layer (sandboxing de skills, isolamento de processo), persistence
layer (memória em SQLite/Postgres). Já tem suporte multiagente nativo.

Referência real de "fork corporativo" (o mesmo padrão descrito no edital): **NemoClaw**, da
NVIDIA (GTC 2026) — adiciona guardrails de segurança/privacidade e suporte a modelos locais
via Nemotron sobre o runtime OpenClaw, rodando sobre "NVIDIA OpenShell" (policy-based privacy/
security: identity boundaries, capability scoping, data-handling rules). Vale usar como
referência de como esse tipo de extensão corporativa costuma ser estruturada, caso surja a
decisão de portar o Harbor para essa plataforma.

**Implicação para o fit**: portar o Harbor para OpenClaw não seria "trocar uma lib por outra"
(como trocaria `requests`+Ollama por LangChain) — seria adotar uma plataforma de execução
inteira, com modelo de segurança/sandboxing/runtime próprios. Mudança de arquitetura bem mais
profunda que o que o Harbor faz hoje (scripts Python + Streamlit + FastAPI orquestrados
manualmente). Por isso o item 6 abaixo continua "fora do escopo hoje", não "impossível".

## Avaliação de fit (atualizar esta seção conforme os itens abaixo avançam)

**O que o Harbor já prova, sem trabalho adicional:**
- SLMs locais em produção (llama3.2:3b, qwen2.5:7b/14b via Ollama), com seletor de
  velocidade/qualidade no dashboard.
- Domínio vertical em português resolvido no nível de aplicação (RAG + prompts + NL-to-SQL,
  vocabulário técnico de manutenção industrial).
- Disciplina de avaliação sistemática de **qualidade** (harness, golden set, faithfulness,
  Recall@k/MRR) comparando modelos diferentes — falta a metade de **custo computacional**
  (ver item 2 abaixo).
- O padrão central da sublinha (SLM por padrão, escalar só quando justificável) já existe
  de forma embrionária: os answerability gates de `dashboard/roteador.py` interceptam com
  Python determinístico ANTES de gastar uma chamada de LLM, exatamente quando o padrão de
  erro é recorrente e conhecido — é "não confiar no modelo caro/genérico quando algo mais
  barato resolve", aplicado ao eixo de dados, não ainda ao eixo de escolha de modelo.

**Gaps reais, não simulável só com o que existe hoje:**
- Fine-tuning/LoRA/quantização/destilação — nunca foi feito no Harbor (usa modelos prontos).
- Hardware aberto compacto (SoC, memória unificada, fator de forma reduzido) — Harbor roda
  em desktop Windows convencional. Precisa de um dispositivo físico (ex: Raspberry Pi,
  Jetson, similar) para virar evidência real — não dá para simular em desktop e contar como
  validação de hardware compacto.
- OpenClaw — não é algo que se "implementa dentro do Harbor"; é uma plataforma separada.
  Só vira relevante se houver decisão de portar/integrar o Harbor sobre ela (decisão maior,
  fora do escopo de um item de lista).

## Itens implementáveis, em ordem de esforço/retorno

Marque `[x]` conforme for implementando, e adicione uma linha de resultado abaixo do item.

### Alto retorno, baixo esforço (fazer primeiro)

- [ ] **1. Documentar os answerability gates como "padrão de design para agentes proativos"
      generalizável** — reescrever o que já existe em `dashboard/roteador.py` +
      `eval/alucinacoes.md` (catálogo de bugs reais, com causa raiz e data) como um documento
      de padrão de design não-específico ao domínio industrial. É quase só reformulação —
      o trabalho empírico já foi feito.
- [ ] **2. Avaliação sistemática de custo computacional por modelo** — estender o harness
      (`eval/rodar_golden.py` ou script novo) para medir latência/tokens-por-segundo/uso de
      memória por modelo (llama3.2:3b vs qwen2.5:7b vs qwen2.5:14b), não só qualidade. Roda
      no hardware atual, mas cria o método reaproveitável para quando houver hardware
      compacto disponível.

### Esforço médio (fazer depois dos dois acima)

- [ ] **3. Generalizar o roteador para decidir MODELO, não só rota de dados** — hoje
      `roteador.py` decide contexto/rag/sql; adicionar um eixo que decide SLM rápido vs SLM
      "qualidade" vs (futuramente) LLM frontier, automatizando o que hoje é escolha manual
      do usuário no dashboard. É a peça de código que mais demonstra o conceito central da
      sublinha (escalonamento condicional SLM→LLM).
- [ ] **4. Fine-tuning leve (LoRA) de um SLM pequeno no domínio industrial em português** —
      usar o golden set + manuais técnicos como dataset de supervisão para um LoRA sobre um
      modelo pequeno (ex: llama3.2:1b/3b). Fecha a lacuna de nunca ter treinado nada, com
      escopo controlado. Exige GPU minimamente decente, mas não hardware exótico.

### Fora do escopo do Harbor hoje (não simular, exige aquisição/decisão maior)

- **5. Hardware aberto compacto** — só vira implementável com acesso físico a um SoC
  compacto (Raspberry Pi/Jetson/similar).
- **6. OpenClaw** — só vira relevante com decisão explícita de portar o Harbor para rodar
  sobre a plataforma.

## Como usar esta skill

- Ao ser invocada, releia esta seção de avaliação de fit e a lista de itens antes de sugerir
  próximos passos — não repita a análise do zero.
- Ao implementar qualquer item 1-4, volte aqui, marque `[x]`, e adicione uma linha curta
  descrevendo o resultado (bate o padrão já usado no projeto: fato + data).
- Se o usuário perguntar "o que falta para a proposta", responda com base nesta lista, não
  reconstrua a avaliação de fit do zero.
