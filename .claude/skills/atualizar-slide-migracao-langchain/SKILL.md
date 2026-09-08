---
name: atualizar-slide-migracao-langchain
description: Gera/atualiza slides Beamer (.tex, para colar no Overleaf) narrando o progresso da migração do Harbor para LangChain/LangGraph — comparação Python puro x LangChain com números reais e a decisão tomada em cada módulo. Use depois de qualquer módulo migrado/comparado (ver checklist em .claude/skills/migrar-para-langchain/SKILL.md), quando o usuário pedir para "atualizar os slides da migração", "preparar isso pro Overleaf", "fazer a comparação em slide", ou "documentar o progresso da migração".
---

# Slides Beamer — progresso da migração para LangChain

## Formato: Beamer (.tex) para Overleaf, não Markdown

Diferente de `atualizar-slide-resposta` (que gera `.md` de resposta preparada para reunião),
esta skill produz **LaTeX/Beamer** — o usuário monta a apresentação no Overleaf, não no
PPTX/`slides.html` do resto do projeto. Saída: um arquivo `slides/migracao_langchain.tex`
(documento Beamer completo, um `\documentclass{beamer}` por conta própria — não é um trecho
para colar dentro de outro arquivo, a menos que o usuário já tenha um projeto Overleaf e peça
para anexar frames a ele).

Usar `references/tema_cerise.sty` (cores da identidade CERISE já usada no dashboard/slides do
projeto) e seguir a estrutura de `references/exemplo_frame_comparacao.tex` como modelo de
frame — reler os dois antes de gerar/atualizar o `.tex`, não reinventar o estilo a cada vez.

## Princípio central: narrativa, não só tabela

Cada módulo migrado vira um **frame com 3 batidas**, nessa ordem (ver exemplo em
`references/exemplo_frame_comparacao.tex`):

1. **O que medimos** — a tabela de números reais (Python puro x LangChain), num bloco
   `numerobloco`. Nunca um número estimado ou "deveria melhorar" — só o que o harness/golden
   set realmente produziu, com a fonte (script + data) anotada.
2. **O que isso significou** — 1-2 frases interpretando o número: o ganho veio do framework ou
   do algoritmo? regrediu ou não? Isso é o que transforma uma tabela fria em argumento.
3. **O que decidimos** — bloco `decisaobloco` com a decisão tomada e a razão. Se a decisão foi
   "não migrar" (número não justificou), isso também é decisão narrável, não um item para
   omitir. Se foi "migrar mesmo sem ganho isolado" por outra razão (ex.: consistência com
   LangGraph em módulos futuros — caso real do RAG, 2026-09-08), essa razão TEM que aparecer
   explicitamente no frame — é a parte que mais defende a decisão numa arguição.

Isso é o oposto de um slide de changelog ("migramos X, resultado Y") — a narrativa é sobre o
**raciocínio de engenharia**: medir antes de migrar, e decidir com base no que foi medido (ou
declarar explicitamente quando a decisão pesou outro critério além do número).

## Passo 1 — Obter o estado e os números atuais

Nunca escrever a partir de memória:

1. Reler `.claude/skills/migrar-para-langchain/SKILL.md` — checklist (`[ ]`/`[x]`) e a seção
   "Decisão registrada" (se existir) de cada módulo já avaliado.
2. Para cada módulo com checklist marcado ou em andamento, pegar os números reais do
   commit/relatório daquela comparação — ex.: `experiments/langchain_rag/README.md` (seção
   "Atualização <data>") e o CSV de resultados mais recente, para o caso do RAG híbrido. Não
   reconstruir números de memória.
3. Se um módulo está "em progresso", registrar como tal no slide, com o que já se sabe até
   agora — não inventar números finais antes de existirem.

## Passo 2 — Estrutura do documento Beamer

```latex
\documentclass{beamer}
\usepackage[utf8]{inputenc}
\usepackage[brazilian]{babel}
\usepackage{booktabs}
\usepackage{xcolor}
\input{tema_cerise}  % ou colar o conteúdo de references/tema_cerise.sty direto aqui

\title{Migração do Harbor para LangChain/LangGraph}
\subtitle{Progresso, comparações e decisões}
\author{Yan Santos Leite}
\date{\today}

\begin{document}

\frame{\titlepage}

\begin{frame}{Por que migrar}
  % 2-3 bullets: motivação real (uso futuro de LangGraph no roteador/self-repair),
  % não um argumento genérico de "LangChain é melhor"
\end{frame}

\begin{frame}{Ordem de migração (por risco)}
  % lista os 5 módulos da tabela de risco de migrar-para-langchain/SKILL.md,
  % com um ícone/cor indicando status: migrado / avaliado-e-recusado / não iniciado
\end{frame}

% -- um frame de 3 batidas (numerobloco/decisaobloco) por módulo já avaliado --

\begin{frame}{Próximos passos}
  % próximo item não marcado do checklist, mesma lógica de migrar-para-langchain/SKILL.md
\end{frame}

\end{document}
```

## Passo 3 — Conferir consistência

- Os números em cada frame devem bater com o commit/relatório real daquela migração (branch
  `migrar-langchain-*`) — se não bater, o slide está na frente do código; corrigir o slide.
- Um módulo só aparece como "migrado" no frame de status se o checklist em
  `migrar-para-langchain/SKILL.md` já tiver esse item em `[x]` — os dois arquivos concordam.
- Se uma migração regrediu e foi revertida, isso também vira um frame (3 batidas: medimos →
  regrediu → revertemos e ficamos com X) — é dado real do processo, mais forte numa
  apresentação do que só mostrar sucessos (mesmo princípio de `slides/resposta_pipeline2.md`).

## Não fazer

- Não gerar PPTX/HTML — a saída é sempre `.tex` Beamer; se o usuário quiser outro formato,
  perguntar antes de trocar.
- Não anunciar um módulo como migrado/pronto para produção no slide antes do critério de
  aceite da skill `migrar-para-langchain` ter sido cumprido (harness sem regressão, ou decisão
  explícita do usuário de migrar mesmo sem ganho isolado, registrada por escrito na skill).
- Não misturar com `slides/resposta_pipeline2.md` ou outros arquivos de assunto diferente —
  arquivo Beamer próprio (`slides/migracao_langchain.tex`), um assunto por documento.
