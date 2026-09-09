---
name: slide-progresso-sessao
description: Gera um slide Beamer panorâmico de tudo que foi produzido num período (não uma única decisão técnica) — conquistas mensuráveis, decisões com o porquê, métricas com contexto, riscos/pendências com indicador de status, próximos passos. Use quando o usuário pedir "um slide de tudo que fizemos", "resumo de progresso para apresentar", "status report", ou "panorama da sessão/semana", em contraste com `atualizar-slide-migracao-langchain`/`slides-beamer-decisao-tecnica` (uma decisão técnica de cada vez).
---

# Slide panorâmico de progresso (status report)

## Diferença central em relação às outras skills de slide do projeto

`atualizar-slide-migracao-langchain` e `slides-beamer-decisao-tecnica` produzem um frame de
"3 batidas" (o que medimos → o que significou → o que decidimos) para **uma** decisão técnica
isolada. Esta skill produz o nível acima: um **panorama de várias frentes de trabalho** num
período — várias decisões, vários números, o estado de cada uma. Não recria o trabalho daquelas
skills; reusa o mesmo formato de 3 batidas **dentro** de cada frame de frente de trabalho, quando
aplicável.

## Padrão-ouro pesquisado (2026-09-09): estrutura de status report de engenharia

Pesquisa em templates/práticas de status report e sprint review de times de engenharia — a
estrutura que se repete em toda fonte, adaptada aqui para slide técnico (não gerencial):

1. **Visão geral do período** — o que estava em escopo, datas/período coberto, estado geral.
2. **Conquistas mensuráveis** (2-5 itens por frame, não uma lista longa) — cada item factual e
   com número real, nunca "trabalhamos em X" sem resultado. Se não há número, não é uma conquista
   ainda, é trabalho em andamento (vai na seção de riscos/pendências).
3. **Decisões tomadas, com o porquê** — não só "decidimos migrar para Y", mas a razão (custo
   medido, ganho medido, ou razão explícita não relacionada a número, ex. consistência
   arquitetural). Reusa o `decisaobloco` das outras skills quando a decisão já tem esse formato
   pronto.
4. **Métricas com contexto** — todo número vem acompanhado do que ele significa para o objetivo
   maior, não solto. Nunca uma tabela de números sem uma frase de interpretação ao lado.
5. **Riscos/pendências com indicador de status** — usar um indicador visual (verde=fechado,
   amarelo=em andamento/parcial, vermelho=bloqueado ou esquecido) em vez de listar tudo like
   uma pilha plana de TODOs — permite que quem lê rápido veja o que precisa de atenção.
6. **Próximos passos, com dono e critério de conclusão** — não uma lista aberta, mas o que será
   entregue antes da próxima atualização e como saber que terminou.

**Diferença de audiência a decidir antes de escrever**: audiência técnica quer detalhe de
implementação; audiência executiva quer resultado/trajetória/risco, sem detalhe de código. Perguntar
ao usuário qual é o público antes de decidir o nível de detalhe de cada frame, se não estiver óbvio
pelo contexto do pedido.

## Fontes de dado deste projeto (nunca inventar número, sempre coletar das fontes reais)

Antes de escrever qualquer frame, coletar do estado real do repositório, na ordem:

1. **Commits do período** (`git log --oneline <desde>..HEAD`) — a lista bruta de trabalho feito;
   filtrar para o que é conquista real (não WIP/typo fix).
2. **Rollout Cards** (`eval/traces/*_rollout_card.md`, `shared/rollout_card.py`) — já são
   relatórios de replicação prontos para qualquer agente/experimento medido; reusar diretamente
   como fonte de métrica, não recalcular.
3. **Resultados de harness/benchmark** (`eval/resultados_*.csv`, `eval/resultados_*.json`) —
   números reais já persistidos, nunca de memória.
4. **Checklists de skills vivas** (`migrar-para-langchain`, `roadmap-rag-survey`,
   `roadmap-slm-multiagente`, etc.) — o que está `[x]` vs `[ ]` já é o indicador de status pronto
   para a seção 5 acima.
5. **Memórias de sessão** (`C:\Users\USER\.claude\projects\<projeto>\memory\`) — decisões e
   achados já registrados, com o "why" já escrito — reaproveitar a redação, não reescrever do
   zero.
6. **Plano de implementação ativo** (`C:\Users\USER\.claude\plans\*.md`, se existir) — a fila de
   trabalho pendente já registrada é a fonte direta da seção de riscos/próximos passos.

## Formato de saída

Beamer (.tex, para Overleaf) — mesmo padrão das outras skills de slide do projeto. Usar
`references/tema_cerise.sty` de `atualizar-slide-migracao-langchain` (não duplicar o arquivo,
referenciar/copiar de lá) para manter identidade visual consistente entre todos os slides do
Harbor. Arquivo de saída: perguntar ao usuário o nome (ex. `slides/progresso_<periodo>.tex`) —
não sobrescrever `slides/migracao_langchain.tex`, que é de outra skill.

## Como usar esta skill

1. Perguntar (se não estiver claro) o período coberto e a audiência (técnica vs. executiva).
2. Coletar dados das 6 fontes acima — nunca escrever frame a partir de memória da conversa.
3. Montar um frame de "visão geral" primeiro, depois um frame por frente de trabalho relevante
   (cada um podendo usar a estrutura de 3 batidas das outras skills quando aplicável), um frame
   de riscos/pendências com indicador, e um frame final de próximos passos.
4. Não incluir mais de 5-7 frames de conteúdo (fora título) — se há mais frentes de trabalho que
   isso, agrupar por tema ou perguntar ao usuário o que priorizar, não tentar caber tudo.
5. Conferir que todo número do slide bate com uma fonte real citada — mesma disciplina das
   outras skills de slide do projeto.

## Não fazer

- Não confundir com `atualizar-slide-migracao-langchain`/`slides-beamer-decisao-tecnica` — esta
  skill é para o panorama, aquelas para uma decisão técnica isolada. Se o pedido for sobre uma
  decisão só, usar a skill certa em vez desta.
- Não listar "trabalho em andamento" como "conquista" — só o que já tem resultado medido entra
  na seção de conquistas; o resto vai em riscos/pendências ou próximos passos.
- Não gerar mais de uma dezena de frames sem antes checar com o usuário se ele quer esse nível de
  detalhe — um panorama que vira changelog completo perde a função de panorama.
