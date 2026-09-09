---
name: slide-progresso-sessao
description: Gera um slide Beamer panorâmico de tudo que foi produzido num período (não uma única decisão técnica) — conquistas mensuráveis, decisões com o porquê, métricas com contexto, riscos/pendências com indicador de status, próximos passos, com um passo obrigatório de autocrítica adversarial e revisão gramatical antes de considerar o slide pronto. Use quando o usuário pedir "um slide de tudo que fizemos", "resumo de progresso para apresentar", "status report", ou "panorama da sessão/semana", em contraste com `atualizar-slide-migracao-langchain`/`slides-beamer-decisao-tecnica` (uma decisão técnica de cada vez).
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

1. **Resumo no início** — todo slide gerado por esta skill abre com um frame de resumo (1
   frame, poucas frases) descrevendo o que será apresentado a seguir. Não é um índice de
   tópicos — é uma orientação real do que o leitor vai encontrar, escrita depois que o resto do
   slide já existe (resumir é mais fácil de acertar quando o conteúdo já está pronto).
2. **Visão geral do período** — o que estava em escopo, datas/período coberto, estado geral.
3. **Conquistas mensuráveis** (2-5 itens por frame, não uma lista longa) — cada item factual e
   com número real, nunca "trabalhamos em X" sem resultado. Se não há número, não é uma conquista
   ainda, é trabalho em andamento (vai na seção de riscos/pendências).
4. **Decisões tomadas, com o porquê** — não só "decidimos migrar para Y", mas a razão (custo
   medido, ganho medido, ou razão explícita não relacionada a número, ex. consistência
   arquitetural). Reusa o `decisaobloco` das outras skills quando a decisão já tem esse formato
   pronto.
5. **Métricas com contexto** — todo número vem acompanhado do que ele significa para o objetivo
   maior, não solto. Nunca uma tabela de números sem uma frase de interpretação ao lado.
6. **Riscos/pendências com indicador de status** — usar um indicador visual (verde=fechado,
   amarelo=em andamento/parcial, vermelho=bloqueado ou esquecido) em vez de listar tudo como
   uma pilha plana de TODOs — permite que quem lê rápido veja o que precisa de atenção.
7. **Próximos passos, com dono e critério de conclusão** — não uma lista aberta, mas o que será
   entregue antes da próxima atualização e como saber que terminou.

**Diferença de audiência a decidir antes de escrever**: audiência técnica quer detalhe de
implementação; audiência executiva quer resultado/trajetória/risco, sem detalhe de código. Perguntar
ao usuário qual é o público antes de decidir o nível de detalhe de cada frame, se não estiver óbvio
pelo contexto do pedido.

## Fundamentação por escrito — o "como" e o "porquê", não só o "o quê"

Cada frame de decisão/resultado deve poder responder, no texto ou nas notas do slide:

- **Como isso foi feito?** Passo a passo verificável (arquivo, comando, script) — não um resumo
  vago tipo "melhoramos o sistema". Se o "como" não está registrado em nenhum lugar acessível
  (commit, memória, plano), voltar e reconstruir antes de escrever o frame — não aproximar.
- **Por que essa abordagem e não outra?** Razão explícita, nunca implícita. Se a decisão foi
  arbitrária (nenhum dado ou princípio a favor, só "pareceu razoável"), isso precisa aparecer
  como tal — não travestir de decisão fundamentada uma escolha que não foi.
- **Existe artigo/survey de estado da arte apoiando essa escolha?** Citado, não só mencionado de
  memória. Reusa a mesma disciplina de citação já usada nas skills de pesquisa deste projeto
  (`roadmap-rag-survey`, `rag-multimodal`) — nunca afirmar "padrão-ouro"/"estado da arte" sem uma
  fonte real por trás (nome do paper/survey, não só "a literatura diz").

## Introdução e motivação (quando o slide introduz um tema, não só reporta progresso)

Para slides que apresentam um tema pela primeira vez (não um update incremental de algo já
conhecido pela audiência), incluir explicitamente:

- **Por que o problema importa** — convencer o leitor de que vale a pena resolver, não assumir
  que é óbvio. Uma frase que situe o custo real de não resolver (tempo perdido, erro que já
  aconteceu, decisão que já foi tomada errada por falta disso).
- **Qual é a lacuna real** — o que falta hoje, defendida com evidência (um número, um exemplo
  concreto de falha), não apenas afirmada como fato.
- **Qual a contribuição concreta** e o que ela tem de diferente do que já existe — se é uma
  técnica da literatura, dizer isso; se é original ao projeto, dizer isso também.
- **Um exemplo de uso concreto** que reforça a proposta, com uma frase acessível para quem não é
  familiarizado com o domínio (manutenção industrial/IA) — não assumir conhecimento prévio do
  leitor sobre os termos técnicos do Harbor.

## Vocabulário técnico internacional

Usar os termos padrão-ouro em inglês quando existirem (ex. "Faithfulness", "nDCG@k", "late
interaction", "Contextual Retrieval" — não traduzir/inventar termo em português para um conceito
que já tem nome estabelecido na literatura). Mesma disciplina já aplicada na skill
`metricas-avaliacao-ia-industrial`.

## Passo de autocrítica adversarial (obrigatório, depois de escrever o conteúdo, antes de considerar pronto)

Depois de escrever o slide inteiro, adotar a postura de um revisor cético — de preferência numa
passada separada, relendo como se fosse a primeira vez — e responder por escrito, para cada
frame de decisão/resultado:

- O que eu mudaria para evitar perguntas ou críticas?
- Toda decisão está justificada com dado ou razão explícita, nunca "porque sim"?
- Os benchmarks/números citados são reais e recentes (não inventados, não de memória vaga)?
- Estamos de fato seguindo o padrão-ouro e o estado da arte do tema, com citação — não afirmação
  solta?
- Existe alguma decisão arbitrária que alguém razoavelmente perguntaria "por que assim e não de
  outro jeito"?
- Quais perguntas eu não saberia responder se alguém perguntasse agora, olhando este slide?
- Quais são os pontos cegos — o que eu não estou vendo que uma pessoa de fora enxergaria?

**Se a resposta honesta a qualquer uma dessas é "não" ou "não sei", o slide não está pronto** —
precisa de mais uma rodada de ajuste (buscar a fonte que falta, reescrever a frase vaga, admitir
a limitação em vez de escondê-la) antes de considerar fechado. Não pular esta etapa achando que o
conteúdo "parece bom" — o objetivo é achar o furo antes que a audiência ache.

## Revisão gramatical e de fluência em português (passo final, separado do de conteúdo)

Depois do conteúdo técnico e da autocrítica estarem resolvidos, uma passada final revisando
linha por linha como um gramático e lógico rigoroso — este passo é de natureza diferente dos
anteriores (forma, não conteúdo) e não deve ser misturado com eles:

- **Coesão**: os conectivos entre frases estão corretos (não repetidos, não faltando, não
  contraditórios com o que a frase realmente diz)?
- **Coerência**: a sequência de ideias faz sentido sozinha, sem depender do que estava na cabeça
  de quem escreveu? Um leitor que não acompanhou o processo consegue seguir o raciocínio só pelo
  texto do slide?
- **Fluidez**: sem construções estranhas ou tradução literal de termo técnico em inglês quando
  existe forma natural em português (mantendo o vocabulário técnico internacional da seção
  acima só onde ele de fato deve ficar em inglês).

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
   para a seção de riscos/pendências acima.
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
3. Montar um frame por frente de trabalho relevante (cada um podendo usar a estrutura de 3
   batidas das outras skills quando aplicável, com fundamentação do "como"/"por quê" e, quando
   aplicável, introdução/motivação), um frame de riscos/pendências com indicador, e um frame
   final de próximos passos.
4. Escrever o frame de resumo inicial por último, depois que o resto já existe.
5. Rodar o passo de autocrítica adversarial — corrigir o que ela apontar.
6. Rodar a revisão gramatical/de fluência como passada final e separada.
7. Não incluir mais de 5-7 frames de conteúdo (fora título e resumo) — se há mais frentes de
   trabalho que isso, agrupar por tema ou perguntar ao usuário o que priorizar, não tentar caber
   tudo.
8. Conferir que todo número do slide bate com uma fonte real citada — mesma disciplina das
   outras skills de slide do projeto.

## Não fazer

- Não confundir com `atualizar-slide-migracao-langchain`/`slides-beamer-decisao-tecnica` — esta
  skill é para o panorama, aquelas para uma decisão técnica isolada. Se o pedido for sobre uma
  decisão só, usar a skill certa em vez desta.
- Não listar "trabalho em andamento" como "conquista" — só o que já tem resultado medido entra
  na seção de conquistas; o resto vai em riscos/pendências ou próximos passos.
- Não gerar mais de uma dezena de frames sem antes checar com o usuário se ele quer esse nível de
  detalhe — um panorama que vira changelog completo perde a função de panorama.
- Não pular o passo de autocrítica adversarial achando que "está óbvio que está bom" — é
  exatamente esse tipo de suposição que ela existe para capturar.
- Não misturar a revisão gramatical com a revisão de conteúdo — são passes de natureza diferente,
  fazer os dois ao mesmo tempo faz perder itens de ambas as listas.
