---
name: refatorar-organizar-repositorio
description: Audita um repositório inteiro em busca de dívida técnica estrutural (arquivos órfãos, lógica duplicada entre módulos, nomenclatura inconsistente, organização de pastas confusa, dependências não usadas), prioriza os achados e aplica refatoração com comportamento preservado, em mudanças pequenas e reversíveis. Use quando o usuário pedir para "organizar o repositório", "limpar o código", "refatorar isso", "esse projeto está bagunçado", "tem código duplicado em algum lugar?", "arruma a estrutura de pastas", ou antes de um checkpoint/entrega grande quando o usuário quiser saber o estado da dívida técnica do projeto.
---

# Refatorar e organizar repositório

Esta skill não escreve funcionalidade nova e não é um linter automático. Ela faz três coisas em
sequência: audita a dívida técnica estrutural do repositório inteiro, prioriza o que vale a pena
mexer, e executa a limpeza em mudanças pequenas e reversíveis que preservam o comportamento
existente. Nunca misture "organizar" com "adicionar/mudar comportamento" no mesmo passo — são
duas operações com riscos diferentes e devem ser avaliadas (e commitadas) separadamente.

## Passo 1 — Estabelecer um baseline reversível

Antes de tocar em qualquer arquivo:

- Rode `git status` e garanta working tree limpo. Se houver mudanças não commitadas do usuário,
  pare e pergunte — não organize por cima de trabalho em progresso alheio.
- Descubra se existe suíte de testes/smoke (`README`, `package.json`, `pyproject.toml`, pasta
  `tests/`) e rode-a antes de mexer em qualquer coisa. Se falhar já no baseline, documente isso
  separadamente — não é uma regressão sua, mas você precisa saber que já existia.
- Se não houver suíte automatizada, documente explicitamente qual verificação manual (rodar a
  aplicação, checar uma tela, um comando específico) vai substituir o "rodar teste" em cada
  mudança do Passo 4.
- Garanta um ponto de checkpoint para poder reverter (um commit do estado atual, ou pelo menos
  confirmação de que o usuário pode reverter via `git diff`/`git restore` a qualquer momento).

## Passo 2 — Auditar o repositório inteiro

Antes de decidir o que mudar, levante os achados brutos, sem julgar prioridade ainda:

- **Arquivos órfãos**: arquivos que não são importados/referenciados por nenhum outro lugar do
  código, configuração ou documentação. Confirme com busca textual pelo nome do arquivo/módulo
  antes de listar como órfão — um arquivo pode ser carregado dinamicamente (glob, plugin,
  string de config) sem aparecer em um import estático.
- **Lógica duplicada entre módulos**: a mesma decisão de negócio ou algoritmo reimplementado em
  dois ou mais lugares. Procure por padrões de definição de função/classe com nomes ou
  assinaturas parecidas em arquivos diferentes, e leia as duas implementações inteiras antes de
  marcar como duplicata — não decida só pelo nome.
- **Nomenclatura inconsistente**: convenções diferentes para a mesma coisa entre pastas (ex.:
  `snake_case` num módulo e `camelCase` em outro sem razão técnica), nomes genéricos demais
  (`utils.py`, `helpers.js`, `misc/`) que viraram gaveta de tudo.
- **Estrutura de pastas confusa**: aninhamento sem necessidade, módulos que deveriam estar
  juntos espalhados, ou splitting artificial que só existe por causa de como o projeto cresceu
  historicamente.
- **Dependências declaradas mas não usadas** (no manifesto de pacotes) e **imports não usados**
  dentro dos arquivos.

O resultado deste passo é uma lista de achados concretos, cada um com localização exata (caminho
de arquivo, e trecho relevante quando aplicável) — ainda sem decidir o que fazer com eles.

## Passo 3 — Priorizar os achados

Classifique cada achado do Passo 2 cruzando impacto (risco de bug real, atrito recorrente no dia
a dia de quem edita o código) com custo da mudança (quantos arquivos/linhas afetados, quão
espalhada está a referência):

| Prioridade | Critério | Ação |
|---|---|---|
| Crítico | Duplicação ativa que já causou ou pode causar inconsistência silenciosa (duas cópias divergindo sem ninguém perceber) | Corrigir nesta sessão |
| Alto valor | Atrito real e recorrente, custo de correção baixo/médio | Corrigir se o usuário topar o escopo |
| Nice-to-have | Melhoraria legibilidade, mas ninguém tropeçou nisso ainda | Listar, não aplicar sem pedir |
| Não vale | Custo de mudança maior que o benefício, ou risco de quebrar algo sem cobertura de teste | Documentar e deixar como está, com o motivo |

Regra explícita para duplicação: DRY é sobre **conhecimento**, não sobre **sintaxe**. Dois
trechos parecidos só são candidatos a unificação se representam a mesma decisão/regra de
negócio. Código que coincidentemente se parece, mas responde a motivações diferentes e pode
evoluir de forma independente, deve continuar separado — unificá-lo cria acoplamento artificial.

## Passo 4 — Executar mudanças pequenas e reversíveis

- Uma mudança por vez, cada uma isolável e verificável sozinha — não empacote três reorganizações
  não relacionadas no mesmo commit.
- Depois de cada mudança, rode a suíte de testes/smoke (ou a verificação manual definida no
  Passo 1). Não acumule várias mudanças para só verificar no final.
- Prefira mover/renomear/extrair a reescrever do zero — quanto menor o diff, mais fácil revisar
  e reverter.
- Ao remover um arquivo "órfão", confirme de novo com uma busca textual pelo nome antes de
  deletar, e prefira mover para fora do projeto (ou um commit dedicado, fácil de reverter) em vez
  de apagar direto se houver qualquer dúvida remanescente.
- Ao unificar lógica duplicada, garanta que o comportamento resultante é o correto para todos os
  chamadores — se as cópias já haviam divergido de propósito (uma tem uma correção que a outra
  não tem), a unificação é a oportunidade de decidir qual comportamento é o certo, não apenas
  colar as duas.

## Passo 5 — Quando não refatorar/reorganizar

Pare e não aplique a mudança quando:

- O motivo de existência do código não está claro — investigue a razão antes de assumir que é
  lixo (histórico de commits, comentários, quem chama).
- A mudança misturaria reorganização estrutural com mudança de comportamento no mesmo commit.
- A refatoração é especulativa, feita para um "possível uso futuro" que ninguém pediu.
- Não existe nenhuma forma de verificar que nada quebrou (sem testes, sem possibilidade de rodar
  a aplicação, sem revisão humana disponível) — nesse caso, liste o achado e pare, não aplique.

## Não fazer

- Não misture reorganização de pastas com mudança de lógica no mesmo commit — dificulta revisão
  e reversão, e esconde qual das duas causou uma eventual regressão.
- Não delete um arquivo "aparentemente não usado" sem confirmar via busca textual por todas as
  referências possíveis (imports, strings dinâmicas, arquivos de configuração, scripts de build).
- Não trate "os dois trechos se parecem" como prova suficiente de duplicação — leia as duas
  implementações inteiras antes de unificar.
- Não pule o baseline reversível mesmo em repositórios pequenos ou mudanças que parecem triviais.

## Checklist final de aceitação

- Havia um baseline reversível antes de começar (working tree limpo + checkpoint).
- Testes/smoke (ou verificação manual equivalente) rodaram e passaram antes e depois de cada
  mudança, não só no final.
- Cada mudança aplicada é isolável e revertível independentemente das outras.
- Nenhum commit mistura reorganização estrutural com mudança de comportamento.
- Achados de prioridade "nice-to-have" ou "não vale" foram documentados com o motivo, não
  silenciosamente ignorados.
- Nenhum arquivo foi removido sem confirmação de que não há referência restante a ele.
