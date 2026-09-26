---
name: varrer-residuo-de-ia
description: Varre um repositório em busca de resíduo deixado por sessões de agente de IA — arquivos que não deveriam existir (scratch que virou permanente, pasta duplicada, resultado de experimento abandonado, cache versionado por engano, untracked sem decisão) e marcadores de slop dentro do código (placeholder, stub vazio, docstring que descreve um estado que não existe mais, comentário inflado). Classifica cada achado em certeza ALTA/MÉDIA/BAIXA com a evidência da busca e nunca remove nada sozinha, nem em certeza ALTA. Use quando o usuário pedir para "achar arquivos inúteis gerados pela IA", "tem arquivo temporário sobrando?", "varre o lixo do repo", "limpar resíduo de sessão", "isso aqui foi gerado e ficou?", ou antes de um commit/entrega grande para conferir que nada de scratch vazou para o versionamento.
---

# Varrer resíduo de IA

Esta skill responde a uma pergunta específica: **o que nesta árvore não deveria ter sido criado?**
Sessões longas de agente produzem arquivos de apoio que nunca foram limpos, e código com marcas de
geração que ninguém revisou. A skill encontra esses casos, classifica cada um por nível de certeza,
e **nunca remove nada sozinha** — nem os achados de certeza ALTA.

Isso é um desvio deliberado do padrão-ouro público da categoria (`agent-sh/deslop`,
`kirankunapuli/stop-ai-slop`), que auto-corrige achados de certeza alta. Aqui a certeza governa
**a ordem e a força da recomendação, não a autonomia**, por coerência com a regra já fixada no
projeto: análise estática é evidência adicional, nunca veredito.

## Fronteira — quando delegar em vez de continuar

Esta skill cobre *resíduo* (o que não deveria existir). Dívida técnica estrutural é de outras
skills. Ao encontrar um achado dos tipos abaixo, **reporte e delegue**, não reimplemente:

| Achado | Skill |
|---|---|
| Script standalone de propósito duvidoso, mas plausivelmente útil | `auditar-scripts-orfaos` |
| Função/método definido e nunca chamado | `revisar-funcoes-orfas` |
| Lógica duplicada entre módulos, estrutura de pastas confusa | `refatorar-organizar-repositorio` |

A diferença prática: um arquivo que *alguém pode querer rodar de novo* não é resíduo — é escopo de
`auditar-scripts-orfaos`. Resíduo é o que existe por acidente de processo.

## Regras de segurança — valem em todas as fases

1. **Use a ferramenta Grep (ripgrep), nunca `grep -r`.** `grep -r` entra em `.venv/` e estoura o
   timeout (achado real: aconteceu duas vezes na sessão que originou esta skill, 120s cada).
2. **Nunca varra** `.venv/`, `venv/`, `node_modules/`, `outputs/`, `rag/chroma_db*`,
   `wheelhouse/`, `.git/`. Prefira `git ls-files` como fonte da lista de arquivos: ele já respeita
   o `.gitignore` de graça.
3. **Desconte auto-referência antes de declarar um diretório órfão.** Ao checar se `X/` é
   referenciado, exclua os matches que estão *dentro de* `X/`. Um diretório de documentação cita
   o próprio nome dezenas de vezes e parece "muito referenciado" sem ter nenhum consumidor real.
   E ancore o padrão no caminho (`X/`, com barra): sem a barra, `\.agents` casa com
   `from langchain.agents import ...` (20 arquivos no Harbor, todos falsos positivos).
4. **Um match não é um consumidor — leia a linha.** Depois de reduzir a lista, abra cada match e
   pergunte se aquela linha de fato *carrega* uma dependência. Menção condicional
   (`"If X/ exists, read it"`) cuja condição nunca é satisfeita não é consumo. Foi esse passo que
   distinguiu, no Harbor, 4 matches restantes de 0 consumidores reais.
5. **Arquivo não-versionado e não-ignorado é achado de classificação, não de remoção.** A pergunta
   correta é "isso entra no git ou no `.gitignore`?", nunca "posso apagar?".
6. **Leia o arquivo antes de recomendar remoção.** Vale especialmente para os `.json`/`.csv` de
   resultado: alguns são o número de referência citado em `CLAUDE.md` ou num slide.

## Fase 1 — Certeza ALTA: arquivos inequívocos

**1a. Nomes de scratch e backup.** Sobre `git ls-files` e também sobre os untracked
(`git ls-files --others --exclude-standard`):

- extensões: `*.bak`, `*.orig`, `*.tmp`, `*.temp`, `*.save`, `*.swp`, `*.old`
- prefixos: `scratch*`, `temp_*`, `tmp_*`, `debug_*`, `teste_rapido*`, `rascunho*`, `wip_*`
- sufixos: `*_old`, `*_copy`, `*_copia`, `*_backup`, `*_bkp`, `*_final`, `*_final2`, `*_v2`
- padrões de duplicata do sistema: `* (1).*`, `* - Copia.*`, `Untitled*`, `sem_titulo*`

**1b. Cache versionado por engano.** `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.ruff_cache/`,
`.ipynb_checkpoints/` aparecendo em `git ls-files` (não apenas em disco). Se aparecem em
`git ls-files`, o `.gitignore` foi adicionado depois do commit — precisa de `git rm --cached`.

**1c. Diretório duplicado, detectado por conteúdo.** É o achado mais valioso e o mais fácil de
perder no olho. Para dois diretórios candidatos, compare a lista de arquivos relativa e depois o
conteúdo byte a byte por amostra. Se um é subconjunto do outro e as amostras são idênticas, o
subconjunto é uma cópia estagnada. Ver `references/comandos_varredura.md` para o comando pronto.

Só classifique como ALTA se, além de idêntico, o diretório passar na regra de segurança 3
(nenhuma referência externa a ele).

## Fase 2 — Certeza MÉDIA: precisa de julgamento

**2a. Untracked sem decisão.** Arquivos em `git ls-files --others --exclude-standard`: não estão
no git nem no `.gitignore`. Cada um precisa de uma decisão explícita. Um par gerador + saída
(ex.: um `.py` que produz um `.pptx`, ambos untracked) é o caso típico: normalmente o gerador
entra no git e a saída no `.gitignore`.

**2b. Resultado de experimento com sufixo de variante.** Arquivos de resultado cujo nome carrega
a variante do experimento (`_v2`, `_final`, `_100`, `_qwen`, `_sem_temperature`, `_completo`).
Acumulam porque cada execução cria um nome novo. Para cada um, busque um consumidor: é citado em
`CLAUDE.md`, num slide, num `.md` de documentação, ou lido por algum `.py`? Sem consumidor, é
candidato — mas **um resultado sem consumidor pode ser o registro histórico de uma medição que
não deve ser perdido**; reporte, não conclua.

**2c. Stub e placeholder de verdade.** Função cujo corpo é só `pass`, `return None`, `...` ou
`raise NotImplementedError`, sem decorator de interface/protocolo que justifique. Strings de
gabarito: `your_api_key`, `sua_chave_aqui`, `insert here`, `coloque aqui`, `lorem ipsum`,
`example.com`, `TODO:`/`FIXME:`/`XXX:` **com dois-pontos**. Ver os falsos positivos em
`references/padroes_residuo_ia.md` antes de reportar qualquer coisa desta subfase.

**2d. Docstring ou comentário que descreve um estado morto.** Docstring que cita um arquivo,
função ou parâmetro que não existe mais; comentário "temporário, remover depois" com data
antiga; bloco comentado grande de código anterior. Detectável cruzando nomes citados em
docstrings com os símbolos que existem de fato.

**2e. Razão comentário-por-código anormal.** Arquivo cujo volume de comentário é muito acima da
mediana do repositório — sintoma clássico de geração. **Atenção:** neste projeto, comentário
denso que documenta um "achado real" com data é intencional e valioso, não slop. Só reporte
comentário que *reafirma o que o código já diz*.

**2f. Arquivo adicionado e nunca mais tocado.** `git log --diff-filter=A` para achar a data de
entrada; cruzada com a data do último commit que tocou o arquivo. Entrou num único commit, nunca
mais foi alterado e ninguém o referencia é um perfil forte de resíduo de sessão.

## Fase 3 — Certeza BAIXA: ferramenta externa, opcional

Só se as fases 1 e 2 deixaram dúvida, e só como **evidência adicional, nunca veredito** — regra
já fixada no projeto para estas duas ferramentas:

- `vulture` — código morto em Python, com score de confiança próprio. Trate o score como pista.
- `jscpd` — bloco duplicado entre arquivos.

Se as ferramentas não estiverem instaladas, **não instale**: registre que a Fase 3 foi pulada.

## Passo final — Relatório e confirmação item por item

Monte uma tabela única, ordenada por certeza decrescente:

| Arquivo/símbolo | Certeza | Categoria | Evidência | Ação sugerida |
|---|---|---|---|---|

Regras do relatório:

- **Evidência é obrigatória e específica.** "Nenhum match para `X` fora de `X/` em `*.py`,
  `*.md`, `*.ps1`" é evidência; "parece não usado" não é.
- **Ação sugerida é uma proposta, não uma conclusão.** Distinga `git rm --cached` (sai do
  versionamento, fica em disco) de remoção do disco — são decisões diferentes.
- **Declare o que foi pulado**: fase não executada, diretório não varrido, ferramenta ausente.
- **Peça confirmação item por item.** Nunca em lote, nem para os ALTA. Cada achado é uma decisão
  separada, e a taxa de falso positivo desta categoria de análise é alta o suficiente para que
  aprovação em bloco seja insegura.
- Ao remover algo confirmado, verifique se a remoção deixa outra coisa órfã por tabela (um
  `.gitignore` apontando para um caminho que deixou de existir, um import quebrado, uma linha de
  índice em `CLAUDE.md` ou `MEMORY.md`).

## Verificação de que a skill está funcionando

Rodada no repositório Harbor, a skill deve reencontrar os achados conhecidos (ver a tabela de
baseline no plano que originou esta skill) e, sobretudo, **não** reportar os dois falsos
positivos catalogados em `references/padroes_residuo_ia.md`: a palavra portuguesa "todo" e o
atributo `self.placeholder` do Streamlit. Falso positivo em qualquer um dos dois indica que a
regex da Fase 2c foi aplicada sem as exceções.

`git status --porcelain` antes e depois da varredura deve ser idêntico: a varredura só lê.
