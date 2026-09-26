# Comandos de varredura

Todos os comandos abaixo foram executados no Harbor e a saída conferida. Usam `git ls-files` como
fonte da lista de arquivos, o que já exclui `.venv/`, `outputs/` e `rag/chroma_db*` de graça pelo
`.gitignore` — é por isso que nenhum deles precisa de lista de exclusão manual.

Todos são **somente leitura**.

## Fase 1a — Nomes de scratch e backup

```bash
# versionados
git ls-files | grep -iE '\.(bak|orig|tmp|temp|save|swp|old)$|(^|/)(scratch|temp_|tmp_|debug_|rascunho|wip_)|(_old|_copy|_copia|_backup|_bkp|_final|_v2)\.|\([0-9]\)\.|Untitled'

# untracked (não estão no git nem no .gitignore)
git ls-files --others --exclude-standard
```

PowerShell:
```powershell
git ls-files | Select-String -Pattern '\.(bak|orig|tmp|temp|save|swp|old)$|(^|/)(scratch|temp_|tmp_|debug_|rascunho|wip_)|(_old|_copy|_copia|_backup|_bkp|_final|_v2)\.|\([0-9]\)\.|Untitled'
```

## Fase 1b — Cache versionado por engano

```bash
git ls-files | grep -E '__pycache__|\.pyc$|\.pytest_cache|\.ipynb_checkpoints|\.ruff_cache'
```

Achado aqui exige `git rm --cached`, não remoção do disco — o `.gitignore` provavelmente já cobre
o caminho, mas foi adicionado depois do commit.

## Fase 1c — Diretório duplicado, por conteúdo

O comando mais valioso da skill. Compara dois diretórios candidatos arquivo por arquivo:

```bash
A=.agents/skills; B=.claude/skills
comuns=$(comm -12 <(git ls-files "$A" | sed "s|^$A/||" | sort) \
                  <(git ls-files "$B" | sed "s|^$B/||" | sort))
tot=0; ident=0
for f in $comuns; do
  tot=$((tot+1))
  cmp -s "$A/$f" "$B/$f" && ident=$((ident+1))
done
echo "em comum: $tot | byte-identicos: $ident"
echo "só em $A: $(comm -23 <(git ls-files "$A" | sed "s|^$A/||" | sort) <(git ls-files "$B" | sed "s|^$B/||" | sort) | wc -l)"
echo "só em $B: $(comm -13 <(git ls-files "$A" | sed "s|^$A/||" | sort) <(git ls-files "$B" | sed "s|^$B/||" | sort) | wc -l)"
```

Saída no Harbor: `em comum: 53 | byte-identicos: 53`, `só em .agents/skills: 0`,
`só em .claude/skills: 22`. Leitura: `.agents/skills` é **subconjunto próprio e perfeito** de
`.claude/skills`. Zero arquivos exclusivos + 100% de identidade byte a byte = cópia estagnada.

Compare a contagem, não uma amostra. Amostra de 3 arquivos daria a mesma pista, mas "53/53" é o
que sustenta classificar como certeza ALTA.

## Regra 3 — Consumidor externo, descontando auto-referência

```bash
# CORRETO: âncora na barra + exclui matches dentro do próprio diretório
git ls-files | grep -v '^\.agents/' | xargs grep -lE '\.agents/'

# ERRADO: casa com 'from langchain.agents import ...' em 20 arquivos
git ls-files | xargs grep -l '\.agents'
```

No Harbor, a versão correta devolve 4 arquivos e a ingênua 20. Mas os 4 exigem um segundo passo de
leitura: são a cópia em `.claude/` da skill `eval-engineering`, e as duas linhas são **condicionais**
— `"If .agents/skills/<project>-world/SKILL.md exists, read it"`. Como nenhum `*-world/SKILL.md`
existe no repositório, a condição nunca dispara: não é consumidor.

Lição para a skill: um match não é um consumidor. Leia a linha e pergunte se ela de fato *carrega*
uma dependência, ou apenas menciona um caminho hipotético.

## Fase 2b — Resultado sem consumidor documentado

```bash
# lista os resultados
git ls-files 'eval/resultados_*' 'eval/resultado_*'

# para um candidato, busca consumidor em código e documentação
nome="resultados_golden_sem_temperature.csv"
git ls-files '*.py' '*.md' '*.tex' | xargs grep -l "$nome"
```

## Fase 2c — Placeholder e stub, com as exceções aplicadas

```bash
# TODO/FIXME REAIS: exige dois-pontos e maiúsculas (ver falso positivo 2.1)
git ls-files '*.py' | xargs grep -nE '\b(TODO|FIXME|XXX|HACK):'

# gabaritos, como conteúdo de string
git ls-files '*.py' | xargs grep -niE '(your_api_key|sua_chave_aqui|insert here|coloque aqui|lorem ipsum|example\.com)'

# stubs
git ls-files '*.py' | xargs grep -nE -B1 '^\s+(pass|\.\.\.|return None|raise NotImplementedError)\s*$'
```

A primeira linha devolve **zero** no Harbor — e é o resultado certo. A variante ingênua
(`grep -i todo`) devolve ~18 linhas, todas da palavra portuguesa "todo". Nunca use a ingênua.

## Fase 2f — Adicionado num commit e nunca mais tocado

```bash
f="caminho/do/arquivo"
echo "add=$(git log --diff-filter=A --format=%ad --date=short -- "$f" | tail -1)"
echo "último=$(git log -1 --format=%ad --date=short -- "$f")"
echo "commits=$(git log --oneline -- "$f" | wc -l)"

# para um diretório inteiro
git log --format='%ad %h %s' --date=short -- .agents/
```

`commits=1` com `add == último` é o perfil de resíduo de sessão. No Harbor, `.agents/` inteiro tem
exatamente **um** commit (`ff356a9`, 2026-09-08) e nunca foi tocado depois — enquanto
`slides/harbor_narrativa_2026-09.tex` tem 5 commits ao longo de 4 dias, o perfil de um arquivo vivo.

## Arquivos pesados versionados

```bash
git ls-files -z | xargs -0 ls -l 2>/dev/null | awk '$5>1000000 {printf "%.1f MB  %s\n", $5/1048576, $9}' | sort -rn
```

Tamanho isolado não é sinal — cruze com busca por consumidor. No Harbor o único hit
(`rag/corpus_openpack_janelas.json`, 5,0 MB) é corpus de produção.

## Fase 3 — Ferramentas externas (opcional)

```bash
vulture . --min-confidence 80 --exclude .venv,outputs
jscpd --pattern "**/*.py" --ignore ".venv/**,outputs/**"
```

Se não estiverem instaladas, **não instale**: registre a Fase 3 como pulada.

## Conferir que a varredura não alterou nada

```bash
git status --porcelain   # deve ser idêntico ao estado anterior à varredura
```
