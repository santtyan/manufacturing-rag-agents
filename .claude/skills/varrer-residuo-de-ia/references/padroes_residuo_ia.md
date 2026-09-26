# Catálogo de padrões de resíduo e falsos positivos conhecidos

Complemento de `../SKILL.md`. A segunda metade deste arquivo é a parte mais importante: são os
falsos positivos que a regex ingênua do padrão-ouro público produz num repositório como o Harbor.

## Parte 1 — Padrões de resíduo, por certeza

### Certeza ALTA

| Padrão | Por que é resíduo |
|---|---|
| `*.bak`, `*.orig`, `*.tmp`, `*.save`, `*.swp` | Subproduto de editor ou de merge; nunca é fonte |
| `scratch*`, `temp_*`, `debug_*`, `rascunho*`, `wip_*` | Nome declara a intenção descartável |
| `*_old`, `*_copy`, `*_copia`, `*_backup`, `*_bkp` | Versionamento manual — o git já faz isso |
| `* (1).ext`, `* - Copia.ext` | Duplicata criada pelo sistema de arquivos |
| `Untitled*`, `sem_titulo*` | Arquivo criado e nunca nomeado |
| `__pycache__/`, `*.pyc`, `.pytest_cache/` em `git ls-files` | Cache que entrou no versionamento antes do `.gitignore` |
| Diretório byte-idêntico e subconjunto de outro, sem referência externa | Cópia estagnada — divergirá silenciosamente |

### Certeza MÉDIA

| Padrão | Por que exige julgamento |
|---|---|
| Untracked fora do `.gitignore` | Pode ser trabalho em andamento legítimo |
| Resultado com sufixo de variante (`_v2`, `_qwen`, `_100`) | Pode ser registro histórico de medição |
| Corpo só `pass` / `return None` / `...` | Pode ser implementação de protocolo ou classe base |
| Docstring citando símbolo inexistente | Pode ser renomeação incompleta, não resíduo |
| Comentário muito acima da mediana | Neste projeto, comentário denso costuma ser intencional |
| Adicionado num commit e nunca mais tocado | Pode ser código estável e correto |

### Certeza BAIXA

Saída de `vulture` e `jscpd`. Pista, nunca conclusão.

## Parte 2 — Falsos positivos conhecidos

### 2.1 A palavra portuguesa "todo" (crítico)

Este é o falso positivo de maior volume. O repositório é inteiramente em português, e "todo"/
"toda" é uma palavra comum:

```
eval/verificacao.py:6:      "todo numero citado na resposta precisa aparecer no contexto"
tests/test_roteador_langgraph.py:53: "concordam em TODO o golden set"
eval/rodar_golden.py:10:    "antes reimplementava call_ollama() e TODO o roteamento"
rag/rag_openpack_texto.py:185: "aceleracao/rotacao ao todo neste segmento"
```

Um `grep -i todo` no Harbor devolve ~18 linhas e **nenhuma é um marcador de tarefa pendente**.

**Regra:** busque `TODO:` / `FIXME:` / `XXX:` **com dois-pontos**, ou `# TODO` no início do
comentário. Nunca `todo` case-insensitive e solto. Em repositórios em português, prefira ainda
exigir maiúsculas: `\b(TODO|FIXME|XXX|HACK):`.

### 2.2 `placeholder` como API de framework

`dashboard/app.py:209-221` usa `self.placeholder` para guardar um `st.empty()` do Streamlit — é o
nome oficial do conceito no framework, não texto de gabarito:

```python
self.placeholder = None
...
self.placeholder = st.empty()
self.placeholder.markdown(...)
self.placeholder.empty()
```

**Regra:** `placeholder` só é achado quando é *conteúdo* (uma string literal de gabarito), nunca
quando é *identificador* (nome de variável, atributo ou parâmetro). Streamlit, Tkinter, HTML e
Pydantic todos usam a palavra legitimamente.

### 2.3 `langchain.agents` vs. o diretório `.agents/`

Ao checar se o diretório `.agents/` tem consumidores, `grep '\.agents'` casa com todo
`from langchain.agents import create_agent` — dezenas de hits que nada têm a ver com o diretório.

**Regra:** ancore o padrão no caminho (`\.agents/`, com barra) e exclua matches dentro do próprio
diretório candidato. Aplicando as duas correções no Harbor, o número de consumidores reais de
`.agents/` cai de "dezenas" para **zero**.

### 2.4 Função referenciada só por despacho em dicionário

Uma função pode nunca aparecer como `nome(` e ainda assim ser chamada via
`dicionario[chave](...)`. Mesmo falso positivo que `revisar-funcoes-orfas` já trata — se o achado
é uma função, delegue para aquela skill em vez de reproduzir a análise aqui.

### 2.5 Arquivo vazio ou quase vazio que é legítimo

`__init__.py` vazio (marca pacote), `py.typed` (marca tipagem), `.gitkeep` (preserva diretório
vazio), `conftest.py` mínimo. Nenhum é resíduo.

### 2.6 Arquivo grande que é dado de produção

`rag/corpus_openpack_janelas.json` (5,0 MB) é o maior arquivo versionado do Harbor e é corpus de
produção, consumido pelo RAG. Tamanho isolado não é sinal de resíduo — cruze sempre com a busca
por consumidor.

### 2.7 Comentário denso que documenta um achado real

Convenção do Harbor: comentários registram um "achado real" (bug encontrado ao vivo, com data)
que motivou a regra atual, e `CLAUDE.md` instrui explicitamente a lê-los antes de mexer em
roteamento/gates. A razão comentário-por-código alta nesses arquivos é intencional.

**Regra:** só é slop o comentário que *reafirma o que o código já diz*
(`# incrementa o contador` sobre `contador += 1`). Comentário que carrega contexto histórico que
não está em nenhum outro lugar é o oposto de resíduo.

### 2.8 Script cujo caminho de re-execução está documentado

Um `.py` standalone sem import em nenhum lugar parece órfão, mas pode ter seu comando de execução
documentado em `CLAUDE.md` ou numa SKILL.md. Cheque essas duas fontes antes de classificar — e,
se houver dúvida real, delegue para `auditar-scripts-orfaos`, que existe para exatamente isso.
