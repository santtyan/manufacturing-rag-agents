---
name: atualizar-slide-resposta
description: Cria ou atualiza um arquivo de resposta preparada em slides/ (padrão "Resposta curta / Se perguntarem X / Números na ponta da língua") com números atuais do harness ou dos pipelines. Use quando o usuário pedir para "preparar resposta para a reunião sobre X", "atualizar o slide com os números novos", ou depois de rodar o harness/pipeline e os números mudarem em relação ao que já está documentado em slides/*.md.
---

# Atualizar um slide de resposta preparada

## Objetivo

`slides/*.md` (ex.: `slides/resposta_pipeline2.md`) guarda respostas prontas para perguntas difíceis que podem surgir em reunião — não são os slides em si (`slides/slides.html` é o rascunho de trabalho; o PPTX final é transcrito manualmente pelo usuário e **não precisa ficar sincronizado automaticamente** com o `.html`, ver memória sobre isso). Este arquivo é puramente para o usuário "ter a resposta na ponta da língua".

## Passo 1 — Obter os números atuais

Antes de escrever qualquer coisa, rode a fonte real do número que vai citar — nunca copie de memória ou de uma resposta anterior sem confirmar que ainda bate:

- Números de roteamento/faithfulness → skill `rodar-harness` (`eval/rodar_golden.py`)
- Números de um pipeline específico (MTTR/MTBF, precision/recall, etc) → releia o output em `outputs/pipelineN_*/*.json` ou `.csv` diretamente, não confie em um slide antigo como fonte.
- Números de retrieval (Recall@5/MRR) → `eval/avaliar_retrieval.py` (golden set do domínio) ou `eval/avaliar_retrieval_nanobeir.py` (benchmark externo) — não misture os dois no mesmo texto sem deixar claro qual é qual.

## Passo 2 — Escrever/atualizar no template do projeto

Siga a estrutura já usada em `slides/resposta_pipeline2.md`:

```markdown
# Resposta preparada — "<pergunta que pode surgir na reunião>"

**Resposta curta (30s):**
"<resposta direta, em 1a pessoa, citando os números exatos>"

**Por que isso é bom, não ruim, de mostrar:** (opcional — usar quando o número é fraco/negativo)
- <razão 1>
- <razão 2>

**Se perguntarem "<objeção previsível>":**
"<resposta de acompanhamento>"

**Números para ter na ponta da língua:**
- <métrica>: <valor> | <métrica>: <valor>
- <contexto numérico de apoio>
```

Nomeie o arquivo `slides/resposta_<assunto>.md` (padrão existente: `resposta_pipeline2.md`).

## Passo 3 — Conferir consistência

- Os números citados aqui devem bater com os números já publicados em `slides/slides.html` (se o assunto já estiver lá). Se divergirem, é sinal de que o slide está desatualizado — avise o usuário em vez de silenciosamente citar números diferentes em dois lugares.
- Se o número for fraco/inesperado (ex.: recall baixo, faithfulness caindo), prefira o padrão já estabelecido no projeto: explicar a causa raiz com dado real, não maquiar — ver `slides/resposta_pipeline2.md` como exemplo (baixa separabilidade entre classes, não "modelo ruim").

## Não fazer

- Não invente números ou extrapole de uma métrica para outra sem recalcular — esse é exatamente o tipo de alucinação que o projeto tem gates para evitar no chatbot; o mesmo cuidado vale para os slides.
- Não publique um número novo em `slides/slides.html` ou no PPTX a partir só deste arquivo de rascunho sem confirmar com o usuário — eles não sincronizam automaticamente.
