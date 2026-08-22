---
name: adicionar-gate-roteamento
description: Cria um "answerability gate" determinístico novo em dashboard/roteador.py quando o chatbot alucina ou roteia errado uma pergunta específica. Use quando o usuário reportar uma resposta errada/alucinada do chat, pedir para "corrigir essa pergunta", ou quando um teste manual no Streamlit expuser um bug de roteamento ou de cruzamento de dados impossível no schema.
---

# Adicionar um gate de roteamento determinístico

Este é o padrão já estabelecido no projeto para bugs de alucinação recorrentes: em vez de confiar no LLM para se autopoliciar, intercepta a pergunta ANTES do LLM decidir, com uma checagem determinística em Python. `dashboard/roteador.py` já tem ~10 gates desse tipo — leia o arquivo inteiro primeiro, cada função `pede_*` documenta o "achado real" (bug ao vivo, com data) que a motivou.

## Passo a passo

1. **Reproduza e entenda a causa raiz.** Rode a pergunta no dashboard (ou via `dashboard/roteador.py` isolado) e identifique exatamente o que deu errado:
   - Roteou para a rota errada (ex.: foi para `sql` quando devia ir para `contexto`)?
   - Foi para a rota certa, mas o LLM/SQL gerou algo impossível (cruzamento de tabelas sem chave em comum, coluna inexistente, GROUP BY errado)?
   - O LLM interpretou a direção/magnitude de uma métrica errado mesmo com o dado certo no contexto?

2. **Verifique o schema real** (`nl_to_sql/nl_to_sql.py`, variável `ESQUEMA`) antes de escrever a regra — várias interceptações existentes são porque duas tabelas simplesmente não têm chave de junção, não porque o LLM "poderia ter tentado mais".

3. **Escreva a função `pede_<nome_do_caso>(pergunta)`** em `dashboard/roteador.py`, seguindo o padrão das existentes:
   - Docstring com: o que detecta, por que existe (achado real + data), o que acontecia sem o gate.
   - Corpo: `p = pergunta.lower()` + checagem de palavras-chave/combinações. Evite over-fitting a uma frase exata — pense em paráfrases razoáveis, mas não tente cobrir sinônimos infinitos (esse é o papel do desempate por LLM).
   - Se precisar de novas listas de palavras-chave, declare-as como constantes `PALAVRAS_CHAVE_*` no topo do arquivo, perto de onde são usadas.

4. **Registre o gate** no ponto certo:
   - Se a pergunta deve virar uma resposta 100% determinística (dado calculado, sem chamar LLM) ou "não respondível": adicione a checagem em `rotear_por_keyword()` (retorna uma rota especial, ex. `"nao_respondivel_xxx"`) ou em `rotear_pergunta()` como trava de prioridade máxima (se o risco for o desempate por LLM reclassificar errado — ver `pede_confirmacao_alarme_automatico` como exemplo desse padrão).
   - Gates de "prioridade máxima" (que interceptam antes até do keyword normal) vão no topo de `rotear_pergunta()`, na mesma ordem de execução das checagens existentes.

5. **Se a rota for `nao_respondivel_*` ou uma resposta fixa**, confira em `dashboard/app.py` (e em `eval/rag_gerador.py` se for resposta gerada) que existe tratamento para essa rota — um gate novo sem handler correspondente só muda a classificação, não a resposta final.

6. **Adicione a pergunta ao golden set** (`eval/golden_questions.json`) com a rota/resposta esperada — é assim que o harness passa a proteger contra essa regressão específica no futuro. Sem isso, o gate pode quebrar silenciosamente numa mudança futura sem que ninguém perceba.

7. **Rode o harness** (skill `rodar-harness`) para confirmar que o gate novo não regrediu nenhuma pergunta existente — um gate mal escrito pode capturar perguntas que deveriam ir para outra rota.

## Não fazer

- Não duplique a lógica de roteamento em `eval/rodar_golden.py` ou em qualquer outro lugar — `dashboard/roteador.py` é a única fonte de verdade, importada por todos os consumidores. Isso já causou uma regressão fantasma no passado (52% vs 67,9%, puramente por desatualização de cópia).
- Não tente resolver com "melhorar o prompt" um bug que já foi comprovado como recorrente mesmo com instrução explícita no prompt — o padrão do projeto é intercept determinístico em código, não confiar no LLM se autocorrigir.
