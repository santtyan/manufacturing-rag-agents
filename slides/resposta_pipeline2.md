# Resposta preparada — "Por que o pipeline 2 teve precision/recall tão baixos?"

**Resposta curta (30s):**
"O Isolation Forest teve precision de 0.25 e recall de 0.10 porque, quando comparei a média de cada
sensor entre as classes Fault e Normal, os valores são praticamente idênticos — por exemplo,
temperatura média de 69.4°C em Fault contra 70.1°C em Normal, uma diferença de menos de 1°C. Isso
significa que, neste dataset especificamente, o rótulo Fault não está fortemente correlacionado com
os valores numéricos dos sensores — provavelmente foi gerado de forma sintética ou depende mais do
texto (Operator_Notes, Error_Message) do que dos números."

**Por que isso é bom, não ruim, de mostrar:**
- Prova que você validou o modelo contra a realidade em vez de aceitar cegamente
- É exatamente a disciplina de "backtesting contra ground truth" que você citou na abordagem
- Reportar um resultado fraco com explicação é mais forte cientificamente do que reportar métricas altas sem checar

**Se perguntarem "e daí, o pipeline não serve pra nada?":**
"Serve para provar a arquitetura de 3 camadas funcionando ponta a ponta (regra → Isolation Forest →
LLM), que é o que precisamos validar agora. A qualidade do resultado depende do dataset — com dados
reais da HarboR, onde a correlação sensor-falha é fisicamente real (não sintética), a mesma
arquitetura deve performar muito melhor. É a arquitetura que estamos entregando, não uma alegação de
alta acurácia neste dataset específico."

**Números para ter na ponta da língua:**
- Precision: 0.248 | Recall: 0.097 | F1: 0.14
- 639 casos de Fault reais no dataset, 250 anomalias detectadas pelo Isolation Forest
- Diferença de temperatura entre classes: < 1°C (69.4 vs 70.1) — evidência direta da baixa separabilidade
