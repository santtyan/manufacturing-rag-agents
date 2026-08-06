# Manual Tecnico de Manutencao - Manufatura Discreta SME (Dado Tipo B)

## 1. Introducao

Este manual descreve os procedimentos de manutencao e monitoramento de estado de maquina para
o dataset de manufatura discreta, que cobre duas empresas anonimas independentes: company_A e
company_B. As duas empresas usam esquemas de status DIFERENTES e nao devem ser tratadas como
o mesmo parque de equipamentos.

## 2. Estados de Maquina - company_A

A company_A registra o estado da maquina como codigo numerico traduzido para rotulo textual:
idle (ociosa), manual (operacao manual), automatic (operacao automatica) e alarm (alarme
ativo). O estado "automatic" e o que mais aparece no historico imediatamente antes de uma
ocorrencia de alarme, seguido de "manual" -- ou seja, a maquina tende a entrar em alarme
vindo de producao automatica, nao de estado ocioso.

Procedimento de manutencao: monitorar a transicao automatic -> alarm com mais atencao que as
demais transicoes, ja que e o padrao mais frequente que precede uma parada por alarme nesta
empresa.

## 3. Estados de Maquina - company_B

A company_B usa nomenclatura textual DIFERENTE: Alarm, Standby, MachineOn, Production, Loading
e Tooling. NUNCA misturar os codigos de status de company_A com os de company_B na mesma
analise -- os valores nao sao equivalentes, mesmo quando parecem sinonimos (ex: "automatic" de
company_A nao e o mesmo conceito que "Production" de company_B, sao esquemas de captura
diferentes).

Procedimento de manutencao: o status "Alarm" de company_B nao consome mais energia que os
estados "Loading" ou "Tooling" nesta empresa -- ao contrario do que se poderia esperar de uma
falha, o consumo de energia durante alarme e MENOR que durante os estados de operacao normal
Loading/Tooling. Isso indica que o alarme aqui interrompe o processo produtivo (motores param),
nao e um pico de consumo por mau funcionamento eletrico.

## 4. Mudanca de Regime de Consumo de Energia (CUSUM)

A analise de mudanca de regime de energia (deteccao CUSUM) so esta calculada para os 3 assets
com mais registros de cada empresa (9 assets no total por empresa, 3 analisados) -- os demais
6 assets de cada empresa nao tem essa metrica calculada. Se a pergunta pedir ranking de "todos
os N assets", a resposta deve deixar claro que a analise cobre so os assets mais frequentes,
nao todos.

Procedimento de manutencao: o asset com maior numero de pontos de mudanca de regime
(instabilidade de consumo) nao e necessariamente o de maior consumo medio de energia -- sao
duas dimensoes diferentes. Investigar separadamente "quem consome mais" e "quem e mais
instavel" antes de priorizar manutencao.

## 5. Arquitetura de Diagnostico Recomendada (3 camadas)

1. **Camada 1 (regra deterministica):** thresholds fixos por estado de maquina, especificos
   de cada empresa (ex: tempo maximo aceitavel em "alarm"/"Alarm" antes de escalar).
2. **Camada 2 (estatistica/ML):** deteccao de mudanca de regime de consumo de energia (CUSUM)
   sobre os assets mais frequentes, capturando instabilidade que uma regra fixa nao capturaria.
3. **Camada 3 (validacao semantica com LLM):** para transicoes de estado atipicas, um modelo
   de linguagem analisa o contexto (sequencia de estados anteriores) e decide se a transicao e
   uma FALHA REAL, CICLO NORMAL DE PRODUCAO ou DADO INCONSISTENTE.

## 6. Escala de Criticidade

- **Critico**: transicao automatic/Production -> alarm/Alarm com duracao de alarme acima do
  historico tipico do asset -- investigar imediatamente.
- **Alerta**: asset entre os 3 com mudanca de regime de energia calculada mostrando aumento de
  instabilidade (mais pontos de mudanca que o periodo anterior) -- monitorar.
- **Informativo**: transicao manual/Standby sem alarme subsequente -- comportamento esperado
  de operacao, registrar sem acao.
