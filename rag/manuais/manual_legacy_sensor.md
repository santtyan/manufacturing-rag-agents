# Manual Tecnico de Manutencao Preditiva - Sensores Legados (Dado Tipo B)

## 1. Introducao

Este manual descreve os procedimentos de manutencao preditiva para o parque de equipamentos
industriais legados monitorado pelo dataset Legacy Industrial Equipment Sensor Logs. O
equipamento e classificado em cada leitura como Normal ou Fault (rotulo real, ground truth),
com um modelo de deteccao de anomalia (Isolation Forest) avaliado contra esse rotulo.

## 2. Limitacao Conhecida dos Sensores Numericos

Os sensores numericos (Temperature_C, Pressure_bar, Vibration_Level, Current_A, entre outros)
sozinhos tem baixo poder discriminativo para prever falha (Fault) neste equipamento -- a media
de Temperature_C em leituras Fault e muito proxima da media em leituras Normal (diferenca
menor que 1 grau Celsius), o mesmo padrao se repete nas demais features numericas. Isso
significa que o rotulo de falha depende mais do contexto textual (Operator_Notes,
Error_Message) do que de qualquer leitura numerica isolada.

Procedimento de manutencao: nunca decidir se um equipamento esta em falha usando so os
sensores numericos -- sempre cruzar com as notas do operador e mensagens de erro antes de
abrir ou fechar uma ordem de servico.

## 3. Qualidade do Modelo de Deteccao Atual (Isolation Forest)

O modelo de deteccao de anomalia (Isolation Forest, Camada 2 da arquitetura) tem recall baixo
neste equipamento -- ele deixa passar a maioria das falhas reais sem detectar, mesmo marcando
corretamente uma fracao pequena delas. Um recall baixo NAO significa "acerta a maioria" --
significa o oposto: o modelo so identifica uma pequena fracao das falhas de verdade, deixando
a maior parte passar sem alerta.

Procedimento de manutencao: nao tratar a ausencia de alerta do modelo como garantia de que o
equipamento esta saudavel -- o modelo atual erra (nao detecta) a maioria das falhas reais,
entao a inspecao manual periodica continua necessaria mesmo sem alerta automatico.

## 4. Precisao do Modelo e Falso Alarme

Dos casos que o modelo marca como anomalia, uma parcela significativa e falso alarme (nao era
falha real) -- a precisao do modelo e baixa. Isso significa que nem todo alerta automatico
deve disparar uma ordem de servico imediata sem checagem humana previa.

Procedimento de manutencao: todo alerta do Isolation Forest deve passar por uma validacao de
Camada 3 (contexto textual) antes de virar ordem de servico -- reduz o desperdicio de tempo de
manutencao investigando falso alarme.

## 5. Arquitetura de Diagnostico Recomendada (3 camadas)

1. **Camada 1 (regra deterministica):** thresholds fixos por sensor.
2. **Camada 2 (estatistica/ML):** Isolation Forest sobre o conjunto de sensores -- recall
   baixo neste equipamento (secao 3).
3. **Camada 3 (validacao semantica com LLM):** para os casos marcados como anomalos pelas
   camadas 1 e 2, um modelo de linguagem analisa o contexto textual (Operator_Notes,
   Error_Message) e decide se a anomalia e REAL, FALSO_POSITIVO ou INCONCLUSIVO. Dado o baixo
   poder discriminativo dos sensores numericos isolados (secao 2), esta camada tem peso maior
   na decisao final aqui do que teria num equipamento onde os sensores ja separassem bem
   Normal de Fault.

## 6. Escala de Criticidade

- **Critico**: Error_Message presente E qualquer sensor fora da faixa normal -- parada
  imediata recomendada, mesmo que o Isolation Forest nao tenha marcado anomalia (ver
  limitacao de recall, secao 3).
- **Alerta**: Isolation Forest marcou anomalia, mas sem Error_Message correspondente --
  investigar antes de agir, risco de falso alarme (ver secao 4).
- **Informativo**: Operator_Notes com termo qualitativo (ex: "ruido estranho") sem leitura
  numerica anomala correspondente -- registrar, sem acao imediata.
