# Manual Tecnico de Manutencao - Sensores Industriais (Dado Tipo A)

## 1. Introducao

Este manual descreve os procedimentos de manutencao preventiva e diagnostico de falhas para
equipamentos industriais monitorados por sensores de temperatura, pressao, vibracao, corrente
eletrica e fluxo. Aplica-se a maquinas com sistema de monitoramento continuo (Tipo C) e historico
de paradas (Tipo B).

## 2. Sensor de Temperatura (Temperature_C)

A temperatura normal de operacao dos motores varia entre 60 C e 75 C dependendo da carga.
Leituras acima de 85 C por mais de 10 minutos indicam risco de superaquecimento e devem
disparar parada preventiva. Causas comuns de superaquecimento: falha no sistema de refrigeracao,
sobrecarga do motor, ou obstrucao de fluxo de ar.

Procedimento de manutencao: verificar ventiladores de refrigeracao, limpar filtros de ar,
conferir se a carga esta dentro da especificacao da placa do motor.

## 3. Sensor de Pressao (Pressure_bar)

A faixa operacional segura e de 2.5 a 4.5 bar. Quedas abruptas de pressao geralmente indicam
vazamento em juntas ou mangueiras. Picos de pressao acima de 5 bar podem indicar obstrucao
a jusante do sistema hidraulico/pneumatico.

Procedimento de manutencao: inspecionar juntas e conexoes, verificar valvulas de alivio,
testar sensores de pressao com manometro calibrado.

## 4. Sensor de Vibracao (Vibration_Level)

Niveis de vibracao acima de 1.5 (escala normalizada) sao considerados anomalos e geralmente
precedem falha mecanica (desalinhamento de eixo, desgaste de rolamento, desbalanceamento).
Vibracao excessiva e um dos indicadores mais confiaveis de falha iminente em equipamentos
rotativos, segundo a literatura de manutencao preditiva.

Procedimento de manutencao: realizar analise espectral de vibracao, verificar alinhamento
de acoplamentos, inspecionar rolamentos.

## 5. Corrente Eletrica e Consumo de Energia

Aumento anormal de corrente (Current_A) sem aumento correspondente de producao (Production_Rate)
indica ineficiencia mecanica ou eletrica -- possivel atrito excessivo, desgaste de componentes
ou problema no motor eletrico. Deve ser correlacionado com Energy_Consumption_kWh para confirmar.

## 6. Interpretacao de Notas do Operador (Operator_Notes)

Notas como "Flow irregular", "Temperature high" e "Pressure fluctuation" sao sinais qualitativos
que devem ser cruzados com os valores quantitativos dos sensores antes de abrir uma ordem de
servico. Um unico sinal isolado (ex: nota textual sem leitura anomala correspondente) tem alta
chance de ser falso positivo.

## 7. Arquitetura de Diagnostico Recomendada (3 camadas)

1. **Camada 1 (regra deterministica):** thresholds fixos por sensor (ex: Temperature_C > 85).
2. **Camada 2 (estatistica/ML):** deteccao de anomalia multivariada (Isolation Forest) sobre
   o conjunto de sensores, capturando combinacoes que nenhuma regra isolada capturaria.
3. **Camada 3 (validacao semantica com LLM):** para os casos marcados como anomalos pelas
   camadas 1 e 2, um modelo de linguagem analisa o contexto textual (notas do operador,
   mensagens de erro) e decide se a anomalia e REAL, FALSO_POSITIVO ou INCONCLUSIVO.

Esta arquitetura em camadas reduz falsos positivos e concentra a atencao da equipe de manutencao
nos casos com maior probabilidade de falha real.

## 8. Escala de Criticidade

- **Critico**: Temperature_C > 90 OU Vibration_Level > 2.0 -- parada imediata recomendada.
- **Alerta**: Anomalia detectada por Isolation Forest sem violacao de threshold fixo -- monitorar.
- **Informativo**: Nota do operador sem leitura anomala correspondente -- registrar, sem acao imediata.
