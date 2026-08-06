# Legacy Industrial Equipment Sensor Logs (Kaggle)

## Descricao

Dados de sensores de equipamentos industriais antigos, incluindo notas do operador, mensagens de
erro e uma coluna de status (Normal ou Fault). Util para analises de falhas.

## Colunas

- Timestamp: data e hora da leitura
- Machine_ID: identificador da maquina
- Temperature_C: temperatura em graus Celsius
- Pressure_bar: pressao em bar
- Vibration_Level: nivel de vibracao (escala normalizada)
- Voltage_V: tensao em volts
- Current_A: corrente eletrica em amperes
- Sound_dB: nivel de som em decibeis
- FlowRate_Lmin: taxa de fluxo em litros por minuto
- Humidity_%: umidade relativa
- Oil_Quality_Index: indice de qualidade do oleo
- Energy_Consumption_kWh: consumo de energia
- Production_Rate: taxa de producao
- Load_Percentage: percentual de carga da maquina
- Operator_Notes: notas textuais do operador
- Error_Message: mensagem de erro registrada pela maquina
- Target: rotulo real do estado da maquina (Normal ou Fault) -- usado como ground truth

## Achado da analise (pipeline 2)

A media das 12 features numericas por classe (Normal vs. Fault) e quase identica -- por exemplo,
Temperature_C tem media de 69.4C em Fault contra 70.1C em Normal, diferenca menor que 1 grau. Isso
indica que, neste dataset especificamente, os sensores numericos sozinhos tem baixo poder
discriminativo para prever falha -- o rotulo Fault provavelmente depende mais do contexto textual
(Operator_Notes, Error_Message) do que dos valores numericos brutos.

## Origem e Escopo do Dataset

Fonte: Kaggle, "Legacy Industrial Equipment Sensor Logs" (autor: Colabsss). Licenca: CC0
(dominio publico). O dataset contem 2.500 registros e 17 colunas, coletados ao longo do ano
de 2024 (medicoes operacionais e ambientais de equipamentos industriais legados).
