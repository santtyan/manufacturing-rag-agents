# Manual Tecnico de Manutencao - Paradas de Producao OEE (Dado Tipo B)

## 1. Introducao

Este manual descreve os procedimentos de manutencao e escalonamento de paradas para a linha
de producao monitorada pelo dataset de OEE (Overall Equipment Effectiveness) e Downtime.
Aplica-se a equipamentos com historico de paradas categorizado por StopGroup (categoria),
StopType (planejada/nao planejada) e StopLocation (posto de trabalho onde a parada ocorreu).

## 2. Categorias de Parada (StopGroup)

As paradas sao classificadas em quatro categorias: FAILURE (falha de equipamento), MATERIALS
(falta ou problema de material), OPERATIONAL (parada operacional) e SETUPS-CHANGEOVERS (troca
de configuracao/setup). A categoria FAILURE e a que mais consome minutos no total da linha,
somando as paradas de todos os postos de trabalho -- deve ser tratada como prioridade de
manutencao preventiva.

Procedimento de manutencao: revisar o historico de paradas por StopGroup semanalmente,
priorizando o posto de trabalho com maior soma de minutos dentro da categoria FAILURE.

## 3. Parada Planejada vs. Nao Planejada (StopType)

Paradas Unplanned (nao planejadas) consomem mais minutos no total que paradas Planned
(planejadas) nesta linha -- isso e esperado, ja que falhas/imprevistos tendem a ser mais
dificeis de conter que manutencao programada. O ideal e monitorar se essa proporcao esta
piorando ao longo do tempo: um aumento sustentado de Unplanned em relacao a Planned indica
degradacao da confiabilidade do equipamento.

Procedimento de manutencao: se a proporcao Unplanned/Planned crescer por 3 semanas
consecutivas, abrir investigacao de causa raiz antes que a parada nao planejada vire o padrao
dominante da linha.

## 4. MTTR e MTBF (Indicadores de Confiabilidade)

MTTR (Mean Time To Repair, tempo medio de reparo) e MTBF (Mean Time Between Failures, tempo
medio entre falhas) sao os dois indicadores centrais de confiabilidade da linha. Um MTTR baixo
sozinho nao significa que a linha esta saudavel -- se o MTBF tambem cair (falhas mais
frequentes), a linha esta reparando rapido, mas quebrando mais vezes, o que e um sinal de
alerta, nao de melhoria.

Procedimento de manutencao: sempre reportar MTTR e MTBF juntos, nunca isoladamente. Uma queda
de MTTR acompanhada de queda de MTBF deve ser tratada como piora da confiabilidade geral, nao
como sucesso da manutencao.

## 5. Localizacao da Parada (StopLocation)

O posto de trabalho (StopLocation) que mais acumula minutos de parada no total, somando todas
as categorias, deve ser o foco prioritario de investimento em manutencao preventiva -- troca
de pecas de desgaste, calibracao de sensores, revisao de rotina.

Procedimento de manutencao: cruzar StopLocation com StopGroup antes de decidir uma acao
corretiva -- o mesmo posto pode ter mais de uma causa raiz diferente (ex: falha mecanica em um
periodo, falta de material em outro).

## 6. Arquitetura de Diagnostico Recomendada (3 camadas)

1. **Camada 1 (regra deterministica):** thresholds fixos por StopGroup/StopType (ex: alertar
   se Unplanned > Planned num periodo movel de 4 semanas).
2. **Camada 2 (estatistica/ML):** deteccao de tendencia sobre a serie temporal de MTTR/MTBF,
   capturando degradacao gradual que uma regra fixa isolada nao capturaria.
3. **Camada 3 (validacao semantica com LLM):** para paradas marcadas como atipicas pelas
   camadas 1 e 2, um modelo de linguagem analisa o contexto textual (ExtraText, notas do
   operador) e decide se a parada e uma FALHA RECORRENTE, EVENTO ISOLADO ou DADO
   INCONSISTENTE.

Esta arquitetura em camadas concentra a atencao da equipe de manutencao nos postos de trabalho
com maior probabilidade de impacto real na eficiencia da linha (OEE).

## 7. Escala de Criticidade

- **Critico**: StopGroup = FAILURE E StopLocation com soma de minutos entre as 3 maiores da
  linha -- acao corretiva imediata recomendada.
- **Alerta**: MTBF caindo por 2 periodos consecutivos, mesmo com MTTR estavel ou em queda --
  monitorar e investigar causa raiz.
- **Informativo**: parada Planned dentro do OEE (Planned - Included in OEE) -- registrar, sem
  acao corretiva imediata, e comportamento esperado da operacao.

## 8. Nota sobre Dados Ausentes

Nao existe, nas tabelas deste dataset, nenhuma coluna de custo ou investimento financeiro
associado as paradas -- apenas metricas operacionais (StopDuration, OEE, MTTR, MTBF). Qualquer
calculo de ROI ou retorno financeiro sobre reducao de downtime exigiria um dado de custo que
nao esta disponivel nesta base.
