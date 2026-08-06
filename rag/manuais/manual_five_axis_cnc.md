# Manual Tecnico de Manutencao - Usinagem Five-Axis CNC (Dado Tipo A)

## 1. Introducao

Este manual descreve os procedimentos de manutencao preditiva para a fresadora CNC de 5 eixos
monitorada pelo dataset Five-Axis CNC Milling. O equipamento tem sensores de temperatura em 5
componentes: Spindle (motor da fresa), e os motores dos eixos X, Y, Z, alem de uma leitura
geral (General_temperature).

## 2. Deteccao de Anomalia por Componente

A deteccao de leituras anomalas de temperatura e feita por metodo ESTATISTICO (z-score, desvio
acima da media historica de cada componente) -- nao por um limite absoluto fixo. O eixo Z
concentra a grande maioria das leituras anomalas detectadas entre os 5 componentes monitorados,
seguido pelo Spindle; os eixos X e Y e a leitura geral praticamente nao apresentam leituras
anomalas nesta base.

Procedimento de manutencao: priorizar inspecao do motor do eixo Z antes dos demais
componentes, dado o volume desproporcional de leituras anomalas detectadas nesse eixo.

## 3. Anomalia Estatistica NAO E Violacao de Limite Absoluto

Uma leitura ser marcada como "anomala" pelo metodo estatistico (z-score) NAO significa que ela
violou um limite absoluto de seguranca do fabricante -- sao dois criterios diferentes,
calculados de formas diferentes, e um nao implica o outro. A media historica de temperatura do
eixo Z neste equipamento e bem abaixo de qualquer limite de risco de superaquecimento; uma
leitura "anomala" por z-score, mesmo sendo a maior do periodo, dificilmente se aproxima de um
limite absoluto de seguranca.

Procedimento de manutencao: nunca tratar "N leituras anomalas" como equivalente a "violacao de
limite de temperatura" sem checar o valor real da leitura contra o limite absoluto do
fabricante. Decisao de parar a linha por superaquecimento exige o valor real da leitura, nao
so a contagem de anomalias estatisticas.

## 4. Limitacao de Cruzamento entre Produto e Componente

Nao existe, nas tabelas deste dataset, uma coluna que cruze o ciclo de producao por produto
(Program_path) com a anomalia de temperatura por componente -- sao duas agregacoes
independentes: uma por produto (tempo de ciclo), outra agregada globalmente por componente
(anomalias de temperatura). Nao ha como responder "o produto com ciclo mais longo tambem gera
mais anomalia no Spindle?" com os dados disponiveis atualmente.

Procedimento de manutencao: se essa correlacao for necessaria no futuro, e preciso registrar a
leitura de temperatura POR PRODUTO E POR CICLO, nao so agregada por componente -- mudanca de
instrumentacao, nao de analise.

## 5. Arquitetura de Diagnostico Recomendada (3 camadas)

1. **Camada 1 (regra deterministica):** limite absoluto fixo por componente (ex: temperatura
   do Spindle acima do limite de seguranca do fabricante = parada preventiva).
2. **Camada 2 (estatistica/ML):** deteccao de anomalia por z-score sobre a serie historica de
   cada componente, capturando desvio da media que um limite fixo isolado nao capturaria.
3. **Camada 3 (validacao semantica com LLM):** para leituras marcadas como anomalas pela
   Camada 2, um modelo de linguagem confere se o valor real tambem viola o limite absoluto da
   Camada 1 antes de recomendar parada de linha -- nunca decide so pela contagem de anomalias
   estatisticas (ver secao 3).

## 6. Escala de Criticidade

- **Critico**: leitura de temperatura acima do limite absoluto de seguranca do fabricante,
  independente de ter sido marcada como anomalia estatistica -- parada imediata.
- **Alerta**: leitura marcada como anomalia estatistica (z-score) SEM violar o limite absoluto
  -- monitorar tendencia, nao parar a linha so por esse sinal.
- **Informativo**: variacao de ciclo de producao por produto (Program_path) sem leitura de
  temperatura anomala associada -- registrar, sem acao de manutencao.
