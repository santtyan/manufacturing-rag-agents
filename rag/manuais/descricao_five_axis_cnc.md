# Five-Axis CNC Milling Dataset (Zenodo)

## Descricao

Dados de um processo de fresagem CNC de 5 eixos. Tres produtos diferentes foram fabricados, com
matriz de changeover garantindo todas as combinacoes possiveis. Producao repetida 5 vezes (30
sessoes de manufatura). Dados registrados a partir de um controle Siemens 840D-SL em uma fresadora
de 5 eixos "Spinner U5-620".

## Colunas relevantes usadas na analise

- time: timestamp da leitura
- Program_path: identificador do programa/produto em execucao
- Program_status: status numerico do programa (valores observados: 1, 2, 3, 5)
- Cycle_time_program: tempo de ciclo do programa em execucao
- Spindle_motor_temperature: temperatura do motor do eixo-arvore (spindle)
- X_Axis_motor_temperature, Y_Axis_Motor_temperature, Z_Axis_Motor_temperature: temperatura dos
  motores de cada eixo linear
- General_temperature: temperatura geral do equipamento

## Achado da analise (pipeline 4)

Deteccao de anomalia de temperatura via z-score (limiar > 3 desvios-padrao) identificou 1.907
leituras anomalas no total. O eixo Z concentrou a maior parte dessas anomalias (1.634 leituras),
indicando maior variancia termica nesse motor especifico comparado aos demais eixos monitorados
(Spindle, X, Y).

## Origem, Proveniencia e Material Suplementar

Fonte: Zenodo, "Production Data Set for Five-Axis CNC Milling with Multiple Changeovers"
(DOI: 10.5281/zenodo.15735480, versao 1.0.1, publicado por Technische Hochschule
Wuerzburg-Schweinfurt - THWS). Autores: Bastian Engelmann, Anna-Maria Schmitt, Mario
Martinez. Licenca: CC BY 4.0. Arquivo unico `data_v1_0_1.csv`, 78.7 MB.

Ha um data paper publicado descrevendo o dataset em detalhe (Scientific Data/Nature, DOI:
10.1038/s41597-025-05294-0). O material suplementar inclui NC-code dos 3 produtos,
informacao de ferramentas e um Jupyter notebook de exemplo, disponiveis no GitHub:
github.com/ElMoe/Production-Data-Set-for-Five-Axis-CNC-Milling-with-Multiple-Changeovers.

ERRATUM IMPORTANTE (correcao da v1.0.0 para v1.0.1): a coluna que antes se chamava
`smoothed_DC_voltage_Drive4` foi renomeada para `Tool_number_Magazine_Place_49` -- o nome
antigo estava incorreto. Ao consultar ou analisar esse dataset, usar sempre o nome
corrigido (`Tool_number_Magazine_Place_49`).
