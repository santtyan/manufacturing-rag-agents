"""
Ground truth factual de cada imagem sintetica de rag/manuais_imagens/, usado por:
- eval/checks_fidelidade_caption.py (checks deterministicos de fidelidade do VLM)
- eval/golden_questions_multimodal.json (perguntas derivadas deste ground truth)

Item 1 do plano "Evoluir o RAG multimodal" (2026-09-09). Nao e um sidecar entregue ao VLM
(isso seria trapaca -- ver secao A2/oraculo do plano) -- e o ground truth usado SO para avaliar
se o caption gerado pelo VLM e fiel ao que a imagem de fato mostra. As imagens sao geradas por
rag/gerar_imagens_sinteticas.py; os campos abaixo replicam manualmente titulo/eixos/tipo real de
cada funcao geradora, um a um, na mesma ordem em que aparecem la.

Campos por imagem:
- tipo: "boxplot" | "barra" | "scatter" | "linha_temporal"
- titulo: string exata usada em ax.set_title()
- eixo_x / eixo_y: strings exatas usadas em ax.set_xlabel()/set_ylabel()
- ranking: None, ou lista de categorias na ordem correta (decrescente por padrao) quando o
  grafico tem uma ordem que a legenda deveria capturar
- variaveis_fonte: nomes de coluna do CSV de origem, para o check sem_numeros_inventados
    procurar valores plausveis
"""

GROUND_TRUTH = {
    "grafico_temperatura_por_classe": {
        "tipo": "boxplot", "titulo": "Temperatura do sensor por classe de operação",
        "eixo_x": "Classe", "eixo_y": "Temperatura (°C)", "ranking": None,
        "variaveis_fonte": ["Temperature_C"],
    },
    "grafico_vibracao_por_classe": {
        "tipo": "boxplot", "titulo": "Nível de vibração por classe de operação",
        "eixo_x": "Classe", "eixo_y": "Vibração (nível)", "ranking": None,
        "variaveis_fonte": ["Vibration_Level"],
    },
    "grafico_pressao_vazao": {
        "tipo": "scatter", "titulo": "Pressão vs. vazão por classe de operação",
        "eixo_x": "Pressão (bar)", "eixo_y": "Vazão (L/min)", "ranking": None,
        "variaveis_fonte": ["Pressure_bar", "FlowRate_Lmin"],
    },
    "grafico_anomalias_componentes_cnc": {
        "tipo": "barra", "titulo": "Anomalias de temperatura por componente (CNC 5 eixos)",
        "eixo_x": "Componente", "eixo_y": "Número de leituras anômalas",
        "ranking": "decrescente",  # ordenado desc no gerador
        "variaveis_fonte": ["Spindle_motor_temperature_anomalo", "X_Axis_motor_temperature_anomalo",
                             "Y_Axis_Motor_temperature_anomalo", "Z_Axis_Motor_temperature_anomalo",
                             "General_temperature_anomalo"],
    },
    "grafico_temperatura_barras": {
        "tipo": "barra", "titulo": "Temperatura média do sensor por classe de operação",
        "eixo_x": "Classe", "eixo_y": "Temperatura média (°C)", "ranking": None,
        "variaveis_fonte": ["Temperature_C"],
    },
    "grafico_temperatura_scatter_tempo": {
        "tipo": "scatter", "titulo": "Temperatura do sensor ao longo das leituras, por classe",
        "eixo_x": "Índice da leitura", "eixo_y": "Temperatura (°C)", "ranking": None,
        "variaveis_fonte": ["Temperature_C"],
    },
    "grafico_vibracao_barras": {
        "tipo": "barra", "titulo": "Vibração média por classe de operação",
        "eixo_x": "Classe", "eixo_y": "Vibração média (nível)", "ranking": None,
        "variaveis_fonte": ["Vibration_Level"],
    },
    "grafico_pressao_vazao_eixos_trocados": {
        "tipo": "scatter", "titulo": "Vazão vs. pressão por classe de operação",
        "eixo_x": "Vazão (L/min)", "eixo_y": "Pressão (bar)", "ranking": None,
        "variaveis_fonte": ["Pressure_bar", "FlowRate_Lmin"],
    },
    "grafico_voltagem_corrente_eixos_trocados": {
        "tipo": "scatter", "titulo": "Corrente vs. voltagem por classe de operação",
        "eixo_x": "Voltagem (V)", "eixo_y": "Corrente (A)", "ranking": None,
        "variaveis_fonte": ["Voltage_V", "Current_A"],
    },
    "grafico_corrente_voltagem_eixos_trocados": {
        "tipo": "scatter", "titulo": "Voltagem vs. corrente por classe de operação",
        "eixo_x": "Corrente (A)", "eixo_y": "Voltagem (V)", "ranking": None,
        "variaveis_fonte": ["Voltage_V", "Current_A"],
    },
    "grafico_anomalias_cnc_ascendente": {
        "tipo": "barra",
        "titulo": "Anomalias de temperatura por componente, do menor para o maior (CNC 5 eixos)",
        "eixo_x": "Componente", "eixo_y": "Número de leituras anômalas",
        "ranking": "ascendente",
        "variaveis_fonte": ["Spindle_motor_temperature_anomalo", "X_Axis_motor_temperature_anomalo",
                             "Y_Axis_Motor_temperature_anomalo", "Z_Axis_Motor_temperature_anomalo",
                             "General_temperature_anomalo"],
    },
    "grafico_producao_por_classe_ordenado": {
        "tipo": "barra", "titulo": "Taxa de produção média por classe, do maior para o menor",
        "eixo_x": "Classe", "eixo_y": "Taxa de produção média",
        "ranking": "decrescente", "variaveis_fonte": ["Production_Rate"],
    },
    "grafico_voltagem_por_classe": {
        "tipo": "boxplot", "titulo": "Voltagem por classe de operação",
        "eixo_x": "Classe", "eixo_y": "Voltagem (V)", "ranking": None,
        "variaveis_fonte": ["Voltage_V"],
    },
    "grafico_corrente_por_classe": {
        "tipo": "boxplot", "titulo": "Corrente por classe de operação",
        "eixo_x": "Classe", "eixo_y": "Corrente (A)", "ranking": None,
        "variaveis_fonte": ["Current_A"],
    },
    "grafico_som_por_classe": {
        "tipo": "boxplot", "titulo": "Nível de som por classe de operação",
        "eixo_x": "Classe", "eixo_y": "Som (dB)", "ranking": None,
        "variaveis_fonte": ["Sound_dB"],
    },
    "grafico_umidade_por_classe": {
        "tipo": "boxplot", "titulo": "Umidade por classe de operação",
        "eixo_x": "Classe", "eixo_y": "Umidade (%)", "ranking": None,
        "variaveis_fonte": ["Humidity_%"],
    },
    "grafico_qualidade_oleo_por_classe": {
        "tipo": "boxplot", "titulo": "Índice de qualidade do óleo por classe de operação",
        "eixo_x": "Classe", "eixo_y": "Índice de qualidade do óleo", "ranking": None,
        "variaveis_fonte": ["Oil_Quality_Index"],
    },
    "grafico_consumo_energia_por_classe": {
        "tipo": "boxplot", "titulo": "Consumo de energia por classe de operação",
        "eixo_x": "Classe", "eixo_y": "Consumo de energia (kWh)", "ranking": None,
        "variaveis_fonte": ["Energy_Consumption_kWh"],
    },
    "grafico_carga_por_classe": {
        "tipo": "boxplot", "titulo": "Percentual de carga por classe de operação",
        "eixo_x": "Classe", "eixo_y": "Carga (%)", "ranking": None,
        "variaveis_fonte": ["Load_Percentage"],
    },
    "grafico_energia_vs_carga": {
        "tipo": "scatter", "titulo": "Consumo de energia vs. carga por classe de operação",
        "eixo_x": "Carga (%)", "eixo_y": "Consumo de energia (kWh)", "ranking": None,
        "variaveis_fonte": ["Energy_Consumption_kWh", "Load_Percentage"],
    },
    "grafico_som_vs_vibracao": {
        "tipo": "scatter", "titulo": "Som vs. vibração por classe de operação",
        "eixo_x": "Vibração (nível)", "eixo_y": "Som (dB)", "ranking": None,
        "variaveis_fonte": ["Sound_dB", "Vibration_Level"],
    },
    "grafico_anomalia_spindle_tempo": {
        "tipo": "linha_temporal",
        "titulo": "Anomalia de temperatura ao longo do tempo — Spindle (CNC 5 eixos)",
        "eixo_x": "Tempo (índice da leitura)", "eixo_y": "Anomalia (0=normal, 1=anômalo)",
        "ranking": None, "variaveis_fonte": ["Spindle_motor_temperature_anomalo"],
    },
    "grafico_anomalia_eixo_x_tempo": {
        "tipo": "linha_temporal",
        "titulo": "Anomalia de temperatura ao longo do tempo — Eixo X (CNC 5 eixos)",
        "eixo_x": "Tempo (índice da leitura)", "eixo_y": "Anomalia (0=normal, 1=anômalo)",
        "ranking": None, "variaveis_fonte": ["X_Axis_motor_temperature_anomalo"],
    },
    "grafico_anomalia_eixo_y_tempo": {
        "tipo": "linha_temporal",
        "titulo": "Anomalia de temperatura ao longo do tempo — Eixo Y (CNC 5 eixos)",
        "eixo_x": "Tempo (índice da leitura)", "eixo_y": "Anomalia (0=normal, 1=anômalo)",
        "ranking": None, "variaveis_fonte": ["Y_Axis_Motor_temperature_anomalo"],
    },
    "grafico_anomalia_eixo_z_tempo": {
        "tipo": "linha_temporal",
        "titulo": "Anomalia de temperatura ao longo do tempo — Eixo Z (CNC 5 eixos)",
        "eixo_x": "Tempo (índice da leitura)", "eixo_y": "Anomalia (0=normal, 1=anômalo)",
        "ranking": None, "variaveis_fonte": ["Z_Axis_Motor_temperature_anomalo"],
    },
    "grafico_anomalia_geral_tempo": {
        "tipo": "linha_temporal",
        "titulo": "Anomalia de temperatura ao longo do tempo — Geral (CNC 5 eixos)",
        "eixo_x": "Tempo (índice da leitura)", "eixo_y": "Anomalia (0=normal, 1=anômalo)",
        "ranking": None, "variaveis_fonte": ["General_temperature_anomalo"],
    },
}
