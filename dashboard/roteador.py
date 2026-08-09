"""
Roteador do chat: decide se uma pergunta vai para contexto/RAG/SQL, ou para um dos gates
deterministicos (answerability gates + calculos pre-processados em Python).

EXTRAIDO de dashboard/app.py (2026-08-07): antes essas funcoes viviam dentro de app.py, que
tem chamadas Streamlit em nivel de modulo (st.set_page_config etc) -- por isso NUNCA foi
possivel importar o roteamento direto num script. eval/rodar_golden.py reimplementava esse
bloco inteiro como "copia fiel" (comentarios do arquivo admitiam isso explicitamente),
sujeita a desatualizar sempre que um gate novo fosse adicionado aqui -- e desatualizou (rerun
do harness deu 52% em vez de 67,9% por essa causa, nao por regressao real). Mesmo padrao ja
resolvido para o RAG (rag_gerador.py, modulo compartilhado entre app.py e o harness).

dashboard/app.py e eval/rodar_golden.py agora importam deste modulo -- fonte unica de
verdade para o roteamento. Nenhuma logica mudou nesta extracao, so o local onde vive.

OLLAMA_URL/OLLAMA_MODEL sao parametros de rotear_por_llm()/rotear_pergunta() (nao constantes
de modulo fixas) para que cada chamador (dashboard real vs. harness) possa passar sua propria
config sem precisar duplicar a funcao -- mesmo principio de parametrizacao usado em
rag/rag_hibrido.py (chroma_dir/colecao) e nl_to_sql/nl_to_sql.py (gerar_sql_com_schema).
"""
import json

import requests

PALAVRAS_CHAVE_RAG = [
    "manual", "manutencao", "manutenção", "procedimento", "threshold", "limiar",
    "arquitetura", "camada", "temperatura maxima", "temperatura máxima", "seguranca", "segurança",
    "rag", "retrieval",
]
PALAVRAS_CHAVE_SQL = [
    "select", "tabela", "banco de dados", "quantos registros", "quantas linhas",
    "sql", "consulta no banco", "query",
]
# Pedidos de agregacao com filtro/recorte especifico (ex: "media de temperatura NESSE PERIODO",
# "media da maquina X") devem ir para SQL -- o Postgres calcula AVG()/SUM() de verdade sobre os
# dados filtrados, em vez do LLM "estimar" a partir de uma amostra pequena do prompt (risco real:
# LLM faz conta errada ou generaliza de poucos exemplos). So dispara quando ha sinal de recorte
# (periodo, maquina, data, faixa) -- perguntas sobre METRICA JA CALCULADA E PRONTA no pipeline
# (ex: "qual o recall do modelo?", "qual o OEE?") continuam indo para contexto, que ja tem o
# numero certo sem precisar calcular nada.
PALAVRAS_CHAVE_AGREGACAO_COM_FILTRO = [
    "nesse periodo", "nesse período", "no periodo", "no período", "nessa faixa",
    "dessa maquina", "dessa máquina", "desse periodo", "desse período",
]
# Pedidos de "qual categoria mais/menos X" exigem GROUP BY + ORDER BY corretos para comparar
# categorias entre si -- achado real (2026-07-12): o LLM confundiu a maior LINHA individual
# (StopGroup+StopLocation) com a maior CATEGORIA (StopGroup agregado), porque o contexto so
# tinha as linhas cruas. Ranking por categoria sempre vai para SQL, que faz o GROUP BY de
# verdade, em vez de contexto/LLM estimando a partir de uma amostra de linhas.
PALAVRAS_CHAVE_RANKING_CATEGORIA = [
    "qual categoria", "qual grupo", "qual tipo", "que categoria", "que grupo", "que tipo",
    "mais consumiu", "menos consumiu", "mais gerou", "menos gerou", "mais teve", "menos teve",
    "mais comum", "menos comum", "no total",
    # Comparacao "X vs Y" entre duas categorias/grupos (achado real, 2026-07-12: "as paradas
    # planejadas consomem mais ou menos minutos que as nao planejadas" nao batia com nenhuma
    # palavra-chave acima, caiu em 'contexto' por default e o LLM alucinou uma resposta
    # "nao e possivel determinar" mesmo com o dado certo ja anotado no contexto).
    "consomem mais", "consomem menos", "consome mais", "consome menos",
    "mais ou menos", "comparado a", "comparado com", "em relacao a", "em relação a",
]
# Perguntas que cruzam "categoria/grupo/tipo de parada" com "antes/depois do LSS" pedem um dado
# que NAO EXISTE no schema: oee_agregacao_paradas (por StopGroup) e oee_comparacao_lss (por
# metrica OEE, sem quebra por StopGroup) nao tem chave em comum. Achado real (2026-07-12): tanto
# o LLM (SQL, inventando JOIN/colunas) quanto o fallback de contexto (mesmo com instrucao
# explicita no prompt) alucinaram uma atribuicao "antes"/"depois" para StopGroup que nao existe.
# Em vez de confiar no LLM se auto-policiar, detecta essa combinacao e responde de forma
# deterministica, combinando os dois dados reais SEM fingir que estao cruzados.
PALAVRAS_CHAVE_PERIODO_LSS = [
    "antes", "depois", "lean six sigma", "lss", "mudou apos", "mudou depois",
]
# Metricas do comparativo LSS (oee_comparacao_lss: MTTR, MTBF, OEE, Availability, Performance,
# Quality) NAO tem quebra por StopGroup no banco -- qualquer pergunta que junte uma dessas
# metricas com "categoria"/"tipo"/"grupo" de parada cai no mesmo cruzamento impossivel do
# Ataque LSS, mesmo sem usar as palavras "antes"/"depois" (achado real, 2026-07-12: "a queda
# no MTTR se correlaciona com alguma categoria de parada" driblou o gate original porque nao
# tem "antes/depois" explicito -- SQL rodou uma query que nao responde a pergunta e o LLM
# inventou uma "correlacao" que os dados nao suportam).
PALAVRAS_CHAVE_METRICA_LSS = ["mttr", "mtbf", "oee", "availability", "disponibilidade"]
PALAVRAS_CHAVE_CATEGORIA_PARADA = ["categoria", "tipo de parada", "grupo de parada", "stopgroup"]


def pede_agregacao_com_filtro(pergunta):
    """Deteta pedido de calculo (media/soma/total/maximo/minimo) combinado com recorte
    especifico (periodo/maquina/faixa) que exige consultar o banco, nao estimar do contexto."""
    p = pergunta.lower()
    tem_verbo_agregacao = any(v in p for v in ("media", "média", "soma", "total", "maximo", "máximo", "minimo", "mínimo"))
    tem_recorte = any(kw in p for kw in PALAVRAS_CHAVE_AGREGACAO_COM_FILTRO)
    return tem_verbo_agregacao and tem_recorte


def pede_ranking_categoria(pergunta):
    """Deteta pedido de ranking/comparacao entre categorias (ex: 'qual categoria mais
    consumiu minutos') -- exige GROUP BY correto, risco de o LLM confundir linha com categoria."""
    p = pergunta.lower()
    tem_superlativo = any(v in p for v in ("mais", "menos", "maior", "menor", "maximo", "máximo", "minimo", "mínimo", "top"))
    tem_categoria = any(kw in p for kw in PALAVRAS_CHAVE_RANKING_CATEGORIA)
    return tem_superlativo and tem_categoria


def pede_cruzamento_categoria_x_periodo_lss(pergunta):
    """Deteta pedido de cruzamento StopGroup x periodo LSS -- combinacao impossivel no schema
    atual (sem chave de juncao). Ver comentario acima (PALAVRAS_CHAVE_PERIODO_LSS) para o
    achado que motivou isso. Duas formas de disparar: (1) ranking de categoria + palavra de
    periodo LSS explicita ("antes"/"depois"), ou (2) qualquer metrica do comparativo LSS
    (MTTR/MTBF/OEE/Availability) junto com mencao a categoria/tipo/grupo de parada -- cobre
    parafrases como "correlaciona-se" que nao usam "antes/depois" mas pedem o mesmo cruzamento
    impossivel."""
    p = pergunta.lower()
    forma_1 = pede_ranking_categoria(pergunta) and any(kw in p for kw in PALAVRAS_CHAVE_PERIODO_LSS)
    forma_2 = (any(kw in p for kw in PALAVRAS_CHAVE_METRICA_LSS)
               and any(kw in p for kw in PALAVRAS_CHAVE_CATEGORIA_PARADA))
    return forma_1 or forma_2


PALAVRAS_CHAVE_INVESTIMENTO = ("investimos", "investimento", "roi", "retorno sobre", "payback", "se paga")


def pede_roi_dado_inexistente(pergunta):
    """Deteta pedido de ROI/payback financeiro sobre o Lean Six Sigma -- nao existe coluna de
    custo/investimento em NENHUMA tabela do dataset OEE (so metricas operacionais: OEE, MTTR,
    MTBF, StopDuration). Achado real, 2026-07-14: SEM esse gate, o NL-to-SQL gerou SQL com uma
    constante monetaria INVENTADA (fator "8.33" por minuto de parada, sem origem em nenhum dado
    real) para fabricar um numero de ROI plausivel (R$ 114.495, 46 meses de payback). E
    exatamente o cenario de maior risco de negocio catalogado (Ataque 4 do dataset 1): LLM
    'confirmando' um investimento financeiro com numero fabricado. Intercepta ANTES de gerar
    SQL -- nao ha como responder isso com os dados disponiveis, tem que admitir a limitacao."""
    p = pergunta.lower()
    return any(kw in p for kw in PALAVRAS_CHAVE_INVESTIMENTO) and ("lean six sigma" in p or "lss" in p)


def pede_matriz_confusao_precalculada(pergunta):
    """Deteta pergunta pedindo para derivar precision/recall a partir da matriz de confusao
    (tp/fp/tn/fn) -- ja pre-calculado em precision_derivada_txt (dataset 2). Achado real,
    2026-07-14: sem esse gate, a pergunta ia para SQL, que gerava consulta com erro de tipo
    (JOIN sensor_predicoes com "if_anomaly" booleano comparado a inteiro) -- o fallback de
    contexto ja pegava o numero certo (62) depois do erro, mas mostrava um erro de SQL feio na
    tela antes disso, sem necessidade. Forca contexto direto, evitando o erro visivel."""
    p = pergunta.lower()
    return "matriz de confus" in p and ("marcou como anomalia" in p or "realmente" in p)


def pede_downtime_dataset_errado(pergunta):
    """Deteta pergunta sobre 'minutos de downtime' no contexto do dataset 2 (Legacy Sensor
    Logs) -- essa coluna NAO existe la, so existe StopDuration(min) no dataset 1 (OEE). Achado
    real, 2026-07-14: sem esse gate, o roteamento por keyword ja acerta 'contexto' (sem sinal
    forte de sql), mas o desempate por LLM (rotear_por_llm) reclassificava errado para 'sql' --
    o NL-to-SQL entao consultava oee_downtime_raw (dataset ERRADO) filtrando por um valor de
    ExtraText que nao existe ('falha nao detectada'), sem dar erro (retornou NULL/None, lido
    como resposta valida 'nenhum downtime'). Mesmo padrao do bug de roteamento ja corrigido no
    dataset 3 (pede_confirmacao_alarme_automatico) -- intercepta ANTES do desempate por LLM."""
    p = pergunta.lower()
    tem_downtime = "downtime" in p or "tempo de parada" in p
    tem_falha_nao_detectada = "nao detectad" in p or "não detectad" in p
    return tem_downtime and tem_falha_nao_detectada


PALAVRAS_CHAVE_CICLO_PRODUTO = ("ciclo", "produto", "program_path", "programa")
PALAVRAS_CHAVE_ANOMALIA_COMPONENTE = ("anomalia", "anomalo", "spindle", "eixo", "motor", "temperatura")


def pede_cruzamento_ciclo_x_anomalia_cnc(pergunta):
    """Deteta pedido de cruzar ciclo de producao por produto (cnc_ciclo_por_produto, chave
    Program_path) com anomalia de temperatura por componente (cnc_resumo_anomalias_por_componente,
    chave componente/eixo) -- combinacao impossivel no schema atual: as duas tabelas nao
    compartilham chave de juncao (uma e por produto, outra e agregada globalmente por eixo do
    motor, sem granularidade de Program_path). Mesma classe de bug de
    pede_cruzamento_categoria_x_periodo_lss (achado real, 2026-07-14): o LLM alucinava um JOIN
    inexistente entre cnc_ciclo_por_produto e uma tabela por eixo que nao existia no banco."""
    p = pergunta.lower()
    return (any(kw in p for kw in PALAVRAS_CHAVE_CICLO_PRODUTO)
            and any(kw in p for kw in PALAVRAS_CHAVE_ANOMALIA_COMPONENTE)
            and ("correla" in p or "junto" in p or ("e tambem" in p) or ("e o que" in p)))


def pede_planned_vs_unplanned(pergunta):
    """Deteta comparacao Planned vs Unplanned (valores da coluna StopType) -- achado real,
    2026-07-12: o LLM gera SQL sintaticamente valido mas agrupa pela coluna ERRADA (StopGroup
    em vez de StopType), retornando uma tabela que responde outra pergunta. Aconteceu tanto no
    llama3.2 quanto no qwen2.5:7b, mesmo com regra explicita de GROUP BY no prompt -- intercepta
    e usa os valores pre-calculados (dashboard/app.py, total_planned_min/total_unplanned_min)
    em vez de arriscar outro SQL malformado."""
    p = pergunta.lower()
    tem_planned = "planned" in p or "planejada" in p
    tem_unplanned = "unplanned" in p or "nao planejada" in p or "não planejada" in p
    return tem_planned and tem_unplanned


def pede_lss_melhorou_tudo(pergunta):
    """Deteta pergunta 'o LSS melhorou tudo, ou piorou algo' -- achado real, 2026-07-12: mesmo
    com cada linha do comparativo ja rotulada [melhorou]/[PIOROU] no contexto (calculado pelo
    sinal de variacao_%), o LLM (llama3.2 E qwen2.5:7b) ignorava o rotulo pronto e reinterpretava
    os numeros brutos sozinho, invertendo a direcao de quase todas as metricas (chamou
    Availability +8.3% de 'reduzida', Performance +2.2% de 'diminuiu', Quality -1.3% de
    'aumento de 1.3%'). Anotar nao bastou -- intercepta e responde 100% em Python."""
    p = pergunta.lower()
    tem_lss = "lean six sigma" in p or "lss" in p
    tem_pergunta_direcao = any(kw in p for kw in (
        "melhorou tudo", "piorou", "todas as metricas", "alguma que", "teve alguma",
    ))
    return tem_lss and tem_pergunta_direcao


def pede_interpretacao_recall(pergunta):
    """Deteta pergunta pedindo para interpretar a DIRECAO do recall/precision (ex: 'isso
    significa que o modelo acerta quase 10 em cada 10, ou erra quase 10 em cada 10?') --
    achado real, 2026-07-15: mesmo com a metrica ja anotada no contexto com a conversao
    fracao->percentual pronta ("0.097 (equivale a 9.7%)", ver metrics_anotado na aba do
    dataset 2), o LLM lia o percentual CERTO mas invertia a interpretacao, concluindo
    "acerta quase 10 em cada 10" quando recall baixo significa o OPOSTO (detecta poucas
    falhas reais, erra a maioria). Mesma classe do bug #4 do catalogo (LSS direcao
    invertida) -- anotar o dado nao bastou, so resolveu quando a resposta virou 100%
    determinística. Intercepta ANTES do LLM decidir a direcao sozinho."""
    p = pergunta.lower()
    tem_recall_ou_precision = "recall" in p or "precision" in p or "precisao" in p
    tem_pergunta_direcao = any(kw in p for kw in (
        "acerta", "erra", "significa que", "na pratica", "na prática",
    ))
    return tem_recall_ou_precision and tem_pergunta_direcao


def pede_confirmacao_alarme_automatico(pergunta):
    """Deteta pergunta com premissa numerica sobre alarme/automatico no dataset de manufatura
    discreta (ex: '79 mil registros em alarme contra 24 mil em automatico') pedindo 'confirme
    esses numeros' -- achado real, 2026-07-14: essa pergunta tem sinal fraco de keyword
    ('registros', 'confirme') que cai em 'contexto' por default, mas o desempate por LLM
    (rotear_por_llm) reclassifica errado para 'sql' so por causa da palavra 'confirme', mesmo
    a resposta ja estando pronta e rotulada no contexto (estados_antes_do_alarme). Isso levou
    o NL-to-SQL a consultar a tabela ERRADA (sensor_predicoes, outro dataset) com numeros
    completamente diferentes (250 vs 2250 em vez de 79475 vs 23886). Intercepta ANTES do
    roteador LLM ter chance de reclassificar -- forca 'contexto', onde o dado certo ja esta."""
    p = pergunta.lower()
    tem_alarme = "alarme" in p or "alarm" in p
    tem_automatico = "automatico" in p or "automático" in p
    tem_pedido_confirmacao = "confirme" in p or "confirmar" in p or "esses numeros" in p or "esses números" in p
    return tem_alarme and tem_automatico and tem_pedido_confirmacao


PALAVRAS_CHAVE_RANKING_ASSETS = ("asset", "assets", "regime de energia", "mudancas de regime", "mudanças de regime")


def pede_ranking_assets_duas_empresas(pergunta):
    """Deteta pedido de ranking de assets por mudanca de regime (manufacturing_regime_a/_b) --
    achado real, 2026-07-14: sem esse gate, o NL-to-SQL respondia com UNION ALL entre
    manufacturing_regime_a e manufacturing_regime_b SEM coluna identificando a empresa de
    origem, produzindo um "ranking unico" comparando asset_id de company_A com asset_id de
    company_B como se fossem a mesma numeracao -- nao existe "3o lugar geral" entre duas
    fabricas com numeracao propria de asset, apesar da regra explicita no schema dizendo para
    nunca misturar as duas empresas. Intercepta ANTES do SQL, responde em Python com as duas
    listas separadas e rotuladas."""
    p = pergunta.lower()
    tem_ranking = any(v in p for v in ("ranque", "ranking", "ordene", "classifique", "mais instave", "mais estave"))
    tem_assets = any(kw in p for kw in PALAVRAS_CHAVE_RANKING_ASSETS)
    return tem_ranking and tem_assets


def rotear_por_keyword(pergunta):
    """Roteamento por palavra-chave: rapido, 0 latencia, mas fragil a sinonimos.
    'nao_respondivel' tem prioridade maxima: e um "answerability gate" deterministico (padrao-
    ouro para unanswerable questions em text-to-SQL -- ver LatentRefusal/arXiv:2601.10398 e
    "Query Carefully"/arXiv:2512.21345, 2026) que intercepta ANTES de gerar SQL ou chamar o LLM.
    Motivo: tanto o SQL quanto o fallback de contexto, mesmo com instrucao explicita no prompt,
    alucinaram um cruzamento StopGroup x periodo LSS que nao existe no schema (achado real,
    2026-07-12) -- a literatura confirma que nem modelos grandes (70B) resolvem isso so por
    prompt (~80% acuracia mesmo com few-shot), entao a checagem tem que ser no codigo, nao no LLM."""
    if pede_cruzamento_categoria_x_periodo_lss(pergunta):
        return "nao_respondivel_lss"
    if pede_roi_dado_inexistente(pergunta):
        return "nao_respondivel_roi"
    if pede_cruzamento_ciclo_x_anomalia_cnc(pergunta):
        return "nao_respondivel_cnc"
    if pede_planned_vs_unplanned(pergunta):
        return "planned_vs_unplanned"
    if pede_lss_melhorou_tudo(pergunta):
        return "lss_melhorou_tudo"
    p = pergunta.lower()
    if any(kw in p for kw in PALAVRAS_CHAVE_SQL) or pede_agregacao_com_filtro(pergunta) or pede_ranking_categoria(pergunta):
        return "sql"
    # Excecao a PALAVRAS_CHAVE_RAG: "camada" tambem aparece em perguntas sobre a Camada 3 (LLM)
    # do pipeline de deteccao de anomalia -- um DADO calculado (vereditos_llm.csv), nao uma
    # pergunta sobre a arquitetura descrita no manual. Achado real, 2026-07-14: "Na amostra de
    # vereditos do LLM (Camada 3), quantos concordaram com o Target?" roteou errado para RAG
    # (que nao tem o dado, so a descricao textual da arquitetura), quando a resposta pronta ja
    # estava calculada em concordancia_txt (rota "contexto"). Sinais fortes de pergunta sobre
    # dado calculado (amostra/veredito/concordou) tem prioridade sobre a keyword generica.
    sinal_dado_calculado = any(kw in p for kw in ("veredito", "amostra", "concordou", "concordaram"))
    if any(kw in p for kw in PALAVRAS_CHAVE_RAG) and not sinal_dado_calculado:
        return "rag"
    return "contexto"


def rotear_por_llm(pergunta, ollama_url, ollama_model):
    """Classificacao de intencao via LLM com saida estruturada (Parte 4 do plano padrao-ouro).
    Usa o `format` JSON-schema do Ollama para forcar rota valida (Literal contexto/rag/sql) --
    mais robusto a sinonimos que o keyword. Retorna None se o LLM falhar (cai no keyword).

    ollama_url/ollama_model sao parametros (nao constantes fixas) -- cada chamador (dashboard
    real ou harness de avaliacao) passa a propria config, sem precisar duplicar esta funcao."""
    prompt = f"""Classifique a intencao da pergunta em UMA categoria. Regra de ouro: se a pergunta
pode ser respondida com METRICAS/NUMEROS/COMPARACOES que um pipeline de dados ja calculou
(percentuais, medias, contagens, ranking de categorias, antes/depois), a categoria e "contexto" --
mesmo que a pergunta nao diga explicitamente "dados calculados". So use "rag" se a pergunta pedir
uma REGRA, PROCEDIMENTO ou LIMITE ESCRITO EM TEXTO (ex: "qual a temperatura maxima segura?",
"qual o procedimento de manutencao?").

- contexto: metricas/resultados/comparacoes JA CALCULADOS E PRONTOS pelo pipeline, que NAO exigem
  comparar/ranquear categorias entre si. Exemplos: "o OEE antes e depois do Lean Six Sigma", "qual
  o recall do modelo?", "o MTTR caiu, isso e bom ou ruim?"
- rag: regras/procedimentos/limites escritos no MANUAL TECNICO. Exemplos: "qual a temperatura
  maxima de operacao?", "qual o procedimento de manutencao preventiva?", "como o sistema decide
  se e falha real?"
- sql: pede um CALCULO NOVO (media/soma/total/maximo/minimo) sobre um RECORTE especifico
  (periodo, maquina, faixa de datas), OU pede RANKING/COMPARACAO entre categorias (qual categoria/
  grupo/tipo mais ou menos algo) -- ambos exigem GROUP BY/agregacao exata no banco, que o LLM erra
  se tentar estimar de uma amostra de linhas. Exemplos: "qual a media de temperatura nesse
  periodo?", "qual categoria de parada mais consumiu minutos?", "qual produto tem o maior tempo
  de ciclo?", "quantos registros tem a tabela X?", "mostre a tabela Y"

Pergunta: {pergunta}

Responda em JSON com a chave "rota"."""
    try:
        resp = requests.post(
            ollama_url,
            json={
                "model": ollama_model, "prompt": prompt, "stream": False,
                "format": {"type": "object",
                           "properties": {"rota": {"type": "string", "enum": ["contexto", "rag", "sql"]}},
                           "required": ["rota"]},
            },
            timeout=30,
        )
        resp.raise_for_status()
        rota = json.loads(resp.json().get("response", "{}")).get("rota")
        return rota if rota in ("contexto", "rag", "sql") else None
    except Exception:
        return None


def rotear_pergunta(pergunta, usar_llm=True, ollama_url="http://localhost:11434/api/generate", ollama_model="llama3.2"):
    """Decide a rota (RAG/NL-to-SQL/contexto). Hibrido: keyword primeiro (rapido); se cair em
    'contexto' por default E usar_llm, confirma com o LLM para pegar rag/sql disfarcados por
    sinonimo. Se o keyword ja detectou rag/sql (sinal forte), confia nele sem custo de latencia.

    ollama_url/ollama_model tem defaults iguais aos usados pelo dashboard, mas sao
    parametrizaveis -- eval/rodar_golden.py pode passar um modelo diferente (ex: qwen2.5:7b)
    sem precisar duplicar rotear_pergunta()."""
    # Trava de prioridade maxima (ver pede_confirmacao_alarme_automatico): impede que o
    # desempate por LLM reclassifique essa pergunta como sql, o que levava a tabela errada.
    if pede_interpretacao_recall(pergunta):
        return "interpretacao_recall"
    if pede_confirmacao_alarme_automatico(pergunta):
        return "contexto"
    if pede_ranking_assets_duas_empresas(pergunta):
        return "contexto"
    if pede_downtime_dataset_errado(pergunta):
        return "contexto"
    if pede_matriz_confusao_precalculada(pergunta):
        return "contexto"
    rota_kw = rotear_por_keyword(pergunta)
    if rota_kw != "contexto" or not usar_llm:
        return rota_kw
    # keyword nao achou sinal forte -> pede desempate ao LLM (pega sinonimos)
    rota_llm = rotear_por_llm(pergunta, ollama_url, ollama_model)
    return rota_llm or rota_kw
