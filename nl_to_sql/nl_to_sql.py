"""
NL-to-SQL - traduz pergunta em portugues para consulta SQL no Postgres via Ollama local,
executa (somente SELECT, com validacao de seguranca) e formata a resposta.
Cronograma: Eixo 5, E1-E6 (NL-to-SQL, LangChain SQL Database Agent, validacao/sanitizacao).

Fluxo deste arquivo, do mais simples ao mais completo:
  1. gerar_sql() + corrigir_sql(): LLM traduz a pergunta, com self-repair se o SQL falhar.
  2. validar_sql_seguro() + perguntar(): so aceita SELECT, executa no Postgres real.
  3. RAG de exemplos: few-shot com consultas ja validadas, para reduzir erro de coluna.
  4. DBA-Agent: segunda opiniao do LLM sobre se o RESULTADO (nao so o SQL) responde a
     pergunta -- pega o caso raro de SQL que roda sem erro mas responde a coisa errada.
  5. resposta_amigavel(): traduz a tabela crua numa frase em portugues para o usuario.
"""
import json
import re
from pathlib import Path

import pandas as pd
import requests
from sqlalchemy import create_engine, text

ENGINE_URL = "postgresql+psycopg2://harbor:harbor123@localhost:5432/harbor_manufatura"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
EXEMPLOS_SQL_PATH = Path(__file__).parent / "exemplos_sql.json"

ESQUEMA = """
Tabelas disponiveis no banco harbor_manufatura, agrupadas por DATASET DE ORIGEM. Cada
pergunta pertence a UM dataset -- nunca misture tabelas de datasets diferentes na mesma
consulta, mesmo que os nomes de coluna pareçam parecidos (ex: "Machine_ID" existe em
sensor_predicoes, mas o dataset de manufatura discreta usa "asset", nao "Machine_ID").

--- Dataset 1: OEE/Downtime (linha de tijolos, Kakoyiannis Bricks) ---
oee_downtime_raw("Date", "Productcode", "StopGroup", "Stop", "StopType", "StopLocation", "ExtraText", "StopStartTime", "StopEndTime", "StopDuration(min)")
oee_agregacao_paradas("StopGroup", "StopType", "StopLocation", sum, count, mean)
oee_comparacao_lss("Unnamed: 0", "antes_LSS", "depois_LSS", "variacao_%")

--- Dataset 2: Legacy Sensor Logs (sensores industriais legados, rotulo Normal/Fault) ---
sensor_predicoes("Timestamp", "Machine_ID", "Target", if_anomaly)
sensor_backtest_separacao("Target", "Temperature_C", "Pressure_bar", "Vibration_Level", ...)

--- Dataset 3: Discrete Manufacturing (SME, duas empresas anonimas, company_A e company_B) ---
manufacturing_duracao_estados_a(asset, status_label, n_registros)
manufacturing_regime_a(asset, n_registros, power_avg_medio, pontos_mudanca_regime)
manufacturing_estados_antes_alarme_a(status_label, count)
manufacturing_duracao_estados_b(asset, status, n_registros)
manufacturing_regime_b(asset, n_registros, power_avg_medio, pontos_mudanca_regime)
manufacturing_consumo_energia_status_b(status, power_avg, power_min, power_max)

NOTA Dataset 3: company_A usa status numerico traduzido para status_label
(idle/manual/automatico/alarme, ingles: idle/manual/automatic/alarm); company_B usa status
TEXTUAL diferente (Alarm/Standby/MachineOn/Production/Loading/Tooling) -- NUNCA misture as
duas empresas na mesma consulta, os valores de status nao sao equivalentes. Perguntas sobre
"Alarm", "Loading" ou "Tooling" (com esses nomes exatos) sao SEMPRE de company_B, nunca de
company_A. manufacturing_consumo_energia_status_b e a UNICA tabela com energia por status
(so existe para company_B, nao ha equivalente para company_A).
manufacturing_regime_a e manufacturing_regime_b so tem 3 assets cada -- sao os 3 assets com
MAIS registros da respectiva empresa (9 assets no total em cada), escolhidos deliberadamente
pelo pipeline para analise de mudanca de regime (CUSUM). Os outros 6 assets de cada empresa
NAO tem essa analise calculada -- se a pergunta pedir "todos os N assets" e a tabela so tiver
3, avise explicitamente que a analise cobre so os 3 mais frequentes, nao inclua isso como se
fossem todos.
manufacturing_estados_antes_alarme_a NAO e contagem de registros por status -- e o estado
que ANTECEDE cada ocorrencia de alarme (transicao), sempre 3 linhas (alarme/automatico/manual
como estado anterior). Nao existe coluna "Machine_ID" em nenhuma tabela deste dataset, use
"asset".

--- Dataset 4: Five-Axis CNC Milling (usinagem, Program_path/Program_status por leitura) ---
cnc_ciclo_por_produto("Program_path", n_registros, cycle_time_medio, cycle_time_max, running_time_medio)
cnc_distribuicao_program_status("Program_status", n_registros)
cnc_resumo_anomalias_por_componente(componente, media, std, n_anomalias)

NOTA Dataset 4: nao existe tabela com colunas de temperatura por leitura individual, nem por
"Program_path" cruzado com componente/eixo -- os dados de temperatura estao SO agregados
por componente em cnc_resumo_anomalias_por_componente (5 linhas: Spindle_motor_temperature,
X_Axis_motor_temperature, Z_Axis_Motor_temperature, Y_Axis_Motor_temperature,
General_temperature). Nao ha como cruzar anomalia de temperatura com "Program_path" especifico
nem com "ExtraText" -- essas colunas nao existem em nenhuma tabela do banco.

IMPORTANTE: qualquer nome de coluna com letra maiuscula (ex: "Machine_ID", "Timestamp", "Target")
DEVE ser referenciado entre aspas duplas exatamente como mostrado acima. O Postgres e case-sensitive
para identificadores entre aspas -- sem aspas, "Machine_ID" e lido como "machine_id" (minusculo) e
a coluna nao e encontrada. Colunas totalmente minusculas (ex: sum, count, mean, asset) nao precisam de aspas.
"""


def call_ollama(prompt, timeout=120):
    """Chamada crua ao Ollama local -- toda geracao de SQL/resposta passa por aqui.

    timeout=120 (era 60): achado real ao rodar eval/avaliar_bird_sql.py com 25 perguntas
    (2026-08-07) -- 3 timeouts em schemas grandes (10-37K chars, bem acima do schema fixo
    do Harbor, ~2-3K chars) que passavam de 60s no llama3.2 local. Producao real nunca
    exercitou esse caso (schema sempre pequeno), so apareceu com gerar_sql_com_schema()
    contra benchmarks academicos de schema variavel."""
    resp = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _limpar_sql(resposta):
    """Remove markdown/crases que o LLM as vezes adiciona, garante ';' no final."""
    sql = resposta.strip()
    sql = re.sub(r"^```sql\s*|```$", "", sql, flags=re.MULTILINE).strip()
    sql = sql.rstrip(";") + ";"
    return sql


# ── 1. RAG de exemplos SQL (few-shot, sugestao do Gustavo na reuniao CERISE) ───────────────
# "Pra ele parar de errar essas colunas, tem que ter um RAG pra ele consultar no NL2SQL. Um
# banco de exemplos de consultas validas. Minimiza esse problema." Reusa o mesmo modelo E5
# do RAG hibrido (rag/rag_hibrido.py) para recuperar os exemplos mais parecidos com a pergunta.
_EXEMPLOS_STATE = {"modelo": None, "exemplos": None, "embeddings": None}


def _carregar_exemplos_sql():
    """Carrega e embeda o banco de exemplos uma unica vez (cacheado no processo)."""
    if _EXEMPLOS_STATE["exemplos"] is not None:
        return
    from sentence_transformers import SentenceTransformer
    import numpy as np

    dados = json.loads(EXEMPLOS_SQL_PATH.read_text(encoding="utf-8"))
    exemplos = dados["exemplos"]
    modelo = SentenceTransformer("intfloat/multilingual-e5-small")
    perguntas = [f"query: {ex['pergunta']}" for ex in exemplos]
    embeddings = modelo.encode(perguntas, normalize_embeddings=True)

    _EXEMPLOS_STATE["modelo"] = modelo
    _EXEMPLOS_STATE["exemplos"] = exemplos
    _EXEMPLOS_STATE["embeddings"] = np.array(embeddings)


def recuperar_exemplos_similares(pergunta_nl, k=3):
    """Retorna os k exemplos {pergunta, sql} mais parecidos com a pergunta do usuario,
    por similaridade de cosseno (mesmo padrao E5 do RAG neural: prefixos query:/passage:)."""
    import numpy as np

    _carregar_exemplos_sql()
    modelo = _EXEMPLOS_STATE["modelo"]
    emb_query = modelo.encode([f"query: {pergunta_nl}"], normalize_embeddings=True)[0]
    embeddings = _EXEMPLOS_STATE["embeddings"]

    scores = embeddings @ emb_query  # cosseno, ja normalizado
    top_idx = np.argsort(scores)[::-1][:k]
    return [_EXEMPLOS_STATE["exemplos"][i] for i in top_idx]


def _formatar_exemplos_prompt(exemplos):
    """Formata os exemplos recuperados como pares Pergunta/SQL, prontos para colar no prompt."""
    blocos = []
    for ex in exemplos:
        blocos.append(f"Pergunta: {ex['pergunta']}\nSQL: {ex['sql']}")
    return "\n\n".join(blocos)


# ── 2. Geração de SQL (com self-repair) ─────────────────────────────────────────────────

def _montar_prompt_gerar_sql(pergunta_nl, esquema, exemplos_txt):
    """Monta o prompt de geracao de SQL. Extraido de gerar_sql() para que o texto do prompt
    viva em UM lugar so, reusado tanto pela producao (esquema fixo do Harbor) quanto por
    gerar_sql_com_schema() (esquema dinamico, benchmarks BIRD-SQL/Spider) -- ver
    gerar_sql_com_schema() abaixo para o motivo de precisar de schema parametrizavel."""
    return f"""Voce e um especialista em SQL PostgreSQL. Traduza a pergunta abaixo em uma consulta SQL
usando APENAS as tabelas e colunas listadas no esquema. Responda SOMENTE com o SQL, sem explicacao,
sem markdown, sem ```sql. A query deve ser um SELECT (nunca INSERT/UPDATE/DELETE/DROP).

REGRA CRITICA DE SINTAXE: as colunas com letras maiusculas (ex: StopGroup, Machine_ID, Program_path)
DEVEM ser escritas entre aspas duplas exatamente como no esquema (ex: "StopGroup", "Machine_ID"),
porque o PostgreSQL e case-sensitive dentro de aspas duplas e minusculiza tudo sem elas.

REGRA DE SIMPLICIDADE: prefira consultas de UMA UNICA TABELA. So use JOIN se a pergunta
exigir explicitamente combinar dados de duas tabelas diferentes. Nao invente JOINs.

REGRA CRITICA DE AGREGACAO PRE-CALCULADA: antes de usar COUNT(*)/SUM()/AVG() para "ranquear"
ou "contar quantos", verifique se a tabela do esquema JA TEM uma coluna com esse numero pronto
(ex: "n_anomalias", "n_registros", "pontos_mudanca_regime", "power_avg_medio", "sum", "count",
"mean" ja sao totais pre-agregados, uma linha por categoria). Se a coluna pronta existir, use
ORDER BY direto nela -- NUNCA GROUP BY + COUNT(*)/COUNT(coluna) sobre uma tabela que ja e o
resultado agregado, porque isso so conta "1 por categoria" (numero de linhas da tabela), nao o
total real. Exemplos CORRETOS: "ranqueie os componentes por numero de anomalias" ->
SELECT componente, n_anomalias FROM cnc_resumo_anomalias_por_componente ORDER BY n_anomalias DESC;
"ranqueie os assets por mudancas de regime" -> SELECT asset, pontos_mudanca_regime FROM
manufacturing_regime_a ORDER BY pontos_mudanca_regime DESC.

=== ESQUEMA ===
{esquema}
{exemplos_txt}
=== PERGUNTA ===
{pergunta_nl}

SQL:"""


def _montar_exemplos_txt(pergunta_nl):
    """Bloco de few-shot formatado, ou string vazia se o RAG de exemplos falhar
    (E5 indisponivel) -- mesmo fallback silencioso que gerar_sql() sempre teve."""
    try:
        exemplos = recuperar_exemplos_similares(pergunta_nl, k=3)
        return f"""
=== EXEMPLOS DE CONSULTAS VALIDAS JA TESTADAS (use como referencia de sintaxe e colunas) ===
{_formatar_exemplos_prompt(exemplos)}
"""
    except Exception:
        return ""


def gerar_sql(pergunta_nl, usar_few_shot=True):
    """Traduz a pergunta em SQL: monta o prompt (esquema + regras + exemplos few-shot
    opcionais) e pede ao LLM. Não executa a query -- só gera o texto do SQL.

    Assinatura e comportamento INTOCADOS por design -- usada pela producao real
    (perguntar()/perguntar_com_dba()). Sempre usa o ESQUEMA fixo do Harbor."""
    exemplos_txt = _montar_exemplos_txt(pergunta_nl) if usar_few_shot else ""
    prompt = _montar_prompt_gerar_sql(pergunta_nl, ESQUEMA, exemplos_txt)
    return _limpar_sql(call_ollama(prompt))


def gerar_sql_com_schema(pergunta_nl, esquema, usar_few_shot=False):
    """Mesma geracao de SQL de gerar_sql(), mas com ESQUEMA DINAMICO -- para benchmarks
    academicos (BIRD-SQL, Spider) cujo schema varia por pergunta, diferente do banco fixo
    do Harbor. usar_few_shot=False por padrao: os exemplos de exemplos_sql.json sao
    especificos do schema do Harbor, nao fazem sentido como referencia de sintaxe para um
    schema de terceiros.

    Nao reusa gerar_sql() diretamente (ela so aceita o ESQUEMA fixo do modulo) -- ambas
    chamam _montar_prompt_gerar_sql() para o texto do prompt viver em um lugar so."""
    exemplos_txt = _montar_exemplos_txt(pergunta_nl) if usar_few_shot else ""
    prompt = _montar_prompt_gerar_sql(pergunta_nl, esquema, exemplos_txt)
    return _limpar_sql(call_ollama(prompt))


def _montar_prompt_corrigir_sql(pergunta_nl, esquema, sql_ruim, erro):
    """Prompt de self-repair. Extraido de corrigir_sql() pelo mesmo motivo de
    _montar_prompt_gerar_sql() -- reusado por corrigir_sql() (esquema fixo) e
    corrigir_sql_com_schema() (esquema dinamico)."""
    return f"""Voce e um especialista em SQL PostgreSQL. A consulta abaixo, gerada para responder a
pergunta, FALHOU ao ser executada. Corrija-a usando APENAS as tabelas/colunas do esquema.
Responda SOMENTE com o SQL corrigido, sem explicacao, sem markdown.

Lembre-se: colunas com maiusculas vao entre aspas duplas; prefira uma unica tabela; so SELECT.

=== ESQUEMA ===
{esquema}

=== PERGUNTA ORIGINAL ===
{pergunta_nl}

=== SQL QUE FALHOU ===
{sql_ruim}

=== ERRO DO POSTGRES ===
{erro}

SQL corrigido:"""


def corrigir_sql(pergunta_nl, sql_ruim, erro):
    """Self-repair: reenvia ao LLM o SQL que falhou + a mensagem de erro, pedindo correcao.
    Sobe muito a taxa de sucesso do modelo local, que erra sintaxe/colunas na primeira tentativa.

    Assinatura e comportamento INTOCADOS -- usada pela producao real (perguntar())."""
    prompt = _montar_prompt_corrigir_sql(pergunta_nl, ESQUEMA, sql_ruim, erro)
    return _limpar_sql(call_ollama(prompt))


def corrigir_sql_com_schema(pergunta_nl, esquema, sql_ruim, erro):
    """Self-repair com esquema dinamico -- irma de gerar_sql_com_schema(). Existe para
    benchmarks que tenham um banco real para executar e obter erro (BIRD-SQL/Spider hoje
    NAO tem, ver docstring de avaliar_bird_sql.py -- fica disponivel para uso futuro se
    isso mudar)."""
    prompt = _montar_prompt_corrigir_sql(pergunta_nl, esquema, sql_ruim, erro)
    return _limpar_sql(call_ollama(prompt))


# ── 3. Validação e execução (só SELECT, no Postgres real) ──────────────────────────────

def validar_sql_seguro(sql):
    """So permite SELECT -- nunca executa DDL/DML gerado pelo LLM sem checagem."""
    sql_normalizado = sql.strip().lower()
    if not sql_normalizado.startswith("select"):
        raise ValueError(f"Query rejeitada por seguranca (nao e SELECT): {sql}")
    proibidos = ["insert", "update", "delete", "drop", "alter", "truncate", "grant", "create"]
    for palavra in proibidos:
        if re.search(rf"\b{palavra}\b", sql_normalizado):
            raise ValueError(f"Query rejeitada por conter palavra proibida '{palavra}': {sql}")
    return True


_ENGINE = None


def get_engine():
    """Reusa um unico engine em vez de recriar a cada pergunta."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(ENGINE_URL)
    return _ENGINE


def _executar(sql):
    validar_sql_seguro(sql)
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn)


def perguntar(pergunta_nl, tentar_corrigir=True):
    """Gera SQL, valida (so SELECT) e executa. Se falhar e tentar_corrigir, faz UMA rodada de
    self-repair reenviando o erro ao LLM. Retorna (sql_final, DataFrame)."""
    sql = gerar_sql(pergunta_nl)
    try:
        resultado = _executar(sql)
        return sql, resultado
    except Exception as erro:
        if not tentar_corrigir:
            raise
        # Self-repair: uma unica tentativa de correcao.
        sql_corrigido = corrigir_sql(pergunta_nl, sql, str(erro))
        resultado = _executar(sql_corrigido)  # se falhar de novo, propaga
        return sql_corrigido, resultado


# ── 4. DBA-Agent (Parte 6 do plano padrao-ouro, padrao text2sql_agent_formacao) ────────────
# O self-repair acima so pega SQL que FALHA ao executar (erro do Postgres). Mas o bug real
# relatado por Vinicius na reuniao CERISE ("NL-to-SQL as vezes consulta a coluna errada") e SQL
# que RODA SEM ERRO mas nao responde a pergunta (ex: pergunta pede media de vibracao, LLM
# consultou media de temperatura por engano -- sintaticamente valido, semanticamente errado).
# O DBA-Agent fecha essa lacuna: pede ao LLM para avaliar se o RESULTADO responde a pergunta.
def verificar_resultado_responde(pergunta_nl, sql, resultado_df, timeout=60):
    """Segundo LLM (papel de 'DBA') avalia se o resultado da query realmente responde a
    pergunta original. Retorna (responde: bool, motivo: str). So roda se o resultado nao
    for vazio -- resultado vazio ja e auto-evidente (a query nao achou nada)."""
    if resultado_df.empty:
        return False, "A consulta nao retornou nenhuma linha."

    amostra = resultado_df.head(5).to_string()
    prompt = f"""Voce e um DBA revisando se uma consulta SQL respondeu corretamente a pergunta de um
usuario. Responda em JSON: {{"responde": true ou false, "motivo": "explicacao curta"}}.

=== PERGUNTA ORIGINAL ===
{pergunta_nl}

=== SQL EXECUTADO ===
{sql}

=== AMOSTRA DO RESULTADO (ate 5 linhas) ===
{amostra}

O resultado (colunas retornadas) realmente responde ao que foi perguntado? Por exemplo, se a
pergunta pede sobre "vibracao" mas o SQL trouxe colunas de "temperatura", isso NAO responde.
Responda SOMENTE o JSON."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                "format": {
                    "type": "object",
                    "properties": {"responde": {"type": "boolean"}, "motivo": {"type": "string"}},
                    "required": ["responde", "motivo"],
                },
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        import json
        corpo = json.loads(resp.json().get("response", "{}"))
        return bool(corpo.get("responde", True)), corpo.get("motivo", "")
    except Exception:
        # Se o DBA-Agent falhar (Ollama indisponivel, JSON invalido), nao bloqueia a resposta
        # original -- so deixa de adicionar a segunda opiniao.
        return True, "DBA-Agent indisponivel, resultado nao revisado."


def perguntar_com_dba(pergunta_nl, tentar_corrigir=True, tentar_novo_sql_se_nao_responde=True):
    """Fluxo completo: gera+executa (com self-repair de erro), depois pede ao DBA-Agent para
    checar se o resultado responde a pergunta. Se nao responder, tenta gerar um SQL novo UMA vez
    (mesmo padrao de limite de tentativas do self-repair, evita loop -- achado do estudo empirico
    de frameworks de agente: >1/3 das falhas de agentes sao loops sem teto de iteracao).
    Retorna dict: sql, resultado, dba_responde, dba_motivo."""
    sql, resultado = perguntar(pergunta_nl, tentar_corrigir=tentar_corrigir)
    dba_responde, dba_motivo = verificar_resultado_responde(pergunta_nl, sql, resultado)

    if not dba_responde and tentar_novo_sql_se_nao_responde:
        prompt_novo = f"""Voce e um especialista em SQL PostgreSQL. A consulta abaixo RODOU SEM ERRO
mas um DBA revisor apontou que ela NAO responde a pergunta original. Motivo do DBA: {dba_motivo}
Gere uma NOVA consulta que responda corretamente. Responda SOMENTE com o SQL, sem explicacao.

REGRA CRITICA DE SINTAXE: colunas com maiusculas entre aspas duplas. So SELECT. Prefira 1 tabela.

=== ESQUEMA ===
{ESQUEMA}

=== PERGUNTA ORIGINAL ===
{pergunta_nl}

=== SQL QUE RODOU MAS NAO RESPONDEU (segundo o DBA) ===
{sql}

SQL corrigido:"""
        sql_novo = _limpar_sql(call_ollama(prompt_novo))
        try:
            resultado_novo = _executar(sql_novo)
            dba_responde_novo, dba_motivo_novo = verificar_resultado_responde(pergunta_nl, sql_novo, resultado_novo)
            sql, resultado, dba_responde, dba_motivo = sql_novo, resultado_novo, dba_responde_novo, dba_motivo_novo
        except Exception:
            pass  # mantem o resultado original (rodou sem erro) se a 2a tentativa falhar

    return {"sql": sql, "resultado": resultado, "dba_responde": dba_responde, "dba_motivo": dba_motivo}


# ── 5. Resposta amigável ("Marcelo Agent", padrão text2sql_agent_formacao) ─────────────────
def resposta_amigavel(pergunta_nl, sql, resultado_df, timeout=60):
    """Transforma a tabela crua numa resposta conversacional em portugues -- a 'recomendacao ao
    usuario' priorizada por Yan, em vez de so mostrar SQL+dataframe. Usa o proprio resultado
    como contexto (grounding), instrucao anti-alucinacao explicita (mesmo padrao do chat)."""
    if resultado_df.empty:
        return "Não encontrei nenhum registro que respondesse a essa pergunta no banco de dados."

    amostra = resultado_df.head(10).to_string()
    prompt = f"""Voce e um assistente que traduz resultados de consultas SQL em respostas claras em
portugues para um usuario nao-tecnico. Use APENAS os dados abaixo -- nao invente numeros que nao
estejam na tabela. Responda em 2-3 frases diretas, indo direto ao ponto da pergunta.

=== PERGUNTA DO USUARIO ===
{pergunta_nl}

=== RESULTADO DA CONSULTA (tabela) ===
{amostra}

Resposta:"""
    return call_ollama(prompt, timeout=timeout)


def main():
    perguntas_teste = [
        "Qual categoria de parada (StopGroup) tem mais minutos totais de parada?",
        "Quantos registros existem na tabela sensor_predicoes onde if_anomaly e verdadeiro?",
        "Qual e o cycle_time_medio maximo na tabela cnc_ciclo_por_produto?",
    ]

    for pergunta in perguntas_teste:
        print(f"\nPERGUNTA: {pergunta}")
        try:
            sql, resultado = perguntar(pergunta)
            print(f"SQL GERADO: {sql}")
            print(f"RESULTADO:\n{resultado.to_string()}")
        except Exception as exc:
            print(f"ERRO: {exc}")
        print("-" * 70)


if __name__ == "__main__":
    main()
