"""
Verificacao anti-alucinacao: depois que o LLM responde, confere se o que ele disse esta
FUNDAMENTADO no contexto que foi dado a ele (RAGAS Faithfulness / "hallucination check").

Duas checagens independentes, combinadas em verificar_resposta() no final do arquivo:
  1. NUMEROS: todo numero citado na resposta precisa aparecer no contexto (ou na pergunta).
     Um numero que nao aparece em nenhum dos dois e candidato a alucinacao -- o LLM inventou.
  2. CLAIMS (opcional, mais cara): cada frase da resposta precisa ter vocabulario em comum
     com alguma frase do contexto. Pega afirmacoes inventadas sem numero nenhum.

Fonte unica reusada pelo harness (eval/rodar_golden.py) e pelo dashboard (dashboard/app.py).
Sem dependencia nova (so re, biblioteca padrao).
"""
import re

# Numeros pequenos/triviais (ex: "3 frases", "as 2 empresas") nao contam como alucinacao
# mesmo ausentes do contexto -- sao contagens genericas, nao dados do dominio.
NUMEROS_IGNORADOS_PEQUENOS = 3  # inteiros <= este valor sao ignorados


# ── 1. Checagem de números: resposta vs. contexto ──────────────────────────────────────

def extrair_numeros(texto):
    """Extrai todos os numeros de um texto, tratando decimal BR (virgula) e internacional
    (ponto). So trata VIRGULA como separador decimal quando nao for seguida de outro numero
    com ponto colado -- '19,6%' vira 19.6 (decimal BR real), mas '32444,558.03...' de um CSV
    cru nao gruda, porque ali a virgula e separador de coluna."""
    nums = []
    texto = texto or ""
    padrao = r"-?\d+\.\d+|-?\d+,\d{1,4}(?!\d)(?!\.\d)|-?\d+"
    for m in re.finditer(padrao, texto):
        b = m.group(0)
        try:
            nums.append(float(b.replace(",", ".")))
        except ValueError:
            pass
    return nums


def _partes_inteiras(numeros):
    """Amplia cada numero com sua parte inteira truncada, para casar o caso de o LLM citar
    a versao arredondada de um float longo de CSV (ex: contexto tem '558.0306660098335',
    resposta cita '558 segundos' -- sem isso, o numero gigante nunca bateria)."""
    partes = set()
    for n in numeros:
        partes.add(n)
        partes.add(float(int(n)))
    return partes


def numero_equivalente(a, b, tol=0.5):
    """Compara dois numeros por valor absoluto (percentuais aparecem com/sem sinal: -19,6
    == 19,6). Valores pequenos (<10) usam tolerancia RELATIVA apertada (2%), para nao
    confundir 0.97 com 0.643 -- proporcoes/medias precisam de match mais estrito que
    percentuais grandes, que usam a tolerancia absoluta padrao (tol)."""
    a, b = abs(a), abs(b)
    if max(a, b) < 10:
        return abs(a - b) <= max(0.02, 0.02 * max(a, b))
    return abs(a - b) <= tol


def numeros_nao_fundamentados(resposta, contexto, tol=0.5, pergunta=None):
    """Retorna os numeros citados na RESPOSTA que NAO aparecem no CONTEXTO -- os
    candidatos a alucinacao numerica. Ignora inteiros pequenos (contagens triviais).

    pergunta (opcional): numeros que ja apareciam na PERGUNTA do usuario tambem contam
    como fundamentados -- cobre o caso do LLM ecoar um numero da propria pergunta (ex:
    pergunta "ganho em cada uma das 12 linhas?" -> resposta "nao tenho essa informacao
    sobre as 12 linhas"). Isso e parafrasear o usuario, nao inventar um numero novo."""
    nums_resp = extrair_numeros(resposta)
    nums_ctx_ampliado = _partes_inteiras(extrair_numeros(contexto))
    if pergunta:
        nums_ctx_ampliado |= _partes_inteiras(extrair_numeros(pergunta))

    suspeitos = []
    for n in nums_resp:
        if n == int(n) and abs(n) <= NUMEROS_IGNORADOS_PEQUENOS:
            continue  # contagem trivial (ex: "em 3 frases"), nao e dado do dominio
        if not any(numero_equivalente(n, c, tol) for c in nums_ctx_ampliado):
            suspeitos.append(n)
    return suspeitos


# ── 2. Checagem de claims: overlap de vocabulário (opcional, mais cara) ────────────────

STOPWORDS_PT = {
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas",
    "um", "uma", "uns", "umas", "e", "ou", "que", "com", "sem", "para", "por", "se", "ao",
    "aos", "à", "às", "é", "foi", "sao", "são", "ser", "esta", "está", "isso", "essa", "esse",
    "como", "mais", "muito", "ja", "já", "nao", "não", "tem", "ha", "há", "seu", "sua",
}


def _frases(texto):
    """Divide o texto em frases (por pontuacao final), descartando as curtas demais para
    ter sinal (menos de 4 palavras -- saudacoes, conectores)."""
    partes = re.split(r"(?<=[.!?])\s+", (texto or "").strip())
    return [p.strip() for p in partes if len(p.split()) >= 4]


def _palavras_significativas(frase):
    """Tokeniza a frase e remove stopwords/pontuacao, sobrando so vocabulario de conteudo."""
    tokens = re.findall(r"[a-zà-ú0-9]+", frase.lower())
    return {t for t in tokens if t not in STOPWORDS_PT and len(t) > 2}


def claims_nao_fundamentados(resposta, contexto, limiar_overlap=0.3):
    """Para cada frase da resposta, mede a fracao de palavras de conteudo que tambem
    aparecem em alguma frase do contexto. Frases com overlap baixo sao candidatas a claim
    inventado (ex: "12 linhas de producao" quando o contexto nunca fala de linhas).

    Proxy barato (bag-of-words), NAO um verificador semantico completo (NLI) -- pega casos
    grosseiros que a checagem numerica sozinha ignora (afirmacoes sem numero nenhum). Tem
    risco de falso positivo com parafrase forte (sinonimos), por isso o limiar e permissivo
    (30%) e a checagem so roda quando pedida explicitamente (ver checar_claims abaixo)."""
    palavras_ctx = _palavras_significativas(contexto or "")
    if not palavras_ctx:
        return []

    suspeitos = []
    for frase in _frases(resposta):
        palavras_frase = _palavras_significativas(frase)
        if not palavras_frase:
            continue
        overlap = len(palavras_frase & palavras_ctx) / len(palavras_frase)
        if overlap < limiar_overlap:
            suspeitos.append(frase)
    return suspeitos


# ── 3. Fachada: combina as duas checagens num resultado único ──────────────────────────

def verificar_resposta(resposta, contexto, tol=0.5, checar_claims=False, pergunta=None):
    """Verificacao completa de uma resposta contra seu contexto. Retorna um dict com:
      - fundamentada: True se nenhuma checagem encontrou suspeita.
      - suspeitos: números da resposta ausentes do contexto (checagem 1, sempre roda).
      - claims_suspeitos: frases com baixo overlap de vocabulário (checagem 2, opt-in).
      - aviso: mensagem pronta para mostrar ao usuário, ou None se tudo fundamentado.

    checar_claims=False por padrão: a checagem numérica é barata e roda sempre; a de
    claims é mais cara e tem mais ruído (falso positivo com paráfrase), então fica opt-in.
    pergunta (opcional): repassado à checagem 1 -- ver numeros_nao_fundamentados()."""
    suspeitos = numeros_nao_fundamentados(resposta, contexto, tol, pergunta=pergunta)
    claims_suspeitos = claims_nao_fundamentados(resposta, contexto) if checar_claims else []
    fundamentada = len(suspeitos) == 0 and len(claims_suspeitos) == 0

    aviso = None
    if not fundamentada:
        partes = []
        if suspeitos:
            nums = ", ".join(f"{s:g}" for s in suspeitos)
            partes.append(f"numero(s) que nao encontrei no contexto: {nums}")
        if claims_suspeitos:
            partes.append(f"{len(claims_suspeitos)} afirmacao(oes) com baixo respaldo no contexto")
        aviso = f"[verificacao] a resposta cita {' e '.join(partes)}. Trate com cautela."

    return {
        "fundamentada": fundamentada,
        "suspeitos": suspeitos,
        "claims_suspeitos": claims_suspeitos,
        "aviso": aviso,
    }


if __name__ == "__main__":
    # teste rapido
    ctx = "OEE subiu de 0.5907 para 0.643 (variacao 8.9%). MTTR caiu 26.3%."
    print("Caso 1 (fundamentada):", verificar_resposta("O OEE subiu 8.9%, de 0.5907 para 0.643.", ctx))
    print("Caso 2 (alucinada):   ", verificar_resposta("O OEE subiu 15% e o custo caiu 42%.", ctx))
    print("Caso 3 (Bruno 0,97):  ", verificar_resposta("A media foi 0,97.", ctx))
