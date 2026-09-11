"""
Checks deterministicos de fidelidade de caption gerado por VLM (item 1 do plano "Evoluir o RAG
multimodal", 2026-09-09) -- mede GERACAO, nao retrieval. Recall@k/MRR/Precision@k (ja existentes
em eval/avaliar_rag_multimodal.py) medem se o RANKING acerta o documento certo; estes checks
medem se o TEXTO gerado pelo VLM e FIEL ao que a imagem de fato mostra, comparando contra o
ground truth factual de rag/metadados_imagens_ground_truth.py (as imagens sao geradas
programaticamente, entao o ground truth e exato, nao uma aproximacao).

Sem LLM-judge -- checks 100% deterministicos e reproduziveis, porque o ground truth ja e
conhecido (nao ha ambiguidade de "o que a imagem realmente mostra" para resolver via
julgamento).

6 checks, do mais simples ao mais critico:
1. nao_degenerado -- a legenda tem conteudo minimo (nao e vazia/vetor de floats/lixo)
2. idioma_pt -- a legenda esta em portugues, nao em ingles (achado real: 3 de 4 legendas do
   moondream vieram em ingles apesar do prompt pedir portugues)
3. tipo_grafico_correto -- a legenda menciona o tipo certo (boxplot/barra/scatter/linha
   temporal), nao um tipo errado (achado real: "Line graph" para um scatter)
4. menciona_eixos_corretos -- a legenda cita (aproximadamente) os rotulos reais dos eixos
   (achado real: legenda trocou titulo por rotulo do eixo X)
5. contem_ranking -- quando a imagem tem ranking (ground_truth["ranking"] != None), a legenda
   menciona qual categoria e maior/menor, nao so lista categorias (achado real: legenda nunca
   mencionava ranking, so "bar graph with temperature and component")
6. sem_numeros_inventados -- o MAIS IMPORTANTE: nenhum numero mencionado na legenda pode ser
   claramente incompativel com os dados de origem (achado real: "frequency 0.6/0.4" nao existe
   no boxplot de vibracao -- e alucinacao, nao imprecisao)

Uso: python eval/checks_fidelidade_caption.py [caminho_para_cache_legendas.json]
     (default: rag/legendas_cache.json)
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))
from metadados_imagens_ground_truth import GROUND_TRUTH

# Palavras-chave de tipo de grafico em portugues E ingles (VLMs frequentemente respondem em
# ingles apesar do prompt pedir portugues -- ver check idioma_pt, mas tipo_grafico_correto deve
# aceitar ambos os idiomas para nao confundir os dois checks).
PALAVRAS_TIPO = {
    "boxplot": ["boxplot", "box plot", "diagrama de caixa", "caixa"],
    "barra": ["barra", "bar graph", "bar chart", "gráfico de barras"],
    "scatter": ["dispersão", "scatter", "dispersao", "gráfico de dispersão"],
    "linha_temporal": ["linha", "line graph", "line chart", "série temporal", "serie temporal"],
}

# Deteccao simples de idioma: presenca de palavras funcionais comuns em portugues vs. ingles.
PALAVRAS_FUNCIONAIS_PT = {"o", "a", "de", "que", "com", "para", "por", "uma", "os", "as", "não", "é"}
PALAVRAS_FUNCIONAIS_EN = {"the", "is", "with", "and", "of", "graph", "shows", "has", "this"}


def nao_degenerado(legenda: str) -> bool:
    """Legenda tem conteudo minimo -- nao vazia, nao e so numeros/colchetes, tamanho razoavel."""
    texto = legenda.strip()
    if len(texto) < 15:
        return False
    if re.fullmatch(r"[\[\]0-9.,\s\-]+", texto):
        return False  # so numeros/colchetes -- caso real "[0.0, 0.13, 0.99, 0.28]"
    palavras = re.findall(r"[a-zA-Zà-úÀ-Ú]+", texto)
    return len(palavras) >= 4


def idioma_pt(legenda: str) -> bool:
    """Legenda esta em portugues -- conta palavras funcionais de cada idioma, português vence
    se tiver pelo menos 1 marcador e mais marcadores que ingles (ou empate com acentuacao)."""
    texto_lower = legenda.lower()
    palavras = set(re.findall(r"[a-zà-ú]+", texto_lower))
    pt_hits = len(palavras & PALAVRAS_FUNCIONAIS_PT)
    en_hits = len(palavras & PALAVRAS_FUNCIONAIS_EN)
    tem_acento = bool(re.search(r"[àáâãéêíóôõúç]", texto_lower))
    if pt_hits == 0 and not tem_acento:
        return False
    return pt_hits >= en_hits


def tipo_grafico_correto(legenda: str, tipo_esperado: str) -> bool:
    """Legenda menciona o tipo de grafico certo -- aceita portugues ou ingles (o check de
    idioma e separado, este so valida CONTEUDO)."""
    texto_lower = legenda.lower()
    palavras_esperadas = PALAVRAS_TIPO.get(tipo_esperado, [])
    return any(p in texto_lower for p in palavras_esperadas)


def menciona_eixos_corretos(legenda: str, eixo_x: str, eixo_y: str) -> bool:
    """Legenda cita pelo menos uma palavra significativa de cada rotulo de eixo real (ignora
    unidades entre parenteses e palavras curtas, para tolerar parafraseamento)."""
    texto_lower = legenda.lower()

    def palavras_significativas(rotulo):
        sem_unidade = re.sub(r"\([^)]*\)", "", rotulo)
        return [p for p in re.findall(r"[a-zà-ú]+", sem_unidade.lower()) if len(p) >= 4]

    palavras_x = palavras_significativas(eixo_x)
    palavras_y = palavras_significativas(eixo_y)
    x_ok = any(p in texto_lower for p in palavras_x) if palavras_x else True
    y_ok = any(p in texto_lower for p in palavras_y) if palavras_y else True
    return x_ok and y_ok


def contem_ranking(legenda: str, ranking_esperado) -> bool:
    """Quando a imagem tem ranking, a legenda deve conter linguagem comparativa (maior/menor/
    mais/menos/primeiro/etc) -- nao valida QUAL categoria e a maior, so que o VLM tentou
    reportar uma ordem (validar a ordem exata exigiria NER de categoria, fora de escopo destes
    checks deterministicos simples)."""
    if ranking_esperado is None:
        return True  # nao se aplica
    texto_lower = legenda.lower()
    marcadores = ["maior", "menor", "mais", "menos", "top", "primeiro", "último", "ultimo",
                  "máximo", "maximo", "mínimo", "minimo", "ranking", "ordem", "highest", "lowest"]
    return any(m in texto_lower for m in marcadores)


def sem_numeros_inventados(legenda: str, limites_plausiveis: dict) -> bool:
    """Nenhum numero na legenda pode estar claramente fora da faixa plausivel das variaveis de
    origem. `limites_plausiveis`: dict var -> (min, max) ja calculado a partir do CSV real.
    Heuristica conservadora: só reprova se o numero for claramente fora de QUALQUER faixa
    conhecida (evita falso positivo em numeros de contexto, ex: "classe 1")."""
    # Achado real (2026-09-10): o VLM escreve decimais em pt-BR (virgula), ex. "0,099 V" -- um
    # regex so com ponto decimal quebra "0,099" em dois numeros falsos ("0" e "099"), inflando
    # falsos positivos de "numero inventado" quando o numero na verdade estava correto.
    numeros = [
        float(n.replace(",", "."))
        for n in re.findall(r"-?\d+(?:[.,]\d+)?", legenda)
    ]
    if not numeros or not limites_plausiveis:
        return True  # sem numero para checar, ou sem faixa conhecida -- nao reprova
    todas_faixas = list(limites_plausiveis.values())
    margem = 0.5  # 50% de folga sobre a AMPLITUDE da faixa (max-min), nao sobre o valor
    # absoluto -- folga proporcional ao valor absoluto (ex: +-1 fixo) explode para variaveis
    # com faixa real pequena perto de zero (achado real ao testar: Vibration_Level varia so
    # 0,78-0,81, mas uma folga de +-1 cobria ate 2,2, deixando "0.6"/"0.4" inventados passarem
    # como "plausiveis").
    for n in numeros:
        if n in (0, 1, 2, 3, 4, 5):
            continue  # indices/contadores pequenos comuns em contexto (ex: "classe 1",
            # "2 categorias") -- mas NAO ignora decimais como 0.6/0.4, que sao valores
            # reportados como dado, nao indices (achado real: "frequency 0.6/0.4" inventado).
            # Achado real (2026-09-10): "5" tambem aparece sempre no titulo "CNC 5 eixos" do
            # corpus deste dataset -- sem isso, 5 imagens reprovavam por citar o nome do
            # equipamento, nao um dado inventado.
        dentro_de_alguma_faixa = any(
            (mn - (mx - mn) * margem) <= n <= (mx + (mx - mn) * margem)
            for mn, mx in todas_faixas
        )
        if not dentro_de_alguma_faixa:
            return False
    return True


def avaliar_legenda(doc_id: str, legenda: str, limites_plausiveis: dict = None) -> dict:
    """Roda os 6 checks para uma legenda, contra o ground truth de doc_id. Retorna dict com o
    resultado de cada check + `passou_todos` (bool) + `taxa_aprovacao` (float 0-1)."""
    gt = GROUND_TRUTH.get(doc_id)
    if gt is None:
        raise KeyError(f"Sem ground truth para '{doc_id}' -- ver rag/metadados_imagens_ground_truth.py")

    resultados = {
        "nao_degenerado": nao_degenerado(legenda),
        "idioma_pt": idioma_pt(legenda),
        "tipo_grafico_correto": tipo_grafico_correto(legenda, gt["tipo"]),
        "menciona_eixos_corretos": menciona_eixos_corretos(legenda, gt["eixo_x"], gt["eixo_y"]),
        "contem_ranking": contem_ranking(legenda, gt["ranking"]),
        "sem_numeros_inventados": sem_numeros_inventados(legenda, limites_plausiveis or {}),
    }
    n_passou = sum(resultados.values())
    resultados["passou_todos"] = n_passou == len(resultados)
    resultados["taxa_aprovacao"] = n_passou / len(resultados)
    return resultados


def _calcular_limites_plausiveis():
    """Calcula (min, max) real de cada variavel de origem, para o check
    sem_numeros_inventados -- lido uma vez, reusado para todas as legendas."""
    import pandas as pd
    raiz = Path(r"C:\Projetos\Harbor")
    csv2 = raiz / "outputs" / "pipeline2_legacy_sensor" / "separacao_features_por_classe.csv"
    csv4 = raiz / "outputs" / "pipeline4_five_axis_cnc" / "anomalias_temperatura.csv"
    limites = {}
    if csv2.exists():
        df2 = pd.read_csv(csv2)
        for col in df2.select_dtypes("number").columns:
            limites[col] = (float(df2[col].min()), float(df2[col].max()))
    if csv4.exists():
        df4 = pd.read_csv(csv4)
        for col in df4.select_dtypes("number").columns:
            limites[col] = (float(df4[col].min()), float(df4[col].max()))
    return limites


def main():
    caminho_cache = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Projetos\Harbor\rag\legendas_cache.json")
    if not caminho_cache.exists():
        print(f"Cache nao encontrado: {caminho_cache}")
        print("Rode primeiro: python rag/rag_multimodal_langchain.py --captionar")
        return

    docs = json.loads(caminho_cache.read_text(encoding="utf-8"))
    limites_plausiveis = _calcular_limites_plausiveis()

    resultados_por_doc = {}
    for doc in docs:
        doc_id = doc["id"]
        legenda = doc["texto"]
        if doc_id not in GROUND_TRUTH:
            print(f"[{doc_id:40}] SEM GROUND TRUTH -- pulado")
            continue
        r = avaliar_legenda(doc_id, legenda, limites_plausiveis)
        resultados_por_doc[doc_id] = r
        status = "PASSOU" if r["passou_todos"] else f"{r['taxa_aprovacao']*100:.0f}%"
        falhas = [k for k, v in r.items() if k not in ("passou_todos", "taxa_aprovacao") and not v]
        print(f"[{doc_id:40}] {status:8} falhas={falhas}")

    if not resultados_por_doc:
        print("\nNenhuma legenda avaliada.")
        return

    n_docs = len(resultados_por_doc)
    taxa_media = sum(r["taxa_aprovacao"] for r in resultados_por_doc.values()) / n_docs
    n_passou_tudo = sum(1 for r in resultados_por_doc.values() if r["passou_todos"])
    n_sem_numero_inventado = sum(1 for r in resultados_por_doc.values() if r["sem_numeros_inventados"])

    print("\n" + "=" * 60)
    print(f"Documentos avaliados      : {n_docs}")
    print(f"Taxa de aprovação média   : {taxa_media*100:.1f}%")
    print(f"Passou todos os checks    : {n_passou_tudo}/{n_docs}")
    print(f"Sem números inventados    : {n_sem_numero_inventado}/{n_docs}")
    print("\nCritério de promoção do plano: >=90% de taxa média E 100% em "
          "sem_numeros_inventados.")
    if taxa_media >= 0.9 and n_sem_numero_inventado == n_docs:
        print("RESULTADO: aprovado pelo critério de promoção.")
    else:
        print("RESULTADO: NÃO aprovado pelo critério de promoção.")


if __name__ == "__main__":
    main()
