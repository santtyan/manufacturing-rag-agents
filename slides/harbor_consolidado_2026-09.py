"""
Deck consolidado do projeto Harbor -- setembro de 2026.

Gera harbor_consolidado_2026-09.pptx a partir dos 6 slides Beamer (.tex)
existentes, unificados num unico deck panoramico em python-pptx.

Rodar de dentro de slides/:
    python harbor_consolidado_2026-09.py
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
import os

# ---------------------------------------------------------------------------
# Paleta Cerise
# ---------------------------------------------------------------------------
MAGENTA = RGBColor(166, 24, 107)      # #A6186B - titulos, destaques
ROSA = RGBColor(230, 24, 109)         # #E6186D
CORAL = RGBColor(245, 100, 45)        # #F5642D
ROXO = RGBColor(91, 26, 107)          # #5B1A6B
BRANCO = RGBColor(255, 255, 255)
QUASE_PRETO = RGBColor(40, 40, 40)
CINZA_CLARO = RGBColor(240, 240, 240)
CINZA_MEDIO = RGBColor(120, 120, 120)
FUNDO_ALERTA = RGBColor(253, 231, 220)   # coral bem clarinho
FUNDO_CRITICO = RGBColor(253, 210, 200)  # vermelho/laranja claro
VERMELHO = RGBColor(178, 34, 34)
BORDA_ALERTA = CORAL

FONTE = "Calibri"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGEM = Inches(0.55)
CONTEUDO_W = SLIDE_W - 2 * MARGEM


# ---------------------------------------------------------------------------
# Helpers genericos
# ---------------------------------------------------------------------------

def slide_em_branco(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def add_fundo(slide, cor=BRANCO):
    """Preenche o fundo do slide com uma cor solida."""
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = cor


def add_titulo(slide, texto, cor=MAGENTA, top=Inches(0.35), tamanho=30, largura=None):
    largura = largura or CONTEUDO_W
    box = slide.shapes.add_textbox(MARGEM, top, largura, Inches(0.9))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = texto
    run.font.size = Pt(tamanho)
    run.font.bold = True
    run.font.name = FONTE
    run.font.color.rgb = cor
    return box


def add_subtitulo_slide(slide, texto, top=Inches(1.05), cor=ROXO, tamanho=16, italico=True):
    box = slide.shapes.add_textbox(MARGEM, top, CONTEUDO_W, Inches(0.5))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = texto
    run.font.size = Pt(tamanho)
    run.font.italic = italico
    run.font.name = FONTE
    run.font.color.rgb = cor
    return box


def add_paragrafo(tf, texto, tamanho=16, cor=QUASE_PRETO, bold=False, italic=False,
                   bullet=False, level=0, space_after=8, primeiro=False):
    if primeiro and len(tf.paragraphs) == 1 and not tf.paragraphs[0].runs:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.level = level
    p.space_after = Pt(space_after)
    run = p.add_run()
    prefixo = "•  " if bullet else ""
    run.text = prefixo + texto
    run.font.size = Pt(tamanho)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = FONTE
    run.font.color.rgb = cor
    return p


def add_caixa_texto(slide, left, top, width, height):
    box = slide.shapes.add_textbox(left, top, width, height)
    box.text_frame.word_wrap = True
    return box.text_frame


def add_rodape(slide, texto, numero=None):
    box = slide.shapes.add_textbox(MARGEM, SLIDE_H - Inches(0.45), CONTEUDO_W, Inches(0.35))
    tf = box.text_frame
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = texto
    run.font.size = Pt(10)
    run.font.italic = True
    run.font.name = FONTE
    run.font.color.rgb = CINZA_MEDIO
    if numero is not None:
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.RIGHT


def set_shape_sem_sombra(shape):
    shape.shadow.inherit = False


def add_bloco_destaque(slide, left, top, width, height, titulo, corpo_linhas,
                        cor_borda=CORAL, cor_fundo=FUNDO_ALERTA, titulo_tam=15, corpo_tam=13.5):
    """Bloco tipo 'alertblock' do Beamer: fundo leve + barra lateral colorida."""
    caixa = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    caixa.fill.solid()
    caixa.fill.fore_color.rgb = cor_fundo
    caixa.line.color.rgb = cor_borda
    caixa.line.width = Pt(1)
    set_shape_sem_sombra(caixa)

    barra = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, Inches(0.09), height)
    barra.fill.solid()
    barra.fill.fore_color.rgb = cor_borda
    barra.line.fill.background()
    set_shape_sem_sombra(barra)

    tf = caixa.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.25)
    tf.margin_right = Inches(0.15)
    tf.margin_top = Inches(0.1)
    tf.margin_bottom = Inches(0.1)

    p0 = tf.paragraphs[0]
    r0 = p0.add_run()
    r0.text = titulo
    r0.font.bold = True
    r0.font.size = Pt(titulo_tam)
    r0.font.name = FONTE
    r0.font.color.rgb = ROXO
    p0.space_after = Pt(6)

    if isinstance(corpo_linhas, str):
        corpo_linhas = [corpo_linhas]
    for linha in corpo_linhas:
        p = tf.add_paragraph()
        p.space_after = Pt(4)
        run = p.add_run()
        run.text = linha
        run.font.size = Pt(corpo_tam)
        run.font.name = FONTE
        run.font.color.rgb = QUASE_PRETO
    return caixa


def add_nota(slide, left, top, width, height, texto, cor=CINZA_MEDIO, tamanho=12):
    tf = add_caixa_texto(slide, left, top, width, height)
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = texto
    run.font.size = Pt(tamanho)
    run.font.italic = True
    run.font.name = FONTE
    run.font.color.rgb = cor
    return tf


def _set_cell(cell, texto, tamanho=13, bold=False, cor_fonte=QUASE_PRETO, cor_fundo=None,
              alinhar_centro=False):
    cell.text = ""
    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = str(texto)
    run.font.size = Pt(tamanho)
    run.font.bold = bold
    run.font.name = FONTE
    run.font.color.rgb = cor_fonte
    if alinhar_centro:
        p.alignment = PP_ALIGN.CENTER
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    if cor_fundo is not None:
        cell.fill.solid()
        cell.fill.fore_color.rgb = cor_fundo
    else:
        cell.fill.solid()
        cell.fill.fore_color.rgb = BRANCO


def add_tabela(slide, left, top, width, height, cabecalho, linhas, larguras_rel=None,
               tam_cabecalho=13, tam_corpo=12.5, listras=True):
    n_linhas = len(linhas) + 1
    n_cols = len(cabecalho)
    grafico = slide.shapes.add_table(n_linhas, n_cols, left, top, width, height)
    tabela = grafico.table

    if larguras_rel:
        total = sum(larguras_rel)
        for i, rel in enumerate(larguras_rel):
            tabela.columns[i].width = Emu(int(width * (rel / total)))

    for j, titulo in enumerate(cabecalho):
        _set_cell(tabela.cell(0, j), titulo, tamanho=tam_cabecalho, bold=True,
                  cor_fonte=BRANCO, cor_fundo=MAGENTA, alinhar_centro=True)

    for i, linha in enumerate(linhas, start=1):
        cor_fundo = CINZA_CLARO if (listras and i % 2 == 0) else BRANCO
        for j, valor in enumerate(linha):
            _set_cell(tabela.cell(i, j), valor, tamanho=tam_corpo, cor_fundo=cor_fundo,
                      alinhar_centro=(j > 0))
    return grafico


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------

def slide_titulo(prs):
    slide = slide_em_branco(prs)
    add_fundo(slide, BRANCO)

    faixa = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(2.6), SLIDE_W, Inches(2.4))
    faixa.fill.solid()
    faixa.fill.fore_color.rgb = MAGENTA
    faixa.line.fill.background()
    set_shape_sem_sombra(faixa)

    tf = faixa.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = "Harbor — Panorama do Projeto"
    run.font.size = Pt(40)
    run.font.bold = True
    run.font.name = FONTE
    run.font.color.rgb = BRANCO

    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    p2.space_before = Pt(14)
    run2 = p2.add_run()
    run2.text = ("Pipelines, RAG híbrido, NL-to-SQL, OpenPack e a migração "
                 "para LangGraph — setembro de 2026")
    run2.font.size = Pt(18)
    run2.font.italic = True
    run2.font.name = FONTE
    run2.font.color.rgb = BRANCO

    rodape_tf = add_caixa_texto(slide, MARGEM, Inches(6.6), CONTEUDO_W, Inches(0.7))
    p3 = rodape_tf.paragraphs[0]
    p3.alignment = PP_ALIGN.CENTER
    r3 = p3.add_run()
    r3.text = "Yan Santos Leite  •  25 de setembro de 2026"
    r3.font.size = Pt(16)
    r3.font.name = FONTE
    r3.font.color.rgb = ROXO


def slide_ato0(prs):
    slide = slide_em_branco(prs)
    add_fundo(slide)
    add_titulo(slide, "Ato 0 — A tese")
    add_subtitulo_slide(slide, "A pergunta que organiza tudo")

    tf = add_caixa_texto(slide, MARGEM, Inches(1.75), CONTEUDO_W, Inches(2.6))
    add_paragrafo(tf, "Até 08/09, o Harbor não conseguia responder: "
                       "“essa mudança melhorou o sistema?”", tamanho=18,
                  bullet=True, primeiro=True)
    add_paragrafo(tf, "Sem trace, sem custo medido, sem comparação antes/depois.",
                  tamanho=18, bullet=True)
    add_paragrafo(tf, "Duas etapas seguidas nesta sessão:", tamanho=18, bullet=True)
    add_paragrafo(tf, "(1) construir capacidade de medir", tamanho=17, bullet=True, level=1)
    add_paragrafo(tf, "(2) usar essa capacidade em decisões reais — em várias delas "
                       "a resposta foi “não”", tamanho=17, bullet=True, level=1)

    add_bloco_destaque(
        slide, MARGEM, Inches(4.7), CONTEUDO_W, Inches(1.9),
        "Tese",
        ["Instrumentação não serve para confirmar o que já queríamos fazer.",
         "Serve para descobrir o que NÃO devemos fazer."],
        cor_borda=ROXO, cor_fundo=RGBColor(238, 227, 240), titulo_tam=17, corpo_tam=16,
    )


def slide_ato1(prs):
    slide = slide_em_branco(prs)
    add_fundo(slide)
    add_titulo(slide, "Ato 1 — Construir a régua")

    tf = add_caixa_texto(slide, MARGEM, Inches(1.3), CONTEUDO_W, Inches(2.3))
    add_paragrafo(tf, "3 peças novas de instrumentação:", tamanho=17, bold=True,
                  primeiro=True)
    add_paragrafo(tf, "shared/trace.py — esquema gen_ai.* / OpenTelemetry + proveniência",
                  tamanho=15.5, bullet=True)
    add_paragrafo(tf, "shared/ollama_client.py — captura tokens/modelo (antes, 13 pontos "
                       "descartavam essa informação)", tamanho=15.5, bullet=True)
    add_paragrafo(tf, "shared/rollout_card.py — relatório de replicação",
                  tamanho=15.5, bullet=True)

    tf2 = add_caixa_texto(slide, MARGEM, Inches(3.65), CONTEUDO_W, Inches(1.6))
    add_paragrafo(tf2, "Régua em uso: agente ReAct sobre benchmark", tamanho=17, bold=True,
                  primeiro=True)
    add_paragrafo(tf2, "SQL self-repair: 3 execuções, 1/3 sucesso "
                        "(27.163 / 1.342 tokens in/out)", tamanho=15.5, bullet=True)
    add_paragrafo(tf2, "HotpotQA: 2 execuções, 1/2 sucesso", tamanho=15.5, bullet=True)

    add_bloco_destaque(
        slide, MARGEM, Inches(5.4), CONTEUDO_W, Inches(1.55),
        "Achado + regra",
        ["2/3 execuções de SQL travaram no mesmo erro de sintaxe repetido, "
         "sem o modelo se corrigir.",
         "Regra: todo experimento com agente entrega código + trace + "
         "relatório de replicação."],
    )


def slide_ato2_langgraph(prs):
    slide = slide_em_branco(prs)
    add_fundo(slide)
    add_titulo(slide, "Ato 2 — Migração para LangGraph (EM PRODUÇÃO)")

    cabecalho = ["Métrica", "Antes", "Depois"]
    linhas = [
        ["Roteamento (golden set, 67 perguntas)", "56/67", "55/67"],
        ["Faithfulness", "64%", "64%"],
        ["Alucinações", "4", "3"],
        ["Divergências do roteador migrado", "—", "0/67"],
    ]
    add_tabela(slide, MARGEM, Inches(1.35), CONTEUDO_W, Inches(2.0), cabecalho, linhas,
               larguras_rel=[3, 1, 1])

    add_nota(slide, MARGEM, Inches(3.45), CONTEUDO_W, Inches(0.4),
             "A perda de 1 pergunta é uma pergunta de fronteira instável nos dois lados, não regressão.")

    tf = add_caixa_texto(slide, MARGEM, Inches(3.95), CONTEUDO_W, Inches(1.1))
    add_paragrafo(tf, "Migração estrutural, não comportamental: mesmos prompts, "
                       "14 gates (não 11 — contagem corrigida em 2026-09-25; os 2 gates "
                       "do OpenPack de 09-12 não estavam contados originalmente), "
                       "mesmo teto de 1 tentativa.", tamanho=15.5, bullet=True, primeiro=True)

    add_bloco_destaque(
        slide, MARGEM, Inches(5.15), CONTEUDO_W, Inches(1.85),
        "Nota adicionada nesta sessão (2026-09-25) — não estava nos slides originais",
        ["dashboard/roteador_langgraph.py e nl_to_sql/nl_to_sql_langgraph.py estão "
         "CONFIRMADOS em produção desde 2026-09-09",
         "(dashboard/app.py:521, eval/rodar_golden.py:35) — fato verificado nesta "
         "sessão, não estava explícito nos decks originais."],
        cor_borda=ROXO, cor_fundo=RGBColor(238, 227, 240),
    )


def slide_ato2_regua_nao_pega(prs):
    slide = slide_em_branco(prs)
    add_fundo(slide)
    add_titulo(slide, "Ato 2 — O que a régua não pega")

    tf = add_caixa_texto(slide, MARGEM, Inches(1.6), CONTEUDO_W, Inches(2.2))
    add_paragrafo(tf, "Testes automatizados passavam.", tamanho=18, bullet=True, primeiro=True)
    add_paragrafo(tf, "Mas streamlit run app.py quebrava com ModuleNotFoundError — bug de "
                       "import só visível em produção real, por causa de sys.path.",
                  tamanho=18, bullet=True)

    add_bloco_destaque(
        slide, MARGEM, Inches(4.1), CONTEUDO_W, Inches(1.4),
        "Fix",
        ["Carregar módulos por caminho de arquivo explícito (importlib)."],
    )


def slide_ato3_agentic_rag(prs):
    slide = slide_em_branco(prs)
    add_fundo(slide)
    add_titulo(slide, "Ato 3 — Quando a régua diz não: Agentic RAG")

    tf = add_caixa_texto(slide, MARGEM, Inches(1.25), CONTEUDO_W, Inches(1.5))
    add_paragrafo(tf, "Critic node reformulando query (Self-RAG/CRAG) sobre "
                       "RAGHibrido.buscar() — LangGraph com LLM juiz.", tamanho=16,
                  bullet=True, primeiro=True)
    add_paragrafo(tf, "Causa raiz: o documento certo é recuperado, mas o chunk específico "
                       "às vezes não contém a resposta (outra seção do mesmo arquivo) — "
                       "reformular query não resolve problema de chunking.", tamanho=16,
                  bullet=True)

    add_tabela(
        slide, MARGEM, Inches(2.75), CONTEUDO_W, Inches(1.7),
        ["Métrica", "Baseline", "Agentic", "Delta"],
        [
            ["Recall@k (documento)", "93,9%", "91,8%", "-2,0pp"],
            ["MRR (documento)", "0,857", "0,827", "-0,031"],
            ["Recall@k (chunk, 40/49)", "90,0%", "85,0%", "-5,0pp"],
            ["Duração total (49 perguntas)", "81,6s", "1885,2s", "23,09x mais caro"],
        ],
    )

    add_bloco_destaque(
        slide, MARGEM, Inches(4.75), CONTEUDO_W, Inches(1.3),
        "Decisão final (2026-09-25) — ENCERRADO, não promovido",
        ["Reexecução formal com n=49 (eval/avaliar_rag_agentic.py) confirma e agrava o "
         "resultado original de n=15 (Recall@1 12/15→11/15): regride em TODAS as métricas "
         "e custa 23x mais (18/49 = 36,7% das perguntas reformularam query). Duas medições "
         "independentes convergem — não reabrir sem hipótese nova."],
    )

    add_nota(slide, MARGEM, Inches(6.25), CONTEUDO_W, Inches(0.6),
             "Resultado persistido em eval/resultados_rag_agentic.json. "
             "CLAUDE.md atualizado com o veredito final em 2026-09-25.", tamanho=12)


def slide_ato3_contextual_retrieval(prs):
    slide = slide_em_branco(prs)
    add_fundo(slide)
    add_titulo(slide, "Ato 3 — Contextual Retrieval")

    tf = add_caixa_texto(slide, MARGEM, Inches(1.3), CONTEUDO_W, Inches(1.1))
    add_paragrafo(tf, "Prependar resumo LLM a cada chunk.", tamanho=17, bullet=True,
                  primeiro=True)

    cabecalho = ["Métrica", "Antes", "Depois", "Δ"]
    linhas = [
        ["Recall@5", "98,0%", "98,0%", "idêntico"],
        ["Precision@5", "50,0%", "46,4%", "-3,6pp"],
        ["MRR", "0,899", "0,907", "+0,008"],
    ]
    add_tabela(slide, MARGEM, Inches(2.3), CONTEUDO_W, Inches(1.7), cabecalho, linhas,
               larguras_rel=[2, 1, 1, 1])

    add_bloco_destaque(
        slide, MARGEM, Inches(4.35), CONTEUDO_W, Inches(2.1),
        "Causa e decisão",
        ["Manuais do Harbor têm 40-181 linhas, curtos demais para a técnica "
         "(literatura original mede ganho em documentos longos).",
         "Não promovido, mas mantido como flag de ablação."],
    )


def slide_ato3_calibracao_quebrada(prs):
    slide = slide_em_branco(prs)
    add_fundo(slide)
    add_titulo(slide, "Ato 3 — Calibração de top-p estava quebrada")

    tf = add_caixa_texto(slide, MARGEM, Inches(1.5), CONTEUDO_W, Inches(2.3))
    add_paragrafo(tf, "Benchmark multimodal media score=0,0 fixo (busca sem rerank não "
                       "expõe RRF).", tamanho=18, bullet=True, primeiro=True)
    add_paragrafo(tf, "Testar 3 limiares dava resultado idêntico.", tamanho=18, bullet=True)

    add_bloco_destaque(
        slide, MARGEM, Inches(4.1), CONTEUDO_W, Inches(1.6),
        "Diagnóstico correto",
        ["Não era limiar mal calibrado — era ausência de sinal de score real."],
        cor_borda=ROXO, cor_fundo=RGBColor(238, 227, 240),
    )


def slide_ato3_regua_consertada(prs):
    slide = slide_em_branco(prs)
    add_fundo(slide)
    add_titulo(slide, "Ato 3 — Consertada a régua: 0,7 vence")

    cabecalho = ["Configuração", "nDCG@5", "Recall@5", "Tempo"]
    linhas = [
        ["Sem rerank", "0,864", "93%", "38–62s"],
        ["top-p = 0,7  (vencedor)", "0,889", "97%", "~3015s"],
        ["top-p = 0,85", "0,802", "86%", "~6734s"],
        ["top-p = 0,99", "0,780", "86%", "~7055s"],
    ]
    add_tabela(slide, MARGEM, Inches(1.5), CONTEUDO_W, Inches(2.4), cabecalho, linhas,
               larguras_rel=[2, 1, 1, 1])

    add_bloco_destaque(
        slide, MARGEM, Inches(4.3), CONTEUDO_W, Inches(1.4),
        "Resultado",
        ["TOP_P_LIMIAR atualizado de 0,85 (chute nunca validado) para 0,7."],
    )


def slide_ato3_experimento_decisivo(prs):
    slide = slide_em_branco(prs)
    add_fundo(slide)
    add_titulo(slide, "Ato 3 — O experimento decisivo")
    add_subtitulo_slide(slide, "Rerank não compensa captioning ruim")

    cabecalho = ["Configuração", "nDCG@5", "Recall@5", "Tempo"]
    linhas = [
        ["(A) moondream + rerank top-p=0,7", "0,436", "48%", "4779s"],
        ["(B) qwen3-vl sem rerank", "0,864", "93%", "11,8s"],
    ]
    add_tabela(slide, MARGEM, Inches(1.6), CONTEUDO_W, Inches(1.5), cabecalho, linhas,
               larguras_rel=[2.4, 1, 1, 1])

    add_nota(slide, MARGEM, Inches(3.25), CONTEUDO_W, Inches(0.4),
             "(B) é ~400x mais barato E melhor que (A).")

    add_bloco_destaque(
        slide, MARGEM, Inches(3.85), CONTEUDO_W, Inches(2.0),
        "Conclusão",
        ["Se a legenda não descreve o conteúdo real da imagem, o rerank não tem o que "
         "resgatar.",
         "Prioridade deve ser achar VLM bom, não otimizar rerank."],
    )


def slide_ato37_adaptive_k(prs):
    slide = slide_em_branco(prs)
    add_fundo(slide)
    add_titulo(slide, "Ato 3.7 — Adaptive-k: promovido com ressalva")
    add_subtitulo_slide(slide, "eval/rag_gerador.py::adaptive_k() — decisão final 2026-09-25")

    tf = add_caixa_texto(slide, MARGEM, Inches(1.25), CONTEUDO_W, Inches(1.0))
    add_paragrafo(tf, "Corta o nº de chunks passados ao gerador pelo gap de score RRF "
                       "(Adaptive-k, EMNLP 2025, arXiv:2506.08479), em vez de k fixo=3.",
                  tamanho=16, bullet=True, primeiro=True)
    add_paragrafo(tf, "Ganho original (88,5% vs. 76,9% do melhor k fixo) vinha de 1 "
                       "execução só — literatura de 2026 exige N=3 com média±desvio.",
                  tamanho=16, bullet=True)

    add_tabela(
        slide, MARGEM, Inches(2.35), CONTEUDO_W, Inches(1.6),
        ["Execução", "Faithfulness", "Alucinações", "Duração"],
        [
            ["1/3", "76,9%", "1/49", "2881s"],
            ["2/3", "76,9%", "0/49", "1554s"],
            ["3/3", "88,5%", "0/49", "1225s"],
        ],
    )

    add_bloco_destaque(
        slide, MARGEM, Inches(4.15), CONTEUDO_W, Inches(1.55),
        "Resultado: média 0,8077 ± 0,067 (CV=0,0829)",
        ["Média supera o melhor k fixo (0,769) e o pior caso das 3 execuções empata com "
         "ele, nunca fica abaixo. CV acima do alvo estrito (<0,05), mas a direção é "
         "consistente nas 3 execuções — PROMOVIDO COM RESSALVA: usar_adaptive_k=True "
         "passa a ser o default em eval/rag_gerador.py."],
    )

    add_nota(slide, MARGEM, Inches(5.85), CONTEUDO_W, Inches(0.7),
             "Resultado persistido em eval/resultados_variancia_adaptive_k.json. "
             "A variância real (CV=0,083) é maior que a ressalva original antecipava "
             "— reportar sempre com o intervalo, nunca só a média.", tamanho=12)


def slide_ato35_qwen_reprovado(prs):
    slide = slide_em_branco(prs)
    add_fundo(slide)
    add_titulo(slide, "Ato 3.5 — Busca por um VLM de captioning aprovado")
    add_subtitulo_slide(slide, "10–11/09 — sessão anterior ao Ato 3")

    tf = add_caixa_texto(slide, MARGEM, Inches(1.65), CONTEUDO_W, Inches(0.9))
    add_paragrafo(tf, "qwen3-vl:4b testado: melhoria real sobre moondream (42,3% → 77,5% "
                       "de fidelidade), mas REPROVADO pelo critério de promoção "
                       "(≥90% taxa média E 100% sem números inventados).",
                  tamanho=15.5, bullet=True, primeiro=True)

    cabecalho = ["Versão", "Fidelidade", "Sem invent.", "Passou tudo"]
    linhas = [
        ["moondream (baseline)", "42,3%", "—", "0–26"],
        ["qwen3-vl bruto", "66,5%", "6–26", "13–26"],
        ["qwen3-vl corpus completo", "76,4%", "11–26", "15–26"],
        ["qwen3-vl checks corrigidos", "77,5%", "13–26", "17–26"],
    ]
    add_tabela(slide, MARGEM, Inches(2.7), CONTEUDO_W, Inches(2.0), cabecalho, linhas,
               larguras_rel=[2.3, 1, 1, 1])


def slide_ato35_bugs_metodologia(prs):
    slide = slide_em_branco(prs)
    add_fundo(slide)
    add_titulo(slide, "Ato 3.5 — Dois bugs de metodologia corrigidos na régua")

    tf = add_caixa_texto(slide, MARGEM, Inches(1.35), CONTEUDO_W, Inches(2.1))
    add_paragrafo(tf, "Bug 1: regex não reconhecia vírgula decimal pt-BR "
                       "(“0,099” virava dois números falsos).", tamanho=16.5,
                  bullet=True, primeiro=True)
    add_paragrafo(tf, "Bug 2: “5” de “CNC 5 eixos” contado como número inventado.",
                  tamanho=16.5, bullet=True)
    add_paragrafo(tf, "Sem esses fixes, a fidelidade estava subestimada: 66,5% → 77,5% é "
                       "toda atribuível à régua, não ao VLM.", tamanho=16.5, bullet=True)

    add_bloco_destaque(
        slide, MARGEM, Inches(4.5), CONTEUDO_W, Inches(1.7),
        "Achado que sobrevive",
        ["Alucinação real confirmada: VLM reportou voltagem “~0,09V” quando o real "
         "é “~220,2V” (3 ordens de grandeza), em 2 imagens."],
        cor_borda=VERMELHO, cor_fundo=FUNDO_CRITICO,
    )


def slide_ato35_profiling(prs):
    slide = slide_em_branco(prs)
    add_fundo(slide)
    add_titulo(slide, "Ato 3.5 — Profiling revisou o custo do rerank para baixo")

    tf = add_caixa_texto(slide, MARGEM, Inches(1.5), CONTEUDO_W, Inches(2.0))
    add_paragrafo(tf, "Estimativa inicial: 48,76s/imagem — quase causou matar um processo "
                       "saudável por engano.", tamanho=17, bullet=True, primeiro=True)
    add_paragrafo(tf, "Revisada por profiling isolado para ~14,5s/imagem real "
                       "(encode_document).", tamanho=17, bullet=True)

    add_bloco_destaque(
        slide, MARGEM, Inches(3.85), CONTEUDO_W, Inches(1.6),
        "Achado operacional",
        ["Log vazio não é sinônimo de travado — medir delta de CPU real antes de "
         "matar processo longo."],
        cor_borda=ROXO, cor_fundo=RGBColor(238, 227, 240),
    )


def slide_ato35_veredito(prs):
    slide = slide_em_branco(prs)
    add_fundo(slide)
    add_titulo(slide, "Ato 3.5 — Veredito final: nenhum VLM aprovado")

    tf = add_caixa_texto(slide, MARGEM, Inches(1.3), CONTEUDO_W, Inches(1.6))
    add_paragrafo(tf, "qwen3-vl:4b é o melhor testado: 93,6% de fidelidade real "
                       "(18/26 passou tudo, 22/26 sem números inventados).", tamanho=16.5,
                  bullet=True, primeiro=True)
    add_paragrafo(tf, "Abaixo do critério de 90%/100% simultâneo (passar TODOS os checks "
                       "em 100% das imagens) — reprovado por esse critério mais rígido.",
                  tamanho=16.5, bullet=True)

    add_bloco_destaque(
        slide, MARGEM, Inches(3.1), CONTEUDO_W, Inches(1.5),
        "Reconciliado: o número inicial (77,5%) estava subestimado por bug",
        ["eval/checks_fidelidade_caption.py capturava len(resultados) DEPOIS de inserir "
         "campos derivados no dict, dividindo por 7 chaves em vez de 6 e subestimando a "
         "taxa mesmo quando os 6 checks reais passavam (6/7=85,7% em vez de 6/6=100%). "
         "Corrigido; 93,6% é o número final e definitivo — ver CLAUDE.md."],
        cor_borda=ROXO, cor_fundo=RGBColor(238, 227, 240),
    )

    add_bloco_destaque(
        slide, MARGEM, Inches(4.85), CONTEUDO_W, Inches(1.35),
        "Decisão final",
        ["Abandonar a via VLM para captioning de gráfico técnico."],
    )


def slide_ato36_deterministico(prs):
    slide = slide_em_branco(prs)
    add_fundo(slide)
    add_titulo(slide, "Ato 3.6 — Decisão final: geração determinística vence")
    add_subtitulo_slide(slide, "14–15/09 — EM PRODUÇÃO")

    tf = add_caixa_texto(slide, MARGEM, Inches(1.55), CONTEUDO_W, Inches(0.9))
    add_paragrafo(tf, "Em vez de VLM lendo a imagem, template gera a legenda do dado "
                       "tabular de origem (mesmo DataFrame que desenhou o gráfico) — "
                       "rag/legendas_deterministicas.py.", tamanho=15, bullet=True, primeiro=True)

    cabecalho = ["Critério", "VLM", "Determinístico"]
    linhas = [
        ["Taxa de aprovação", "93,6%", "100%"],
        ["Passou tudo", "18/26", "26/26"],
        ["Sem números inventados", "22/26", "26/26"],
        ["Custo", "412s/imagem (~3h/corpus)", "segundos"],
        ["Depende de Ollama", "Sim", "Não"],
    ]
    add_tabela(slide, MARGEM, Inches(2.55), CONTEUDO_W, Inches(2.3), cabecalho, linhas,
               larguras_rel=[2, 1.4, 1.4])

    add_nota(slide, MARGEM, Inches(5.0), CONTEUDO_W, Inches(0.5),
             "Retrieval empata dentro do ruído (Recall@3/MRR): 93%/0,724 (VLM) vs. "
             "90%/0,747 (determinístico).")


def slide_ato36_validacao_externa(prs):
    slide = slide_em_branco(prs)
    add_fundo(slide)
    add_titulo(slide, "Ato 3.6 — Validação externa")
    add_subtitulo_slide(slide, "3 referências de 2026, buscadas DEPOIS da decisão")

    tf = add_caixa_texto(slide, MARGEM, Inches(1.7), CONTEUDO_W, Inches(2.6))
    add_paragrafo(tf, "ChartFI (arXiv:2605.23694) — “table-grounded methods generally "
                       "demonstrate higher faithfulness rates”.", tamanho=15.5,
                  bullet=True, primeiro=True)
    add_paragrafo(tf, "Chart-to-Text/VisText — “image-only models perform the worst, "
                       "showing high confidence in inaccurate captions”.", tamanho=15.5,
                  bullet=True)
    add_paragrafo(tf, "arXiv:2312.10160 — 82,06% de erro factual em legendas de LVLM "
                       "(GPT-4V incluso).", tamanho=15.5, bullet=True)
    add_paragrafo(tf, "As três confirmam a escolha de forma independente.", tamanho=15.5,
                  bold=True, bullet=True)

    add_bloco_destaque(
        slide, MARGEM, Inches(5.0), CONTEUDO_W, Inches(1.9),
        "Contribuição candidata",
        ["A taxonomia 2026 de RAG multimodal (arXiv:2510.15253) nomeia 3 arquiteturas "
         "dominantes (caption-and-index, unified vision embeddings, page-as-image/late "
         "interaction) — a variante do Harbor (legenda do dado de origem, nunca do "
         "pixel) não é nenhuma delas. Fidelidade vira invariante estrutural garantido "
         "pela construção, não métrica que se torce testando mais modelos."],
        cor_borda=ROXO, cor_fundo=RGBColor(238, 227, 240),
    )


def slide_ato4(prs):
    slide = slide_em_branco(prs)
    add_fundo(slide)
    add_titulo(slide, "Ato 4 — Escolhas que a medição sustentou")

    top = Inches(1.35)
    altura = Inches(1.75)
    espaco = Inches(0.15)

    add_bloco_destaque(
        slide, MARGEM, top, CONTEUDO_W, altura,
        "TF-IDF vs. BM25 (agosto)",
        ["Empate no pipeline completo: 98%/44%/0,866 vs. 98%/43%/0,866.",
         "BM25 perde no lexical isolado (94% vs. 98%). Mantido TF-IDF."],
        cor_borda=MAGENTA, cor_fundo=RGBColor(250, 235, 243), corpo_tam=13,
    )

    top2 = top + altura + espaco
    add_bloco_destaque(
        slide, MARGEM, top2, CONTEUDO_W, altura,
        "Python puro vs. LangChain no RAG (setembro)",
        ["Recall@5/MRR idênticos; Precision@5 sobe 11,4pp (38,4%→49,8%) — mas o "
         "ganho vem do algoritmo BM25+RRF, não do framework.",
         "Migrado assim mesmo, por consistência com LangGraph nos módulos seguintes "
         "(razão declarada, não disfarçada de ganho técnico). Implementação usada: "
         "a que combina framework + BM25+RRF juntos."],
        cor_borda=MAGENTA, cor_fundo=RGBColor(250, 235, 243), corpo_tam=13,
    )

    top3 = top2 + altura + espaco
    add_bloco_destaque(
        slide, MARGEM, top3, CONTEUDO_W, altura,
        "qwen2.5:7b vs. DeepSeek-R1:7b",
        ["2/2 sucessos em 112,6s vs. 1/2 em 258,2s (2,3x mais lento, 3x mais tokens).",
         "Mantido qwen2.5:7b."],
        cor_borda=MAGENTA, cor_fundo=RGBColor(250, 235, 243), corpo_tam=13,
    )


def slide_ato5(prs):
    slide = slide_em_branco(prs)
    add_fundo(slide)
    add_titulo(slide, "Ato 5 — Até onde a régua alcança")
    add_subtitulo_slide(slide, "Mapa de métricas CERISE")

    cabecalho = ["Categoria", "Estado", "O que temos"]
    linhas = [
        ["Recuperação", "Completa", "Recall@k / Precision@k / MRR / nDCG@k"],
        ["LLM", "Parcial", "Faithfulness / Hallucination Rate"],
        ["Sistema", "Parcial", "Tokens/latência só nos agentes de teste"],
        ["Ferramentas", "LACUNA", "Sem SQL/Tool Selection Accuracy formal"],
        ["Industrial", "LACUNA", "Sem Diagnosis/Root Cause Accuracy"],
    ]
    add_tabela(slide, MARGEM, Inches(1.35), CONTEUDO_W, Inches(2.4), cabecalho, linhas,
               larguras_rel=[1.3, 1, 3.2])

    add_nota(slide, MARGEM, Inches(3.9), CONTEUDO_W, Inches(0.5),
             "Falta nomeado: Answer Relevancy e BERTScore (LLM); latência no dashboard de "
             "PRODUÇÃO, não só nos agentes de teste (Sistema).")

    add_bloco_destaque(
        slide, MARGEM, Inches(4.5), CONTEUDO_W, Inches(2.55),
        "2 lacunas novas encontradas nesta sessão (2026-09-25) — não estavam no deck original",
        ["(1) Métricas de falha de planejamento no NL-to-SQL (taxonomia: tool inválida / "
         "parâmetros inválidos / valores de parâmetro errados — esta última é a classe "
         "do bug de case-sensitive em coluna maiúscula já visto no Harbor); testado nesta "
         "sessão: llama3.2:1b teve 0/20 falhas de formato JSON, mas 9/20 (45%) de erro de "
         "CLASSIFICAÇÃO de rota — taxa alta que vale monitorar.",
         "(2) O chat roteia cada pergunta isoladamente, sem memória de turno — não "
         "suporta perguntas de follow-up; gap identificado via leitura do whitepaper "
         "“Agents” do Google (2025), que define memória de sessão como traço "
         "distintivo de agente vs. modelo."],
        corpo_tam=12.5,
    )


def slide_ato6(prs):
    slide = slide_em_branco(prs)
    add_fundo(slide)
    add_titulo(slide, "Ato 6 — OpenPack: a régua testada num dado real")
    add_subtitulo_slide(slide, "8º dataset")

    tf = add_caixa_texto(slide, MARGEM, Inches(1.6), CONTEUDO_W, Inches(1.9))
    add_paragrafo(tf, "Até 13/09 toda medição RAG multimodal usava dado sintético "
                       "(26 gráficos gerados pelo próprio time).", tamanho=16.5,
                  bullet=True, primeiro=True)
    add_paragrafo(tf, "Integração do OpenPack (dataset público, operações de "
                       "embalagem logística, sensores IMU vestíveis, 102 sessões reais) "
                       "aplicando RAG-HAR (arXiv:2512.08984) — estatísticas de sensor "
                       "viram texto por template, sem modelo de visão, sem treino.",
                  tamanho=16.5, bullet=True)

    add_bloco_destaque(
        slide, MARGEM, Inches(3.9), CONTEUDO_W, Inches(2.3),
        "Achado",
        ["Recall@5 deu 0% inicialmente — não era bug: RAG-HAR busca vizinhos da mesma "
         "classe (retrieval por categoria), não o documento exato (retrieval por "
         "instância).",
         "Métrica corrigida: k-NN label purity = 12,9% contra 10% de acaso — sinal "
         "real, mas fraco."],
    )


def slide_ato7_correcao_critica(prs):
    slide = slide_em_branco(prs)
    add_fundo(slide, RGBColor(255, 245, 243))

    # Barra lateral vermelha de destaque forte
    barra = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.28), SLIDE_H)
    barra.fill.solid()
    barra.fill.fore_color.rgb = VERMELHO
    barra.line.fill.background()
    set_shape_sem_sombra(barra)

    faixa_titulo = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.28), 0, SLIDE_W - Inches(0.28), Inches(1.15))
    faixa_titulo.fill.solid()
    faixa_titulo.fill.fore_color.rgb = FUNDO_CRITICO
    faixa_titulo.line.fill.background()
    set_shape_sem_sombra(faixa_titulo)

    tf_t = faixa_titulo.text_frame
    tf_t.word_wrap = True
    tf_t.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf_t.margin_left = Inches(0.3)
    p = tf_t.paragraphs[0]
    run = p.add_run()
    run.text = "CORRIGIDO em 2026-09-15 — o F1-macro do OpenPack estava inflado"
    run.font.size = Pt(24)
    run.font.bold = True
    run.font.name = FONTE
    run.font.color.rgb = VERMELHO

    tf = add_caixa_texto(slide, Inches(0.7), Inches(1.35), CONTEUDO_W, Inches(1.5))
    add_paragrafo(tf, "O F1-macro originalmente reportado (0,91–0,92) vinha de um "
                       "VAZAMENTO DE RÓTULO: o texto indexado citava sujeito/sessão/"
                       "operação/classe na primeira linha, fazendo o k-NN casar rótulo "
                       "com rótulo (BM25 lexical) em vez de sinal de sensor real.",
                  tamanho=14.5, bullet=True, primeiro=True)
    add_paragrafo(tf, "Corrigido em rag/rag_openpack_texto.py::montar_texto_janela() — "
                       "identificação agora só em metadado, nunca no texto embeddado.",
                  tamanho=14.5, bullet=True)

    cabecalho = ["Modelo", "F1-macro", "Treino"]
    linhas = [
        ["RAG-HAR/Harbor (amostra, 26 janelas)", "0,0750", "Não (training-free)"],
        ["RAG-HAR/Harbor (completo, 2.591 janelas)", "0,1664", "Não (training-free)"],
        ["RAG-HAR/Harbor (LOSO, 21 sujeitos)", "0,1626 ± 0,0686", "Não (training-free)"],
        ["UNet (oficial)", "0,3451", "Sim, supervisionado"],
        ["ST-GCN (oficial)", "0,7024", "Sim, supervisionado"],
        ["DeepConvLSTM (oficial)", "0,7081", "Sim, supervisionado"],
    ]
    add_tabela(slide, Inches(0.7), Inches(2.95), CONTEUDO_W - Inches(0.15), Inches(2.15),
               cabecalho, linhas, larguras_rel=[3, 1.3, 1.6], tam_cabecalho=12.5, tam_corpo=12)

    add_nota(slide, Inches(0.7), Inches(5.2), CONTEUDO_W, Inches(0.55),
             "O método training-free NÃO supera os baselines supervisionados neste "
             "dataset — resultado honesto, não fracasso. LOSO (protocolo padrão-ouro "
             "cross-subject em HAR) é consistente com o teste completo, confirmando que "
             "não é artefato de split favorável.", cor=QUASE_PRETO, tamanho=12.5)

    add_bloco_destaque(
        slide, Inches(0.7), Inches(5.85), CONTEUDO_W - Inches(0.15), Inches(1.15),
        "A CONTRIBUIÇÃO REAL desta sessão",
        ["3 controles de sanidade validam a correção: ausência léxica de identificação "
         "(0/4000 docs), rótulo embaralhado (F1=0,0182 < acaso 0,10), LOSO sem "
         "F1=1,0000 exato. É a DETECÇÃO e CORREÇÃO do vazamento, não o F1-macro em "
         "si, que é defensável como contribuição metodológica."],
        cor_borda=VERMELHO, cor_fundo=RGBColor(255, 226, 219), corpo_tam=11.5, titulo_tam=13,
    )


def slide_ato7_custo(prs):
    slide = slide_em_branco(prs)
    add_fundo(slide)
    add_titulo(slide, "Ato 7 — Custo (não afetado pelo bug)")

    tf = add_caixa_texto(slide, MARGEM, Inches(1.7), CONTEUDO_W, Inches(2.5))
    add_paragrafo(tf, "Indexação: ~50–72s", tamanho=19, bullet=True, primeiro=True)
    add_paragrafo(tf, "Busca: ~0,5s/janela", tamanho=19, bullet=True)
    add_paragrafo(tf, "Teste completo (2.591 janelas): ~1360s", tamanho=19, bullet=True)

    add_nota(slide, MARGEM, Inches(4.4), CONTEUDO_W, Inches(0.6),
             "Esses números de custo continuam válidos — não foram afetados pela "
             "correção do vazamento de rótulo (Ato 7).")


def slide_fechamento(prs):
    slide = slide_em_branco(prs)
    add_fundo(slide)
    add_titulo(slide, "Fechamento — O que fica")

    add_bloco_destaque(
        slide, MARGEM, Inches(1.15), CONTEUDO_W, Inches(1.9),
        "Sim (promovido)",
        ["Migração do self-repair e do roteador para LangGraph (EM PRODUÇÃO desde "
         "2026-09-09); top-p=0,7 como limiar de rerank multimodal; geração "
         "determinística de legendas (EM PRODUÇÃO desde 14/09, 100% fidelidade); "
         "OpenPack como 8º dataset no chat/Streamlit sem regressão de roteamento "
         "(55/67→59/69, reconfirmado pelo harness em 2026-09-25 após a promoção do "
         "Adaptive-k — dois slides originais citavam 58/69 e 59/69; 59/69 é o número "
         "correto e atual); "
         "Adaptive-k como default do gerador RAG (2026-09-25, N=3 execuções, "
         "faithfulness médio 0,8077±0,067, PROMOVIDO COM RESSALVA — ver Ato 3.7)."],
        cor_borda=MAGENTA, cor_fundo=RGBColor(250, 235, 243), corpo_tam=12,
    )

    add_bloco_destaque(
        slide, MARGEM, Inches(3.15), CONTEUDO_W, Inches(1.4),
        "Não (medido e não promovido)",
        ["Agentic RAG (2026-09-25, ENCERRADO: n=49 confirma regressão em todas as "
         "métricas e 23x mais custo — não reabrir sem hipótese nova); Contextual "
         "Retrieval; rerank multimodal como forma de compensar captioning ruim; "
         "qualquer VLM para captioning de gráfico técnico (decisão fechada, sem "
         "critério de reabertura)."],
        cor_borda=CORAL, cor_fundo=FUNDO_ALERTA, corpo_tam=12,
    )

    add_bloco_destaque(
        slide, MARGEM, Inches(4.55), CONTEUDO_W, Inches(1.0),
        "Corrigido",
        ["F1-macro do OpenPack (vazamento de rótulo, ver Ato 7); calibração de top-p "
         "(score=0,0 no 1º estágio, corrigido)."],
        cor_borda=ROXO, cor_fundo=RGBColor(238, 227, 240), corpo_tam=12.5,
    )

    caixa = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, MARGEM, Inches(5.7), CONTEUDO_W, Inches(1.0))
    caixa.fill.solid()
    caixa.fill.fore_color.rgb = ROXO
    caixa.line.fill.background()
    set_shape_sem_sombra(caixa)
    tf = caixa.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.25)
    tf.margin_right = Inches(0.25)
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = ("Boa parte dos resultados medidos nesta narrativa foi negativa — e isso é "
                "o argumento mais forte deste documento. Uma régua que só confirma o que "
                "a gente já queria não é régua.")
    run.font.size = Pt(15)
    run.font.italic = True
    run.font.bold = True
    run.font.name = FONTE
    run.font.color.rgb = BRANCO


def slide_proximos_passos(prs):
    slide = slide_em_branco(prs)
    add_fundo(slide)
    add_titulo(slide, "Próximos passos")

    tf = add_caixa_texto(slide, MARGEM, Inches(1.5), CONTEUDO_W, Inches(4.5))
    add_paragrafo(tf, "CONCLUÍDO (2026-09-25): variância do Adaptive-k, N=3 execuções — "
                       "faithfulness 0,8077±0,067 (CV=0,0829), promovido com ressalva",
                  tamanho=18, bullet=True, primeiro=True)
    add_paragrafo(tf, "CONCLUÍDO (2026-09-25): veredito do rag_agentic com n=49 — "
                       "regrediu em tudo, 23x mais custo, frente encerrada",
                  tamanho=18, bullet=True)
    add_paragrafo(tf, "Answer Relevancy como métrica formal", tamanho=18, bullet=True)
    add_paragrafo(tf, "Latência no dashboard de produção", tamanho=18, bullet=True)
    add_paragrafo(tf, "RGB do OpenPack (aprovação institucional pendente, sem prazo)",
                  tamanho=18, bullet=True)


# ---------------------------------------------------------------------------
# Montagem do deck
# ---------------------------------------------------------------------------

def construir_deck():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    slide_titulo(prs)
    slide_ato0(prs)
    slide_ato1(prs)
    slide_ato2_langgraph(prs)
    slide_ato2_regua_nao_pega(prs)
    slide_ato3_agentic_rag(prs)
    slide_ato3_contextual_retrieval(prs)
    slide_ato3_calibracao_quebrada(prs)
    slide_ato3_regua_consertada(prs)
    slide_ato3_experimento_decisivo(prs)
    slide_ato37_adaptive_k(prs)
    slide_ato35_qwen_reprovado(prs)
    slide_ato35_bugs_metodologia(prs)
    slide_ato35_profiling(prs)
    slide_ato35_veredito(prs)
    slide_ato36_deterministico(prs)
    slide_ato36_validacao_externa(prs)
    slide_ato4(prs)
    slide_ato5(prs)
    slide_ato6(prs)
    slide_ato7_correcao_critica(prs)
    slide_ato7_custo(prs)
    slide_fechamento(prs)
    slide_proximos_passos(prs)

    return prs


if __name__ == "__main__":
    apresentacao = construir_deck()
    diretorio = os.path.dirname(os.path.abspath(__file__))
    caminho_saida = os.path.join(diretorio, "harbor_consolidado_2026-09.pptx")
    apresentacao.save(caminho_saida)
    print(f"Deck salvo em: {caminho_saida}")
    print(f"Total de slides: {len(apresentacao.slides)}")
