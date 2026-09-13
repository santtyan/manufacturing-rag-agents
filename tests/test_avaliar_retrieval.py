"""Testes de eval/avaliar_retrieval.py -- especificamente as funcoes novas de chunk-level
(2026-09-14, investigacao do resultado negativo do Agentic RAG). Cada teste cobre um achado
real, mesmo padrao ja usado em tests/test_checks_fidelidade_caption.py.

Rodar via: python -m pytest tests/test_avaliar_retrieval.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from avaliar_retrieval import secao_do_chunk, secao_esperada


def test_secao_esperada_extrai_numero_da_nota():
    """Achado real: 41/49 perguntas de rota rag ja citam o numero da secao esperada na nota
    (ex. "Manual (secao 5): ..."), escrito por quem criou a pergunta ao verificar contra o
    manual -- nao precisa reanotar o golden set para medir chunk-level."""
    pergunta = {"nota": "Manual (secao 5): threshold de vibracao e 1.5."}
    assert secao_esperada(pergunta) == 5


def test_secao_esperada_pega_a_primeira_ocorrencia_em_nota_com_duas_secoes():
    """Achado real: armadilhas deliberadas citam DUAS secoes na mesma nota (a errada que o
    sistema poderia confundir, e a certa) -- ex. "85C e o threshold de superaquecimento
    (secao 2), mas a escala de CRITICIDADE (secao 8) usa 90C". A extracao pega a primeira
    ocorrencia por design simples; perguntas assim precisam de checagem manual antes de
    confiar cegamente no numero extraido para chunk-level (ver nota do modulo)."""
    pergunta = {"nota": "85C e o threshold (secao 2), mas a escala usa 90C (secao 8)."}
    assert secao_esperada(pergunta) == 2


def test_secao_esperada_retorna_none_sem_numero_de_secao_na_nota():
    """Perguntas sem 'secao N' explicito na nota (ex. so descricao de conteudo, sem
    referencia estrutural) nao entram na metrica de chunk-level -- retorna None, nunca um
    numero inventado."""
    pergunta = {"nota": "Manual: niveis acima de 1.5 sao considerados anomalos."}
    assert secao_esperada(pergunta) is None


def test_secao_esperada_retorna_none_sem_campo_nota():
    assert secao_esperada({}) is None


def test_secao_do_chunk_extrai_numero_do_cabecalho_preservado():
    """rag_hibrido.py::chunk_texto() preserva o cabecalho '## N. Titulo' dentro do proprio
    texto do chunk -- e o que permite comparar contra secao_esperada() sem precisar de um
    campo extra no candidato retornado por buscar()."""
    texto = "## 2. Sensor de Temperatura (Temperature_C)\n\nA temperatura normal..."
    assert secao_do_chunk(texto) == 2


def test_secao_do_chunk_retorna_none_sem_cabecalho_no_inicio():
    """Chunk sem cabecalho '## N.' no comeco (ex. veio do fallback _chunk_por_linhas sem
    cabecalho proprio, ou o texto foi cortado antes do cabecalho) retorna None, nunca um
    numero de outra parte do texto por acidente."""
    texto = "continuacao do texto sem cabecalho no topo, secao 99 mencionada no meio"
    assert secao_do_chunk(texto) is None


def test_secao_do_chunk_ignora_numero_de_secao_fora_do_padrao_cabecalho():
    """Regex exige '## ' no inicio de linha -- um numero solto no corpo do texto (ex. 'ver
    secao 3 para mais detalhes') nao deve ser confundido com o cabecalho da propria secao."""
    texto = "## 5. Titulo\n\nTexto que menciona ver secao 3 para mais detalhes."
    assert secao_do_chunk(texto) == 5
