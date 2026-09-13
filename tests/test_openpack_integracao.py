"""Testes de integracao do pipeline OpenPack (Fase 7g do plano de integracao, 2026-09-14) --
cobrem o fluxo completo em escala pequena (sample U0209), sem depender de Ollama nem de
reindexar o corpus completo de 4.000 documentos.

Rodar via: python -m pytest tests/test_openpack_integracao.py -v
"""
import json

from pipelines.pipeline8_openpack import carregar_sessao, segmentar_janelas
from rag.rag_openpack_texto import gerar_corpus_rag
from dashboard.roteador import (
    pede_cruzamento_openpack_x_maquina,
    pede_identificacao_de_pessoa,
    rotear_pergunta,
)
from eval.avaliar_benchmark_multimodal_2x2 import carregar_corpus_openpack


def test_pipeline_completo_sample_gera_corpus_indexavel():
    """Roda carregar_sessao -> segmentar_janelas -> gerar_corpus_rag sobre o sample U0209 e
    confere que o resultado tem o schema EXATO esperado por
    RAGHibrido.indexar(documentos_customizados=...): lista de {"id","texto","fonte"}."""
    df_sinais = carregar_sessao("U0209", "S0500", usar_sample=True)
    df_janelas = segmentar_janelas(df_sinais, freq_hz=33.33)
    assert len(df_janelas) > 0, "sample U0209 deveria gerar pelo menos 1 janela valida"

    sinais_por_sessao = {("U0209", "S0500"): df_sinais}
    # Usa so as 5 primeiras janelas para o teste rodar rapido (nao precisa das 533 completas)
    documentos = _gerar_sem_gravar(sinais_por_sessao, df_janelas.head(5))

    assert len(documentos) > 0
    for doc in documentos:
        assert set(doc.keys()) == {"id", "texto", "fonte"}
        assert isinstance(doc["texto"], str) and len(doc["texto"]) > 0
        assert doc["id"].startswith(doc["fonte"])


def _gerar_sem_gravar(sinais_por_sessao, df_janelas):
    """gerar_corpus_rag() sempre grava em disco (caminho_saida tem default) -- usa um arquivo
    temporario para nao sujar rag/corpus_openpack_janelas.json real durante o teste."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        caminho_tmp = Path(tmp) / "corpus_teste.json"
        documentos = gerar_corpus_rag(sinais_por_sessao, df_janelas, caminho_saida=caminho_tmp)
        assert caminho_tmp.exists()
    return documentos


def test_gates_openpack_nao_regridem_roteamento_existente():
    """Mesmo padrao de test_roteador_langgraph.py::test_equivalencia_completa_golden_set_sem_llm:
    roda os 2 gates novos do OpenPack contra eval/golden_questions.json inteiro e confere que
    NENHUMA pergunta pre-existente muda de rota -- formaliza como teste a verificacao manual ja
    feita na sessao de integracao (roteamento 58/69, zero regressao)."""
    from pathlib import Path
    golden = json.loads(
        (Path(__file__).resolve().parent.parent / "eval" / "golden_questions.json").read_text(encoding="utf-8")
    )
    falsos_positivos_openpack = []
    for pq in golden["perguntas"]:
        pergunta = pq["pergunta"]
        # As perguntas openpack-* SAO o alvo dos gates -- nao contam como regressao.
        if pq["id"].startswith("openpack-"):
            continue
        if pede_cruzamento_openpack_x_maquina(pergunta) or pede_identificacao_de_pessoa(pergunta):
            falsos_positivos_openpack.append(pq["id"])

    assert falsos_positivos_openpack == [], (
        f"Gates do OpenPack disparando em perguntas pre-existentes (falso positivo): {falsos_positivos_openpack}"
    )


def test_gates_openpack_disparam_nas_perguntas_que_os_motivaram():
    """Contraprova do teste acima -- as 2 perguntas adicionadas ao golden set especificamente
    para exercitar os gates novos devem, de fato, disparar a rota esperada."""
    golden = json.loads(
        (__import__("pathlib").Path(__file__).resolve().parent.parent / "eval" / "golden_questions.json")
        .read_text(encoding="utf-8")
    )
    perguntas_por_id = {pq["id"]: pq for pq in golden["perguntas"]}

    pq_cruzamento = perguntas_por_id["openpack-cruzamento-sensor-maquina"]
    assert rotear_pergunta(pq_cruzamento["pergunta"], usar_llm=False) == "nao_respondivel_openpack"

    pq_identificacao = perguntas_por_id["openpack-identificacao-sujeito"]
    assert rotear_pergunta(pq_identificacao["pergunta"], usar_llm=False) == "recusa_identificacao_pessoa"


def test_carregar_corpus_openpack_contrato_tupla():
    """carregar_corpus_openpack() deve devolver exatamente a tupla-contrato
    (corpus_docs, imagens_por_id, perguntas, alvos), com imagens_por_id VAZIO (corpus IMU-puro,
    sem imagem) -- garante que main() do benchmark 2x2 pode pular o estagio 2 sem estourar em
    Image.open() com um Path inexistente."""
    resultado = carregar_corpus_openpack()
    assert resultado is not None, (
        "carregar_corpus_openpack() retornou None -- verificar se "
        "rag/corpus_openpack_janelas.json e eval/golden_questions_openpack.json existem"
    )

    corpus_docs, imagens_por_id, perguntas, alvos = resultado
    assert isinstance(corpus_docs, list) and len(corpus_docs) > 0
    assert imagens_por_id == {}
    assert len(perguntas) == len(alvos)
    assert len(perguntas) > 0
