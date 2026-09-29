"""
Bateria de testes de sanidade (item T2 do plano de 2026-09-15) -- generaliza para TODOS os
harnesses de classificacao/retrieval do Harbor o teste que expos o bug de vazamento de rotulo no
OpenPack (ver memoria bug_vazamento_rotulo_openpack_2026-09-15).

MOTIVACAO: nenhum harness do projeto tinha controle negativo antes desta suite. O vazamento de
rotulo em rag_openpack_texto.py sobreviveu porque nada testava "o resultado desaba quando nao
deveria haver sinal". Esses testes sao o padrao-ouro publicado (data randomization test, Zhang
et al. 2016; Sanity Checks for Saliency Maps, NeurIPS 2018) aplicado sistematicamente.

4 baterias (ver item T2 do plano):
1. Rotulo embaralhado (classificacao) -- metrica deve desabar para o acaso.
2. Corpus embaralhado (retrieval) -- Recall@k deve cair para k/N quando a query nao corresponde
   a nenhum documento real.
3. Controle positivo -- um caso trivial que DEVE passar com score maximo; se nao passar, o
   harness em si esta quebrado (nao o metodo avaliado).
4. Ausencia lexica do rotulo no corpus indexado -- teste direto, sem precisar rodar retrieval.

Rodar via: python -m pytest tests/test_sanidade_harnesses.py -v
Ou standalone (pytest indisponivel na maquina por instabilidade de rede -- ver CLAUDE.md):
    python tests/test_sanidade_harnesses.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import datasets  # noqa: F401 (import de protecao contra access violation torch x pyarrow)

HARBOR_ROOT = Path(__file__).resolve().parent.parent
CORPUS_OPENPACK = HARBOR_ROOT / "rag" / "corpus_openpack_janelas.json"
CORPUS_MULTIMODAL = HARBOR_ROOT / "rag" / "legendas_deterministicas.json"

# 10 operacoes do OpenPack (pipelines/pipeline8_openpack.py::CLASSES_OPERACAO) -- usado para
# a checagem de ausencia lexica, nao reimportado do pipeline para o teste ficar independente
# de estado do pipeline (o teste deve continuar valido mesmo se o pipeline mudar de nomenclatura
# -- se isso acontecer e o teste falhar, e sinal de que a lista abaixo precisa ser atualizada).
OPERACOES_OPENPACK = [
    "Picking", "Relocate Item Label", "Assemble Box", "Insert Items", "Close Box",
    "Attach Box Label", "Scan Label", "Attach Shipping Label", "Put on Back Table",
    "Fill out Order",
]


# ---------------------------------------------------------------------------------------------
# Bateria 4: ausencia lexica do rotulo no corpus indexado (mais barata, roda primeiro)
# ---------------------------------------------------------------------------------------------

def test_corpus_openpack_nao_contem_rotulo_de_operacao():
    """CORRIGE o bug de 2026-09-15: nenhum documento do corpus indexado (o que entra no
    embedding/BM25) pode conter o nome de uma operacao nem 'classe <N>' -- essas informacoes
    vivem em metadados, nao no texto. Roda em segundos, e o teste mais barato desta suite;
    teria pego o bug original no primeiro dia se existisse antes."""
    docs = json.loads(CORPUS_OPENPACK.read_text(encoding="utf-8"))
    assert len(docs) > 0, "corpus vazio -- rodar rag/rag_openpack_texto.py primeiro"

    vazamentos = []
    for doc in docs:
        texto = doc["texto"]
        for operacao in OPERACOES_OPENPACK:
            if operacao in texto:
                vazamentos.append((doc["id"], operacao))
        if "classe " in texto.lower():
            vazamentos.append((doc["id"], "classe <N>"))

    assert not vazamentos, (
        f"VAZAMENTO DE ROTULO DETECTADO em {len(vazamentos)} ocorrencias -- "
        f"exemplos: {vazamentos[:5]}. Ver rag/rag_openpack_texto.py::montar_texto_janela()."
    )


# ---------------------------------------------------------------------------------------------
# Bateria 1: rotulo embaralhado -- a metrica de CLASSIFICACAO deve desabar para o acaso
# ---------------------------------------------------------------------------------------------

def test_classificacao_openpack_com_rotulo_embaralhado_desaba_para_acaso():
    """Reexecuta o teste de sanidade formal (eval/teste_sanidade_rotulo_embaralhado.py) como
    parte da suite automatizada, para nao depender de alguem lembrar de rodar o script solto.
    Sobe o mesmo indice ja persistido (nao reindexar aqui -- caro; assume que
    rag/rag_openpack_texto.py ja rodou)."""
    sys.path.insert(0, str(HARBOR_ROOT / "eval"))
    from avaliar_classificacao_openpack import (
        K_DEFAULT, SPLIT_TREINO, SPLIT_TESTE, classificar_por_vizinhos,
        montar_texto_pergunta_para_janela, JANELAS_AMOSTRADAS, CHROMA_DIR, COLECAO,
    )
    from sklearn.metrics import f1_score

    if not CHROMA_DIR.exists():
        print("SKIP: indice OpenPack nao existe, rodar rag/rag_openpack_texto.py primeiro")
        return

    sys.path.insert(0, str(HARBOR_ROOT / "rag"))
    from rag_hibrido_langchain import RAGHibrido

    df_janelas = pd.read_csv(JANELAS_AMOSTRADAS)
    n_classes = df_janelas["operacao"].nunique()
    acaso_esperado = 1.0 / n_classes

    random.seed(42)
    janela_ids = list(df_janelas["janela_id"])
    rotulos_reais = list(df_janelas["operacao"])
    rotulos_embaralhados = rotulos_reais.copy()
    random.shuffle(rotulos_embaralhados)
    operacao_embaralhada = dict(zip(janela_ids, rotulos_embaralhados))

    janelas_treino = df_janelas[df_janelas[["sujeito", "sessao"]].apply(tuple, axis=1).isin(SPLIT_TREINO)]
    janelas_teste = df_janelas[df_janelas[["sujeito", "sessao"]].apply(tuple, axis=1).isin(SPLIT_TESTE)]
    janelas_treino_ids = set(janelas_treino["janela_id"])

    corpus_docs = json.loads(CORPUS_OPENPACK.read_text(encoding="utf-8"))
    rag = RAGHibrido(chroma_dir=CHROMA_DIR, colecao=COLECAO)
    rag.indexar(forcar=False, documentos_customizados=corpus_docs)

    y_true, y_pred = [], []
    for _, janela in janelas_teste.iterrows():
        texto_query = montar_texto_pergunta_para_janela(janela["janela_id"], df_janelas)
        if texto_query is None:
            continue
        predicao = classificar_por_vizinhos(rag, texto_query, operacao_embaralhada, janelas_treino_ids, k=K_DEFAULT)
        if predicao is None:
            continue
        y_true.append(operacao_embaralhada[janela["janela_id"]])
        y_pred.append(predicao)

    assert y_true, "nenhuma janela classificada -- verificar indice/split"
    f1_embaralhado = f1_score(y_true, y_pred, average="macro", zero_division=0)

    assert f1_embaralhado <= acaso_esperado * 2, (
        f"F1 com rotulo embaralhado = {f1_embaralhado:.4f}, muito acima do acaso "
        f"(~{acaso_esperado:.4f}) -- AINDA HA VAZAMENTO por algum caminho alem do texto "
        f"indexado. Nao promover nenhum numero de F1-macro ate investigar."
    )


def test_indice_numerico_openpack_com_rotulo_embaralhado_desaba_para_acaso():
    """Mesma bateria acima (item 1, rotulo embaralhado), aplicada ao indice numerico
    (eval/indice_numerico_openpack.py, plano 'OpenPack: corrigir o indice antes de trocar de
    modalidade', 2026-09-28). Risco de vazamento aqui e conceitualmente diferente do bug de
    2026-09-15: nao ha texto embeddado citando o rotulo, mas o vetor kNN ainda usa
    operacoes_treino diretamente da coluna 'operacao' -- se o pareamento janela<->rotulo
    estivesse errado em algum ponto do pipeline, o F1 embaralhado exporia isso do mesmo jeito
    (o F1 real desabaria se o rotulo fosse aleatorio, entao o teste so faz sentido comparando
    o F1 com rotulo real vs. embaralhado)."""
    sys.path.insert(0, str(HARBOR_ROOT / "eval"))
    from indice_numerico_openpack import (
        JANELAS_AMOSTRADAS, SPLIT_TREINO, SPLIT_TESTE, carregar_sinais,
        construir_matriz, classificar_knn,
    )
    from sklearn.metrics import f1_score
    import numpy as np

    if not JANELAS_AMOSTRADAS.exists():
        print("SKIP: janelas_amostradas.csv nao existe, rodar pipeline8_openpack.py primeiro")
        return

    df_janelas = pd.read_csv(JANELAS_AMOSTRADAS)
    n_classes = df_janelas["operacao"].nunique()
    acaso_esperado = 1.0 / n_classes

    janelas_treino = df_janelas[df_janelas[["sujeito", "sessao"]].apply(tuple, axis=1).isin(SPLIT_TREINO)]
    janelas_teste = df_janelas[df_janelas[["sujeito", "sessao"]].apply(tuple, axis=1).isin(SPLIT_TESTE)]
    sinais = carregar_sinais(pd.concat([janelas_treino, janelas_teste]))

    X_treino, y_treino, _ = construir_matriz(janelas_treino, sinais)
    X_teste, y_teste, _ = construir_matriz(janelas_teste, sinais)

    random.seed(42)
    y_treino_embaralhado = y_treino.copy()
    random.shuffle(y_treino_embaralhado)

    media_treino = X_treino.mean(axis=0)
    desvio_treino = X_treino.std(axis=0)
    desvio_treino[desvio_treino == 0] = 1.0
    X_treino_norm = (X_treino - media_treino) / desvio_treino
    X_teste_norm = (X_teste - media_treino) / desvio_treino

    y_pred = classificar_knn(X_treino_norm, y_treino_embaralhado, X_teste_norm, k=5)
    y_true_validos = [yt for yt, yp in zip(y_teste, y_pred) if yp is not None]
    y_pred_validos = [yp for yp in y_pred if yp is not None]

    assert y_true_validos, "nenhuma janela classificada -- verificar split/features"
    f1_embaralhado = f1_score(y_true_validos, y_pred_validos, average="macro", zero_division=0)

    assert f1_embaralhado <= acaso_esperado * 2, (
        f"F1 (indice numerico) com rotulo de TREINO embaralhado = {f1_embaralhado:.4f}, muito "
        f"acima do acaso (~{acaso_esperado:.4f}) -- indica vazamento no pareamento janela<->"
        f"rotulo. Nao promover o indice numerico ate investigar."
    )


def test_indice_pose_openpack_com_rotulo_embaralhado_desaba_para_acaso():
    """Mesma bateria acima (item 1, rotulo embaralhado), aplicada ao indice de pose
    (eval/indice_pose_openpack.py, plano 'OpenPack Fase 2: classificador de pose sobre keypoints
    3D', 2026-09-29). Risco de vazamento aqui e o mesmo do indice numerico de IMU (rotulo vem
    direto da coluna 'operacao' de janelas_amostradas.csv, nao de texto embeddado) -- mas o
    RISCO DE ENGANO na extracao de features e maior: se normalizar_pose() esquecesse de
    centralizar no quadril ou escalar por osso, a posicao absoluta na cena vazaria identidade/
    sessao para o kNN de forma sutil, sem aparecer como vazamento textual obvio. Split e LOSO
    restrito aos 6 sujeitos com video (nao o SPLIT_TREINO/SPLIT_TESTE oficial do indice de
    IMU, que nao tem overlap com esses sujeitos -- ver docstring do modulo avaliado)."""
    sys.path.insert(0, str(HARBOR_ROOT / "eval"))
    from indice_pose_openpack import (
        JANELAS_AMOSTRADAS, SUJEITOS_COM_VIDEO, carregar_keypoints_dos_sujeitos,
        construir_matriz,
    )
    from indice_numerico_openpack import classificar_knn
    from sklearn.metrics import f1_score
    import numpy as np

    if not JANELAS_AMOSTRADAS.exists():
        print("SKIP: janelas_amostradas.csv nao existe, rodar pipeline8_openpack.py primeiro")
        return

    df_janelas = pd.read_csv(JANELAS_AMOSTRADAS)
    df_janelas = df_janelas[df_janelas["sujeito"].isin(SUJEITOS_COM_VIDEO)]
    n_classes = df_janelas["operacao"].nunique()
    acaso_esperado = 1.0 / n_classes

    keypoints_por_sessao = carregar_keypoints_dos_sujeitos(df_janelas)
    if not keypoints_por_sessao:
        print("SKIP: nenhum keypoint carregado -- verificar zips em Desktop/OpenPack/zenodo")
        return

    sujeito_teste = SUJEITOS_COM_VIDEO[0]
    janelas_treino = df_janelas[df_janelas["sujeito"] != sujeito_teste]
    janelas_teste = df_janelas[df_janelas["sujeito"] == sujeito_teste]

    X_treino, y_treino, _, _ = construir_matriz(janelas_treino, keypoints_por_sessao)
    X_teste, y_teste, _, _ = construir_matriz(janelas_teste, keypoints_por_sessao)

    if len(X_treino) == 0 or len(X_teste) == 0:
        print(f"SKIP: sem janelas classificaveis para {sujeito_teste} apos filtro de keypoints")
        return

    random.seed(42)
    y_treino_embaralhado = y_treino.copy()
    random.shuffle(y_treino_embaralhado)

    media_treino = X_treino.mean(axis=0)
    desvio_treino = X_treino.std(axis=0)
    desvio_treino[desvio_treino == 0] = 1.0
    X_treino_norm = (X_treino - media_treino) / desvio_treino
    X_teste_norm = (X_teste - media_treino) / desvio_treino

    y_pred = classificar_knn(X_treino_norm, y_treino_embaralhado, X_teste_norm, k=5)
    y_true_validos = [yt for yt, yp in zip(y_teste, y_pred) if yp is not None]
    y_pred_validos = [yp for yp in y_pred if yp is not None]

    assert y_true_validos, "nenhuma janela classificada -- verificar split/features de pose"
    f1_embaralhado = f1_score(y_true_validos, y_pred_validos, average="macro", zero_division=0)

    assert f1_embaralhado <= acaso_esperado * 2, (
        f"F1 (indice de pose) com rotulo de TREINO embaralhado = {f1_embaralhado:.4f}, muito "
        f"acima do acaso (~{acaso_esperado:.4f}) -- indica vazamento no pareamento janela<->"
        f"rotulo, ou normalizacao de pose deixando passar posicao absoluta/identidade. Nao "
        f"promover o indice de pose ate investigar."
    )


# ---------------------------------------------------------------------------------------------
# Bateria 2: corpus embaralhado -- Recall@k deve cair para k/N quando nao ha documento certo
# ---------------------------------------------------------------------------------------------

def test_retrieval_multimodal_com_pergunta_desconectada_do_corpus():
    """Para a Trilha A (graficos): uma pergunta sobre um assunto que NAO existe em nenhuma
    legenda do corpus deve ter Recall@k baixo (nao encontrar nada com confianca) -- controle
    negativo simetrico ao teste de rotulo embaralhado, adaptado para retrieval em vez de
    classificacao (aqui nao ha "rotulo" para embaralhar, o equivalente e uma query fora de
    distribuicao)."""
    if not CORPUS_MULTIMODAL.exists():
        print("SKIP: corpus multimodal nao existe, rodar rag/legendas_deterministicas.py primeiro")
        return

    sys.path.insert(0, str(HARBOR_ROOT / "rag"))
    from rag_hibrido_langchain import RAGHibrido

    docs = json.loads(CORPUS_MULTIMODAL.read_text(encoding="utf-8"))
    rag = RAGHibrido(
        chroma_dir=HARBOR_ROOT / "rag" / "chroma_db_multimodal_deterministico",
        colecao="graficos_harbor_deterministico_v1",
    )
    rag.indexar(forcar=False, documentos_customizados=docs)

    # Pergunta deliberadamente fora de qualquer assunto do corpus (graficos tecnicos de
    # manufatura) -- nenhum documento deve ser um match forte.
    candidatos = rag.buscar(
        "qual e a capital da Mongolia e quantos habitantes ela tem",
        k=3, usar_rerank=False, usar_hybrid=True, usar_score_rrf=True,
    )
    scores = [c.get("score", 0.0) for c in candidatos]
    # Nao exigimos score exatamente zero (RRF sempre retorna algo), so que o melhor score fique
    # bem abaixo do que se observa em perguntas legitimas do golden set (tipicamente > 3, ver
    # slide de validacao externa) -- serve de sinal, nao de gate binario rigido.
    assert max(scores, default=0.0) < 2.0, (
        f"Query fora de distribuicao recebeu score alto ({max(scores):.2f}) -- "
        f"o retrieval pode estar 'inventando' relevancia em vez de discriminar de verdade."
    )


# ---------------------------------------------------------------------------------------------
# Bateria 3: controle positivo -- caso trivial que DEVE passar com score maximo
# ---------------------------------------------------------------------------------------------

def test_retrieval_multimodal_controle_positivo_pergunta_literal():
    """Complementa a bateria 1/2 (controle negativo) com o controle POSITIVO: perguntar
    literalmente o titulo de um grafico conhecido deve retorna-lo em 1o lugar. Se isso falhar,
    o harness em si esta quebrado -- nao adianta desconfiar do metodo antes de confirmar que o
    equipamento de medicao funciona (mesma logica da licao de 2026-09-14 sobre checks de
    fidelidade que nunca foram exercitados por um caso 100%)."""
    if not CORPUS_MULTIMODAL.exists():
        print("SKIP: corpus multimodal nao existe, rodar rag/legendas_deterministicas.py primeiro")
        return

    sys.path.insert(0, str(HARBOR_ROOT / "rag"))
    from rag_hibrido_langchain import RAGHibrido

    docs = json.loads(CORPUS_MULTIMODAL.read_text(encoding="utf-8"))
    assert docs, "corpus multimodal vazio"
    primeiro_doc = docs[0]

    rag = RAGHibrido(
        chroma_dir=HARBOR_ROOT / "rag" / "chroma_db_multimodal_deterministico",
        colecao="graficos_harbor_deterministico_v1",
    )
    rag.indexar(forcar=False, documentos_customizados=docs)

    # Usa as primeiras palavras do proprio texto do documento como query -- caso trivial,
    # o documento tem que ser o top-1.
    query_literal = primeiro_doc["texto"][:80]
    candidatos = rag.buscar(query_literal, k=3, usar_rerank=False, usar_hybrid=True, usar_score_rrf=True)
    ids_recuperados = [c.get("id") or c.get("fonte") for c in candidatos]

    assert primeiro_doc["id"] in ids_recuperados[:1], (
        f"Controle positivo FALHOU: query literal do proprio documento nao o recuperou em "
        f"1o lugar (top-1 foi {ids_recuperados[0] if ids_recuperados else None}). "
        f"O harness de retrieval pode estar quebrado, nao o metodo avaliado."
    )


if __name__ == "__main__":
    testes = [
        test_corpus_openpack_nao_contem_rotulo_de_operacao,
        test_classificacao_openpack_com_rotulo_embaralhado_desaba_para_acaso,
        test_indice_numerico_openpack_com_rotulo_embaralhado_desaba_para_acaso,
        test_indice_pose_openpack_com_rotulo_embaralhado_desaba_para_acaso,
        test_retrieval_multimodal_com_pergunta_desconectada_do_corpus,
        test_retrieval_multimodal_controle_positivo_pergunta_literal,
    ]
    falhas = []
    for teste in testes:
        nome = teste.__name__
        try:
            teste()
            print(f"OK   {nome}")
        except AssertionError as e:
            print(f"FAIL {nome}: {e}")
            falhas.append(nome)
        except Exception as e:
            print(f"ERRO {nome}: {type(e).__name__}: {e}")
            falhas.append(nome)

    print()
    if falhas:
        print(f"{len(falhas)}/{len(testes)} testes de sanidade FALHARAM: {falhas}")
        sys.exit(1)
    else:
        print(f"Todos os {len(testes)} testes de sanidade passaram.")
