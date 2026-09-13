"""
Complemento da Fase 7a: mesma classificacao F1-macro de eval/avaliar_classificacao_openpack.py,
mas usando TODAS as janelas de teste do split oficial (U0106/S0100,S0300,S0500 SEM amostragem
estratificada), nao so as ~26 que sobraram na amostra balanceada de 1.000 janelas.

ACHADO REAL (2026-09-13): a primeira rodada (avaliar_classificacao_openpack.py) deu
F1-macro=0,9217, superando os 3 baselines oficiais -- investigado e confirmado sem vazamento de
dado (zero overlap de sujeito treino/teste), mas com uma diferenca metodologica honesta: o
teste usava a amostra ESTRATIFICADA balanceada (100/classe), mais "limpa" que a avaliacao
oficial sobre a sessao inteira segmentada (com ruido/desbalanceamento real). Este script fecha
essa lacuna processando as sessoes de teste completas, sem amostragem -- mais proximo do
protocolo oficial, mesmo treino (amostra estratificada de U0102/U0103/U0105, ja validada).
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import datasets  # noqa: F401 (import de protecao contra access violation torch x pyarrow)
from sklearn.metrics import f1_score

EVAL_DIR = Path(__file__).resolve().parent
HARBOR_ROOT = EVAL_DIR.parent

sys.path.insert(0, str(HARBOR_ROOT / "pipelines"))
from pipeline8_openpack import carregar_sessao, segmentar_janelas, JANELA_SEGUNDOS, PASSO_SEGUNDOS

sys.path.insert(0, str(HARBOR_ROOT / "rag"))
from rag_openpack_texto import features_todos_canais, montar_texto_janela, _discretizar_perfil, CANAIS

JANELAS_AMOSTRADAS = HARBOR_ROOT / "outputs" / "pipeline8_openpack" / "janelas_amostradas.csv"
CORPUS_TREINO = HARBOR_ROOT / "rag" / "corpus_openpack_janelas.json"
CHROMA_DIR = HARBOR_ROOT / "rag" / "chroma_db_openpack"
COLECAO = "openpack_janelas_v1"

SESSOES_TESTE_OFICIAL = [("U0106", "S0100"), ("U0106", "S0300"), ("U0106", "S0500")]
K = 5

BASELINES_OFICIAIS = {"UNet": 0.3451, "ST-GCN": 0.7024, "DeepConvLSTM": 0.7081}


def gerar_textos_teste_sem_amostragem():
    """Segmenta as 3 sessoes de teste oficiais na integra (sem estratificar por classe),
    devolvendo (janela_id, operacao, texto_completo) para cada janela valida (pureza >= 0.7,
    exclui Null) -- mesmo criterio de pipeline8_openpack.segmentar_janelas(), sem descartar
    nenhuma classe por cota de amostragem."""
    limiares_intensidade = None
    limiares_regularidade = None
    todas_janelas = []

    for sujeito, sessao in SESSOES_TESTE_OFICIAL:
        df_sinais = carregar_sessao(sujeito, sessao, usar_sample=False)
        deltas = df_sinais["timestamp_ms"].diff().dropna()
        freq_hz = 1000 / deltas.median()
        janelas = segmentar_janelas(df_sinais, freq_hz=freq_hz, janela_s=JANELA_SEGUNDOS, passo_s=PASSO_SEGUNDOS)

        for _, janela in janelas.iterrows():
            bloco = df_sinais.iloc[janela["indice_inicio"]: janela["indice_fim"]]
            todas_janelas.append((janela, bloco))

    # limiares de discretizacao calculados sobre TODO o conjunto de teste (mesma logica de
    # gerar_corpus_rag, mas aqui e so para a query -- nao entra no corpus de treino)
    desvios, cvs = [], []
    for janela, bloco in todas_janelas:
        feats = features_todos_canais(bloco)
        desvio_medio = sum(feats[c]["desvio_padrao"] for c in CANAIS) / len(CANAIS)
        media_valores = [feats[c]["media"] for c in CANAIS if feats[c]["media"] != 0]
        media_media = sum(media_valores) / len(media_valores) if media_valores else 0
        desvios.append(desvio_medio)
        cvs.append(desvio_medio / media_media if media_media else 0.0)

    import numpy as np
    limiares_intensidade = tuple(np.percentile(desvios, [33, 66])) if desvios else (0.0, 0.0)
    limiares_regularidade = tuple(np.percentile(cvs, [50, 50])) if cvs else (0.0, 0.0)

    resultado = []
    for janela, bloco in todas_janelas:
        feats = features_todos_canais(bloco)
        perfil = _discretizar_perfil(feats, limiares_intensidade, limiares_regularidade)
        meta = {"sujeito": janela["sujeito"], "sessao": janela["sessao"],
                "operacao": janela["operacao"], "operation_label": janela["operation_label"]}
        texto = montar_texto_janela(meta, feats, "completo", perfil)
        resultado.append((janela["janela_id"], janela["operacao"], texto))

    return resultado


def agregar_por_fonte(candidatos):
    vistos = []
    for c in candidatos:
        doc_id = c.get("id") or c.get("fonte", "")
        fonte = doc_id.rsplit("__", 1)[0] if "__" in doc_id else doc_id
        if fonte not in vistos:
            vistos.append(fonte)
    return vistos


def main():
    print("Segmentando as 3 sessoes de teste oficiais na integra (sem amostragem)...")
    t0 = time.time()
    teste_completo = gerar_textos_teste_sem_amostragem()
    print(f"{len(teste_completo)} janelas de teste (todas as classes, sem amostragem) "
          f"em {time.time()-t0:.1f}s.")

    df_amostrado = pd.read_csv(JANELAS_AMOSTRADAS)
    operacao_por_janela_treino = dict(zip(df_amostrado["janela_id"], df_amostrado["operacao"]))
    split_treino = [("U0102", s) for s in ("S0100", "S0200", "S0300", "S0400", "S0500")] + \
                   [("U0103", s) for s in ("S0100", "S0200", "S0300", "S0400", "S0500")] + \
                   [("U0105", s) for s in ("S0100", "S0200", "S0300", "S0400", "S0500")]
    treino = df_amostrado[df_amostrado[["sujeito", "sessao"]].apply(tuple, axis=1).isin(split_treino)]
    janelas_treino_ids = set(treino["janela_id"])
    print(f"Base de retrieval (treino, ja indexada): {len(treino)} janelas de U0102/U0103/U0105.")

    sys.path.insert(0, str(HARBOR_ROOT / "rag"))
    from rag_hibrido_langchain import RAGHibrido

    corpus_docs = json.loads(CORPUS_TREINO.read_text(encoding="utf-8"))
    rag = RAGHibrido(chroma_dir=CHROMA_DIR, colecao=COLECAO)
    t0_indexacao = time.time()
    rag.indexar(forcar=False, documentos_customizados=corpus_docs)
    tempo_indexacao_s = time.time() - t0_indexacao

    y_true, y_pred = [], []
    tempos_busca = []
    n_sem_vizinho = 0

    for janela_id, operacao_real, texto in teste_completo:
        t0_busca = time.time()
        candidatos = rag.buscar(texto, k=25, usar_rerank=False, usar_hybrid=True, usar_score_rrf=True)
        tempos_busca.append(time.time() - t0_busca)

        vizinhos = agregar_por_fonte(candidatos)
        vizinhos_treino = [v for v in vizinhos if v in janelas_treino_ids][:K]
        if not vizinhos_treino:
            n_sem_vizinho += 1
            continue

        votos = {}
        for v in vizinhos_treino:
            op = operacao_por_janela_treino.get(v)
            if op is not None:
                votos[op] = votos.get(op, 0) + 1
        if not votos:
            n_sem_vizinho += 1
            continue

        predicao = max(votos.items(), key=lambda item: item[1])[0]
        y_true.append(operacao_real)
        y_pred.append(predicao)

    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    tempo_medio_busca_s = sum(tempos_busca) / len(tempos_busca) if tempos_busca else 0.0

    print("=" * 74)
    print("CLASSIFICACAO F1-MACRO -- teste COMPLETO (sem amostragem), split oficial")
    print("=" * 74)
    print(f"Janelas classificadas: {len(y_true)}/{len(teste_completo)} ({n_sem_vizinho} sem vizinho de treino)")
    print(f"F1-macro (Harbor/RAG-HAR, teste completo): {f1_macro:.4f}")
    print()
    for nome, f1_oficial in BASELINES_OFICIAIS.items():
        diff = f1_macro - f1_oficial
        print(f"  {nome:15} F1={f1_oficial:.4f}  (Harbor {'supera' if diff > 0 else 'fica abaixo'} em {abs(diff):.4f})")
    print()
    print(f"Tempo de indexacao   : {tempo_indexacao_s:.1f}s")
    print(f"Tempo medio de busca : {tempo_medio_busca_s:.3f}s/janela")
    print(f"Tempo total          : {tempo_indexacao_s + sum(tempos_busca):.1f}s")

    resultado = {
        "protocolo": "teste_completo_sem_amostragem",
        "n_janelas_teste_total": len(teste_completo),
        "n_classificadas": len(y_true),
        "n_sem_vizinho_treino": n_sem_vizinho,
        "f1_macro": round(float(f1_macro), 4),
        "baselines_oficiais": BASELINES_OFICIAIS,
        "tempo_indexacao_s": round(tempo_indexacao_s, 1),
        "tempo_medio_busca_s": round(tempo_medio_busca_s, 3),
    }
    caminho_saida = EVAL_DIR / "resultados_classificacao_openpack_completo.json"
    caminho_saida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados salvos em {caminho_saida}")


if __name__ == "__main__":
    main()
