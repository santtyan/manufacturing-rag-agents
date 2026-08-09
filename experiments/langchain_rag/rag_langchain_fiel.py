"""
Experimento LangChain - Versao A: replica FIEL do pipeline de producao (rag/rag_hibrido.py).

Objetivo: isolar o efeito de TROCAR DE FRAMEWORK, mantendo os MESMOS algoritmos que a
producao usa hoje (TF-IDF lexical, uniao+dedupe na fusao, sem RRF). A Versao B
(rag_langchain_bm25rrf.py) troca TF-IDF->BM25 e uniao->RRF, o padrao idiomatico do
LangChain -- mas isso muda o algoritmo, nao so o framework. Comparar producao contra
a Versao B sozinha misturaria as duas variaveis; esta versao existe para isolar
"o que o LangChain trouxe" mantendo o resto constante.

Reaproveita rag/rag_hibrido.py::chunk_texto (import direto, sem duplicar a logica de
particionamento por secao Markdown + fallback por linha) para que as duas comparacoes
usem exatamente os mesmos chunks -- qualquer diferenca de metrica vem da camada de
embedding/busca/rerank, nao de uma diferenca de chunking.

Interface identica a RAGHibrido.buscar(): list[{"texto", "fonte", "score"}].
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"C:\Projetos\Harbor\rag")))
from rag_hibrido import chunk_texto, MANUAIS_DIR, MODELO_EMBEDDING, MODELO_RERANK  # noqa: E402

CHROMA_DIR = Path(r"C:\Projetos\Harbor\experiments\langchain_rag\chroma_db_fiel")
COLECAO = "manuais_harbor_langchain_fiel"


class RAGLangChainFiel:
    """Mesmo pipeline de rag_hibrido.py (E5 denso + TF-IDF lexical + uniao/dedupe +
    rerank Cross-Encoder), mas usando componentes LangChain no lugar do codigo
    artesanal, onde existe um componente LangChain equivalente:

        - Embedding E5 com prefixos:  HuggingFaceEmbeddings (langchain-huggingface)
        - Vetor store:                Chroma (langchain-chroma)
        - Busca lexical TF-IDF:       sem equivalente LangChain -- mantido manual
          (scikit-learn), pois nao ha TFIDFRetriever no LangChain com o mesmo
          comportamento; forcar um substituto mudaria o algoritmo, nao so o framework.
        - Uniao + dedupe:             logica propria, identica a producao
        - Rerank Cross-Encoder:       ContextualCompressionRetriever + CrossEncoderReranker
    """

    def __init__(self):
        self._vectorstore = None
        self._embeddings = None
        self._reranker = None
        self._tfidf = None  # (vectorizer, matriz, documentos, metadados)

    def _carregar_embeddings(self):
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            # Prefixos E5 nativos do langchain-huggingface (confirmado em pesquisa:
            # encode_kwargs / query_encode_kwargs repassam "prompt" ao encode() do
            # sentence-transformers -- equivalente aos prefixos manuais de rag_hibrido.py).
            self._embeddings = HuggingFaceEmbeddings(
                model_name=MODELO_EMBEDDING,
                encode_kwargs={"prompt": "passage: ", "normalize_embeddings": True},
                query_encode_kwargs={"prompt": "query: ", "normalize_embeddings": True},
            )
        return self._embeddings

    def _carregar_reranker(self):
        if self._reranker is None:
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder
            self._reranker = HuggingFaceCrossEncoder(model_name=MODELO_RERANK)
        return self._reranker

    def indexar(self, forcar=False):
        from langchain_chroma import Chroma
        import chromadb

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        cliente = chromadb.PersistentClient(path=str(CHROMA_DIR))
        nomes_existentes = [c.name for c in cliente.list_collections()]

        embeddings = self._carregar_embeddings()

        if COLECAO in nomes_existentes and not forcar:
            self._vectorstore = Chroma(
                client=cliente, collection_name=COLECAO, embedding_function=embeddings,
                collection_metadata={"hnsw:space": "cosine"},
            )
            self._reconstruir_tfidf()
            return self._vectorstore._collection.count()

        if COLECAO in nomes_existentes:
            cliente.delete_collection(COLECAO)

        documentos, metadados, ids = [], [], []
        for caminho in sorted(MANUAIS_DIR.glob("*.md")):
            texto = caminho.read_text(encoding="utf-8").strip()
            for i, chunk in enumerate(chunk_texto(texto)):
                documentos.append(chunk)
                metadados.append({"file_name": caminho.name, "chunk_index": i})
                ids.append(f"{caminho.stem}_{i}")

        self._vectorstore = Chroma(
            client=cliente, collection_name=COLECAO, embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._vectorstore.add_texts(texts=documentos, metadatas=metadados, ids=ids)

        self._construir_tfidf(documentos, metadados)
        return self._vectorstore._collection.count()

    def _construir_tfidf(self, documentos, metadados):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words=None, max_features=2000)
        matriz = vectorizer.fit_transform(documentos)
        self._tfidf = (vectorizer, matriz, documentos, metadados)

    def _reconstruir_tfidf(self):
        dados = self._vectorstore._collection.get(include=["documents", "metadatas"])
        self._construir_tfidf(dados["documents"], dados["metadatas"])

    def _buscar_tfidf(self, pergunta, k):
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        if self._tfidf is None:
            self.indexar()
        vectorizer, matriz, documentos, metadados = self._tfidf
        vetor_pergunta = vectorizer.transform([pergunta])
        scores = cosine_similarity(vetor_pergunta, matriz)[0]
        top_idx = np.argsort(scores)[::-1][:k]
        return [
            {"texto": documentos[i], "fonte": metadados[i]["file_name"], "score": round(float(scores[i]), 4)}
            for i in top_idx if scores[i] > 0
        ]

    def buscar(self, pergunta, k=3, score_min=0.75, usar_rerank=True, usar_hybrid=True, k_candidatos=10):
        if self._vectorstore is None:
            self.indexar()

        # 1. Busca semantica (E5 via LangChain Chroma).
        n_buscar = max(k_candidatos, k) if (usar_rerank or usar_hybrid) else k
        resultados = self._vectorstore.similarity_search_with_relevance_scores(pergunta, k=n_buscar)

        candidatos = []
        vistos = set()
        for doc, score in resultados:
            candidatos.append({"texto": doc.page_content, "fonte": doc.metadata["file_name"], "score": round(float(score), 4)})
            vistos.add(doc.page_content)

        # 2. Busca lexical TF-IDF + uniao/dedupe -- mesma logica de producao (nao RRF).
        if usar_hybrid:
            for c_lex in self._buscar_tfidf(pergunta, k=n_buscar):
                if c_lex["texto"] not in vistos:
                    candidatos.append(c_lex)
                    vistos.add(c_lex["texto"])

        if not usar_rerank:
            candidatos.sort(key=lambda c: c["score"], reverse=True)
            return [c for c in candidatos if c["score"] >= score_min][:k]

        # 3. Rerank via CrossEncoderReranker (LangChain). A classe instalada
        # (langchain_classic) descarta o score no retorno de compress_documents --
        # so devolve os Documents reordenados (ver cross_encoder_rerank.py:47-50,
        # scores calculados internamente mas nao anexados ao metadata). Reproduzimos
        # o mesmo model.score() aqui para expor o score na interface de saida,
        # igual a producao faz.
        try:
            reranker_model = self._carregar_reranker()
            scores_rerank = reranker_model.score([(pergunta, c["texto"]) for c in candidatos])
            for c, s in zip(candidatos, scores_rerank):
                c["score"] = round(float(s), 4)
            candidatos.sort(key=lambda c: c["score"], reverse=True)
            return candidatos[:k]
        except Exception:
            candidatos.sort(key=lambda c: c["score"], reverse=True)
            return [c for c in candidatos if c["score"] >= score_min][:k]
