"""
Experimento LangChain - Versao B: padrao IDIOMATICO do LangChain para hybrid search.

Diferente da Versao A (rag_langchain_fiel.py), esta versao NAO tenta replicar os
algoritmos exatos da producao -- usa o componente que a propria documentacao/estado
da arte do LangChain recomenda para busca hibrida:

    - BM25Retriever (langchain-community)   no lugar do TF-IDF+cosseno da producao
    - EnsembleRetriever (RRF ponderado)      no lugar da uniao+dedupe manual da producao

Isso muda DUAS coisas em relacao a producao ao mesmo tempo: o framework E o algoritmo
de busca lexical/fusao. Por isso os resultados desta versao devem ser lidos como
"o que o padrao idiomatico do LangChain entrega", nao como "o efeito isolado de trocar
de framework" -- essa segunda leitura e o papel da Versao A. Ver README.md para a
metodologia completa e por que as duas versoes existem separadamente.

Interface identica a RAGHibrido.buscar(): list[{"texto", "fonte", "score"}].
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"C:\Projetos\Harbor\rag")))
from rag_hibrido import chunk_texto, MANUAIS_DIR, MODELO_EMBEDDING, MODELO_RERANK  # noqa: E402

CHROMA_DIR = Path(r"C:\Projetos\Harbor\experiments\langchain_rag\chroma_db_bm25rrf")
COLECAO = "manuais_harbor_langchain_bm25rrf"


class RAGLangChainBM25RRF:
    """Hybrid search idiomatico do LangChain: EnsembleRetriever(vetor E5 + BM25Retriever),
    RRF ponderado (k=60 fixo, interno ao EnsembleRetriever), seguido do mesmo rerank
    Cross-Encoder da Versao A."""

    def __init__(self):
        self._vectorstore = None
        self._embeddings = None
        self._reranker = None
        self._bm25 = None  # BM25Retriever, guarda os Documents indexados

    def _carregar_embeddings(self):
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
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
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document
        import chromadb

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        cliente = chromadb.PersistentClient(path=str(CHROMA_DIR))
        nomes_existentes = [c.name for c in cliente.list_collections()]

        embeddings = self._carregar_embeddings()

        documentos, metadados, ids = [], [], []
        for caminho in sorted(MANUAIS_DIR.glob("*.md")):
            texto = caminho.read_text(encoding="utf-8").strip()
            for i, chunk in enumerate(chunk_texto(texto)):
                documentos.append(chunk)
                metadados.append({"file_name": caminho.name, "chunk_index": i})
                ids.append(f"{caminho.stem}_{i}")

        docs_lc = [Document(page_content=t, metadata=m) for t, m in zip(documentos, metadados)]
        # BM25Retriever indexa em memoria a cada processo -- nao ha persistencia em disco
        # nativa como o ChromaDB; reconstruido sempre a partir dos mesmos chunks.
        self._bm25 = BM25Retriever.from_documents(docs_lc)

        if COLECAO in nomes_existentes and not forcar:
            self._vectorstore = Chroma(
                client=cliente, collection_name=COLECAO, embedding_function=embeddings,
                collection_metadata={"hnsw:space": "cosine"},
            )
            return self._vectorstore._collection.count()

        if COLECAO in nomes_existentes:
            cliente.delete_collection(COLECAO)

        self._vectorstore = Chroma(
            client=cliente, collection_name=COLECAO, embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._vectorstore.add_texts(texts=documentos, metadatas=metadados, ids=ids)
        return self._vectorstore._collection.count()

    def buscar(self, pergunta, k=3, score_min=0.75, usar_rerank=True, usar_hybrid=True, k_candidatos=10):
        from langchain_classic.retrievers import EnsembleRetriever

        if self._vectorstore is None:
            self.indexar()

        n_buscar = max(k_candidatos, k) if (usar_rerank or usar_hybrid) else k
        retriever_denso = self._vectorstore.as_retriever(search_kwargs={"k": n_buscar})

        if usar_hybrid:
            self._bm25.k = n_buscar
            ensemble = EnsembleRetriever(retrievers=[retriever_denso, self._bm25], weights=[0.5, 0.5])
            docs = ensemble.invoke(pergunta)
        else:
            docs = retriever_denso.invoke(pergunta)

        candidatos = [{"texto": d.page_content, "fonte": d.metadata["file_name"], "score": 0.0} for d in docs[:n_buscar]]

        if not usar_rerank:
            return candidatos[:k]

        # CrossEncoderReranker (langchain_classic) descarta o score no retorno -- ver
        # comentario equivalente em rag_langchain_fiel.py::buscar(). Reproduzimos
        # model.score() diretamente para expor o score na interface de saida.
        reranker_model = self._carregar_reranker()
        scores_rerank = reranker_model.score([(pergunta, c["texto"]) for c in candidatos])
        for c, s in zip(candidatos, scores_rerank):
            c["score"] = round(float(s), 4)
        candidatos.sort(key=lambda c: c["score"], reverse=True)
        return candidatos[:k]
