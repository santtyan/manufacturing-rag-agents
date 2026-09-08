"""
RAG hibrido via LangChain -- implementacao de PRODUCAO desde 2026-09-08 (substitui a classe
RAGHibrido de rag/rag_hibrido.py como implementacao usada por dashboard/MCP/harness).

Promovido de experiments/langchain_rag/rag_langchain_bm25rrf.py::RAGLangChainBM25RRF (a versao
"idiomatica LangChain": EnsembleRetriever(vetor E5 + BM25Retriever) com fusao RRF, nao a
uniao/dedupe manual da implementacao Python pura original). Ver
.claude/skills/migrar-para-langchain/SKILL.md, secao "Decisao registrada: migrar mesmo sem
ganho isolado de qualidade (2026-09-08)" -- decisao do usuario, nao motivada por ganho de
Recall@5/MRR isolado (medido identico entre as tres implementacoes, ver
experiments/langchain_rag/README.md), e sim por consistencia arquitetural: roteador e
self-repair vao migrar para LangGraph em seguida e vao consumir este retriever dentro de um
grafo. O ganho real medido (+11,4pp Precision@5) vem do algoritmo BM25+RRF, nao do framework
em si -- mas essa classe combina os dois.

rag/rag_hibrido.py (implementacao Python pura, TF-IDF/BM25 manual) fica INTACTA e continua
existindo -- so deixa de ser a importada pelos consumidores de producao
(dashboard/app.py, eval/rag_gerador.py, mcp/servidor_harbor.py). Os benchmarks de comparacao
(eval/avaliar_retrieval_nanobeir.py, eval/avaliar_retrieval.py, eval/comparar_tfidf_bm25.py)
continuam usando a versao antiga por enquanto -- decisao de migra-los tambem fica para depois.

Interface DROP-IN compativel com RAGHibrido: mesmo nome de classe, mesma assinatura de
__init__/indexar/buscar, mesmo formato de retorno de buscar() ({"texto", "fonte", "id",
"score"}), para que os consumidores nao precisem mudar nada alem do import.

Gaps resolvidos ao promover a versao experimental para producao (ela nao tinha isso):
  1. buscar() nao devolvia "id" -- adicionado (mesmo id usado na indexacao: f"{stem}_{i}"),
     usado por eval/avaliar_retrieval_nanobeir.py para comparar contra qrels.
  2. __init__/indexar() nao aceitavam chroma_dir/colecao/documentos_customizados -- adicionados,
     preservando o default (None = usa MANUAIS_DIR/colecao de producao), mesmo motivo do
     RAGHibrido original: benchmarks indexam corpus proprio sem colidir com o indice real.
  3. Parametro "lexico" (tfidf vs bm25) do RAGHibrido original NAO EXISTE mais aqui -- esta
     classe SEMPRE usa BM25+RRF (e a decisao tomada, ver skill). Nenhum consumidor de producao
     passava lexico= explicitamente (confirmado via grep antes da promocao); se algum vier a
     passar, vai falhar com TypeError (nao silenciosamente ignorado).
  4. Import de chunk_texto/MANUAIS_DIR/etc trocado de sys.path hack para import relativo normal
     (mesmo pacote rag/), ja que o arquivo agora mora em rag/ e nao mais em experiments/.
"""
from pathlib import Path

from rag_hibrido import chunk_texto, MANUAIS_DIR, MODELO_EMBEDDING, MODELO_RERANK

CHROMA_DIR = Path(r"C:\Projetos\Harbor\rag\chroma_db_langchain")
COLECAO = "manuais_harbor_langchain_bm25rrf"


class RAGHibrido:
    """Hybrid search idiomatico do LangChain: EnsembleRetriever(vetor E5 + BM25Retriever),
    RRF ponderado (k=60 fixo, interno ao EnsembleRetriever), seguido do mesmo rerank
    Cross-Encoder da implementacao Python pura. Ver docstring do modulo para o porque desta
    classe ter o MESMO NOME que rag/rag_hibrido.py::RAGHibrido -- substituto drop-in."""

    def __init__(self, chroma_dir=None, colecao=None):
        """chroma_dir/colecao opcionais -- default None usa os valores fixos do modulo
        (CHROMA_DIR/COLECAO de producao). Parametrizavel para benchmarks indexarem um corpus
        separado sem colidir com o indice de producao (mesmo padrao do RAGHibrido original)."""
        self._chroma_dir = chroma_dir or CHROMA_DIR
        self._colecao_nome = colecao or COLECAO
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

    def indexar(self, forcar=False, documentos_customizados=None):
        """Le os manuais .md, corta em chunks, embeda e indexa no ChromaDB + BM25Retriever
        (em memoria). Se a colecao ja existir e forcar=False, reaproveita o indice persistido
        em disco para o vetor denso (BM25 e sempre reconstruido em memoria -- BM25Retriever
        do langchain-community nao persiste em disco nativamente).

        documentos_customizados (opcional): lista de {"id", "texto", "fonte"} para indexar um
        corpus diferente dos manuais .md, SEM chunking -- mesmo contrato do RAGHibrido original,
        usado por benchmarks (ex.: NanoBEIR) que precisam do id original para comparar contra
        o gold (qrels)."""
        from langchain_chroma import Chroma
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document
        import chromadb

        self._chroma_dir.mkdir(parents=True, exist_ok=True)
        cliente = chromadb.PersistentClient(path=str(self._chroma_dir))
        nomes_existentes = [c.name for c in cliente.list_collections()]

        embeddings = self._carregar_embeddings()

        documentos, metadados, ids = [], [], []
        if documentos_customizados is not None:
            for doc in documentos_customizados:
                documentos.append(doc["texto"])
                metadados.append({"file_name": doc.get("fonte", doc["id"])})
                ids.append(str(doc["id"]))
        else:
            for caminho in sorted(MANUAIS_DIR.glob("*.md")):
                texto = caminho.read_text(encoding="utf-8").strip()
                for i, chunk in enumerate(chunk_texto(texto)):
                    documentos.append(chunk)
                    metadados.append({"file_name": caminho.name, "chunk_index": i})
                    ids.append(f"{caminho.stem}_{i}")

        # "_id" entra tambem nos metadados (nao so na lista "ids" separada do Chroma) porque
        # o retriever denso (Chroma.as_retriever) reconstroi Document a partir dos metadados
        # salvos, nao da lista de ids do add_texts -- sem isso, buscar() retornava "id": None
        # para todo resultado vindo do lado denso do EnsembleRetriever (achado ao promover
        # para producao, 2026-09-08: bug so aparecia com usar_hybrid=True, o BM25Retriever
        # guarda o Document original em memoria e por isso "escondia" o problema).
        metadados_com_id = [{**m, "_id": i} for m, i in zip(metadados, ids)]

        docs_lc = [Document(page_content=t, metadata=m)
                   for t, m in zip(documentos, metadados_com_id)]
        # BM25Retriever indexa em memoria a cada processo -- nao ha persistencia em disco
        # nativa como o ChromaDB; reconstruido sempre a partir dos mesmos chunks.
        self._bm25 = BM25Retriever.from_documents(docs_lc)

        if self._colecao_nome in nomes_existentes and not forcar:
            self._vectorstore = Chroma(
                client=cliente, collection_name=self._colecao_nome, embedding_function=embeddings,
                collection_metadata={"hnsw:space": "cosine"},
            )
            return self._vectorstore._collection.count()

        if self._colecao_nome in nomes_existentes:
            cliente.delete_collection(self._colecao_nome)

        self._vectorstore = Chroma(
            client=cliente, collection_name=self._colecao_nome, embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._vectorstore.add_texts(texts=documentos, metadatas=metadados_com_id, ids=ids)
        return self._vectorstore._collection.count()

    def buscar(self, pergunta, k=3, score_min=0.75, usar_rerank=True, usar_hybrid=True, k_candidatos=10):
        """Retorna os k trechos mais relevantes: [{texto, fonte, id, score}, ...]. Mesma
        assinatura/contrato de retorno de rag/rag_hibrido.py::RAGHibrido.buscar()."""
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

        candidatos = [
            {"texto": d.page_content, "fonte": d.metadata["file_name"],
             "id": d.metadata.get("_id"), "score": 0.0}
            for d in docs[:n_buscar]
        ]

        if not usar_rerank:
            return candidatos[:k]

        # CrossEncoderReranker (langchain_classic) descarta o score no retorno -- reproduzimos
        # model.score() diretamente para expor o score na interface de saida (mesmo padrao dos
        # dois experimentos que originaram esta classe).
        reranker_model = self._carregar_reranker()
        scores_rerank = reranker_model.score([(pergunta, c["texto"]) for c in candidatos])
        for c, s in zip(candidatos, scores_rerank):
            c["score"] = round(float(s), 4)
        candidatos.sort(key=lambda c: c["score"], reverse=True)
        return candidatos[:k]


if __name__ == "__main__":
    rag = RAGHibrido()
    n = rag.indexar(forcar=True)
    print(f"Indexados {n} chunks (LangChain: E5 + BM25 + RRF + Cross-Encoder).\n")

    perguntas_teste = [
        "a maquina esta esquentando demais, e perigoso?",
        "o equipamento esta tremendo muito, o que faco?",
        "como o sistema decide se um alerta e falha de verdade?",
    ]
    for pergunta in perguntas_teste:
        print(f"PERGUNTA: {pergunta}")
        for d in rag.buscar(pergunta, k=2):
            print(f"  [{d['fonte']} score={d['score']}] {d['texto'][:120]}...")
        print("-" * 70)
