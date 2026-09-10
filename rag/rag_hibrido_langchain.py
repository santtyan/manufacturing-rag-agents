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
MODELO_CONTEXTUALIZACAO = "qwen2.5:7b"


def _contextualizar_chunk(chunk: str, documento_completo: str, nome_arquivo: str) -> str:
    """Contextual Retrieval (Anthropic, arXiv/engineering blog 2024, adaptado para Ollama
    local -- ver docstring de indexar() para o motivo desta flag existir). Gera um resumo
    curto (50-100 tokens) situando o chunk dentro do documento completo, e PREPENDA ao texto
    original do chunk -- o texto final e o que entra tanto no embedding denso quanto no BM25
    (por isso "Contextual Embeddings" + "Contextual BM25" na nomenclatura original).

    O documento_completo e truncado a 3000 caracteres no prompt para nao inflar o contexto do
    LLM local com manuais grandes -- suficiente para o modelo entender de que assunto o
    documento trata, nao precisa do texto inteiro para gerar um resumo situacional curto."""
    from shared.ollama_client import chamar as chamar_ollama

    prompt = f"""Aqui esta um documento tecnico de manutencao industrial:
<documento>
{documento_completo[:3000]}
</documento>

Aqui esta um trecho especifico desse documento:
<trecho>
{chunk}
</trecho>

Escreva um contexto curto (1-2 frases, maximo 100 palavras) situando este trecho dentro do
documento, para melhorar sua recuperacao em busca. Responda SOMENTE o contexto, sem repetir o
trecho."""
    try:
        contexto = str(chamar_ollama(prompt, modelo=MODELO_CONTEXTUALIZACAO)).strip()
        return f"{contexto}\n\n{chunk}" if contexto else chunk
    except Exception:
        # Ollama indisponivel/erro -- fail-open, indexa o chunk sem contexto em vez de travar
        # a indexacao inteira (mesma postura ja usada no DBA-Agent e no RAG agentic).
        return chunk


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

    def indexar(self, forcar=False, documentos_customizados=None, usar_contexto=False):
        """Le os manuais .md, corta em chunks, embeda e indexa no ChromaDB + BM25Retriever
        (em memoria). Se a colecao ja existir e forcar=False, reaproveita o indice persistido
        em disco para o vetor denso (BM25 e sempre reconstruido em memoria -- BM25Retriever
        do langchain-community nao persiste em disco nativamente).

        documentos_customizados (opcional): lista de {"id", "texto", "fonte"} para indexar um
        corpus diferente dos manuais .md, SEM chunking -- mesmo contrato do RAGHibrido original,
        usado por benchmarks (ex.: NanoBEIR) que precisam do id original para comparar contra
        o gold (qrels).

        usar_contexto (Contextual Retrieval, Anthropic -- item 4 da rodada de fechamento da
        fila de trabalho, 2026-09-09): se True, prependa a cada chunk um resumo curto gerado
        por LLM situando o chunk dentro do documento, ANTES de embedar e de indexar no BM25
        ("Contextual Embeddings" + "Contextual BM25"). Default False -- flag de ablacao, nunca
        branch de codigo (mesmo padrao ja usado em usar_rerank/usar_hybrid), para medir
        Recall@5/MRR/Precision@5 com e sem antes de decidir se vira produção. So se aplica ao
        caminho de chunking automatico dos manuais -- documentos_customizados (benchmarks)
        nunca sao contextualizados, ja vem prontos por definicao."""
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
                chunks = chunk_texto(texto)
                for i, chunk in enumerate(chunks):
                    chunk_final = _contextualizar_chunk(chunk, texto, caminho.name) if usar_contexto else chunk
                    documentos.append(chunk_final)
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

    def buscar(self, pergunta, k=3, score_min=0.75, usar_rerank=True, usar_hybrid=True,
               k_candidatos=10, usar_score_rrf=False):
        """Retorna os k trechos mais relevantes: [{texto, fonte, id, score}, ...]. Mesma
        assinatura/contrato de retorno de rag/rag_hibrido.py::RAGHibrido.buscar().

        score_min: parametro morto, nunca usado no corpo da funcao -- achado real ao investigar
        o bug de score=0.0 abaixo (2026-09-09, plano 'Evoluir o RAG multimodal', item 2).
        Mantido na assinatura por compatibilidade de interface com consumidores existentes
        (nenhum passa score_min hoje, confirmado via grep antes desta mudanca), mas nao filtra
        nada -- nao remover silenciosamente sem avisar quem eventualmente passar esse argumento
        esperando que ele funcione.

        usar_score_rrf (2026-09-09, item 2 do plano acima): quando usar_hybrid=True e
        usar_rerank=False, o EnsembleRetriever (RRF) do LangChain nao expoe score nenhum --
        ACHADO REAL: candidatos sempre vinham com score=0.0 literal, o que invalidava qualquer
        calibracao de limiar de score rio abaixo (ex. selecionar_top_p() do benchmark 2x2
        multimodal sempre caia no caso "sem sinal, retorna 1 candidato", fazendo TOP_P_LIMIAR
        0,7/0,85/0,99 produzirem resultado identico). Com a flag ligada, o RRF e recalculado
        explicitamente aqui (chamando os dois retrievers em separado e fundindo por
        1/(60+rank), mesma constante k=60 que o EnsembleRetriever usa internamente) para expor
        o score fundido de verdade. Default False -- flag de ablacao (mesmo padrao de
        usar_contexto em indexar()): comportamento de producao (dashboard, avaliar_retrieval.py,
        NanoBEIR) fica inalterado ate medicao confirmar nao-regressao."""
        from langchain_classic.retrievers import EnsembleRetriever

        if self._vectorstore is None:
            self.indexar()

        n_buscar = max(k_candidatos, k) if (usar_rerank or usar_hybrid) else k
        retriever_denso = self._vectorstore.as_retriever(search_kwargs={"k": n_buscar})

        if usar_hybrid and usar_score_rrf:
            self._bm25.k = n_buscar
            docs_densos = retriever_denso.invoke(pergunta)
            docs_bm25 = self._bm25.invoke(pergunta)

            # RRF explicito: 1/(60+rank), rank comecando em 1 -- mesma constante k=60 default
            # do EnsembleRetriever do LangChain (nao documentada como parametrizavel na versao
            # instalada), fundindo os dois ranks por chave "_id" (unico entre os dois lados,
            # ver metadados_com_id em indexar()).
            RRF_K = 60
            scores_rrf = {}
            docs_por_id = {}
            for rank, d in enumerate(docs_densos, start=1):
                doc_id = d.metadata.get("_id")
                scores_rrf[doc_id] = scores_rrf.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
                docs_por_id[doc_id] = d
            for rank, d in enumerate(docs_bm25, start=1):
                doc_id = d.metadata.get("_id")
                scores_rrf[doc_id] = scores_rrf.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
                docs_por_id.setdefault(doc_id, d)

            ids_ordenados = sorted(scores_rrf, key=scores_rrf.get, reverse=True)[:n_buscar]
            candidatos = [
                {"texto": docs_por_id[i].page_content, "fonte": docs_por_id[i].metadata["file_name"],
                 "id": i, "score": round(scores_rrf[i], 6)}
                for i in ids_ordenados
            ]
        elif usar_hybrid:
            self._bm25.k = n_buscar
            ensemble = EnsembleRetriever(retrievers=[retriever_denso, self._bm25], weights=[0.5, 0.5])
            docs = ensemble.invoke(pergunta)
            candidatos = [
                {"texto": d.page_content, "fonte": d.metadata["file_name"],
                 "id": d.metadata.get("_id"), "score": 0.0}
                for d in docs[:n_buscar]
            ]
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
