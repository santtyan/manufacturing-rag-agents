"""
Trilha 3 - RAG HIBRIDO sobre manual tecnico de equipamento (dado Tipo A).

Renomeado de rag_neural.py (2026-07-15): o nome "Neural" descrevia a mudanca historica de
TF-IDF puro para embeddings neurais, mas o modulo ja reincorporou o TF-IDF como metade do
hybrid search -- "Hibrido" descreve melhor o comportamento atual (E5 denso + TF-IDF lexical
+ rerank Cross-Encoder).

Substitui o TF-IDF lexical (rag_manual_tecnico.py) por embeddings neurais, agora que os
pacotes puderam ser baixados. Reaproveita o padrao ja pago e documentado em
LIA---TRABALHO-FINAL (docs/indexacao.md):

  1. Modelo intfloat/multilingual-e5-small via sentence-transformers.
  2. PREFIXOS E5 OBRIGATORIOS: documentos como "passage: {texto}", queries como "query: {texto}".
     Sem isso, o retrieval retorna lixo com score alto (prova experimental documentada:
     score 0.196 retornando texto errado).
  3. ChromaDB com metadata={"hnsw:space": "cosine"} explicito (evita ambiguidade L2 vs cosine).
  4. Indexar texto corrido chunkado, nao blocos curtos (chunks de tabela "vencem" no cosseno
     contra paragrafos ricos).

Interface identica ao TF-IDF (buscar -> lista de {texto, fonte, id, score}) para ser
plugavel no lugar do modulo antigo sem tocar em quem consome. Campo "id" adicionado para
permitir uso por benchmarks (eval/avaliar_retrieval_nanobeir.py) que precisam comparar
contra um id de documento original, nao so o nome do arquivo -- aditivo, nao quebra codigo
que so usa texto/fonte/score.
"""
import re
from pathlib import Path

MANUAIS_DIR = Path(r"C:\Projetos\Harbor\rag\manuais")
CHROMA_DIR = Path(r"C:\Projetos\Harbor\rag\chroma_db")
MODELO_EMBEDDING = "intfloat/multilingual-e5-small"
MODELO_RERANK = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLECAO = "manuais_harbor_v2"  # nome novo: forca reindexacao com o chunk maior (v2)

# Parte 5 do plano padrao-ouro: chunk subiu de 500 para ~900 chars. A medicao da NVIDIA aponta
# sweet spot 512-1024 tokens (~2000-4000 chars) para corpora GRANDES, mas o corpus do Harbor e
# pequeno (5 docs, a maioria com 1200-3600 chars -- ver `outputs de teste`): testando 2000 chars,
# 4 dos 5 documentos viraram 1 UNICO chunk (documento inteiro), o que eliminou a granularidade e
# fez o Cross-Encoder de rerank nao ter nada para discriminar (mesma ordem sempre). 900 chars da
# ~2-4 chunks por documento nos maiores, preservando alguma granularidade real, e ainda e maior
# que os 500 originais (que cortavam no meio de frase). Ajustado ao TAMANHO REAL do corpus, nao
# ao numero generico da literatura -- exatamente o principio "meça no seu dado" do plano.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150  # ~17%, dentro da faixa 10-20% recomendada (Pinecone/NVIDIA)


def _chunk_por_linhas(texto, chunk_size, overlap):
    """Fallback quando uma secao Markdown sozinha excede chunk_size: agrupa LINHAS inteiras
    ate perto do limite, nunca cortando no meio de uma linha. Essencial para as tabelas de
    schema dos datasets (readme_oee_downtime.md, readme_discrete_manufacturing.md) -- corte
    por caractere cru particiona uma linha "| Availability | ... | Closed Unit Interval" ao
    meio, perdendo o par coluna-unidade. Overlap aqui e por LINHAS (nao por char), repete as
    ultimas linhas do chunk anterior no comeco do proximo."""
    linhas = texto.split("\n")
    chunks = []
    bloco = []
    tam_bloco = 0
    for linha in linhas:
        tam_linha = len(linha) + 1
        if tam_bloco + tam_linha > chunk_size and bloco:
            chunks.append("\n".join(bloco))
            n_overlap = 0
            tam_recuo = 0
            for l in reversed(bloco):
                tam_recuo += len(l) + 1
                n_overlap += 1
                if tam_recuo >= overlap:
                    break
            bloco = bloco[-n_overlap:]
            tam_bloco = sum(len(l) + 1 for l in bloco)
        bloco.append(linha)
        tam_bloco += tam_linha
    if bloco:
        chunks.append("\n".join(bloco))
    return chunks


def chunk_texto(texto, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chunking por SECAO Markdown (##), nao por caracteres fixos.

    Achado do harness (2026-07-11): corte por caracteres cru quebra tabelas de schema no
    meio de uma linha ("| Availability... | Closed Unit Int" cortado ao meio) e separa um
    cabecalho de secao do seu conteudo (ex: "## 8. Escala de Criticidade" ficou orfao,
    o chunk seguinte comecava em "elo de linguagem analisa..." sem indicar que aquilo era
    a secao 8) -- isso fez o retrieval preferir um chunk vizinho errado (secao 2, threshold
    85 C) sobre o chunk certo (secao 8, threshold 90 C) numa pergunta sobre criticidade.

    Cada bloco delimitado por um cabecalho "## " vira um chunk (preserva tabelas inteiras e
    o cabecalho junto do conteudo). Se uma secao sozinha ultrapassar chunk_size (comum nas
    tabelas de schema grandes), ela e sub-dividida por LINHAS (nunca no meio de uma linha de
    tabela) via _chunk_por_linhas -- mesmo cuidado de granularidade documentado acima (linha 27).

    LIMITACAO CONHECIDA: este split e especifico ao formato dos documentos atuais, que usam
    cabecalhos Markdown "## N. Titulo" de forma consistente. Um documento com outra convencao
    de estrutura (ex: "CAPITULO N" / "Art. N" de um PDF de politica interna, ou HTML sem
    cabecalhos Markdown) nao teria nenhum "## " para o split reconhecer -- o documento inteiro
    cairia no fallback _chunk_por_linhas, perdendo a granularidade por secao (mesmo problema
    que motivou esta reescrita, so que sem o corte no meio de linha). Se um documento assim
    entrar no corpus, ajustar o regex de split (linha abaixo) ou pre-processar o documento para
    ter cabecalhos "## " antes de indexar.

    RESSALVA IMPORTANTE (nao tratar como regra universal): um estudo empirico da NVIDIA
    (blog "Finding the Best Chunking Strategy for Accurate AI Responses", jun/2025, testado
    em 5 datasets/PDFs longos) achou o OPOSTO do que se poderia supor -- page-level chunking
    (por pagina fisica do PDF) venceu section-level chunking (por estrutura, como aqui) na
    maioria dos datasets testados, mesmo controlando pelo mesmo extrator. O mesmo estudo
    tambem mostrou que datasets dentro da MESMA categoria (documentos financeiros) tiveram
    estrategias otimas diferentes entre si. Conclusao: chunking por secao Markdown e a
    escolha certa PARA ESTE CORPUS (documentos .md curtos, sem paginas fisicas, com
    cabecalhos consistentes -- validado empiricamente com Recall@5=100%/MRR=1.0 nesta
    sessao), nao uma "melhor pratica" universal. Se o corpus mudar de natureza (ex: PDFs
    longos e paginados), reavaliar chunking por pagina como alternativa, nao assumir que
    chunking estrutural continua vencendo."""
    blocos = re.split(r"(?=^## )", texto, flags=re.MULTILINE)
    blocos = [b for b in blocos if b.strip()]

    chunks = []
    for bloco in blocos:
        bloco = bloco.strip()
        if len(bloco) <= chunk_size:
            chunks.append(bloco)
        else:
            chunks.extend(_chunk_por_linhas(bloco, chunk_size, overlap))
    return chunks


class RAGHibrido:
    """Motor de busca hibrida (RAG) sobre o manual tecnico: 3 etapas em sequencia.

        1. BUSCA SEMANTICA (E5)   -> entende significado, erra termo tecnico exato.
        2. BUSCA LEXICAL (TF-IDF) -> entende termo exato, erra sinonimo/parafrase.
        3. RERANK (Cross-Encoder) -> reordena a uniao das duas buscas por relevancia real.

    E5 e TF-IDF sao bibliotecas prontas (sentence-transformers / scikit-learn), usadas sem
    alterar o algoritmo. A engenharia do projeto esta em COMO essas pecas se conectam: unir
    os dois resultados sem duplicar (buscar(), etapa 2) e so entao rerankear o conjunto
    combinado (buscar(), etapa 3) -- ver docstring de cada metodo abaixo.
    """

    def __init__(self, chroma_dir=None, colecao=None):
        """chroma_dir/colecao opcionais -- default None usa os valores fixos do modulo
        (CHROMA_DIR/COLECAO), comportamento de producao INTOCADO. Parametrizavel para que
        benchmarks (ex: NanoBEIR, eval/avaliar_retrieval_nanobeir.py) possam indexar um
        corpus diferente sem colidir com o indice de producao -- ver indexar()."""
        self._chroma_dir = chroma_dir or CHROMA_DIR
        self._colecao_nome = colecao or COLECAO
        self._modelo = None      # SentenceTransformer (E5), carregado sob demanda
        self._colecao = None     # colecao do ChromaDB (indice dos embeddings)
        self._reranker = None    # CrossEncoder, carregado sob demanda
        self._tfidf = None       # (vectorizer, matriz, chunks, metadados) do indice lexical

    # ── Modelos (carregamento preguiçoso, uma vez por instância) ──────────────────

    def _carregar_modelo(self):
        """Modelo de embedding E5 (sentence-transformers), uso padrao da biblioteca."""
        if self._modelo is None:
            from sentence_transformers import SentenceTransformer
            self._modelo = SentenceTransformer(MODELO_EMBEDDING)
        return self._modelo

    def _carregar_reranker(self):
        """Cross-Encoder (sentence-transformers): le pergunta+trecho JUNTOS (nao isolados)
        e da uma nota de relevancia mais precisa que a similaridade de embedding sozinha."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(MODELO_RERANK)
        return self._reranker

    def _cliente_chroma(self):
        """Cliente do ChromaDB, o banco vetorial onde os embeddings ficam persistidos."""
        import chromadb
        self._chroma_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(self._chroma_dir))

    # ── Busca semântica: texto -> vetor (E5) ───────────────────────────────────────

    def _embed_passages(self, textos):
        """Converte trechos do manual em vetores, para indexar. Prefixo 'passage:' e
        exigencia do modelo E5 -- sem ele o retrieval piora (score alto, texto errado)."""
        modelo = self._carregar_modelo()
        return modelo.encode([f"passage: {t}" for t in textos], normalize_embeddings=True).tolist()

    def _embed_query(self, texto):
        """Converte a pergunta do usuario em vetor, no mesmo espaco dos passages acima.
        Prefixo 'query:' e o par obrigatorio de 'passage:' -- mesma exigencia do E5."""
        modelo = self._carregar_modelo()
        return modelo.encode([f"query: {texto}"], normalize_embeddings=True).tolist()

    # ── Indexação (roda uma vez, antes de qualquer pergunta) ───────────────────────

    def indexar(self, forcar=False, documentos_customizados=None):
        """Le os manuais .md, corta em chunks, embeda e salva no ChromaDB + indice TF-IDF.
        Se a colecao ja existir e forcar=False, reaproveita o indice persistido em disco.

        documentos_customizados (opcional): lista de {"id", "texto", "fonte"} para indexar
        um corpus diferente dos manuais .md, SEM chunking (cada item vira 1 documento
        indexado como esta, usando o "id" fornecido em vez de um chunk_index sintetico).
        Usado por benchmarks (ex: NanoBEIR) cujo corpus ja vem em unidades de documento
        prontas e precisa do id original para comparar contra o gold (qrels) -- ver
        eval/avaliar_retrieval_nanobeir.py. None (default) preserva o comportamento de
        producao (le MANUAIS_DIR, chunking por secao)."""
        cliente = self._cliente_chroma()

        nomes_existentes = [c.name for c in cliente.list_collections()]
        if self._colecao_nome in nomes_existentes and not forcar:
            self._colecao = cliente.get_collection(self._colecao_nome)
            self._reconstruir_tfidf()
            return self._colecao.count()

        if self._colecao_nome in nomes_existentes:
            cliente.delete_collection(self._colecao_nome)

        # hnsw:space cosine explicito -- casa com normalize_embeddings=True.
        self._colecao = cliente.create_collection(self._colecao_nome, metadata={"hnsw:space": "cosine"})

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

        # ChromaDB aceita lotes de ate 5461 itens por chamada (limite do backend) -- corpus
        # de benchmark pode passar disso, ao contrario dos poucos chunks dos manuais.
        LOTE = 5000
        embeddings = self._embed_passages(documentos)
        for i in range(0, len(documentos), LOTE):
            self._colecao.add(
                documents=documentos[i:i + LOTE], embeddings=embeddings[i:i + LOTE],
                metadatas=metadados[i:i + LOTE], ids=ids[i:i + LOTE],
            )

        self._construir_tfidf(documentos, metadados, ids)
        return self._colecao.count()

    # ── Busca lexical: texto -> índice TF-IDF ──────────────────────────────────────

    def _construir_tfidf(self, documentos, metadados, ids=None):
        """Indice lexical (TF-IDF, scikit-learn puro) sobre os MESMOS chunks do indice E5 --
        e a metade 'palavra exata' do hybrid search: acerta siglas/codigos que o E5 erra.

        ids opcional: preserva o id de cada documento (usado por buscar()/_buscar_tfidf()
        para incluir "id" no resultado, alem de "fonte") -- None quando reconstruido a
        partir de colecao antiga sem essa informacao explicita (retrocompatibilidade)."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words=None, max_features=2000)
        matriz = vectorizer.fit_transform(documentos)
        self._tfidf = (vectorizer, matriz, documentos, metadados, ids)

    def _reconstruir_tfidf(self):
        """Reconstroi o indice TF-IDF em memoria a partir dos chunks ja salvos no ChromaDB --
        usado ao reaproveitar uma colecao existente num processo novo (self._tfidf vazio)."""
        dados = self._colecao.get(include=["documents", "metadatas"])
        self._construir_tfidf(dados["documents"], dados["metadatas"], dados.get("ids"))

    def _buscar_tfidf(self, pergunta, k):
        """Busca lexical pura: similaridade de cosseno entre a pergunta e os chunks indexados."""
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        if self._tfidf is None:
            self.indexar()
        vectorizer, matriz, documentos, metadados, ids = self._tfidf
        vetor_pergunta = vectorizer.transform([pergunta])
        scores = cosine_similarity(vetor_pergunta, matriz)[0]
        top_idx = np.argsort(scores)[::-1][:k]
        return [
            {"texto": documentos[i], "fonte": metadados[i]["file_name"],
             "id": ids[i] if ids else None, "score": round(float(scores[i]), 4)}
            for i in top_idx if scores[i] > 0
        ]

    # ── Busca principal: as 3 etapas descritas no docstring da classe ──────────────

    def buscar(self, pergunta, k=3, score_min=0.75, usar_rerank=True, usar_hybrid=True, k_candidatos=10):
        """Retorna os k trechos mais relevantes para a pergunta: [{texto, fonte, score}, ...].

        k_candidatos trechos são buscados nas etapas 1-2 antes do corte final; o rerank da
        etapa 3 decide os k que sobram. score_min só é usado se usar_rerank=False."""
        if self._colecao is None:
            self.indexar()

        # 1. BUSCA SEMANTICA (E5): embeda a pergunta e busca os vizinhos mais próximos.
        emb_query = self._embed_query(pergunta)
        n_buscar = max(k_candidatos, k) if (usar_rerank or usar_hybrid) else k
        res = self._colecao.query(query_embeddings=emb_query, n_results=n_buscar)

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        distancias = res["distances"][0]
        res_ids = res["ids"][0]

        candidatos = []
        vistos = set()
        for texto, meta, dist, doc_id in zip(docs, metas, distancias, res_ids):
            score_e5 = 1.0 - float(dist)
            candidatos.append({"texto": texto, "fonte": meta["file_name"], "id": doc_id, "score": round(score_e5, 4)})
            vistos.add(texto)

        # 2. BUSCA LEXICAL (TF-IDF) + COMBINACAO: une os candidatos do TF-IDF aos do E5,
        # sem duplicar um chunk que as duas buscas já tenham encontrado. Esta união é a
        # parte de engenharia própria do "hybrid search" — não vem de nenhuma biblioteca.
        if usar_hybrid:
            for c_lex in self._buscar_tfidf(pergunta, k=n_buscar):
                if c_lex["texto"] not in vistos:
                    candidatos.append(c_lex)
                    vistos.add(c_lex["texto"])

        if not usar_rerank:
            candidatos.sort(key=lambda c: c["score"], reverse=True)
            return [c for c in candidatos if c["score"] >= score_min][:k]

        # 3. RERANK (Cross-Encoder): reordena a lista combinada por relevância real
        # (pergunta + trecho lidos juntos), substituindo o score de E5/TF-IDF.
        try:
            reranker = self._carregar_reranker()
            pares = [[pergunta, c["texto"]] for c in candidatos]
            scores_rerank = reranker.predict(pares)
            for c, s in zip(candidatos, scores_rerank):
                c["score"] = round(float(s), 4)
            candidatos.sort(key=lambda c: c["score"], reverse=True)
            return candidatos[:k]
        except Exception:
            # Fallback: se o Cross-Encoder falhar (memória/pacote ausente), usa os scores
            # de retrieval (E5/TF-IDF) que já temos, sem quebrar a resposta.
            candidatos.sort(key=lambda c: c["score"], reverse=True)
            return [c for c in candidatos if c["score"] >= score_min][:k]


if __name__ == "__main__":
    rag = RAGHibrido()
    n = rag.indexar(forcar=True)
    print(f"Indexados {n} chunks (E5 + ChromaDB).\n")

    # Perguntas com SINONIMOS que o TF-IDF lexical erraria (palavra diferente da do manual):
    perguntas_teste = [
        "a maquina esta esquentando demais, e perigoso?",   # manual fala em "temperatura maxima"
        "o equipamento esta tremendo muito, o que faco?",    # manual fala em "vibracao alta"
        "como o sistema decide se um alerta e falha de verdade?",  # arquitetura em camadas
    ]
    for pergunta in perguntas_teste:
        print(f"PERGUNTA: {pergunta}")
        for d in rag.buscar(pergunta, k=2):
            print(f"  [{d['fonte']} score={d['score']}] {d['texto'][:120]}...")
        print("-" * 70)
