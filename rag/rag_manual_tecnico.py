"""
Trilha 3 - RAG sobre manual tecnico de equipamento (dado Tipo A), sem dependencias pesadas.
Adaptado do padrao validado em LIA---TRABALHO-FINAL (aplicacao_ocr.py / chat_rag.py), mas usando
TF-IDF (scikit-learn, ja disponivel) em vez de embeddings neurais (chromadb/sentence-transformers
falharam por instabilidade de rede ao baixar os pacotes ~50-100MB). Mesmos principios preservados:
- Chunking por texto corrido (nao por bloco truncado)
- Similaridade de cosseno para retrieval
- Citacao da fonte na resposta
- LLM (Ollama local) so gera a resposta final a partir do contexto recuperado -- nao alucina fonte
"""
from pathlib import Path

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MANUAIS_DIR = Path(r"C:\Projetos\Harbor\rag\manuais")
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80


def chunk_texto(texto, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chunking simples por caracteres com overlap, sobre o texto corrido (nao por paragrafo truncado)."""
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fim = min(inicio + chunk_size, len(texto))
        chunks.append(texto[inicio:fim])
        if fim == len(texto):
            break
        inicio += chunk_size - overlap
    return chunks


class RAGManualTecnico:
    def __init__(self):
        self.chunks = []
        self.metadados = []
        self.vectorizer = None
        self.matriz_tfidf = None

    def indexar(self):
        for caminho in sorted(MANUAIS_DIR.glob("*.md")):
            texto = caminho.read_text(encoding="utf-8").strip()
            partes = chunk_texto(texto)
            for i, chunk in enumerate(partes):
                self.chunks.append(chunk)
                self.metadados.append({"file_name": caminho.name, "chunk_index": i})

        self.vectorizer = TfidfVectorizer(stop_words=None, max_features=2000)
        self.matriz_tfidf = self.vectorizer.fit_transform(self.chunks)
        return len(self.chunks)

    def buscar(self, pergunta, k=3):
        vetor_pergunta = self.vectorizer.transform([pergunta])
        scores = cosine_similarity(vetor_pergunta, self.matriz_tfidf)[0]
        top_idx = np.argsort(scores)[::-1][:k]
        resultados = []
        for idx in top_idx:
            if scores[idx] <= 0:
                continue
            resultados.append({
                "texto": self.chunks[idx],
                "fonte": self.metadados[idx]["file_name"],
                "score": round(float(scores[idx]), 4),
            })
        return resultados


def call_ollama(prompt, timeout=180):
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as exc:
        return f"[Ollama indisponivel: {exc}]"


def responder(rag, pergunta):
    documentos = rag.buscar(pergunta, k=3)

    if not documentos:
        return "Nao encontrei trechos relevantes no manual tecnico para essa pergunta.", []

    contexto = "\n\n".join(f"[Fonte: {d['fonte']}]\n{d['texto']}" for d in documentos)

    prompt = f"""Voce e um assistente tecnico de manutencao industrial. Responda a pergunta usando
APENAS o contexto abaixo, extraido do manual tecnico. Cite a fonte entre colchetes ao final da
resposta. Se o contexto nao tiver a resposta, diga isso claramente -- nao invente informacao.

=== CONTEXTO (trechos do manual) ===
{contexto}

=== PERGUNTA ===
{pergunta}

Responda em portugues, de forma direta (2-4 frases), citando a fonte."""

    resposta = call_ollama(prompt)
    return resposta, documentos


def main():
    rag = RAGManualTecnico()
    n_chunks = rag.indexar()
    print(f"Indexados {n_chunks} chunks do manual tecnico.\n")

    perguntas_teste = [
        "Qual a temperatura maxima segura para os motores?",
        "O que fazer quando a vibracao esta alta?",
        "Como funciona a arquitetura de diagnostico em camadas?",
    ]

    for pergunta in perguntas_teste:
        print(f"PERGUNTA: {pergunta}")
        resposta, docs = responder(rag, pergunta)
        print(f"RESPOSTA: {resposta}")
        print(f"Fontes usadas: {[d['fonte'] + ' (score=' + str(d['score']) + ')' for d in docs]}")
        print("-" * 60)


if __name__ == "__main__":
    main()
