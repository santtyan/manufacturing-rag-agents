"""
RAG multimodal (texto + imagem) -- smoke test, skill rag-multimodal.

Arquitetura: caption-then-embed (ver .claude/skills/rag-multimodal/SKILL.md e
references/resumo_rag_multimodal_2025_2026.md para o porque desta escolha, nao
embeddings compartilhados tipo CLIP). Um VLM local via Ollama gera uma legenda textual de
cada imagem; a legenda vira Document de texto e entra no MESMO pipeline LangChain ja em
producao (RAGHibrido de rag_hibrido_langchain.py, via seu parametro documentos_customizados),
numa colecao SEPARADA da producao ate ser medido (mesmo cuidado ja tomado com
chroma_db_langchain durante a migracao do RAG hibrido).

ACHADO REAL (2026-09-08): qwen2.5vl (modelo recomendado pela pesquisa) falhou ao carregar o
projetor multimodal via Ollama nesta maquina --
"Failed to load CLIP model ... llama-server process has terminated: exit status 1" -- confirma
o risco de compatibilidade ja documentado na skill. Fallback usado: moondream (~1.7GB). Captioning
com moondream e funcional mas INCONSISTENTE/POBRE em 2 testes manuais (uma vez so descreveu
"gráfico de barras roxo", outra vez leu o titulo mas nao os rotulos dos eixos) -- limitacao
esperada de um VLM pequeno (<2B), documentada no achado 2 da pesquisa. Nao usar isto como
avaliacao final da arquitetura caption-then-embed -- reavaliar com um VLM maior assim que houver
espaco em disco/compatibilidade Ollama resolvida (ver SKILL.md item 2).

ACHADO REAL #2 (2026-09-08): importar langchain_ollama (ChatOllama) e depois
sentence-transformers/HuggingFaceEmbeddings NO MESMO PROCESSO causa segmentation fault (exit
139) nesta maquina -- reproduzido de forma isolada, o crash acontece so com os dois imports
presentes, mesmo sem nenhuma chamada de rede. Provavel conflito de biblioteca nativa
compartilhada entre o cliente do Ollama e o backend de tensor (torch/onnxruntime). Por isso
TODOS os imports de langchain_ollama/RAGHibrido sao feitos DENTRO das funcoes que os usam
(nunca no topo do modulo) -- rodar `--captionar` (so Ollama) e a indexacao/busca (so
sentence-transformers) como duas invocacoes de processo Python SEPARADAS, nunca no mesmo
processo. Ver bloco __main__.
"""
import base64
from pathlib import Path

IMAGENS_DIR = Path(r"C:\Projetos\Harbor\rag\manuais_imagens")
CHROMA_DIR = Path(r"C:\Projetos\Harbor\rag\chroma_db_multimodal")
COLECAO = "imagens_harbor_multimodal_v1"
MODELO_VLM = "moondream"  # fallback -- qwen2.5vl falhou ao carregar o projetor (ver docstring)

PROMPT_CAPTION = (
    "Descreva este gráfico técnico em português, de forma objetiva, mencionando: "
    "o tipo de gráfico, o título/eixos, e quais categorias/componentes aparecem com "
    "valores mais altos ou mais baixos. Máximo 4 frases."
)


def gerar_legenda(caminho_imagem, modelo=MODELO_VLM):
    """Chama o VLM local via ChatOllama para descrever uma imagem -- passo 'caption' do
    caption-then-embed. Retorna a legenda como string de texto puro.

    Import de langchain_ollama feito AQUI DENTRO (nao no topo do modulo) -- ver ACHADO REAL #2
    na docstring do modulo: importar isso junto com sentence-transformers no mesmo processo
    causa segmentation fault nesta maquina."""
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage

    with open(caminho_imagem, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    llm = ChatOllama(model=modelo, temperature=0.1)
    msg = HumanMessage(content=[
        {"type": "text", "text": PROMPT_CAPTION},
        {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"},
    ])
    resposta = llm.invoke([msg])
    return resposta.content.strip()


def gerar_legendas(pasta=IMAGENS_DIR):
    """Gera legenda de cada imagem em `pasta` via VLM (Ollama) e retorna a lista de
    documentos_customizados pronta para indexar. Etapa SEPARADA de indexar_imagens() -- achado
    real (2026-09-08): rodar captioning via Ollama e depois carregar sentence-transformers/torch
    NO MESMO PROCESSO causa segmentation fault (exit 139) nesta maquina, mesmo com cada etapa
    funcionando perfeitamente isolada. Causa provavel: conflito de alocacao nativa entre o
    cliente HTTP do Ollama e o backend de tensor do PyTorch carregado depois. Rodar como dois
    processos Python separados evita o problema -- ver bloco __main__."""
    documentos_customizados = []
    for caminho in sorted(pasta.glob("*.png")):
        legenda = gerar_legenda(caminho)
        documentos_customizados.append({
            "id": caminho.stem,
            "texto": legenda,
            "fonte": caminho.name,
        })
        print(f"[{caminho.name}] legenda: {legenda[:100]}...")
    return documentos_customizados


def indexar_imagens(documentos_customizados, forcar=False):
    """Indexa documentos_customizados (ja com legenda gerada por gerar_legendas()) no pipeline
    LangChain de producao (RAGHibrido), numa colecao separada (CHROMA_DIR/COLECAO) para nao
    misturar com os manuais de producao. Retorna a instancia RAGHibrido pronta para buscar().

    Import de RAGHibrido feito AQUI DENTRO (nao no topo do modulo) -- mesmo motivo do import
    tardio de langchain_ollama em gerar_legenda(), ver ACHADO REAL #2 na docstring do modulo."""
    from rag_hibrido_langchain import RAGHibrido
    rag = RAGHibrido(chroma_dir=CHROMA_DIR, colecao=COLECAO)
    rag.indexar(forcar=forcar, documentos_customizados=documentos_customizados)
    return rag


if __name__ == "__main__":
    import sys
    import json

    CACHE_LEGENDAS = Path(r"C:\Projetos\Harbor\rag\legendas_cache.json")

    if len(sys.argv) > 1 and sys.argv[1] == "--captionar":
        # Etapa 1 (processo A): so gera e salva as legendas, sem tocar em torch/sentence-transformers.
        docs = gerar_legendas()
        CACHE_LEGENDAS.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n{len(docs)} legendas salvas em {CACHE_LEGENDAS}")
    else:
        # Etapa 2 (processo B): le as legendas do cache e indexa/busca, sem tocar em Ollama.
        docs = json.loads(CACHE_LEGENDAS.read_text(encoding="utf-8"))
        rag = indexar_imagens(docs, forcar=True)
        print(f"\nIndexadas {len(docs)} legendas de imagem.\n")

        perguntas_teste = [
            "qual componente do CNC teve mais anomalias de temperatura?",
            "existe alguma diferença de vibração entre operação normal e com falha?",
            "mostre um gráfico relacionando pressão e vazão",
        ]
        for pergunta in perguntas_teste:
            print(f"PERGUNTA: {pergunta}")
            for d in rag.buscar(pergunta, k=2, usar_rerank=False):
                print(f"  [{d['fonte']} score={d['score']}] {d['texto'][:150]}")
            print("-" * 70)
