"""
Answer Relevancy (item 7.2 do plano de 2026-09-15, implementado em 2026-09-25).

MOTIVACAO: o harness mede Faithfulness (os numeros esperados aparecem na resposta?) e
Hallucination Rate (a resposta cita numero proibido/nao-fundamentado?), mas nenhuma das duas
mede se a resposta de fato RESPONDE a pergunta -- uma resposta pode ser 100% fiel aos numeros
do contexto e ainda ser tangencial ou generica o suficiente para nao satisfazer o que foi
perguntado. Isso e a lacuna nomeada na categoria "LLM" da tabela de metricas CERISE
(skill metricas-avaliacao-ia-industrial): "Faithfulness sozinha nao conclui".

METODO (RAGAS, Es et al. 2023, arXiv:2309.15217): gerar N perguntas artificiais a partir da
resposta via LLM (perguntas que a resposta responderia bem), embeddar cada uma com o mesmo
modelo de embedding ja usado em producao (intfloat/multilingual-e5-small, ver
rag/rag_hibrido.py::MODELO_EMBEDDING -- reuso deliberado, nao uma segunda dependencia de
embedding), e medir a similaridade de cosseno media entre elas e a pergunta ORIGINAL. Score
baixo = a resposta e generica/nao-comprometida (o LLM nao consegue reconstruir perguntas
parecidas com a original a partir dela); score alto = a resposta e especifica o suficiente
para "apontar de volta" para o que foi perguntado.

Nao usa LLM-judge de score direto (0-10) por ser mais sensivel a variancia entre modelos/runs
que embedding similarity -- mesma razao documentada no achado real de 2026-09-09 (score de
Cross-Encoder e pessimo preditor de acerto; medir por proxy de embedding e mais estavel que
pedir a um LLM pequeno para "notar" a propria resposta).

Uso: python eval/answer_relevancy.py (roda uma demonstracao standalone sobre 3 exemplos fixos)
Uso como biblioteca: from answer_relevancy import answer_relevancy_score
"""
import json
import re
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent / "shared"))
from ollama_client import chamar as _chamar_ollama  # noqa: E402

N_PERGUNTAS_GERADAS = 3
OLLAMA_URL = "http://localhost:11434/api/generate"
MODELO_JUIZ_DEFAULT = "qwen2.5:7b"  # mesmo modelo usado como juiz em avaliar_rag_agentic.py

_MODELO_EMBEDDING_CACHE = None


def _carregar_modelo_embedding():
    """Carrega intfloat/multilingual-e5-small (mesmo modelo de rag/rag_hibrido.py) uma unica
    vez por processo -- import tardio para nao pagar custo de sentence-transformers em quem
    so importa este modulo para outra funcao."""
    global _MODELO_EMBEDDING_CACHE
    if _MODELO_EMBEDDING_CACHE is None:
        import datasets  # noqa: F401 -- protecao de ordem de import (access violation
        # torch x pyarrow no Windows, ver aprendizados_sessao_2026-09-11_12)
        from sentence_transformers import SentenceTransformer
        _MODELO_EMBEDDING_CACHE = SentenceTransformer("intfloat/multilingual-e5-small")
    return _MODELO_EMBEDDING_CACHE


def _gerar_perguntas_a_partir_da_resposta(resposta, n=N_PERGUNTAS_GERADAS,
                                           modelo=MODELO_JUIZ_DEFAULT, url=OLLAMA_URL):
    """Pede ao LLM juiz para gerar N perguntas que a resposta dada responderia bem --
    formato JSON-schema para parsing robusto (mesmo padrao de rotear_por_llm())."""
    prompt = f"""Dada a resposta abaixo, gere {n} perguntas DIFERENTES que essa resposta
responderia bem. As perguntas devem ser especificas o suficiente para que so fariam sentido
para ESSA resposta, nao perguntas genericas que qualquer resposta responderia.

Resposta: {resposta}

Responda em JSON com a chave "perguntas" (lista de {n} strings, em portugues)."""
    formato = {
        "type": "object",
        "properties": {
            "perguntas": {"type": "array", "items": {"type": "string"}, "minItems": n, "maxItems": n},
        },
        "required": ["perguntas"],
    }
    try:
        resp = _chamar_ollama(prompt, modelo=modelo, timeout=60, formato=formato, url=url)
        dados = json.loads(str(resp) or "{}")
        perguntas = dados.get("perguntas", [])
        return [p for p in perguntas if isinstance(p, str) and p.strip()]
    except Exception as exc:
        print(f"[answer_relevancy] falha ao gerar perguntas: {exc}")
        return []


def _similaridade_cosseno(modelo, pergunta_original, perguntas_geradas):
    """Similaridade de cosseno media entre a pergunta original e as perguntas geradas,
    usando o mesmo prefixo query:/passage: do E5 (ambas sao 'query' aqui -- comparacao
    simetrica pergunta-pergunta, nao pergunta-documento)."""
    import numpy as np
    if not perguntas_geradas:
        return 0.0
    textos = [f"query: {pergunta_original}"] + [f"query: {p}" for p in perguntas_geradas]
    embeddings = modelo.encode(textos, normalize_embeddings=True)
    emb_original = embeddings[0]
    emb_geradas = embeddings[1:]
    similaridades = emb_geradas @ emb_original  # ja normalizado -> produto interno = cosseno
    return float(np.mean(similaridades))


def answer_relevancy_score(pergunta_original, resposta, n_perguntas=N_PERGUNTAS_GERADAS,
                            modelo_juiz=MODELO_JUIZ_DEFAULT, url=OLLAMA_URL):
    """Retorna (score, perguntas_geradas). Score em [0,1] (cosseno normalizado, tipicamente
    positivo com E5 multilingue) -- quanto maior, mais a resposta "aponta de volta" para o
    que foi perguntado. Score baixo/negativo sinaliza resposta generica ou fora do assunto."""
    if not resposta or not resposta.strip():
        return 0.0, []
    perguntas_geradas = _gerar_perguntas_a_partir_da_resposta(
        resposta, n=n_perguntas, modelo=modelo_juiz, url=url)
    if not perguntas_geradas:
        return 0.0, []
    modelo_emb = _carregar_modelo_embedding()
    score = _similaridade_cosseno(modelo_emb, pergunta_original, perguntas_geradas)
    return score, perguntas_geradas


def _remover_pensamento(texto):
    """Remove blocos <think>...</think> que alguns modelos (ex. deepseek-r1) emitem antes
    da resposta final -- nao deveriam contaminar nem a geracao de perguntas nem o embedding."""
    return re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL).strip()


if __name__ == "__main__":
    exemplos = [
        {
            "pergunta": "Qual o limiar de temperatura que classifica uma leitura como crítica?",
            "resposta": "Segundo o manual, leituras de Temperature_C acima de 90°C são "
                        "classificadas como críticas (seção 8 de criticidade).",
        },
        {
            "pergunta": "Qual o limiar de temperatura que classifica uma leitura como crítica?",
            "resposta": "O sistema monitora diversos sensores industriais para garantir a "
                        "segurança da operação e evitar falhas.",
        },
        {
            "pergunta": "Qual foi a categoria de parada que mais consumiu minutos?",
            "resposta": "A categoria FAILURE consumiu 3233,6 minutos no total, somando todas "
                        "as StopLocation dessa categoria.",
        },
    ]
    print("Demonstracao standalone (3 exemplos, Ollama precisa estar no ar):\n")
    for ex in exemplos:
        score, perguntas = answer_relevancy_score(ex["pergunta"], ex["resposta"])
        print(f"Pergunta original : {ex['pergunta']}")
        print(f"Resposta          : {ex['resposta'][:80]}...")
        print(f"Perguntas geradas : {perguntas}")
        print(f"Answer Relevancy  : {score:.4f}")
        print("-" * 74)
