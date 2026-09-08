"""Wrapper unico para chamadas ao Ollama local (Fase 2 do plano de disciplina
experimental do PDC: instrumentar custo + pin de versao).

Antes deste modulo, 13 pontos do Harbor chamavam `requests.post` para
`/api/generate` e descartavam a resposta com `.get("response", "")` -- em 3
deles (`dashboard/app.py`, `eval/rodar_golden.py`, `eval/rodar_golden_qwen.py`)
o dict completo ja era materializado em memoria e jogado fora de qualquer
jeito. O Ollama sempre retornou `eval_count` (tokens de saida),
`prompt_eval_count` (tokens de entrada) e `total_duration` (nanossegundos) --
nenhum desses campos aparecia em lugar nenhum do repositorio.

`chamar()` devolve uma string (mesma assinatura de comportamento que os
`call_ollama()` duplicados ja tinham, para nao quebrar call sites de uma vez)
E anexa os metadados como atributos na propria string via subclasse `str`
(`RespostaOllama`) -- assim quem so quer o texto continua funcionando sem
mudar nada, e quem quer instrumentar (Fase 2+) acessa `.metadados`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass
class MetadadosOllama:
    """Espelha os nomes gen_ai.* da convencao OpenTelemetry GenAI (ver
    shared/trace.py para a justificativa completa dessa escolha)."""

    gen_ai_request_model: str
    gen_ai_response_model: str
    gen_ai_usage_input_tokens: int
    gen_ai_usage_output_tokens: int
    duracao_s: float


class RespostaOllama(str):
    """String com o texto da resposta, carregando `.metadados` (MetadadosOllama)
    como atributo extra. Compativel com todo codigo existente que trata o
    retorno de call_ollama como string pura."""

    metadados: MetadadosOllama


def chamar(
    prompt: str,
    modelo: str,
    timeout: int = 120,
    formato: dict | str | None = None,
    temperature: float | None = None,
    url: str = OLLAMA_URL,
) -> RespostaOllama:
    """Chamada instrumentada ao Ollama. Substitui as implementacoes duplicadas
    de call_ollama() espalhadas pelo projeto -- ver Fase 2 do plano de
    disciplina experimental (shared/trace.py para o contexto completo)."""
    body: dict = {"model": modelo, "prompt": prompt, "stream": False}
    if formato is not None:
        body["format"] = formato
    if temperature is not None:
        body["options"] = {"temperature": temperature}

    inicio = time.monotonic()
    resp = requests.post(url, json=body, timeout=timeout)
    duracao = time.monotonic() - inicio
    resp.raise_for_status()
    corpo = resp.json()
    if "error" in corpo:
        # Ollama as vezes retorna HTTP 200 com um campo "error" no corpo (ex: modelo
        # nao carregado por falta de memoria) -- raise_for_status() sozinho nao pega isso.
        raise RuntimeError(corpo["error"])

    texto = RespostaOllama(corpo.get("response", "").strip())
    texto.metadados = MetadadosOllama(
        gen_ai_request_model=modelo,
        gen_ai_response_model=corpo.get("model", modelo),
        gen_ai_usage_input_tokens=corpo.get("prompt_eval_count", 0),
        gen_ai_usage_output_tokens=corpo.get("eval_count", 0),
        duracao_s=duracao,
    )
    return texto
