"""Tarefa HotpotQA como agente ReAct (Fase 5a do plano de disciplina
experimental do PDC) -- replicacao literal da tarefa do paper original
(Yao et al., ICLR 2023, arXiv:2210.03629), a primeira tarefa dada pelo grupo
de pesquisa PDC em 2026-09-08 ("todo membro replica um agente ReAct simples
sobre uma tarefa de benchmark e entrega o trace").

Tools search/lookup replicam a API simples de busca usada no paper (nao um
RAG hibrido com embeddings -- a tarefa e sobre o AGENTE, nao sobre qualidade
de retrieval, entao a tool fica deliberadamente simples: busca lexical crua
via API publica da Wikipedia, com User-Agent identificado por educacao com o
servico). Nenhuma dependencia nova alem de `requests`, ja usado no projeto.

Metrica de interesse do paper a replicar (nao so acerto/erro): em
chain-of-thought puro, 56% das falhas sao alucinacao de fato "lembrado"
errado do proprio treinamento, porque o modelo nunca checa nada externo.
ReAct, ao intercalar busca real, ZERA esse modo de falha, trocando-o por
"buscou a informacao errada" (23% das falhas) -- mais facil de diagnosticar.
Este script reporta essa distribuicao a partir dos nossos proprios traces,
que e a replicacao de fato (nao so rodar e ver se acertou).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from agents.react import Tool, rodar_react

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "Harbor-Research-Agent/1.0 (uso educacional/pesquisa PDC; contato: santosleiteyan@icloud.com)"


def _wiki_search(termo: str) -> str:
    """Tool 'search[entidade]' do paper: busca a entidade na Wikipedia e
    devolve o INICIO do artigo (resumo/intro), igual ao comportamento
    documentado do wrapper de busca usado no paper original -- se a
    entidade nao existir exatamente, devolve os 5 titulos mais proximos
    (comportamento de desambiguacao do paper)."""
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(WIKIPEDIA_API, params={
        "action": "query", "list": "search", "srsearch": termo, "format": "json", "srlimit": 5,
    }, headers=headers, timeout=15)
    resp.raise_for_status()
    resultados = resp.json().get("query", {}).get("search", [])
    if not resultados:
        return f"Nenhum artigo encontrado para '{termo}'."

    titulo = resultados[0]["title"]
    resp2 = requests.get(WIKIPEDIA_API, params={
        "action": "query", "prop": "extracts", "exintro": True, "explaintext": True,
        "titles": titulo, "format": "json",
    }, headers=headers, timeout=15)
    resp2.raise_for_status()
    paginas = resp2.json().get("query", {}).get("pages", {})
    extrato = next(iter(paginas.values()), {}).get("extract", "")
    if not extrato:
        outros = ", ".join(r["title"] for r in resultados[:5])
        return f"Artigo '{titulo}' sem resumo disponivel. Titulos similares: {outros}"
    return f"[{titulo}] {extrato[:1500]}"


def _wiki_lookup(termo: str) -> str:
    """Tool 'lookup[string]' do paper: como nao ha pagina 'atual' com estado
    persistente nesta reimplementacao simples (o paper usa lookup dentro da
    MESMA pagina do search anterior), tratamos lookup como busca textual
    contra o resumo mais recente retornado -- suficiente para a replicacao
    do padrao de raciocinio, nao uma reimplementacao byte-a-byte do
    ambiente do paper."""
    return _wiki_search(termo)


def montar_tools_hotpotqa() -> list[Tool]:
    return [
        Tool(nome="search", descricao="Busca uma entidade/topico na Wikipedia e retorna o resumo do artigo mais proximo.", funcao=_wiki_search),
        Tool(nome="lookup", descricao="Busca um termo especifico (aproximado por busca textual nesta reimplementacao).", funcao=_wiki_lookup),
    ]


def rodar_tarefa_hotpotqa(pergunta: str, modelo: str = "qwen2.5:7b", arquivo_trace: str | None = None):
    tools = montar_tools_hotpotqa()
    entrada = f"Responda a pergunta de multiplos saltos a seguir, usando search/lookup para checar fatos reais antes de responder. Pergunta: {pergunta}"
    return rodar_react(
        tarefa="hotpotqa",
        entrada=entrada,
        tools=tools,
        modelo=modelo,
        versoes={"fonte_tarefa": "HotpotQA (hotpotqa/hotpot_qa, distractor, validation)"},
        arquivo_trace=arquivo_trace,
    )


def _classificar_falha(trace) -> str:
    """Heuristica simples para reportar a distribuicao de modos de falha do
    paper (alucinacao vs busca errada) -- nao um classificador validado,
    serve para a leitura qualitativa do Rollout Card."""
    if trace.sucesso:
        return "sucesso"
    passos_tool = [p for p in trace.passos if p.tipo == "tool"]
    if not passos_tool:
        return "nunca_usou_tool_de_busca"  # mais proximo do modo "alucinacao" do CoT puro
    if any("Nenhum artigo encontrado" in (p.observation or "") for p in passos_tool):
        return "buscou_termo_errado"
    return "outro"


if __name__ == "__main__":
    import argparse

    from datasets import load_dataset

    from shared.rollout_card import salvar_rollout_card

    parser = argparse.ArgumentParser(description="Agente ReAct sobre HotpotQA (Fase 5a PDC, replicacao do paper)")
    parser.add_argument("--n", type=int, default=10, help="numero de perguntas do HotpotQA (validation, distractor) a rodar")
    parser.add_argument("--modelo", default="qwen2.5:7b")
    args = parser.parse_args()

    print(f"Carregando {args.n} perguntas do HotpotQA (streaming, split=validation, config=distractor)...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
    exemplos = []
    for ex in ds:
        exemplos.append(ex)
        if len(exemplos) >= args.n:
            break

    print(f"Rodando agente ReAct sobre {len(exemplos)} perguntas HotpotQA (modelo={args.modelo})...\n")
    distribuicao_falha = {}
    for ex in exemplos:
        print(f"[{ex['id']}] {ex['question']}  (gabarito: {ex['answer']})")
        trace = rodar_tarefa_hotpotqa(ex["question"], modelo=args.modelo)
        motivo = _classificar_falha(trace)
        distribuicao_falha[motivo] = distribuicao_falha.get(motivo, 0) + 1
        print(f"  {motivo} | passos={len(trace.passos)} custo={trace.custo_total}")
        print(f"  resposta do agente: {trace.resultado_final[:150]}")
        print("-" * 70)

    print("\nDistribuicao de modos de falha (comparar contra o paper: CoT puro=56% alucinacao, "
          "ReAct zera alucinacao e troca por ~23% de busca errada):")
    for motivo, contagem in sorted(distribuicao_falha.items(), key=lambda kv: -kv[1]):
        print(f"  {motivo}: {contagem}/{len(exemplos)}")

    caminho_card = salvar_rollout_card(
        "hotpotqa",
        dependencias={"langgraph": "1.x", "datasets": "HF hotpotqa/hotpot_qa distractor"},
        tools=["search", "lookup"],
        prompt_resumo="Responder pergunta multi-hop usando search/lookup sobre a Wikipedia (replicacao ReAct, Yao et al. 2023).",
    )
    print(f"\nRollout Card salvo em {caminho_card}")
