"""Gera um Rollout Card (relatorio de replicacao) em Markdown a partir de uma
colecao de traces JSONL (Fase 1 do plano de disciplina experimental do PDC).

Rollout Cards (arXiv:2605.12131) e o padrao proposto para RELATAR uma serie de
execucoes de agente, no espirito de Model Cards: modelo+versao, prompt, tools,
dependencias, custo, e o que quebrou - resolvendo o problema de que scaffolds
de agente (modelo + prompt + tools + retry logic + ambiente) sao sensiveis a
implementacao, e mudar qualquer peca muda o resultado medido sem que isso
fique registrado em lugar nenhum.

Trace (shared/trace.py) registra UMA execucao; Rollout Card resume MUITAS
execucoes para leitura humana. Os dois sao artefatos distintos e complementares
- ver a regra dos "tres artefatos" (codigo, trace, relatorio) documentada no
plano e nas skills padrao-react-raciocinio-acao / migrar-para-langchain.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from shared.trace import carregar_traces


def _custo_total(traces: list[dict[str, Any]]) -> dict[str, int]:
    total_in = sum(t.get("custo_total", {}).get("input_tokens", 0) for t in traces)
    total_out = sum(t.get("custo_total", {}).get("output_tokens", 0) for t in traces)
    return {"input_tokens": total_in, "output_tokens": total_out}


def _distribuicao_falha(traces: list[dict[str, Any]]) -> Counter:
    """Classifica cada trace com sucesso=False por motivo aproximado, usando
    o ultimo passo como pista. Heuristica simples de primeira passada -
    suficiente para o relatorio, nao para analise fina (usar os traces
    brutos para isso)."""
    motivos: Counter = Counter()
    for t in traces:
        if t.get("sucesso"):
            continue
        passos = t.get("passos", [])
        if not passos:
            motivos["sem_passos"] += 1
            continue
        ultimo = passos[-1]
        obs = (ultimo.get("observation") or "").lower()
        if "erro" in obs or "exception" in obs or "syntax" in obs:
            motivos["acao_invalida_ou_erro_execucao"] += 1
        elif not obs.strip():
            motivos["sem_observacao_util"] += 1
        else:
            motivos["outro"] += 1
    return motivos


def gerar_rollout_card(
    tarefa: str,
    arquivo_traces: Path | str | None = None,
    dependencias: dict[str, str] | None = None,
    tools: list[str] | None = None,
    prompt_resumo: str = "",
) -> str:
    """Monta o Rollout Card em Markdown para todos os traces de uma tarefa."""
    caminho = Path(arquivo_traces) if arquivo_traces else (
        Path(__file__).resolve().parent.parent / "eval" / "traces" / f"{tarefa}.jsonl"
    )
    traces = carregar_traces(caminho)

    if not traces:
        return f"# Rollout Card - {tarefa}\n\nNenhum trace encontrado em `{caminho}`.\n"

    n = len(traces)
    sucessos = sum(1 for t in traces if t.get("sucesso"))
    taxa_sucesso = sucessos / n * 100
    custo = _custo_total(traces)
    duracoes = [t.get("duracao_total_s", 0.0) for t in traces]
    duracao_media = sum(duracoes) / n if n else 0.0
    modelos = Counter(t.get("modelo", "?") for t in traces)
    n_passos = [len(t.get("passos", [])) for t in traces]
    media_passos = sum(n_passos) / n if n else 0.0
    falhas = _distribuicao_falha(traces)

    linhas = [
        f"# Rollout Card - {tarefa}",
        "",
        f"Gerado a partir de `{caminho}` ({n} execucoes).",
        "",
        "## Modelo e versoes",
        "",
        f"- Modelo(s): {', '.join(f'{m} ({c}x)' for m, c in modelos.items())}",
    ]
    if dependencias:
        linhas.append("- Dependencias:")
        for pacote, versao in dependencias.items():
            linhas.append(f"  - {pacote}: {versao}")
    if tools:
        linhas.append(f"- Tools disponiveis ao agente: {', '.join(tools)}")
    if prompt_resumo:
        linhas += ["", "## Prompt (resumo)", "", prompt_resumo]

    linhas += [
        "",
        "## Resultado",
        "",
        f"- Execucoes: {n}",
        f"- Taxa de sucesso: {taxa_sucesso:.1f}% ({sucessos}/{n})",
        f"- Passos por execucao (media): {media_passos:.1f}",
        f"- Duracao por execucao (media): {duracao_media:.2f}s",
        "",
        "## Custo (tokens, somado sobre todas as execucoes)",
        "",
        f"- Input: {custo['input_tokens']}",
        f"- Output: {custo['output_tokens']}",
        f"- Total: {custo['input_tokens'] + custo['output_tokens']}",
    ]

    if falhas:
        linhas += ["", "## O que quebrou (execucoes sem sucesso)", ""]
        for motivo, contagem in falhas.most_common():
            linhas.append(f"- {motivo}: {contagem}")

    linhas += [
        "",
        "## Reprodutibilidade",
        "",
        "- Trace bruto (JSONL) disponivel em `" + str(caminho) + "` - cada linha e uma "
        "execucao completa, com thought/action/observation por passo, custo e modelo "
        "usados naquele passo especifico.",
        "- Regra do projeto: nenhuma conclusao sobre este resultado e valida sem o "
        "baseline comparavel a custo de tokens normalizado (ver skill "
        "`migrar-para-langchain`, secao de baseline justo).",
    ]

    return "\n".join(linhas) + "\n"


def salvar_rollout_card(tarefa: str, destino: Path | str | None = None, **kwargs: Any) -> Path:
    conteudo = gerar_rollout_card(tarefa, **kwargs)
    caminho = Path(destino) if destino else (
        Path(__file__).resolve().parent.parent / "eval" / "traces" / f"{tarefa}_rollout_card.md"
    )
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(conteudo, encoding="utf-8")
    return caminho
