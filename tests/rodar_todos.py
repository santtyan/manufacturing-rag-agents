"""
Runner unico da camada de verificacao do Harbor (item T1b/6b do plano de 2026-09-15).

MOTIVACAO: nao existe CI (.github/workflows/ nao existe) nem Makefile. Os testes so rodam se
alguem lembrar de rodar cada um manualmente -- e foi exatamente essa ausencia de execucao
sistematica que permitiu um teste unitario (tests/test_rag_openpack_texto.py) afirmar um bug de
vazamento de rotulo como comportamento esperado por dias sem ninguem confrontar isso com um
caso que o contradissesse. Este runner nao substitui pytest (que roda os arquivos tests/test_*
individualmente com mais detalhe) -- e um resumo rapido para rodar ANTES de qualquer commit que
toque rag/, eval/ ou pipelines/.

Composicao (nesta ordem, do mais barato ao mais caro):
1. Smoke test (tests/smoke_test.py) -- pipelines + servicos.
2. Bateria de sanidade (tests/test_sanidade_harnesses.py) -- controles positivo/negativo.
3. Checks de fidelidade de caption multimodal (eval/checks_fidelidade_caption.py).

Uso: python tests/rodar_todos.py
     python tests/rodar_todos.py --pular-servicos   (so os que nao dependem de Docker/Ollama/
                                                       Streamlit no ar)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

# Console do Windows costuma abrir em cp1252 -- forcar UTF-8 na saida deste processo tambem
# (nao so no subprocesso filho) para os textos em portugues com acento nao virarem mojibake
# ao reimprimir o stdout capturado dos scripts chamados.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent


def rodar_passo(nome: str, comando: list[str], timeout_s: int = 300) -> tuple[bool, str]:
    """Roda um comando como subprocesso, capturando stdout/stderr -- isola cada passo (se um
    travar ou lancar excecao nao tratada, os outros ainda rodam)."""
    print(f"\n{'=' * 74}\n{nome}\n{'=' * 74}")
    t0 = time.time()
    try:
        import os
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        resultado = subprocess.run(
            comando, cwd=ROOT, timeout=timeout_s, env=env,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        duracao = time.time() - t0
        # Filtra ruido comum (download de pesos HF) do stdout impresso, mas preserva no log
        # completo se precisar depurar depois.
        saida_filtrada = "\n".join(
            linha for linha in resultado.stdout.splitlines()
            if "Loading weights" not in linha and "HF Hub" not in linha
        )
        print(saida_filtrada[-3000:])  # ultimas linhas bastam para o resumo
        if resultado.returncode != 0:
            print(f"[stderr, ultimas linhas]\n{resultado.stderr[-1500:]}")
        ok = resultado.returncode == 0
        print(f"\n{'OK' if ok else 'FALHA'} ({duracao:.1f}s)")
        return ok, f"{duracao:.1f}s"
    except subprocess.TimeoutExpired:
        print(f"\nTIMEOUT apos {timeout_s}s")
        return False, "timeout"
    except Exception as e:
        print(f"\nERRO ao rodar: {type(e).__name__}: {e}")
        return False, "erro"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pular-servicos", action="store_true",
                         help="Pula o smoke test completo (que exige Docker/Ollama/Streamlit no "
                              "ar) e roda so sanidade + fidelidade -- util para checagem rapida "
                              "antes de commit, sem precisar subir toda a infraestrutura.")
    args = parser.parse_args()

    resultados = {}

    if not args.pular_servicos:
        ok, info = rodar_passo(
            "1/3 -- Smoke test (pipelines + servicos)",
            [sys.executable, "tests/smoke_test.py"], timeout_s=180,
        )
        resultados["smoke_test"] = (ok, info)
    else:
        print("\n[1/3 -- Smoke test PULADO (--pular-servicos)]")
        resultados["smoke_test"] = (None, "pulado")

    ok, info = rodar_passo(
        "2/3 -- Bateria de sanidade (controles positivo/negativo)",
        [sys.executable, "tests/test_sanidade_harnesses.py"], timeout_s=300,
    )
    resultados["sanidade"] = (ok, info)

    caminho_legendas = ROOT / "rag" / "legendas_deterministicas.json"
    if caminho_legendas.exists():
        ok, info = rodar_passo(
            "3/4 -- Checks de fidelidade de caption multimodal",
            [sys.executable, "eval/checks_fidelidade_caption.py", str(caminho_legendas)],
            timeout_s=60,
        )
        resultados["fidelidade_caption"] = (ok, info)

        ok, info = rodar_passo(
            "4/4 -- Proveniencia numerica das legendas (item 5.5)",
            [sys.executable, "eval/teste_provenencia_numerica_legendas.py"],
            timeout_s=60,
        )
        resultados["provenencia_numerica"] = (ok, info)
    else:
        print(f"\n[3-4/4 -- PULADO: {caminho_legendas} nao existe]")
        resultados["fidelidade_caption"] = (None, "arquivo ausente")
        resultados["provenencia_numerica"] = (None, "arquivo ausente")

    print(f"\n{'=' * 74}\nRESUMO\n{'=' * 74}")
    houve_falha = False
    for nome, (ok, info) in resultados.items():
        if ok is None:
            simbolo = "-"
        elif ok:
            simbolo = "OK"
        else:
            simbolo = "FALHA"
            houve_falha = True
        print(f"  [{simbolo:5}] {nome} ({info})")

    if houve_falha:
        print("\nHA FALHA(S) -- nao commitar ate investigar (ver output detalhado acima).")
        sys.exit(1)
    else:
        print("\nTudo verde (ou pulado deliberadamente).")


if __name__ == "__main__":
    main()
