---
name: auditar-scripts-orfaos
description: Classifica cada script Python standalone (arquivo com "if __name__ == '__main__':") de um repositório como útil, órfão, ou arquivado intencionalmente, cruzando evidência de chamada real (import, subprocess, docstring "Uso:", entrada em documentação/skills) com sinais de abandono (docstring desatualizado, paths hardcoded de máquina específica, ausência de qualquer caminho de re-execução documentado). Use quando o usuário pedir para "ver quais scripts ainda são usados", "achar scripts órfãos", "limpar os scripts soltos", "esse script ainda roda?", "audita os scripts do projeto", ou depois de uma auditoria estrutural mais ampla (skill de refatoração/organização) que tenha marcado "preciso mais investigação" em algum arquivo executável.
---

# Auditar scripts órfãos

Esta skill classifica scripts Python standalone (arquivos com ponto de entrada de execução
direta, não módulos puros de biblioteca) em útil, órfão ou arquivado intencionalmente. Ela não
decide sozinha remover nada — entrega um diagnóstico com evidência, e a decisão final sobre
mover/deletar um script ambíguo é sempre do usuário. Não cobre arquivos sem ponto de entrada
próprio (esses são só bibliotecas importadas, fora do escopo aqui).

## Passo 1 — Levantar candidatos

Busque todo arquivo `.py` do repositório com `if __name__ == "__main__":` (ou o equivalente
idiomático da linguagem do projeto, se não for Python — o critério geral é "arquivo com ponto
de entrada de execução direta, feito para rodar como programa"). Produza a lista bruta, sem
julgar ainda. Ignore pastas de ambiente virtual/dependências instaladas.

## Passo 2 — Coletar evidência de uso, com peso por tipo de sinal

Para cada candidato, busque evidência de chamada real, na seguinte ordem de força de sinal (do
mais forte ao mais fraco):

1. **Import por outro módulo** (`import nome_do_arquivo` / `from nome_do_arquivo import ...`) —
   sinal forte; indica que o arquivo é biblioteca além de (ou em vez de) script solto. Um
   arquivo pode ter `__main__` e ainda assim ser importado por outro lugar — não assuma que
   "tem `__main__`" implica "só roda standalone".
2. **Chamada via subprocess/equivalente** a partir de código real da aplicação (não de scratch
   ou teste isolado).
3. **Comando de uso documentado** — um "Uso: python caminho/arquivo.py" no próprio docstring de
   topo, ou em README/documentação central do projeto.
4. **Entrada explícita numa skill/comando/runbook** do repositório.
5. **Menção ao nome do arquivo dentro de um comentário de outro arquivo** — sinal fraco. Pode
   ser só uma referência analógica ("assim como X faz"), não uma chamada real. Nunca conte isso
   sozinho como prova de uso — sempre cheque o contexto ao redor do match.

Um script com só sinais fracos (ou zero sinais) é candidato a órfão, mas ainda não está
confirmado — passe pelos Passos 3 e 4 antes de rotular.

## Passo 3 — Checar a categoria "arquivado intencionalmente" antes de julgar órfão

Se o script vive numa pasta com sua própria documentação local (um README, por exemplo) que
descreve um experimento ou protótipo concluído e isolado por design, classifique como
**arquivado**, não órfão. São categorias com ações diferentes: arquivado significa manter como
está, já documentado; órfão significa decidir com o usuário se remove ou atualiza. Um
experimento com zero referências externas pode ser exatamente isso por desenho — checar a
documentação local do diretório antes de julgar pela ausência de referência é obrigatório.

## Passo 4 — Ler o próprio script em busca de sinais de auto-abandono

Leia o docstring e os comentários de topo do candidato em busca de:

- Linguagem que admite estado desatualizado ("antes fazia X", "cópia de", "reescrito",
  "substituído por").
- Paths absolutos hardcoded de uma máquina/usuário específico (forte indício de script rodado
  uma vez, num ambiente específico, nunca generalizado).
- Números ou contagens que não batem com o estado atual do repositório (ex.: docstring cita "N
  itens/módulos/pipelines" quando o projeto hoje tem um número diferente).

Esses sinais aumentam a suspeita de órfão mesmo quando a contagem de referências não é
literalmente zero — um script pode ainda ser referenciado por outro código igualmente
desatualizado.

## Passo 5 — Tratar scripts de efeito one-shot como categoria própria

Scripts que produzem um efeito externo único (criar um recurso remoto, carga inicial de banco,
registro que só precisa acontecer uma vez) não têm "chamador" recorrente por design — não é
razoável esperar um import ou uma referência ativa para eles. A pergunta certa não é "quem chama
isso hoje", e sim "o efeito que este script produziu ainda existe e ainda é necessário" — isso só
o usuário sabe responder. Não classifique esse tipo de script como órfão pela ausência de
referência; relate como "setup one-shot, verificar se o efeito já foi efetivado" e pergunte.

## Passo 6 — Produzir a tabela de classificação final

Monte uma tabela com: caminho do script, evidência encontrada (citando onde e qual força de
sinal), classificação (útil / órfão / arquivado intencionalmente / setup one-shot a confirmar) e
ação sugerida. Não aplique nenhuma remoção sozinha — a skill entrega o diagnóstico. Quando o
usuário confirmar que um script é de fato órfão e deve sair, siga o mesmo cuidado de baseline
reversível de refatoração: prefira mover para fora do controle de versão ou um commit dedicado e
fácil de reverter, nunca apagar direto sem essa confirmação explícita.

## Não fazer

- Não classifique como órfão só pela contagem de referências ser zero, sem antes checar se há
  documentação local do diretório (Passo 3) e sem ler o próprio docstring (Passo 4).
- Não conte menção em comentário de outro arquivo como prova de uso — sempre leia o contexto ao
  redor do match antes de contar como sinal.
- Não trate scripts de setup one-shot (infraestrutura, carga inicial) como órfãos só porque
  ninguém os importa — eles não têm chamador recorrente por natureza.
- Não delete nada automaticamente, mesmo com alta confiança de que um script é órfão — sempre
  confirme com o usuário antes de remover do controle de versão.

## Checklist final de aceitação

- Todos os candidatos (arquivos com ponto de entrada de execução direta) foram levantados.
- Cada candidato tem evidência coletada com peso de sinal, não só contagem bruta de menções.
- As categorias "arquivado intencionalmente" e "setup one-shot" foram checadas antes de
  qualquer script ser rotulado como órfão.
- Nenhuma remoção foi feita sem confirmação explícita do usuário.
