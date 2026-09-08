---
name: revisar-funcoes-orfas
description: Revisa um arquivo ou pasta (indicado pelo usuário, não o repositório inteiro por padrão) linha por linha em busca de funções/métodos definidos mas nunca chamados em nenhum lugar, e reporta cada candidato com a evidência da busca — nunca remove nada sozinha. Cuida especificamente de três fontes conhecidas de falso positivo: despacho por dicionário (função só referenciada como valor, nunca como chamada direta), métodos homônimos entre classes distintas, e funções com underscore importadas cross-file apesar da convenção de "privado ao módulo". Use quando o usuário pedir para "revisar esse arquivo linha por linha", "achar funções órfãs", "tem função sem uso nesse módulo?", "limpar funções mortas", ou depois de auditar-scripts-orfaos ter marcado um arquivo como útil mas o usuário suspeitar que nem todo o conteúdo interno dele é usado.
---

# Revisar funções órfãs

Esta skill analisa o arquivo ou pasta que o usuário indicar — nunca o repositório inteiro por
padrão, porque revisão linha por linha é cara e deve ser direcionada. Ela identifica funções e
métodos definidos mas nunca chamados em lugar nenhum, e nunca remove nada sozinha, mesmo com
alta confiança. Complementa (não substitui) uma skill de auditoria de scripts inteiros: aquela
decide se um ARQUIVO é útil ou órfão; esta decide, dentro de um arquivo já considerado útil, se
alguma FUNÇÃO específica dele deixou de ser usada.

## Passo 1 — Listar todas as definições do escopo

Levante todo `def` de função ou método dentro do arquivo ou pasta indicada, com nome, linha, e
se é método de classe (indentado sob `class`) ou função solta de módulo — essa distinção importa
para o Passo 2.

## Passo 2 — Buscar evidência de chamada em camadas, na ordem abaixo

Para cada definição, pare na primeira camada que confirmar uso:

1. **Chamada direta `nome(` dentro do próprio arquivo.**
2. **Chamada direta `nome(`, `.nome(` ou `self.nome(` em qualquer outro arquivo do repositório**
   — busca cross-file. Funções com prefixo underscore (convenção "privada ao módulo") NÃO estão
   isentas dessa busca: a convenção pode não ser respeitada de fato, então confirme sempre.
3. **Referência como valor de estrutura de despacho** — busque o nome próximo de um `= {` ou
   `dict(` (como `nome,`, `nome}`, ou `"chave": nome`). Se encontrar, confirme manualmente que a
   estrutura é de fato usada com indexação dinâmica (algo como `dicionario[chave](...)`) antes
   de aceitar isso como prova de uso — uma menção solta no dicionário sem uso posterior não
   conta.
4. **Para métodos de classe, ao achar um match cross-file**: confirme que o match pertence à
   MESMA classe, não a uma classe homônima definida em outro arquivo com métodos de mesmo nome.
   Leia a classe do arquivo onde o match apareceu antes de aceitar como prova de uso — nomes de
   método curtos e genéricos colidem com frequência entre classes não relacionadas.

## Passo 3 — Tratar como uso válido, nunca como órfão

- Chamada dentro de `if __name__ == "__main__":` do próprio arquivo, incluindo `main()`.
- Função decorada por um framework que a invoca por mecanismo próprio, não por chamada textual
  direta (rotas web, cache do framework, registro de ferramenta/plugin). Confirme que o
  decorator é de fato desse tipo antes de aplicar a isenção — não trate qualquer decorator como
  prova automática de uso.
- Função exportada publicamente (sem underscore) de um módulo que o projeto trata como
  biblioteca compartilhada, destinada a ser importada por consumidores futuros — mesmo sem uso
  interno atual, verifique se há uma razão de API pública antes de sugerir remoção.

## Passo 4 — Reportar os candidatos confirmados

Só reporte como candidato a órfão uma função que não teve nenhum match em nenhuma das 4 camadas
do Passo 2 E que não se enquadra em nenhuma exceção do Passo 3. Para cada candidato, informe:
nome, arquivo, linha, um resumo de uma frase do que a função faz, e qual busca não encontrou
nada — para o usuário poder auditar o raciocínio, não só aceitar a conclusão.

## Passo 5 — Pedir confirmação função por função antes de remover

Nunca remova em lote sem revisão individual — cada função candidata é uma decisão separada.
Ao remover uma função confirmada, verifique também se ela deixa algo órfão por tabela (uma
constante ou import que só ela usava), mas não expanda isso para uma segunda auditoria de módulo
inteiro sem avisar — isso é fora do escopo desta skill.

## Não fazer

- Não rode no repositório inteiro sem o usuário ter indicado um escopo (arquivo ou pasta).
- Não trate ausência de chamada direta como prova suficiente sem checar despacho por dicionário
  primeiro — é a fonte de falso positivo mais comum.
- Não aceite um match cross-file de método sem confirmar que pertence à mesma classe.
- Não remova nada sem confirmação explícita do usuário função por função, mesmo com alta
  confiança de que é órfã.

## Checklist final de aceitação

- O escopo foi indicado pelo usuário, não assumido como o repositório inteiro.
- Cada candidato passou pelas 4 camadas de busca do Passo 2 antes de ser reportado como órfão.
- As exceções do Passo 3 (bloco `__main__`, decorators de framework, API pública) foram checadas.
- Nenhuma remoção ocorreu sem confirmação explícita, função por função.
