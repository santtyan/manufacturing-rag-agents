# Dataset OpenPack — licença e atribuição

Registro obrigatório antes de baixar qualquer arquivo do dataset (ver plano de integração,
`pipelines/pipeline8_openpack.py`). Ler antes de adicionar qualquer artefato novo derivado do
OpenPack ao repositório.

## O que é

OpenPack: dataset multimodal de reconhecimento de operações de trabalho em embalagem logística,
53,8 horas, 16 sujeitos, 20.129 anotações de operação (10 classes) + 52.529 anotações de ação
(16 subclasses). 9 modalidades: aceleração, giroscópio, quaternion, BVP, EDA, keypoints, LiDAR,
depth, RGB.

## Citação obrigatória

> Naoya Yoshimura, Jaime Morales, Takuya Maekawa, Takahiro Hara, "OpenPack: A Large-scale
> Dataset for Recognizing Packaging Works in IoT-enabled Logistic Environments". Proceedings of
> the IEEE International Conference on Pervasive Computing and Communications (PerCom 2024).

## Duas licenças distintas — não confundir

- **OpenPack Dataset (sem RGB)** — CC BY-NC-SA 4.0 (Atribuição-NãoComercial-CompartilhaIgual).
  Permite uso e adaptação, exige atribuição e a mesma licença em derivados, **proíbe uso
  comercial**. É a licença que cobre o que este projeto usa: dados IMU pré-processados
  (`imuWithOperationLabel`), rótulos de operação/ação.
- **OpenPack Dataset (+RGB)** — licença própria e mais restritiva: **proíbe redistribuição sob
  qualquer forma**, uso limitado a pesquisa acadêmica, proíbe uso comercial. Cobre qualquer
  keyframe/imagem RGB extraída do dataset (Fase 5 do plano de integração).

## Regras aplicadas neste repositório

1. **Uso é compatível**: o Harbor é projeto de pesquisa acadêmica (bolsa FUNAPE/CERISE), sem fim
   comercial — atende as duas licenças.
2. **Dado bruto nunca entra no git** — fica em `C:\Users\USER\Downloads\OpenPack\` (fora do
   repositório), mesma convenção dos outros pipelines (`DATA_PATH` absoluto).
3. **Nenhuma imagem RGB do OpenPack é commitada ou redistribuída** — `rag/manuais_imagens_openpack/`
   está no `.gitignore`. A licença +RGB proíbe redistribuição mesmo para fins não-comerciais.
4. **Artefatos derivados versionados** (texto agregado, features estatísticas, CSVs de outputs
   de pipeline) são transformações que não reproduzem o dado bruto — compatível com CC
   BY-NC-SA 4.0, mas mantêm a atribuição acima citada em qualquer publicação/slide que os use.
5. Ao publicar qualquer slide/resultado derivado do OpenPack (via `slide-progresso-sessao` ou
   outra skill), incluir a citação do paper.

## Acesso ao RGB completo — pendência de continuidade (2026-09-13)

O sample público (GitHub, sem autorização) só disponibiliza **1 segundo de vídeo** por sessão
(15 frames, U0209/S0500, todos entre 14:25:11.030 e 14:25:11.963) — cobre um instante **antes**
do início da primeira operação anotada (Picking começa às 14:25:22), não uma operação real.
Isso inviabiliza a Fase 5 completa do plano de integração (12 keyframes cobrindo as 10
operações) com o material disponível sem aprovação externa.

**Acesso ao dataset RGB completo exige**:
1. Preencher o "OpenPack Dataset - Access Request Form" (formulário Google, 5 páginas, pede
   nome, email institucional de universidade/empresa, e opcionalmente orientador/gestor e sua
   afiliação): https://docs.google.com/forms/d/e/1FAIpQLScrRWe-qTQV5CKTBxtLQZ7ScgLsHFWxXRmD5he04qXRVBAtqg/viewform
2. Aguardar aprovação manual dos autores do dataset (processo assíncrono, sem prazo definido).
3. Após aprovado, acesso é compartilhado como leitura numa pasta do Google Drive vinculada à
   conta Google informada no formulário.

**Preenchimento requer dados institucionais pessoais do usuário (Yan) — não preenchido por
Claude nesta sessão.** Se aprovado no futuro, retomar a Fase 5 com o dataset completo permite
amostrar keyframes reais de cada uma das 10 operações, em vez do único frame pré-operação hoje
disponível.

**STATUS (2026-09-14): formulário enviado pelo usuário.** Aguardando aprovação manual dos
autores (processo assíncrono, sem prazo definido — ver item 2 acima). Nenhuma ação adicional
possível até a aprovação chegar (acesso via pasta do Google Drive vinculada à conta usada no
formulário). Quando aprovado, retomar a Fase 5 do plano histórico de integração: baixar
keyframes reais das 10 operações (não o único frame pré-operação hoje disponível), reindexar
com `rag/rag_multimodal_langchain.py` usando `PROMPT_CAPTION_CENA` (já validado no smoke test
abaixo), e decidir se a legenda de cena real também deveria ser determinística ou se, nesse
caso, VLM é mesmo necessário (não há dado tabular de origem para gerar template, ao contrário
dos gráficos técnicos — ver skill `rag-multimodal`, seção "Item 6b").

### Smoke test já executado (2026-09-13), válido independente da aprovação

Com o único frame disponível (`U0209_S0500_frame_meio.jpg`, fora do repo,
`C:\Users\USER\Downloads\OpenPack\rgb_keyframes\`), o pipeline de captioning RGB do Harbor foi
testado e **aprovado tecnicamente**:

- **Achado real**: `rag/rag_multimodal_langchain.py::PROMPT_CAPTION` é específico para gráficos
  técnicos sintéticos ("tipo de gráfico, título/eixos") — usá-lo numa foto de cena real produziu
  legenda **vazia** em 794,8s (o modelo aparentemente "trava" buscando algo que não existe na
  imagem; mesmo padrão intermitente já documentado na skill `rag-multimodal`, seção "4 das 26
  imagens deram resposta vazia"). Corrigido com um prompt novo, `PROMPT_CAPTION_CENA`
  (parâmetro opcional `prompt=` adicionado a `gerar_legenda()`), específico para cena de
  embalagem: "o que a pessoa está fazendo com as mãos, quais objetos visíveis, postura".
- Com `PROMPT_CAPTION_CENA`, `qwen3-vl:4b` gerou legenda coerente e factualmente correta em
  698,5s: identificou luvas, camisa, cortinas, caixas de papelão, material plástico, mesa,
  scanner, fita adesiva, etiquetas, relógio digital — todos confirmados na imagem real.
- **Custo real**: 698-795s por imagem, ~35% mais caro que os 412s medidos em gráficos técnicos
  sintéticos — cena real tem mais elementos a descrever que um gráfico matplotlib.
- **Conclusão**: o mecanismo técnico funciona (VLM + prompt certo + pipeline de indexação já
  validado nas Fases 2-4). O que falta é dado real anotado em escala suficiente (múltiplas
  operações, múltiplos sujeitos) para repetir o desenho de golden set das Fases 2-4 — bloqueado
  pela aprovação de acesso acima, não por limitação técnica do pipeline.
