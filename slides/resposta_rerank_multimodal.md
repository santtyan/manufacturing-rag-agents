# Resposta preparada — "O rerank multimodal (ColModernVBERT) vale a pena no RAG de imagens?"

**Resposta curta (30s):**
"Calibramos o limiar de corte do rerank (top-p) testando 0,7, 0,85 e 0,99 no corpus de 26
imagens do Harbor com score real de retrieval (não o score fixo de antes, que era um bug). O
limiar 0,7 venceu em todas as métricas — nDCG@5 de 0,889 e Recall@5 de 97%, contra 0,864/93% sem
rerank — e ainda foi mais barato que os limiares mais permissivos. Mas o achado mais importante
veio do experimento seguinte: comparamos rerank sobre um VLM de captioning ruim contra nenhum
rerank sobre um VLM bom, e o VLM bom sozinho ganhou disparado — 0,864 de nDCG e 93% de recall em
12 segundos, contra 0,436 e 48% em quase 80 minutos com rerank sobre o VLM ruim. Conclusão: o
rerank ajuda quando o captioning já é bom, mas não conserta um captioning ruim — a prioridade
real agora é achar um VLM de qualidade suficiente, não otimizar o rerank."

**Por que isso é bom, não ruim, de mostrar:**
- Testamos a hipótese mais otimista da literatura (rerank caro compensa retrieval barato/fraco,
  padrão HEAVEN) e ela não se sustentou aqui — reportar isso evita investir esforço futuro no
  lugar errado da arquitetura
- O resultado é desproporcional o bastante (quase 400x de diferença de custo, o dobro de
  qualidade) para ser conclusão forte, não ruído estatístico
- Veio de um experimento desenhado especificamente para testar a hipótese antes de decidir
  arquitetura, não de uma decisão tomada e depois racionalizada

**Se perguntarem "então o rerank foi desperdício de trabalho?":**
"Não — ele ainda é o melhor resultado quando o VLM já é o mesmo dos dois lados: no item anterior
(calibração de top-p), rerank com top-p=0,7 bateu 'sem rerank' em todas as métricas usando o
mesmo VLM (qwen3-vl) nos dois casos. O que ele não faz é compensar a escolha de um VLM ruim — são
dois problemas diferentes, e o rerank resolve só um deles."

**Números para ter na ponta da língua:**
- Recalibração de top-p (mesmo VLM nos 2 lados): top-p=0,7 → nDCG@5=0,889/Recall@5=97%/MRR=0,702
  (vs. 0,864/93%/0,679 sem rerank); 0,85 e 0,99 pioraram (nDCG 0,802 e 0,780) e custaram mais caro
- Rerank vs. captioning (VLM diferente nos 2 lados): qwen3-vl sem rerank
  nDCG@5=0,864/Recall@5=93%/custo=11,8s vs. moondream+rerank nDCG@5=0,436/Recall@5=48%/
  custo=4779,4s
- VLM aprovado ainda não existe: qwen3-vl chegou a 77,5% de fidelidade nos checks
  determinísticos, abaixo do critério de promoção de ≥90%
