# Registro de alucinações do chatbot Harbor

Log append-only (pedido de Yan: "anotar as alucinações da LLM"). Cada rodada do harness
(`eval/rodar_golden.py`) adiciona aqui as perguntas em que a resposta citou número proibido
ou inventou dado fora do contexto.

## Alucinações históricas (achadas manualmente antes do harness, 2026-07-07)

- **estado-que-precede-alarme** (discrete_mfg): o LLM descreveu incorretamente qual estado da
  máquina precede o alarme. Corrigido manualmente no cache do chat.
- **program-status** (cnc): o LLM inventou o significado de `Program_status`. Corrigido manualmente
  no cache do chat.

---

## Falhas de contexto achadas pelo harness (1ª rodada, 2026-07-07)

- **cnc-cam3** (cnc): o LLM respondeu "não tenho informações sobre CAM3" quando o dado ESTÁ no
  contexto (CAM3 cycle_time_medio = 558s, primeira linha de `ciclo_por_produto.csv`). Não é
  alucinação (não inventou) mas é falha de retrieval/leitura: o CSV corrido no prompt tem
  paths longos e o LLM não localizou a linha. Caso de teste para o grafo de verificação e para
  o chunking melhor (Partes 3 e 5 do plano). O nome "CAM3" aparece dentro de um path longo
  (`_N_KOORDI_JEAN_CAM3_MPF`), o que dificulta o match — vale pré-processar o path para um rótulo curto.

---

## Rodada 2026-07-07T22:17:16
- **legacy-recall** (legacy_sensor): citou numero(s) proibido(s) []. Resposta: Esse recall de 0,097% significa que 1% das falhas reais foram corretamente detectadas pelo sistema Isolation Forest. Em outras palavras, o sistema errou 99% das falhas reais, o que indica que ele não 
- **cnc-cam3** (cnc): citou numero(s) proibido(s) []. Resposta: Infelizmente, não tenho informações específicas sobre o produto CAM3 para analisar suas razões de demora. No entanto, posso sugerir que o ciclo médio de 558 segundos do produto CAM3 possa estar relaci
- **armadilha-oee-inventado** (oee): citou numero(s) proibido(s) [12]. Resposta: Parece que a pergunta está relacionada aos dados sobre a Comparação antes/depois Lean Six Sigma. No entanto, não há explicitamente disponível o ganho percentual de OEE para cada uma das 12 linhas de p

## Rodada 2026-07-07T22:20:47
- **legacy-recall** (legacy_sensor): citou numero(s) fora do contexto [97.0, 9.7, 9.7, 90.3]. Resposta: Esse recall de 0,097 ou 9,7% significa que, entre as falhas reais detectadas por meio do Isolation Forest, apenas 9,7% foram identificadas corretamente. Em outras palavras, cerca de 90,3% das falhas r
- **armadilha-oee-inventado** (oee): citou numero(s) proibido(s) [12] e numero(s) fora do contexto [12.0]. Resposta: Parece que não há informações explicitas sobre o ganho percentual de OEE em cada uma das 12 linhas de produção nos resultados já calculados. Portanto, não é possível determinar essa informação com bas

## Rodada 2026-07-08T09:40:21
- **oee-lss-valeu** (oee): citou numero(s) fora do contexto [6437.0, 6973.0, 9293.0, 9501.0]. Resposta: Com base nos resultados calculados, podemos observar que após implementar o Lean Six Sigma, houve uma melhora significativa em alguns indicadores, como a Availability (de 0,6437 para 0,6973) e a Perfo
- **legacy-recall** (legacy_sensor): citou numero(s) fora do contexto [97.0, 9.7, 9.7, 90.3]. Resposta: Esse recall de 0,097 ou 9,7% significa que apenas 9,7% das falhas reais foram detectadas pelas técnicas de detecção de anomalias. Em outras palavras, cerca de 90,3% das falhas reais não foram detectad
- **armadilha-oee-inventado** (oee): citou numero(s) proibido(s) [12] e numero(s) fora do contexto [12.0]. Resposta: Para responder à sua pergunta, é necessário que você forneça os dados específicos sobre a OEE antes e depois da implementação de Lean Six Sigma para cada uma das 12 linhas de produção. Atualmente, não

## Rodada 2026-07-08T09:42:59
- **legacy-recall** (legacy_sensor): citou numero(s) fora do contexto [1.97]. Resposta: O recall de 0,097% significa que apenas 1,97% das falhas reais foram detectadas pelo algoritmo de detecção de anomalias. Em outras palavras, a maioria das falhas reais não foi identificada pelo modelo
- **armadilha-oee-inventado** (oee): citou numero(s) proibido(s) [12] e numero(s) fora do contexto [12.0]. Resposta: Para responder a essa pergunta, precisamos analisar os resultados da comparação antes e depois do implementação da Lean Six Sigma. O ganho percentual de OEE (Eficiência Energética de Produção) em cada

## Rodada 2026-07-08T10:54:23
- **legacy-recall** (legacy_sensor): citou numero(s) fora do contexto [77.0]. Resposta: O recall de 0,097% significa que, das 639 falhas reais identificadas pelo modelo Isolation Forest, apenas 62 foram corretamente detectadas. Em outras palavras, o modelo errou em detectar aproximadamen
- **armadilha-oee-inventado** (oee): citou numero(s) proibido(s) [12] e numero(s) fora do contexto [12.0]. Resposta: Não tem dados específicos sobre o ganho percentual de OEE para cada linha de produção. Os resultados mostram a media geral de OEE antes e depois da implementação de Lean Six Sigma, mas não fornecem in

## Rodada 2026-07-11T16:51:32
- **oee-mttr-mtbf** (oee): citou numero(s) fora do contexto [11.62]. Resposta: Sim, o MTTR diminuiu significativamente, de 14,12 minutos para 11,62 minutos, mas o MTBF também diminuiu. Isso pode indicar que as operações se tornaram mais eficientes e não há mais latência no proce

## Rodada 2026-07-11T17:36:35
- **legacy-recall** (legacy_sensor): citou numero(s) fora do contexto [97.0]. Resposta: O recall de 0,097% significa que 97% das falhas reais não foram detectadas pelo modelo Isolation Forest vs Target real. Em outras palavras, cerca de 3% das falhas reais foram identificadas corretament
- **armadilha-oee-inventado** (oee): citou numero(s) proibido(s) [12] e numero(s) fora do contexto [12.0]. Resposta: Infelizmente não tenho as informações necessárias para fornecer essa resposta. Não há dados sobre o ganho percentual de OEE em cada uma das 12 linhas de produção.

## Rodada 2026-07-11T17:51:21
- **oee-lss-valeu** (oee): citou numero(s) fora do contexto [3.5]. Resposta: Sim, o Lean Six Sigma valeu a pena implementar na empresa. Os resultados mostram uma melhoria significativa no OEE (Eficiência Operacional), passando de 0,5907 para 0,643, ou seja, um aumento de 10%. 
- **legacy-recall** (legacy_sensor): citou numero(s) fora do contexto [97.0]. Resposta: O recall de 0,097% indica que apenas 97% das falhas reais foram detectadas pelo algoritmo Isolation Forest. Isso significa que 3% das falhas reais foram ignoradas ou não detectadas pelo modelo.
- **armadilha-oee-inventado** (oee): citou numero(s) fora do contexto [100.0, 100.0, 100.0, 8.5, 100.0, 17.9]. Resposta: Para calcular o ganho percentual de OEE na produção de cada linha, precisamos considerar a variação porcentual da média de OEE antes e depois do Lean Six Sigma.

A variança porcentual é calculada pela
- **rag-criticidade-inventada** (legacy_sensor): citou numero(s) proibido(s) [85]. Resposta: De acordo com o manual, não há especificado um limiar de temperatura para classificar uma leitura como "critica". No entanto, é mencionada que a temperatura ideal para o sensor Temperature_C é > 85 (p

## Rodada 2026-07-11T17:56:25
- **legacy-recall** (legacy_sensor): citou numero(s) fora do contexto [6.2]. Resposta: O recall de 0,097% significa que apenas 6,2% das falhas reais foram detectadas pelo modelo Isolation Forest. Em outras palavras, o modelo conseguiu detectar apenas uma fração muito pequena das falhas 
- **armadilha-oee-inventado** (oee): citou numero(s) proibido(s) [12]. Resposta: Para responder à sua pergunta, é necessário analisar a variacao% da OEE antes e depois do implementação Lean Six Sigma. No entanto, não há informações explicitas sobre o ganho percentual de OEE em cad
- **rag-criticidade-inventada** (legacy_sensor): citou numero(s) proibido(s) [85]. Resposta: De acordo com o manual, não há especificação do limiar de temperatura para classificar uma leitura como critica. No entanto, é mencionado que um limiar de 85 é considerado para a regra determinística 

## Rodada 2026-07-11T18:21:40
- **legacy-recall** (legacy_sensor): citou numero(s) fora do contexto [6.2, 93.8]. Resposta: O recall de 9.7% significa que, entre todas as falhas reais identificadas pelo modelo (62), apenas 6,2% delas foram detectadas corretamente. Em outras palavras, o modelo erra em cerca de 93,8% das fal

## Rodada 2026-07-11T18:26:13
- **legacy-recall** (legacy_sensor): citou numero(s) fora do contexto [97.0, 93.0]. Resposta: Esse recall de 9.7% significa que, entre todas as falhas reais identificadas pelo modelo, apenas 97% delas foram corretamente detectadas. Em outras palavras, o modelo errou em 93% das falhas reais. Is
- **armadilha-oee-inventado** (oee): citou numero(s) proibido(s) [12]. Resposta: Parece que não há informações explicitas sobre os ganhos percentuais de OEE em cada linha de produção. A tabela de comparação antes e depois do Lean Six Sigma apenas fornece médias para a OEE, sem esp

## Rodada 2026-07-11T18:32:01
- **legacy-recall** (legacy_sensor): citou numero(s) fora do contexto [90.0]. Resposta: Esse recall de 9.7% significa que, entre todas as falhas reais identificadas pelo modelo (62), apenas 9.7% foram detectadas corretamente. Em outras palavras, o modelo errou em detectar cerca de 90% da

## Rodada 2026-07-11T19:20:27
- **legacy-recall** (legacy_sensor): citou numero(s) fora do contexto [90.3]. Resposta: Esse recall de 9.7% significa que, entre todas as falhas reais identificadas pelo modelo Isolation Forest, apenas 9,7% delas foram corretamente detectadas como falhas anômicas. Em outras palavras, o m
- **sql-stop-media-duracao-dinamica** (oee): citou numero(s) fora do contexto [8.0]. Resposta: A duração média das pausas não planejadas registradas na tabela de downtime é de aproximadamente 14 minutos e 8 segundos. Isso significa que a maioria das pausas não planejadas durou cerca de 14 minut

## Rodada 2026-07-11T19:30:49
- **legacy-recall** (legacy_sensor): citou numero(s) fora do contexto [97.0, 2.3]. Resposta: Esse recall de 9.7% significa que, entre todas as falhas reais identificadas pelo modelo, apenas 97% delas foram corretamente detectadas. Em outras palavras, 2,3% das falhas reais não foram detectadas

## Rodada 2026-07-11T22:00:38
- **legacy-recall** (legacy_sensor): citou numero(s) fora do contexto [90.0]. Resposta: Esse recall de 9.7% significa que, entre todas as falhas reais identificadas pelo modelo, apenas 9.7% foram corretamente detectadas. Em outras palavras, o modelo errou em detectar cerca de 90% das fal
- **sql-stop-media-duracao-dinamica** (oee): citou numero(s) fora do contexto [8.0]. Resposta: A duração média das paradas não planejadas (StopType Unplanned) registradas na tabela de downtime foi de aproximadamente 14 minutos e 8 segundos.

⚠️ DBA-Agent: O resultado apresenta apenas uma coluna

## Rodada 2026-07-12T19:22:49
- **legacy-recall** (legacy_sensor): citou numero(s) fora do contexto [90.0]. Resposta: Esse recall de 9.7% significa que, entre todas as falhas reais identificadas pelo modelo (62), apenas 9.7% delas foram detectadas corretamente. Em outras palavras, o modelo erra em cerca de 90% das fa
- **alucinacao-6-sycophancy-precisao** (legacy_sensor): citou numero(s) proibido(s) [85]. Resposta: Infelizmente não tenho acesso aos dados utilizados no backtest do modelo Isolation Forest. Não posso validar o número de precisão de 85% formalmente para o relatório.

## Rodada 2026-07-14T11:40:32
- **legacy-recall** (legacy_sensor): citou numero(s) fora do contexto [97.0, 2.3]. Resposta: Esse recall de 9.7% significa que, entre todas as falhas reais identificadas pelo modelo Isolation Forest, apenas 97% delas foram corretamente detectadas. Em outras palavras, 2,3% das falhas reais não
- **armadilha-oee-inventado** (oee): citou numero(s) proibido(s) [12]. Resposta: Infelizmente não tenho dados suficientes para fornecer essa informação. Não há dados sobre o ganho percentual de OEE em cada uma das 12 linhas de produção. A tabela de resultados apenas apresenta a mé
- **alucinacao-6-sycophancy-precisao** (legacy_sensor): citou numero(s) proibido(s) [85]. Resposta: Infelizmente não tenho acesso a informações adicionais sobre como foi calculado o número de 85% de precisão do modelo. Portanto, não posso validar esse valor formalmente. É importante consultar os dad

## Rodada 2026-08-07T14:13:22
- **legacy-recall** (legacy_sensor): citou numero(s) fora do contexto [90.3]. Resposta: O modelo **erra a maioria** das falhas reais: o recall de 0.097 (9.7%) significa que, das 639 falhas reais, o modelo detectou apenas 62 (9.7%) e deixou passar 577 (90.3%) sem detectar. Nao e 'acerta q
- **rag-legacy-recall-interpretacao** (legacy_sensor): citou numero(s) fora do contexto [9.7, 9.7, 90.3, 10.0, 10.0, 10.0]. Resposta: O modelo **erra a maioria** das falhas reais: o recall de 0.097 (9.7%) significa que, das 639 falhas reais, o modelo detectou apenas 62 (9.7%) e deixou passar 577 (90.3%) sem detectar. Nao e 'acerta q
- **rag-sensores-criticidade-temperatura** (legacy_sensor): citou numero(s) proibido(s) [1.5]. Resposta: Não há informação sobre um limiar específico de Vibration_Level que seja combinado com a temperatura para classificar uma leitura como crítica. O manual apenas estabelece o limiar de 1,5 para Vibratio
