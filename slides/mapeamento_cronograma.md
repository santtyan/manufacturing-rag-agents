# Mapeamento da entrega de 2026-07-07 ao cronograma FUNAPE (A1-J9)

Referência para justificar a entrega como progresso real do plano de trabalho, não um exercício avulso.

## Eixo 1 — Fundamentação e Capacitação Técnica
- **A2** (LLMs, NLP na manufatura) — os 4 pipelines usam LLM (Ollama) para resumo/diagnóstico
- **A7** (modelos locais: LLaMA, Mistral, Phi-3 via Ollama) — `llama3.2` rodando localmente em todos os pipelines
- **A15** (Docker: Dockerfile, docker-compose) — `docker-compose.yml` com Postgres + N8N
- **A16** (APIs RESTful) — API FastAPI (`/diagnostico`, `/amostra`, `/health`)
- **A18** (ambientação com dados Tipo A/B/C) — os 4 datasets cobrem paradas/histórico (Tipo B) e telemetria (Tipo C)
- **A19** (capacitação em Python, LangChain, Ollama, Git) — todos os pipelines em Python + Ollama
- **A20** (capacitação em APIs: DeepSeek, Groq, Together AI) — testado Groq no repositório pessoal; entrega final usa Ollama

## Eixo 2 — Engenharia de Dados e Análise Exploratória
- **B1** (coleta, limpeza, transformação de dados industriais) — todos os 4 pipelines
- **B3** (organização de CSVs: sensores, logs, OEE, paradas) — os 4 datasets são exatamente isso
- **B4-B10** (EDA, séries temporais, outliers Z-score/Isolation Forest, correlação) — pipelines 1, 2 e 4
- **B11** (banco SQL para dados industriais) — Postgres com 8 tabelas carregadas
- **B16** (matriz de correlação falhas-sensores) — análise de separabilidade do pipeline 2

## Eixo 3 — Docker, APIs, MCP e Modelos Locais
- **C1-C3** (Dockerfiles, docker-compose, containers para Ollama) — infraestrutura Docker completa
- **C4-C5** (API de chatbot, seleção de LLM) — endpoint `/diagnostico`
- **C8** (API de diagnóstico de falhas) — literalmente o endpoint principal da entrega
- **C13** (documentação Swagger/OpenAPI) — Swagger automático do FastAPI em `/docs`
- **C17-C21** (execução local via Ollama, avaliação latência/precisão) — todos os pipelines usam Ollama local

## Eixo 4 — Automação com N8N
- **D1-D2** (deploy N8N em Docker, criação de workflows) — container N8N + workflow ativo
- **D3-D9** (processamento CSV periódico, alertas automáticos) — workflow webhook → diagnóstico → resultado
- **D10-D13** (integração N8N com APIs via webhook/HTTP) — exatamente o que foi construído

## Eixo 5 — NL to SQL, PandasQuery e Agentes de Dados
- **E17-E20** (SymPy, fórmulas de manufatura como OEE/MTBF/MTTR) — MTTR/MTBF calculados no pipeline 1

## Eixo 6 — Agentes PDF, CSV e Multiagentes
- **F1-F7** (agente PDF: RAG com citação de fonte) — pipeline RAG sobre manual técnico, citando fonte
- **F8-F14** (agente CSV: EDA automatizada, detecção de anomalias) — os 4 pipelines de dados

## Eixo 7 — Diagnóstico de Falhas
- **G1-G4** (classificação de falhas, detecção de anomalias em séries temporais, RCA assistida por LLM) — arquitetura híbrida de 3 camadas (pipeline 2 + API)
- **G9** (chatbot com diagnóstico) — chatbots especializados do dashboard

## Não coberto ainda (fica para as próximas trilhas)
- **A17, C14-C16** (MCP) — mencionado como próximo passo, não implementado
- **A11-A14, F15-F23** (Multiagentes, Continual Learning, LoRA/JumpLoRA) — fora do escopo desta entrega
- **E1-E6** (NL-to-SQL) — Postgres está pronto como base, mas o agente de tradução NL→SQL ainda não foi construído

**Conclusão prática:** a entrega de 2026-07-07 toca efetivamente itens de 5 dos 10 eixos do cronograma (1, 2, 3, 4 parcialmente, 6, 7), o que é uma cobertura de escopo real do plano de 30 meses, não um exercício isolado.
