-- Trilha 1 -- 3 queries uteis sobre as tabelas carregadas dos 4 pipelines.
-- Rodar com: docker exec -i harbor_postgres psql -U harbor -d harbor_manufatura -f /caminho/queries_uteis.sql
-- ou colar direto num cliente SQL (DBeaver/pgAdmin) conectado em localhost:5432.

-- 1. Tempo total de parada por categoria (StopGroup), do maior para o menor
SELECT
    "StopGroup",
    ROUND(SUM("StopDuration(min)")::numeric, 1) AS minutos_totais,
    COUNT(*) AS ocorrencias,
    ROUND(AVG("StopDuration(min)")::numeric, 2) AS media_por_parada
FROM oee_downtime_raw
GROUP BY "StopGroup"
ORDER BY minutos_totais DESC;

-- 2. Comparativo OEE antes vs depois do Lean Six Sigma (usando a tabela ja pivotada pelo pipeline)
SELECT * FROM oee_comparacao_lss;

-- 3. Taxa de falha real (Target) por maquina, comparada com o que o Isolation Forest detectou
SELECT
    "Machine_ID",
    COUNT(*) AS total_leituras,
    SUM(CASE WHEN "Target" = 'Fault' THEN 1 ELSE 0 END) AS fault_real,
    SUM(CASE WHEN if_anomaly THEN 1 ELSE 0 END) AS anomalias_detectadas,
    ROUND(100.0 * SUM(CASE WHEN "Target" = 'Fault' THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_fault_real
FROM sensor_predicoes
GROUP BY "Machine_ID"
ORDER BY pct_fault_real DESC;
