# Sobe toda a infraestrutura da entrega Harbor e confere a saude de cada servico.
# Uso: powershell -ExecutionPolicy Bypass -File start_all.ps1

$ErrorActionPreference = "Continue"
$ollamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"

function Test-Endpoint($nome, $url, $timeoutSec = 5) {
    try {
        $resp = Invoke-WebRequest -Uri $url -TimeoutSec $timeoutSec -UseBasicParsing -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            Write-Host "[OK]   $nome ($url)" -ForegroundColor Green
            return $true
        }
        Write-Host "[FALHA] $nome respondeu $($resp.StatusCode) ($url)" -ForegroundColor Yellow
        return $false
    } catch {
        Write-Host "[FALHA] $nome nao respondeu ($url)" -ForegroundColor Red
        return $false
    }
}

Write-Host "`n=== 1. Docker Desktop ===" -ForegroundColor Cyan
$dockerRunning = docker ps 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker Desktop nao esta rodando -- iniciando..." -ForegroundColor Yellow
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    Write-Host "Aguardando Docker Desktop inicializar (ate 60s)..."
    $tentativas = 0
    while ($tentativas -lt 20) {
        Start-Sleep -Seconds 3
        docker ps > $null 2>&1
        if ($LASTEXITCODE -eq 0) { break }
        $tentativas++
    }
}
docker ps --format "{{.Names}}: {{.Status}}" 2>&1

Write-Host "`n=== 2. Containers (Postgres + N8N) ===" -ForegroundColor Cyan
Push-Location "$PSScriptRoot\infra"
docker compose up -d
Pop-Location

Write-Host "`n=== 3. Ollama ===" -ForegroundColor Cyan
$ollamaProc = Get-Process ollama -ErrorAction SilentlyContinue
if (-not $ollamaProc) {
    Write-Host "Ollama nao esta rodando -- iniciando..." -ForegroundColor Yellow
    Start-Process $ollamaExe -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

Write-Host "`n=== 4. FastAPI (porta 8000) ===" -ForegroundColor Cyan
$apiRodando = Test-Endpoint "FastAPI" "http://localhost:8000/health" 3
if (-not $apiRodando) {
    Write-Host "Iniciando FastAPI..." -ForegroundColor Yellow
    Push-Location "$PSScriptRoot\api"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m uvicorn main:app --port 8000" -WindowStyle Minimized
    Pop-Location
    Start-Sleep -Seconds 6
}

Write-Host "`n=== 5. Streamlit (porta 8501) ===" -ForegroundColor Cyan
$dashRodando = Test-Endpoint "Streamlit" "http://localhost:8501" 3
if (-not $dashRodando) {
    Write-Host "Iniciando Streamlit..." -ForegroundColor Yellow
    Push-Location "$PSScriptRoot\dashboard"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m streamlit run app.py --server.port 8501" -WindowStyle Minimized
    Pop-Location
    Start-Sleep -Seconds 6
}

Write-Host "`n=== Checagem final de saude ===" -ForegroundColor Cyan
Start-Sleep -Seconds 2
$resultados = @{
    "Ollama"    = Test-Endpoint "Ollama"    "http://localhost:11434/api/tags"
    "FastAPI"   = Test-Endpoint "FastAPI"   "http://localhost:8000/health"
    "Streamlit" = Test-Endpoint "Streamlit" "http://localhost:8501"
    "N8N"       = Test-Endpoint "N8N"       "http://localhost:5678"
}

Write-Host "`n=== Resumo ===" -ForegroundColor Cyan
$falhas = 0
foreach ($k in $resultados.Keys) {
    if (-not $resultados[$k]) { $falhas++ }
}
if ($falhas -eq 0) {
    Write-Host "Tudo pronto para a demo." -ForegroundColor Green
} else {
    Write-Host "$falhas servico(s) com problema -- verifique os logs acima antes da reuniao." -ForegroundColor Red
}

Write-Host "`nEnderecos:"
Write-Host "  Dashboard : http://localhost:8501"
Write-Host "  API Docs  : http://localhost:8000/docs"
Write-Host "  N8N       : http://localhost:5678"
Write-Host "  Webhook   : http://localhost:5678/webhook/diagnostico-automatico"
