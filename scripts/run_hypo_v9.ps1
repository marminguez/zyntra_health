param(
    [Parameter(Mandatory=$true)][string]$DataDir,
    [string]$OutputDir = "ml/models/v9_alert_intelligence"
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$python = Join-Path $repo ".venv-hypo-v6\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Python environment not found: $python" }
& $python "ml\lopo_hypo_v9.py" --data-dir $DataDir --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code ${LASTEXITCODE}: Hypo V9" }
Write-Host "Hypo V9 completed successfully." -ForegroundColor Green
Write-Host "Results: $OutputDir"
Write-Host " - v9_per_patient.csv"
Write-Host " - v9_report.json"
