param(
  [Parameter(Mandatory=$true)][string]$DataDir,
  [string]$OutputDir = "ml/models/v10_alert_state_machine"
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$python = Join-Path $repo ".venv-hypo-v6\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Python environment not found: $python" }
& $python "ml/lopo_hypo_v10.py" --data-dir $DataDir --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) { throw "Hypo V10 failed with exit code ${LASTEXITCODE}" }
Write-Host "Hypo V10 completed successfully." -ForegroundColor Green
Write-Host "Results: $OutputDir"
Write-Host " - v10_per_patient.csv"
Write-Host " - v10_report.json"
