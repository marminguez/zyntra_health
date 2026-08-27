param(
  [Parameter(Mandatory=$true)][string]$DataDir,
  [string]$OutputDir = "ml/models/v12_1_temporal_safety_gate"
)
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$Python = Join-Path $RepoRoot ".venv-hypo-v6\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Python environment not found: $Python" }
if (-not (Test-Path $DataDir)) { throw "Data directory not found: $DataDir" }
& $Python "ml/lopo_hypo_v12_1.py" --data-dir $DataDir --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) { throw "Hypo V12.1 failed with exit code ${LASTEXITCODE}" }
Write-Host ""
Write-Host "Hypo V12.1 completed successfully." -ForegroundColor Green
Write-Host "Results: $OutputDir"
Write-Host " - v12_1_per_patient.csv"
Write-Host " - v12_1_report.json"
