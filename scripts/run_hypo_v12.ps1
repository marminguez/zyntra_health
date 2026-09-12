param(
  [Parameter(Mandatory=$true)][string]$DataDir,
  [string]$OutputDir = "ml/models/v12_personalized_finetuning"
)
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$Python = Join-Path $RepoRoot ".venv-hypo-v6\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Python environment not found: $Python" }
if (-not (Test-Path $DataDir)) { throw "Data directory not found: $DataDir" }
& $Python "ml/lopo_hypo_v12.py" --data-dir $DataDir --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) { throw "Hypo V12 failed with exit code ${LASTEXITCODE}" }
Write-Host ""
Write-Host "Hypo V12 completed successfully." -ForegroundColor Green
Write-Host "Results: $OutputDir"
Write-Host " - v12_report.json"
Write-Host " - v12_window_summary.csv"
Write-Host " - v12_per_patient_window.csv"
