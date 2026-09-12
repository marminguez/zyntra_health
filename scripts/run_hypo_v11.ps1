param(
  [Parameter(Mandatory=$true)][string]$DataDir,
  [string]$OutputDir = "ml/models/v11_personalized_calibration"
)
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$Python = Join-Path $RepoRoot ".venv-hypo-v6\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Python environment not found: $Python" }
if (-not (Test-Path $DataDir)) { throw "Data directory not found: $DataDir" }
& $Python "ml/lopo_hypo_v11.py" --data-dir $DataDir --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) { throw "Hypo V11 failed with exit code ${LASTEXITCODE}" }
Write-Host ""
Write-Host "Hypo V11 completed successfully." -ForegroundColor Green
Write-Host "Results: $OutputDir"
Write-Host " - v11_per_patient.csv"
Write-Host " - v11_report.json"
