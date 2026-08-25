param(
  [Parameter(Mandatory=$true)]
  [string]$DataDir
)

$ErrorActionPreference='Stop'
$RepoRoot=Split-Path -Parent $PSScriptRoot
$Venv=Join-Path $RepoRoot '.venv-hypo-v6'
$Python=Join-Path $Venv 'Scripts\python.exe'
$OutputDir=Join-Path $RepoRoot 'ml\models\v7_alert_intelligence'

function Invoke-Checked {
  param([Parameter(Mandatory=$true)][string]$Exe,[Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
  & $Exe @Args
  if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code ${LASTEXITCODE}: $Exe $($Args -join ' ')" }
}

if (-not (Test-Path $DataDir)) { throw "Data directory not found: $DataDir" }
if (-not (Test-Path $Python)) { throw "Expected existing V6 Python environment at $Python" }

Invoke-Checked $Python -c "import pandas, numpy, sklearn, tensorflow; print('ML imports OK')"
Push-Location (Join-Path $RepoRoot 'ml')
try {
  Invoke-Checked $Python lopo_hypo_v7.py --data-dir $DataDir --output-dir $OutputDir
}
finally { Pop-Location }

$Expected=@((Join-Path $OutputDir 'dataset_summary.csv'),(Join-Path $OutputDir 'v7_per_patient.csv'),(Join-Path $OutputDir 'v7_report.json'))
foreach($file in $Expected){ if(-not(Test-Path $file)){ throw "Missing expected output: $file" } }
Write-Host ""
Write-Host "Hypo V7 completed successfully." -ForegroundColor Green
Write-Host "Results: $OutputDir"
Write-Host " - dataset_summary.csv"
Write-Host " - v7_per_patient.csv"
Write-Host " - v7_report.json"
