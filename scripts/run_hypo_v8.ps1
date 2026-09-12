param([Parameter(Mandatory=$true)][string]$DataDir)
$ErrorActionPreference='Stop'
$RepoRoot=Split-Path -Parent $PSScriptRoot
$Python=Join-Path $RepoRoot '.venv-hypo-v6\Scripts\python.exe'
$OutputDir=Join-Path $RepoRoot 'ml\models\v8_contextual_risk'
function Invoke-Checked { param([Parameter(Mandatory=$true)][string]$Exe,[Parameter(ValueFromRemainingArguments=$true)][string[]]$Args) & $Exe @Args; if($LASTEXITCODE -ne 0){throw "Command failed with exit code ${LASTEXITCODE}: $Exe $($Args -join ' ')"} }
if(-not(Test-Path $DataDir)){throw "Data directory not found: $DataDir"}
if(-not(Test-Path $Python)){throw "Expected existing ML environment at $Python"}
Invoke-Checked $Python -c "import pandas,numpy,sklearn,tensorflow; print('ML imports OK')"
Push-Location (Join-Path $RepoRoot 'ml')
try{Invoke-Checked $Python lopo_hypo_v8.py --data-dir $DataDir --output-dir $OutputDir}finally{Pop-Location}
$Expected=@((Join-Path $OutputDir 'dataset_summary.csv'),(Join-Path $OutputDir 'v8_per_patient.csv'),(Join-Path $OutputDir 'v8_report.json'))
foreach($f in $Expected){if(-not(Test-Path $f)){throw "Missing expected output: $f"}}
Write-Host ""; Write-Host "Hypo V8 completed successfully." -ForegroundColor Green; Write-Host "Results: $OutputDir"; Write-Host " - v8_per_patient.csv"; Write-Host " - v8_report.json"
