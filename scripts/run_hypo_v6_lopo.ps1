param(
  [Parameter(Mandatory=$true)]
  [string]$DataDir
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $RepoRoot '.venv-hypo-v6'
$Python = Join-Path $Venv 'Scripts\python.exe'
$OutputDir = Join-Path $RepoRoot 'ml\models\v6_lopo'

function Invoke-Checked {
  param(
    [Parameter(Mandatory=$true)][string]$Exe,
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$Args
  )
  & $Exe @Args
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code ${LASTEXITCODE}: $Exe $($Args -join ' ')"
  }
}

if (-not (Test-Path $DataDir)) {
  throw "Data directory not found: $DataDir"
}

$xml = Get-ChildItem -Path $DataDir -Filter '*-ws-*.xml' -File
if ($xml.Count -lt 12) {
  throw "Expected at least 12 OhioT1DM XML files in $DataDir, found $($xml.Count)."
}

if (-not (Test-Path $Venv)) {
  if (Test-Path 'D:\Python313\python.exe') {
    & 'D:\Python313\python.exe' -m venv $Venv
  } else {
    py -3.13 -m venv $Venv
  }
  if ($LASTEXITCODE -ne 0) { throw "Could not create Python 3.13 virtual environment." }
}

Invoke-Checked $Python -m pip install --upgrade pip
Invoke-Checked $Python -m pip install -r (Join-Path $RepoRoot 'ml\requirements.txt')
Invoke-Checked $Python -c "import pandas, numpy, sklearn, tensorflow; print('ML imports OK')"

Push-Location (Join-Path $RepoRoot 'ml')
try {
  Invoke-Checked $Python lopo_hypo_v6.py --data-dir $DataDir --output-dir $OutputDir
}
finally {
  Pop-Location
}

$Expected = @(
  (Join-Path $OutputDir 'dataset_summary.csv'),
  (Join-Path $OutputDir 'lopo_per_patient.csv'),
  (Join-Path $OutputDir 'lopo_report.json')
)
foreach ($file in $Expected) {
  if (-not (Test-Path $file)) { throw "Training finished without expected output: $file" }
}

Write-Host ""
Write-Host "Hypo V6 LOPO completed successfully." -ForegroundColor Green
Write-Host "Results: $OutputDir"
Write-Host " - dataset_summary.csv"
Write-Host " - lopo_per_patient.csv"
Write-Host " - lopo_report.json"
