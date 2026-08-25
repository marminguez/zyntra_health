param(
  [Parameter(Mandatory=$true)]
  [string]$DataDir
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $RepoRoot '.venv-hypo-v6'
$Python = Join-Path $Venv 'Scripts\python.exe'
$OutputDir = Join-Path $RepoRoot 'ml\models\v6_lopo'

if (-not (Test-Path $DataDir)) {
  throw "Data directory not found: $DataDir"
}

$xml = Get-ChildItem -Path $DataDir -Filter '*-ws-*.xml' -File
if ($xml.Count -lt 12) {
  throw "Expected at least 12 OhioT1DM XML files in $DataDir, found $($xml.Count)."
}

if (-not (Test-Path $Venv)) {
  py -3.11 -m venv $Venv
}

& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $RepoRoot 'ml\requirements.txt')

Push-Location (Join-Path $RepoRoot 'ml')
try {
  & $Python lopo_hypo_v6.py --data-dir $DataDir --output-dir $OutputDir
}
finally {
  Pop-Location
}

Write-Host ""
Write-Host "Hypo V6 LOPO completed." -ForegroundColor Green
Write-Host "Results: $OutputDir"
Write-Host " - dataset_summary.csv"
Write-Host " - lopo_per_patient.csv"
Write-Host " - lopo_report.json"
