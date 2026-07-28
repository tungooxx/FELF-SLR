$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

python tools\compute_paper_logit_metrics.py
if ($LASTEXITCODE -ne 0) {
    throw "Canonical paper metric generation failed with exit code $LASTEXITCODE"
}

python tools\compute_canonical_fusion_sensitivity.py
if ($LASTEXITCODE -ne 0) {
    throw "Fusion sensitivity generation failed with exit code $LASTEXITCODE"
}

