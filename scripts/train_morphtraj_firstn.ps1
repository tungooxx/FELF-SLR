param(
    [ValidateSet(100, 300)]
    [int]$NumGlosses = 100,
    [Parameter(Mandatory = $true)]
    [string]$CacheDir,
    [int]$Seed = 1
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot
$env:PYTHONPATH = (Join-Path $RepoRoot "code")

python -u code\train_morph_traj_expert.py `
    --num-glosses $NumGlosses `
    --action-source json_first_n `
    --cache-dir $CacheDir `
    --seed $Seed

if ($LASTEXITCODE -ne 0) {
    throw "MorphTraj training failed with exit code $LASTEXITCODE"
}
