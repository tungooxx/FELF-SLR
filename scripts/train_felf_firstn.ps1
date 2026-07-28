param(
    [ValidateSet(100, 300)]
    [int]$NumGlosses = 100,
    [Parameter(Mandatory = $true)]
    [string]$CacheDir,
    [Parameter(Mandatory = $true)]
    [string]$BaselineCheckpoint,
    [Parameter(Mandatory = $true)]
    [string]$LrgCheckpoint,
    [Parameter(Mandatory = $true)]
    [string]$RfCheckpoint,
    [int]$Seed = 1
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot
$env:PYTHONPATH = (Join-Path $RepoRoot "code")

python -u code\wlasl_train_felf_slr_subset.py `
    --num-glosses $NumGlosses `
    --action-source json_first_n `
    --feature-kind old `
    --cache-dir $CacheDir `
    --architecture-name FELF `
    --left-branch-dim 96 `
    --right-branch-dim 96 `
    --global-branch-dim 96 `
    --dropout 0.2 `
    --use-lrg-head `
    --use-rf-head `
    --lrg-logit-weight 1.0 `
    --rf-logit-weight 0.75 `
    --normalize-fused-logits `
    --preload-main-checkpoint $BaselineCheckpoint `
    --preload-lrg-checkpoint $LrgCheckpoint `
    --preload-rf-checkpoint $RfCheckpoint `
    --freeze-main `
    --output-prefix "wlasl${NumGlosses}_FELF_seed${Seed}" `
    --epochs 50 `
    --swa-epochs 20 `
    --batch-size 32 `
    --seed $Seed

if ($LASTEXITCODE -ne 0) {
    throw "FELF training failed with exit code $LASTEXITCODE"
}

