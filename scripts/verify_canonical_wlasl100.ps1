param(
    [Parameter(Mandatory = $true)]
    [string]$DataDir,

    [Parameter(Mandatory = $true)]
    [string]$JsonPath,

    [Parameter(Mandatory = $true)]
    [string]$FeatureCache,

    [Parameter(Mandatory = $true)]
    [string]$FelfCache,

    [Parameter(Mandatory = $true)]
    [string]$MtCache
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

Push-Location $RepoRoot
try {
    python -u code\verify_canonical_wlasl100.py `
        --data-dir $DataDir `
        --json-path $JsonPath `
        --feature-cache $FeatureCache `
        --felf-cache $FelfCache `
        --mt-cache $MtCache `
        --checkpoint-dir checkpoints\canonical `
        --output-dir diagnostic\canonical_checkpoint_reproduction\wlasl100

    if ($LASTEXITCODE -ne 0) {
        throw "Canonical WLASL-100 verification failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
