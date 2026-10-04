# ENG-TCN-R1R2 on Colab

This path reproduces the **final custody-closed** ENG-TCN-R1R2 code/data bytes without distributing or touching the WLASL-100 test258 split or any WLASL-300 outcomes.

## 1. Clone at the expected absolute path

```bash
mkdir -p /workspace/local-vlm/SLR
cd /workspace/local-vlm/SLR
git clone https://github.com/tungooxx/FELF-SLR.git
cd FELF-SLR
```

## 2. Install runtime + Kaggle CLI

Use a CUDA PyTorch environment compatible with the project. Then:

```bash
python -m pip install kaggle numpy
```

Configure your Kaggle API token in Colab **outside the repository**. Never commit credentials.

## 3. Download the private cache and verify exact bytes

```bash
bash scripts/download_eng_tcn_kaggle_data.sh /content/eng_tcn_r1r2_data /workspace/local-vlm/SLR/FELF-SLR
```

Default dataset slug:

`chuckies/felf-slr-eng-tcn-r1r2-wlasl100-dev-cache`

The preparation script verifies the archive manifest and source dependency SHA256 values, installs the exact vendored `factorial_dev.py` at the path frozen by the runner, and copies only the 14 development-cache files (train, validation, frozen train augmentation). **No `test_*` or WLASL-300 file is included.**

## 4. Safe pre-science checks

```bash
python diagnostic/eng_tcn_r1r2_runner.py --mode audit
python diagnostic/eng_tcn_r1r2_runner.py --mode smoke --arm ENGINEERED_REPARCH_CONTROL --seed 20261031
python diagnostic/eng_tcn_r1r2_runner.py --mode smoke --arm ENGINEERED_MULTISCALE_TCN --seed 20261031
```

These are audit/train-only smoke checks. They are not the scientific validation tournament.

## 5. Scientific execution

Do **not** run `--mode execute` until the Research OS independent implementation-fidelity review and final pre-science gate for Experiment `27e54160-30b8-440b-b517-2bffbeae897c` are PASS. Once released, run the six frozen cells in the preregistered order without partial-outcome adaptation.

Frozen seeds: `20261031`, `20261103`, `20261107`.
