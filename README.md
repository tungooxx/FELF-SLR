# FELF-SLR

Official research code and canonical evaluation artifacts for **Factor-Expert
Logit Fusion for Class-Crowded Isolated Sign Recognition**.

## Canonical Results

The paper uses one frozen seed-1 protocol:

| Dataset | System | Top-1 | Top-5 | Per-class Top-1 | Per-class Top-5 |
|---|---:|---:|---:|---:|---:|
| WLASL-100 | Baseline | 86.39 | 94.24 | 86.37 | 94.27 |
| WLASL-100 | Stage 1 FELF | 86.39 | 94.24 | 87.85 | 94.36 |
| WLASL-100 | FELF-SLR | **88.48** | **95.29** | **88.63** | **95.83** |
| WLASL-300 | Baseline | 72.41 | 89.85 | 74.54 | 90.21 |
| WLASL-300 | Stage 1 FELF | 77.20 | 90.80 | 78.63 | 91.61 |
| WLASL-300 | FELF-SLR | **77.97** | **92.34** | **79.44** | **92.83** |

Historical runs using different caches, checkpoint families, or direct fusion
rules are not canonical paper results.

## Fusion Definition

Stage 1 includes the Baseline:

```text
z_FELF = (z_Baseline + w_LRG*z_LRG + w_RF*z_RF)
         / (1 + abs(w_LRG) + abs(w_RF))
```

The canonical Stage-1 coefficients are `w_LRG=1.0` and `w_RF=0.75`.
Stage 2 is:

```text
z_final = 0.5*z_Baseline + 1.0*z_FELF + 0.5*z_MT
```

The implementation does **not** apply sample-wise z-score normalization.
Fusion weights are selected from validation logits and frozen before test
evaluation.

## Repository Layout

- `code/`: training, feature extraction, expert, and latency implementations.
- `scripts/`: PowerShell training/evaluation wrappers.
- `tools/`: canonical metric, significance, sensitivity, and fusion-ablation tools.
- `diagnostic/paper_metrics/`: paper-ready metrics and protocol manifests.
- `diagnostic/canonical/`: canonical summaries, stress tests, and dropout results.
- `docs/`: additional experimental notes.

## Reproducing Paper Metrics

The metric scripts consume saved logits referenced by
`diagnostic/paper_metrics/canonical_protocol_manifest.json`:

```powershell
python tools\compute_paper_logit_metrics.py
python tools\compute_canonical_fusion_sensitivity.py
python tools\run_fusion_space_ablation.py --datasets wlasl100 wlasl300 --seed 1
```

The manifest records the exact local artifact paths and SHA-256 hashes used to
generate the tables. Raw logits and checkpoints are not committed because of
size; place them at the manifest paths or update the paths without changing
their hashes.

## Training

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

The principal entry points are:

```text
code/wlasl_train_b6_baseline_subset.py
code/wlasl_train_felf_experts_subset.py
code/wlasl_train_felf_slr_subset.py
code/train_morph_traj_expert.py
code/evaluate_tri_felf_mt.py
```

PowerShell wrappers under `scripts/` document the exact arguments used in the
working environment. Set dataset/cache paths explicitly before running them.

## Data

WLASL videos and derived pose caches are not redistributed. Obtain WLASL from
its official source and follow its Computational Use of Data Agreement. This
repository contains only code and derived aggregate diagnostics.

## Reproduction Scope

External comparison rows in the paper are same-split reproductions: public
architectures were adapted at the data-loader boundary and retrained on the
shared retained WLASL samples. They are not copied original-paper results.

