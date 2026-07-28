# Final Paper Metrics

Top-1/Top-5 are reported as per-instance accuracy; per-class values are macro averages across glosses.
MRR and mean true rank use the complete test ranking.

## Classification and Ranking

| Dataset | System | Per-instance Top-1 | Per-instance Top-5 | Per-class Top-1 | Per-class Top-5 | MRR | ECE | NLL | AURC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wlasl100 | Baseline | 86.39% | 94.24% | 86.37% | 94.27% | 0.897 | 8.90% | 0.805 | 0.039 |
| wlasl100 | FELF | 86.39% | 94.24% | 87.85% | 94.36% | 0.898 | 6.71% | 0.760 | 0.036 |
| wlasl100 | Tri-FELF-MT | 88.48% | 95.29% | 88.63% | 95.83% | 0.914 | 9.37% | 1.178 | 0.032 |
| wlasl300 | Baseline | 72.41% | 89.85% | 74.54% | 90.21% | 0.803 | 11.73% | 1.403 | 0.127 |
| wlasl300 | FELF | 77.20% | 90.80% | 78.63% | 91.61% | 0.832 | 8.44% | 1.276 | 0.120 |
| wlasl300 | Tri-FELF-MT | 77.97% | 92.34% | 79.44% | 92.83% | 0.843 | 16.63% | 1.850 | 0.103 |

## Neighborhood Preservation

`Overlap@5` is the mean fraction of the reference Top-5 candidates retained in the final Top-5.
`TrueClassDrop@5` counts samples whose true class was in the reference Top-5 but not in the final Top-5.

| Dataset | Reference | Final | Overlap@5 | TrueClassDrop@5 | Drop rate |
|---|---|---|---:|---:|---:|
| wlasl100 | Baseline | FELF | 60.10% | 3 | 1.57% |
| wlasl100 | Baseline | Tri-FELF-MT | 71.94% | 1 | 0.52% |
| wlasl100 | FELF | Tri-FELF-MT | 76.02% | 1 | 0.52% |
| wlasl300 | Baseline | FELF | 61.80% | 16 | 3.07% |
| wlasl300 | Baseline | Tri-FELF-MT | 72.03% | 6 | 1.15% |
| wlasl300 | FELF | Tri-FELF-MT | 79.62% | 2 | 0.38% |

## Paired Significance

Confidence intervals and p-values use 20,000 paired bootstrap resamples of test examples.
McNemar p-values are exact two-sided tests on discordant predictions.

| Dataset | Reference | Top-1 delta [95% CI] | Bootstrap p | McNemar p |
|---|---|---:|---:|---:|
| wlasl100 | Baseline | 2.09 [0.00, 4.71] | 0.1315 | 0.2188 |
| wlasl100 | FELF | 2.09 [0.00, 4.71] | 0.1281 | 0.2188 |
| wlasl300 | Baseline | 5.56 [3.07, 8.05] | 0.0001 | 1.537e-05 |
| wlasl300 | FELF | 0.77 [-0.96, 2.49] | 0.4243 | 0.5034 |

## Seed Stability

| Dataset | Seeds | Top-1 mean +- std | Top-5 mean +- std |
|---|---|---:|---:|
| WLASL-100 | 1,2,3,42 | 87.43 +- 0.96% | 94.63 +- 0.66% |
| WLASL-300 | 1,2,3 | 77.78 +- 0.19% | 92.27 +- 0.29% |
