# Paper metrics

`classification_rank_metrics_seed1.csv` reports Top-1, Top-5, MRR, and mean true rank from saved logits.

`rank_movement_metrics_seed1.csv` reports baseline/FELF/Tri rank movement and Top-5 retention diagnostics.

`paired_significance_seed1.csv` reports paired bootstrap confidence intervals and exact McNemar tests.

`canonical_stage1_weight_sensitivity.csv` and
`canonical_stage2_weight_sensitivity.csv` report validation-only one-at-a-time
coefficient sweeps. Any test columns in the Stage 2 file are diagnostic-only and
must not be used for selection.

`canonical_protocol_manifest.json` records every source logit and label path used by
the report, together with source-metadata, retained, and combined-exclusion counts.

The manifest paths are taken from the canonical final-run summaries; historical branch-logit banks are intentionally excluded.
