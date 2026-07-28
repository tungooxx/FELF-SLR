# Quality-Apex and FineShape-v1

## Seed-42 Uniform B6

- Uniform-40 B6: `74.71 / 91.19`
- Uniform B6 + `0.5 * Quality-Apex B6`: `75.10 / 91.00`

Quality-Apex extracts 64 evenly spaced candidates, keeps 20 uniform anchors,
selects 20 additional frames using:

```text
1.5 * hand visibility + 0.75 * normalized hand motion + 0.35 * middle bias
```

MediaPipe configuration:

```text
static_image_mode=false
model_complexity=1
min_detection_confidence=0.3
min_tracking_confidence=0.3
```

The extraction implementation is in `code/build_quality_apex_wlasl_cache.py`.

## FineShape-v1

FineShape-v1 is a conservative pairwise top-5 reranker over:

- PalmNormVec statistics
- finger curl/spread
- fingertip and thumb-index geometry
- Uniform and Quality-Apex frame statistics
- morphology prototype similarities

It activates only for low-margin, morphology-similar top-5 candidates.
Training uses cross-fitted validation predictions; test labels are not used for
selection.

Three-seed result:

```text
Tri + 0.1 Quality B6: 78.29 +/- 0.22 / 92.15 +/- 0.19
FineShape-v1:         78.35 +/- 0.19 / 92.15 +/- 0.19
```

FineShape-v1 preserves retrieval and causes no harmful flips, but does not meet
the required `78.6` mean Top-1 target.
