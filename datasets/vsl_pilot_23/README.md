# VSL Pilot 23: Derived Pose Release Candidate

This directory contains the pose-derived representation used by the controlled
23-class Vietnamese Sign Language pilot in the FELF-SLR paper. It contains no
source videos, images, face crops, filenames, or camera metadata.

Each split is a compressed NumPy archive with:

- `left`: 40-frame left-hand features with dimension 165;
- `right`: 40-frame right-hand features with dimension 165;
- `global_context`: 40-frame global-context features with dimension 23;
- `labels`: integer class labels corresponding to `metadata.json`.

The split is 414 training, 138 validation, and 138 test sequences across 23
classes. This is a derived pose-feature release, not a raw-video dataset.

## Release status

This is a release candidate. Do not publish or mirror it until the author-signer
has provided documented consent for public release of derived pose data. Pose
trajectories can remain sensitive even when source videos are not included.

The original recordings are not redistributed. See `RELEASE_CHECKLIST.md` and
`metadata.json` for the representation and integrity information.
