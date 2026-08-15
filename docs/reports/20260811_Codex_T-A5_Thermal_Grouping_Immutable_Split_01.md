# Thermal T-A5 — Grouping, Leakage-Resistant Split, and Immutable Assignment Policy

## Decision

T-A5 selects `S0_OFFICIAL_SOURCE_PARTITION_PRESERVATION` under `THERMAL_SPLIT_POLICY_001`.  The official SDT train, validation, and test boundaries are preserved; the real test partition is assigned only to `REAL_EVAL_DEVELOPMENT` because it was used for T-A2 geometry selection and subsequent T-A3/T-A4 development evidence.  No pristine Thermal `LOCKED_TEST` currently exists.

The selected source is the [SDT Dataset official documentation](https://cvl.tuwien.ac.at/research/cvl-databases/sdt-icip/) and [Zenodo distribution record](https://zenodo.org/records/4124309) (doi:10.5281/zenodo.4124309).  Official documentation distinguishes 32,000 synthetic train images, 8,000 synthetic validation images, and 8,000 real test images.  The local T-A1 archive inventory independently verifies the materialized real test members and labels.

## Grouping and access

No authoritative subject, session, recording, sequence, event, timestamp, scene, or per-frame camera identifier is available.  Frame index and label are provenance fields only and are never used as groups.  Consequently subject/session/event generalization is `NOT_VERIFIABLE`; a frame-random or frame-hash resplit is rejected.  The strongest verified unit is the official source partition.

T-A0 and T-A1 established source identity and bounded reader evidence.  T-A2 used a 48-frame real test pilot to compare and select geometry; T-A3 reused that pilot for temporal capability analysis; T-A4 reused the pilot and audited all 8,000 labels for semantic policy.  This history disqualifies the real test partition from pristine locked-test status even though it remains useful for development evaluation.

## Assignment roles

| Official partition | Domain | SafeNest role | Materialization | Count |
|---|---|---|---|---:|
| train | SYNTHETIC | TRAIN (planned) | LOCAL_CLOUD_PLACEHOLDER | 32,000 |
| validation | SYNTHETIC | VALIDATION (planned) | LOCAL_CLOUD_PLACEHOLDER | 8,000 |
| test | REAL | REAL_EVAL_DEVELOPMENT | LOCALLY_MATERIALIZED | 8,000 |
| independent holdout | — | LOCKED_TEST | NOT_AVAILABLE | 0 |

Every real test assignment preserves its source member/frame identity and inherits the T-A4 semantic policy.  No random seed, hash assignment, canonical tensor, augmentation, or model metric is introduced.  Derived artifacts must inherit the parent assignment; TRAIN-only augmentation is deferred to later phases.

## Limitations and gate

The T-A5 contract is `PASS_WITH_LIMITATIONS`.  T-A6 is authorized for policy/integrity work, but its full completion requires explicit authorization to hydrate the multi-gigabyte train/validation placeholders.  T-A6 does not create an unbiased final holdout.  Until an independent holdout exists, T-B may use the real test only conditionally for development evaluation.

Machine-readable evidence is under `datasets/thermal/manifests/T-A5_grouping_immutable_split/`; its checksum registry covers every JSON artifact.  The standalone validator independently recomputes candidate admissibility and selection, rechecks predecessor gates, verifies all 8,000 real assignments, and rejects tampering, absolute paths, frame-random/hash splitting, and retroactive locked-test claims.
