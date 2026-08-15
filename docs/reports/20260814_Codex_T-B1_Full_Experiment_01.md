# SafeNest Thermal T-B1 Full Experiment

## Decision

`T-B1` completed as `T_B1_FULL_COMPLETE_WITH_LIMITATIONS`. The full experiment ran on the owner-authorized external SSD canonical artifacts after a `TRAINING_RUN_READY` dry-run. T-B2 was not started. The validator records `T-B2 authorization: YES_WITH_LIMITATIONS`; this is an authorization gate only, not permission to begin that phase in this branch.

Evidence categories used below: `LOCALLY_MEASURED` for SSD measurements, `REPOSITORY_CODE_VERIFIED` for the runner/validator contract, and `OFFICIAL_EXTERNAL_SOURCE_VERIFIED` only where inherited T-A6/T-B0 evidence already carried that status.

## SSD import and canonical verification

The external volume was mounted as `SafeNestssd` and remained read-write with sufficient free capacity. The two downloaded T-A6 archives were preserved, and their source archive SHA-256 values were recorded in `ssd_import_manifest.json`. The canonical archive was retained under the imported source area and its role files were copied into the standard logical layout:

```text
SafeNestAI/thermal/canonical/
├── TRAIN/
├── VALIDATION/
└── REAL_EVAL_DEVELOPMENT/
```

The T-A6 execution evidence was preserved under `SafeNestAI/thermal/evidence/T-A6_execution_result/`. Existing SDT ZIPs were not recompressed, concatenated, deleted, or used as a silent fallback; they remained at their original SSD location. No destructive deletion was performed.

All three canonical roles were independently re-hashed and checked for tensor/provenance row alignment:

| Role | Rows | Shape | Dtype/unit | Tensor SHA-256 | Provenance SHA-256 | Target distribution |
|---|---:|---|---|---|---|---|
| TRAIN | 32,000 | 32,000 × 62 × 80 | little-endian float32 / Celsius | `749c847fc9ab50ea5eee8827f0d47b5ebaa48165732a382c59f8b96c565b9d93` | `b4ec8228e35703bac3319ca218a69fdef43013ea44b23376da4929b274d24888` | NOT_HUMAN 8,000; HUMAN_NORMAL 16,000; HUMAN_FALL 8,000 |
| VALIDATION | 8,000 | 8,000 × 62 × 80 | little-endian float32 / Celsius | `5d16451702c1bccfa945d9188d9b29a26ce11c8b33bf7a0dfbb25bfa86d74610` | `48ebd03ca6f8d738ad7048aa72d4c454fd821140aa887971c27c5b49c1d7ec63` | NOT_HUMAN 2,000; HUMAN_NORMAL 4,000; HUMAN_FALL 2,000 |
| REAL_EVAL_DEVELOPMENT | 8,000 | 8,000 × 62 × 80 | little-endian float32 / Celsius | `cd696e68aeec063cbc8185719b4f4dad3d038cb3d28eec0d3701b8311e4ad8f1` | `c9d12f12d845d218e5636dad84a4a094e869faa29d95feb4a6f69603c195e550` | NOT_HUMAN 2,000; HUMAN_NORMAL 4,000; HUMAN_FALL 2,000 |

`LYING → HUMAN_FALL` remains a derived posture proxy, not temporal fall-event ground truth. The REAL role was never used for preprocessing fitting or winner selection and is reported only as post-selection development characterization.

## Controlled full experiment

The runner used `SMALL_CNN_BASELINE_V1` (312,131 parameters), architecture fingerprint `937fceb88900a779d67a8e407cebc3362cc21346721f92cea5ae8de1413aba2a`, seed `20260813`, and the frozen T-B0 unweighted training budget. The host used CPU execution with TensorFlow 2.20.0, NumPy 1.26.4, Python 3.9.6, and no visible GPU. The three profiles used identical data roles, architecture, initial weights, seed, and budget; only preprocessing differed.

| Profile | Best epoch | Validation Macro F1 | Balanced accuracy | HUMAN_FALL proxy recall |
|---|---:|---:|---:|---:|
| P0 canonical Celsius direct | 11 | 0.9724235030 | 0.9690833333 | 0.9270 |
| P1 TRAIN-fitted global z-score | 15 | **0.9951295333** | **0.9957500000** | **0.9940** |
| P2 legacy per-frame min-max | 10 | 0.9916275375 | 0.9920000000 | 0.9855 |

Winner selection was recomputed from VALIDATION only under `THERMAL_T_B0_WINNER_RULE_001` with tolerance `1e-5`; P1 won. The persistent SSD checkpoint identity is:

```text
logical path: checkpoints/P1_TRAIN_FITTED_GLOBAL_ZSCORE.weights.h5
size: 3,777,416 bytes
SHA-256: 7aba32fe8d0e241546429bdc3e8cd059b10d4d8f548e9e12c4085abeba308a75
```

P0 and P2 checkpoint SHAs are also retained in `checkpoint_registry.json`. Checkpoint bytes are intentionally not tracked in Git.

### REAL_EVAL_DEVELOPMENT

The frozen P1 winner was evaluated once on the REAL role after selection:

```text
Macro F1:          0.5939265236
Accuracy:          0.6782500000
Balanced accuracy: 0.5827500000
HUMAN_FALL recall: 0.4460000000
```

This is not a pristine locked test, deployment validation, or clinical result. The complete confusion matrix and per-class metrics are in `real_eval_development.json`.

## Validation and limitations

The standalone full validator passed both against the materialized SSD bundle and against the compact Git evidence mirror. The SSD bundle reports 3 warnings; the Git mirror reports 6 because the three checkpoint byte files are intentionally external and only their SHA/size identities are tracked. T-A6 and T-B0 live predecessor validators also passed.

The following inherited limitations remain mandatory disclosures:

- 14,514 TRAIN–VALIDATION near-duplicate pairs were measured; the clean sensitivity subset is not materialized in the compact evidence.
- Subject/session/event generalization is not verifiable from the available provenance.
- REAL_EVAL_DEVELOPMENT is not an untouched `LOCKED_TEST`.
- `LYING → HUMAN_FALL` is a posture proxy and cannot support temporal fall-event or safety claims.
- The synthetic-to-real domain gap is large: the selected P1 profile scores 0.9951 Macro F1 on synthetic VALIDATION versus 0.5939 on REAL_EVAL_DEVELOPMENT. This is a development characterization, not deployment validation.
- Thermal-44 domain equivalence and hardware behavior remain deferred to T-C.
- T-B0's non-commercial/license release restriction remains subject to manual review.

## Tracked evidence

The compact mirror is `datasets/thermal/manifests/T-B1_full_experiment/`. Its `checksums.sha256` covers every tracked JSON artifact. The SSD import/canonical manifest records the preserved source archives and explicitly records that raw archives, canonical tensors, and bulk checkpoints are not tracked in Git.

No T-B2 code, model comparison, quantization, deployment artifact, or hardware validation was started.
