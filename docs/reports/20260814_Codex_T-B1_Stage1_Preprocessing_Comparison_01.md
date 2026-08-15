# Thermal T-B1 Stage 1 — Controlled Preprocessing Comparison

## Scope and gate

This report records **T-B1 Stage 1 only**. The implementation establishes a
controlled P0/P1/P2 comparison runner and validator; it does not perform the
full 32,000-sample experiment, select a preprocessing winner, create a new
deployment model, or authorize T-B2.

| Item | Result |
| --- | --- |
| Stage-1 gate | `T_B1_STAGE1_IMPLEMENTATION_READY_WITH_LIMITATIONS` |
| Full T-B1 experiment | `PENDING_EXTERNAL_SSD_EXECUTION` |
| T-B2 | `NOT_AUTHORIZED` |
| Full canonical training | `NO` |
| New trained model | `NO` |
| Performance winner | `NOT_SELECTED` |
| T-B0 base | `0aa6cb8d1c282f07c769f108d6b6482e112c85d2` |
| Branch | `feature/thermal-t-b1-preprocessing-comparison` |

The branch is intentionally stacked on T-B0 PR #57 and is not based on, or
merged into, the current intermediate-release `main`.

## Scientific contract

The only experimental factor is the input preprocessing profile:

- `P0_CANONICAL_CELSIUS_DIRECT`: preserves the T-A6 float32 Celsius values and
  adds only the channel dimension.
- `P1_TRAIN_FITTED_GLOBAL_ZSCORE`: fits one scalar mean/std over TRAIN pixels
  only, persists the statistics identity, and applies the frozen statistics to
  VALIDATION and REAL_EVAL_DEVELOPMENT.
- `P2_LEGACY_PER_FRAME_MINMAX`: reproduces the current
  `ThermalInterpreter._prepare_float_frame` behavior, including finite checks,
  per-frame range handling, constant-frame clipping, and `[0,1]` output.

The source labels remain `LYING`, `SITTING`, `STANDING`, and `EMPTY_ROOM`.
The compatibility mapping remains `LYING → HUMAN_FALL`, `SITTING/STANDING →
HUMAN_NORMAL`, and `EMPTY_ROOM → NOT_HUMAN`; `HUMAN_FALL` is a derived posture
proxy, not temporal fall-event ground truth.

All profiles use the preregistered `SMALL_CNN_BASELINE_V1` (312,131
parameters), the same initial-weight fingerprint, seed `20260813`, Adam,
unweighted sparse categorical cross-entropy, 20-epoch maximum budget, batch
size 64, validation Macro F1 early stopping, and the T-B0 deterministic winner
hierarchy with tolerance `1e-5`. Augmentation, class weighting, oversampling,
focal loss, extra trials, and architecture changes are disabled.

## Dataset and role policy

The runner accepts only T-A6 canonical artifacts with the following pinned
identities:

| Role | Rows | Expected SHA-256 | Permitted use |
| --- | ---: | --- | --- |
| `TRAIN` | 32,000 | `749c847fc9ab50ea5eee8827f0d47b5ebaa48165732a382c59f8b96c565b9d93` | weights, P1 fit, deterministic shuffle |
| `VALIDATION` | 8,000 | `5d16451702c1bccfa945d9188d9b29a26ce11c8b33bf7a0dfbb25bfa86d74610` | early stopping and winner selection |
| `REAL_EVAL_DEVELOPMENT` | 8,000 | `cd696e68aeec063cbc8185719b4f4dad3d038cb3d28eec0d3701b8311e4ad8f1` | selected-checkpoint post-selection characterization only |

The legacy `datasets/thermal/processed_thermal_80x62.npz` remains prohibited as
new training authority. Raw SDT ZIP archives are not required by T-B1 because
the later full run consumes verified T-A6 canonical arrays.

The official VALIDATION partition remains authoritative despite the T-A6/T-B0
known `14,514` TRAIN↔VALIDATION near-duplicate pairs. Random/hash resplitting,
sample deletion, role reassignment, and an invented clean subset are
prohibited. The sensitivity subset remains
`SENSITIVITY_SUBSET_NOT_MATERIALIZABLE_FROM_CURRENT_COMPACT_EVIDENCE`.

## Runner and storage safety

`scripts/run_thermal_t_b1.py` supports configurable `--canonical-root`,
`--work-root`, and `--output-root` values. Stage 1 is fully runnable without an
external SSD and permits a tiny fixture-only one-epoch smoke run. The later
`FULL_EXPERIMENT --dry-run` verifies all role files, sizes, shape/dtype,
provenance labels, SHA-256, capacity, TensorFlow backend, model identity, and
output writability before returning `TRAINING_RUN_READY`. Execute additionally
requires an explicit owner authorization flag. Missing or disconnected
external storage fails closed; partial output is never finalized.

The intended topology is external storage for verified canonical inputs and
final results, with Mac local scratch/RAM for high-frequency training I/O.
Apple Silicon Metal is detected when available, but CPU is supported and GPU
absence does not block Stage 1. CPU/Metal bit identity is not claimed.

## Evidence and validation

Machine-readable Stage-1 evidence is under
`datasets/thermal/manifests/T-B1_preprocessing_comparison/` and includes the
execution, dataset, preprocessing, baseline, runtime, storage, initialization,
result-schema, limitation, validation, and checksum contracts.

Implemented reusable modules:

- `datasets/thermal/t_b1_preprocessing.py`
- `datasets/thermal/t_b1_model.py`
- `datasets/thermal/t_b1_runner.py`
- `scripts/run_thermal_t_b1.py`
- `scripts/validate_thermal_t_b1.py`

The standalone validator live-validates T-A6 `FULL_DATASET` and T-B0 before
checking the T-B1 evidence. It rejects stale predecessor/result evidence,
legacy-NPZ promotion, changed role hashes, non-TRAIN P1 fitting, P0/P1/P2
contract drift, baseline/seed/budget drift, REAL winner use, nonportable paths,
archive/cross-track references, and incomplete checksums.

Focused tests: **35 passed, 0 failed, 0 errors**. The fixture smoke run
executed all three profile paths with ephemeral `FIXTURE_ONLY` metrics and
reported no full-training or new-model claims. TensorFlow emitted only
environment/retracing warnings; these are not project metrics.

Predecessor validators:

- T-A6 `FULL_DATASET`: `PASS`, `PASS_WITH_LIMITATIONS` outcome.
- T-B0: `PASS`, `PASS_WITH_LIMITATIONS` outcome, `YES_WITH_LIMITATIONS` T-B1 authorization.
- T-B1 Stage 1: `PASS`, `T_B1_STAGE1_IMPLEMENTATION_READY_WITH_LIMITATIONS`.

## Limitations and next owner action

Subject/session/event generalization remains not verifiable, REAL is not a
pristine locked test, and `HUMAN_FALL` remains a posture proxy. No final TFLite
conversion, hardware validation, Thermal-44 equivalence, or performance claim
is made.

The next authorized sequence is: prepare and verify the external canonical
TRAIN/VALIDATION/REAL artifacts and provenance, run the same-configuration
full dry-run until `TRAINING_RUN_READY`, then obtain separate owner approval
for the full P0/P1/P2 training and post-selection REAL characterization.
