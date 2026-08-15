# SafeNest Thermal T-B3 — Frame-Only Multi-Seed Confirmation

## THERMAL T-B3 RESULT

### PHASE

- Phase: `T-B3 — frame-only multi-seed stability confirmation`
- Gate: `T_B3_COMPLETE_WITH_LIMITATIONS`
- Roadmap reconciliation present on main: `YES` — PR #70 merge `22f745ac2f730e3a6c73d0a335c3e12644b03913`
- Temporal training performed: `NO`
- Multi-seed experiment completed: `YES` — frozen seeds `20260813`, `20260814`, `20260815`
- Next phase: `T-B4 — Float→TFLite→INT8 equivalence`
- Next-phase authorization: `YES_WITH_LIMITATIONS` after owner review; T-B4 was not started

### GIT

- Starting `origin/main`: `22f745ac2f730e3a6c73d0a335c3e12644b03913`
- T-B3 branch: `feature/thermal-t-b3-frame-multiseed-confirmation`
- Current `origin/main` at final audit: `22f745ac2f730e3a6c73d0a335c3e12644b03913`
- Direct commits: readiness contract `2ee8aa6`; second direct commit contains final evidence, validator, tests, and report
- PR base: `main`
- PR head: `feature/thermal-t-b3-frame-multiseed-confirmation`
- Merge: `NOT PERFORMED`

### PREDECESSORS

- T-A6: `PASS_WITH_LIMITATIONS`
- T-B0: `PASS_WITH_LIMITATIONS`
- T-B1 FULL: `T_B1_FULL_COMPLETE_WITH_LIMITATIONS`
- T-B2: `T_B2_COMPLETE_WITH_LIMITATIONS`
- T-B2 architecture winner: `SMALL_CNN_BASELINE_V1`, `312131` parameters
- Temporal feasibility: `NOT_SUPPORTED_BY_CURRENT_DATASET_PROVENANCE`

### DATASET

- TRAIN: `32000` frames, float32 little-endian Celsius, SHA-256 `749c847fc9ab50ea5eee8827f0d47b5ebaa48165732a382c59f8b96c565b9d93`
- VALIDATION: `8000` frames, float32 little-endian Celsius, SHA-256 `5d16451702c1bccfa945d9188d9b29a26ce11c8b33bf7a0dfbb25bfa86d74610`
- REAL: `8000` frames, `REAL_EVAL_DEVELOPMENT` only, SHA-256 `cd696e68aeec063cbc8185719b4f4dad3d038cb3d28eec0d3701b8311e4ad8f1`; not evaluated for new seeds
- Canonical checksum validation: `PASS`
- Legacy NPZ used: `NO`
- Raw ZIP used: `NO`

### P1

- Mean: `22.769290618485442`
- Std: `2.8684523405441222`
- Checksum: `10b5da044ef33f26715544c3d4bf56e9d999d6c65139e931f6997583b3f5b816`
- Reused from T-B1: `YES`
- Refit: `NO`
- VALIDATION/REAL fit: `NO`

### ARCHITECTURE

- Candidate: `SMALL_CNN_BASELINE_V1`, fingerprint `937fceb88900a779d67a8e407cebc3362cc21346721f92cea5ae8de1413aba2a`
- Parameters: `312131`
- Device-domain validation: `NOT_PERFORMED — deferred to T-C`

### MULTI-SEED VALIDATION

| Seed | Source | Best epoch | VALIDATION Macro F1 | Accuracy | Balanced accuracy | HUMAN_FALL posture-proxy recall | Checkpoint |
|---:|---|---:|---:|---:|---:|---:|---|
| 20260813 | reused verified T-B1 | 15 | 0.9951295333 | 0.995125 | 0.995750 | 0.994 | inherited SHA `7aba32fe…`, 3,777,416 bytes |
| 20260814 | new training | 11 | 0.9956265742 | 0.995625 | 0.9959167 | 0.993 | external SHA `b33dff43…`, 3,777,416 bytes |
| 20260815 | new training | 12 | 0.9943760728 | 0.994375 | 0.9947500 | 0.991 | external SHA `d2d9c4ec…`, 3,777,416 bytes |

Aggregate over the three frozen seeds (population standard deviation):

- Macro F1 mean `0.9950440601`, std `0.0005140802`, min `0.9943760728`, max `0.9956265742`, range `0.0012505014`
- Balanced accuracy mean `0.9954722222`, worst `0.9947500000`
- HUMAN_FALL posture-proxy recall mean `0.9926666667`, worst `0.9910000000`
- No predefined threshold and no best-seed checkpoint cherry-picking

### LIMITATIONS

- TRAIN↔VALIDATION near-duplicate audit remains `14,514` pairs.
- `REAL_EVAL_DEVELOPMENT` is not a pristine `LOCKED_TEST`; no new multi-seed REAL evaluation was performed.
- `HUMAN_FALL` remains the source `LYING` posture proxy, not temporal fall-event ground truth.
- Subject/session/event-independent generalization is `NOT_VERIFIABLE`.
- The observed inherited P1 synthetic-to-REAL development gap is `0.4012030097`; it is not causally attributed here.
- Thermal-44/device-domain validation was not performed and remains deferred to T-C.
- T-B3 does not change the candidate checkpoint and does not create a deployable TFLite/INT8 artifact.

### STORAGE

- SSD output root: `SSD_EXTERNAL_OUTPUT_ROOT/T-B3_execution_result` (resolved by the runner under the configured external SSD namespace)
- Bulk checkpoints on SSD: `2` new checkpoints, plus the inherited reference identity; retained outside Git
- Bulk artifacts committed to Git: `0`
- Git contains compact JSON evidence, checksums, validator, tests, and this report only

### VALIDATION

- T-B3 standalone validator: `PASS`, `0` errors, `4` limitation warnings
- Focused T-B3 tests: `17 passed`, `0 failed`, `0 errors`, `0 skipped`
- Predecessor validators: A6/B0/B1/B2 all live `PASS`
- Compile/import checks: `PASS`
- `git diff --check`: `PASS`
- New T-B3 failures: none after the summary-envelope correction; no retraining was needed

### PARALLEL SAFETY

- Direct T-B3 files: only `datasets/thermal/t_b3_runner.py`, `scripts/run_thermal_t_b3.py`, `scripts/validate_thermal_t_b3.py`, T-B3 manifest files, T-B3 tests, and this report
- CO₂: `0`
- mmWave: `0`
- Integration/shared files: `0`
- Raw payload staged: `0`
- Canonical NPY staged: `0`
- Bulk checkpoint staged: `0`

### NEXT

- Exact roadmap phase: `T-B4 — Float→TFLite→INT8 equivalence`
- Authorization: `YES_WITH_LIMITATIONS`, pending owner review/merge
- T-B4 work performed: `NONE`

## Evidence and provenance note

All metrics above are `LOCALLY_MEASURED` from the external SSD execution bundle and independently recomputed/validated by `scripts/validate_thermal_t_b3.py`. Dataset identities and inherited metrics are `REPOSITORY_CODE_VERIFIED` plus `LOCALLY_MEASURED` where the canonical files were re-hashed. The posture, grouping, near-duplicate, REAL-domain, and device-domain statements are retained as explicit limitations, not upgraded to unsupported claims.

## Hard stop

T-B3 is terminal at `T_B3_COMPLETE_WITH_LIMITATIONS`. No fourth seed, REAL multi-seed evaluation, temporal sequence construction, class weighting, augmentation, TFLite conversion, INT8 quantization, candidate replacement, T-C, or later phase was started.
