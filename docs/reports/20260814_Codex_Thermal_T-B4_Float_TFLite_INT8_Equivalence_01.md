# SafeNest Thermal T-B4 — Float/TFLite FP32/Full INT8 Equivalence

## THERMAL T-B4 RESULT

### PHASE

- Phase: `T-B4 — Frozen Float → TFLite FP32 → full INT8 equivalence and quantization audit`
- Gate: `T_B4_COMPLETE_WITH_LIMITATIONS`
- T-B3 merged on main: `YES` (`c65d2e32e6f14089790a8c576312eb9873e367f7`)
- Conversion performed: `YES`
- Retraining performed: `NO`
- Corrective closure: `TRUE_UNQUANTIZED_FP32_REGENERATED`; former `TFLITE_FP32` reclassified as diagnostic `TFLITE_DYNAMIC_RANGE`
- Next phase: `T-B5 — robustness · latency · candidate lock`
- T-B5 authorization: `YES_WITH_LIMITATIONS` (validator result; not executed here)

### GIT

- Starting `origin/main`: `c65d2e32e6f14089790a8c576312eb9873e367f7`
- Branch: `feature/thermal-t-b4-float-tflite-int8-equivalence`
- Final HEAD: recorded at commit time
- PR: created after validation
- PR base: `main`
- PR head: `feature/thermal-t-b4-float-tflite-int8-equivalence`
- Merge: `NOT_PERFORMED`

### PREDECESSORS

- T-A6: `PASS / PASS_WITH_LIMITATIONS`
- T-B0: `PASS / PASS_WITH_LIMITATIONS`
- T-B1 FULL: `PASS / T_B1_FULL_COMPLETE_WITH_LIMITATIONS`
- T-B2: `PASS / T_B2_COMPLETE_WITH_LIMITATIONS`
- T-B3: `PASS / T_B3_COMPLETE_WITH_LIMITATIONS`
- Reference candidate: `SMALL_CNN_BASELINE_V1` with `P1_TRAIN_FITTED_GLOBAL_ZSCORE`
- Reference seed: `20260813` (reused; no new seed selection)

### FLOAT SOURCE

- Checkpoint path: `experiments/T-B1/T-B1_execution_result/checkpoints/P1_TRAIN_FITTED_GLOBAL_ZSCORE.weights.h5` on the external SSD
- Checkpoint SHA: `7aba32fe8d0e241546429bdc3e8cd059b10d4d8f548e9e12c4085abeba308a75`
- Architecture: `SMALL_CNN_BASELINE_V1`, frame input `[1,62,80,1]`, output `[1,3]`
- Architecture fingerprint: `937fceb88900a779d67a8e407cebc3362cc21346721f92cea5ae8de1413aba2a`
- Parameters: `312131`
- P1 identity: mean `22.769290618485442`, std `2.8684523405441222`, checksum `10b5da044ef33f26715544c3d4bf56e9d999d6c65139e931f6997583b3f5b816`; TRAIN fit only
- VALIDATION Macro F1 reproduced: `0.9951295332536425`
- REAL inherited Macro F1: `0.593926523563344` (development characterization)

### DATASET

- TRAIN SHA: `749c847fc9ab50ea5eee8827f0d47b5ebaa48165732a382c59f8b96c565b9d93`
- VALIDATION SHA: `5d16451702c1bccfa945d9188d9b29a26ce11c8b33bf7a0dfbb25bfa86d74610`
- REAL SHA: `cd696e68aeec063cbc8185719b4f4dad3d038cb3d28eec0d3701b8311e4ad8f1`
- Legacy NPZ used: `NO`
- Raw ZIP used: `NO`

### REPRESENTATIVE CALIBRATION

- Policy ID: `T-B4_TRAIN_ONLY_STRATIFIED_CALIBRATION_512`
- Source role: `TRAIN`
- TRAIN-only: `YES`
- Sample count: `512` unique canonical indices
- Selection algorithm: 4 source labels × 8 TRAIN per-frame-mean Celsius quantile bands × up to 16 evenly spaced indices per stratum, with deterministic global-index backfill for sparse strata
- Class/source coverage: `EMPTY_ROOM`, `SITTING`, `STANDING`, `LYING`; all eight bands represented; 20 deterministic backfill samples due sparse outlier strata
- Temperature coverage: boundaries frozen from TRAIN frame-mean quantiles `[0,.125,.25,.375,.5,.625,.75,.875,1]`
- P1 applied: `YES`, float32 NHWC representative tensors
- Sample manifest: `representative_sample_manifest.json` and external `calibration/representative_sample_manifest.json`
- Manifest checksum: `51bbced6b40ab14d547e3c80afd99b92a24c016c1853e66c634b69d1dc4b30a4`
- VALIDATION samples used: `0`
- REAL samples used: `0`

### TFLITE FP32 (CORRECTED)

- Artifact: `artifacts/SMALL_CNN_BASELINE_V1_P1_float32.tflite` (external SSD)
- SHA: `fbe891520f07e0534b1a7074dc819d8ed44bca58b27e35c78916c3ddae73a779`
- Size: `1252048` bytes
- Input shape/dtype: `[1,62,80,1]`, `float32`
- Output shape/dtype: `[1,3]`, `float32`
- Converter TensorFlow: `2.20.0`; `optimizations: []`; no representative dataset; no float16; no dynamic-range mode; `quantization_mode: NONE`
- Internal tensor dtype inventory: `float32: 17`, `int32: 1`; quantized tensors: `0`; quantized parameter tensors: `0`; nonzero quantization tensors: `0`
- Ops: `CONV_2D ×2`, `MAX_POOL_2D ×2`, `RESHAPE`, `FULLY_CONNECTED ×2`, `SOFTMAX` (delegate shown only by interpreter inspection)
- SELECT_TF_OPS used: `NO`; builtin-only: `YES`

The former `317344`-byte file is retained externally as
`artifacts/SMALL_CNN_BASELINE_V1_P1_dynamic_range.tflite` with ID
`TFLITE_DYNAMIC_RANGE` and diagnostic-only status. It had float32 I/O but
`Optimize.DEFAULT` and two internal int8 pseudo-constants, so it was not a
true FP32 baseline. Its near-INT8 size was the expected dynamic-range
quantization size anomaly, not evidence of a smaller FP32 model.

### FULL INT8

- Artifact: `artifacts/SMALL_CNN_BASELINE_V1_P1_full_int8.tflite` (external SSD)
- SHA: `fa9730c29535477a3994c11e664474a0ca0116afaaa172889f47446ab2ac46be`
- Size: `318280` bytes (`933768` bytes / `74.579249%` smaller than the corrected true FP32 artifact; the former dynamic-range file must not be used as the denominator)
- Input dtype: `int8`
- Input scale: `0.31791284680366516`
- Input zero point: `-125`
- Output dtype: `int8`
- Output scale: `0.00390625`
- Output zero point: `-128`
- Full INT8 ops: same builtin graph; `TFLITE_BUILTINS_INT8`, no Select TF ops
- Float fallback present: `NO`
- Representative policy checksum: `c5ce8a54898a19d0b9dad156aee89feeafbf85f79a64a6e424d7912b24a95179`

### VALIDATION FLOAT↔TRUE FP32

- Argmax agreement: `1.0` (`8000/8000`)
- Disagreement count: `0`
- Probability MAE: `3.7420780291378165e-09`
- Max absolute error: `1.9073486328125e-06`
- Float Macro F1: `0.9951295332536425`
- True FP32 TFLite Macro F1: `0.9951295332536425`
- Macro F1 delta: `0.0`
- Balanced accuracy delta: `0.0`
- HUMAN_FALL proxy recall delta: `0.0`

### VALIDATION FLOAT↔INT8

- Argmax agreement: `0.997125` (`7977/8000`)
- Disagreement count: `23`
- Probability MAE: `0.0029979798683472344`
- Max absolute error: `0.8184627704322338`
- Float Macro F1: `0.9951295332536425`
- INT8 Macro F1: `0.9939946506452692`
- Macro F1 delta: `-0.0011348826083733554`
- Accuracy delta: `-0.0011250000000000426`
- Balanced accuracy delta: `-0.0017499999999999183`
- HUMAN_FALL proxy recall delta: `-0.006000000000000005`

### TRUE FP32↔INT8

- Argmax agreement: `0.997125` (`7977/8000`)
- Disagreement count: `23`
- Probability MAE: `0.0029979794821153086`
- Max absolute error: `0.8184627573937178`
- Macro F1 delta: `-0.0011348826083733554`
- Accuracy delta: `-0.0011250000000000426`
- Balanced accuracy delta: `-0.0017499999999999183`
- HUMAN_FALL proxy recall delta: `-0.006000000000000005`

### SATURATION

- Total input elements: `39,680,000` (VALIDATION)
- Lower boundary count: `14`; upper boundary count: `0`
- Actual clipping fraction: `6 / 39,680,000 = 1.5120967741935485e-7`
- Frames with clipping: `6`; boundary-value frames: `14`
- Output boundary count: lower `15,730`, upper `7,819` of `24,000` (`0.9812083333333333`)
- Interpretation: input clipping is rare; output boundary occupancy is high because the int8 softmax probability tensor uses scale `1/256` and zero point `-128`. It is reported descriptively, with no post-hoc pass threshold.

### TEMPERATURE-RANGE AUDIT

- Temperature statistic: per-frame mean Celsius
- Band boundaries source: frozen TRAIN quantiles; policy checksum `05acc4f93e07a2149e817a34832c8f83b2d3e062727b8faa1ce5ada08f1e09a5`
- Bands: eight ordered bands, VALIDATION counts `1016, 1012, 1018, 987, 982, 992, 1037, 956`
- Worst agreement band: band `7`, `0.9947698744769874`
- Worst probability-error band: band `3`, MAE `0.003485933442105815` (band 7 is effectively tied at `0.0034851443379576544`)
- Worst clipping band: not separately concentrated; total required clipping is `1.5120967741935485e-7` and is reported without a post-hoc threshold
- Interpretation: no temperature band shows a large argmax collapse; moderate probability error varies by band and must not be interpreted as deployment acceptance.

### REAL_EVAL_DEVELOPMENT

- Diagnostic performed: `YES`, fixed post-artifact parity diagnostic on all `8000` REAL rows
- Frozen before REAL evaluation: `YES`
- Float Macro F1: `0.593926523563344`
- True FP32 TFLite Macro F1: `0.593926523563344` (not used for selection)
- INT8 Macro F1: `0.6389937024875953`
- Float↔true FP32: argmax agreement `1.0`, disagreements `0`, probability MAE `2.8682436757510464e-08`, Macro F1 delta `0.0`
- True FP32↔INT8: argmax agreement `0.8565`, disagreements `1148`, probability MAE `0.08870920102212597`, Macro F1 delta `+0.04506717892425138`
- Float↔INT8: argmax agreement `0.8565`, disagreements `1148`, probability MAE `0.08870920170182328`, Macro F1 delta `+0.04506717892425138`
- INT8 clipping: `43,614 / 39,680,000 = 0.0010991431451612902`; boundary fraction `0.0068174395161290325`
- REAL used for recalibration: `NO`
- Existing synthetic→REAL gap separated from quantization gap: `YES`; the inherited gap is `0.40120300969029854`, and REAL remains `REAL_EVAL_DEVELOPMENT`, not LOCKED_TEST

### SEMANTICS

- HUMAN_FALL: Lying-derived posture proxy; not temporal fall-event ground truth
- Temporal event parity: not evaluated; T-B4 is frame-level only
- Temporal fall claim: none

### LEGACY MODEL

- Existing legacy TFLite: `models/thermal/thermal_fall_int8_v0.1.0.tflite`
- Overwritten: `NO`
- Runtime switched: `NO`
- Production manifest changed: `NO`

### ARTIFACTS

- External T-B4 root: `EXTERNAL_SSD_SAFE_NEST_AI_THERMAL_EXPERIMENTS_T_B4/T-B4_execution_result`
- Float reference: external checkpoint `experiments/T-B1/T-B1_execution_result/checkpoints/P1_TRAIN_FITTED_GLOBAL_ZSCORE.weights.h5`
- True FP32 TFLite: external `artifacts/SMALL_CNN_BASELINE_V1_P1_float32.tflite`
- Former dynamic-range artifact: external `artifacts/SMALL_CNN_BASELINE_V1_P1_dynamic_range.tflite` (diagnostic-only)
- INT8 TFLite: external `artifacts/SMALL_CNN_BASELINE_V1_P1_full_int8.tflite`
- Representative manifest: compact Git JSON plus external calibration copy
- Parity evidence: compact JSON plus external checksum-backed `parity/*.npz`
- Saturation evidence: `input_saturation_audit.json`, `output_saturation_audit.json`
- Temperature audit: `temperature_range_policy.json`, `temperature_range_error.json`
- Compact Git evidence: `datasets/thermal/manifests/T-B4_tflite_int8_equivalence/`
- Binary candidates committed to Git: `NO`

### VALIDATION

- T-B4 validator: `PASS`, `T_B4_COMPLETE_WITH_LIMITATIONS`, `6` warnings, `0` errors
- Errors: `0`
- Warnings: `6`
- Focused tests: `11 passed`, `0 failed`, `0 errors`, `0 skipped`
- Thermal regression: `318 passed`, `0 failed`, `0 errors`, `5 skipped` (21 warnings)
- Contract regression: T-A6, T-B0, T-B1 FULL, T-B2, T-B3 live validators all `PASS`
- Broader regression: `1038 passed`, `201 failed`, `5 errors`, `6 skipped` (86 warnings); failures/errors are pre-existing archive/CO₂/mmWave dependency or payload-availability issues outside T-B4; no Thermal T-B4 failures
- Compile/import: `python3 -m py_compile` passed for T-B4 runner, validator, and focused tests
- git diff --check: passed before staging
- New T-B4 failures: none

### LIMITATIONS

- TRAIN↔VALIDATION near duplicates: `14,514` inherited pairs
- VALIDATION near saturation: output boundary occupancy is high under the measured softmax int8 tensor; input clipping is rare
- pristine LOCKED_TEST: unavailable; REAL is development-only
- HUMAN_FALL posture proxy: Lying-derived, not temporal event ground truth
- temporal provenance: not verified/evaluated in this frame phase
- subject/session/event generalization: not verifiable
- synthetic→REAL domain gap: inherited observed gap `0.40120300969029854`, separate from quantization parity
- Thermal-44 validation: not performed; deferred to T-C
- Pi latency: not measured; deferred to T-B5/device work

### PARALLEL SAFETY

- Direct T-B4 files: runner, validator, focused tests, report, and compact T-B4 evidence only
- CO₂: untouched
- mmWave: untouched
- Integration: untouched
- Shared: untouched
- Raw payload staged: `NO`
- Canonical payload staged: `NO`
- H5 checkpoint staged: `NO`
- Production model replacement: `NO`

### GIT FINAL

- Commit: recorded after final validation
- Push: performed for the T-B4 branch
- PR: created against `main`
- CI: pending GitHub review/CI
- Merge: not performed

### NEXT

- Exact next phase: `T-B5 — robustness · latency · candidate lock`
- Authorization: `YES_WITH_LIMITATIONS`; do not start it in this branch
- T-B5 work performed: `NONE`

## Evidence classification

- Locally measured: SSD artifact hashes/sizes, canonical role hashes, tensor metadata, conversion graph, parity, saturation, and temperature-band metrics.
- Repository-code verified: frozen P1, architecture, class map, predecessor validators, and legacy-model non-replacement policy.
- Official/inherited machine-readable evidence: T-A6 through T-B3 role identities and limitations.
- Inferred/interpretive: output boundary occupancy explanation; it is explicitly not an acceptance claim.
- Unknown/deferred: subject/session/event generalization, pristine locked test, Thermal-44 unit/domain equivalence, and Pi latency.
