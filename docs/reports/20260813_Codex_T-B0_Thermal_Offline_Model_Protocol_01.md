# SafeNest Thermal T-B0 — Offline Model Comparison Protocol & Baseline Design

## Decision

`T-B0` is complete with limitations. The protocol is registered on the
Thermal-only development branch `feature/thermal-t-b0-offline-model-protocol`
from the current `origin/main` A-stage release baseline. No full model was
trained, no canonical tensor was changed, no split was created, and no new
model artifact was generated.

`T-B1` is authorized with limitations on this development branch only. The
branch must remain unmerged until the multisensor intermediate release is
tagged.

## A-stage authority

The protocol consumes only the compact T-A6 evidence and pins the following
roles:

| Role | Rows | Representation | SHA-256 | T-B use |
| --- | ---: | --- | --- | --- |
| TRAIN | 32,000 | `[62,80]` little-endian float32 Celsius | `749c847fc9ab50ea5eee8827f0d47b5ebaa48165732a382c59f8b96c565b9d93` | fitting and TRAIN-only statistics |
| VALIDATION | 8,000 | `[62,80]` little-endian float32 Celsius | `5d16451702c1bccfa945d9188d9b29a26ce11c8b33bf7a0dfbb25bfa86d74610` | model comparison and winner selection |
| REAL_EVAL_DEVELOPMENT | 8,000 | `[62,80]` little-endian float32 Celsius | `cd696e68aeec063cbc8185719b4f4dad3d038cb3d28eec0d3701b8311e4ad8f1` | post-selection real-domain characterization only |

The official partition is preserved. Random or hash resplitting and use of
`datasets/thermal/processed_thermal_80x62.npz` as authority are prohibited.
No pristine `LOCKED_TEST` exists.

## Existing model and preprocessing

The current artifact is measured as:

```text
models/thermal/thermal_fall_int8_v0.1.0.tflite
size: 318184 bytes
sha256: 5b56da8d127ccef85f30b6459cc0cfe2d86490e41f3caa5bd2a7b70bbc46ae84
input: [1,62,80,1] int8, scale 0.003921568859368563, zero point -128
output: [1,3] int8, scale 0.00390625, zero point -128
class order: NOT_HUMAN, HUMAN_NORMAL, HUMAN_FALL
```

The model is classified as `LEGACY_DEPLOYED_REFERENCE`. Its T-A6 canonical
training lineage is not established, and it is not eligible as a reproducibly
trained B-phase winner. The actual runtime path
`inference/thermal_interpreter.py` performs finite/shape checks, per-frame
min-max normalization when a frame is outside `[0,1]`, then int8 quantization
in NHWC. This discards absolute Celsius information and is not equivalent to
the T-A6 physical artifact. The mismatch is registered for T-B1 comparison;
the interpreter and model are unchanged in T-B0.

Three small preprocessing profiles are frozen:

* `P0_CANONICAL_CELSIUS_DIRECT`: identity on Celsius, no fit, preserves physical values.
* `P1_TRAIN_FITTED_GLOBAL_ZSCORE`: mean/std fit on TRAIN only, applied unchanged to later roles.
* `P2_LEGACY_PER_FRAME_MINMAX`: compatibility-only per-frame min-max profile; it does not preserve absolute temperature.

The canonical physical tensors remain separate from model-specific preprocessing.

## Targets and evaluation roles

Source labels remain immutable: `LYING`, `SITTING`, `STANDING`, and
`EMPTY_ROOM`. The three-class compatibility target is explicit and separate:
`EMPTY_ROOM -> NOT_HUMAN`, `SITTING/STANDING -> HUMAN_NORMAL`, and
`LYING -> HUMAN_FALL`. The last mapping is a `DERIVED_POSTURE_PROXY`, not
temporal fall-event ground truth.

The primary comparison view is the full official VALIDATION partition. The
REAL partition cannot fit preprocessing, tune a model, select a winner, or
serve as a final unbiased test. A secondary near-duplicate sensitivity view
is defined, but the current compact evidence contains only counts and
truncated witnesses; therefore the exact implicated VALIDATION subset is
recorded as
`SENSITIVITY_SUBSET_NOT_MATERIALIZABLE_FROM_CURRENT_COMPACT_EVIDENCE`.

## Near-duplicate policy

The frozen T-A6 profile `THERMAL_T_A6_NEAR_DUPLICATE_SCREEN_V1` is retained.
It measured 58,467 within-TRAIN pairs and 14,514 TRAIN↔VALIDATION pairs. The
screen is deterministic but explicitly non-exhaustive. These are not exact
duplicates, but the overlap is a material validation-sensitivity limitation.
No rows may be moved, deleted, or resplit to conceal it. Full official
VALIDATION remains the winner-selection view; any later sensitivity metrics
are diagnostic only.

## Candidate and winner protocol

The legacy artifact is reference-only. The registered trainable candidates
are `SMALL_CNN_BASELINE_V1` and one deployment-oriented
`DEPTHWISE_SEPARABLE_CNN_V1`; both are preregistered and untrained. No
architecture tournament is authorized.

Winner selection is frozen before training:

1. highest VALIDATION macro F1;
2. highest VALIDATION balanced accuracy;
3. highest HUMAN_FALL posture-proxy recall;
4. lowest trainable parameter count;
5. lowest TFLite artifact size;
6. lexicographically smallest candidate ID.

The tie tolerance is `1e-5`. REAL_EVAL_DEVELOPMENT and the near-duplicate
sensitivity view cannot be tie breakers. No hard safety recall floor is
introduced because the HUMAN_FALL target is not safety ground truth.

The metric contract freezes accuracy, balanced accuracy, per-class precision,
recall and F1, macro F1, and an integer confusion matrix. Class order is
`NOT_HUMAN, HUMAN_NORMAL, HUMAN_FALL`; averaging is unweighted and zero
division is `0.0` for class metrics.

## Reproducibility and budget

The primary seed is `20260813` for Python, NumPy, TensorFlow, shuffling and
initialization. The later multi-seed confirmation list is
`[20260813, 20260814, 20260815]` and is not run in T-B0. Same-architecture
preprocessing comparisons must record pre-training weight fingerprints.

The fair baseline budget is 20 epochs, batch size 64, Adam at `0.001`,
unweighted sparse categorical cross-entropy, validation-macro-F1 early
stopping (patience 5, restore best weights), and ReduceLROnPlateau (factor
0.5, patience 3, minimum learning rate `1e-6`). One tuning trial per
candidate/profile is allowed. Augmentation, class weighting, oversampling,
and focal loss are disabled in the baseline and isolated to later phases.

## Deployment and license boundaries

Later phases must measure parameter count, TFLite size/checksum, tensor dtypes,
quantization, operators, memory where measurable, desktop latency, and actual
Raspberry Pi 5 latency separately. Desktop timing is not a Pi claim. Thermal-44
packet dtype, endianness, orientation, invalid-pixel semantics, and hardware
latency remain T-C questions.

The SDT source is cleared in the evidence for non-commercial research/model
training with citation/attribution. Commercial use, raw redistribution, and
public model-artifact release require separate terms review.

## Validation and release protection

The standalone validator independently reruns T-A6 `FULL_DATASET`, checks all
T-B0 contracts and checksums, rejects legacy- or proxy-label escalation,
rejects role/resplit changes, and verifies that no training/model result is
present. Focused tests use small metadata fixtures only. The A-stage
intermediate release remains unchanged; this B branch is explicitly held for
post-intermediate-release merge.
