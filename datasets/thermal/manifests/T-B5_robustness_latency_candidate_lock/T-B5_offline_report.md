# Thermal T-B5 — Offline Robustness, Mac Latency, Candidate Lock

- Status: `T_B5_COMPLETE_WITH_LIMITATIONS` (compact validator required)
- Selection role: `VALIDATION` only; `REAL_EVAL_DEVELOPMENT` was diagnostic only.
- Candidate lock: `FULL_INT8` / `OFFLINE_INT8_CANDIDATE_LOCKED_WITH_LIMITATIONS`.
- LOCKED_TEST: unavailable; no final unbiased test claim.

## Frozen contract

- Profile: `THERMAL_T_B5_ROBUSTNESS_PROFILE_001`; checksum `c333d38de170eb7acb6245adf23d536f09effe75db620afde7ab1000c8406877`; deterministic VALIDATION sample count: 512.
- Architecture: `SMALL_CNN_BASELINE_V1`; P1 is TRAIN-fitted; seed `20260813`; retraining/recalibration/conversion: none.
- Dynamic-range TFLite is diagnostic-only and ineligible for the equivalence chain.

## Clean VALIDATION metrics (same deterministic subset)

- `FLOAT_KERAS`: macro F1 `0.998050667`, accuracy `0.998046875`, balanced accuracy `0.998697917`, HUMAN_FALL posture-proxy recall `1.000000000`.
- `TFLITE_FP32`: macro F1 `0.998050667`, accuracy `0.998046875`, balanced accuracy `0.998697917`, HUMAN_FALL posture-proxy recall `1.000000000`.
- `FULL_INT8`: macro F1 `0.994128993`, accuracy `0.994140625`, balanced accuracy `0.993489583`, HUMAN_FALL posture-proxy recall `0.984375000`.

## Robustness

Perturbations are synthetic offline diagnostics, not Thermal-44 validation. Missing-frame handling is fail-closed with no zero/last/mean imputation.

- `AMBIENT_OFFSET` `-2.0`: FULL_INT8 macro F1 `0.905687665`; delta vs clean `-0.088441328`.
- `AMBIENT_OFFSET` `2.0`: FULL_INT8 macro F1 `0.493051903`; delta vs clean `-0.501077091`.
- `DEAD_PIXEL` `0.01`: FULL_INT8 macro F1 `0.995446501`; delta vs clean `0.001317507`.
- `DEAD_PIXEL` `0.05`: FULL_INT8 macro F1 `0.900902230`; delta vs clean `-0.093226763`.
- `PARTIAL_OCCLUSION` `0.05`: FULL_INT8 macro F1 `0.990233071`; delta vs clean `-0.003895922`.
- `PARTIAL_OCCLUSION` `0.15`: FULL_INT8 macro F1 `0.936651135`; delta vs clean `-0.057477858`.
- `HOT_OBJECT` `5.0`: FULL_INT8 macro F1 `0.509695961`; delta vs clean `-0.484433033`.
- `HOT_OBJECT` `10.0`: FULL_INT8 macro F1 `0.387827166`; delta vs clean `-0.606301828`.
- `MISSING_FRAME` `single_frame`: `PIPELINE_CONTRACT_FAULT_FAIL_CLOSED`; model inference performed: `false`.
- `ORIENTATION_ERROR` `horizontal_flip`: FULL_INT8 macro F1 `0.992156262`; delta vs clean `-0.001972732`.
- `ORIENTATION_ERROR` `vertical_flip`: FULL_INT8 macro F1 `0.460990894`; delta vs clean `-0.533138099`.
- `ORIENTATION_ERROR` `rotate_180`: FULL_INT8 macro F1 `0.464238311`; delta vs clean `-0.529890683`.

## Mac latency (microseconds)

CPU/XNNPACK-if-available, one thread, batch one, 20 warmups and 200 measured iterations. This is not Raspberry Pi or sensor-to-alarm latency.

- `TFLITE_FP32` invoke-only mean/median/p95/p99: `216.791` / `215.250` / `221.904` / `236.067` us; preprocess+invoke mean: `227.054` us.
- `FULL_INT8` invoke-only mean/median/p95/p99: `149.186` / `148.166` / `156.642` / `168.048` us; preprocess+invoke mean: `159.927` us.

## Limitations and handoff

- REAL_EVAL_DEVELOPMENT is not LOCKED_TEST; its T-B4 INT8 sensitivity is diagnostic only.
- HUMAN_FALL is a Lying-derived posture proxy, not temporal fall ground truth.
- Subject/session/event generalization is not verifiable; TRAIN-VALIDATION near-duplicate overlap (14,514 pairs) remains disclosed.
- Thermal-44 device-domain validation and Raspberry Pi latency are deferred to T-C.
