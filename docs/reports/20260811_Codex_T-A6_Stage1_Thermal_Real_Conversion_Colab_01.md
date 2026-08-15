# SafeNest Thermal T-A6 Stage 1 — Real Conversion and Colab Package

## Decision

- Stage-1 gate: `T_A6_STAGE1_COMPLETE`
- Full T-A6 gate: `NOT_YET_COMPLETE`
- T-B authorized: `NO`
- Mac synthetic access: `PROHIBITED_STAGE_1`

This report covers only the T-A6 Stage-1 implementation.  It does not train,
evaluate, normalize, quantize, or invoke the Thermal model, and it does not
infer any Thermal-44 hardware contract.

## Real source and artifact

The only Stage-1 source is `datasets/raw_archives/thermal_split_zips/test.zip` with the locked T-A1 identity.
The intended output is `datasets/thermal/artifacts/T-A6_real_eval_development/real_eval_development_canonical.npy`,
an `(8000, 62, 80)` little-endian float32 Celsius memmap in ascending source
frame order.  Current conversion status is `REAL_CONVERSION_FINALIZED`;
artifact status is `FINALIZED`.

## Integrity findings

- Source accounting: `8000` measured of 8,000; status counts `{"EXCLUDED": 0, "FAILED": 0, "SUCCESS": 8000, "SUCCESS_WITH_WARNING": 0}`.
- Quality status: `PASS`; quality summary `{"channel_mismatch": 0, "constant_canonical": 0, "constant_source": 0, "corrupt": 0, "dtype_mismatch": 0, "full_extreme": 0, "full_invalid": 0, "nonfinite": 0, "partial_extreme_warnings": 0, "shape_mismatch": 0, "silent_skips": 0, "temperature_distribution": {"finite_pixel_count": 39680000, "frame_maximum_p99": 35.16240539550781, "frame_mean_p01": 20.742431080818175, "frame_mean_p50": 24.226159711614734, "frame_mean_p99": 26.020483127236368, "frame_minimum_p01": 19.484200191497802, "maximum_celsius": 36.17790222167969, "mean_celsius": 24.046884845543822, "minimum_celsius": 18.418468475341797, "std_celsius": 1.8435828784710004}, "truncated": 0, "warning_code_counts": {}}`.
- Exact duplicates: `COMPLETE`; scope is within REAL_EVAL_DEVELOPMENT only.
- Near duplicates: `COMPLETE`; profile is deterministic, label/model independent, and explicitly screening rather than exhaustive.
- Determinism: `PASS`; repeated checksum match `True`.
- Subject/session/event generalization remains `NOT_VERIFIABLE` because the source does not provide those identifiers.

## Deliberate Stage-1 stop

The Mac runner never reads, hashes, copies, extracts, reconstructs, or streams
`train.zip.001`–`.004` or `validation.zip`.  Synthetic TRAIN/VALIDATION and all
cross-partition duplicate/leakage audits remain `PENDING_COLAB_STAGE2`.  The
Colab runner accepts configurable Drive/work/output roots, rejects incomplete
uploads, identifies multipart format before reconstruction, stages heavy I/O
through `/content` when possible, supports partition-level resume, and writes a
small execution-result bundle.  It is not auto-started here.

## Evidence

Compact evidence is under `datasets/thermal/manifests/T-A6_full_conversion_integrity/`; full canonical tensors and JSONL
provenance remain Git-ignored.  The standalone validator independently
rechecks T-A0–T-A5, finalization, 1:1 alignment, quality, duplicate scope,
determinism, path portability, and the synthetic pending gate.
