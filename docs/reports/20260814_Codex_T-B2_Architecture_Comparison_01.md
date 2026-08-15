# SafeNest Thermal T-B2 — Controlled Architecture Comparison

## Decision

`T-B2_COMPLETE_WITH_LIMITATIONS` — the frozen B1 P1 preprocessing contract was
held constant while the registered `DEPTHWISE_SEPARABLE_CNN_V1` candidate was
trained once and compared with the verified B1 `SMALL_CNN_BASELINE_V1` result.
The deterministic winner is `SMALL_CNN_BASELINE_V1`; its B1 REAL result was
reused and the losing depthwise candidate was not evaluated on REAL.

The experiment used the corrected T-B2 architecture-comparison scope. The
current master roadmap still labels its T-B2 subsection “imbalance·hard-negative
strategy”; that naming discrepancy is recorded here and does not change the
frozen experiment contract or silently authorize a later imbalance study.

## Predecessor and data locks

- Execution started from current `origin/main` commit `6cfaeeec26b7c455ab38bcad172b24b63d6c4ac2`, after PR #57 (T-B0) and PR #64 (T-B1) were merged.
- Live T-A6, T-B0, and T-B1 FULL validators all returned `PASS`.
- Canonical roles were reused without modification: TRAIN 32,000, VALIDATION 8,000, and REAL_EVAL_DEVELOPMENT 8,000; all are 62×80 little-endian float32 Celsius artifacts with the T-A6 SHA identities.
- `datasets/thermal/processed_thermal_80x62.npz` and raw SDT ZIPs were not used.
- P1 was reused from `datasets/thermal/manifests/T-B1_full_experiment/p1_preprocessing.json`: mean `22.769290618485442`, std `2.8684523405441222`, checksum `10b5da044ef33f26715544c3d4bf56e9d999d6c65139e931f6997583b3f5b816`; no VALIDATION/REAL refit occurred.
- Target semantics remained `EMPTY_ROOM→NOT_HUMAN`, `SITTING/STANDING→HUMAN_NORMAL`, `LYING→HUMAN_FALL`; HUMAN_FALL remains a derived posture proxy, not temporal fall ground truth.

## Controlled comparison

Training used primary seed `20260813`, one trial per architecture, 20-epoch maximum,
batch size 64, Adam at 0.001, unweighted sparse categorical cross-entropy,
VALIDATION Macro F1 early stopping, ReduceLROnPlateau, and no augmentation,
class weighting, oversampling, focal loss, or independent tuning.

| Candidate | Source | Parameters | VALIDATION Macro F1 | Balanced accuracy | HUMAN_FALL proxy recall |
| --- | --- | ---: | ---: | ---: | ---: |
| `SMALL_CNN_BASELINE_V1` | Reused verified T-B1 P1 result | 312,131 | 0.9951295333 | 0.995750 | 0.994 |
| `DEPTHWISE_SEPARABLE_CNN_V1` | New T-B2 training, architecture frozen before metrics | 347 | 0.9212330381 | 0.913000 | 0.842 |

The depthwise model reduces parameter count by approximately 99.889%, but its
VALIDATION Macro F1 is lower by `0.0738964952`; it is therefore not the winner.
VALIDATION is near saturation for the reused baseline, so small differences in
that regime would not establish broad architectural superiority.

## Winner and REAL characterization

- Winner rule: `THERMAL_T_B0_WINNER_RULE_001`, VALIDATION-only, tie tolerance `1e-5`.
- Winner: `SMALL_CNN_BASELINE_V1` with the existing B1 P1 checkpoint SHA `7aba32fe8d0e241546429bdc3e8cd059b10d4d8f548e9e12c4085abeba308a75`.
- REAL result source: `REUSED_VERIFIED_T_B1_RESULT`; no new REAL evaluation was needed.
- REAL_EVAL_DEVELOPMENT Macro F1 `0.5939265236`, accuracy `0.67825`, balanced accuracy `0.58275`, HUMAN_FALL proxy recall `0.446`.
- Observed winner VALIDATION→REAL Macro F1 gap: `0.4012030097`.
- REAL was not used for selection or preprocessing and is not a LOCKED_TEST, final unbiased test, device validation, clinical validation, or safety validation.

## Limitations

- 14,514 confirmed TRAIN↔VALIDATION near-duplicate pairs remain disclosed; no random/hash resplit was made.
- Subject/session/event generalization remains `NOT_VERIFIABLE`.
- The HUMAN_FALL label is a lying/posture proxy, not a temporal fall event.
- The synthetic-to-real gap is observed and not causally attributed.
- Thermal-44 device-domain validation, latency, MACs, quantization, and deployment claims were not performed.
- No pristine LOCKED_TEST exists in this experiment.

## Storage and evidence

- Bulk output/checkpoint namespace: `SafeNestAI/thermal/experiments/T-B2/T-B2_execution_result` on the external SSD.
- Git contains compact JSON evidence only under `datasets/thermal/manifests/T-B2_architecture_comparison/`; raw arrays, ZIPs, and bulk `.weights.h5` files are not tracked.
- Standalone validator: `scripts/validate_thermal_t_b2.py` — `PASS`, 0 errors, 3 non-blocking inherited limitations warnings.
- Focused tests: `tests/test_thermal_t_b2.py`.
- Next roadmap phase is `T-B3` (frame vs temporal architecture), but it is **not authorized or started** by T-B2.
