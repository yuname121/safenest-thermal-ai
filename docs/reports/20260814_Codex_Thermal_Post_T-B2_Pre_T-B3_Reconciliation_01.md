# SafeNest Thermal post-T-B2 / pre-T-B3 reconciliation

## Decision

`NEXT_ACTION = DEFER_TEMPORAL_COMPARISON_AND_DEFINE_VALID_FRAME_ONLY_NEXT_PHASE`.

This is a pre-execution gate, not a model experiment. No T-B3 training,
pseudo-sequence construction, additional seed, quantization, hardware test, or
later Thermal phase was performed.

## Git and phase authority

- Current live `origin/main`: `07f1cdefa0775c6525101bb83546b89acc5e3c13`.
- PR #69 is merged. Its merge commit is
  `d429bd787f320d33bcf892187d3d80ece27c7c53`, and it is an ancestor of the
  current `origin/main`.
- T-B2 evidence is present under
  `datasets/thermal/manifests/T-B2_architecture_comparison/`; its standalone
  validator returns `evidence_validation=PASS` and
  `overall_outcome=T_B2_COMPLETE_WITH_LIMITATIONS`.
- The only active B-series roadmap found is the master roadmap
  `docs/20260810_ChatGPT_SafeNest_Multisensor_Parallel_Execution_Roadmap_01.md`.
  No newer Thermal-specific roadmap supersedes it.
- The roadmap therefore has `MASTER_ROADMAP_PHASE_DRIFT`: its T-B2 heading
  still says imbalance/hard-negative strategy, while the committed T-B2
  evidence records a controlled frame-architecture comparison.

## Evidence-backed phase mapping

| Phase | Roadmap intent | Completed operation | Authority |
| --- | --- | --- | --- |
| T-B0 | Protocol and baseline | Offline protocol, role policy, candidate preregistration; no training | `T-B0` validator PASS |
| T-B1 | Preprocessing/augmentation | Three preprocessing profiles; P1 selected from VALIDATION | `T-B1` FULL validator PASS |
| T-B2 | Imbalance/hard negatives in the old roadmap | Controlled frame-architecture comparison; SMALL_CNN winner | `T-B2` validator PASS |
| T-B3 | Frame vs temporal architecture | Not started; temporal branch is currently blocked | This gate |

No historical T-B1/T-B2 evidence, branch, manifest, or report is renamed or
rewritten by this reconciliation.

## Temporal feasibility gate

The active T-A3 evidence classifies SDT as frame-level only:

| Field | Status | Evidence |
| --- | --- | --- |
| Acquisition timestamp | `ABSENT` | `T-A3 temporal_capability_contract.json` |
| FPS/cadence | `NOT_VERIFIABLE` | `T-A3 temporal_evidence_registry.json` |
| Sequence/recording ID | `ABSENT` | T-A3 source-schema contract |
| Session ID | `ABSENT` | T-A3 source-schema contract |
| Event ID and fall boundaries | `ABSENT` / `NOT_VERIFIABLE` | `event_policy.json` |
| Chronological adjacency | `NOT_VERIFIABLE` | `sequence_policy.json` |
| Continuous-recording provenance | `NOT_VERIFIABLE` | T-A3 grouping/temporal evidence |
| Frame index as time | `NO` | Structural provenance only; filename/archive order is explicitly not time |

`TEMPORAL_TRAINING_FEASIBILITY = NO`. Constructing sequences by grouping rows,
filenames, ZIP order, or assuming an FPS would manufacture temporal evidence.

## T-B2 state and unfinished original work

The frozen P1 preprocessing profile is
`P1_TRAIN_FITTED_GLOBAL_ZSCORE`. The architecture comparison selected
`SMALL_CNN_BASELINE_V1` (312,131 parameters) over
`DEPTHWISE_SEPARABLE_CNN_V1` (347 parameters):

- SMALL_CNN VALIDATION Macro F1: `0.9951295332536425`.
- DEPTHWISE VALIDATION Macro F1: `0.9212330380736017`.
- Winner REAL_EVAL_DEVELOPMENT Macro F1: `0.593926523563344`.
- Observed validation-to-real gap: `0.40120300969029854`.
- REAL remains development characterization, not `LOCKED_TEST`.

The original imbalance/hard-negative study remains unperformed. The verified
source labels are only `LYING`, `SITTING`, `STANDING`, and `EMPTY_ROOM`.
`BENDING` and `PARTIAL_BODY` are not represented by verified source labels, so
those hard-negative slices cannot be fabricated. The current T-B2 contract
kept class weighting, oversampling, and focal loss disabled.

Only primary seed `20260813` was used in T-B1/T-B2. The registered list
`[20260813, 20260814, 20260815]` contains reserved seeds, not completed
experiments. The roadmap minimum-three-seed requirement is therefore
`UNSATISFIED`.

## Gate P2 audit

| Item | Status | Basis |
| --- | --- | --- |
| Preprocessing | `DONE_WITH_LIMITATIONS` | T-B1 P1 winner; proxy labels and domain gap remain |
| Architecture | `DONE_WITH_LIMITATIONS` | T-B2 frame comparison; no device latency/MAC evidence |
| Multi-seed | `UNSATISFIED` | One primary seed; three seeds only registered |
| Float model | `DONE_WITH_LIMITATIONS` | Winner checkpoint exists externally; compact SHA/size tracked |
| Representative dataset policy | `DONE_WITH_LIMITATIONS` | T-B0 policy registered; T-B4 representative set not executed/locked |
| TFLite conversion | `NOT_DONE` | T-B4 not started |
| INT8 quantization | `NOT_DONE` | T-B4 not started |
| Float/TFLite/INT8 equivalence | `NOT_DONE` | No parity evidence |
| Robustness | `NOT_DONE` | T-B5 not started |
| Latency | `NOT_DONE` | No Mac/Pi candidate measurements |
| Candidate lock | `NOT_DONE` | No final artifact/metadata lock |
| `LOCKED_TEST` | `NOT_POSSIBLE_WITH_CURRENT_DATA` | No pristine holdout; REAL is development-only |

Overall P2 is not satisfied.

## Required roadmap correction

The minimal roadmap note records that the committed T-B2 architecture result
is authoritative, the old imbalance/hard-negative question remains open, and
the next eligible offline work is a proposed **frame-only multi-seed
confirmation** of the frozen candidate. Temporal comparison remains deferred
until a separately approved source/provenance amendment supplies ordered
recordings and event context. T-B4 and T-B5 retain their conversion and
robustness/lock roles, but inherit the amended frame-level scope if no temporal
source is added.

## Future temporal data requirement

Any future temporal candidate needs consented, safely staged or public data
with verified subject, session, recording/sequence, event, timestamp/FPS,
pre-fall, transition/impact, post-fall lying, and recovery provenance. This
does not authorize data collection or move temporal training automatically into
T-C; T-C remains device-domain validation of a frozen contract.

## Hard stop

- T-B3 training: **NO**
- T-B4: **NO**
- T-C: **NO**
- T-D: **NO**
