# Thermal T-A4 — Label Semantics, Proxy Mapping, and Ambiguity Contract

Date: 2026-08-10

Phase: `T-A4`

Outcome: `PASS_WITH_LIMITATIONS`

T-A5 authorized: `YES`

## Source truth

The selected source is the SDT real `test` split (`local_sdt_zenodo_4124309`, doi:10.5281/zenodo.4124309) with archive SHA-256 `3a838bd70835e579ecfaa820a6c0b4cbc6ba7b76729417c73845f0c959281449`. The official [Zenodo SDT record](https://zenodo.org/records/4124309) and [TU Wien documentation](https://cvl.tuwien.ac.at/research/cvl-databases/sdt-icip/) describe pose/presence labels: `LYING`, `SITTING`, `STANDING`, and `EMPTY_ROOM`. The source distribution is 2,000 rows per label, independently measured from `labels.txt`.

The original label remains immutable. Its source meaning is separate from what it may defensibly support in SafeNest. A `LYING` source annotation is verified as a lying posture, while fall-event interpretation remains ambiguous/not verifiable because T-A3 provides no timestamp, sequence, transition, onset, impact, or end evidence.

## Selected semantic policy

T-A4 evaluated L0 source-only, L1 dual-layer source-plus-proxy, and L2 direct legacy three-class collapse. The declared gates reject L2 because it rewrites source truth and creates unsupported semantic escalation. `L1_DUAL_LAYER_SOURCE_PLUS_PROXY` was selected by the declared deterministic ranking because it preserves source truth while retaining an explicitly qualified compatibility layer.

Layer B frame evidence is `HUMAN_LYING_POSTURE`, `HUMAN_SITTING_POSTURE`, `HUMAN_STANDING_POSTURE`, and `NO_ANNOTATED_HUMAN_IN_REPRESENTED_FRAME`. Layer C is optional compatibility only: `LYING`→`HUMAN_FALL` is `DERIVED_POSTURE_PROXY`, `SITTING/STANDING`→`HUMAN_NORMAL` are non-lying posture proxies, and `EMPTY_ROOM`→`NOT_HUMAN` is frame-scoped presence equivalence. None is source ground truth, temporal event ground truth, or general worker-safety ground truth.

## Coverage and ambiguity

The inventory contains 8000 deterministic label rows. Mapping types are `{'DERIVED_POSTURE_PROXY': 6000, 'DIRECT_SOURCE_EQUIVALENT': 2000}` and compatibility targets are `{'HUMAN_FALL': 2000, 'HUMAN_NORMAL': 4000, 'NOT_HUMAN': 2000}`. Unsupported activity categories are explicitly marked not represented or not verifiable; their absence is never turned into a negative example. Bending, kneeling, walking, entering, exiting, transitions, impacts, post-fall intervals, recovery, and natural/staged fall events are not established by this source.

Source-label ambiguity is false for known labels. Fall-interpretation ambiguity is represented independently. Unknown labels, unsupported targets, fake pre/post-fall fields, safety claims, and temporal escalation fail closed. Bounding boxes remain provenance/geometry evidence and do not redefine labels.

## Boundaries

T-A4 does not train, convert all frames, create splits, construct events, modify the runtime model, modify risk/fusion, or make Thermal-44, clinical, medical, or hardware claims. T-A3 remains inherited: frame-level supported; sequence/event not verifiable; window not applicable. T-A5 must preserve this semantic contract while solving grouping and splits.
