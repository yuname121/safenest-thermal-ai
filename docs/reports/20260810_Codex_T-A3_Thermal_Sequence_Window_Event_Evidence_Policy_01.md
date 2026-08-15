# Thermal T-A3 — Sequence, Window, and Event-Evidence Policy

Date: 2026-08-10

Phase: `T-A3`

Outcome: `PASS_WITH_LIMITATIONS`

T-A4 authorized: `YES`

## Source decision

The selected local source is the SDT test archive `datasets/raw_archives/thermal_split_zips/test.zip` with SHA-256 `3a838bd70835e579ecfaa820a6c0b4cbc6ba7b76729417c73845f0c959281449`. The [Zenodo SDT record](https://zenodo.org/records/4124309) and [TU Wien SDT documentation](https://cvl.tuwien.ac.at/research/cvl-databases/sdt-icip/) describe thermal/depth image pairs and pose labels (LYING, SITTING, STANDING, EMPTY_ROOM), not a timestamped fall-event stream. The active T-A1 reader independently measures one-to-one member/index/label linkage and explicitly records timestamp, subject, session, sequence, and event fields as absent.

## Temporal boundary

T-A3 freezes a supported `FRAME_LEVEL` contract for 48 real frames ({'0': 12, '1': 12, '2': 12, '3': 12} by original class). A member filename or integer index identifies source provenance only. It is not a timestamp, FPS proxy, or proof that neighboring indices belong to one sequence. An index gap is reported as structural archive evidence; it is not labeled a dropped acquisition frame.

`SEQUENCE_LEVEL` is `NOT_VERIFIABLE`, `EVENT_LEVEL` is `NOT_VERIFIABLE`, and `WINDOW_LEVEL` is `NOT_APPLICABLE`. No FPS, timestamps, sequence IDs, event IDs, fall onset/end, pre/during/post ranges, duration, stride, or overlap values are fabricated.

## Labels and events

The original `LYING` label is preserved as posture semantics. It is not silently relabeled as a fall event: without transition, onset, impact, end, and surrounding context, a lying frame cannot establish when or whether a fall occurred. A later event policy must use authoritative temporal evidence rather than a posture run.

## Representation and geometry

The source remains radiometric temperature encoded as uint16 PNG values with the T-A1 Celsius conversion. Each pilot record retains source archive/member/index, raw encoded hash, original bbox and pose label, and the selected T-A2 geometry profile `G1_FIXED_ASPECT_CROP_BILINEAR` with canonical frame hash. T-A3 does not train, infer, normalize, relabel, split, or perform full conversion.

## Gap, duplicate, and downstream limits

Duplicate member names/indices and missing structural indices are checked from the ZIP central directory. Exact duplicate content would retain both provenance records and be flagged; near-duplicate/complete acquisition-level audit is deferred to T-A6 because no timeline exists. Train/validation placeholders are not hydrated. Thermal-44 FPS, clock, drop semantics, buffering, and hardware validation remain deferred to T-C.

Evidence categories used: `LOCALLY_MEASURED`, `VALIDATOR_INHERITED`, `OFFICIAL_EXTERNAL_SOURCE_VERIFIED`, `UNKNOWN`, and `NOT_APPLICABLE`.
