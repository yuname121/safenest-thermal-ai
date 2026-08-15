# Thermal real-data acquisition contract — implementation report

## Scope and stop boundary

This change creates the pre-T-C governance package requested for real Thermal collection. It does not collect real frames, modify Thermal firmware, change ESP32 transport, change the Raspberry Pi runtime, convert real frames to the SDT canonical form, train a model, tune T-B candidates, declare a locked test, or execute T-C/T-D.

## Current evidence used

The active standalone repository's T-A evidence was read through T-A6 and the latest merged T-B evidence. The important inherited limits are:

- the selected SDT source is frame-level evidence;
- SDT does not provide verified timestamps, FPS, subject/session/recording/sequence/event IDs, or true fall-transition ground truth;
- LYING is a posture observation and a qualified SafeNest proxy only;
- the current real SDT evaluation partition is development evidence, not a pristine locked test;
- the standalone Thermal-44 backend is not a verified real hardware driver;
- physical Thermal-44 unit, native packet contract, orientation, and effective FPS remain deferred to T-C.

The team repository was inspected read-only at its main base. Its current code-level path contains an 80×62 big-endian uint16 Thermal type-2 packet, frame sequence, ESP32 uptime, and raw min/max metadata over SNST/TCP. The Pi capture helper currently stores converted arrays and numbered filenames but does not preserve the complete raw packet/timestamp/session/event contract. Those code facts are recorded as implementation context only and are not promoted to independently verified hardware facts.

## Deliverables

1. Korean teammate guide: docs/20260814_Codex_Thermal_Real_Data_Acquisition_Guide_KO_01.md
2. English normative contract: docs/20260814_Codex_Thermal_Real_Data_Acquisition_Contract_EN_01.md
3. JSON Schema contracts under datasets/thermal/manifests/real_capture_contract_v1/
4. Standalone validator: scripts/validate_thermal_real_capture.py
5. Focused tests: tests/test_thermal_real_capture_validator.py
6. Synthetic examples under datasets/thermal/manifests/real_capture_contract_v1/examples/

The examples include obvious SYNTHETIC_NOT_REAL_* payload markers and are not measurements. No bulk frames, canonical arrays, trained models, or participant captures are included.

## Contract decisions

### Raw lineage

The session contract keeps raw packet/native-frame paths separate from any future canonical or model-input path. The frame record carries frame identity, sequence/counter identity, independent clock fields, raw representation, unit status, validity, packet status, and annotation status. Unknown values remain explicit rather than guessed.

### Chronology

The validator distinguishes verified temporal provenance from order-only and insufficient evidence. It requires explicit counters/order, timestamps, a continuous session, and ordered event phases before emitting TEMPORAL_PROVENANCE_VERIFIED. Filename order alone is rejected.

### Semantics and safety

Source posture and derived SafeNest compatibility labels are separate. LYING cannot become a verified fall event. The session contract rejects an uncontrolled free-fall flag. Future transition collection is conditional on appropriate safety controls.

### Roles and leakage

Pilot, development, future-train-candidate, and locked-test roles are separate. The v1 capture validator rejects TRAIN/VALIDATION self-promotion and TRAINING_ALLOWED model access; later T-D promotion/split evidence must grant those roles explicitly. It also rejects frame-random split metadata, subject/session/event role leakage, and locked-test access violations. A passing capture result never grants model-use authorization.

### Integrity

The validator checks JSON/JSONL structure, identity, declared-vs-discovered collection inventory, frame representation requirements, raw/decoded file registration, extra files, SHA-256 coverage/mismatch, timing reversals and gaps, packet/decode status, annotation references/ranges, temporal range non-overlap and event-ID integrity, and scalar-only or preprocessed-only classification. TEMPORAL_PROVENANCE_VERIFIED additionally requires the same event_id to have validated PRE_EVENT, FALL_TRANSITION, and POST_FALL_LYING phase_ranges; frame-scoped event_phase labels alone cannot satisfy that gate.

## Synthetic smoke evidence

The static synthetic example validates with structural limitations and is classified FULL_FRAME_RAW plus TEMPORAL_ORDER_ONLY. The temporal synthetic example validates with structural limitations and is classified FULL_FRAME_RAW plus TEMPORAL_PROVENANCE_VERIFIED because it contains explicit sequence/counter values, host monotonic timestamps, a continuous session, and ordered pre/transition/post/recovery event ranges.

These results are validator wiring evidence only. They are not sensor, model, clinical, Raspberry Pi, or T-C evidence.

## First team action after merge

Perform one small, safe device-contract pilot. Keep the session continuous, preserve native full frames and all available clocks/counters, complete the manifest and annotation files, create SHA-256 checksums, and run:

~~~text
python3 scripts/validate_thermal_real_capture.py <pilot-collection-directory>
~~~

Send the collection/session manifests, raw and decoded-native folders, checksum registry, validator result, hardware/firmware/collector metadata, and all reported unknowns or packet exceptions for review. Do not begin T-C, mass collection, or T-D from this contract automatically.
