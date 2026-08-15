# SafeNest Thermal Real-World Acquisition Contract v1

Status: pre-T-C collection contract
Scope: metadata, raw-frame preservation, chronology, annotation provenance, integrity, and role governance
Out of scope: data collection execution, firmware changes, runtime changes, model training, T-C execution, T-D execution, and performance claims

## 1. Contract intent

The collection unit MUST be a traceable session, not an unconnected pile of images. Every future reviewer MUST be able to reconstruct the lineage:

~~~
physical native frame → capture identity/timing → annotation provenance
                     → verified T-C decoder → canonical representation
                     → separately authorized model input
~~~

The collection validator reports capture structure only. A passing result MUST NOT be interpreted as TRAINING_AUTHORIZED, T_C_COMPLETE, T_D_AUTHORIZED, or REAL_LOCKED_TEST_VALID.

## 2. Current implementation context

Read-only inspection of the current team repository found a code-level path in which an ESP32 Thermal HAT/MI48xx implementation reads an 80×62 frame, sends a big-endian uint16 type-2 SNST/TCP packet, and includes a frame sequence, ESP32 uptime, and raw min/max metadata. The Pi receiver currently visualizes the frame; the older capture helper converts raw words with raw / 100.0 and saves numbered arrays.

These are implementation claims, not an independently verified physical sensor contract. The exact sensor identity, native encoding, physical unit, orientation, calibration, effective FPS, packet-loss behavior, and end-to-end clock relationship remain UNKNOWN or NOT_VERIFIED until T-C evidence is created. The standalone Thermal-44 adapter also has no installed real hardware backend. The collection contract MUST preserve evidence rather than promote these code claims to verified facts.

## 3. Three representation layers

### 3.1 RAW

The collector MUST preserve the closest available representation of the physical measurement before resize, crop, rotation, normalization, z-score, quantization, colorization, frame deletion, or label mapping. Where possible it MUST preserve both:

1. raw packet/frame bytes; and
2. a decoded native numerical frame.

The collector MUST NOT retain only screenshots, PNG/JPEG visualizations, normalized arrays, resized 62×80 tensors, model inputs, model predictions, or a scalar such as thermal_max_c.

### 3.2 CANONICAL

Canonical representation MUST be produced only after the real physical contract is verified. The SDT-specific Kelvin centiunit conversion and geometry profile MUST NOT be applied blindly to a real capture. A canonical artifact MUST retain a reference to its immutable raw evidence.

### 3.3 MODEL INPUT

P1 global z-score, quantization, or any other model-ready representation MUST be regenerable from raw/canonical evidence. Model input files MUST NOT replace raw evidence.

## 4. Identity and privacy

The collection MUST contain pseudonymous identifiers:

~~~
collection_id
subject_id
session_id
recording_id, when distinct
sequence_id, when distinct
event_id, when applicable
frame_id
sequence_index
~~~

IDs MUST be deterministic within the collection, unique at their declared scope, and free of names, student numbers, or other direct identifiers. frame_id MUST NOT be reconstructed later from ZIP order, filesystem order, or filesystem timestamps.

UNKNOWN, NOT_VERIFIED, and NOT_APPLICABLE MUST remain explicit values or explicit status fields. A missing fact MUST NOT be filled with a plausible guess.

## 5. Session manifest requirements

Each session MUST have a session.json containing the fields represented by datasets/thermal/manifests/real_capture_contract_v1/session.schema.json. At minimum it MUST record:

- collection_id, subject_id, session_id, and recording_id;
- capture start/end wall time and timezone, or explicit null/status;
- sensor vendor/model/hardware revision/pseudonymous device ID;
- firmware, collector software, and collector commit;
- native width/height/dtype, raw encoding, raw unit claim, and unit verification status;
- configured FPS and whether effective FPS is verified;
- orientation, mount height/angle, and sensor-to-subject distance, or explicit unknowns;
- transport path, protocol, full-frame availability, and scalar-only status;
- sensor/device/host timestamp and clock-domain coverage;
- environment and scenario metadata;
- storage paths for frames.jsonl, annotations.jsonl, raw/, decoded_native/, and checksums;
- expected and received frame counts;
- collection/model role governance.

All persisted paths MUST be session-relative POSIX paths resolved from the session directory. Absolute paths, file:// URIs, home-relative paths, backslashes, and parent traversal MUST be rejected.

## 6. Frame record requirements

Each line of frames.jsonl MUST conform to datasets/thermal/manifests/real_capture_contract_v1/frame.schema.json. Each record MUST preserve, where available:

- frame/collection/subject/session/recording/sequence/event identity;
- sequence index and sensor frame counter with independent status fields;
- sensor timestamp and clock domain/unit;
- device monotonic timestamp;
- host receive monotonic timestamp;
- host wall-clock time and timezone;
- raw and decoded-native relative paths;
- byte count, native shape, native dtype, raw encoding, and raw-unit claim;
- CRC/packet-loss status;
- validity status, error code, and exclusion reason;
- annotation status and per-file hashes when available.

The file-reference matrix is explicit: `RAW_PACKET_AND_NATIVE` requires both
`raw_file` and `decoded_native_file`; `RAW_PACKET_ONLY` requires only
`raw_file`; `DECODED_NATIVE_ONLY` requires only `decoded_native_file`; and
`SCALAR_ONLY`, `PREPROCESSED_ONLY`, and `SCREENSHOT_ONLY` require `raw_file`
and forbid `decoded_native_file`. A valid `DECODED_NATIVE_ONLY` frame therefore
MUST NOT be rejected merely because no raw packet file exists.

Invalid, corrupt, late, duplicate, partial, missing, or decode-failed frames MUST remain represented as evidence. The collector MUST NOT silently delete a bad frame or renumber the following sequence.

## 7. Timing and transport

The collector SHOULD preserve all available clocks rather than collapsing them into one unlabeled time. A wall-clock timestamp and a monotonic timestamp SHOULD both be retained on the host. Every numeric timestamp MUST identify its unit and clock domain.

The validator MUST report sequence monotonicity, duplicate counters, sequence gaps, timestamp reversals, negative intervals, large timing gaps, configured FPS, measured effective FPS, expected/received counts, packet loss, CRC status, and decode failures. A configured FPS MUST NOT be presented as measured FPS.

An observed counter gap is evidence of a possible dropped frame or transport condition; it MUST be reported. It MUST NOT be silently repaired or treated as proof of a particular physical cause without T-C evidence.

## 8. Annotation contract

The original observation MUST be separate from any SafeNest compatibility label. annotations.jsonl MUST preserve annotation ID, annotator code, annotation time, method, confidence, notes, revision, and correction history.

The source posture vocabulary MAY include:

~~~
EMPTY, STANDING, SITTING, LYING, UNKNOWN, NOT_ANNOTATED
~~~

LYING MUST mean an observed lying posture only. It MUST NOT be represented as a verified temporal fall event. A derived HUMAN_FALL label, if ever used, MUST be marked as an explicitly qualified DERIVED_POSTURE_PROXY and MUST NOT be called event ground truth.

For a genuinely authorized, safe transition capture, the annotation MUST preserve event_id, ordered frame or timestamp ranges, and phase evidence such as PRE_EVENT, FALL_TRANSITION, POST_FALL_LYING, and RECOVERY. To emit TEMPORAL_PROVENANCE_VERIFIED, the validator requires validated, non-overlapping phase_ranges for the same event_id; frame-scoped event_phase labels alone are insufficient. Uncontrolled or unprotected free-fall experiments MUST NOT be improvised or required.

## 9. Storage and immutability

The semantic layout SHOULD be:

~~~
<collection_id>/
├── collection.json
└── subjects/<subject_id>/sessions/<session_id>/
    ├── session.json
    ├── raw/
    ├── decoded_native/
    ├── frames.jsonl
    ├── annotations.jsonl
    └── checksums.sha256
~~~

After finalization, raw/ and decoded_native/ artifacts MUST be immutable. Corrections MUST be represented as annotation revisions or derived metadata, not by overwriting capture bytes. SHA-256 MUST cover the session manifest, frames manifest, annotations manifest, and every registered raw or decoded-native artifact. Extra unregistered files, missing references, and checksum mismatches MUST fail validation.

## 10. Collection roles and split governance

The contract recognizes these roles:

| Role | Meaning | Model authority |
|---|---|---|
| DEVICE_CONTRACT_PILOT | Verify capture wiring, file integrity, timing, and annotation linkage | Not training or pristine test authorization |
| REAL_DEVELOPMENT | Development/domain analysis | Not a locked test |
| FUTURE_TRAIN_CANDIDATE | Candidate for a later authorized T-D decision | Not automatically training |
| REAL_LOCKED_TEST | New, untouched subject/session/event after contract/protocol freeze | Must not be fitted, trained, tuned, calibrated, or repeatedly debugged |

Frame-random train/test splitting MUST be rejected. Assignment MUST use the strongest available group, preferably subject, then session, then event. The same subject, session, or event MUST NOT appear in conflicting roles. The split MUST be frozen before model inspection.

Pilot data MAY be inspected repeatedly and therefore MUST NOT later be called pristine REAL_LOCKED_TEST. TRAIN/VALIDATION promotion and training authority MUST be issued only by a later T-D promotion/split artifact, never by this capture contract.

## 11. Validator result and temporal classification

Run:

~~~text
python3 scripts/validate_thermal_real_capture.py <collection-or-session-path>
~~~

The validator MUST classify temporal evidence from actual fields, never from filenames alone:

- TEMPORAL_PROVENANCE_VERIFIED: explicit order/counter, adequate timestamps, continuous session, ordered event phases, and no unaccounted sequence gaps;
- TEMPORAL_ORDER_ONLY: explicit ordered records exist, but complete event/time provenance is not established;
- TEMPORAL_PROVENANCE_INSUFFICIENT: chronology or event provenance cannot be established from the captured evidence.

The result MUST separately report:

~~~text
CAPTURE_STRUCTURE_VALID
CAPTURE_STRUCTURE_VALID_WITH_LIMITATIONS
CAPTURE_INVALID
~~~

It MUST never emit training authorization. Scalar-only evidence MUST be classified as limited runtime/transport evidence, not full-frame AI evidence. Preprocessed-only collections MUST be rejected as sufficient raw evidence.

## 12. T-C and T-D handoff

T-C MUST later determine the actual sensor raw encoding, physical unit, geometry, orientation, calibration, effective FPS, packet integrity, device-to-Pi behavior, and real canonical conversion. This contract preserves the evidence needed to answer those questions; it does not answer them.

T-D MAY later decide whether audited sessions become training, validation, hard-negative, domain-adaptation, or temporal data. That decision requires provenance audit, physical-contract validation, annotation review, grouping, leakage analysis, and a frozen split. Raw collection alone MUST NOT authorize training.

## 13. Prohibited shortcuts

The following are contract violations:

- scalar-only thermal_max_c capture presented as full-frame evidence;
- screenshot-only or PNG/JPEG-only capture;
- preprocessing before raw preservation;
- silent deletion or renumbering of bad frames;
- filename order or filesystem mtime used as timestamps;
- session/event IDs omitted from a temporal capture;
- LYING called a verified fall event;
- pilot promoted to pristine locked test;
- locked test used for training, calibration, threshold tuning, or INT8 representative data;
- frame-random split;
- uncontrolled hazardous fall experiments;
- blind reuse of the SDT conversion for a real sensor capture.

## 14. Contract files

Machine-readable schemas and synthetic, non-measurement examples are under:

~~~text
datasets/thermal/manifests/real_capture_contract_v1/
~~~

The validator implementation and focused tests are:

~~~text
scripts/validate_thermal_real_capture.py
tests/test_thermal_real_capture_validator.py
~~~
