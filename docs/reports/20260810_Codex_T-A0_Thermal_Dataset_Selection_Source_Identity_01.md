# SafeNest Thermal T-A0 Dataset Selection and Source Identity

- Phase: `T-A0`
- Audit date: `2026-08-10`
- Overall outcome: `PASS_WITH_LIMITATIONS`
- Selection status: `LOCAL_DATASET_SELECTED_WITH_LIMITATIONS`
- T-A1 authorized: `YES`

## Decision

The local SDT source (`doi:10.5281/zenodo.4124309`) is selected with explicit limitations as the T-A1 source basis. The local payload exists and is intentionally Git-ignored; absence from Git is not absence from the owner workspace.

SDT source label 0 remains **lying**. SafeNest derives it as `HUMAN_FALL` only in the narrower sense of **post-fall lying-posture evidence**: a single frame does not establish that a fall event occurred. Persistence and corroboration from other sensors are responsible for escalating suspicion. This matches the intended sensor-fusion architecture while preserving the original source semantics.

SDT has no subject/session/sequence/event identifiers, so T-A1 must preserve its official synthetic-train, synthetic-validation and real-test split exactly. Subject-wise and event-level generalization are `NOT_VERIFIABLE`, and frame-random resplitting is prohibited. Family A and the additional human/not-human tree remain unselected because their source provenance is insufficient. The processed NPZ remains legacy mixed-source evidence and is not canonical.

## Candidate comparison

| Candidate | Representation | Label/group evidence | Access/license | T-A0 status |
|---|---|---|---|---|
| Local Family A | RGB thermal colorized rendering | Unknown labels and grouping | Identity/license unknown | `REJECTED_PROVENANCE` |
| Local SDT | 16-bit thermal Kelvin encoding + depth; synthetic train/validation, real test | Lying as derived post-fall posture proxy; official split is the accepted grouping limitation | Non-commercial research restriction, citation/attribution; official metadata conflict retained | `SELECTED` |
| Local human/not-human tree | RGB/RGBA thermal screenshots/exports | Presence polygons only | Identity/license unknown | `REJECTED_PROVENANCE` |
| eHomeSeniors | Numeric thermal temperature and raw fields | Six subjects and staged fall types; no documented normal sequences or explicit repeated-event boundaries | Open supplement; dataset-specific terms need review | `NEEDS_MANUAL_REVIEW` |
| MUVIM | Encoded thermal video plus other modalities | Strong publication-level subject/ADL/fall structure | Author request; terms unverified | `ACCESS_BLOCKED` |
| Thermal Fall 66 | Thermal representation not inspectable | Publication claims 66 participants | Author request; terms unverified | `ACCESS_BLOCKED` |

## Local inventory

- Family A: 6,748 PNG, 224,906,370 logical bytes; 3,723 readable RGB 230×226 files and 3,025 dataless placeholders.
- SDT: `test.zip` is materialized and byte-identical to official MD5; it contains 8,000 `image_t`, 8,000 `image_d`, and 8,000 five-field labels. Four train parts and validation are dataless placeholders. No large hydration was attempted.
- Processed NPZ: 330,777,971 bytes, SHA-256 `3d6ad1eb2ed0438f0faaf83abed8b6e2c175074dfa031dcb4a5739c45984d06e`; only `X` `(54218,62,80)` float32 and `y` `(54218,)` int32 survive.
- Additional tree: 410 images and 410 JSON annotations; all JSON and 213 images are readable, while 197 images are dataless placeholders.

## Processed NPZ lineage

Selected SDT test samples exactly match NPZ rows 40,000–47,999 under the current preparation transform. Code order, segment counts and local spot matches support a partial reconstruction of 32,000 SDT train + 8,000 SDT validation + 8,000 SDT test + 20 additional-tree images + 6,198 Family A images. Exact per-row source IDs, generation commit, skip reasons and original grouping are absent, so the artifact remains `PROCESSED_LINEAGE_PARTIALLY_RECONSTRUCTED` and `NOT_T_A_CANONICAL`.

`thermal_prep.py` merges original train/validation/test sources and silently swallows broad exceptions. `thermal_train.py` then defines a seeded frame-level 80:20 permutation. This is confirmed code risk; execution against the current TFLite artifact is not independently proven by an immutable training record.

## Contract boundaries preserved

Per-frame min-max normalization discards absolute Celsius context. Thermal-44 physical unit, dtype, endianness, raw-count conversion, invalid pixels, 9,920-versus-10,080 bytes, real driver and hardware/Pi evidence remain `NOT_VERIFIABLE` and deferred to `T-C`. No T-A1 split, tensor regeneration, training or model-performance claim was created.

## T-A1 gate

`T-A1 authorized: YES`, with these mandatory conditions: use SDT under the stricter non-commercial research and attribution terms; obtain owner authorization before hydrating multi-GB placeholders; read the original archives rather than the mixed legacy NPZ; preserve the official train/validation/test split; retain the source labels and derived-proxy mapping in row provenance; and do not claim temporal fall-event, subject-generalization, Thermal-44 hardware or model-performance validation from this T-A0 decision.
