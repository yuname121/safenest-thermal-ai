# Thermal T-A2 — Geometry, Calibration, and Canonical Frame Contract

Date: 2026-08-10

Phase: `T-A2`

Outcome: `PASS_WITH_LIMITATIONS`

T-A3 authorized: `YES`

## Decision

The selected software canonical profile is `G1_FIXED_ASPECT_CROP_BILINEAR`. It was derived from all nine candidate metric records using policy `THERMAL_T_A2_GEOMETRY_SELECTION_POLICY_002` v2.0; no profile ID is hardcoded as the winner. The canonical physical unit is Celsius and the canonical dtype is float32. No model score, model inference, normalization, or SafeNest label remapping was used.

## Geometry boundary

The verified SDT distributed frame is `(480,640)` and already contains the authors' bilinear enlargement from the FLIR Lepton 3.5 native `(120,160)` grid. T-A2 does not reverse that operation or claim a restored native frame. Thermal-44 physical orientation and packet ordering remain `UNVERIFIED / DEFERRED_T_C`.

The predeclared candidate set contains 3 fixed geometry policies (direct stretch, fixed aspect crop, masked aspect pad) crossed with nearest, bilinear, and exact area interpolation. The policy first applies mandatory semantic gates, then the declared FOV/bbox/padding admissibility thresholds, then lexicographically ranks anisotropy, padding, interpolation preference, Celsius-statistic distortion, round-trip diagnostic MAE, and finally candidate ID.

The selected geometry crop is `[10, 0, 630, 480]` and retains `96.875%` of source area. Candidate evidence records each gate, admissibility result, rejection reason, rank, tie group, and final status. Source-frame bbox overflow is clipped before measuring incremental candidate-crop damage: `5` source bboxes were outside the distributed frame, `2` received additional crop intersection, and total additional crop loss was `406.500000` source-pixel².

The selected interpolation is `BILINEAR` with coordinate mapping `HALF_PIXEL_CENTER`, edge handling `EDGE_CLAMPING`, and `NO_EXPLICIT_ANTIALIAS_PREFILTER`. Coordinate mapping is not described as antialiasing.

## Physical calibration

SDT Celsius conversion remains `(encoded_uint16 - 27315) / 100`. Ambient/reference compensation and hardware-specific calibration are not applied because no verified parameter source exists. Float32 was selected after comparison with float64 reference conversion: maximum measured conversion error `1.83105469e-06 °C`, below the source `0.01 °C` encoded resolution.

## Invalid pixels and provenance

T-A1's no-sentinel policy is inherited. NaN/Inf or a supplied partial invalid source mask fails closed; no neighbor, mean-temperature, zero, ambient, or other synthetic value is inserted. The selected crop has an all-true validity mask. Every pilot record retains the original encoded source hash and exact source member/frame index separately from the canonical frame hash.

## Pilot and visual check

The bounded real-data pilot uses 12 evenly spaced sorted source indices per original pose class (48 total), with all four classes represented. Repeated canonicalization is byte-stable. Coordinate traces and asymmetric synthetic fixtures show row/column order preserved with no transpose, rotation, or flip. The tracked visual is a colorized human diagnostic only and is not radiometric model input.

## Deferred boundaries

T-A2 does not create temporal windows, SafeNest fall labels, grouping/splits, full canonical conversion, model comparisons, or Thermal-44 hardware claims. Train/validation split placeholders remain unhydrated.
