# Thermal T-A1 — Safe Reader and Raw Unit Contract

Date: 2026-08-10

Phase: `T-A1`

Outcome: `PASS_WITH_LIMITATIONS`

T-A2 authorized: `YES`

## Decision

The T-A0-selected SDT real `test` split is readable through a deterministic, read-only, fail-closed Thermal reader. The reader preserves each distributed 16-bit single-channel `image_t` value and its original source label (`LYING`, `SITTING`, `STANDING`, `EMPTY_ROOM`). It performs no resize, normalization, int8 conversion, SafeNest label rewrite, or model inference.

## Verified source contract

- Official source: <https://zenodo.org/records/4124309> (`doi:10.5281/zenodo.4124309`)
- Local archive: `datasets/raw_archives/thermal_split_zips/test.zip`
- Identity: 1740348425 bytes; MD5 `d59a739f3b5ecf373c94046fb94cd94f`; SHA-256 `3a838bd70835e579ecfaa820a6c0b4cbc6ba7b76729417c73845f0c959281449`
- Linkage: 8000 Thermal images, 8000 depth images, and 8000 label rows linked 1:1 by zero-based index
- Distributed Thermal shape/dtype: `480 × 640`, one channel, PNG 16-bit, decoded `uint16`
- Native real Thermal sensor: FLIR Lepton 3.5, `120 × 160`; the source author documents bilinear upscaling to the distributed `640 × 480` geometry

## Raw unit contract

Official SDT documentation defines Kelvin centiunits: `K = raw / 100` and `°C = (raw - 27315) / 100`. The witness `raw=30000` yields `300 K` and `26.85 °C`. The 12-frame deterministic pilot observed encoded range `29277..30814` and Celsius range `19.62..34.99`.

The official “16/14-bit” wording is preserved as an unresolved encoding description. Because the official 30000 witness exceeds an unsigned 14-bit container, T-A1 does not invent a mask, ADC rule, or saturation threshold.

## Label and provenance boundary

`LYING` is an original posture label and is not rewritten to a raw fall-event label. A single frame does not establish fall onset, a transition, or a temporal event. Subject, session, sequence, event, and timestamp identifiers are absent. Each emitted record instead retains source DOI, official split, archive identity, exact member and row index, member hashes, original pose/bbox, and a hash of the preserved encoded array.

## Pilot and failure behavior

The pilot uses first/middle/last frames per original class (12 total), represents all four labels, and produces identical arrays and provenance on repeat decode. Focused fixtures establish fail-closed behavior for corrupt/unsupported images, wrong shape/bit depth/channels, missing or invalid labels, linkage failures, duplicate or unsafe members, nonfinite conversion inputs, archive identity mismatch, and constant container-extreme frames. No invalid sample is silently skipped.

## Local payload limitations

Only `test.zip` was read. `train.zip.001` through `.004` and `validation.zip` remain `LOCAL_CLOUD_PLACEHOLDER`; no hydration, extraction, reconstruction, or download occurred. The official synthetic-train/synthetic-validation/real-test split must remain intact. License use remains restricted to the stricter common denominator recorded in T-A0: non-commercial research/model development with citation; redistribution or commercial use needs separate review.

## Deferred work

Geometry/calibration belongs to T-A2; temporal sequence/event policy to T-A3; SafeNest label policy to T-A4; grouping/split policy to T-A5; full conversion to T-A6; and Thermal-44 unit, packet dtype, endianness, and hardware validation to T-C.
