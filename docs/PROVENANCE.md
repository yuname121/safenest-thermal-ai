# Provenance and curation record

## Curated source

This repository is a focused derivative of the following team working materials, curated on 2026-08-15 (KST):

- `https://github.com/sheepmeat/test`, source commit `5000354536cf3fd2bdaea10a5db3ce2c6fe1f219`
- `https://github.com/jinsu1011/safenest-embedded-competition`, source commit `3d86bf2a7a4e527d7aba2dfabcb087201ffeb46e` (architecture reference only; no code copied from it)

Copied scope is deliberately limited to thermal-AI dataset contracts, T-A0~T-B5 preparation/training/validation scripts, tests, manifests, and associated reports. Communication, ESP firmware, integrated-node logic, risk/fusion logic, legacy interpreter code, raw data, and model binaries are excluded.

## Artifact and data exclusions

- The T-B5 selected `.tflite` binary remains on the approved external artifact store; its SHA-256 is documented in `models/README.md`.
- No raw thermal frames, identifiable participant data, deployment captures, checkpoints, or proprietary artifact binaries have been copied.
- This is not a relicensing statement. Before adding or publishing further material, confirm its authorship, data consent, and redistribution permission with the original team owners.
