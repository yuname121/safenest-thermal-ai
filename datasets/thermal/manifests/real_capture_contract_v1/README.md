# Thermal real-capture contract v1

This directory contains the machine-readable contract for pre-T-C Thermal
real-world collection. It is a provenance and capture-integrity contract, not
a model-training or model-evaluation authorization.

The five JSON Schema files describe `collection.json`, `session.json`, one
`frames.jsonl` record, one `annotations.jsonl` record, and the validator result.
The `examples/` directories are synthetic and are explicitly not real sensor
measurements. They are intentionally compact and exist only to exercise the
contract and validator.

The canonical validator is:

```text
scripts/validate_thermal_real_capture.py <collection-or-session-directory>
```

Paths in manifests are session-relative POSIX paths resolved from each session
directory. A validator result never authorizes TRAIN, VALIDATION, T-C, T-D, or
`REAL_LOCKED_TEST` use.
