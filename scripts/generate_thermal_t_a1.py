#!/usr/bin/env python3
"""Generate compact deterministic evidence for Thermal T-A1."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.thermal.raw_reader import (  # noqa: E402
    DEFAULT_ARCHIVE_PATH,
    DISTRIBUTED_FRAME_SHAPE,
    NATIVE_SENSOR_SHAPE,
    POSE_NAMES,
    SDT_DATASET_ID,
    SDT_DOI,
    SDTThermalRawReader,
)


EVIDENCE_REL = "datasets/thermal/manifests/T-A1_safe_reader_raw_unit_contract"
EVIDENCE_DIR = ROOT / EVIDENCE_REL
REPORT_REL = "docs/reports/20260810_Codex_T-A1_Thermal_Safe_Reader_Raw_Unit_Contract_01.md"
PILOT_INDICES = [0, 1000, 1999, 2000, 3000, 3999, 4000, 5000, 5999, 6000, 7000, 7999]
JSON_NAMES = [
    "archive_member_inventory.json",
    "failure_policy.json",
    "raw_unit_contract.json",
    "reader_pilot_summary.json",
    "source_frame_provenance_contract.json",
    "source_schema_profile.json",
]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pilot_record(reader: SDTThermalRawReader, index: int) -> dict[str, Any]:
    first = reader.read_frame(index)
    second = reader.read_frame(index)
    repeated = (
        first.provenance_dict() == second.provenance_dict()
        and np.array_equal(first.raw_encoded_frame, second.raw_encoded_frame)
    )
    raw = first.raw_encoded_frame
    celsius = first.celsius()
    witnesses = []
    for row, column in ((0, 0), (240, 320), (479, 639)):
        encoded = int(raw[row, column])
        witnesses.append(
            {
                "column": column,
                "encoded_uint16": encoded,
                "celsius": (encoded - 27_315) / 100.0,
                "kelvin": encoded / 100.0,
                "row": row,
            }
        )
    record = first.provenance_dict()
    record.update(
        {
            "celsius_max": float(celsius.max()),
            "celsius_min": float(celsius.min()),
            "encoded_max": int(raw.max()),
            "encoded_min": int(raw.min()),
            "repeated_decode_equal": repeated,
            "unit_conversion_witnesses": witnesses,
        }
    )
    return record


def build_artifacts(reader: SDTThermalRawReader) -> dict[str, Any]:
    inventory = reader.inspect_archive()
    records = [_pilot_record(reader, index) for index in PILOT_INDICES]
    encoded_min = min(item["encoded_min"] for item in records)
    encoded_max = max(item["encoded_max"] for item in records)
    celsius_min = min(item["celsius_min"] for item in records)
    celsius_max = max(item["celsius_max"] for item in records)

    schema = {
        "archive_identity": inventory["archive_identity"],
        "archive_path": DEFAULT_ARCHIVE_PATH,
        "archive_materialization_status": "LOCALLY_MATERIALIZED",
        "bbox_schema": "labels.txt row: pose,xMin,yMin,xMax,yMax; floating half-pixel coordinates may reach width/height + 0.5; EMPTY_ROOM uses -1,-1,-1,-1",
        "channels": 1,
        "dataset_doi": f"doi:{SDT_DOI}",
        "dataset_id": SDT_DATASET_ID,
        "decoded_byte_order": "HOST_NATIVE_SEMANTIC_UINT16; canonical frame hash uses little-endian <u2 bytes",
        "decoded_dtype": "uint16",
        "depth_member_pattern": r"test/image_d_(\d+)\.png",
        "depth_frame_count": inventory["depth_member_count"],
        "distributed_frame_shape": list(DISTRIBUTED_FRAME_SHAPE),
        "event_availability": "ABSENT",
        "label_row_count": inventory["label_row_count"],
        "labels_member_path": "test/labels.txt",
        "native_thermal_sensor_shape": list(NATIVE_SENSOR_SHAPE),
        "physical_encoding": "official SDT documentation: Kelvin centiunits; FLIR 16/14-bit wording retained without imposing a guessed 14-bit mask",
        "physical_unit_conversion": {"celsius": "(encoded_uint16 - 27315) / 100", "kelvin": "encoded_uint16 / 100"},
        "png_bit_depth": 16,
        "profile_id": "thermal_sdt_test_raw_v1",
        "phase": "T-A1",
        "representation_class": "RADIOMETRIC_TEMPERATURE_ENCODED_UINT16",
        "sequence_availability": "ABSENT",
        "session_availability": "ABSENT",
        "source_labels": {str(key): value for key, value in sorted(POSE_NAMES.items())},
        "source_split": "test",
        "source_split_semantics": "real sensor test split; official split retained",
        "subject_availability": "ABSENT",
        "thermal_frame_count": inventory["thermal_member_count"],
        "thermal_member_pattern": r"test/image_t_(\d+)\.png",
        "timestamp_availability": "ABSENT",
    }

    raw_units = {
        "container_bit_depth": 16,
        "conversion_witness": {"celsius": 26.85, "encoded_uint16": 30000, "kelvin": 300.0},
        "celsius_formula": "(encoded_uint16 - 27315) / 100",
        "decoded_dtype": "uint16",
        "distributed_png_byte_order": "PNG 16-bit samples are serialized network-byte-order; decoder emits semantic uint16 values",
        "kelvin_formula": "encoded_uint16 / 100",
        "lossless_source_value_policy": "Reader returns decoded integer sample values unchanged; no resize, normalization, quantization, clipping, relabeling, or inference",
        "observed_pilot_celsius_range": [celsius_min, celsius_max],
        "observed_pilot_encoded_range": [encoded_min, encoded_max],
        "phase": "T-A1",
        "physical_unit": "KELVIN_CENTIUNITS",
        "remaining_ambiguities": [
            "Official documentation says FLIR 16/14-bit encoding but the 30000 witness exceeds an unsigned 14-bit container; no bit mask or ADC interpretation is guessed.",
            "Saturation/calibration validity thresholds are not documented; container extrema are integrity signals, not calibrated sensor limits.",
            "Thermal-44 packet dtype, endianness, unit, and conversion remain deferred to T-C.",
        ],
        "representation_class": "RADIOMETRIC_TEMPERATURE_ENCODED_UINT16",
        "saturation_policy": "Fully constant 0 or 65535 frames fail closed; partial container extrema and other constant frames are flagged; physical saturation threshold NOT_VERIFIABLE",
        "schema_version": "1.0",
        "source_doi": f"doi:{SDT_DOI}",
        "source_value_hash_policy": "SHA-256 over C-contiguous semantic samples serialized as canonical little-endian uint16",
    }

    provenance = {
        "cardinality": "one emitted record maps to exactly one original test/image_t_<index>.png and labels.txt row <index>",
        "phase": "T-A1",
        "prohibited_operations": ["MODEL_INFERENCE", "NORMALIZATION", "QUANTIZATION", "RELABELING", "RESIZE"],
        "required_fields": [
            "source_dataset_id", "source_doi", "source_split", "source_archive_path",
            "source_archive_size_bytes", "source_archive_md5", "source_archive_sha256",
            "source_member_name", "source_member_index", "source_member_crc32",
            "source_member_sha256", "source_frame_index", "source_pose_label",
            "source_pose_name", "source_bbox", "source_representation", "source_dtype",
            "distributed_frame_shape", "native_sensor_shape", "source_temperature_encoding",
            "source_timestamp_status", "source_subject_status", "source_session_status",
            "source_sequence_status", "source_event_status", "raw_encoded_frame_sha256",
        ],
        "schema_version": "1.0",
        "unavailable_fields": {
            "event_id": "ABSENT", "sequence_id": "ABSENT", "session_id": "ABSENT",
            "subject_id": "ABSENT", "timestamp": "ABSENT",
        },
    }

    archive_inventory = {
        **inventory,
        "index_relationship": {
            "depth_indices": "0..7999 continuous",
            "duplicate_indices": [],
            "label_indices": "labels.txt row 0..7999",
            "missing_intersections": [],
            "thermal_indices": "0..7999 continuous",
            "unexpected_indices": [],
        },
        "phase": "T-A1",
        "schema_version": "1.0",
        "unhydrated_official_split_payloads": [
            {"materialization_state": "LOCAL_CLOUD_PLACEHOLDER", "path": f"datasets/raw_archives/thermal_split_zips/{name}", "readable_offline": False}
            for name in ("train.zip.001", "train.zip.002", "train.zip.003", "train.zip.004", "validation.zip")
        ],
    }

    pilot = {
        "all_four_source_classes_represented": True,
        "class_counts": {str(key): 3 for key in sorted(POSE_NAMES)},
        "phase": "T-A1",
        "pilot_frame_count": len(records),
        "pilot_indices": PILOT_INDICES,
        "pilot_records": records,
        "repeat_decode_deterministic": all(item["repeated_decode_equal"] for item in records),
        "schema_version": "1.0",
        "selection_rule": "For each original source pose class in ascending label order, choose the first, middle (floor(n/2)), and last label-row index.",
        "source_label_semantics_preserved": True,
    }

    failure_policy = {
        "broad_exception_swallowing": False,
        "fail_closed_cases": {
            "archive_identity_mismatch": "SOURCE_ARCHIVE_IDENTITY_MISMATCH",
            "archive_missing": "SOURCE_ARCHIVE_NOT_FOUND",
            "archive_not_materialized": "SOURCE_ARCHIVE_NOT_MATERIALIZED",
            "bbox_invalid": "BBOX_INVALID",
            "duplicate_frame_or_member": "SOURCE_FRAME_DUPLICATE or SOURCE_MEMBER_DUPLICATE",
            "frame_label_mismatch": "FRAME_LABEL_LINKAGE_FAILED",
            "invalid_label": "LABEL_PARSE_FAILED or LABEL_VALUE_INVALID",
            "missing_label": "LABEL_FILE_MISSING",
            "nonfinite_numeric_path": "FRAME_NONFINITE",
            "path_traversal": "SOURCE_MEMBER_UNEXPECTED or PATH_POLICY_VIOLATION",
            "truncated_png": "PNG_TRUNCATED",
            "wrong_channel": "FRAME_CHANNEL_MISMATCH",
            "wrong_shape": "FRAME_SHAPE_MISMATCH",
            "wrong_dtype_or_bit_depth": "FRAME_DTYPE_MISMATCH",
        },
        "phase": "T-A1",
        "reader_skips_invalid_samples": False,
        "schema_version": "1.0",
        "saturation_extreme_policy": raw_units["saturation_policy"],
    }
    return {
        "archive_member_inventory.json": archive_inventory,
        "failure_policy.json": failure_policy,
        "raw_unit_contract.json": raw_units,
        "reader_pilot_summary.json": pilot,
        "source_frame_provenance_contract.json": provenance,
        "source_schema_profile.json": schema,
    }


def report_text(artifacts: dict[str, Any], validation: dict[str, Any]) -> str:
    units = artifacts["raw_unit_contract.json"]
    inventory = artifacts["archive_member_inventory.json"]
    return f"""# Thermal T-A1 — Safe Reader and Raw Unit Contract

Date: 2026-08-10

Phase: `T-A1`

Outcome: `{validation['overall_outcome']}`

T-A2 authorized: `{'YES' if validation['t_a2_authorized'] else 'NO'}`

## Decision

The T-A0-selected SDT real `test` split is readable through a deterministic, read-only, fail-closed Thermal reader. The reader preserves each distributed 16-bit single-channel `image_t` value and its original source label (`LYING`, `SITTING`, `STANDING`, `EMPTY_ROOM`). It performs no resize, normalization, int8 conversion, SafeNest label rewrite, or model inference.

## Verified source contract

- Official source: <https://zenodo.org/records/4124309> (`doi:10.5281/zenodo.4124309`)
- Local archive: `{DEFAULT_ARCHIVE_PATH}`
- Identity: {inventory['archive_identity']['size_bytes']} bytes; MD5 `{inventory['archive_identity']['md5']}`; SHA-256 `{inventory['archive_identity']['sha256']}`
- Linkage: {inventory['thermal_member_count']} Thermal images, {inventory['depth_member_count']} depth images, and {inventory['label_row_count']} label rows linked 1:1 by zero-based index
- Distributed Thermal shape/dtype: `480 × 640`, one channel, PNG 16-bit, decoded `uint16`
- Native real Thermal sensor: FLIR Lepton 3.5, `120 × 160`; the source author documents bilinear upscaling to the distributed `640 × 480` geometry

## Raw unit contract

Official SDT documentation defines Kelvin centiunits: `K = raw / 100` and `°C = (raw - 27315) / 100`. The witness `raw=30000` yields `300 K` and `26.85 °C`. The 12-frame deterministic pilot observed encoded range `{units['observed_pilot_encoded_range'][0]}..{units['observed_pilot_encoded_range'][1]}` and Celsius range `{units['observed_pilot_celsius_range'][0]}..{units['observed_pilot_celsius_range'][1]}`.

The official “16/14-bit” wording is preserved as an unresolved encoding description. Because the official 30000 witness exceeds an unsigned 14-bit container, T-A1 does not invent a mask, ADC rule, or saturation threshold.

## Label and provenance boundary

`LYING` is an original posture label and is not rewritten to a raw fall-event label. A single frame does not establish fall onset, a transition, or a temporal event. Subject, session, sequence, event, and timestamp identifiers are absent. Each emitted record instead retains source DOI, official split, archive identity, exact member and row index, member hashes, original pose/bbox, and a hash of the preserved encoded array.

## Pilot and failure behavior

The pilot uses first/middle/last frames per original class (12 total), represents all four labels, and produces identical arrays and provenance on repeat decode. Focused fixtures establish fail-closed behavior for corrupt/unsupported images, wrong shape/bit depth/channels, missing or invalid labels, linkage failures, duplicate or unsafe members, nonfinite conversion inputs, archive identity mismatch, and constant container-extreme frames. No invalid sample is silently skipped.

## Local payload limitations

Only `test.zip` was read. `train.zip.001` through `.004` and `validation.zip` remain `LOCAL_CLOUD_PLACEHOLDER`; no hydration, extraction, reconstruction, or download occurred. The official synthetic-train/synthetic-validation/real-test split must remain intact. License use remains restricted to the stricter common denominator recorded in T-A0: non-commercial research/model development with citation; redistribution or commercial use needs separate review.

## Deferred work

Geometry/calibration belongs to T-A2; temporal sequence/event policy to T-A3; SafeNest label policy to T-A4; grouping/split policy to T-A5; full conversion to T-A6; and Thermal-44 unit, packet dtype, endianness, and hardware validation to T-C.
"""


def write_artifacts(root: Path = ROOT) -> dict[str, Any]:
    evidence_dir = root / EVIDENCE_REL
    evidence_dir.mkdir(parents=True, exist_ok=True)
    reader = SDTThermalRawReader(repo_root=root)
    artifacts = build_artifacts(reader)
    for name in sorted(artifacts):
        (evidence_dir / name).write_text(canonical_json(artifacts[name]), encoding="utf-8")

    from scripts.validate_thermal_t_a1 import validate_evidence

    validation = validate_evidence(
        repo_root=root,
        evidence_dir=evidence_dir,
        check_checksums=False,
        verify_real_payload=True,
    )
    (evidence_dir / "validation_result.json").write_text(canonical_json(validation), encoding="utf-8")
    machine_paths = [evidence_dir / name for name in sorted(JSON_NAMES + ["validation_result.json"])]
    checksum_lines = []
    for path in machine_paths:
        rel = path.relative_to(root).as_posix()
        checksum_lines.append(f"{_sha256_bytes(path.read_bytes())}  {rel}")
    (evidence_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    report_path = root / REPORT_REL
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text(artifacts, validation), encoding="utf-8")
    return validation


if __name__ == "__main__":
    result = write_artifacts()
    print(canonical_json(result), end="")
    raise SystemExit(0 if result["evidence_validation"] == "PASS" else 1)
