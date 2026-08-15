#!/usr/bin/env python3
"""Standalone validator for Thermal T-A1 compact evidence and real pilot."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.thermal.raw_reader import (  # noqa: E402
    DEFAULT_ARCHIVE_MD5,
    DEFAULT_ARCHIVE_PATH,
    DEFAULT_ARCHIVE_SHA256,
    DEFAULT_ARCHIVE_SIZE,
    SDT_DOI,
    SDTThermalRawReader,
    encoded_to_celsius,
    encoded_to_kelvin,
)


DEFAULT_EVIDENCE = ROOT / "datasets/thermal/manifests/T-A1_safe_reader_raw_unit_contract"
T_A0_EVIDENCE_REL = "datasets/thermal/manifests/T-A0_source_identity"
REQUIRED_JSON = [
    "archive_member_inventory.json",
    "failure_policy.json",
    "raw_unit_contract.json",
    "reader_pilot_summary.json",
    "source_frame_provenance_contract.json",
    "source_schema_profile.json",
]
PILOT_INDICES = [0, 1000, 1999, 2000, 3000, 3999, 4000, 5000, 5999, 6000, 7000, 7999]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _error(errors: list[dict[str, str]], code: str, location: str, message: str) -> None:
    errors.append({"code": code, "location": location, "message": message})


def _warning(warnings: list[dict[str, str]], code: str, location: str, message: str) -> None:
    warnings.append({"code": code, "location": location, "message": message})


def _walk(value: Any, location: str = "$") -> Iterable[tuple[str, Any]]:
    yield location, value
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _walk(value[key], f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{location}[{index}]")


def _is_portable_path(value: str) -> bool:
    if value.startswith(("/", "~/", "file://")) or "\\" in value or re.match(r"^[A-Za-z]:", value):
        return False
    pure = PurePosixPath(value)
    return not pure.is_absolute() and ".." not in pure.parts


def _validate_static_reader(repo_root: Path, errors: list[dict[str, str]]) -> None:
    path = repo_root / "datasets/thermal/raw_reader.py"
    if not path.is_file():
        _error(errors, "READER_MISSING", path.as_posix(), "T-A1 safe reader is missing.")
        return
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        _error(errors, "READER_SYNTAX_ERROR", "datasets/thermal/raw_reader.py", str(exc))
        return
    forbidden_text = {
        "processed_thermal_80x62.npz": "LEGACY_NPZ_CONSUMPTION",
        "ThermalInterpreter": "MODEL_INFERENCE_REFERENCE",
        "extractall(": "ARCHIVE_EXTRACTION",
        ".resize(": "RESIZE_OPERATION",
    }
    for needle, code in forbidden_text.items():
        if needle in source:
            _error(errors, code, "datasets/thermal/raw_reader.py", f"Forbidden reader text: {needle}")
    lowered = source.lower()
    for needle, code in (("min-max", "NORMALIZATION_OPERATION"), ("int8", "INT8_CONVERSION"), ("tflite", "MODEL_INFERENCE_REFERENCE")):
        if needle in lowered:
            _error(errors, code, "datasets/thermal/raw_reader.py", f"Forbidden reader semantic: {needle}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"resize", "extractall"}:
                _error(errors, "FORBIDDEN_READER_CALL", f"datasets/thermal/raw_reader.py:{node.lineno}", node.func.attr)
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            _error(errors, "BARE_EXCEPTION_HANDLER", f"datasets/thermal/raw_reader.py:{node.lineno}", "Bare exception handlers are forbidden.")


def _validate_checksums(
    repo_root: Path,
    evidence_dir: Path,
    required_paths: list[Path],
    errors: list[dict[str, str]],
) -> None:
    path = evidence_dir / "checksums.sha256"
    if not path.is_file():
        _error(errors, "CHECKSUM_REGISTRY_MISSING", "checksums.sha256", "Required checksum registry missing.")
        return
    entries: dict[str, str] = {}
    previous = ""
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", line)
        if not match:
            _error(errors, "CHECKSUM_LINE_INVALID", f"checksums.sha256:{line_number}", line)
            continue
        digest, rel = match.groups()
        if rel <= previous:
            _error(errors, "CHECKSUM_ORDER_NONDETERMINISTIC", f"checksums.sha256:{line_number}", rel)
        previous = rel
        if not _is_portable_path(rel):
            _error(errors, "NONPORTABLE_PATH", f"checksums.sha256:{line_number}", rel)
        entries[rel] = digest
    for artifact in required_paths:
        rel = artifact.relative_to(repo_root).as_posix()
        if rel not in entries:
            _error(errors, "CHECKSUM_COVERAGE_MISSING", rel, "Machine-readable artifact is not covered.")
        elif hashlib.sha256(artifact.read_bytes()).hexdigest() != entries[rel]:
            _error(errors, "CHECKSUM_MISMATCH", rel, "Artifact checksum differs.")


def validate_evidence(
    *,
    repo_root: Path,
    evidence_dir: Path,
    check_checksums: bool = True,
    verify_real_payload: bool = True,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    documents: dict[str, Any] = {}
    paths: list[Path] = []

    # Predecessor gate is independently rerun.
    try:
        from scripts.validate_thermal_t_a0 import validate_evidence as validate_t_a0

        t_a0_dir = repo_root / T_A0_EVIDENCE_REL
        predecessor = validate_t_a0(t_a0_dir, repo_root)
        if predecessor.get("evidence_validation") != "PASS":
            _error(errors, "T_A0_VALIDATION_FAILED", T_A0_EVIDENCE_REL, canonical_json(predecessor).strip())
    except Exception as exc:
        _error(errors, "T_A0_VALIDATOR_FAILED", T_A0_EVIDENCE_REL, str(exc))
        predecessor = {"evidence_validation": "FAIL", "overall_outcome": "NOT_VERIFIABLE"}

    selected_path = repo_root / T_A0_EVIDENCE_REL / "selected_source_identity.json"
    try:
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        if selected.get("selected_candidate_id") != "local_sdt_zenodo_4124309":
            _error(errors, "T_A0_SOURCE_CHANGED", selected_path.as_posix(), "Selected source is not SDT.")
        if selected.get("stable_identifier") != f"doi:{SDT_DOI}":
            _error(errors, "T_A0_DOI_CHANGED", selected_path.as_posix(), "Selected DOI changed.")
        if selected.get("t_a1_authorized") is not True:
            _error(errors, "T_A0_T_A1_NOT_AUTHORIZED", selected_path.as_posix(), "T-A1 predecessor gate is closed.")
    except (OSError, json.JSONDecodeError) as exc:
        _error(errors, "T_A0_SELECTED_SOURCE_UNREADABLE", selected_path.as_posix(), str(exc))

    for name in REQUIRED_JSON:
        path = evidence_dir / name
        if not path.is_file():
            _error(errors, "REQUIRED_ARTIFACT_MISSING", name, "Required T-A1 artifact is missing.")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _error(errors, "JSON_READ_FAILED", name, str(exc))
            continue
        documents[name] = data
        paths.append(path)
        if path.read_text(encoding="utf-8") != canonical_json(data):
            _error(errors, "NONDETERMINISTIC_JSON", name, "JSON is not canonical sorted-key content.")
        for location, value in _walk(data):
            if isinstance(value, str) and ("/Users/" in value or value.startswith(("file://", "~/"))):
                _error(errors, "ABSOLUTE_PATH_LEAKAGE", f"{name}:{location}", value)
            if isinstance(value, str) and ("Thermal-44" in value or "Thermal44" in value):
                if any(token in value for token in ("VERIFIED_UNIT", "VERIFIED_ENDIANNESS", "VERIFIED_DTYPE")):
                    _error(errors, "UNSUPPORTED_THERMAL44_ASSERTION", f"{name}:{location}", value)

    if set(REQUIRED_JSON) - set(documents):
        return _result(predecessor, errors, warnings, False)

    schema = documents["source_schema_profile.json"]
    expected_schema = {
        "dataset_doi": f"doi:{SDT_DOI}", "source_split": "test", "thermal_frame_count": 8000,
        "depth_frame_count": 8000, "label_row_count": 8000, "distributed_frame_shape": [480, 640],
        "native_thermal_sensor_shape": [120, 160], "png_bit_depth": 16, "channels": 1,
        "decoded_dtype": "uint16", "representation_class": "RADIOMETRIC_TEMPERATURE_ENCODED_UINT16",
    }
    for key, expected in expected_schema.items():
        if schema.get(key) != expected:
            _error(errors, "SCHEMA_CONTRACT_MISMATCH", f"source_schema_profile.json:{key}", f"expected={expected!r}, found={schema.get(key)!r}")
    if schema.get("source_labels") != {"0": "LYING", "1": "SITTING", "2": "STANDING", "3": "EMPTY_ROOM"}:
        _error(errors, "SOURCE_LABELS_REWRITTEN", "source_schema_profile.json:source_labels", "Original labels must be preserved.")
    for field in ("subject_availability", "session_availability", "sequence_availability", "event_availability", "timestamp_availability"):
        if schema.get(field) != "ABSENT":
            _error(errors, "UNSUPPORTED_PROVENANCE_ASSERTION", f"source_schema_profile.json:{field}", "SDT does not expose this identifier.")

    units = documents["raw_unit_contract.json"]
    witness = units.get("conversion_witness", {})
    if witness != {"celsius": 26.85, "encoded_uint16": 30000, "kelvin": 300.0}:
        _error(errors, "UNIT_WITNESS_INVALID", "raw_unit_contract.json", str(witness))
    encoded = np.array([30000], dtype=np.uint16)
    if encoded_to_kelvin(encoded).tolist() != [300.0] or encoded_to_celsius(encoded).tolist() != [26.85]:
        _error(errors, "READER_UNIT_CONVERSION_INVALID", "datasets/thermal/raw_reader.py", "Official witness failed.")
    if units.get("kelvin_formula") != "encoded_uint16 / 100" or units.get("celsius_formula") != "(encoded_uint16 - 27315) / 100":
        _error(errors, "UNIT_FORMULA_INVALID", "raw_unit_contract.json", "Official conversion formulas must be preserved exactly.")
    policy = str(units.get("lossless_source_value_policy", ""))
    for term in ("no resize", "normalization", "quantization", "relabeling", "inference"):
        if term not in policy.lower():
            _error(errors, "LOSSLESS_POLICY_INCOMPLETE", "raw_unit_contract.json:lossless_source_value_policy", term)

    provenance = documents["source_frame_provenance_contract.json"]
    required_fields = set(provenance.get("required_fields", []))
    for field in ("source_archive_sha256", "source_member_name", "source_frame_index", "source_pose_label", "source_bbox", "raw_encoded_frame_sha256"):
        if field not in required_fields:
            _error(errors, "PROVENANCE_FIELD_MISSING", "source_frame_provenance_contract.json", field)

    inventory = documents["archive_member_inventory.json"]
    identity = inventory.get("archive_identity", {})
    expected_identity = {"path": DEFAULT_ARCHIVE_PATH, "size_bytes": DEFAULT_ARCHIVE_SIZE, "md5": DEFAULT_ARCHIVE_MD5, "sha256": DEFAULT_ARCHIVE_SHA256, "materialization_state": "LOCALLY_MATERIALIZED"}
    if identity != expected_identity:
        _error(errors, "ARCHIVE_IDENTITY_CONTRACT_MISMATCH", "archive_member_inventory.json:archive_identity", str(identity))
    for key, expected in (("unexpected_members", []), ("missing_thermal_indices", []), ("missing_depth_indices", []), ("duplicate_member_names", []), ("duplicate_thermal_indices", []), ("duplicate_depth_indices", [])):
        if inventory.get(key) != expected:
            _error(errors, "ARCHIVE_LINKAGE_INVALID", f"archive_member_inventory.json:{key}", str(inventory.get(key)))
    placeholders = inventory.get("unhydrated_official_split_payloads", [])
    if len(placeholders) != 5 or any(item.get("materialization_state") != "LOCAL_CLOUD_PLACEHOLDER" or item.get("readable_offline") is not False for item in placeholders):
        _error(errors, "PLACEHOLDER_STATE_HIDDEN", "archive_member_inventory.json:unhydrated_official_split_payloads", "Five placeholders must remain explicit.")

    pilot = documents["reader_pilot_summary.json"]
    records = pilot.get("pilot_records", [])
    if pilot.get("pilot_indices") != PILOT_INDICES or len(records) != 12:
        _error(errors, "PILOT_SELECTION_INVALID", "reader_pilot_summary.json", "Expected deterministic 12-index pilot.")
    if pilot.get("class_counts") != {"0": 3, "1": 3, "2": 3, "3": 3}:
        _error(errors, "PILOT_CLASS_COVERAGE_INVALID", "reader_pilot_summary.json:class_counts", str(pilot.get("class_counts")))
    if any(record.get("source_pose_name") == "HUMAN_FALL" for record in records):
        _error(errors, "LYING_SILENTLY_RENAMED_FALL", "reader_pilot_summary.json", "Original LYING label must remain intact.")
    if not pilot.get("repeat_decode_deterministic") or any(not item.get("repeated_decode_equal") for item in records):
        _error(errors, "REPEAT_DECODE_NONDETERMINISTIC", "reader_pilot_summary.json", "Repeat decode differs.")

    failures = documents["failure_policy.json"]
    required_failures = {
        "archive_identity_mismatch", "archive_missing", "archive_not_materialized", "bbox_invalid",
        "duplicate_frame_or_member", "frame_label_mismatch", "invalid_label", "missing_label",
        "nonfinite_numeric_path", "path_traversal", "truncated_png", "wrong_channel", "wrong_shape",
        "wrong_dtype_or_bit_depth",
    }
    if not required_failures.issubset(set(failures.get("fail_closed_cases", {}))):
        _error(errors, "FAILURE_POLICY_INCOMPLETE", "failure_policy.json", "Required failure modes missing.")
    if failures.get("reader_skips_invalid_samples") is not False or failures.get("broad_exception_swallowing") is not False:
        _error(errors, "SILENT_SKIP_POLICY", "failure_policy.json", "Reader must not skip errors.")

    _validate_static_reader(repo_root, errors)

    if verify_real_payload:
        try:
            reader = SDTThermalRawReader(repo_root=repo_root)
            measured = reader.inspect_archive()
            if measured["archive_identity"] != identity:
                _error(errors, "REAL_ARCHIVE_IDENTITY_MISMATCH", DEFAULT_ARCHIVE_PATH, "Real archive differs from compact inventory.")
            for expected, record in zip(PILOT_INDICES, records):
                frame = reader.read_frame(expected)
                repeat = reader.read_frame(expected)
                if frame.source_frame_index != expected or record.get("source_frame_index") != expected:
                    _error(errors, "REAL_PILOT_INDEX_MISMATCH", f"pilot:{expected}", "Frame index not retained.")
                if frame.source_pose_label != record.get("source_pose_label") or frame.source_pose_name != record.get("source_pose_name"):
                    _error(errors, "REAL_PILOT_LABEL_MISMATCH", f"pilot:{expected}", "Source label not retained.")
                if frame.raw_encoded_frame_sha256 != record.get("raw_encoded_frame_sha256"):
                    _error(errors, "REAL_PILOT_VALUE_MISMATCH", f"pilot:{expected}", "Encoded array differs.")
                if not np.array_equal(frame.raw_encoded_frame, repeat.raw_encoded_frame):
                    _error(errors, "REAL_PILOT_REPEAT_MISMATCH", f"pilot:{expected}", "Repeated arrays differ.")
        except Exception as exc:
            _error(errors, "REAL_PAYLOAD_VALIDATION_FAILED", DEFAULT_ARCHIVE_PATH, str(exc))

    if check_checksums:
        result_path = evidence_dir / "validation_result.json"
        if not result_path.is_file():
            _error(errors, "VALIDATION_RESULT_MISSING", "validation_result.json", "Stored result is required.")
        else:
            paths.append(result_path)
        _validate_checksums(repo_root, evidence_dir, paths, errors)

    _warning(warnings, "NO_TEMPORAL_FALL_EVENTS", "SDT labels", "LYING is posture evidence, not a fall event.")
    _warning(warnings, "GROUP_IDENTIFIERS_ABSENT", "SDT provenance", "Subject/session/sequence/event/timestamp IDs are absent.")
    _warning(warnings, "PLACEHOLDERS_NOT_HYDRATED", "local payload", "Train and validation bytes remain unavailable offline.")
    _warning(warnings, "LICENSE_RESTRICTED", "T-A0 license", "Use the stricter non-commercial research common denominator.")
    return _result(predecessor, errors, warnings, not errors)


def _result(
    predecessor: dict[str, Any],
    errors: list[dict[str, str]],
    warnings: list[dict[str, str]],
    gate: bool,
) -> dict[str, Any]:
    sorted_errors = sorted(errors, key=lambda item: (item["code"], item["location"], item["message"]))
    sorted_warnings = sorted(warnings, key=lambda item: (item["code"], item["location"], item["message"]))
    return {
        "error_count": len(sorted_errors),
        "errors": sorted_errors,
        "evidence_validation": "PASS" if not sorted_errors else "FAIL",
        "overall_outcome": "PASS_WITH_LIMITATIONS" if gate and not sorted_errors else "NOT_VERIFIABLE",
        "phase": "T-A1",
        "schema_version": "1.0",
        "t_a0_outcome": predecessor.get("overall_outcome", "NOT_VERIFIABLE"),
        "t_a0_validation": predecessor.get("evidence_validation", "FAIL"),
        "t_a2_authorized": bool(gate and not sorted_errors),
        "warning_count": len(sorted_warnings),
        "warnings": sorted_warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--skip-real-payload", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    evidence_dir = args.evidence_dir or repo_root / "datasets/thermal/manifests/T-A1_safe_reader_raw_unit_contract"
    result = validate_evidence(
        repo_root=repo_root,
        evidence_dir=evidence_dir,
        check_checksums=True,
        verify_real_payload=not args.skip_real_payload,
    )
    print(canonical_json(result), end="")
    return 0 if result["evidence_validation"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
