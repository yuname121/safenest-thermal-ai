#!/usr/bin/env python3
"""Validate SafeNest Thermal real-capture provenance and integrity.

This validator deliberately does not import a model, inspect model outputs, or
authorize any dataset role.  It validates collection structure only.  A
successful result means that the capture is structurally usable evidence; it
does not mean that the capture is suitable for T-C, T-D, training, or a locked
test.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Iterable


VALIDATOR_NAME = "validate_thermal_real_capture.py"
CONTRACT_ID = "safenest.thermal.real_capture.v1"
FRAME_SCHEMA = "safenest.thermal.real_capture.frame.v1"
ANNOTATION_SCHEMA = "safenest.thermal.real_capture.annotation.v1"

FULL_FRAME_REPRESENTATIONS = {
    "RAW_PACKET_AND_NATIVE",
    "RAW_PACKET_ONLY",
    "DECODED_NATIVE_ONLY",
}
LIMITED_REPRESENTATIONS = {"SCALAR_ONLY"}
PREPROCESSED_REPRESENTATIONS = {"PREPROCESSED_ONLY", "SCREENSHOT_ONLY"}
REPRESENTATION_FILE_REQUIREMENTS: dict[str, dict[str, bool | None]] = {
    "RAW_PACKET_AND_NATIVE": {"raw_file": True, "decoded_native_file": True},
    "RAW_PACKET_ONLY": {"raw_file": True, "decoded_native_file": False},
    "DECODED_NATIVE_ONLY": {"raw_file": False, "decoded_native_file": True},
    "SCALAR_ONLY": {"raw_file": True, "decoded_native_file": False},
    "PREPROCESSED_ONLY": {"raw_file": True, "decoded_native_file": False},
    "SCREENSHOT_ONLY": {"raw_file": True, "decoded_native_file": False},
}
VALID_ROLES = {
    "DEVICE_CONTRACT_PILOT",
    "REAL_DEVELOPMENT",
    "FUTURE_TRAIN_CANDIDATE",
    "REAL_LOCKED_TEST",
}
VALID_MODEL_ACCESS_STATUSES = {"UNTOUCHED", "DEVELOPMENT_ALLOWED", "UNKNOWN"}
SOURCE_LABELS = {"EMPTY", "STANDING", "SITTING", "LYING", "UNKNOWN", "NOT_ANNOTATED"}
EVENT_PHASES = {
    "PRE_EVENT",
    "FALL_TRANSITION",
    "POST_FALL_LYING",
    "RECOVERY",
    "NORMAL_ACTIVITY",
    "NOT_APPLICABLE",
    "UNKNOWN",
}
PHASE_ORDER = {
    "PRE_EVENT": 0,
    "FALL_TRANSITION": 1,
    "POST_FALL_LYING": 2,
    "RECOVERY": 3,
    "NORMAL_ACTIVITY": 0,
    "NOT_APPLICABLE": 99,
    "UNKNOWN": 99,
}
CHECKSUM_LINE = re.compile(r"^([0-9a-fA-F]{64})  ([^\r\n]+)$")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _error(errors: list[dict[str, str]], code: str, path: str, message: str) -> None:
    errors.append({"code": code, "path": path, "message": message})


def _warning(warnings: list[dict[str, str]], code: str, path: str, message: str) -> None:
    warnings.append({"code": code, "path": path, "message": message})


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _portable_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or value.startswith(("/", "~")):
        return False
    if "\\" in value or "\x00" in value:
        return False
    path = PurePosixPath(value)
    return not any(part in {"", ".", ".."} for part in path.parts)


def _safe_join(base: Path, value: Any) -> Path | None:
    if not _portable_relative_path(value):
        return None
    candidate = (base / PurePosixPath(value)).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        return None
    return candidate


def _display_path(path: Path) -> str:
    """Keep generated result paths portable when the input is absolute."""

    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name or "."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, errors: list[dict[str, str]], label: str) -> Any | None:
    if not path.is_file():
        _error(errors, "REQUIRED_FILE_MISSING", label, f"Required file does not exist: {label}.")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _error(errors, "JSON_INVALID", label, f"Could not read valid JSON: {exc}.")
        return None


def _load_jsonl(path: Path, errors: list[dict[str, str]], label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        _error(errors, "REQUIRED_FILE_MISSING", label, f"Required file does not exist: {label}.")
        return []
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        _error(errors, "JSONL_INVALID", label, f"Could not read JSONL: {exc}.")
        return records
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            _error(errors, "JSONL_BLANK_LINE", f"{label}:{line_number}", "Blank JSONL lines are not allowed.")
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            _error(errors, "JSONL_INVALID", f"{label}:{line_number}", f"Invalid JSONL record: {exc}.")
            continue
        if not isinstance(record, dict):
            _error(errors, "JSONL_RECORD_NOT_OBJECT", f"{label}:{line_number}", "Each JSONL record must be an object.")
            continue
        records.append(record)
    return records


def _required_fields(value: Any, fields: Iterable[str], path: str, errors: list[dict[str, str]]) -> None:
    if not isinstance(value, dict):
        _error(errors, "MANIFEST_NOT_OBJECT", path, "Manifest record must be a JSON object.")
        return
    for field in fields:
        if field not in value:
            _error(errors, "REQUIRED_FIELD_MISSING", f"{path}:{field}", f"Required field '{field}' is missing.")


def _check_schema_version(value: dict[str, Any], expected: str, path: str, errors: list[dict[str, str]]) -> None:
    if value.get("schema_version") != expected:
        _error(
            errors,
            "SCHEMA_VERSION_INVALID",
            f"{path}:schema_version",
            f"Expected {expected!r}, got {value.get('schema_version')!r}.",
        )


def _check_id(value: Any, field: str, path: str, errors: list[dict[str, str]]) -> None:
    if not _is_nonempty_string(value):
        code = "MISSING_SUBJECT_ID" if field == "subject_id" else "MISSING_SESSION_ID" if field == "session_id" else "INVALID_ID"
        _error(errors, code, f"{path}:{field}", f"{field} must be a non-empty pseudonymous identifier.")
        return
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        _error(errors, "ID_NOT_PORTABLE", f"{path}:{field}", f"{field} contains non-portable characters.")


def _check_relative_manifest_path(value: Any, field: str, base: Path, path: str, errors: list[dict[str, str]]) -> Path | None:
    if not _portable_relative_path(value):
        _error(
            errors,
            "NONPORTABLE_PATH",
            f"{path}:{field}",
            "Manifest paths must be session-relative POSIX paths without '..', absolute prefixes, or backslashes.",
        )
        return None
    resolved = _safe_join(base, value)
    if resolved is None:
        _error(errors, "PATH_ESCAPES_SESSION", f"{path}:{field}", "Manifest path escapes the session directory.")
    return resolved


def _validate_representation_file_matrix(
    representation: Any,
    raw_file: Any,
    decoded_native_file: Any,
    validity_status: Any,
    path: str,
    errors: list[dict[str, str]],
) -> None:
    requirements = REPRESENTATION_FILE_REQUIREMENTS.get(representation)
    if requirements is None:
        return
    for field, required in requirements.items():
        value = raw_file if field == "raw_file" else decoded_native_file
        if required is True and value is None and validity_status == "VALID":
            code = "RAW_FRAME_REFERENCE_MISSING" if field == "raw_file" else "DECODED_NATIVE_REFERENCE_MISSING"
            _error(errors, code, f"{path}:{field}", f"{representation} requires {field} for a valid frame.")
        elif required is False and value is not None:
            code = "RAW_FILE_NOT_ALLOWED_FOR_REPRESENTATION" if field == "raw_file" else "DECODED_NATIVE_FILE_NOT_ALLOWED_FOR_REPRESENTATION"
            _error(errors, code, f"{path}:{field}", f"{representation} must not register {field}.")


def _validate_collection_manifest(collection: Any, path: str, errors: list[dict[str, str]], warnings: list[dict[str, str]]) -> None:
    _required_fields(
        collection,
        [
            "schema_version",
            "contract_id",
            "collection_id",
            "collection_role",
            "created_at",
            "subject_ids",
            "session_ids",
            "full_frame_collection_status",
            "split_policy",
            "source",
        ],
        path,
        errors,
    )
    if not isinstance(collection, dict):
        return
    if collection.get("schema_version") != "safenest.thermal.real_capture.collection.v1":
        _error(errors, "SCHEMA_VERSION_INVALID", f"{path}:schema_version", "Collection schema version is not v1.")
    if collection.get("contract_id") != CONTRACT_ID:
        _error(errors, "CONTRACT_ID_INVALID", f"{path}:contract_id", "Unexpected Thermal real-capture contract ID.")
    _check_id(collection.get("collection_id"), "collection_id", path, errors)
    role = collection.get("collection_role")
    if role not in {"DEVICE_CONTRACT_PILOT", "REAL_DEVELOPMENT", "FUTURE_TRAIN_CANDIDATE", "REAL_LOCKED_TEST"}:
        _error(errors, "COLLECTION_ROLE_INVALID", f"{path}:collection_role", "Unknown collection role.")
    for field in ("subject_ids", "session_ids"):
        values = collection.get(field)
        if not isinstance(values, list):
            _error(errors, "COLLECTION_ID_LIST_INVALID", f"{path}:{field}", "Collection ID lists must be arrays.")
            continue
        if len(values) != len(set(values)):
            _error(errors, "DUPLICATE_COLLECTION_ID", f"{path}:{field}", f"Duplicate values found in {field}.")
        for index, value in enumerate(values):
            if not _is_nonempty_string(value):
                _error(errors, "INVALID_COLLECTION_ID", f"{path}:{field}[{index}]", "Collection IDs must be non-empty strings.")
    source = collection.get("source")
    if not isinstance(source, dict):
        _error(errors, "SOURCE_BLOCK_MISSING", f"{path}:source", "Source provenance block is required.")
    elif source.get("source_status") == "SYNTHETIC_EXAMPLE_NOT_MEASUREMENT":
        _warning(warnings, "SYNTHETIC_EXAMPLE", f"{path}:source", "This collection is synthetic example metadata, not a real measurement.")
    split = collection.get("split_policy")
    if not isinstance(split, dict):
        _error(errors, "SPLIT_POLICY_MISSING", f"{path}:split_policy", "Split policy is required.")
    else:
        assignment_unit = split.get("assignment_unit")
        if assignment_unit not in {"SUBJECT", "SESSION", "EVENT", "NOT_ASSIGNED"}:
            _error(errors, "SPLIT_UNIT_INVALID", f"{path}:split_policy:assignment_unit", "Frame-level assignment is not allowed.")
        if split.get("frame_random_split_allowed") is not False:
            _error(errors, "FRAME_RANDOM_SPLIT_REJECTED", f"{path}:split_policy:frame_random_split_allowed", "Frame-random splitting must be explicitly false.")
        assignment_method = str(split.get("assignment_method", "")).upper()
        if "FRAME" in assignment_method or "RANDOM" in assignment_method or "HASH" in assignment_method:
            _error(errors, "FRAME_RANDOM_SPLIT_REJECTED", f"{path}:split_policy:assignment_method", "Assignment method suggests frame-random/hash splitting.")
    privacy = collection.get("privacy")
    if isinstance(privacy, dict):
        if privacy.get("pseudonymous_ids_only") is not True:
            _error(errors, "PRIVACY_ID_POLICY_INVALID", f"{path}:privacy:pseudonymous_ids_only", "Subject IDs must be pseudonymous.")
        if privacy.get("names_in_filenames") is True:
            _error(errors, "PERSONAL_ID_IN_FILENAME", f"{path}:privacy:names_in_filenames", "Personal names are prohibited in filenames.")
        if privacy.get("unnecessary_personal_metadata_present") is True:
            _error(errors, "UNNECESSARY_PERSONAL_METADATA", f"{path}:privacy", "Unnecessary personal metadata must not be collected.")


def _resolve_timestamp(value: Any, unit: Any = None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        unit_text = str(unit or "s").lower()
        if unit_text in {"ns", "nanoseconds"}:
            return number / 1_000_000_000.0
        if unit_text in {"ms", "milliseconds"}:
            return number / 1_000.0
        return number
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.timestamp()
        except ValueError:
            return None
    return None


def _series_checks(
    frames: list[dict[str, Any]],
    field: str,
    path_prefix: str,
    errors: list[dict[str, str]],
    warnings: list[dict[str, str]],
) -> dict[str, Any]:
    values: list[tuple[int, int]] = []
    missing = 0
    for index, frame in enumerate(frames):
        value = frame.get(field)
        if value is None:
            missing += 1
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            _error(errors, "TIMING_VALUE_INVALID", f"{path_prefix}:{index}:{field}", f"{field} must be a non-negative integer or null.")
            continue
        values.append((index, value))
    duplicates = 0
    seen: dict[int, int] = {}
    gap_count = 0
    reversal_count = 0
    for index, value in values:
        if value in seen:
            duplicates += 1
            _error(errors, "DUPLICATE_SEQUENCE_INDEX" if field == "sequence_index" else "DUPLICATE_SENSOR_COUNTER", f"{path_prefix}:{index}:{field}", f"Duplicate {field} {value}.")
        seen[value] = index
    for (_, previous), (index, current) in zip(values, values[1:]):
        if current < previous:
            reversal_count += 1
            _error(errors, "SEQUENCE_REVERSAL" if field == "sequence_index" else "SENSOR_COUNTER_REVERSAL", f"{path_prefix}:{index}:{field}", f"{field} decreases from {previous} to {current}.")
        if current > previous + 1:
            gap = current - previous - 1
            gap_count += gap
            _warning(warnings, "SEQUENCE_GAP" if field == "sequence_index" else "SENSOR_COUNTER_GAP", f"{path_prefix}:{index}:{field}", f"Observed {gap} missing value(s) between {previous} and {current}; this is reported, not silently removed.")
    return {
        "field": field,
        "observed_count": len(values),
        "missing_count": missing,
        "duplicate_count": duplicates,
        "gap_count": gap_count,
        "reversal_count": reversal_count,
        "first": values[0][1] if values else None,
        "last": values[-1][1] if values else None,
    }


def _timestamp_checks(
    frames: list[dict[str, Any]],
    field: str,
    unit_field: str | None,
    path_prefix: str,
    errors: list[dict[str, str]],
    warnings: list[dict[str, str]],
) -> dict[str, Any]:
    values: list[tuple[int, float]] = []
    missing = 0
    for index, frame in enumerate(frames):
        raw = frame.get(field)
        value = _resolve_timestamp(raw, frame.get(unit_field) if unit_field else None)
        if value is None:
            missing += 1
            continue
        values.append((index, value))
    reversal_count = 0
    zero_interval_count = 0
    large_gap_count = 0
    intervals: list[float] = []
    for index, ((_, previous), (_, current)) in enumerate(zip(values, values[1:]), 1):
        delta = current - previous
        if delta < 0:
            reversal_count += 1
            _error(errors, "TIMESTAMP_REVERSAL", f"{path_prefix}:{index}:{field}", f"{field} moves backwards by {delta:.9f} seconds.")
        elif delta == 0:
            zero_interval_count += 1
            _warning(warnings, "DUPLICATE_TIMESTAMP", f"{path_prefix}:{index}:{field}", f"{field} repeats at two adjacent records.")
        else:
            intervals.append(delta)
    if intervals:
        median = sorted(intervals)[len(intervals) // 2]
        threshold = max(1.0, median * 3.0)
        for index, delta in enumerate(intervals, 1):
            if delta > threshold:
                large_gap_count += 1
                _warning(warnings, "LARGE_TIMING_GAP", f"{path_prefix}:{index}:{field}", f"Inter-frame interval {delta:.6f}s exceeds the report threshold {threshold:.6f}s.")
    return {
        "field": field,
        "observed_count": len(values),
        "missing_count": missing,
        "reversal_count": reversal_count,
        "zero_interval_count": zero_interval_count,
        "large_gap_count": large_gap_count,
        "first": values[0][1] if values else None,
        "last": values[-1][1] if values else None,
    }


def _effective_fps(frames: list[dict[str, Any]]) -> tuple[float | None, str | None]:
    candidates: list[tuple[str, list[float]]] = []
    for field, unit_field, label in (
        ("host_receive_monotonic_timestamp_ns", None, "host_receive_monotonic"),
        ("device_monotonic_timestamp_ns", None, "device_monotonic"),
        ("host_wall_time", None, "host_wall_clock"),
    ):
        values = []
        for frame in frames:
            value = _resolve_timestamp(frame.get(field), "ns" if field.endswith("_ns") else None)
            if value is not None:
                values.append(value)
        if len(values) >= 2:
            candidates.append((label, values))
    if not candidates:
        return None, None
    label, values = candidates[0]
    duration = values[-1] - values[0]
    if duration <= 0:
        return None, label
    return (len(values) - 1) / duration, label


def _validate_checksums(
    session_dir: Path,
    checksum_path: Path | None,
    required_paths: set[str],
    errors: list[dict[str, str]],
) -> str:
    if checksum_path is None or not checksum_path.is_file():
        _error(errors, "CHECKSUM_REGISTRY_MISSING", "checksums.sha256", "A session checksum registry is required.")
        return "FAIL"
    entries: dict[str, str] = {}
    previous = ""
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        _error(errors, "CHECKSUM_REGISTRY_INVALID", "checksums.sha256", f"Could not read checksum registry: {exc}.")
        return "FAIL"
    for line_number, line in enumerate(lines, 1):
        match = CHECKSUM_LINE.fullmatch(line)
        if not match:
            _error(errors, "CHECKSUM_LINE_INVALID", f"checksums.sha256:{line_number}", "Expected '<sha256><two spaces><relative path>'.")
            continue
        digest, relative = match.groups()
        if not _portable_relative_path(relative):
            _error(errors, "CHECKSUM_PATH_NOT_PORTABLE", f"checksums.sha256:{line_number}", "Checksum paths must be session-relative POSIX paths.")
            continue
        if relative <= previous:
            _error(errors, "CHECKSUM_ORDER_NONDETERMINISTIC", f"checksums.sha256:{line_number}", "Checksum entries must be sorted by path.")
        previous = relative
        if relative in entries:
            _error(errors, "CHECKSUM_DUPLICATE_PATH", f"checksums.sha256:{line_number}", f"Duplicate checksum entry for {relative}.")
        entries[relative] = digest.lower()
        resolved = _safe_join(session_dir, relative)
        if resolved is None or not resolved.is_file():
            _error(errors, "CHECKSUM_TARGET_MISSING", f"checksums.sha256:{line_number}", f"Checksum target does not exist: {relative}.")
        elif _sha256(resolved) != digest.lower():
            _error(errors, "CHECKSUM_MISMATCH", relative, "Measured SHA-256 differs from the registry.")
    for relative in sorted(required_paths):
        if relative not in entries:
            _error(errors, "CHECKSUM_COVERAGE_MISSING", relative, "Required finalized artifact has no checksum entry.")
    return "FAIL" if any(item["code"].startswith("CHECKSUM_") for item in errors) else "PASS"


def _raw_classification(representations: list[str]) -> str:
    if not representations:
        return "UNKNOWN_OR_EMPTY"
    values = set(representations)
    full = bool(values & FULL_FRAME_REPRESENTATIONS)
    limited = bool(values & LIMITED_REPRESENTATIONS)
    preprocessed = bool(values & PREPROCESSED_REPRESENTATIONS)
    unknown = "UNKNOWN" in values
    if preprocessed and not full and not limited:
        return "PREPROCESSED_ONLY_INSUFFICIENT"
    if limited and not full and not preprocessed and not unknown:
        return "SCALAR_ONLY_LIMITED"
    if full and (limited or preprocessed or unknown):
        return "MIXED_FULL_FRAME_AND_LIMITED"
    if full:
        return "FULL_FRAME_RAW"
    if preprocessed:
        return "PREPROCESSED_ONLY_INSUFFICIENT"
    return "UNKNOWN_OR_EMPTY"


def _resolve_frame_order(
    frame_ids: dict[str, dict[str, Any]],
    frame_positions: dict[str, int],
    frame_id: Any,
    path: str,
    errors: list[dict[str, str]],
) -> int | None:
    if not _is_nonempty_string(frame_id):
        _error(errors, "EVENT_FRAME_REFERENCE_MISSING", path, "Event range frame IDs must be non-empty strings.")
        return None
    if frame_id not in frame_ids:
        _error(errors, "ANNOTATION_FRAME_REFERENCE_MISSING", path, f"Annotation references unknown frame_id {frame_id!r}.")
        return None
    return frame_positions.get(frame_id)


def _validate_annotations(
    annotations: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    session: dict[str, Any],
    path: str,
    errors: list[dict[str, str]],
    warnings: list[dict[str, str]],
) -> dict[str, Any]:
    frame_map = {frame.get("frame_id"): frame for frame in frames if _is_nonempty_string(frame.get("frame_id"))}
    frame_positions = {
        frame_id: index
        for index, frame_id in enumerate(frame.get("frame_id") for frame in frames)
        if _is_nonempty_string(frame_id)
    }
    annotation_ids: set[str] = set()
    event_record_ids: set[str] = set()
    annotated_frame_ids: set[str] = set()
    source_counts: Counter[str] = Counter()
    event_phases: dict[str, set[str]] = defaultdict(set)
    phase_ranges_by_event: dict[str, list[tuple[int, int, str, str]]] = defaultdict(list)
    for index, annotation in enumerate(annotations):
        item_path = f"{path}:{index}"
        _required_fields(
            annotation,
            [
                "schema_version",
                "annotation_id",
                "collection_id",
                "subject_id",
                "session_id",
                "annotation_scope",
                "event_id",
                "frame_id",
                "source_annotation",
                "derived_safenest_annotation",
                "provenance",
                "revision",
            ],
            item_path,
            errors,
        )
        if not isinstance(annotation, dict):
            continue
        if annotation.get("schema_version") != ANNOTATION_SCHEMA:
            _error(errors, "SCHEMA_VERSION_INVALID", f"{item_path}:schema_version", "Annotation schema version is invalid.")
        annotation_id = annotation.get("annotation_id")
        if not _is_nonempty_string(annotation_id):
            _error(errors, "ANNOTATION_ID_MISSING", f"{item_path}:annotation_id", "Annotation ID is required.")
        elif annotation_id in annotation_ids:
            _error(errors, "DUPLICATE_ANNOTATION_ID", f"{item_path}:annotation_id", f"Duplicate annotation ID {annotation_id!r}.")
        else:
            annotation_ids.add(annotation_id)
        for field in ("collection_id", "subject_id", "session_id"):
            if annotation.get(field) != session.get(field):
                _error(errors, "ANNOTATION_IDENTITY_MISMATCH", f"{item_path}:{field}", f"Annotation {field} does not match session manifest.")
        scope = annotation.get("annotation_scope")
        if scope not in {"FRAME", "SESSION", "EVENT"}:
            _error(errors, "ANNOTATION_SCOPE_INVALID", f"{item_path}:annotation_scope", "Unsupported annotation scope.")
        frame_id = annotation.get("frame_id")
        if frame_id is not None:
            if frame_id not in frame_map:
                _error(errors, "ANNOTATION_FRAME_REFERENCE_MISSING", f"{item_path}:frame_id", f"Unknown frame_id {frame_id!r}.")
            else:
                annotated_frame_ids.add(frame_id)
        if scope == "FRAME" and frame_id is None:
            _error(errors, "FRAME_ANNOTATION_ID_MISSING", f"{item_path}:frame_id", "FRAME annotation must name a frame_id.")
        event_id = annotation.get("event_id")
        if scope == "EVENT" and not _is_nonempty_string(event_id):
            _error(errors, "EVENT_ID_MISSING", f"{item_path}:event_id", "EVENT annotation must name an event_id.")
        if event_id is not None:
            if not _is_nonempty_string(event_id):
                _error(errors, "EVENT_ID_INVALID", f"{item_path}:event_id", "event_id must be a pseudonymous identifier or null.")
            if frame_id in frame_map and frame_map[frame_id].get("event_id") not in {None, event_id}:
                _error(errors, "FRAME_EVENT_ID_MISMATCH", f"{item_path}:event_id", "Frame and annotation event IDs disagree.")
        if scope == "EVENT" and event_id in event_record_ids:
            _error(errors, "DUPLICATE_EVENT_ID", f"{item_path}:event_id", f"Duplicate EVENT record for {event_id!r}.")
        elif scope == "EVENT" and _is_nonempty_string(event_id):
            event_record_ids.add(event_id)
        source = annotation.get("source_annotation")
        if not isinstance(source, dict):
            _error(errors, "SOURCE_ANNOTATION_MISSING", f"{item_path}:source_annotation", "Source annotation block is required.")
        else:
            label = source.get("label")
            if label not in SOURCE_LABELS:
                _error(errors, "UNSUPPORTED_SOURCE_LABEL", f"{item_path}:source_annotation:label", f"Unsupported source label {label!r}.")
            else:
                source_counts[label] += 1
        derived = annotation.get("derived_safenest_annotation")
        if not isinstance(derived, dict):
            _error(errors, "DERIVED_ANNOTATION_MISSING", f"{item_path}:derived_safenest_annotation", "Derived SafeNest annotation block is required.")
        elif source and source.get("label") == "LYING" and derived.get("label") == "HUMAN_FALL" and derived.get("mapping_type") != "DERIVED_POSTURE_PROXY":
            _error(errors, "LYING_PROMOTED_TO_FALL", f"{item_path}:derived_safenest_annotation", "LYING is posture evidence; HUMAN_FALL must remain an explicitly qualified derived posture proxy.")
        phase = annotation.get("event_phase")
        if phase is not None:
            if phase not in EVENT_PHASES:
                _error(errors, "UNSUPPORTED_EVENT_PHASE", f"{item_path}:event_phase", f"Unsupported event phase {phase!r}.")
            elif _is_nonempty_string(event_id):
                event_phases[event_id].add(phase)
        phase_ranges = annotation.get("phase_ranges", [])
        if phase_ranges is None:
            phase_ranges = []
        if not isinstance(phase_ranges, list):
            _error(errors, "PHASE_RANGE_INVALID", f"{item_path}:phase_ranges", "phase_ranges must be an array.")
            phase_ranges = []
        previous_phase_order = -1
        for phase_index, phase_range in enumerate(phase_ranges):
            range_path = f"{item_path}:phase_ranges:{phase_index}"
            if not isinstance(phase_range, dict):
                _error(errors, "PHASE_RANGE_INVALID", range_path, "Each phase range must be an object.")
                continue
            phase_name = phase_range.get("phase")
            if phase_name not in EVENT_PHASES:
                _error(errors, "UNSUPPORTED_EVENT_PHASE", f"{range_path}:phase", f"Unsupported event phase {phase_name!r}.")
            phase_order = PHASE_ORDER.get(phase_name, 99)
            if phase_order < previous_phase_order:
                _error(errors, "EVENT_PHASE_ORDER_INVALID", range_path, "Event phase ranges are not ordered.")
            previous_phase_order = max(previous_phase_order, phase_order)
            start = _resolve_frame_order(
                frame_map,
                frame_positions,
                phase_range.get("start_frame_id"),
                f"{range_path}:start_frame_id",
                errors,
            )
            end = _resolve_frame_order(
                frame_map,
                frame_positions,
                phase_range.get("end_frame_id"),
                f"{range_path}:end_frame_id",
                errors,
            )
            if start is not None and end is not None and start > end:
                _error(errors, "EVENT_RANGE_REVERSED", range_path, "Event phase start must be <= end.")
            elif start is not None and end is not None and _is_nonempty_string(event_id):
                range_frames = frames[start : end + 1]
                mismatched = next(
                    (
                        frame
                        for frame in range_frames
                        if frame.get("event_id") != event_id
                    ),
                    None,
                )
                if mismatched is not None:
                    _error(
                        errors,
                        "EVENT_RANGE_FRAME_EVENT_MISMATCH",
                        range_path,
                        f"All frames in this phase range must carry event_id {event_id!r}.",
                    )
                elif phase_name in EVENT_PHASES:
                    phase_ranges_by_event[event_id].append((start, end, phase_name, range_path))
            if _is_nonempty_string(event_id) and phase_name in EVENT_PHASES:
                event_phases[event_id].add(phase_name)
        start_id = annotation.get("frame_start_id")
        end_id = annotation.get("frame_end_id")
        if start_id is not None or end_id is not None:
            start = _resolve_frame_order(frame_map, frame_positions, start_id, f"{item_path}:frame_start_id", errors)
            end = _resolve_frame_order(frame_map, frame_positions, end_id, f"{item_path}:frame_end_id", errors)
            if start is not None and end is not None and start > end:
                _error(errors, "EVENT_RANGE_REVERSED", item_path, "Event frame_start_id must be <= frame_end_id.")
            elif start is not None and end is not None and _is_nonempty_string(event_id):
                if any(frame.get("event_id") != event_id for frame in frames[start : end + 1]):
                    _error(
                        errors,
                        "EVENT_RANGE_FRAME_EVENT_MISMATCH",
                        item_path,
                        f"All frames in the event envelope must carry event_id {event_id!r}.",
                    )
    for event_id, ranges in phase_ranges_by_event.items():
        ordered_ranges = sorted(ranges, key=lambda item: (item[0], item[1], PHASE_ORDER.get(item[2], 99)))
        previous: tuple[int, int, str, str] | None = None
        previous_phase_order: int | None = None
        for current in ordered_ranges:
            current_phase_order = PHASE_ORDER.get(current[2], 99)
            if previous_phase_order is not None and current_phase_order < previous_phase_order:
                _error(
                    errors,
                    "EVENT_PHASE_ORDER_INVALID",
                    current[3],
                    f"Phase ranges are not in PRE_EVENT to POST_FALL_LYING order for event {event_id!r}.",
                )
            if previous is not None and current[0] <= previous[1]:
                _error(
                    errors,
                    "EVENT_PHASE_OVERLAP",
                    current[3],
                    f"Phase range {current[2]!r} overlaps the preceding {previous[2]!r} range for event {event_id!r}.",
                )
            previous = current
            if previous_phase_order is None:
                previous_phase_order = current_phase_order
            else:
                previous_phase_order = max(previous_phase_order, current_phase_order)
    valid_frame_ids = {frame.get("frame_id") for frame in frames if frame.get("validity_status") == "VALID"}
    coverage = len(annotated_frame_ids & valid_frame_ids) / len(valid_frame_ids) if valid_frame_ids else 0.0
    phase_range_phases_by_event = {
        event_id: {phase_name for _, _, phase_name, _ in ranges}
        for event_id, ranges in phase_ranges_by_event.items()
    }
    temporal_event_ids = [
        event_id
        for event_id, phases in phase_range_phases_by_event.items()
        if {"PRE_EVENT", "FALL_TRANSITION", "POST_FALL_LYING"}.issubset(phases)
    ]
    range_integrity_error_codes = {
        "ANNOTATION_FRAME_REFERENCE_MISSING",
        "EVENT_FRAME_REFERENCE_MISSING",
        "EVENT_RANGE_REVERSED",
        "EVENT_PHASE_ORDER_INVALID",
        "EVENT_PHASE_OVERLAP",
        "EVENT_RANGE_FRAME_EVENT_MISMATCH",
    }
    event_range_integrity_valid = not any(item["code"] in range_integrity_error_codes for item in errors)
    if not annotations:
        _warning(warnings, "ANNOTATIONS_EMPTY", path, "No annotations were supplied; UNKNOWN is preferable to fabricated labels.")
    return {
        "record_count": len(annotations),
        "unique_annotation_count": len(annotation_ids),
        "annotated_frame_count": len(annotated_frame_ids & valid_frame_ids),
        "valid_frame_count": len(valid_frame_ids),
        "coverage_ratio": round(coverage, 6),
        "source_label_counts": dict(sorted(source_counts.items())),
        "event_ids": sorted(event_phases),
        "complete_temporal_event_ids": sorted(temporal_event_ids),
        "event_range_integrity_valid": event_range_integrity_valid,
    }


def _validate_session(
    session_path: Path,
    collection: dict[str, Any] | None,
    inherited_errors: list[dict[str, str]],
    inherited_warnings: list[dict[str, str]],
) -> dict[str, Any]:
    session_dir = session_path.parent
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    session = _load_json(session_path, errors, "session.json")
    if not isinstance(session, dict):
        session = {}
    _required_fields(
        session,
        [
            "schema_version",
            "collection_id",
            "subject_id",
            "session_id",
            "recording_id",
            "capture_start_time",
            "capture_end_time",
            "timezone",
            "sensor",
            "transport",
            "timing",
            "environment",
            "storage",
            "quality",
            "role_governance",
        ],
        "session.json",
        errors,
    )
    _check_schema_version(session, "safenest.thermal.real_capture.session.v1", "session.json", errors)
    _check_id(session.get("subject_id"), "subject_id", "session.json", errors)
    _check_id(session.get("session_id"), "session_id", "session.json", errors)
    if collection and session.get("collection_id") != collection.get("collection_id"):
        _error(errors, "COLLECTION_ID_MISMATCH", "session.json:collection_id", "Session collection_id does not match collection.json.")
    role_block = session.get("role_governance")
    role = role_block.get("role") if isinstance(role_block, dict) else None
    if role not in VALID_ROLES:
        _error(errors, "SESSION_ROLE_INVALID", "session.json:role_governance:role", "Unknown session role.")
    if isinstance(session.get("safety"), dict):
        safety = session["safety"]
        if safety.get("uncontrolled_free_fall_experiment") is True or safety.get("safety_control_status") == "UNCONTROLLED":
            _error(errors, "UNCONTROLLED_FALL_EXPERIMENT", "session.json:safety", "Unprotected free-fall experiments are prohibited.")
    transport = session.get("transport") if isinstance(session.get("transport"), dict) else {}
    framed_udp_v2 = transport.get("protocol") == "SAFENEST_THERMAL_RAW_UDP_V2"
    legacy_stream = transport.get("udp_datagram_mode") == "UNSAFE_LEGACY_STREAM_REASSEMBLY"
    if framed_udp_v2:
        expected_transport = {
            "protocol_version": "2",
            "udp_datagram_mode": "FRAME_ID_CHUNKED_REASSEMBLY",
            "full_frame_status": "PRESERVED",
            "raw_packet_bytes_preserved": True,
            "raw_udp_chunks_preserved": True,
            "chunk_header_bytes": 32,
            "chunk_payload_bytes": 1168,
            "expected_chunks_per_frame": 9,
            "frame_crc_status": "WHOLE_FRAME_CRC32_REQUIRED",
            "reassembly_safety": "FRAME_BOUNDARY_EXPLICIT_FAIL_CLOSED",
        }
        for field, expected in expected_transport.items():
            if transport.get(field) != expected:
                _error(
                    errors,
                    "FRAMED_UDP_V2_CONTRACT_INVALID",
                    f"session.json:transport:{field}",
                    f"Expected {expected!r}, got {transport.get(field)!r}.",
                )
    elif legacy_stream:
        _warning(
            warnings,
            "UNSAFE_LEGACY_STREAM_REASSEMBLY",
            "session.json:transport:udp_datagram_mode",
            "Blind byte-stream chunk concatenation cannot prove frame boundaries after UDP loss; retain only as diagnostic evidence.",
        )
    storage = session.get("storage") if isinstance(session.get("storage"), dict) else {}
    storage_paths: dict[str, Path | None] = {}
    for field in ("frames_file", "annotations_file", "checksums_file", "raw_root", "decoded_native_root"):
        storage_paths[field] = _check_relative_manifest_path(storage.get(field), field, session_dir, "session.json:storage", errors)
    frames_path = storage_paths.get("frames_file")
    annotations_path = storage_paths.get("annotations_file")
    checksums_path = storage_paths.get("checksums_file")
    raw_root = storage_paths.get("raw_root")
    decoded_root = storage_paths.get("decoded_native_root")
    sensor = session.get("sensor") if isinstance(session.get("sensor"), dict) else {}
    raw_chunks_root = None
    if storage.get("raw_chunks_root") is not None:
        raw_chunks_root = _check_relative_manifest_path(
            storage.get("raw_chunks_root"),
            "raw_chunks_root",
            session_dir,
            "session.json:storage",
            errors,
        )
    if framed_udp_v2 and (raw_chunks_root is None or not raw_chunks_root.is_dir()):
        _error(
            errors,
            "RAW_CHUNKS_ROOT_MISSING",
            "session.json:storage:raw_chunks_root",
            "Framed UDP V2 capture must preserve every received datagram in raw_chunks/.",
        )
    frames = _load_jsonl(frames_path, errors, "frames.jsonl") if frames_path else []
    annotations = _load_jsonl(annotations_path, errors, "annotations.jsonl") if annotations_path else []
    frame_ids: set[str] = set()
    representations: list[str] = []
    raw_references: set[str] = set()
    decoded_references: set[str] = set()
    for index, frame in enumerate(frames):
        item_path = f"frames.jsonl:{index}"
        _required_fields(
            frame,
            [
                "schema_version",
                "frame_id",
                "collection_id",
                "subject_id",
                "session_id",
                "sequence_index",
                "sequence_index_status",
                "sensor_frame_counter",
                "sensor_frame_counter_status",
                "raw_file",
                "decoded_native_file",
                "raw_representation",
                "native_shape",
                "native_dtype",
                "raw_encoding",
                "raw_unit_claim",
                "unit_status",
                "validity_status",
                "exclude_reason",
                "annotation_status",
            ],
            item_path,
            errors,
        )
        _check_schema_version(frame, FRAME_SCHEMA, item_path, errors)
        frame_id = frame.get("frame_id")
        if not _is_nonempty_string(frame_id):
            _error(errors, "FRAME_ID_MISSING", f"{item_path}:frame_id", "Every frame row requires a deterministic frame_id.")
        elif frame_id in frame_ids:
            _error(errors, "DUPLICATE_FRAME_ID", f"{item_path}:frame_id", f"Duplicate frame_id {frame_id!r}.")
        else:
            frame_ids.add(frame_id)
        for field in ("collection_id", "subject_id", "session_id"):
            if frame.get(field) != session.get(field):
                code = "MISSING_SUBJECT_ID" if field == "subject_id" and frame.get(field) is None else "MISSING_SESSION_ID" if field == "session_id" and frame.get(field) is None else "FRAME_IDENTITY_MISMATCH"
                _error(errors, code, f"{item_path}:{field}", f"Frame {field} does not match session manifest.")
        representation = frame.get("raw_representation")
        if representation not in FULL_FRAME_REPRESENTATIONS | LIMITED_REPRESENTATIONS | PREPROCESSED_REPRESENTATIONS | {"UNKNOWN"}:
            _error(errors, "RAW_REPRESENTATION_INVALID", f"{item_path}:raw_representation", f"Unsupported raw representation {representation!r}.")
        elif frame.get("validity_status") == "VALID":
            # Missing, corrupt, partial, and duplicate markers must stay in the
            # manifest as transport evidence. They do not, however, change the
            # collection's successful raw-frame representation classification.
            representations.append(representation)
        raw_file = frame.get("raw_file")
        decoded_file = frame.get("decoded_native_file")
        _validate_representation_file_matrix(
            representation,
            raw_file,
            decoded_file,
            frame.get("validity_status"),
            item_path,
            errors,
        )
        if raw_file is not None:
            resolved = _check_relative_manifest_path(raw_file, "raw_file", session_dir, item_path, errors)
            if resolved is not None:
                normalized = resolved.relative_to(session_dir.resolve()).as_posix()
                raw_references.add(normalized)
                if not resolved.is_file():
                    _error(errors, "MISSING_RAW_FRAME", f"{item_path}:raw_file", f"Referenced raw artifact does not exist: {raw_file}.")
                elif frame.get("raw_sha256") and _sha256(resolved) != str(frame["raw_sha256"]).lower():
                    _error(errors, "RAW_FRAME_CHECKSUM_MISMATCH", f"{item_path}:raw_sha256", "Frame raw_sha256 does not match the referenced file.")
        if decoded_file is not None:
            resolved = _check_relative_manifest_path(decoded_file, "decoded_native_file", session_dir, item_path, errors)
            if resolved is not None:
                normalized = resolved.relative_to(session_dir.resolve()).as_posix()
                decoded_references.add(normalized)
                if not resolved.is_file():
                    _error(errors, "MISSING_DECODED_NATIVE_FRAME", f"{item_path}:decoded_native_file", f"Referenced decoded native artifact does not exist: {decoded_file}.")
                elif frame.get("decoded_native_sha256") and _sha256(resolved) != str(frame["decoded_native_sha256"]).lower():
                    _error(errors, "DECODED_FRAME_CHECKSUM_MISMATCH", f"{item_path}:decoded_native_sha256", "Frame decoded_native_sha256 does not match the referenced file.")
        native_shape = frame.get("native_shape")
        expected_shape = [sensor.get("native_height"), sensor.get("native_width")]
        if isinstance(native_shape, list) and expected_shape[0] is not None and expected_shape[1] is not None and native_shape != expected_shape:
            _error(errors, "NATIVE_SHAPE_MISMATCH", f"{item_path}:native_shape", f"Frame shape {native_shape!r} differs from session native shape {expected_shape!r}.")
        if frame.get("validity_status") != "VALID" and not frame.get("exclude_reason"):
            _warning(warnings, "INVALID_FRAME_REASON_MISSING", f"{item_path}:exclude_reason", "Invalid or incomplete frames should retain an explicit exclude_reason.")
    if raw_root and raw_root.is_dir():
        for path in sorted(raw_root.rglob("*")):
            if path.is_file():
                relative = path.relative_to(session_dir.resolve()).as_posix()
                if relative not in raw_references:
                    _error(errors, "EXTRA_UNREGISTERED_RAW_FILE", relative, "Raw file exists but is not registered in frames.jsonl.")
    elif raw_root:
        _error(errors, "RAW_ROOT_MISSING", "session.json:storage:raw_root", "The raw root directory does not exist.")
    if decoded_root and not decoded_root.is_dir():
        _warning(warnings, "DECODED_NATIVE_ROOT_MISSING", "session.json:storage:decoded_native_root", "No decoded_native directory is present; this may be valid for packet-only evidence.")
    elif decoded_root and decoded_root.is_dir():
        for path in sorted(decoded_root.rglob("*")):
            if path.is_file():
                relative = path.relative_to(session_dir.resolve()).as_posix()
                if relative not in decoded_references:
                    _error(errors, "EXTRA_UNREGISTERED_DECODED_FILE", relative, "Decoded-native file exists but is not registered in frames.jsonl.")
    quality = session.get("quality") if isinstance(session.get("quality"), dict) else {}
    if quality.get("received_frame_count") != len(frames):
        _error(errors, "FRAME_COUNT_INCONSISTENCY", "session.json:quality:received_frame_count", "received_frame_count does not equal the number of frames.jsonl records.")
    expected_count = quality.get("expected_frame_count")
    if isinstance(expected_count, int) and expected_count != len(frames):
        _warning(warnings, "EXPECTED_FRAME_COUNT_DIFFERENCE", "session.json:quality:expected_frame_count", f"Expected {expected_count} frame(s), received {len(frames)}; the discrepancy is retained as evidence.")
    transport_metrics = quality.get("transport_metrics") if isinstance(quality.get("transport_metrics"), dict) else {}
    if framed_udp_v2:
        integrity_failures = {
            field: int(transport_metrics.get(field, 0) or 0)
            for field in (
                "invalid_datagrams",
                "incomplete_frames",
                "conflicting_duplicates",
                "reconstruction_timeouts",
                "pending_limit_evictions",
                "checksum_failures",
            )
        }
        nonzero_failures = {field: value for field, value in integrity_failures.items() if value != 0}
        if nonzero_failures:
            _error(
                errors,
                "FRAMED_UDP_V2_INTEGRITY_FAILED",
                "session.json:quality:transport_metrics",
                f"Framed UDP V2 fail-closed metrics contain errors: {nonzero_failures!r}.",
            )
        if int(transport_metrics.get("duplicate_chunks", 0) or 0) > 0:
            _warning(warnings, "UDP_DUPLICATE_CHUNKS", "session.json:quality:transport_metrics", "Duplicate chunks were ignored without corrupting a frame.")
        if int(transport_metrics.get("out_of_order_chunks", 0) or 0) > 0:
            _warning(warnings, "UDP_OUT_OF_ORDER_CHUNKS", "session.json:quality:transport_metrics", "Out-of-order chunks were reassembled by explicit chunk index.")
    sequence_summary = _series_checks(frames, "sequence_index", "frames.jsonl", errors, warnings)
    counter_summary = _series_checks(frames, "sensor_frame_counter", "frames.jsonl", errors, warnings)
    timing_summaries = {
        "sensor_timestamp": _timestamp_checks(frames, "sensor_timestamp", "sensor_timestamp_unit", "frames.jsonl", errors, warnings),
        "device_monotonic_timestamp_ns": _timestamp_checks(frames, "device_monotonic_timestamp_ns", None, "frames.jsonl", errors, warnings),
        "host_receive_monotonic_timestamp_ns": _timestamp_checks(frames, "host_receive_monotonic_timestamp_ns", None, "frames.jsonl", errors, warnings),
        "host_wall_time": _timestamp_checks(frames, "host_wall_time", None, "frames.jsonl", errors, warnings),
    }
    measured_fps, fps_source = _effective_fps(frames)
    configured_fps = sensor.get("configured_fps") if isinstance(sensor, dict) else None
    if isinstance(configured_fps, (int, float)) and measured_fps and abs(measured_fps - float(configured_fps)) / float(configured_fps) > 0.25:
        _warning(warnings, "CONFIGURED_EFFECTIVE_FPS_DIFFERENCE", "session.json:sensor:configured_fps", f"Configured FPS {configured_fps} differs materially from measured {measured_fps:.6f}; no automatic failure is applied.")
    annotation_summary = _validate_annotations(annotations, frames, session, "annotations.jsonl", errors, warnings)
    valid_frames = [frame for frame in frames if frame.get("validity_status") == "VALID"]
    valid_count = len(valid_frames)
    invalid_count = len(frames) - valid_count
    if framed_udp_v2 and int(transport_metrics.get("completed_frames", -1)) != valid_count:
        _error(
            errors,
            "FRAMED_UDP_V2_COMPLETED_COUNT_MISMATCH",
            "session.json:quality:transport_metrics:completed_frames",
            "Completed transport frame count must equal the number of valid persisted frames.",
        )
    sequence_verified = bool(valid_frames) and all(
        frame.get("sequence_index") is not None and frame.get("sequence_index_status") == "VERIFIED" for frame in valid_frames
    )
    counter_verified = bool(valid_frames) and all(
        frame.get("sensor_frame_counter") is not None and frame.get("sensor_frame_counter_status") == "VERIFIED" for frame in valid_frames
    )
    order_only = bool(valid_frames) and all(
        frame.get("sequence_index") is not None and frame.get("sequence_index_status") in {"VERIFIED", "RECEIVED_ORDER_ONLY"}
        for frame in valid_frames
    )
    timestamp_verified = bool(valid_frames) and all(
        frame.get("host_receive_monotonic_timestamp_ns") is not None
        or (frame.get("sensor_timestamp") is not None and str(frame.get("sensor_timestamp_status", "")).upper() == "VERIFIED")
        for frame in valid_frames
    )
    no_sequence_gaps = sequence_summary["gap_count"] == 0 and counter_summary["gap_count"] == 0
    continuous_session = (session.get("timing") or {}).get("continuous_session") is True
    complete_event = bool(annotation_summary["complete_temporal_event_ids"])
    if (
        complete_event
        and annotation_summary.get("event_range_integrity_valid", False)
        and (sequence_verified or counter_verified)
        and timestamp_verified
        and continuous_session
        and no_sequence_gaps
    ):
        temporal_status = "TEMPORAL_PROVENANCE_VERIFIED"
    elif order_only:
        temporal_status = "TEMPORAL_ORDER_ONLY"
    else:
        temporal_status = "TEMPORAL_PROVENANCE_INSUFFICIENT"
    temporal_claim = session.get("temporal_evidence_claim") if isinstance(session.get("temporal_evidence_claim"), dict) else {}
    claim_source = str(temporal_claim.get("source", "")).upper()
    if temporal_claim.get("filename_order_used_as_time") is True or "FILENAME" in claim_source or "FILE_ORDER" in claim_source:
        _error(errors, "FILENAME_ORDER_NOT_TEMPORAL", "session.json:temporal_evidence_claim", "Filename or filesystem order cannot establish chronology.")
    if temporal_claim.get("claimed_status") == "TEMPORAL_PROVENANCE_VERIFIED" and temporal_status != "TEMPORAL_PROVENANCE_VERIFIED":
        _error(errors, "TEMPORAL_CLAIM_NOT_SUPPORTED", "session.json:temporal_evidence_claim:claimed_status", "Claimed temporal provenance is not supported by actual frame evidence.")
    if temporal_status != "TEMPORAL_PROVENANCE_VERIFIED":
        _warning(warnings, "TEMPORAL_LIMITATION", "temporal_provenance_status", f"Session classified as {temporal_status}; do not construct a temporal event from filenames or posture alone.")
    session_root = session_dir.resolve()
    required_checksum_paths = {
        "session.json",
        frames_path.relative_to(session_root).as_posix() if frames_path else "frames.jsonl",
        annotations_path.relative_to(session_root).as_posix() if annotations_path else "annotations.jsonl",
        *raw_references,
        *decoded_references,
    }
    required_checksum_paths.discard("")
    checksum_status = _validate_checksums(session_dir, checksums_path, required_checksum_paths, errors)
    raw_errors = {
        "MISSING_RAW_FRAME",
        "RAW_FRAME_REFERENCE_MISSING",
        "RAW_FILE_NOT_ALLOWED_FOR_REPRESENTATION",
        "EXTRA_UNREGISTERED_RAW_FILE",
        "RAW_ROOT_MISSING",
        "RAW_FRAME_CHECKSUM_MISMATCH",
        "MISSING_DECODED_NATIVE_FRAME",
        "DECODED_NATIVE_REFERENCE_MISSING",
        "DECODED_NATIVE_FILE_NOT_ALLOWED_FOR_REPRESENTATION",
        "EXTRA_UNREGISTERED_DECODED_FILE",
        "DECODED_FRAME_CHECKSUM_MISMATCH",
    }
    raw_integrity_status = "FAIL" if any(item["code"] in raw_errors for item in errors) else "PASS_WITH_LIMITATIONS" if warnings else "PASS"
    raw_class = _raw_classification(representations)
    if raw_class == "PREPROCESSED_ONLY_INSUFFICIENT":
        _error(errors, "PREPROCESSED_ONLY_COLLECTION", "frames.jsonl:raw_representation", "Preprocessed tensors, screenshots, or normalized arrays are not sufficient raw evidence.")
    elif raw_class == "SCALAR_ONLY_LIMITED":
        _warning(warnings, "SCALAR_ONLY_FULL_FRAME_UNAVAILABLE", "frames.jsonl:raw_representation", "Scalar thermal_max_c-style evidence is limited to transport/runtime diagnostics and cannot validate or retrain the full-frame AI pipeline.")
    elif raw_class == "UNKNOWN_OR_EMPTY":
        _error(errors, "RAW_EVIDENCE_UNCLASSIFIED", "frames.jsonl:raw_representation", "No full-frame raw representation was classified.")
    role_block = session.get("role_governance") if isinstance(session.get("role_governance"), dict) else {}
    model_access_status = role_block.get("model_access_status")
    if model_access_status == "TRAINING_ALLOWED":
        _error(
            errors,
            "PRE_T_C_MODEL_ACCESS_FORBIDDEN",
            "session.json:role_governance:model_access_status",
            "Capture-contract v1 cannot grant training authority; use a later T-D promotion artifact.",
        )
    elif model_access_status not in VALID_MODEL_ACCESS_STATUSES:
        _error(errors, "MODEL_ACCESS_STATUS_INVALID", "session.json:role_governance:model_access_status", "Unknown model access status.")
    if role == "REAL_LOCKED_TEST":
        if role_block.get("model_access_status") != "UNTOUCHED" or role_block.get("locked_test_status") != "LOCKED_TEST_UNTOUCHED":
            _error(errors, "LOCKED_TEST_ACCESS_VIOLATION", "session.json:role_governance", "REAL_LOCKED_TEST must remain untouched by fitting, tuning, calibration, and debugging.")
    if collection and collection.get("collection_role") == "REAL_LOCKED_TEST" and role != "REAL_LOCKED_TEST":
        _error(errors, "LOCKED_TEST_ROLE_MISMATCH", "session.json:role_governance:role", "Session role must remain REAL_LOCKED_TEST inside a locked-test collection.")
    limitations: list[str] = []
    sensor_unit = sensor.get("unit_verification_status") if isinstance(sensor, dict) else None
    fps_status = sensor.get("verified_fps_status") if isinstance(sensor, dict) else None
    if sensor_unit != "VERIFIED":
        limitations.append("PHYSICAL_UNIT_NOT_VERIFIED")
    if fps_status != "VERIFIED":
        limitations.append("EFFECTIVE_FPS_NOT_VERIFIED")
    if temporal_status != "TEMPORAL_PROVENANCE_VERIFIED":
        limitations.append(temporal_status)
    if raw_class in {"SCALAR_ONLY_LIMITED", "MIXED_FULL_FRAME_AND_LIMITED", "PREPROCESSED_ONLY_INSUFFICIENT"}:
        limitations.append(raw_class)
    if warnings:
        limitations.append("WARNINGS_PRESENT")
    session_result = {
        "schema_version": "safenest.thermal.real_capture.validation_result.v1",
        "validator": VALIDATOR_NAME,
        "validated_path": _display_path(session_dir),
        "collection_id": session.get("collection_id"),
        "subject_id": session.get("subject_id"),
        "session_id": session.get("session_id"),
        "capture_status": "CAPTURE_INVALID" if errors else "CAPTURE_STRUCTURE_VALID_WITH_LIMITATIONS" if limitations else "CAPTURE_STRUCTURE_VALID",
        "raw_evidence_classification": raw_class,
        "temporal_provenance_status": temporal_status,
        "frame_count": len(frames),
        "valid_frame_count": valid_count,
        "invalid_frame_count": invalid_count,
        "timing_coverage": {
            "sequence_index": sequence_summary,
            "sensor_frame_counter": counter_summary,
            "clocks": timing_summaries,
            "effective_fps": measured_fps,
            "effective_fps_source": fps_source,
        },
        "packet_loss_summary": {
            "expected_frames": expected_count,
            "received_frames": len(frames),
            "sequence_gap_count": sequence_summary["gap_count"],
            "sensor_counter_gap_count": counter_summary["gap_count"],
            "duplicate_sensor_counter_count": counter_summary["duplicate_count"],
            "invalid_frame_count": invalid_count,
        },
        "annotation_coverage": annotation_summary,
        "raw_integrity_status": raw_integrity_status,
        "checksum_status": checksum_status,
        "limitations": sorted(set(limitations)),
        "model_use_eligibility": "NOT_AUTHORIZED_BY_CAPTURE_VALIDATOR",
        "errors": errors,
        "warnings": warnings,
    }
    inherited_errors.extend({**item, "session_id": session.get("session_id")} for item in errors)
    inherited_warnings.extend({**item, "session_id": session.get("session_id")} for item in warnings)
    return session_result


def _find_collection_root(path: Path) -> Path | None:
    current = path.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "collection.json").is_file():
            return candidate
    return None


def _role_leakage(
    sessions: list[dict[str, Any]],
    session_manifests: list[dict[str, Any]],
    errors: list[dict[str, str]],
) -> None:
    subject_roles: defaultdict[str, set[str]] = defaultdict(set)
    session_roles: defaultdict[str, set[str]] = defaultdict(set)
    event_roles: defaultdict[str, set[str]] = defaultdict(set)
    for result, manifest in zip(sessions, session_manifests):
        role = ((manifest.get("role_governance") or {}).get("role"))
        subject = manifest.get("subject_id")
        session_id = manifest.get("session_id")
        if _is_nonempty_string(subject) and _is_nonempty_string(role):
            subject_roles[subject].add(role)
        if _is_nonempty_string(session_id) and _is_nonempty_string(role):
            session_roles[session_id].add(role)
        # Event IDs are collected from the validator's annotation coverage so a
        # role cannot be changed merely by moving frame rows.
        for event_id in result.get("annotation_coverage", {}).get("event_ids", []):
            event_roles[event_id].add(role)
    for subject, roles in sorted(subject_roles.items()):
        if len(roles) > 1:
            _error(errors, "SUBJECT_ROLE_LEAKAGE", f"subject:{subject}", f"Subject appears in multiple roles: {sorted(roles)}.")
    for session_id, roles in sorted(session_roles.items()):
        if len(roles) > 1:
            _error(errors, "SESSION_ROLE_LEAKAGE", f"session:{session_id}", f"Session appears in multiple roles: {sorted(roles)}.")
    for event_id, roles in sorted(event_roles.items()):
        if len(roles) > 1:
            _error(errors, "EVENT_ROLE_LEAKAGE", f"event:{event_id}", f"Event appears in multiple roles: {sorted(roles)}.")


def validate_capture(capture_path: str | Path) -> dict[str, Any]:
    target = Path(capture_path)
    if not target.exists():
        return {
            "schema_version": "safenest.thermal.real_capture.validation_result.v1",
            "validator": VALIDATOR_NAME,
            "validated_path": _display_path(target),
            "capture_status": "CAPTURE_INVALID",
            "raw_evidence_classification": "UNKNOWN_OR_EMPTY",
            "temporal_provenance_status": "TEMPORAL_PROVENANCE_INSUFFICIENT",
            "frame_count": 0,
            "valid_frame_count": 0,
            "invalid_frame_count": 0,
            "timing_coverage": {},
            "packet_loss_summary": {},
            "annotation_coverage": {},
            "raw_integrity_status": "NOT_ASSESSED",
            "checksum_status": "NOT_ASSESSED",
            "limitations": [],
            "model_use_eligibility": "NOT_AUTHORIZED_BY_CAPTURE_VALIDATOR",
            "errors": [{"code": "CAPTURE_PATH_MISSING", "path": _display_path(target), "message": "Capture path does not exist."}],
            "warnings": [],
        }
    if target.is_file():
        if target.name == "collection.json":
            root = target.parent
            session_paths = sorted(root.rglob("session.json"))
        elif target.name == "session.json":
            root = _find_collection_root(target.parent) or target.parent
            session_paths = [target]
        else:
            root = target.parent
            session_paths = []
    else:
        root = target
        if (root / "collection.json").is_file():
            session_paths = sorted(root.rglob("session.json"))
        elif (root / "session.json").is_file():
            session_paths = [root / "session.json"]
        else:
            session_paths = []
    single_session_input = (
        (target.is_file() and target.name == "session.json")
        or (target.is_dir() and (target / "session.json").is_file() and not (target / "collection.json").is_file())
    )
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    collection: dict[str, Any] | None = None
    collection_path = root / "collection.json"
    if collection_path.is_file():
        loaded = _load_json(collection_path, errors, "collection.json")
        if isinstance(loaded, dict):
            collection = loaded
            _validate_collection_manifest(collection, "collection.json", errors, warnings)
    elif len(session_paths) != 1:
        _error(errors, "COLLECTION_MANIFEST_MISSING", _display_path(root), "Provide collection.json or a single session directory.")
    if not session_paths:
        _error(errors, "SESSION_MANIFEST_MISSING", _display_path(root), "No session.json was found.")
    session_results: list[dict[str, Any]] = []
    session_manifests: list[dict[str, Any]] = []
    for session_path in session_paths:
        session_errors: list[dict[str, str]] = []
        session_manifest = _load_json(session_path, session_errors, "session.json")
        if not isinstance(session_manifest, dict):
            session_manifest = {}
        session_manifests.append(session_manifest)
        session_result = _validate_session(session_path, collection, errors, warnings)
        # _validate_session already copied its errors to aggregate lists; add
        # manifest load errors that happened before it was called.
        if session_errors:
            errors.extend(session_errors)
            session_result["errors"].extend(session_errors)
            session_result["capture_status"] = "CAPTURE_INVALID"
        session_results.append(session_result)
    if collection is not None and not single_session_input:
        declared_subject_values = collection.get("subject_ids", [])
        declared_session_values = collection.get("session_ids", [])
        declared_subject_ids = {
            value for value in declared_subject_values if _is_nonempty_string(value)
        } if isinstance(declared_subject_values, list) else set()
        declared_session_ids = {
            value for value in declared_session_values if _is_nonempty_string(value)
        } if isinstance(declared_session_values, list) else set()
        actual_subject_ids = {
            manifest.get("subject_id") for manifest in session_manifests if _is_nonempty_string(manifest.get("subject_id"))
        }
        actual_session_ids = {
            manifest.get("session_id") for manifest in session_manifests if _is_nonempty_string(manifest.get("session_id"))
        }
        if len(actual_session_ids) != len(session_manifests):
            _error(errors, "DUPLICATE_DISCOVERED_SESSION_ID", "session.json:session_id", "Discovered session.json files contain duplicate or missing session IDs.")
        if declared_subject_ids != actual_subject_ids:
            _error(
                errors,
                "COLLECTION_SUBJECT_INVENTORY_MISMATCH",
                "collection.json:subject_ids",
                f"Declared subject_ids do not exactly match discovered session manifests (declared={sorted(declared_subject_ids)}, actual={sorted(actual_subject_ids)}).",
            )
        if declared_session_ids != actual_session_ids:
            _error(
                errors,
                "COLLECTION_SESSION_INVENTORY_MISMATCH",
                "collection.json:session_ids",
                f"Declared session_ids do not exactly match discovered session manifests (declared={sorted(declared_session_ids)}, actual={sorted(actual_session_ids)}).",
            )
    _role_leakage(session_results, session_manifests, errors)
    if not session_results:
        raw_class = "UNKNOWN_OR_EMPTY"
        temporal_status = "TEMPORAL_PROVENANCE_INSUFFICIENT"
    else:
        raw_values = {result["raw_evidence_classification"] for result in session_results}
        raw_class = next(iter(raw_values)) if len(raw_values) == 1 else "MIXED_FULL_FRAME_AND_LIMITED"
        temporal_values = {result["temporal_provenance_status"] for result in session_results}
        if len(temporal_values) == 1:
            temporal_status = next(iter(temporal_values))
        elif "TEMPORAL_PROVENANCE_VERIFIED" in temporal_values:
            temporal_status = "TEMPORAL_PROVENANCE_MIXED"
        else:
            temporal_status = "TEMPORAL_PROVENANCE_MIXED"
    frame_count = sum(result.get("frame_count", 0) for result in session_results)
    valid_count = sum(result.get("valid_frame_count", 0) for result in session_results)
    invalid_count = sum(result.get("invalid_frame_count", 0) for result in session_results)
    limitations = sorted({limitation for result in session_results for limitation in result.get("limitations", [])})
    if collection and collection.get("source", {}).get("source_status") == "SYNTHETIC_EXAMPLE_NOT_MEASUREMENT":
        limitations.append("SYNTHETIC_EXAMPLE_NOT_MEASUREMENT")
    if warnings:
        limitations.append("WARNINGS_PRESENT")
    raw_statuses = {result.get("raw_integrity_status") for result in session_results}
    checksum_statuses = {result.get("checksum_status") for result in session_results}
    result = {
        "schema_version": "safenest.thermal.real_capture.validation_result.v1",
        "validator": VALIDATOR_NAME,
        "validated_path": _display_path(target),
        "collection_id": collection.get("collection_id") if collection else (session_manifests[0].get("collection_id") if session_manifests else None),
        "subject_id": session_manifests[0].get("subject_id") if len(session_manifests) == 1 else None,
        "session_id": session_manifests[0].get("session_id") if len(session_manifests) == 1 else None,
        "capture_status": "CAPTURE_INVALID" if errors or any(result.get("capture_status") == "CAPTURE_INVALID" for result in session_results) else "CAPTURE_STRUCTURE_VALID_WITH_LIMITATIONS" if limitations else "CAPTURE_STRUCTURE_VALID",
        "raw_evidence_classification": raw_class,
        "temporal_provenance_status": temporal_status,
        "frame_count": frame_count,
        "valid_frame_count": valid_count,
        "invalid_frame_count": invalid_count,
        "timing_coverage": {
            "session_count": len(session_results),
            "sessions": [
                {
                    "session_id": item.get("session_id"),
                    "effective_fps": item.get("timing_coverage", {}).get("effective_fps"),
                    "effective_fps_source": item.get("timing_coverage", {}).get("effective_fps_source"),
                }
                for item in session_results
            ],
        },
        "packet_loss_summary": {
            "session_count": len(session_results),
            "sequence_gap_count": sum(item.get("packet_loss_summary", {}).get("sequence_gap_count", 0) or 0 for item in session_results),
            "sensor_counter_gap_count": sum(item.get("packet_loss_summary", {}).get("sensor_counter_gap_count", 0) or 0 for item in session_results),
        },
        "annotation_coverage": {
            "session_count": len(session_results),
            "sessions": [
                {
                    "session_id": item.get("session_id"),
                    "coverage_ratio": item.get("annotation_coverage", {}).get("coverage_ratio", 0.0),
                    "event_ids": item.get("annotation_coverage", {}).get("event_ids", []),
                }
                for item in session_results
            ],
        },
        "raw_integrity_status": "FAIL" if "FAIL" in raw_statuses else "PASS_WITH_LIMITATIONS" if "PASS_WITH_LIMITATIONS" in raw_statuses else "PASS" if raw_statuses else "NOT_ASSESSED",
        "checksum_status": "FAIL" if "FAIL" in checksum_statuses else "PASS" if checksum_statuses and checksum_statuses == {"PASS"} else "NOT_ASSESSED",
        "limitations": sorted(set(limitations)),
        "model_use_eligibility": "NOT_AUTHORIZED_BY_CAPTURE_VALIDATOR",
        "errors": errors,
        "warnings": warnings,
        "sessions": session_results,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="Collection root, collection.json, or one session directory/session.json")
    parser.add_argument("--json-out", type=Path, help="Write the portable JSON validation result to this path")
    args = parser.parse_args(argv)
    result = validate_capture(args.capture)
    rendered = _canonical_json(result)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 1 if result.get("capture_status") == "CAPTURE_INVALID" else 0


if __name__ == "__main__":
    raise SystemExit(main())
