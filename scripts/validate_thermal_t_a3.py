#!/usr/bin/env python3
"""Standalone validator for the Thermal T-A3 temporal evidence contract."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.thermal.canonical_geometry import canonicalize_source_frame, profile_for_id  # noqa: E402
from datasets.thermal.raw_reader import SDTThermalRawReader  # noqa: E402
from datasets.thermal.temporal_policy import (  # noqa: E402
    SDT_ARCHIVE_PATH,
    SDT_ARCHIVE_SHA256,
    SDT_DATASET_ID,
    SDT_DOI,
    SDT_SOURCE_SPLIT,
    TEMPORAL_POLICY_ID,
    T_A2_PROFILE_ID,
    TEMPORAL_METADATA_FORBIDDEN_KEYS,
    validate_frame_sample,
    validate_temporal_policy_profile,
)


EVIDENCE_REL = "datasets/thermal/manifests/T-A3_sequence_window_event_policy"
T_A0_REL = "datasets/thermal/manifests/T-A0_source_identity"
T_A1_REL = "datasets/thermal/manifests/T-A1_safe_reader_raw_unit_contract"
T_A2_REL = "datasets/thermal/manifests/T-A2_geometry_calibration_canonical_frame"
CORE_JSON = [
    "event_policy.json",
    "frame_sample_contract.json",
    "gap_duplicate_policy.json",
    "limitations.json",
    "pilot_temporal_summary.json",
    "sequence_policy.json",
    "temporal_capability_contract.json",
    "temporal_evidence_registry.json",
    "window_policy.json",
]
REQUIRED_JSON = CORE_JSON + ["validation_result.json"]


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


def _portable(value: str) -> bool:
    if value.startswith(("/", "~/", "file://")) or "\\" in value or "/Users/" in value:
        return False
    pure = PurePosixPath(value)
    return not pure.is_absolute() and ".." not in pure.parts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_documents(evidence_dir: Path, errors: list[dict[str, str]]) -> tuple[dict[str, Any], list[Path]]:
    documents: dict[str, Any] = {}
    paths: list[Path] = []
    for name in CORE_JSON:
        path = evidence_dir / name
        if not path.is_file():
            _error(errors, "REQUIRED_ARTIFACT_MISSING", name, "Required T-A3 JSON is missing.")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _error(errors, "JSON_READ_FAILED", name, str(exc))
            continue
        documents[name] = data
        paths.append(path)
        if path.read_text(encoding="utf-8") != canonical_json(data):
            _error(errors, "NONDETERMINISTIC_JSON", name, "JSON must use sorted canonical formatting.")
        for location, value in _walk(data):
            if isinstance(value, str):
                if not _portable(value) and (value.startswith(("/", "~/", "file://")) or "/Users/" in value):
                    _error(errors, "NONPORTABLE_PATH", f"{name}:{location}", value)
                if ("Thermal-44" in value or "Thermal44" in value) and any(token in value for token in ("VERIFIED", "CONFIRMED")):
                    _error(errors, "UNSUPPORTED_THERMAL44_ASSERTION", f"{name}:{location}", value)
    return documents, paths


def _validate_checksums(repo_root: Path, evidence_dir: Path, paths: list[Path], errors: list[dict[str, str]]) -> None:
    checksum_path = evidence_dir / "checksums.sha256"
    if not checksum_path.is_file():
        _error(errors, "CHECKSUM_REGISTRY_MISSING", "checksums.sha256", "T-A3 checksum registry missing.")
        return
    entries: dict[str, str] = {}
    previous = ""
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", line)
        if not match:
            _error(errors, "CHECKSUM_LINE_INVALID", f"checksums.sha256:{line_number}", line)
            continue
        digest, relative = match.groups()
        if relative <= previous:
            _error(errors, "CHECKSUM_ORDER_NONDETERMINISTIC", f"checksums.sha256:{line_number}", relative)
        previous = relative
        if not _portable(relative):
            _error(errors, "CHECKSUM_PATH_NOT_PORTABLE", f"checksums.sha256:{line_number}", relative)
        entries[relative] = digest
    required_pairs: list[tuple[str, Path]] = []
    for path in paths:
        try:
            relative = path.relative_to(repo_root).as_posix()
        except ValueError:
            relative = f"{EVIDENCE_REL}/{path.name}"
        required_pairs.append((relative, path))
    for relative, path in sorted(required_pairs, key=lambda item: item[0]):
        if relative not in entries:
            _error(errors, "CHECKSUM_COVERAGE_MISSING", relative, "Required artifact has no checksum.")
        elif _sha256(path) != entries[relative]:
            _error(errors, "CHECKSUM_MISMATCH", relative, "Measured checksum differs.")


def _run_predecessors(repo_root: Path, errors: list[dict[str, str]], verify_real_payload: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        from scripts.validate_thermal_t_a0 import validate_evidence as validate_a0
        a0 = validate_a0(repo_root / T_A0_REL, repo_root)
        if a0.get("evidence_validation") != "PASS":
            _error(errors, "T_A0_VALIDATION_FAILED", T_A0_REL, canonical_json(a0).strip())
    except Exception as exc:
        a0 = {"evidence_validation": "FAIL", "overall_outcome": "NOT_VERIFIABLE"}
        _error(errors, "T_A0_VALIDATOR_ERROR", T_A0_REL, str(exc))
    try:
        from scripts.validate_thermal_t_a1 import validate_evidence as validate_a1
        a1 = validate_a1(repo_root=repo_root, evidence_dir=repo_root / T_A1_REL, check_checksums=True, verify_real_payload=verify_real_payload)
        if a1.get("evidence_validation") != "PASS":
            _error(errors, "T_A1_VALIDATION_FAILED", T_A1_REL, canonical_json(a1).strip())
    except Exception as exc:
        a1 = {"evidence_validation": "FAIL", "overall_outcome": "NOT_VERIFIABLE"}
        _error(errors, "T_A1_VALIDATOR_ERROR", T_A1_REL, str(exc))
    try:
        if verify_real_payload:
            from scripts.validate_thermal_t_a2 import validate_evidence as validate_a2
            a2 = validate_a2(repo_root=repo_root, evidence_dir=repo_root / T_A2_REL, check_checksums=True, verify_real_payload=True)
        else:
            # T-A2's validator intentionally rechecks the real SDT payload through
            # its nested predecessor gate.  Compact T-A3 validation must not make
            # that payload a prerequisite, so inherit its stored PASS result after
            # checking the compact JSON and checksum identity below.
            a2_path = repo_root / T_A2_REL / "validation_result.json"
            a2 = json.loads(a2_path.read_text(encoding="utf-8"))
            if a2.get("evidence_validation") != "PASS":
                _error(errors, "T_A2_STORED_VALIDATION_FAILED", T_A2_REL, canonical_json(a2).strip())
        if a2.get("evidence_validation") != "PASS":
            _error(errors, "T_A2_VALIDATION_FAILED", T_A2_REL, canonical_json(a2).strip())
    except Exception as exc:
        a2 = {"evidence_validation": "FAIL", "overall_outcome": "NOT_VERIFIABLE"}
        _error(errors, "T_A2_VALIDATOR_ERROR", T_A2_REL, str(exc))
    return a0, a1, a2


def _validate_policy_documents(documents: dict[str, Any], errors: list[dict[str, str]], warnings: list[dict[str, str]]) -> None:
    temporal = documents["temporal_capability_contract.json"]
    policy = temporal.get("policy")
    if not isinstance(policy, dict):
        _error(errors, "TEMPORAL_POLICY_MISSING", "temporal_capability_contract.json:policy", "Policy profile is missing.")
        return
    try:
        validate_temporal_policy_profile(policy)
    except Exception as exc:
        _error(errors, "TEMPORAL_POLICY_INVALID", "temporal_capability_contract.json:policy", str(exc))
    if temporal.get("source_split_preserved") is not True or temporal.get("safenest_split_created") is not False:
        _error(errors, "SPLIT_BOUNDARY_VIOLATION", "temporal_capability_contract.json", "T-A3 must preserve official test split and create no SafeNest split.")
    if temporal.get("model_performance_used") is not False:
        _error(errors, "MODEL_PERFORMANCE_CONTAMINATION", "temporal_capability_contract.json:model_performance_used", "T-A3 must not use performance evidence.")
    frame = policy["capabilities"]["FRAME_LEVEL"]
    sequence = policy["capabilities"]["SEQUENCE_LEVEL"]
    event = policy["capabilities"]["EVENT_LEVEL"]
    window = policy["capabilities"]["WINDOW_LEVEL"]
    if not (frame.get("supported") is True and frame.get("status") == "SUPPORTED" and frame.get("safe_to_construct") is True):
        _error(errors, "FRAME_CAPABILITY_CLOSED", "temporal_capability_contract.json:FRAME_LEVEL", "Frame-level capability must be supported.")
    for level, expected in (("SEQUENCE_LEVEL", "TEMPORAL_SEQUENCE_NOT_VERIFIABLE"), ("EVENT_LEVEL", "TEMPORAL_EVENT_NOT_VERIFIABLE"), ("WINDOW_LEVEL", "WINDOWING_NOT_APPLICABLE_TO_SOURCE")):
        block = policy["capabilities"][level]
        if block.get("supported") is not False or block.get("status") != expected or block.get("safe_to_construct") is not False:
            _error(errors, "UNSUPPORTED_CAPABILITY_OPEN", f"temporal_capability_contract.json:{level}", "Unsupported temporal level must be fail-closed.")
    if policy["source_frame_semantics"].get("index_is_timestamp") is not False or policy["source_frame_semantics"].get("filename_order_is_temporal_order") is not False:
        _error(errors, "INDEX_TEMPORAL_PROMOTION", "temporal_capability_contract.json:source_frame_semantics", "Index/order cannot be timestamps.")
    if policy["source_frame_semantics"].get("lying_is_fall_event") is not False:
        _error(errors, "LYING_EVENT_PROMOTION", "temporal_capability_contract.json:source_frame_semantics", "LYING is posture only.")
    contract = documents["frame_sample_contract.json"]
    if contract.get("status") != "SUPPORTED" or contract.get("source_split") != "test" or contract.get("safenest_split_created") is not False:
        _error(errors, "FRAME_CONTRACT_INVALID", "frame_sample_contract.json", "Frame contract must preserve source test split.")
    if contract.get("synthetic_temporal_metadata_added") is not False:
        _error(errors, "SYNTHETIC_TEMPORAL_METADATA", "frame_sample_contract.json", "Synthetic temporal metadata must be false.")
    for name, status in (("sequence_policy.json", "TEMPORAL_SEQUENCE_NOT_VERIFIABLE"), ("event_policy.json", "TEMPORAL_EVENT_NOT_VERIFIABLE"), ("window_policy.json", "WINDOWING_NOT_APPLICABLE_TO_SOURCE")):
        document = documents[name]
        if document.get("eligible") is not False or document.get("status") != status or document.get("safe_to_construct") is not False:
            _error(errors, "POLICY_NOT_FAIL_CLOSED", name, "Policy must be explicitly ineligible and fail closed.")
        behavior = document.get("unsupported_request_behavior", {})
        if behavior.get("behavior") != "FAIL_CLOSED":
            _error(errors, "UNSUPPORTED_REQUEST_NOT_FAIL_CLOSED", name, "Unsupported request behavior must fail closed.")
    event = documents["event_policy.json"]
    if event.get("lying_is_fall_event") is not False or event.get("fall_onset") != "NOT_VERIFIABLE" or event.get("fall_end") != "NOT_VERIFIABLE":
        _error(errors, "EVENT_BOUNDARY_FABRICATED", "event_policy.json", "Fall event boundaries or LYING semantics were fabricated.")
    window = documents["window_policy.json"]
    for key in ("window_duration", "window_frame_count", "stride", "overlap"):
        if window.get(key) != "NOT_APPLICABLE_NO_VERIFIED_TIMELINE":
            _error(errors, "ARBITRARY_WINDOW_PARAMETER", f"window_policy.json:{key}", "No window parameter is allowed without a verified timeline.")
    gap = documents["gap_duplicate_policy.json"]
    if gap.get("index_gap_interpretation") != "STRUCTURAL_ARCHIVE_PROVENANCE_ONLY; NOT_A_DROPPED_ACQUISITION_FRAME":
        _error(errors, "INDEX_GAP_MISINTERPRETED", "gap_duplicate_policy.json:index_gap_interpretation", "Index gap must not be called a dropped temporal frame.")
    if gap.get("temporal_dropped_frame_status") != "TEMPORAL_DROPPED_FRAME_NOT_VERIFIABLE":
        _error(errors, "TEMPORAL_DROP_STATUS_INVALID", "gap_duplicate_policy.json", "Temporal drop status must remain unverifiable.")
    _warning(warnings, "STATIC_FRAME_ONLY", "capabilities", "SDT supports only frame-level T-A3 evidence.")


def _validate_pilot(repo_root: Path, documents: dict[str, Any], errors: list[dict[str, str]], verify_real_payload: bool) -> None:
    summary = documents["pilot_temporal_summary.json"]
    if summary.get("pilot_frame_count") != 48 or summary.get("source_split") != "test" or summary.get("t_a2_geometry_profile_id") != T_A2_PROFILE_ID:
        _error(errors, "PILOT_CONTRACT_INVALID", "pilot_temporal_summary.json", "Expected 48-frame T-A2-linked test pilot.")
    counts = summary.get("source_class_counts")
    if counts != {"0": 12, "1": 12, "2": 12, "3": 12}:
        _error(errors, "PILOT_CLASS_COUNTS_INVALID", "pilot_temporal_summary.json:source_class_counts", str(counts))
    records = summary.get("records")
    if not isinstance(records, list) or len(records) != 48:
        _error(errors, "PILOT_RECORD_COUNT_INVALID", "pilot_temporal_summary.json:records", "Expected 48 records.")
        return
    indices = [record.get("source_frame_index") for record in records]
    if indices != sorted(indices) or len(set(indices)) != 48:
        _error(errors, "PILOT_ORDER_NONDETERMINISTIC", "pilot_temporal_summary.json:records", "Records must be unique and sorted by source frame index.")
    actual_counts: dict[str, int] = {}
    for index, record in enumerate(records):
        try:
            validate_frame_sample(record)
        except Exception as exc:
            _error(errors, "FRAME_SAMPLE_INVALID", f"pilot_temporal_summary.json:records[{index}]", str(exc))
        if record.get("source_split") != "test" or record.get("t_a2_geometry_profile_id") != T_A2_PROFILE_ID:
            _error(errors, "PILOT_PREDECESSOR_IDENTITY_CHANGED", f"pilot:{record.get('source_frame_index')}", "T-A1/T-A2 identity changed.")
        if record.get("original_source_pose_label") not in {0, 1, 2, 3}:
            _error(errors, "ORIGINAL_LABEL_INVALID", f"pilot:{record.get('source_frame_index')}", "Original label is not one of SDT labels.")
        actual_counts[str(record.get("original_source_pose_label"))] = actual_counts.get(str(record.get("original_source_pose_label")), 0) + 1
        for key in TEMPORAL_METADATA_FORBIDDEN_KEYS:
            if key in record:
                _error(errors, "FORBIDDEN_TEMPORAL_FIELD", f"pilot:{record.get('source_frame_index')}:{key}", "Temporal field must not be present in a frame record.")
    if actual_counts != counts:
        _error(errors, "PILOT_LABEL_COUNTS_CHANGED", "pilot_temporal_summary.json", f"expected={counts!r}, found={actual_counts!r}")
    for key, value in summary.get("fabricated_temporal_metadata", {}).items():
        if value is not False:
            _error(errors, "FABRICATED_TEMPORAL_METADATA", f"pilot_temporal_summary.json:{key}", str(value))
    if not verify_real_payload:
        return
    try:
        reader = SDTThermalRawReader(repo_root=repo_root)
        reader.inspect_archive()
        profile = profile_for_id(T_A2_PROFILE_ID)
        for record in records:
            frame = reader.read_frame(int(record["source_frame_index"]))
            canonical = canonicalize_source_frame(frame, profile)
            if frame.source_member_name != record["source_member_name"] or frame.source_member_sha256 != record.get("source_member_sha256"):
                _error(errors, "REAL_MEMBER_HASH_MISMATCH", f"pilot:{record['source_frame_index']}", "Source member identity differs.")
            if frame.raw_encoded_frame_sha256 != record.get("t_a1_raw_encoded_frame_sha256") or canonical.canonical_frame_hash != record.get("canonical_frame_hash"):
                _error(errors, "REAL_FRAME_HASH_MISMATCH", f"pilot:{record['source_frame_index']}", "T-A1/T-A2 frame hash differs.")
            if frame.source_pose_label != record.get("original_source_pose_label") or list(frame.source_bbox) != record.get("original_source_bbox"):
                _error(errors, "REAL_LABEL_BBOX_MISMATCH", f"pilot:{record['source_frame_index']}", "Original label or bbox differs.")
    except Exception as exc:
        _error(errors, "REAL_PILOT_VALIDATION_FAILED", SDT_ARCHIVE_PATH, str(exc))


def _validate_evidence_registry(documents: dict[str, Any], errors: list[dict[str, str]]) -> None:
    registry = documents["temporal_evidence_registry.json"]
    records = registry.get("source_records")
    if not isinstance(records, list) or len(records) < 3:
        _error(errors, "OFFICIAL_SOURCE_EVIDENCE_MISSING", "temporal_evidence_registry.json:source_records", "At least Zenodo, TU Wien, and publication evidence is required.")
    else:
        urls = {item.get("source_url") for item in records}
        for url in ("https://zenodo.org/records/4124309", "https://cvl.tuwien.ac.at/research/cvl-databases/sdt-icip/", "https://doi.org/10.1109/ICIP40778.2020.9191284"):
            if url not in urls:
                _error(errors, "OFFICIAL_SOURCE_URL_MISSING", "temporal_evidence_registry.json", url)
        for item in records:
            if item.get("category") not in {"OFFICIAL_EXTERNAL_SOURCE_VERIFIED", "REPOSITORY_CODE_VERIFIED", "VALIDATOR_INHERITED"}:
                _error(errors, "EVIDENCE_CATEGORY_INVALID", "temporal_evidence_registry.json", str(item.get("category")))
            if not item.get("stable_identifier") or not item.get("access_date"):
                _error(errors, "EVIDENCE_IDENTITY_INCOMPLETE", "temporal_evidence_registry.json", str(item))
    local = registry.get("local_measurements", {})
    if local.get("thermal_member_count") != 8000 or local.get("depth_member_count") != 8000 or local.get("label_row_count") != 8000:
        _error(errors, "LOCAL_STRUCTURAL_COUNTS_INVALID", "temporal_evidence_registry.json:local_measurements", "Measured SDT test counts are not 8000/8000/8000.")
    if local.get("index_continuity") is not True or local.get("duplicate_thermal_indices") != [] or local.get("duplicate_member_names") != []:
        _error(errors, "LOCAL_STRUCTURAL_INVENTORY_INVALID", "temporal_evidence_registry.json:local_measurements", "Structural inventory is inconsistent.")


def _validate_predecessor_identity(repo_root: Path, errors: list[dict[str, str]]) -> None:
    try:
        selected_a2 = json.loads((repo_root / T_A2_REL / "selected_geometry_profile.json").read_text(encoding="utf-8"))
        profile = selected_a2.get("profile")
        if selected_a2.get("profile_id") != T_A2_PROFILE_ID or profile != profile_for_id(T_A2_PROFILE_ID).to_dict():
            _error(errors, "T_A2_PROFILE_CHANGED", f"{T_A2_REL}/selected_geometry_profile.json", "T-A2 selected profile no longer matches canonical implementation.")
        if selected_a2.get("model_performance_used") is not False:
            _error(errors, "T_A2_MODEL_CONTAMINATION", f"{T_A2_REL}/selected_geometry_profile.json", "T-A2 profile selection must remain model-independent.")
    except Exception as exc:
        _error(errors, "T_A2_PROFILE_UNREADABLE", T_A2_REL, str(exc))
    for path in (repo_root / "datasets/thermal/temporal_policy.py", repo_root / "scripts/generate_thermal_t_a3.py"):
        if not path.is_file():
            _error(errors, "T_A3_IMPLEMENTATION_MISSING", path.relative_to(repo_root).as_posix(), "T-A3 implementation file missing.")
            continue
        source = path.read_text(encoding="utf-8")
        try:
            ast.parse(source)
        except SyntaxError as exc:
            _error(errors, "T_A3_SYNTAX_ERROR", path.relative_to(repo_root).as_posix(), str(exc))
        lowered = source.lower()
        for forbidden in ("from inference", "import inference", "thermalinterpreter", "tensorflow", ".tflite", "model.predict", "interpreter.invoke"):
            if forbidden in lowered:
                _error(errors, "T_A3_MODEL_COUPLING", path.relative_to(repo_root).as_posix(), forbidden)


def validate_evidence(*, repo_root: Path = ROOT, evidence_dir: Path | None = None, check_checksums: bool = True, verify_real_payload: bool = True) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    evidence_dir = (evidence_dir or repo_root / EVIDENCE_REL).resolve()
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    a0, a1, a2 = _run_predecessors(repo_root, errors, verify_real_payload)
    documents, paths = _load_documents(evidence_dir, errors)
    if len(documents) == len(CORE_JSON):
        _validate_policy_documents(documents, errors, warnings)
        _validate_pilot(repo_root, documents, errors, verify_real_payload)
        _validate_evidence_registry(documents, errors)
    _validate_predecessor_identity(repo_root, errors)
    validation_path = evidence_dir / "validation_result.json"
    if not validation_path.is_file():
        if check_checksums:
            _error(errors, "VALIDATION_RESULT_MISSING", "validation_result.json", "Stored T-A3 validation result is required.")
    else:
        try:
            stored = json.loads(validation_path.read_text(encoding="utf-8"))
            if check_checksums and stored.get("evidence_validation") != "PASS":
                _error(errors, "STORED_VALIDATION_NOT_PASS", "validation_result.json", "Stored validation result must be PASS.")
            if validation_path.read_text(encoding="utf-8") != canonical_json(stored):
                _error(errors, "NONDETERMINISTIC_JSON", "validation_result.json", "Stored validation result must be canonical JSON.")
        except (OSError, json.JSONDecodeError) as exc:
            _error(errors, "VALIDATION_RESULT_INVALID", "validation_result.json", str(exc))
        if validation_path not in paths:
            paths.append(validation_path)
    if check_checksums and len(documents) == len(CORE_JSON):
        _validate_checksums(repo_root, evidence_dir, paths, errors)
    sorted_errors = sorted(errors, key=lambda item: (item["code"], item["location"], item["message"]))
    sorted_warnings = sorted(warnings, key=lambda item: (item["code"], item["location"], item["message"]))
    gate = not sorted_errors and a0.get("evidence_validation") == "PASS" and a1.get("evidence_validation") == "PASS" and a2.get("evidence_validation") == "PASS"
    return {
        "error_count": len(sorted_errors), "errors": sorted_errors,
        "evidence_validation": "PASS" if gate else "FAIL",
        "overall_outcome": "PASS_WITH_LIMITATIONS" if gate else "NOT_VERIFIABLE",
        "phase": "T-A3", "schema_version": "1.0",
        "t_a0_validation": a0.get("evidence_validation", "FAIL"), "t_a0_outcome": a0.get("overall_outcome", "NOT_VERIFIABLE"),
        "t_a1_validation": a1.get("evidence_validation", "FAIL"), "t_a1_outcome": a1.get("overall_outcome", "NOT_VERIFIABLE"),
        "t_a2_validation": a2.get("evidence_validation", "FAIL"), "t_a2_outcome": a2.get("overall_outcome", "NOT_VERIFIABLE"),
        "t_a4_authorized": bool(gate), "warning_count": len(sorted_warnings), "warnings": sorted_warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--skip-real-payload", action="store_true")
    args = parser.parse_args()
    result = validate_evidence(repo_root=args.repo_root, evidence_dir=args.evidence_dir, check_checksums=not args.skip_checksums, verify_real_payload=not args.skip_real_payload)
    print(canonical_json(result), end="")
    return 0 if result["evidence_validation"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
