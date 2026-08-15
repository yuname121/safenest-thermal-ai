#!/usr/bin/env python3
"""Standalone validator for compact Thermal T-A0 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "datasets/thermal/manifests/T-A0_source_identity"

REQUIRED_JSON = [
    "candidate_registry.json",
    "limitations.json",
    "local_asset_registry.json",
    "model_artifact_audit.json",
    "processed_lineage.json",
    "selected_source_identity.json",
    "source_license_evidence.json",
]

REQUIRED_CANDIDATE_FIELDS = [
    "candidate_id", "official_dataset_name", "stable_identifier",
    "official_distribution_location", "original_publication", "license_terms",
    "research_use_permission", "model_training_permission",
    "redistribution_restrictions", "access_registration_requirements",
    "genuine_thermal_status", "rgb_colorized_only",
    "representation_classification", "sensor_model", "wavelength",
    "source_frame_shape", "source_orientation", "dtype", "bit_depth",
    "channels", "file_format", "frame_rate", "timestamp_availability",
    "subject_identifiers", "session_identifiers", "scene_identifiers",
    "sequence_identifiers", "event_identifiers", "camera_identifiers",
    "fall_labels", "fall_event_boundary_quality", "normal_activity_coverage",
    "hard_negative_coverage", "staged_vs_natural_fall_semantics",
    "subject_count", "session_count", "sequence_count", "event_count",
    "subject_wise_split_feasibility", "fallback_grouping_feasibility",
    "duplicate_near_duplicate_risk", "event_level_evaluation_compatibility",
    "approximate_download_storage_impact", "checksum_availability",
    "thermal44_relevance", "known_limitations", "materialization_state",
    "overall_status", "explicit_justification", "source_identity_status",
    "license_status", "inventory_status", "label_semantics_status",
    "grouping_status", "safe_reader_documentation_status",
    "official_source_or_limitation", "evidence_category",
]

REPRESENTATIONS = {
    "RADIOMETRIC_TEMPERATURE", "RAW_SENSOR_COUNTS",
    "THERMAL_NUMERIC_UNIT_UNKNOWN", "NORMALIZED_THERMAL",
    "THERMAL_GRAYSCALE_RENDERING", "THERMAL_COLORIZED_RENDERING", "DEPTH",
    "RGB_PHOTOGRAPH", "MULTIMODAL", "UNKNOWN",
}

ALLOWED_CANDIDATE_STATUS = {
    "SELECTED", "ACCEPTABLE_BACKUP", "REJECTED_LICENSE",
    "REJECTED_PROVENANCE", "REJECTED_MODALITY", "REJECTED_GROUPING",
    "REJECTED_LABEL_QUALITY", "ACCESS_BLOCKED", "NEEDS_MANUAL_REVIEW",
}

SELECTED_REQUIREMENT_VALUES = {
    "source_identity_status": {"VERIFIED"},
    "license_status": {"VERIFIED_ACCEPTABLE", "VERIFIED_ACCEPTABLE_WITH_NONCOMMERCIAL_RESEARCH_RESTRICTION"},
    "inventory_status": {"DETERMINISTIC_INVENTORY", "DETERMINISTIC_INVENTORY_WITH_OFFICIAL_CHECKSUMS"},
    "label_semantics_status": {"USABLE", "USABLE_DERIVED_POST_FALL_POSTURE_PROXY"},
    "grouping_status": {"USABLE", "ACCEPTED_OFFICIAL_SPLIT_LIMITATION"},
    "safe_reader_documentation_status": {"DOCUMENTED"},
}


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def walk(value: Any, prefix: str = "$") -> Iterable[tuple[str, str, Any]]:
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}"
            yield child, key, value[key]
            yield from walk(value[key], child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk(item, f"{prefix}[{index}]")


def error(errors: list[dict[str, str]], code: str, location: str, message: str) -> None:
    errors.append({"code": code, "location": location, "message": message})


def validate_checksums(
    evidence_dir: Path,
    repo_root: Path,
    required_paths: list[Path],
    errors: list[dict[str, str]],
) -> None:
    checksum_file = evidence_dir / "checksums.sha256"
    if not checksum_file.is_file():
        error(errors, "CHECKSUM_REGISTRY_MISSING", "checksums.sha256", "Checksum registry is required.")
        return
    entries: dict[str, str] = {}
    for line_no, line in enumerate(checksum_file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", line)
        if not match:
            error(errors, "CHECKSUM_LINE_INVALID", f"checksums.sha256:{line_no}", "Expected sha256, two spaces, and repository-relative path.")
            continue
        digest, rel = match.groups()
        if rel in entries:
            error(errors, "CHECKSUM_PATH_DUPLICATE", rel, "Duplicate checksum path.")
        entries[rel] = digest
    for path in required_paths:
        try:
            rel = path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            error(errors, "CHECKSUM_PATH_OUTSIDE_ROOT", str(path), "Required artifact is outside the repository root.")
            continue
        if rel not in entries:
            error(errors, "CHECKSUM_COVERAGE_MISSING", rel, "Required machine-readable artifact lacks checksum coverage.")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if entries[rel] != actual:
            error(errors, "CHECKSUM_MISMATCH", rel, f"Expected {entries[rel]}, measured {actual}.")


def validate_evidence(evidence_dir: Path, repo_root: Path) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    docs: dict[str, Any] = {}
    paths: list[Path] = []

    for name in REQUIRED_JSON:
        path = evidence_dir / name
        if not path.is_file():
            error(errors, "REQUIRED_ARTIFACT_MISSING", name, "Required T-A0 artifact is missing.")
            continue
        try:
            data = load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            error(errors, "JSON_READ_FAILED", name, str(exc))
            continue
        docs[name] = data
        paths.append(path)
        if path.read_text(encoding="utf-8") != canonical_json(data):
            error(errors, "NONDETERMINISTIC_JSON_ORDER", name, "JSON must use canonical sorted-key formatting.")

    if set(REQUIRED_JSON) - set(docs):
        return {
            "schema_version": "1.0", "phase": "T-A0",
            "evidence_validation": "FAIL", "overall_outcome": "NOT_VERIFIABLE",
            "error_count": len(errors), "warning_count": 0,
            "errors": sorted(errors, key=lambda x: (x["code"], x["location"])),
            "warnings": [],
        }

    prior_result_path = evidence_dir / "validation_result.json"
    if prior_result_path.is_file():
        try:
            prior_result = load_json(prior_result_path)
            if prior_result_path.read_text(encoding="utf-8") != canonical_json(prior_result):
                error(errors, "NONDETERMINISTIC_JSON_ORDER", "validation_result.json", "JSON must use canonical sorted-key formatting.")
            paths.append(prior_result_path)
        except (OSError, json.JSONDecodeError) as exc:
            error(errors, "JSON_READ_FAILED", "validation_result.json", str(exc))

    registry = docs["candidate_registry.json"]
    candidates = registry.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        error(errors, "CANDIDATE_REGISTRY_EMPTY", "candidate_registry.json", "At least one candidate is required.")
        candidates = []
    ids = [item.get("candidate_id") for item in candidates if isinstance(item, dict)]
    if ids != sorted(ids):
        error(errors, "CANDIDATE_ORDER_NONDETERMINISTIC", "candidate_registry.json:candidates", "Candidates must be sorted by candidate_id.")
    if len(ids) != len(set(ids)):
        error(errors, "CANDIDATE_ID_DUPLICATE", "candidate_registry.json:candidates", "Candidate IDs must be unique.")

    candidate_by_id: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(candidates):
        loc = f"candidate_registry.json:candidates[{index}]"
        if not isinstance(item, dict):
            error(errors, "CANDIDATE_NOT_OBJECT", loc, "Candidate must be an object.")
            continue
        candidate_id = item.get("candidate_id", f"index-{index}")
        candidate_by_id[str(candidate_id)] = item
        for field in REQUIRED_CANDIDATE_FIELDS:
            if field not in item:
                error(errors, "CANDIDATE_FIELD_MISSING", f"{loc}.{field}", "Required candidate field is absent.")
        if item.get("overall_status") not in ALLOWED_CANDIDATE_STATUS:
            error(errors, "CANDIDATE_STATUS_INVALID", f"{loc}.overall_status", "Unsupported candidate status.")
        reason = item.get("explicit_justification")
        if not isinstance(reason, str) or not reason.strip():
            error(errors, "CANDIDATE_REASON_MISSING", f"{loc}.explicit_justification", "Every candidate status needs a reason.")
        official = item.get("official_distribution_location")
        limitation = item.get("official_source_or_limitation")
        if not official and (not isinstance(limitation, str) or not limitation.strip()):
            error(errors, "OFFICIAL_SOURCE_OR_LIMITATION_MISSING", loc, "Provide an official source or an explicit evidence limitation.")
        for field in ("license_terms", "license_status", "access_registration_requirements"):
            if field not in item or item.get(field) in (None, ""):
                error(errors, "LICENSE_ACCESS_STATUS_MISSING", f"{loc}.{field}", "License/access status must be explicit.")
        representation = item.get("representation_classification")
        if representation not in REPRESENTATIONS:
            error(errors, "REPRESENTATION_INVALID", f"{loc}.representation_classification", "Use an approved explicit representation classification.")
        if item.get("channels") in (3, 4, [3, 4]) and representation == "RADIOMETRIC_TEMPERATURE":
            error(errors, "RGB_FALSELY_RADIOMETRIC", loc, "RGB/RGBA rendering cannot be labeled radiometric without separate numeric evidence.")
        if item.get("rgb_colorized_only") is True and representation in {"RADIOMETRIC_TEMPERATURE", "RAW_SENSOR_COUNTS"}:
            error(errors, "RGB_FALSELY_RADIOMETRIC", loc, "Colorized-only data cannot be raw/radiometric.")
        if "grouping_status" not in item or item.get("grouping_status") in (None, ""):
            error(errors, "GROUPING_STATUS_MISSING", f"{loc}.grouping_status", "Grouping metadata availability must be recorded.")
        if "label_semantics_status" not in item or item.get("label_semantics_status") in (None, ""):
            error(errors, "LABEL_SEMANTICS_MISSING", f"{loc}.label_semantics_status", "Label semantics status must be recorded.")
        if item.get("overall_status") == "SELECTED":
            for field, accepted in SELECTED_REQUIREMENT_VALUES.items():
                if item.get(field) not in accepted:
                    expected = ", ".join(sorted(accepted))
                    error(errors, "SELECTED_CANDIDATE_REQUIREMENT_FAILED", f"{loc}.{field}", f"Selected candidate requires one of: {expected}.")
            if representation == "UNKNOWN":
                error(errors, "SELECTED_CANDIDATE_REQUIREMENT_FAILED", f"{loc}.representation_classification", "Selected representation cannot be unknown.")
            if item.get("label_semantics_status") == "USABLE_DERIVED_POST_FALL_POSTURE_PROXY":
                role = str(item.get("safenest_sensor_role", ""))
                mapping = item.get("safenest_label_mapping", {})
                lying = mapping.get("0", {}) if isinstance(mapping, dict) else {}
                if "no single thermal frame confirms a fall event" not in role or lying.get("mapping_type") != "DERIVED_POST_FALL_POSTURE_PROXY":
                    error(errors, "POST_FALL_PROXY_GUARD_MISSING", loc, "A selected lying-posture proxy must explicitly prohibit single-frame fall confirmation and preserve its derived mapping type.")
            if item.get("grouping_status") == "ACCEPTED_OFFICIAL_SPLIT_LIMITATION" and "never perform a frame-random resplit" not in str(item.get("fallback_grouping_feasibility", "")):
                error(errors, "OFFICIAL_SPLIT_GUARD_MISSING", f"{loc}.fallback_grouping_feasibility", "An accepted official-split limitation must prohibit frame-random resplitting.")

    local = docs["local_asset_registry.json"]
    assets = local.get("assets")
    if not isinstance(assets, list) or not assets:
        error(errors, "LOCAL_ASSET_REGISTRY_EMPTY", "local_asset_registry.json", "Owner-local assets are required.")
        assets = []
    asset_ids = [item.get("asset_id") for item in assets if isinstance(item, dict)]
    if asset_ids != sorted(asset_ids):
        error(errors, "LOCAL_ASSET_ORDER_NONDETERMINISTIC", "local_asset_registry.json:assets", "Assets must be sorted by asset_id.")
    for index, item in enumerate(assets):
        loc = f"local_asset_registry.json:assets[{index}]"
        required_local = ["path", "observation_source", "existence", "git_visibility", "git_ignore_state", "materialization_state", "logical_size_bytes", "locally_readable_status", "inventory_summary", "representation_status", "source_identity_status", "license_status", "label_status", "grouping_status", "checksum_status", "warnings"]
        for field in required_local:
            if field not in item:
                error(errors, "LOCAL_ASSET_FIELD_MISSING", f"{loc}.{field}", "Required local asset field is absent.")
        observations = item.get("observation_source", [])
        if "OWNER_CONFIRMED_LOCAL_STATE" in observations and item.get("existence") == "ABSENT":
            error(errors, "OWNER_CONFIRMED_SOURCE_MISLABELED_ABSENT", loc, "Git-ignored owner-confirmed local state cannot be labeled absent without contrary local measurement.")
        materialization = str(item.get("materialization_state", ""))
        readable = str(item.get("locally_readable_status", ""))
        if "PLACEHOLDER" in materialization and "PLACEHOLDER" not in readable and "False" not in readable:
            error(errors, "CLOUD_PLACEHOLDER_NOT_DISTINGUISHED", loc, "Placeholder state and offline readability must be explicit.")

    limitations = docs["limitations.json"].get("limitations", [])
    limitation_ids = [item.get("id") for item in limitations if isinstance(item, dict)]
    if limitation_ids != sorted(limitation_ids):
        error(errors, "LIMITATION_ORDER_NONDETERMINISTIC", "limitations.json:limitations", "Limitations must be sorted by id.")

    selected = docs["selected_source_identity.json"]
    selected_id = selected.get("selected_candidate_id")
    authorized = selected.get("t_a1_authorized")
    selection_status = selected.get("selection_status")
    if selected_id is not None:
        item = candidate_by_id.get(str(selected_id))
        if item is None:
            error(errors, "SELECTED_CANDIDATE_UNKNOWN", "selected_source_identity.json:selected_candidate_id", "Selected candidate is absent from the registry.")
        elif item.get("overall_status") != "SELECTED":
            error(errors, "SELECTED_STATUS_MISMATCH", "selected_source_identity.json", "Selected registry candidate must have SELECTED status.")
        if authorized is not True:
            error(errors, "SELECTED_NOT_AUTHORIZED", "selected_source_identity.json:t_a1_authorized", "A valid selection must authorize T-A1.")
        if selection_status not in {"PASS", "PASS_WITH_LIMITATIONS"}:
            error(errors, "SELECTED_OUTCOME_INVALID", "selected_source_identity.json:selection_status", "A selected source requires PASS or PASS_WITH_LIMITATIONS.")
    else:
        if authorized is not False or selection_status not in {"BLOCKED", "NOT_VERIFIABLE"}:
            error(errors, "NO_SELECTION_OUTCOME_INVALID", "selected_source_identity.json", "No selection requires T-A1 false and BLOCKED/NOT_VERIFIABLE.")
    for forbidden in ("splits", "split_assignments", "train", "validation", "locked_test"):
        if forbidden in selected:
            error(errors, "T_A1_SPLIT_CREATED", f"selected_source_identity.json.{forbidden}", "T-A0 must not create a T-A1 split.")

    for name, data in docs.items():
        for location, key, value in walk(data):
            if isinstance(value, str):
                if value.startswith(("/Users/", "/home/", "~/", "file://")) or re.match(r"^[A-Za-z]:\\", value):
                    error(errors, "ABSOLUTE_PATH_LEAKAGE", f"{name}:{location}", "Tracked evidence contains a machine-specific path.")
                if key in {"path", "artifact_path", "model_path"} and value.startswith("archive/"):
                    error(errors, "ARCHIVE_TREATED_AS_ACTIVE", f"{name}:{location}", "Archive content cannot be an active source path.")
                if value in {"VERIFIED_EQUIVALENT", "THERMAL44_VALIDATED", "REAL_SENSOR_VALIDATED", "RASPBERRY_PI_VALIDATED"}:
                    error(errors, "UNSUPPORTED_THERMAL44_ASSERTION", f"{name}:{location}", "Unsupported hardware validation assertion.")
            if key.lower() in {"accuracy", "f1", "macro_f1", "precision", "recall", "auc", "latency_ms"}:
                error(errors, "MODEL_PERFORMANCE_CLAIM_INTRODUCED", f"{name}:{location}", "T-A0 evidence must not introduce model-performance claims.")

    validate_checksums(evidence_dir, repo_root, paths, errors)
    errors = sorted(errors, key=lambda x: (x["code"], x["location"], x["message"]))
    warnings = sorted(warnings, key=lambda x: (x["code"], x["location"], x["message"]))
    if errors:
        evidence_validation = "FAIL"
        overall = "NOT_VERIFIABLE"
    else:
        evidence_validation = "PASS"
        overall = selection_status if selection_status in {"BLOCKED", "NOT_VERIFIABLE"} else ("PASS_WITH_LIMITATIONS" if limitations else "PASS")
    return {
        "schema_version": "1.0",
        "phase": "T-A0",
        "evidence_validation": evidence_validation,
        "overall_outcome": overall,
        "t_a1_authorized": bool(authorized),
        "selected_candidate_id": selected_id,
        "candidate_count": len(candidates),
        "local_asset_count": len(assets),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_evidence(args.evidence_dir.resolve(), args.repo_root.resolve())
    output = args.output or args.evidence_dir / "validation_result.json"
    output.write_text(canonical_json(result), encoding="utf-8")
    print(canonical_json(result), end="")
    return 0 if result["evidence_validation"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
