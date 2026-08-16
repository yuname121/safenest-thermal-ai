#!/usr/bin/env python3
"""Standalone validator for the Thermal T-A4 semantic contract."""

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

from datasets.thermal.label_semantics import (  # noqa: E402
    CLAIM_SCOPES,
    DATASET_DOI,
    DATASET_ID,
    MAPPING_TYPES,
    RUNTIME_CLASS_MAP,
    SEMANTIC_POLICY_ID,
    SELECTION_POLICY_ID,
    SOURCE_ARCHIVE_PATH,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_LABELS,
    SOURCE_SPLIT,
    T_A2_PROFILE_ID,
    T_A3_POLICY_ID,
    candidate_policy_definitions,
    selection_policy_definition,
    validate_mapping_record,
)


EVIDENCE_REL = "datasets/thermal/manifests/T-A4_label_semantics_ambiguity"
T_A0_REL = "datasets/thermal/manifests/T-A0_source_identity"
T_A1_REL = "datasets/thermal/manifests/T-A1_safe_reader_raw_unit_contract"
T_A2_REL = "datasets/thermal/manifests/T-A2_geometry_calibration_canonical_frame"
T_A3_REL = "datasets/thermal/manifests/T-A3_sequence_window_event_policy"
CORE_JSON = [
    "activity_coverage_registry.json",
    "ambiguity_policy.json",
    "claim_scope_contract.json",
    "compatibility_mapping_contract.json",
    "frame_evidence_contract.json",
    "label_mapping_inventory.json",
    "label_semantic_policy.json",
    "limitations.json",
    "original_label_contract.json",
    "pilot_semantic_summary.json",
    "semantic_policy_candidates.json",
    "selected_semantic_policy.json",
]
REQUIRED_JSON = CORE_JSON + ["validation_result.json"]
MODEL_METRIC_KEYS = {"accuracy", "precision", "recall", "f1", "macro_f1", "confusion_matrix", "prediction_distribution", "loss", "auc"}
FORBIDDEN_SEMANTIC_TOKENS = {
    "VERIFIED_FALL_EVENT",
    "DIRECT_FALL_EVENT",
    "FALL_ONSET",
    "FALL_IMPACT",
    "POST_FALL_EVENT",
    "PRE_FALL_EVENT",
    "WORKER_SAFE",
    "GENERAL_WORKER_SAFETY",
    "MEDICAL_DIAGNOSIS",
    "EMERGENCY_CONFIRMED",
}


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
    if value.startswith(("/", "~/", "file://")) or "\\" in value or "/Users/" in value or "/private/tmp/" in value:
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
            _error(errors, "REQUIRED_ARTIFACT_MISSING", name, "Required T-A4 artifact is missing.")
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
                if not _portable(value) and (value.startswith(("/", "~/", "file://")) or "/Users/" in value or "/private/tmp/" in value):
                    _error(errors, "NONPORTABLE_PATH", f"{name}:{location}", value)
                token = value.upper().replace(" ", "_")
                if token in FORBIDDEN_SEMANTIC_TOKENS:
                    _error(errors, "UNSUPPORTED_SEMANTIC_ESCALATION", f"{name}:{location}", value)
                if ("Thermal-44" in value or "Thermal44" in value) and any(flag in value for flag in ("VERIFIED", "CONFIRMED")):
                    _error(errors, "UNSUPPORTED_THERMAL44_ASSERTION", f"{name}:{location}", value)
            if isinstance(value, dict):
                for key in value:
                    if str(key).lower() in MODEL_METRIC_KEYS:
                        _error(errors, "MODEL_METRIC_SEMANTIC_CONTAMINATION", f"{name}:{location}.{key}", "Model metrics cannot select or define semantic policy.")
    return documents, paths


def _validate_checksums(repo_root: Path, evidence_dir: Path, paths: list[Path], errors: list[dict[str, str]]) -> None:
    checksum_path = evidence_dir / "checksums.sha256"
    if not checksum_path.is_file():
        _error(errors, "CHECKSUM_REGISTRY_MISSING", "checksums.sha256", "T-A4 checksum registry missing.")
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
            _error(errors, "CHECKSUM_COVERAGE_MISSING", relative, "Required T-A4 artifact has no checksum.")
        elif _sha256(path) != entries[relative]:
            _error(errors, "CHECKSUM_MISMATCH", relative, "Measured checksum differs.")


def _run_predecessors(repo_root: Path, errors: list[dict[str, str]], verify_real_payload: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
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
            a2 = json.loads((repo_root / T_A2_REL / "validation_result.json").read_text(encoding="utf-8"))
        if a2.get("evidence_validation") != "PASS":
            _error(errors, "T_A2_VALIDATION_FAILED", T_A2_REL, canonical_json(a2).strip())
    except Exception as exc:
        a2 = {"evidence_validation": "FAIL", "overall_outcome": "NOT_VERIFIABLE"}
        _error(errors, "T_A2_VALIDATOR_ERROR", T_A2_REL, str(exc))
    try:
        from scripts.validate_thermal_t_a3 import validate_evidence as validate_a3
        a3 = validate_a3(repo_root=repo_root, evidence_dir=repo_root / T_A3_REL, check_checksums=True, verify_real_payload=verify_real_payload)
        if a3.get("evidence_validation") != "PASS":
            _error(errors, "T_A3_VALIDATION_FAILED", T_A3_REL, canonical_json(a3).strip())
    except Exception as exc:
        a3 = {"evidence_validation": "FAIL", "overall_outcome": "NOT_VERIFIABLE"}
        _error(errors, "T_A3_VALIDATOR_ERROR", T_A3_REL, str(exc))
    return a0, a1, a2, a3


def _validate_source_identity(repo_root: Path, documents: dict[str, Any], errors: list[dict[str, str]]) -> None:
    contract = documents["original_label_contract.json"]
    expected = {
        "dataset_id": DATASET_ID,
        "doi": DATASET_DOI,
        "source_split": SOURCE_SPLIT,
        "archive_path": SOURCE_ARCHIVE_PATH,
        "archive_sha256": SOURCE_ARCHIVE_SHA256,
        "original_labels": {str(key): value for key, value in sorted(SOURCE_LABELS.items())},
        "source_label_counts": {"0": 2000, "1": 2000, "2": 2000, "3": 2000},
        "source_labels_modified": False,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            _error(errors, "SOURCE_CONTRACT_MISMATCH", f"original_label_contract.json:{key}", f"expected={value!r}, found={contract.get(key)!r}")
    if contract.get("source_label_status") != "VERIFIED_IMMUTABLE_SOURCE_ANNOTATION":
        _error(errors, "SOURCE_LABEL_STATUS_INVALID", "original_label_contract.json:source_label_status", str(contract.get("source_label_status")))
    try:
        selected = json.loads((repo_root / "datasets/thermal/manifests/T-A0_source_identity/selected_source_identity.json").read_text(encoding="utf-8"))
        if selected.get("selected_candidate_id") != DATASET_ID or selected.get("stable_identifier") != DATASET_DOI:
            _error(errors, "T_A0_SOURCE_CHANGED", T_A0_REL, "T-A4 source differs from T-A0 selected source.")
    except Exception as exc:
        _error(errors, "T_A0_SOURCE_UNREADABLE", T_A0_REL, str(exc))


def _validate_candidate_selection(documents: dict[str, Any], errors: list[dict[str, str]]) -> str | None:
    registry = documents["semantic_policy_candidates.json"]
    policy = registry.get("selection_policy")
    if policy != selection_policy_definition():
        _error(errors, "SELECTION_POLICY_TAMPERED", "semantic_policy_candidates.json:selection_policy", "Declared selection policy differs from the phase policy definition.")
    if registry.get("selection_policy_content_sha256") != hashlib.sha256(canonical_json(policy).encode("utf-8")).hexdigest():
        _error(errors, "SELECTION_POLICY_CHECKSUM_INVALID", "semantic_policy_candidates.json", "Selection policy checksum is stale.")
    candidates = registry.get("predeclared_candidates")
    evaluated = registry.get("evaluated_candidates")
    canonical_candidates = candidate_policy_definitions()
    if not isinstance(candidates, list) or [item.get("candidate_id") for item in candidates] != sorted(item["candidate_id"] for item in canonical_candidates):
        _error(errors, "CANDIDATE_ORDER_INVALID", "semantic_policy_candidates.json:predeclared_candidates", "Candidates must be sorted and complete.")
        candidates = []
    canonical_by_id = {item["candidate_id"]: item for item in canonical_candidates}
    for candidate in candidates:
        if candidate != canonical_by_id.get(candidate.get("candidate_id")):
            _error(errors, "CANDIDATE_DEFINITION_TAMPERED", f"candidate:{candidate.get('candidate_id')}", "Candidate definition differs from the predeclared semantic alternatives.")
    # Independent admissibility and lexicographic ranking; winner is computed, not hardcoded.
    expected_gate_keys = tuple(policy.get("mandatory_admissibility_gates", {})) if isinstance(policy, dict) else ()
    recomputed: list[dict[str, Any]] = []
    for candidate in candidates:
        checks = {key: candidate.get(key) is True for key in expected_gate_keys if key != "no_model_metrics_used"}
        checks["no_model_metrics_used"] = candidate.get("model_metrics_used") is False
        reasons = sorted(key for key, passed in checks.items() if not passed)
        metrics = tuple(-int(candidate.get(key, 0)) for key in policy["ranking_order"] if key != "candidate_id") + (candidate["candidate_id"],)
        recomputed.append({"candidate_id": candidate["candidate_id"], "admissible": not reasons, "reasons": reasons, "rank": metrics})
    admissible = sorted((item for item in recomputed if item["admissible"]), key=lambda item: item["rank"])
    winner = admissible[0]["candidate_id"] if admissible else None
    if registry.get("selected_candidate_id") != winner:
        _error(errors, "SELECTION_WINNER_MISMATCH", "semantic_policy_candidates.json:selected_candidate_id", f"computed={winner!r}, recorded={registry.get('selected_candidate_id')!r}")
    if not isinstance(evaluated, list) or len(evaluated) != len(candidates):
        _error(errors, "EVALUATED_CANDIDATES_INVALID", "semantic_policy_candidates.json:evaluated_candidates", "Evaluation record count mismatch.")
    else:
        recorded = {item.get("candidate_id"): item for item in evaluated}
        for item in recomputed:
            row = recorded.get(item["candidate_id"])
            if row is None or row.get("admissible") != item["admissible"] or row.get("selected") != (item["candidate_id"] == winner):
                _error(errors, "CANDIDATE_EVALUATION_MISMATCH", f"candidate:{item['candidate_id']}", "Recorded admissibility/ranking differs from independent recomputation.")
    return winner


def _validate_semantic_layers(documents: dict[str, Any], errors: list[dict[str, str]], winner: str | None) -> None:
    profile = documents["label_semantic_policy.json"]
    selected = documents["selected_semantic_policy.json"]
    if profile.get("policy_id") != SEMANTIC_POLICY_ID or selected.get("policy_id") != SEMANTIC_POLICY_ID:
        _error(errors, "SEMANTIC_POLICY_ID_INVALID", "label_semantic_policy.json", "Unexpected semantic policy ID.")
    if profile.get("selected_candidate_id") != selected.get("selected_candidate_id"):
        _error(errors, "SELECTED_PROFILE_MISMATCH", "selected_semantic_policy.json", "Selected profile and semantic profile disagree.")
    if selected.get("selected_candidate_id") != winner:
        _error(errors, "SELECTED_PROFILE_WINNER_MISMATCH", "selected_semantic_policy.json", f"computed={winner!r}, recorded={selected.get('selected_candidate_id')!r}")
    source = profile.get("source", {})
    for key, value in (("dataset_id", DATASET_ID), ("doi", DATASET_DOI), ("source_split", SOURCE_SPLIT), ("archive_path", SOURCE_ARCHIVE_PATH), ("archive_sha256", SOURCE_ARCHIVE_SHA256)):
        if source.get(key) != value:
            _error(errors, "SEMANTIC_SOURCE_MISMATCH", f"label_semantic_policy.json:source.{key}", str(source.get(key)))
    layer_a = profile.get("layer_a_original_source_annotation", {})
    layer_b = profile.get("layer_b_frame_evidence", {})
    layer_c = profile.get("layer_c_compatibility_proxy", {})
    if layer_a.get("status") != "IMMUTABLE_VERIFIED_SOURCE_ANNOTATION" or layer_a.get("labels") != {str(key): value for key, value in sorted(SOURCE_LABELS.items())}:
        _error(errors, "ORIGINAL_LAYER_NOT_IMMUTABLE", "label_semantic_policy.json:layer_a_original_source_annotation", "Original source labels are not preserved.")
    expected_evidence = {"LYING": "HUMAN_LYING_POSTURE", "SITTING": "HUMAN_SITTING_POSTURE", "STANDING": "HUMAN_STANDING_POSTURE", "EMPTY_ROOM": "NO_ANNOTATED_HUMAN_IN_REPRESENTED_FRAME"}
    if layer_b.get("evidence_labels") != expected_evidence or layer_b.get("fall_event_ground_truth") != "NOT_VERIFIABLE" or layer_b.get("worker_safety_ground_truth") != "NOT_SUPPORTED":
        _error(errors, "FRAME_EVIDENCE_LAYER_INVALID", "frame_evidence_contract.json", "Frame evidence layer weakens source/event/safety boundaries.")
    if layer_c.get("enabled") is not True or layer_c.get("runtime_class_map") != {str(key): value for key, value in sorted(RUNTIME_CLASS_MAP.items())}:
        _error(errors, "COMPATIBILITY_LAYER_INVALID", "label_semantic_policy.json:layer_c_compatibility_proxy", "Runtime compatibility layer is not explicit.")
    temporal = profile.get("temporal_inheritance", {})
    expected_temporal = {"t_a3_policy_id": T_A3_POLICY_ID, "frame_level": "SUPPORTED", "sequence_level": "NOT_VERIFIABLE", "event_level": "NOT_VERIFIABLE", "window_level": "NOT_APPLICABLE", "fall_onset": "NOT_VERIFIABLE", "fall_end": "NOT_VERIFIABLE"}
    for key, value in expected_temporal.items():
        if temporal.get(key) != value:
            _error(errors, "T_A3_LIMITATION_WEAKENED", f"label_semantic_policy.json:temporal_inheritance.{key}", str(temporal.get(key)))
    frame_contract = documents["frame_evidence_contract.json"]
    if frame_contract.get("source_annotation_remains_separate") is not True or frame_contract.get("bbox_relabeling") != "PROHIBITED":
        _error(errors, "FRAME_CONTRACT_LAYER_COLLAPSE", "frame_evidence_contract.json", "Source and frame evidence layers are not separated.")
    compatibility = documents["compatibility_mapping_contract.json"]
    if compatibility.get("model_metrics_used") is not False or compatibility.get("enabled") is not True:
        _error(errors, "COMPATIBILITY_METRIC_OR_ENABLEMENT_INVALID", "compatibility_mapping_contract.json", "Compatibility layer must be explicit and metric-independent.")
    claims = documents["claim_scope_contract.json"]
    if claims.get("derived_target_requires_scope") is not True or claims.get("scope_escalation") != "FAIL_CLOSED":
        _error(errors, "CLAIM_SCOPE_GATE_INVALID", "claim_scope_contract.json", "Derived targets require fail-closed claim scopes.")
    ambiguity = documents["ambiguity_policy.json"]
    if any(value is not False for value in ambiguity.get("source_label_ambiguity", {}).values()):
        _error(errors, "SOURCE_LABEL_AMBIGUITY_COLLAPSED", "ambiguity_policy.json", "Known source labels must remain verified.")
    if ambiguity.get("transition_frame_assignment") != "NOT_APPLICABLE_NO_VERIFIED_TEMPORAL_EVENT" or ambiguity.get("boundary_frame_assignment") != "NOT_APPLICABLE_NO_VERIFIED_TEMPORAL_EVENT":
        _error(errors, "TEMPORAL_BOUNDARY_FABRICATED", "ambiguity_policy.json", "Transition/boundary semantics were fabricated.")
    if ambiguity.get("unknown_label_policy") != "FAIL_CLOSED; NEVER_COERCE_TO_NORMAL":
        _error(errors, "UNKNOWN_LABEL_POLICY_INVALID", "ambiguity_policy.json", "Unknown labels must fail closed.")


def _validate_inventory(documents: dict[str, Any], errors: list[dict[str, str]]) -> None:
    inventory = documents["label_mapping_inventory.json"]
    if inventory.get("row_count") != 8000 or inventory.get("split_assignment_created") is not False or inventory.get("model_metrics_used") is not False:
        _error(errors, "INVENTORY_HEADER_INVALID", "label_mapping_inventory.json", "Inventory cardinality/split/metric boundary invalid.")
    records = inventory.get("records")
    if not isinstance(records, list) or len(records) != 8000:
        _error(errors, "INVENTORY_ROW_COUNT_INVALID", "label_mapping_inventory.json:records", "Expected 8,000 mapping records.")
        return
    indices = [record.get("source_frame_index") for record in records]
    if indices != list(range(8000)):
        _error(errors, "INVENTORY_ORDER_INVALID", "label_mapping_inventory.json:records", "Records must be sorted and cover source indices 0..7999 exactly.")
    for index, record in enumerate(records):
        try:
            validate_mapping_record(record)
        except Exception as exc:
            _error(errors, "MAPPING_RECORD_INVALID", f"label_mapping_inventory.json:records[{index}]", str(exc))
    counts = inventory.get("counts", {})
    if counts.get("source_label_counts") != {"EMPTY_ROOM": 2000, "LYING": 2000, "SITTING": 2000, "STANDING": 2000}:
        _error(errors, "SOURCE_DISTRIBUTION_INVALID", "label_mapping_inventory.json:counts", str(counts.get("source_label_counts")))
    if counts.get("mapping_type_counts") != {"DIRECT_SOURCE_EQUIVALENT": 2000, "DERIVED_POSTURE_PROXY": 6000}:
        _error(errors, "MAPPING_TYPE_DISTRIBUTION_INVALID", "label_mapping_inventory.json:counts", str(counts.get("mapping_type_counts")))
    if counts.get("compatibility_target_counts") != {"HUMAN_FALL": 2000, "HUMAN_NORMAL": 4000, "NOT_HUMAN": 2000}:
        _error(errors, "PROXY_DISTRIBUTION_INVALID", "label_mapping_inventory.json:counts", str(counts.get("compatibility_target_counts")))
    if counts.get("source_label_modified_count") != 0 or counts.get("worker_safety_ground_truth_count") != 0 or counts.get("fall_event_ground_truth_count") != 0:
        _error(errors, "SEMANTIC_ESCALATION_IN_INVENTORY", "label_mapping_inventory.json:counts", str(counts))
    if counts.get("ambiguous_target_mapping_count") != 0 or counts.get("unsupported_mapping_count") != 0 or counts.get("direct_verified_fall_mapping_count") != 0 or counts.get("assignment_exclusion_count") != 0:
        _error(errors, "UNEXPECTED_MAPPING_EXCLUSIONS", "label_mapping_inventory.json:counts", str(counts))


def _validate_activity(documents: dict[str, Any], errors: list[dict[str, str]]) -> None:
    entries = documents["activity_coverage_registry.json"].get("entries", {})
    for name, entry in entries.items():
        if entry.get("treated_as_negative") is not False:
            _error(errors, "UNSUPPORTED_ACTIVITY_AS_NEGATIVE", f"activity_coverage_registry.json:{name}", str(entry))
    for name in ("BENDING", "KNEELING", "ENTERING", "EXITING", "WALKING", "FALL_TRANSITION", "FALL_IMPACT", "POST_FALL_INTERVAL", "RECOVERY"):
        if name not in entries or entries[name].get("source_count") != 0:
            _error(errors, "ACTIVITY_COVERAGE_MISSING", f"activity_coverage_registry.json:{name}", "Unsupported activity must be explicit and have no invented count.")
    if documents["activity_coverage_registry.json"].get("unsupported_activity_policy") != "ABSENCE_OF_A_LABEL_IS_NOT_A_NEGATIVE_EXAMPLE":
        _error(errors, "ACTIVITY_ABSENCE_POLICY_INVALID", "activity_coverage_registry.json", "Unsupported activities must not become negatives.")
    expected_generalization = {
        "subject_generalization": "NOT_VERIFIABLE",
        "session_generalization": "NOT_VERIFIABLE",
        "event_generalization": "NOT_VERIFIABLE",
        "temporal_fall_event_performance": "NOT_VERIFIABLE",
    }
    if documents["limitations.json"].get("generalization") != expected_generalization:
        _error(errors, "GENERALIZATION_LIMITATION_WEAKENED", "limitations.json:generalization", str(documents["limitations.json"].get("generalization")))


def _validate_pilot(documents: dict[str, Any], errors: list[dict[str, str]]) -> None:
    pilot = documents["pilot_semantic_summary.json"]
    records = pilot.get("records")
    if pilot.get("pilot_frame_count") != 48 or not isinstance(records, list) or len(records) != 48:
        _error(errors, "PILOT_COUNT_INVALID", "pilot_semantic_summary.json", "T-A4 pilot must reuse 48 T-A3 frames.")
        return
    if [item.get("source_frame_index") for item in records] != sorted(item.get("source_frame_index") for item in records):
        _error(errors, "PILOT_ORDER_INVALID", "pilot_semantic_summary.json:records", "Pilot records must be sorted.")
    for index, record in enumerate(records):
        try:
            validate_mapping_record(record)
        except Exception as exc:
            _error(errors, "PILOT_MAPPING_INVALID", f"pilot_semantic_summary.json:records[{index}]", str(exc))
        if record.get("t_a3_temporal_policy_id") != T_A3_POLICY_ID or record.get("t_a2_geometry_profile_id") != T_A2_PROFILE_ID:
            _error(errors, "PILOT_PREDECESSOR_IDENTITY_INVALID", f"pilot:{record.get('source_frame_index')}", "T-A2/T-A3 identity was not retained.")
        if any(record.get(key) not in {"ABSENT", "NOT_VERIFIABLE"} for key in ("source_timestamp_status", "source_fps_status", "sequence_id_status", "event_id_status")):
            _error(errors, "PILOT_TEMPORAL_ESCALATION", f"pilot:{record.get('source_frame_index')}", "Pilot temporal statuses were weakened.")
    if pilot.get("fall_event_escalation") is not False:
        _error(errors, "PILOT_FALL_ESCALATION", "pilot_semantic_summary.json", "Pilot cannot claim fall event evidence.")


def _validate_model_reference(
    repo_root: Path,
    documents: dict[str, Any],
    errors: list[dict[str, str]],
    warnings: list[dict[str, str]],
) -> None:
    reference = documents["compatibility_mapping_contract.json"].get("runtime_model_reference", {})
    expected_map = {str(key): value for key, value in sorted(RUNTIME_CLASS_MAP.items())}
    expected_reference = {
        "model_id": "thermal_fall_int8",
        "version": "0.1.0",
        "path": "models/thermal/thermal_fall_int8_v0.1.0.tflite",
        "sha256": "5b56da8d127ccef85f30b6459cc0cfe2d86490e41f3caa5bd2a7b70bbc46ae84",
        "size_bytes": 318184,
        "class_map": expected_map,
        "model_metrics_used": False,
        "status": "LEGACY_OR_CURRENT_RUNTIME_CLASS_MAP_NOT_SOURCE_GROUND_TRUTH",
    }
    for key, expected in expected_reference.items():
        if reference.get(key) != expected:
            _error(
                errors,
                "RUNTIME_REFERENCE_MISMATCH",
                f"compatibility_mapping_contract.json:runtime_model_reference.{key}",
                str(reference.get(key)),
            )

    manifest_path = repo_root / "models/model_manifest.json"
    if not manifest_path.is_file():
        _warning(
            warnings,
            "LEGACY_RUNTIME_ARTIFACT_INTENTIONALLY_EXCLUDED",
            "models/model_manifest.json",
            "The focused thermal-AI repository excludes the legacy runtime manifest and binary; only the frozen compatibility reference was validated.",
        )
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["models"]["thermal"]
        model_path = repo_root / manifest["path"]
        measured = {"sha256": _sha256(model_path), "size_bytes": model_path.stat().st_size}
        for key, value in (("model_id", manifest["model_id"]), ("version", manifest["version"]), ("path", manifest["path"]), ("sha256", measured["sha256"]), ("size_bytes", measured["size_bytes"]), ("class_map", expected_map)):
            if reference.get(key) != value:
                _error(errors, "RUNTIME_REFERENCE_MISMATCH", f"compatibility_mapping_contract.json:runtime_model_reference.{key}", str(reference.get(key)))
        if manifest["sha256"] != measured["sha256"] or manifest["size_bytes"] != measured["size_bytes"]:
            _error(errors, "RUNTIME_MODEL_MANIFEST_STALE", "models/model_manifest.json", "Measured model artifact differs from manifest.")
    except Exception as exc:
        _error(errors, "RUNTIME_MODEL_AUDIT_FAILED", "models/model_manifest.json", str(exc))


def _validate_static_implementation(repo_root: Path, errors: list[dict[str, str]]) -> None:
    for relative in ("datasets/thermal/label_semantics.py", "scripts/generate_thermal_t_a4.py"):
        path = repo_root / relative
        if not path.is_file():
            _error(errors, "T_A4_IMPLEMENTATION_MISSING", relative, "T-A4 implementation file missing.")
            continue
        source = path.read_text(encoding="utf-8")
        try:
            ast.parse(source)
        except SyntaxError as exc:
            _error(errors, "T_A4_SYNTAX_ERROR", relative, str(exc))
        lowered = source.lower()
        for forbidden in ("thermalinterpreter", "inference.thermal", "tensorflow", ".tflite", "canonicalize_source_frame", "train_and_quantize", "fit("):
            if forbidden in lowered:
                _error(errors, "T_A4_SCOPE_COUPLING", relative, forbidden)


def validate_evidence(*, repo_root: Path = ROOT, evidence_dir: Path | None = None, check_checksums: bool = True, verify_real_payload: bool = True) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    evidence_dir = (evidence_dir or repo_root / EVIDENCE_REL).resolve()
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    a0, a1, a2, a3 = _run_predecessors(repo_root, errors, verify_real_payload)
    documents, paths = _load_documents(evidence_dir, errors)
    if len(documents) == len(CORE_JSON):
        _validate_source_identity(repo_root, documents, errors)
        winner = _validate_candidate_selection(documents, errors)
        _validate_semantic_layers(documents, errors, winner)
        _validate_inventory(documents, errors)
        _validate_activity(documents, errors)
        _validate_pilot(documents, errors)
        if winner != documents["selected_semantic_policy.json"].get("selected_candidate_id"):
            _error(errors, "SELECTED_POLICY_WINNER_MISMATCH", "selected_semantic_policy.json", f"computed={winner!r}")
        _validate_model_reference(repo_root, documents, errors, warnings)
    _validate_static_implementation(repo_root, errors)
    validation_path = evidence_dir / "validation_result.json"
    if validation_path.is_file():
        try:
            stored = json.loads(validation_path.read_text(encoding="utf-8"))
            if check_checksums and stored.get("evidence_validation") != "PASS":
                _error(errors, "STORED_VALIDATION_NOT_PASS", "validation_result.json", "Stored validation result must be PASS.")
            if validation_path.read_text(encoding="utf-8") != canonical_json(stored):
                _error(errors, "NONDETERMINISTIC_JSON", "validation_result.json", "Stored validation result must be canonical JSON.")
        except (OSError, json.JSONDecodeError) as exc:
            _error(errors, "VALIDATION_RESULT_INVALID", "validation_result.json", str(exc))
        paths.append(validation_path)
    elif check_checksums:
        _error(errors, "VALIDATION_RESULT_MISSING", "validation_result.json", "Stored validation result missing.")
    if check_checksums and len(documents) == len(CORE_JSON):
        _validate_checksums(repo_root, evidence_dir, paths, errors)
    _warning(warnings, "POSTURE_PROXY_NOT_EVENT_GROUND_TRUTH", "semantic policy", "LYING compatibility target remains a frame-level posture proxy only.")
    sorted_errors = sorted(errors, key=lambda item: (item["code"], item["location"], item["message"]))
    sorted_warnings = sorted(warnings, key=lambda item: (item["code"], item["location"], item["message"]))
    gate = not sorted_errors and all(item.get("evidence_validation") == "PASS" for item in (a0, a1, a2, a3))
    return {
        "error_count": len(sorted_errors),
        "errors": sorted_errors,
        "evidence_validation": "PASS" if gate else "FAIL",
        "overall_outcome": "PASS_WITH_LIMITATIONS" if gate else "NOT_VERIFIABLE",
        "phase": "T-A4",
        "schema_version": "1.0",
        "t_a0_validation": a0.get("evidence_validation", "FAIL"),
        "t_a0_outcome": a0.get("overall_outcome", "NOT_VERIFIABLE"),
        "t_a1_validation": a1.get("evidence_validation", "FAIL"),
        "t_a1_outcome": a1.get("overall_outcome", "NOT_VERIFIABLE"),
        "t_a2_validation": a2.get("evidence_validation", "FAIL"),
        "t_a2_outcome": a2.get("overall_outcome", "NOT_VERIFIABLE"),
        "t_a3_validation": a3.get("evidence_validation", "FAIL"),
        "t_a3_outcome": a3.get("overall_outcome", "NOT_VERIFIABLE"),
        "t_a5_authorized": bool(gate),
        "warning_count": len(sorted_warnings),
        "warnings": sorted_warnings,
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
