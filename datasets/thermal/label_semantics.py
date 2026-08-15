#!/usr/bin/env python3
"""Pure T-A4 source-label semantics and fail-closed proxy mapping.

The SDT source labels are immutable posture/presence annotations.  This module
keeps them separate from an optional compatibility target for the historical
three-class runtime vocabulary.  No temporal event, safety, or medical claim
can be produced here.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping, Sequence


SEMANTIC_POLICY_ID = "THERMAL_LABEL_SEMANTIC_POLICY_001"
SEMANTIC_POLICY_VERSION = "1.0"
SELECTION_POLICY_ID = "THERMAL_LABEL_SELECTION_POLICY_001"
SELECTION_POLICY_VERSION = "1.0"
DATASET_ID = "local_sdt_zenodo_4124309"
DATASET_DOI = "doi:10.5281/zenodo.4124309"
SOURCE_SPLIT = "test"
SOURCE_ARCHIVE_PATH = "datasets/raw_archives/thermal_split_zips/test.zip"
SOURCE_ARCHIVE_SHA256 = "3a838bd70835e579ecfaa820a6c0b4cbc6ba7b76729417c73845f0c959281449"
T_A2_PROFILE_ID = "G1_FIXED_ASPECT_CROP_BILINEAR"
T_A3_POLICY_ID = "THERMAL_TEMPORAL_POLICY_001"
SOURCE_LABELS = {0: "LYING", 1: "SITTING", 2: "STANDING", 3: "EMPTY_ROOM"}
RUNTIME_CLASS_MAP = {0: "NOT_HUMAN", 1: "HUMAN_NORMAL", 2: "HUMAN_FALL"}

MAPPING_TYPES = {
    "DIRECT_SOURCE_EQUIVALENT",
    "DERIVED_POSTURE_PROXY",
    "DERIVED_PRESENCE_PROXY",
    "AMBIGUOUS_TARGET",
    "UNSUPPORTED_MAPPING",
    "NOT_APPLICABLE",
}
CLAIM_SCOPES = {
    "SOURCE_POSTURE_ONLY",
    "FRAME_LEVEL_PRESENCE_ONLY",
    "FRAME_LEVEL_POSTURE_PROXY",
    "FRAME_LEVEL_FALL_COMPATIBILITY_PROXY_ONLY",
    "NOT_TEMPORAL_EVENT_GROUND_TRUTH",
    "NOT_SAFETY_GROUND_TRUTH",
}
FALL_EVIDENCE_STRENGTHS = {"NONE", "POSTURE_COMPATIBLE", "AMBIGUOUS_FOR_FALL", "DIRECT_EVENT_ANNOTATION"}
TEMPORAL_FORBIDDEN_FIELDS = frozenset(
    {
        "timestamp",
        "fps",
        "sequence_id",
        "session_id",
        "recording_id",
        "event_id",
        "event_start",
        "event_end",
        "pre_fall",
        "post_fall",
        "transition_frame",
    }
)


class SemanticPolicyError(ValueError):
    code = "THERMAL_SEMANTIC_POLICY_ERROR"

    def __init__(self, message: str) -> None:
        self.detail = message
        super().__init__(f"{self.code}: {message}")


class UnknownSourceLabelError(SemanticPolicyError):
    code = "SOURCE_LABEL_UNKNOWN"


class UnsupportedLabelMappingError(SemanticPolicyError):
    code = "UNSUPPORTED_LABEL_MAPPING"


class SemanticEscalationError(SemanticPolicyError):
    code = "TEMPORAL_LABEL_ESCALATION_NOT_ALLOWED"


class SafetyStateInferenceError(SemanticPolicyError):
    code = "SAFETY_STATE_INFERENCE_NOT_ALLOWED"


class SemanticPolicyMismatchError(SemanticPolicyError):
    code = "SEMANTIC_POLICY_MISMATCH"


def _copy(value: Any) -> Any:
    return deepcopy(value)


def selection_policy_definition() -> dict[str, Any]:
    """Return the declared, versioned candidate admissibility/ranking policy."""

    return _copy(
        {
            "phase": "T-A4",
            "schema_version": "1.0",
            "policy_id": SELECTION_POLICY_ID,
            "policy_version": SELECTION_POLICY_VERSION,
            "selection_criteria": [
                "ORIGINAL_SOURCE_TRUTH_PRESERVATION",
                "NO_UNSUPPORTED_FALL_EVENT_CLAIM",
                "EXPLICIT_AMBIGUITY_AND_PROXY_SEPARATION",
                "DETERMINISTIC_ONE_TO_ONE_MAPPING_PROVENANCE",
                "SOURCE_RUNTIME_COMPATIBILITY_SEPARATION",
                "LATER_T_A5_T_A6_PROVENANCE_COMPATIBILITY",
                "HISTORICAL_RUNTIME_COMPARISON_WITHOUT_SOURCE_REWRITE",
                "SIMPLE_MACHINE_VERIFIABLE_SEMANTICS",
            ],
            "mandatory_admissibility_gates": {
                "preserves_original_source_labels": True,
                "source_and_derived_layers_separate": True,
                "no_verified_fall_event_ground_truth": True,
                "no_temporal_escalation": True,
                "derived_mapping_has_type_and_rule": True,
                "derived_mapping_has_claim_scope": True,
                "unsupported_activities_not_negative": True,
                "no_general_worker_safety_claim": True,
                "no_model_metrics_used": True,
            },
            "ranking_order": [
                "source_truth_preservation",
                "fall_claim_safety",
                "layer_separation",
                "mapping_provenance",
                "runtime_compatibility",
                "semantic_simplicity",
                "candidate_id",
            ],
            "tie_break_rule": "LEXICOGRAPHIC_DESCENDING_DECLARED_METRICS_THEN_ASCENDING_CANDIDATE_ID",
            "model_metrics_used": False,
            "winner_is_not_predeclared": True,
        }
    )


def candidate_policy_definitions() -> list[dict[str, Any]]:
    """Return the predeclared semantic alternatives in stable ID order."""

    candidates = [
        {
            "candidate_id": "L0_SOURCE_ONLY_4_STATE",
            "description": "Keep the four SDT source labels and expose no runtime compatibility target.",
            "preserves_original_source_labels": True,
            "source_and_derived_layers_separate": True,
            "no_verified_fall_event_ground_truth": True,
            "no_temporal_escalation": True,
            "derived_mapping_has_type_and_rule": True,
            "derived_mapping_has_claim_scope": True,
            "unsupported_activities_not_negative": True,
            "no_general_worker_safety_claim": True,
            "model_metrics_used": False,
            "source_truth_preservation": 1,
            "fall_claim_safety": 1,
            "layer_separation": 1,
            "mapping_provenance": 1,
            "runtime_compatibility": 0,
            "semantic_simplicity": 1,
            "compatibility_layer_enabled": False,
            "mapping_mode": "SOURCE_ONLY",
        },
        {
            "candidate_id": "L1_DUAL_LAYER_SOURCE_PLUS_PROXY",
            "description": "Preserve source and frame evidence layers while exposing explicitly qualified runtime proxies.",
            "preserves_original_source_labels": True,
            "source_and_derived_layers_separate": True,
            "no_verified_fall_event_ground_truth": True,
            "no_temporal_escalation": True,
            "derived_mapping_has_type_and_rule": True,
            "derived_mapping_has_claim_scope": True,
            "unsupported_activities_not_negative": True,
            "no_general_worker_safety_claim": True,
            "model_metrics_used": False,
            "source_truth_preservation": 1,
            "fall_claim_safety": 1,
            "layer_separation": 1,
            "mapping_provenance": 1,
            "runtime_compatibility": 1,
            "semantic_simplicity": 1,
            "compatibility_layer_enabled": True,
            "mapping_mode": "SOURCE_PLUS_EXPLICIT_PROXY",
        },
        {
            "candidate_id": "L2_DIRECT_LEGACY_3_CLASS_COLLAPSE",
            "description": "Rewrite SDT labels directly into the historical three-class runtime vocabulary.",
            "preserves_original_source_labels": False,
            "source_and_derived_layers_separate": False,
            "no_verified_fall_event_ground_truth": False,
            "no_temporal_escalation": False,
            "derived_mapping_has_type_and_rule": False,
            "derived_mapping_has_claim_scope": False,
            "unsupported_activities_not_negative": False,
            "no_general_worker_safety_claim": False,
            "model_metrics_used": False,
            "source_truth_preservation": 0,
            "fall_claim_safety": 0,
            "layer_separation": 0,
            "mapping_provenance": 0,
            "runtime_compatibility": 1,
            "semantic_simplicity": 1,
            "compatibility_layer_enabled": True,
            "mapping_mode": "DIRECT_COLLAPSE_UNSUPPORTED",
        },
    ]
    return sorted(_copy(candidates), key=lambda item: item["candidate_id"])


def _candidate_admissibility(candidate: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    gates = policy["mandatory_admissibility_gates"]
    checks = {
        key: candidate.get(candidate_key) is expected
        for key, candidate_key, expected in (
            ("preserves_original_source_labels", "preserves_original_source_labels", True),
            ("source_and_derived_layers_separate", "source_and_derived_layers_separate", True),
            ("no_verified_fall_event_ground_truth", "no_verified_fall_event_ground_truth", True),
            ("no_temporal_escalation", "no_temporal_escalation", True),
            ("derived_mapping_has_type_and_rule", "derived_mapping_has_type_and_rule", True),
            ("derived_mapping_has_claim_scope", "derived_mapping_has_claim_scope", True),
            ("unsupported_activities_not_negative", "unsupported_activities_not_negative", True),
            ("no_general_worker_safety_claim", "no_general_worker_safety_claim", True),
            ("no_model_metrics_used", "model_metrics_used", False),
        )
    }
    # Keep the declared gate names in the result so the report is auditable.
    for key in gates:
        if key not in checks:
            checks[key] = candidate.get(key) is gates[key]
    reasons = [key for key, passed in sorted(checks.items()) if not passed]
    metrics = {key: int(candidate.get(key, 0)) for key in policy["ranking_order"] if key != "candidate_id"}
    return {"candidate_id": candidate.get("candidate_id"), "admissible": not reasons, "gates": checks, "rejection_reasons": reasons, "ranking_metrics": metrics}


def evaluate_candidates(
    candidates: Sequence[Mapping[str, Any]] | None = None,
    policy: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    policy = policy or selection_policy_definition()
    candidates = candidates or candidate_policy_definitions()
    evaluated = []
    for candidate in candidates:
        result = _candidate_admissibility(candidate, policy)
        result["candidate"] = _copy(dict(candidate))
        evaluated.append(result)
    evaluated.sort(key=lambda item: item["candidate_id"])
    admissible = [item for item in evaluated if item["admissible"]]
    ranking_order = policy["ranking_order"]

    def rank_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        metrics = item["ranking_metrics"]
        return tuple(metrics.get(key, 0) for key in ranking_order if key != "candidate_id") + (str(item["candidate_id"]),)

    ranked = sorted(admissible, key=rank_key, reverse=False)
    # Numeric priorities are declared as larger-is-better; candidate ID is the
    # final ascending tie breaker.  Sort explicitly to avoid mixed directions.
    ranked = sorted(
        admissible,
        key=lambda item: tuple(-int(item["ranking_metrics"].get(key, 0)) for key in ranking_order if key != "candidate_id") + (str(item["candidate_id"]),),
    )
    selected_id = ranked[0]["candidate_id"] if ranked else None
    for item in evaluated:
        item["selected"] = item["candidate_id"] == selected_id
        item["selection_status"] = "SELECTED" if item["selected"] else ("ADMISSIBLE_NOT_SELECTED" if item["admissible"] else "REJECTED")
    return evaluated


def selected_candidate(evaluated: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [item for item in evaluated if item.get("selected")]
    if len(selected) != 1:
        raise SemanticPolicyMismatchError("semantic candidate set does not have exactly one selected admissible policy")
    return _copy(dict(selected[0]))


def semantic_policy_profile(selected_candidate_id: str | None = None) -> dict[str, Any]:
    """Return the profile for the candidate selected by the declared policy.

    The default is derived from the candidate registry and selection policy. A
    caller may pass the independently selected ID, but it must agree with that
    derivation; this prevents a hard-coded winner from defining the evidence.
    """

    evaluated = evaluate_candidates()
    derived_selected = selected_candidate(evaluated)["candidate_id"]
    selected_id = selected_candidate_id or derived_selected
    if selected_id != derived_selected:
        raise SemanticPolicyMismatchError(
            f"requested semantic candidate {selected_id!r} differs from derived selection {derived_selected!r}"
        )
    selected_definition = next(item for item in candidate_policy_definitions() if item["candidate_id"] == selected_id)
    compatibility_enabled = bool(selected_definition["compatibility_layer_enabled"])

    return _copy(
        {
            "phase": "T-A4",
            "schema_version": "1.0",
            "policy_id": SEMANTIC_POLICY_ID,
            "policy_version": SEMANTIC_POLICY_VERSION,
            "selected_candidate_id": selected_id,
            "source": {
                "dataset_id": DATASET_ID,
                "doi": DATASET_DOI,
                "source_split": SOURCE_SPLIT,
                "archive_path": SOURCE_ARCHIVE_PATH,
                "archive_sha256": SOURCE_ARCHIVE_SHA256,
                "original_labels": {str(key): value for key, value in sorted(SOURCE_LABELS.items())},
            },
            "layer_a_original_source_annotation": {
                "status": "IMMUTABLE_VERIFIED_SOURCE_ANNOTATION",
                "meaning": "SDT pose/presence annotation exactly as distributed; no SafeNest fall or safety claim.",
                "labels": {str(key): value for key, value in sorted(SOURCE_LABELS.items())},
            },
            "layer_b_frame_evidence": {
                "evidence_labels": {
                    "LYING": "HUMAN_LYING_POSTURE",
                    "SITTING": "HUMAN_SITTING_POSTURE",
                    "STANDING": "HUMAN_STANDING_POSTURE",
                    "EMPTY_ROOM": "NO_ANNOTATED_HUMAN_IN_REPRESENTED_FRAME",
                },
                "claim_scope": "FRAME_LEVEL_ONLY",
                "fall_event_ground_truth": "NOT_VERIFIABLE",
                "worker_safety_ground_truth": "NOT_SUPPORTED",
            },
            "layer_c_compatibility_proxy": {
                "enabled": compatibility_enabled,
                "runtime_class_map": {str(key): value for key, value in sorted(RUNTIME_CLASS_MAP.items())},
                "targets": {
                    "LYING": "HUMAN_FALL",
                    "SITTING": "HUMAN_NORMAL",
                    "STANDING": "HUMAN_NORMAL",
                    "EMPTY_ROOM": "NOT_HUMAN",
                },
                "mapping_types": {
                    "LYING": "DERIVED_POSTURE_PROXY",
                    "SITTING": "DERIVED_POSTURE_PROXY",
                    "STANDING": "DERIVED_POSTURE_PROXY",
                    "EMPTY_ROOM": "DIRECT_SOURCE_EQUIVALENT",
                },
                "claim_scopes": {
                    "LYING": ["FRAME_LEVEL_FALL_COMPATIBILITY_PROXY_ONLY", "NOT_TEMPORAL_EVENT_GROUND_TRUTH", "NOT_SAFETY_GROUND_TRUTH"],
                    "SITTING": ["FRAME_LEVEL_POSTURE_PROXY", "NOT_SAFETY_GROUND_TRUTH"],
                    "STANDING": ["FRAME_LEVEL_POSTURE_PROXY", "NOT_SAFETY_GROUND_TRUTH"],
                    "EMPTY_ROOM": ["FRAME_LEVEL_PRESENCE_ONLY", "NOT_SAFETY_GROUND_TRUTH"],
                },
            },
            "temporal_inheritance": {
                "t_a3_policy_id": T_A3_POLICY_ID,
                "frame_level": "SUPPORTED",
                "sequence_level": "NOT_VERIFIABLE",
                "event_level": "NOT_VERIFIABLE",
                "window_level": "NOT_APPLICABLE",
                "fall_onset": "NOT_VERIFIABLE",
                "fall_end": "NOT_VERIFIABLE",
                "transition_annotations": "NOT_APPLICABLE_NO_VERIFIED_TEMPORAL_EVENT",
            },
            "model_metrics_used": False,
            "split_assigned": False,
            "risk_or_fusion_semantics": "NOT_DEFINED_BY_T_A4",
        }
    )


def _require_hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise SemanticPolicyError(f"{field} must be a lowercase SHA-256")
    return value


def _source_identity(record: Mapping[str, Any]) -> None:
    expected = {
        "dataset_id": DATASET_ID,
        "source_split": SOURCE_SPLIT,
        "source_archive_path": SOURCE_ARCHIVE_PATH,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
    }
    for key, expected_value in expected.items():
        if record.get(key) != expected_value:
            raise SemanticPolicyMismatchError(f"source identity mismatch for {key}")
    if record.get("source_doi") not in {DATASET_DOI, DATASET_DOI.removeprefix("doi:")}:
        raise SemanticPolicyMismatchError("source DOI mismatch")


def map_source_label(record: Mapping[str, Any], *, policy_id: str = SEMANTIC_POLICY_ID) -> dict[str, Any]:
    """Map one source label to separate frame evidence and optional proxy layers."""

    if policy_id != SEMANTIC_POLICY_ID:
        raise SemanticPolicyMismatchError(f"unsupported selected semantic policy: {policy_id}")
    forbidden = sorted(TEMPORAL_FORBIDDEN_FIELDS.intersection(record))
    if forbidden:
        raise SemanticEscalationError("temporal fields are not label semantics: " + ", ".join(forbidden))
    _source_identity(record)
    label_id = record.get("original_label_id", record.get("source_pose_label"))
    if isinstance(label_id, bool) or not isinstance(label_id, int) or label_id not in SOURCE_LABELS:
        raise UnknownSourceLabelError(f"unknown SDT source label: {label_id!r}")
    label_name = record.get("original_label_name", record.get("source_pose_name"))
    if label_name != SOURCE_LABELS[label_id]:
        raise SemanticPolicyMismatchError("source label ID/name mismatch")
    bbox = record.get("original_bbox", record.get("source_bbox"))
    if not isinstance(bbox, list) or len(bbox) != 4 or any(not isinstance(value, (int, float)) for value in bbox):
        raise SemanticPolicyMismatchError("original bbox must be preserved as four numeric values")
    frame_index = record.get("source_frame_index")
    member = record.get("source_member", record.get("source_member_name"))
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0 or frame_index >= 8000:
        raise SemanticPolicyMismatchError("source frame index is invalid")
    if member != f"test/image_t_{frame_index}.png":
        raise SemanticPolicyMismatchError("source member/frame index mismatch")

    evidence = {
        "LYING": ("HUMAN_LYING_POSTURE", "POSTURE_COMPATIBLE", "AMBIGUOUS_OR_NOT_VERIFIABLE_FOR_FALL_EVENT"),
        "SITTING": ("HUMAN_SITTING_POSTURE", "POSTURE_COMPATIBLE", "NOT_VERIFIABLE_FOR_FALL_EVENT"),
        "STANDING": ("HUMAN_STANDING_POSTURE", "POSTURE_COMPATIBLE", "NOT_VERIFIABLE_FOR_FALL_EVENT"),
        "EMPTY_ROOM": ("NO_ANNOTATED_HUMAN_IN_REPRESENTED_FRAME", "NONE", "NOT_VERIFIABLE_FOR_FALL_EVENT"),
    }[label_name]
    target = {"LYING": "HUMAN_FALL", "SITTING": "HUMAN_NORMAL", "STANDING": "HUMAN_NORMAL", "EMPTY_ROOM": "NOT_HUMAN"}[label_name]
    mapping_type = {"LYING": "DERIVED_POSTURE_PROXY", "SITTING": "DERIVED_POSTURE_PROXY", "STANDING": "DERIVED_POSTURE_PROXY", "EMPTY_ROOM": "DIRECT_SOURCE_EQUIVALENT"}[label_name]
    rule_id = {
        "LYING": "THERMAL_MAP_LYING_TO_FALL_COMPAT_PROXY_001",
        "SITTING": "THERMAL_MAP_SITTING_TO_NON_LYING_PROXY_001",
        "STANDING": "THERMAL_MAP_STANDING_TO_NON_LYING_PROXY_001",
        "EMPTY_ROOM": "THERMAL_MAP_EMPTY_ROOM_TO_NO_HUMAN_001",
    }[label_name]
    scopes = {
        "LYING": ["FRAME_LEVEL_FALL_COMPATIBILITY_PROXY_ONLY", "NOT_TEMPORAL_EVENT_GROUND_TRUTH", "NOT_SAFETY_GROUND_TRUTH"],
        "SITTING": ["FRAME_LEVEL_POSTURE_PROXY", "NOT_SAFETY_GROUND_TRUTH"],
        "STANDING": ["FRAME_LEVEL_POSTURE_PROXY", "NOT_SAFETY_GROUND_TRUTH"],
        "EMPTY_ROOM": ["FRAME_LEVEL_PRESENCE_ONLY", "NOT_SAFETY_GROUND_TRUTH"],
    }[label_name]
    result = {
        "dataset_id": DATASET_ID,
        "source_doi": DATASET_DOI,
        "source_split": SOURCE_SPLIT,
        "source_archive_path": SOURCE_ARCHIVE_PATH,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_member": member,
        "source_frame_index": frame_index,
        "original_label_id": label_id,
        "original_label_name": label_name,
        "original_bbox": list(bbox),
        "semantic_policy_id": SEMANTIC_POLICY_ID,
        "frame_evidence_label": evidence[0],
        "compatibility_target": target,
        "compatibility_target_status": "DIRECT_SOURCE_EQUIVALENT" if mapping_type == "DIRECT_SOURCE_EQUIVALENT" else "DERIVED_PROXY",
        "mapping_type": mapping_type,
        "mapping_rule_id": rule_id,
        "source_annotation_status": "VERIFIED",
        "derived_semantic_status": "VERIFIED_FRAME_LEVEL_EVIDENCE",
        "fall_event_semantic_status": "NOT_VERIFIABLE",
        "temporal_event_status": "NOT_VERIFIABLE",
        "ambiguity_status": evidence[2],
        "fall_evidence_strength": evidence[1],
        "claim_scope": scopes,
        "assignment_status": "ELIGIBLE_FOR_LATER_PROXY_LABEL_CONSIDERATION",
        "split_assignment_status": "NOT_ASSIGNED_T_A4",
        "source_label_modified": False,
        "worker_safety_ground_truth": False,
    }
    validate_mapping_record(result)
    return result


def validate_mapping_record(record: Mapping[str, Any]) -> None:
    """Reject semantic escalation or provenance loss in one mapping record."""

    forbidden_keys = sorted(
        str(key).lower()
        for key in record
        if str(key).lower() in TEMPORAL_FORBIDDEN_FIELDS
        or str(key).lower() in {"fall_transition", "pre_fall_event", "post_fall_event"}
    )
    if forbidden_keys:
        raise SemanticEscalationError("temporal fields are unavailable: " + ", ".join(forbidden_keys))
    _source_identity(record)
    required = {
        "source_member", "source_frame_index", "original_label_id", "original_label_name", "original_bbox",
        "semantic_policy_id", "frame_evidence_label", "compatibility_target", "compatibility_target_status", "mapping_type", "mapping_rule_id",
        "source_annotation_status", "derived_semantic_status", "fall_event_semantic_status", "temporal_event_status",
        "ambiguity_status", "fall_evidence_strength", "claim_scope", "assignment_status", "split_assignment_status",
        "source_label_modified", "worker_safety_ground_truth",
    }
    missing = sorted(required - set(record))
    if missing:
        raise SemanticPolicyMismatchError("mapping record missing: " + ", ".join(missing))
    if record["semantic_policy_id"] != SEMANTIC_POLICY_ID or record["source_label_modified"] is not False:
        raise SemanticPolicyMismatchError("source label or semantic policy identity was changed")
    label_id = record["original_label_id"]
    if isinstance(label_id, bool) or not isinstance(label_id, int) or label_id not in SOURCE_LABELS or record["original_label_name"] != SOURCE_LABELS[label_id]:
        raise UnknownSourceLabelError(f"unknown or mismatched source label: {label_id!r}")
    frame_index = record["source_frame_index"]
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or not 0 <= frame_index < 8000:
        raise SemanticPolicyMismatchError("source frame index is invalid")
    if record["source_member"] != f"test/image_t_{record['source_frame_index']}.png":
        raise SemanticPolicyMismatchError("source member is not linked to source frame index")
    bbox = record["original_bbox"]
    if not isinstance(bbox, list) or len(bbox) != 4 or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
        raise SemanticPolicyMismatchError("original bbox must be preserved as four numeric values")
    if record["mapping_type"] not in MAPPING_TYPES or not record["mapping_rule_id"]:
        raise UnsupportedLabelMappingError("mapping type and rule ID are mandatory")
    if not isinstance(record["claim_scope"], list) or not record["claim_scope"] or not set(record["claim_scope"]).issubset(CLAIM_SCOPES):
        raise SemanticPolicyMismatchError("claim scope is missing or unsupported")
    if record["fall_event_semantic_status"] != "NOT_VERIFIABLE" or record["temporal_event_status"] != "NOT_VERIFIABLE":
        raise SemanticEscalationError("source posture cannot become temporal event ground truth")
    if record["worker_safety_ground_truth"] is not False or record["split_assignment_status"] != "NOT_ASSIGNED_T_A4":
        raise SafetyStateInferenceError("T-A4 cannot assert safety or assign a split")
    label_name = record["original_label_name"]
    expected = {
        "LYING": {
            "frame_evidence_label": "HUMAN_LYING_POSTURE",
            "compatibility_target": "HUMAN_FALL",
            "compatibility_target_status": "DERIVED_PROXY",
            "mapping_type": "DERIVED_POSTURE_PROXY",
            "mapping_rule_id": "THERMAL_MAP_LYING_TO_FALL_COMPAT_PROXY_001",
            "ambiguity_status": "AMBIGUOUS_OR_NOT_VERIFIABLE_FOR_FALL_EVENT",
            "fall_evidence_strength": "POSTURE_COMPATIBLE",
            "claim_scope": ["FRAME_LEVEL_FALL_COMPATIBILITY_PROXY_ONLY", "NOT_TEMPORAL_EVENT_GROUND_TRUTH", "NOT_SAFETY_GROUND_TRUTH"],
        },
        "SITTING": {
            "frame_evidence_label": "HUMAN_SITTING_POSTURE",
            "compatibility_target": "HUMAN_NORMAL",
            "compatibility_target_status": "DERIVED_PROXY",
            "mapping_type": "DERIVED_POSTURE_PROXY",
            "mapping_rule_id": "THERMAL_MAP_SITTING_TO_NON_LYING_PROXY_001",
            "ambiguity_status": "NOT_VERIFIABLE_FOR_FALL_EVENT",
            "fall_evidence_strength": "POSTURE_COMPATIBLE",
            "claim_scope": ["FRAME_LEVEL_POSTURE_PROXY", "NOT_SAFETY_GROUND_TRUTH"],
        },
        "STANDING": {
            "frame_evidence_label": "HUMAN_STANDING_POSTURE",
            "compatibility_target": "HUMAN_NORMAL",
            "compatibility_target_status": "DERIVED_PROXY",
            "mapping_type": "DERIVED_POSTURE_PROXY",
            "mapping_rule_id": "THERMAL_MAP_STANDING_TO_NON_LYING_PROXY_001",
            "ambiguity_status": "NOT_VERIFIABLE_FOR_FALL_EVENT",
            "fall_evidence_strength": "POSTURE_COMPATIBLE",
            "claim_scope": ["FRAME_LEVEL_POSTURE_PROXY", "NOT_SAFETY_GROUND_TRUTH"],
        },
        "EMPTY_ROOM": {
            "frame_evidence_label": "NO_ANNOTATED_HUMAN_IN_REPRESENTED_FRAME",
            "compatibility_target": "NOT_HUMAN",
            "compatibility_target_status": "DIRECT_SOURCE_EQUIVALENT",
            "mapping_type": "DIRECT_SOURCE_EQUIVALENT",
            "mapping_rule_id": "THERMAL_MAP_EMPTY_ROOM_TO_NO_HUMAN_001",
            "ambiguity_status": "NOT_VERIFIABLE_FOR_FALL_EVENT",
            "fall_evidence_strength": "NONE",
            "claim_scope": ["FRAME_LEVEL_PRESENCE_ONLY", "NOT_SAFETY_GROUND_TRUTH"],
        },
    }[label_name]
    for field, expected_value in expected.items():
        if record[field] != expected_value:
            raise SemanticPolicyMismatchError(f"{label_name} semantic field {field} changed")
    if record["source_annotation_status"] != "VERIFIED" or record["derived_semantic_status"] != "VERIFIED_FRAME_LEVEL_EVIDENCE":
        raise SemanticPolicyMismatchError("source/evidence status changed")
    if record["assignment_status"] != "ELIGIBLE_FOR_LATER_PROXY_LABEL_CONSIDERATION":
        raise UnsupportedLabelMappingError("unsupported assignment status")
    if record["mapping_type"] in {"DERIVED_POSTURE_PROXY", "DERIVED_PRESENCE_PROXY"} and record["compatibility_target"] is None:
        raise UnsupportedLabelMappingError("derived mapping cannot omit compatibility target")


def reject_semantic_escalation(request: Mapping[str, Any]) -> None:
    """Fail closed for requests that turn posture/proxy evidence into stronger claims."""

    keys = {str(key).lower() for key in request}
    forbidden = sorted(key for key in keys if key in {item.lower() for item in TEMPORAL_FORBIDDEN_FIELDS})
    if forbidden:
        raise SemanticEscalationError("temporal semantic fields are unavailable: " + ", ".join(forbidden))
    for key in ("verified_fall", "fall_event_ground_truth", "direct_fall_event", "fall_onset", "fall_transition", "post_fall_event"):
        if request.get(key) is True or str(request.get(key, "")).upper() in {"VERIFIED", "DIRECT_EVENT_ANNOTATION", "FALL_EVENT"}:
            raise SemanticEscalationError(f"unsupported semantic escalation: {key}")
    for key in ("worker_safe", "general_worker_safety", "medical_diagnosis", "emergency_confirmed"):
        if request.get(key) is True:
            raise SafetyStateInferenceError(f"unsupported safety/medical inference: {key}")
    for key in ("label", "original_label_name", "frame_evidence_label"):
        value = str(request.get(key, "")).upper()
        if value in {"PRE_FALL", "POST_FALL", "FALL_TRANSITION", "FALL_IMPACT", "RECOVERY"}:
            raise SemanticEscalationError(f"unsupported temporal label: {value}")
    if request.get("label") == "UNKNOWN" or request.get("original_label_id") not in (None, 0, 1, 2, 3):
        raise UnknownSourceLabelError("unknown source label cannot be coerced")


__all__ = [
    "CLAIM_SCOPES",
    "DATASET_DOI",
    "DATASET_ID",
    "FALL_EVIDENCE_STRENGTHS",
    "MAPPING_TYPES",
    "RUNTIME_CLASS_MAP",
    "SEMANTIC_POLICY_ID",
    "SEMANTIC_POLICY_VERSION",
    "SELECTION_POLICY_ID",
    "SOURCE_ARCHIVE_PATH",
    "SOURCE_ARCHIVE_SHA256",
    "SOURCE_LABELS",
    "SOURCE_SPLIT",
    "T_A2_PROFILE_ID",
    "T_A3_POLICY_ID",
    "SafetyStateInferenceError",
    "SemanticEscalationError",
    "SemanticPolicyError",
    "SemanticPolicyMismatchError",
    "UnknownSourceLabelError",
    "UnsupportedLabelMappingError",
    "candidate_policy_definitions",
    "evaluate_candidates",
    "map_source_label",
    "reject_semantic_escalation",
    "selected_candidate",
    "selection_policy_definition",
    "semantic_policy_profile",
    "validate_mapping_record",
]
