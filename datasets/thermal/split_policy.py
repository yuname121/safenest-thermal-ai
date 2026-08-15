#!/usr/bin/env python3
"""Fail-closed Thermal T-A5 grouping and immutable assignment policy.

This module contains only provenance and evaluation-governance rules.  It does
not read image payloads, split arrays, train a model, or choose a geometry.  In
particular, a source frame index is an identifier, never a temporal or group
key.  The selected policy preserves the three official SDT partitions and
assigns the already-accessed real test partition to development evaluation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence


DATASET_ID = "local_sdt_zenodo_4124309"
DATASET_NAME = "SDT Dataset"
DATASET_DOI = "doi:10.5281/zenodo.4124309"
OFFICIAL_SOURCE_URL = "https://zenodo.org/records/4124309"
OFFICIAL_DOCUMENTATION_URL = "https://cvl.tuwien.ac.at/research/cvl-databases/sdt-icip/"
SOURCE_ARCHIVE_PATH = "datasets/raw_archives/thermal_split_zips/test.zip"
SOURCE_ARCHIVE_SHA256 = "3a838bd70835e579ecfaa820a6c0b4cbc6ba7b76729417c73845f0c959281449"
SEMANTIC_POLICY_ID = "THERMAL_LABEL_SEMANTIC_POLICY_001"
TEMPORAL_POLICY_ID = "THERMAL_TEMPORAL_POLICY_001"
GROUPING_POLICY_ID = "THERMAL_GROUPING_POLICY_001"
SPLIT_SELECTION_POLICY_ID = "THERMAL_SPLIT_SELECTION_POLICY_001"
SPLIT_POLICY_ID = "THERMAL_SPLIT_POLICY_001"
ASSIGNMENT_RULE_ID = "THERMAL_ASSIGNMENT_RULE_001"
SOURCE_PARTITION_CONTRACT_ID = "THERMAL_SOURCE_PARTITION_CONTRACT_001"
ASSIGNMENT_SCHEMA_VERSION = "1.0"
REAL_TEST_FRAME_COUNT = 8000

SOURCE_SPLITS = ("train", "validation", "test")
SOURCE_DOMAINS = ("SYNTHETIC", "REAL")
SAFE_NEST_ROLES = (
    "TRAIN",
    "VALIDATION",
    "LOCKED_TEST",
    "REAL_EVAL_DEVELOPMENT",
    "UNASSIGNED",
    "EXCLUDED",
)
GROUPING_DIMENSIONS = (
    "subject",
    "session",
    "recording",
    "event",
    "sequence",
    "scene",
    "camera",
)


class SplitPolicyError(ValueError):
    """Base class for fail-closed T-A5 policy errors."""

    code = "THERMAL_T_A5_POLICY_ERROR"

    def __init__(self, message: str) -> None:
        self.detail = message
        super().__init__(f"{self.code}: {message}")


class GroupingProvenanceUnavailableError(SplitPolicyError):
    code = "GROUP_PROVENANCE_UNAVAILABLE"


class FrameRandomSplitError(SplitPolicyError):
    code = "FRAME_RANDOM_SPLIT_NOT_ALLOWED"


class FrameHashSplitError(SplitPolicyError):
    code = "FRAME_HASH_SPLIT_NOT_ALLOWED"


class LockedTestAccessError(SplitPolicyError):
    code = "LOCKED_TEST_ALREADY_ACCESSED"


class SourceDomainMismatchError(SplitPolicyError):
    code = "SOURCE_DOMAIN_MISMATCH"


class GroupCrossSplitError(SplitPolicyError):
    code = "GROUP_CROSS_SPLIT_ASSIGNMENT"


class DerivedAssignmentMismatchError(SplitPolicyError):
    code = "DERIVED_SAMPLE_SPLIT_MISMATCH"


class AssignmentPolicyMismatchError(SplitPolicyError):
    code = "ASSIGNMENT_POLICY_MISMATCH"


class UnverifiedGeneralizationClaimError(SplitPolicyError):
    code = "UNVERIFIED_GROUP_GENERALIZATION_CLAIM"


class UnknownAssignmentRoleError(SplitPolicyError):
    code = "UNKNOWN_ASSIGNMENT_ROLE"


def _copy(value: Any) -> Any:
    return deepcopy(value)


def source_partition_definitions() -> list[dict[str, Any]]:
    """Return the official source partitions, without reading cloud payloads."""

    return _copy(
        [
            {
                "source_split": "train",
                "source_domain": "SYNTHETIC",
                "official_sample_count": 32000,
                "thermal_pair_count": 32000,
                "planned_safenest_role": "TRAIN",
                "materialization_status": "LOCAL_CLOUD_PLACEHOLDER",
                "readable_offline": False,
                "sample_inventory_status": "SAMPLE_LEVEL_INVENTORY_PENDING_MATERIALIZATION",
                "source_partition_preserved": True,
                "pristine_locked_test_eligible": False,
                "locked_test_status": "NOT_APPLICABLE_SYNTHETIC_TRAIN_PARTITION",
                "access_status": "T_A0_METADATA_ONLY_NO_PAYLOAD_READ",
                "assignment_seed": "NOT_APPLICABLE",
                "hash_assignment": "NOT_APPLICABLE",
            },
            {
                "source_split": "validation",
                "source_domain": "SYNTHETIC",
                "official_sample_count": 8000,
                "thermal_pair_count": 8000,
                "planned_safenest_role": "VALIDATION",
                "materialization_status": "LOCAL_CLOUD_PLACEHOLDER",
                "readable_offline": False,
                "sample_inventory_status": "SAMPLE_LEVEL_INVENTORY_PENDING_MATERIALIZATION",
                "source_partition_preserved": True,
                "pristine_locked_test_eligible": False,
                "locked_test_status": "NOT_APPLICABLE_SYNTHETIC_VALIDATION_PARTITION",
                "access_status": "T_A0_METADATA_ONLY_NO_PAYLOAD_READ",
                "assignment_seed": "NOT_APPLICABLE",
                "hash_assignment": "NOT_APPLICABLE",
            },
            {
                "source_split": "test",
                "source_domain": "REAL",
                "official_sample_count": 8000,
                "thermal_pair_count": 8000,
                "planned_safenest_role": "REAL_EVAL_DEVELOPMENT",
                "materialization_status": "LOCALLY_MATERIALIZED",
                "readable_offline": True,
                "sample_inventory_status": "FULL_MEMBER_AND_LABEL_INVENTORY_VERIFIED_T_A1_T_A4",
                "source_partition_preserved": True,
                "pristine_locked_test_eligible": False,
                "locked_test_status": "DISQUALIFIED_BY_PRIOR_ACCESS",
                "locked_test_disqualification_reason": "USED_FOR_PREPROCESSING_GEOMETRY_SELECTION",
                "access_status": "USED_FOR_T_A0_T_A4_DEVELOPMENT_AND_GEOMETRY_SELECTION",
                "assignment_seed": "NOT_APPLICABLE",
                "hash_assignment": "NOT_APPLICABLE",
                "archive_path": SOURCE_ARCHIVE_PATH,
                "archive_sha256": SOURCE_ARCHIVE_SHA256,
            },
        ]
    )


def grouping_evidence_definition() -> dict[str, Any]:
    """Return the measured grouping inventory; absence is explicit evidence."""

    dimensions = {
        "subject": {
            "availability": "ABSENT",
            "status": "NOT_VERIFIABLE",
            "identifier_schema": "subject_id ABSENT",
            "cardinality": "NOT_VERIFIABLE",
            "independence_confidence": "NONE",
            "usable_for_split": False,
            "reason": "T-A1 source schema and official documentation expose no subject identifier.",
        },
        "session": {
            "availability": "ABSENT",
            "status": "NOT_VERIFIABLE",
            "identifier_schema": "session_id/recording_id ABSENT",
            "cardinality": "NOT_VERIFIABLE",
            "independence_confidence": "NONE",
            "usable_for_split": False,
            "reason": "No session or recording identifier is distributed.",
        },
        "recording": {
            "availability": "ABSENT",
            "status": "NOT_VERIFIABLE",
            "identifier_schema": "recording_id ABSENT",
            "cardinality": "NOT_VERIFIABLE",
            "independence_confidence": "NONE",
            "usable_for_split": False,
            "reason": "The source is distributed as frame members without recording provenance.",
        },
        "event": {
            "availability": "ABSENT",
            "status": "NOT_VERIFIABLE",
            "identifier_schema": "event_id ABSENT",
            "cardinality": "NOT_VERIFIABLE",
            "independence_confidence": "NONE",
            "usable_for_split": False,
            "reason": "Labels are frame pose/presence annotations and contain no event boundaries.",
        },
        "sequence": {
            "availability": "ABSENT",
            "status": "NOT_VERIFIABLE",
            "identifier_schema": "sequence_id/clip_id ABSENT",
            "cardinality": "NOT_VERIFIABLE",
            "independence_confidence": "NONE",
            "usable_for_split": False,
            "reason": "Frame index is provenance only; no sequence or temporal cadence is documented.",
        },
        "scene": {
            "availability": "NOT_DOCUMENTED",
            "status": "NOT_VERIFIABLE",
            "identifier_schema": "scene/room subset NOT_DOCUMENTED_PER_FRAME",
            "cardinality": "NOT_VERIFIABLE",
            "independence_confidence": "NONE",
            "usable_for_split": False,
            "reason": "Visual similarity, filename ranges, and bbox position are not authoritative scene groups.",
        },
        "camera": {
            "availability": "SOURCE_SENSOR_DOCUMENTED_ONLY",
            "status": "NOT_VERIFIABLE",
            "identifier_schema": "camera_id ABSENT_PER_FRAME",
            "cardinality": "NOT_VERIFIABLE",
            "independence_confidence": "NONE",
            "usable_for_split": False,
            "reason": "FLIR Lepton 3.5 is documented as the sensor, but no independent camera group key is attached to frames.",
        },
    }
    return {
        "phase": "T-A5",
        "schema_version": ASSIGNMENT_SCHEMA_VERSION,
        "policy_id": GROUPING_POLICY_ID,
        "source_dataset_id": DATASET_ID,
        "grouping_priority": ["subject", "session", "recording", "event", "sequence", "scene", "camera"],
        "dimensions": dimensions,
        "strongest_verified_grouping_unit": "OFFICIAL_SOURCE_PARTITION",
        "strongest_verified_grouping_status": "SOURCE_PARTITION_ONLY_NO_SUBJECT_GROUP",
        "fallback_grouping": "NOT_VERIFIABLE",
        "subject_grouping": "NOT_VERIFIABLE",
        "session_grouping": "NOT_VERIFIABLE",
        "event_grouping": "NOT_VERIFIABLE",
        "sequence_grouping": "NOT_VERIFIABLE",
        "generalization_performance": "NOT_VERIFIABLE",
        "frame_index_as_group": False,
        "label_as_group": False,
        "model_metrics_used": False,
    }


def access_history_definition() -> dict[str, Any]:
    """Return the immutable T-A0–T-A4 partition access audit."""

    rows: list[dict[str, Any]] = []
    def add(phase: str, split: str, domain: str, materialization: str, scope: str, access: str, influenced: bool, contamination: str, evidence: str) -> None:
        rows.append({
            "phase": phase,
            "source_split": split,
            "source_domain": domain,
            "payload_materialization": materialization,
            "sample_scope": scope,
            "access_type": access,
            "decision_influenced": influenced,
            "locked_test_contamination_relevance": contamination,
            "evidence_reference": evidence,
        })
    for phase, scope, access, influenced, evidence in (
        ("T-A0", "metadata_and_bounded_inventory", "SOURCE_IDENTITY_AND_INVENTORY", True, "T-A0 source identity and local asset registry"),
        ("T-A1", "12_frame_real_pilot_plus_member_inventory", "RAW_UNIT_AND_READER_PILOT", True, "T-A1 reader pilot and archive member inventory"),
        ("T-A2", "48_frame_real_pilot", "GEOMETRY_SELECTION", True, "T-A2 pilot geometry summary and selected profile"),
        ("T-A3", "48_frame_real_pilot_reused", "TEMPORAL_CAPABILITY_ANALYSIS", True, "T-A3 pilot temporal summary"),
        ("T-A4", "48_frame_pilot_reused_plus_all_8000_labels", "SEMANTIC_POLICY_SELECTION", True, "T-A4 pilot semantic summary and label mapping inventory"),
    ):
        add(phase, "test", "REAL", "LOCALLY_MATERIALIZED", scope, access, influenced, "DISQUALIFIES_PRISTINE_LOCKED_TEST", evidence)
    for phase, scope, access, evidence in (
        ("T-A0", "official_partition_metadata", "SOURCE_IDENTITY_ONLY", "T-A0 source identity and local asset registry"),
        ("T-A1", "official_partition_metadata", "SOURCE_SCHEMA_ONLY", "T-A1 source schema profile"),
        ("T-A2", "no_payload_access", "NO_PAYLOAD_ACCESS", "T-A2 source partition inheritance"),
        ("T-A3", "no_payload_access", "NO_PAYLOAD_ACCESS", "T-A3 temporal evidence registry"),
        ("T-A4", "no_payload_access", "NO_PAYLOAD_ACCESS", "T-A4 source contract"),
    ):
        add(phase, "train", "SYNTHETIC", "LOCAL_CLOUD_PLACEHOLDER", scope, access, False, "NOT_APPLICABLE", evidence)
        add(phase, "validation", "SYNTHETIC", "LOCAL_CLOUD_PLACEHOLDER", scope, access, False, "NOT_APPLICABLE", evidence)
    rows.sort(key=lambda row: (row["phase"], row["source_split"], row["access_type"]))
    return {
        "phase": "T-A5",
        "schema_version": ASSIGNMENT_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "history_policy_id": GROUPING_POLICY_ID,
        "source_partitions": list(SOURCE_SPLITS),
        "entries": rows,
        "pristine_locked_test_available": "NO",
        "pristine_locked_test_reason": "The real test partition was used for T-A2 geometry selection and later T-A3/T-A4 development evidence.",
        "model_metrics_used": False,
    }


def candidate_policy_definitions() -> list[dict[str, Any]]:
    """Return split candidates.  Evaluation, not list order, chooses a winner."""

    return _copy([
        {
            "candidate_id": "S0_OFFICIAL_SOURCE_PARTITION_PRESERVATION",
            "description": "Preserve train/validation/test exactly and assign test only to real-domain development evaluation.",
            "preserves_official_partitions": True,
            "uses_frame_random_split": False,
            "uses_frame_hash_split": False,
            "has_verified_independent_grouping": False,
            "handles_prior_access": True,
            "separates_source_domains": True,
            "deterministic": True,
            "compatible_with_t_a6": True,
            "supports_t_b_hygiene": True,
            "admissible_without_new_group_evidence": True,
            "ranking_metrics": {"provenance_independence": 1, "official_partition_preservation": 1, "no_cross_group_overlap": 1, "no_retroactive_locked_test": 1, "no_frame_random_leakage": 1, "source_domain_transparency": 1, "deterministic_assignment": 1, "t_a6_compatibility": 1, "t_b_hygiene": 1},
        },
        {
            "candidate_id": "S1_REAL_TEST_FRAME_RANDOM_RESPLIT",
            "description": "Randomly divide real test frames into new roles.",
            "preserves_official_partitions": False,
            "uses_frame_random_split": True,
            "uses_frame_hash_split": False,
            "has_verified_independent_grouping": False,
            "handles_prior_access": False,
            "separates_source_domains": False,
            "deterministic": False,
            "compatible_with_t_a6": False,
            "supports_t_b_hygiene": False,
            "admissible_without_new_group_evidence": False,
            "rejection_reason": "FRAME_RANDOM_SPLIT_WITHOUT_INDEPENDENT_GROUP_PROVENANCE",
            "ranking_metrics": {"provenance_independence": 0, "official_partition_preservation": 0, "no_cross_group_overlap": 0, "no_retroactive_locked_test": 0, "no_frame_random_leakage": 0, "source_domain_transparency": 0, "deterministic_assignment": 0, "t_a6_compatibility": 0, "t_b_hygiene": 0},
        },
        {
            "candidate_id": "S2_REAL_TEST_FRAME_HASH_RESPLIT",
            "description": "Hash frame or member identifiers into new roles.",
            "preserves_official_partitions": False,
            "uses_frame_random_split": False,
            "uses_frame_hash_split": True,
            "has_verified_independent_grouping": False,
            "handles_prior_access": False,
            "separates_source_domains": False,
            "deterministic": True,
            "compatible_with_t_a6": False,
            "supports_t_b_hygiene": False,
            "admissible_without_new_group_evidence": False,
            "rejection_reason": "FRAME_HASH_SPLIT_WITHOUT_INDEPENDENT_GROUP_PROVENANCE",
            "ranking_metrics": {"provenance_independence": 0, "official_partition_preservation": 0, "no_cross_group_overlap": 0, "no_retroactive_locked_test": 0, "no_frame_random_leakage": 0, "source_domain_transparency": 0, "deterministic_assignment": 1, "t_a6_compatibility": 0, "t_b_hygiene": 0},
        },
        {
            "candidate_id": "S3_VERIFIED_GROUP_SPLIT_IF_DISCOVERED",
            "description": "Use a newly verified subject/session/event/scene group if authoritative metadata exists.",
            "preserves_official_partitions": False,
            "uses_frame_random_split": False,
            "uses_frame_hash_split": False,
            "has_verified_independent_grouping": False,
            "handles_prior_access": False,
            "separates_source_domains": True,
            "deterministic": True,
            "compatible_with_t_a6": False,
            "supports_t_b_hygiene": False,
            "admissible_without_new_group_evidence": False,
            "rejection_reason": "NO_AUTHORITATIVE_INDEPENDENT_GROUP_KEY_DISCOVERED",
            "ranking_metrics": {"provenance_independence": 0, "official_partition_preservation": 0, "no_cross_group_overlap": 0, "no_retroactive_locked_test": 0, "no_frame_random_leakage": 1, "source_domain_transparency": 1, "deterministic_assignment": 1, "t_a6_compatibility": 0, "t_b_hygiene": 0},
        },
    ])


def selection_policy_definition() -> dict[str, Any]:
    return {
        "phase": "T-A5",
        "schema_version": ASSIGNMENT_SCHEMA_VERSION,
        "policy_id": SPLIT_SELECTION_POLICY_ID,
        "selection_criteria": [
            "PROVENANCE_SUPPORTED_INDEPENDENCE",
            "OFFICIAL_SOURCE_PARTITION_PRESERVATION",
            "NO_CROSS_GROUP_OVERLAP",
            "NO_RETROACTIVE_PRISTINE_HOLDOUT_CLAIM",
            "NO_FRAME_RANDOM_OR_HASH_LEAKAGE",
            "SOURCE_DOMAIN_TRANSPARENCY",
            "DETERMINISTIC_ASSIGNMENT",
            "T_A6_PROVENANCE_COMPATIBILITY",
            "T_B_EVALUATION_HYGIENE",
        ],
        "ranking_order": [
            "provenance_independence", "official_partition_preservation", "no_cross_group_overlap", "no_retroactive_locked_test", "no_frame_random_leakage", "source_domain_transparency", "deterministic_assignment", "t_a6_compatibility", "t_b_hygiene", "candidate_id",
        ],
        "mandatory_gates": {
            "preserves_official_partitions": True,
            "uses_frame_random_split": False,
            "uses_frame_hash_split": False,
            "handles_prior_access": True,
            "separates_source_domains": True,
            "deterministic": True,
            "compatible_with_t_a6": True,
            "supports_t_b_hygiene": True,
            "no_model_metrics_used": True,
        },
        "tie_break_rule": "LEXICOGRAPHIC_DESCENDING_DECLARED_METRICS_THEN_ASCENDING_CANDIDATE_ID",
        "winner_is_not_predeclared": True,
        "model_metrics_used": False,
    }


def evaluate_candidates(candidates: Sequence[Mapping[str, Any]], policy: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    policy = policy or selection_policy_definition()
    gates = policy["mandatory_gates"]
    evaluated: list[dict[str, Any]] = []
    for candidate in candidates:
        checks = {key: candidate.get(key) == expected for key, expected in gates.items() if key != "no_model_metrics_used"}
        checks["no_model_metrics_used"] = candidate.get("model_metrics_used", False) is False
        admissible = all(checks.values()) and candidate.get("admissible_without_new_group_evidence", False)
        rejection_reasons = [key for key, passed in checks.items() if not passed]
        if not candidate.get("admissible_without_new_group_evidence", False):
            rejection_reasons.append(candidate.get("rejection_reason", "CANDIDATE_NOT_SUPPORTED_BY_PROVENANCE"))
        evaluated.append({**_copy(dict(candidate)), "admissibility_checks": checks, "admissible": admissible, "rejection_reasons": sorted(set(rejection_reasons)), "selected": False})
    admissible = [item for item in evaluated if item["admissible"]]
    if admissible:
        order = policy["ranking_order"]
        winner = sorted(admissible, key=lambda item: tuple((-int(item.get("ranking_metrics", {}).get(key, 0)) if key != "candidate_id" else item["candidate_id"]) for key in order))[0]
        winner["selected"] = True
    return sorted(evaluated, key=lambda item: item["candidate_id"])


def selected_candidate(evaluated: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [item for item in evaluated if item.get("selected") and item.get("admissible")]
    if len(selected) != 1:
        raise AssignmentPolicyMismatchError("exactly one admissible selected split candidate is required")
    return _copy(dict(selected[0]))


def selected_split_policy_profile() -> dict[str, Any]:
    candidates = evaluate_candidates(candidate_policy_definitions(), selection_policy_definition())
    selected = selected_candidate(candidates)
    return {
        "phase": "T-A5",
        "schema_version": ASSIGNMENT_SCHEMA_VERSION,
        "policy_id": SPLIT_POLICY_ID,
        "selection_policy_id": SPLIT_SELECTION_POLICY_ID,
        "selected_candidate_id": selected["candidate_id"],
        "selection_status": "SELECTED_WITH_LIMITATIONS",
        "source_partition_preservation": True,
        "source_domains_preserved": True,
        "random_or_hash_resplit": False,
        "assignment_seed": "NOT_APPLICABLE",
        "hash_assignment": "NOT_APPLICABLE",
        "grouping_unit_type": "OFFICIAL_SOURCE_PARTITION",
        "grouping_unit_id_schema": "dataset_id:source_split",
        "grouping_evidence_status": "SOURCE_PARTITION_ONLY_NO_SUBJECT_GROUP",
        "real_test_role": "REAL_EVAL_DEVELOPMENT",
        "pristine_locked_test_available": "NO",
        "t_b_final_unbiased_locked_test_claim": "NOT_AVAILABLE",
        "model_metrics_used": False,
        "immutability": {
            "policy_version": "1.0",
            "role_change_requires_new_policy_version": True,
            "derived_samples_inherit_parent_assignment": True,
        },
    }


def assignment_rule_contract() -> dict[str, Any]:
    return {
        "phase": "T-A5",
        "schema_version": ASSIGNMENT_SCHEMA_VERSION,
        "assignment_rule_id": ASSIGNMENT_RULE_ID,
        "split_policy_id": SPLIT_POLICY_ID,
        "source_dataset_id": DATASET_ID,
        "rule": "Preserve official SDT partition; test frames receive REAL_EVAL_DEVELOPMENT after prior-access audit.",
        "required_fields": [
            "source_dataset_id", "source_doi", "source_split", "source_domain", "source_member", "source_frame_index", "grouping_unit_type", "grouping_unit_id", "grouping_evidence_status", "safenest_assignment_role", "assignment_eligibility", "locked_test_eligibility", "prior_access_status", "assignment_rule_id", "t_a4_semantic_policy_id", "original_label_id", "original_label_name", "compatibility_target", "mapping_type", "mapping_rule_id",
        ],
        "forbidden_assignment_methods": ["FRAME_RANDOM", "STRATIFIED_FRAME_RANDOM", "FRAME_HASH", "FRAME_INDEX_BLOCK", "LABEL_AS_GROUP"],
        "role_by_source_split": {"train": "TRAIN", "validation": "VALIDATION", "test": "REAL_EVAL_DEVELOPMENT"},
        "locked_test_rule": "Never assign a source partition with prior geometry/semantic/preprocessing access to LOCKED_TEST.",
        "seed": "NOT_APPLICABLE",
        "hash_assignment": "NOT_APPLICABLE",
        "model_metrics_used": False,
    }


def access_type_classification(access_type: str) -> str:
    known = {
        "SOURCE_IDENTITY_AND_INVENTORY": "METADATA_OR_INVENTORY_ACCESS",
        "RAW_UNIT_AND_READER_PILOT": "RAW_SAMPLE_ACCESS",
        "GEOMETRY_SELECTION": "PREPROCESSING_SELECTION_ACCESS",
        "TEMPORAL_CAPABILITY_ANALYSIS": "TEMPORAL_DEVELOPMENT_ACCESS",
        "SEMANTIC_POLICY_SELECTION": "LABEL_SEMANTIC_SELECTION_ACCESS",
        "SOURCE_IDENTITY_ONLY": "METADATA_OR_INVENTORY_ACCESS",
        "SOURCE_SCHEMA_ONLY": "METADATA_OR_INVENTORY_ACCESS",
        "NO_PAYLOAD_ACCESS": "NO_PAYLOAD_ACCESS",
    }
    if access_type not in known:
        raise AssignmentPolicyMismatchError(f"unknown access type: {access_type}")
    return known[access_type]


def evaluate_locked_test_eligibility(source_split: str, access_history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if source_split not in SOURCE_SPLITS:
        raise AssignmentPolicyMismatchError(f"unknown source split: {source_split}")
    relevant = [row for row in access_history if row.get("source_split") == source_split]
    development = [row for row in relevant if row.get("decision_influenced") and row.get("source_domain") == "REAL"]
    if source_split == "test" and development:
        return {"eligible": False, "status": "DISQUALIFIED_BY_PRIOR_ACCESS", "reason": "USED_FOR_PREPROCESSING_GEOMETRY_SELECTION", "access_entries": len(relevant)}
    if source_split in {"train", "validation"}:
        return {"eligible": False, "status": "NOT_APPLICABLE_SOURCE_PARTITION", "reason": "Synthetic planned role is not a pristine SafeNest locked test.", "access_entries": len(relevant)}
    return {"eligible": False, "status": "NOT_VERIFIABLE", "reason": "No independent pristine holdout is present.", "access_entries": len(relevant)}


def assignment_for_real_test_frame(record: Mapping[str, Any]) -> dict[str, Any]:
    """Assign one verified T-A4 test record without changing its semantics."""

    required = ("source_frame_index", "source_member", "source_split", "dataset_id", "semantic_policy_id", "original_label_id", "original_label_name", "compatibility_target", "mapping_type", "mapping_rule_id")
    missing = [key for key in required if key not in record]
    if missing:
        raise AssignmentPolicyMismatchError(f"missing source/T-A4 fields: {','.join(missing)}")
    index = record["source_frame_index"]
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < REAL_TEST_FRAME_COUNT:
        raise AssignmentPolicyMismatchError("source_frame_index must identify exactly one of 8000 test members")
    if record["source_split"] != "test" or record["dataset_id"] != DATASET_ID:
        raise SourceDomainMismatchError("real assignment requires the selected SDT test source")
    expected_member = f"test/image_t_{index}.png"
    if record["source_member"] != expected_member:
        raise AssignmentPolicyMismatchError("source member/frame identity mismatch")
    if record["semantic_policy_id"] != SEMANTIC_POLICY_ID:
        raise AssignmentPolicyMismatchError("T-A4 semantic profile must be inherited")
    return {
        "source_dataset_id": DATASET_ID,
        "source_doi": DATASET_DOI,
        "source_split": "test",
        "source_domain": "REAL",
        "source_member": expected_member,
        "source_frame_index": index,
        "grouping_unit_type": "OFFICIAL_SOURCE_PARTITION",
        "grouping_unit_id": f"{DATASET_ID}:test",
        "grouping_evidence_status": "SOURCE_PARTITION_ONLY_NO_SUBJECT_GROUP",
        "safenest_assignment_role": "REAL_EVAL_DEVELOPMENT",
        "assignment_eligibility": "ELIGIBLE_REAL_DOMAIN_DEVELOPMENT_ONLY",
        "locked_test_eligibility": False,
        "locked_test_status": "DISQUALIFIED_BY_PRIOR_ACCESS",
        "prior_access_status": "USED_FOR_T_A0_T_A4_DEVELOPMENT_AND_GEOMETRY_SELECTION",
        "assignment_rule_id": ASSIGNMENT_RULE_ID,
        "split_policy_id": SPLIT_POLICY_ID,
        "assignment_seed": "NOT_APPLICABLE",
        "hash_assignment": "NOT_APPLICABLE",
        "split_assignment_status": "ASSIGNED_T_A5_IMMUTABLE",
        "t_a4_semantic_policy_id": SEMANTIC_POLICY_ID,
        "original_label_id": record["original_label_id"],
        "original_label_name": record["original_label_name"],
        "compatibility_target": record["compatibility_target"],
        "mapping_type": record["mapping_type"],
        "mapping_rule_id": record["mapping_rule_id"],
        "claim_scope": _copy(record.get("claim_scope", [])),
        "source_archive_path": SOURCE_ARCHIVE_PATH,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
    }


def validate_assignment_record(record: Mapping[str, Any]) -> None:
    role = record.get("safenest_assignment_role")
    if role not in SAFE_NEST_ROLES:
        raise UnknownAssignmentRoleError(str(role))
    if role != "REAL_EVAL_DEVELOPMENT":
        raise AssignmentPolicyMismatchError("this record validator is for the materialized real test inventory")
    if record.get("source_split") != "test" or record.get("source_domain") != "REAL":
        raise SourceDomainMismatchError("real test record has wrong source split/domain")
    if record.get("grouping_unit_type") in {"FRAME_INDEX", "LABEL", "FRAME", "FRAME_BLOCK"}:
        raise GroupingProvenanceUnavailableError("frame index and labels cannot be grouping units")
    if record.get("grouping_evidence_status") != "SOURCE_PARTITION_ONLY_NO_SUBJECT_GROUP":
        raise GroupingProvenanceUnavailableError("unsupported or unverified grouping evidence")
    if record.get("grouping_unit_id") != f"{DATASET_ID}:test":
        raise GroupingProvenanceUnavailableError("grouping unit must be the official source partition")
    if record.get("locked_test_eligibility") is not False or record.get("locked_test_status") != "DISQUALIFIED_BY_PRIOR_ACCESS":
        raise LockedTestAccessError("prior-access test frame cannot be locked test")
    if record.get("assignment_rule_id") != ASSIGNMENT_RULE_ID or record.get("split_policy_id") != SPLIT_POLICY_ID:
        raise AssignmentPolicyMismatchError("assignment rule or split policy mismatch")
    if record.get("assignment_seed") != "NOT_APPLICABLE" or record.get("hash_assignment") != "NOT_APPLICABLE":
        raise AssignmentPolicyMismatchError("random/hash assignment is not allowed")
    if record.get("t_a4_semantic_policy_id") != SEMANTIC_POLICY_ID:
        raise AssignmentPolicyMismatchError("T-A4 semantic inheritance missing")
    index = record.get("source_frame_index")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < REAL_TEST_FRAME_COUNT:
        raise AssignmentPolicyMismatchError("invalid source frame index")
    if record.get("source_member") != f"test/image_t_{index}.png":
        raise AssignmentPolicyMismatchError("source member/frame mismatch")


def validate_assignment_inventory(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(records) != REAL_TEST_FRAME_COUNT:
        raise AssignmentPolicyMismatchError(f"expected {REAL_TEST_FRAME_COUNT} real test records, got {len(records)}")
    indices = []
    members = []
    groups: dict[str, str] = {}
    for record in records:
        validate_assignment_record(record)
        indices.append(record["source_frame_index"])
        members.append(record["source_member"])
        group = str(record["grouping_unit_id"])
        role = str(record["safenest_assignment_role"])
        if group in groups and groups[group] != role:
            raise GroupCrossSplitError(f"group {group} crosses roles")
        groups[group] = role
    if sorted(indices) != list(range(REAL_TEST_FRAME_COUNT)) or len(set(indices)) != len(indices):
        raise AssignmentPolicyMismatchError("real test frame indices must be exactly 0..7999 once")
    if len(set(members)) != REAL_TEST_FRAME_COUNT:
        raise AssignmentPolicyMismatchError("source members must be unique")
    return {"record_count": len(records), "unique_frame_count": len(set(indices)), "unique_member_count": len(set(members)), "cross_role_group_count": 0}


def validate_derived_assignment(parent: Mapping[str, Any], child: Mapping[str, Any]) -> None:
    for field in ("source_dataset_id", "source_split", "source_domain", "source_member", "source_frame_index", "grouping_unit_type", "grouping_unit_id", "safenest_assignment_role", "assignment_rule_id", "split_policy_id"):
        if child.get(field) != parent.get(field):
            raise DerivedAssignmentMismatchError(f"derived sample does not inherit {field}")
    if child.get("augmentation") and parent.get("safenest_assignment_role") != "TRAIN":
        raise DerivedAssignmentMismatchError("augmentation is TRAIN-only")


def reject_unsupported_split_request(request: Mapping[str, Any]) -> None:
    method = str(request.get("method", "")).upper()
    if method in {"RANDOM", "STRATIFIED_RANDOM", "FRAME_RANDOM", "SHUFFLE_RATIO"}:
        raise FrameRandomSplitError("frame-random split is prohibited without independent grouping")
    if method in {"HASH", "FRAME_HASH", "MEMBER_HASH"}:
        raise FrameHashSplitError("frame/member hash split is prohibited without independent grouping")
    if request.get("grouping_unit_type") in {"FRAME_INDEX", "LABEL", "FRAME_BLOCK"}:
        raise GroupingProvenanceUnavailableError("arbitrary frame/label grouping is not provenance")
    if request.get("safenest_assignment_role") == "LOCKED_TEST" and request.get("prior_access_status") not in {"UNUSED", "NOT_ACCESSED"}:
        raise LockedTestAccessError("renaming an accessed partition does not make it pristine")


def validate_generalization_claim(claim: str, grouping: Mapping[str, Any] | None = None) -> None:
    if claim.upper() in {"SUBJECT_WISE", "SESSION_WISE", "EVENT_WISE", "SCENE_WISE", "CAMERA_WISE"}:
        profile = grouping or grouping_evidence_definition()
        if profile.get("generalization_performance") == "NOT_VERIFIABLE":
            raise UnverifiedGeneralizationClaimError("group generalization is not verifiable")


__all__ = [name for name in globals() if not name.startswith("_")]
