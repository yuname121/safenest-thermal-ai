#!/usr/bin/env python3
"""Generate compact, deterministic Thermal T-A5 grouping evidence."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.thermal.split_policy import (  # noqa: E402
    ASSIGNMENT_RULE_ID,
    DATASET_DOI,
    DATASET_ID,
    DATASET_NAME,
    GROUPING_POLICY_ID,
    OFFICIAL_DOCUMENTATION_URL,
    OFFICIAL_SOURCE_URL,
    REAL_TEST_FRAME_COUNT,
    SEMANTIC_POLICY_ID,
    SOURCE_ARCHIVE_PATH,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_PARTITION_CONTRACT_ID,
    SPLIT_POLICY_ID,
    SPLIT_SELECTION_POLICY_ID,
    TEMPORAL_POLICY_ID,
    access_history_definition,
    assignment_for_real_test_frame,
    assignment_rule_contract,
    candidate_policy_definitions,
    evaluate_candidates,
    grouping_evidence_definition,
    selected_candidate,
    selected_split_policy_profile,
    selection_policy_definition,
    source_partition_definitions,
    validate_assignment_inventory,
)


EVIDENCE_REL = "datasets/thermal/manifests/T-A5_grouping_immutable_split"
EVIDENCE_DIR = ROOT / EVIDENCE_REL
REPORT_REL = "docs/reports/20260811_Codex_T-A5_Thermal_Grouping_Immutable_Split_01.md"
ACCESS_DATE = "2026-08-11"
JSON_NAMES = [
    "assignment_rule_contract.json",
    "augmentation_inheritance_policy.json",
    "data_access_history.json",
    "grouping_evidence_registry.json",
    "leakage_policy.json",
    "limitations.json",
    "locked_test_eligibility.json",
    "real_test_assignment_inventory.json",
    "selected_split_policy.json",
    "source_partition_contract.json",
    "split_distribution_summary.json",
    "split_policy_candidates.json",
    "split_selection_policy.json",
    "validation_result.json",
]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def _build_assignments() -> list[dict[str, Any]]:
    inventory = _read_json("datasets/thermal/manifests/T-A4_label_semantics_ambiguity/label_mapping_inventory.json")
    records = inventory.get("records", [])
    if len(records) != REAL_TEST_FRAME_COUNT:
        raise RuntimeError("T-A4 label mapping inventory must contain all 8000 test records")
    assignments = [assignment_for_real_test_frame(record) for record in records]
    assignments.sort(key=lambda row: row["source_frame_index"])
    validate_assignment_inventory(assignments)
    return assignments


def _access_history() -> dict[str, Any]:
    history = access_history_definition()
    history["access_date"] = ACCESS_DATE
    return history


def _source_contract() -> dict[str, Any]:
    partitions = source_partition_definitions()
    return {
        "phase": "T-A5",
        "schema_version": "1.0",
        "contract_id": SOURCE_PARTITION_CONTRACT_ID,
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "doi": DATASET_DOI,
        "official_source_url": OFFICIAL_SOURCE_URL,
        "official_documentation_url": OFFICIAL_DOCUMENTATION_URL,
        "original_publication_status": "OFFICIAL_SOURCE_DOCUMENTATION_VERIFIED",
        "source_partition_preservation": True,
        "partitions": partitions,
        "partition_counts": {row["source_split"]: row["official_sample_count"] for row in partitions},
        "partition_domains": {row["source_split"]: row["source_domain"] for row in partitions},
        "official_partition_total": 48000,
        "source_partitions_must_remain_separate": True,
        "no_random_or_hash_resplit": True,
        "source_archive": {"path": SOURCE_ARCHIVE_PATH, "sha256": SOURCE_ARCHIVE_SHA256, "test_member_count": REAL_TEST_FRAME_COUNT},
        "train_validation_sample_level_inventory": "PENDING_MATERIALIZATION",
        "real_test_inventory": "FULL_MEMBER_LABEL_ASSIGNMENT_INVENTORY_VERIFIED",
        "model_metrics_used": False,
    }


def _locked_test(history: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": "T-A5",
        "schema_version": "1.0",
        "source_dataset_id": DATASET_ID,
        "source_split": "test",
        "source_domain": "REAL",
        "pristine_locked_test_eligible": False,
        "status": "DISQUALIFIED_BY_PRIOR_ACCESS",
        "disqualification_reason": "USED_FOR_PREPROCESSING_GEOMETRY_SELECTION",
        "access_history_entry_count": sum(row["source_split"] == "test" for row in history["entries"]),
        "access_types": sorted({row["access_type"] for row in history["entries"] if row["source_split"] == "test"}),
        "geometry_selection_access": True,
        "semantic_policy_selection_access": True,
        "current_pristine_locked_test_available": "NO",
        "no_renaming_can_restore_pristine_status": True,
        "future_t_b_final_unbiased_lock": "NOT_AVAILABLE_UNTIL_INDEPENDENT_HOLDOUT",
        "train_partition_locked_test_status": "NOT_APPLICABLE_SYNTHETIC_TRAIN_PARTITION",
        "validation_partition_locked_test_status": "NOT_APPLICABLE_SYNTHETIC_VALIDATION_PARTITION",
    }


def _distribution(assignments: list[dict[str, Any]]) -> dict[str, Any]:
    def counts(field: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for row in assignments:
            key = str(row[field])
            result[key] = result.get(key, 0) + 1
        return {key: result[key] for key in sorted(result)}
    return {
        "phase": "T-A5",
        "schema_version": "1.0",
        "policy_id": SPLIT_POLICY_ID,
        "source_partition_preserved": True,
        "roles": {
            "TRAIN": {"source_split": "train", "source_domain": "SYNTHETIC", "count": 32000, "materialization": "LOCAL_CLOUD_PLACEHOLDER", "sample_inventory": "SAMPLE_LEVEL_INVENTORY_PENDING_MATERIALIZATION", "grouping": "NOT_VERIFIABLE"},
            "VALIDATION": {"source_split": "validation", "source_domain": "SYNTHETIC", "count": 8000, "materialization": "LOCAL_CLOUD_PLACEHOLDER", "sample_inventory": "SAMPLE_LEVEL_INVENTORY_PENDING_MATERIALIZATION", "grouping": "NOT_VERIFIABLE"},
            "REAL_EVAL_DEVELOPMENT": {"source_split": "test", "source_domain": "REAL", "count": len(assignments), "materialization": "LOCALLY_MATERIALIZED", "sample_inventory": "FULL_MEMBER_AND_LABEL_INVENTORY_VERIFIED", "grouping": "OFFICIAL_SOURCE_PARTITION_ONLY", "original_label_counts": counts("original_label_name"), "compatibility_target_counts": counts("compatibility_target")},
            "LOCKED_TEST": {"source_split": "NONE", "source_domain": "NONE", "count": 0, "materialization": "NOT_AVAILABLE", "sample_inventory": "NOT_AVAILABLE", "grouping": "NOT_AVAILABLE"},
        },
        "total_planned_samples": 40000,
        "real_test_frame_count": len(assignments),
        "subject_count": "NOT_VERIFIABLE",
        "session_count": "NOT_VERIFIABLE",
        "sequence_count": "NOT_VERIFIABLE",
        "event_count": "NOT_VERIFIABLE",
        "scene_count": "NOT_VERIFIABLE",
        "camera_count": "NOT_VERIFIABLE",
        "model_metrics_used": False,
    }


def _inheritance() -> dict[str, Any]:
    return {
        "phase": "T-A5",
        "schema_version": "1.0",
        "policy_id": SPLIT_POLICY_ID,
        "derived_artifact_rule": "Every canonical, normalized, augmented, or quantized sample inherits source split/domain/group/role and assignment rule; no reassignment.",
        "augmentation_rule": "AUGMENTATION_TRAIN_ONLY",
        "augmentation_parent_role_required": "TRAIN",
        "canonical_conversion_inherits_assignment": True,
        "normalization_inherits_assignment": True,
        "quantization_inherits_assignment": True,
        "role_change_requires_new_policy_version": True,
        "no_augmentations_generated_in_t_a5": True,
    }


def _leakage() -> dict[str, Any]:
    return {
        "phase": "T-A5",
        "schema_version": "1.0",
        "policy_id": SPLIT_POLICY_ID,
        "taxonomy": {
            "exact_source_member_overlap": "PROVEN_ZERO_FOR_T_A5_REAL_ASSIGNMENT_INVENTORY",
            "source_frame_id_overlap": "PROVEN_ZERO_FOR_T_A5_REAL_ASSIGNMENT_INVENTORY",
            "subject_overlap": "NOT_VERIFIABLE_NO_SUBJECT_ID",
            "session_overlap": "NOT_VERIFIABLE_NO_SESSION_ID",
            "recording_overlap": "NOT_VERIFIABLE_NO_RECORDING_ID",
            "event_overlap": "NOT_VERIFIABLE_NO_EVENT_ID",
            "sequence_overlap": "NOT_VERIFIABLE_NO_SEQUENCE_ID",
            "scene_overlap": "NOT_VERIFIABLE_NO_AUTHORITATIVE_SCENE_KEY",
            "camera_overlap": "NOT_VERIFIABLE_NO_PER_FRAME_CAMERA_KEY",
            "derived_artifact_overlap": "CONTROLLED_BY_INHERITANCE_RULE",
            "exact_duplicate_content": "DEFERRED_T_A6",
            "near_duplicate_content": "DEFERRED_T_A6",
            "unknown_dependency": "NOT_VERIFIABLE",
        },
        "cross_role_member_overlap_count": 0,
        "cross_role_frame_overlap_count": 0,
        "generalization_claim": "NOT_VERIFIABLE",
        "near_duplicate_audit": "DEFERRED_T_A6",
        "model_metrics_used": False,
    }


def _limitations() -> dict[str, Any]:
    return {
        "phase": "T-A5",
        "schema_version": "1.0",
        "status": "PASS_WITH_LIMITATIONS",
        "limitations": [
            "No subject/session/recording/event/sequence IDs are distributed; subject/event generalization is NOT_VERIFIABLE.",
            "Scene and camera are not defensible per-frame fallback groups.",
            "The real SDT test partition was used for T-A2 geometry selection and cannot be a pristine LOCKED_TEST.",
            "Train and validation archive bytes are cloud placeholders; only partition-level planned roles are established.",
            "The legacy processed_thermal_80x62.npz is not split authority and has lost source provenance.",
            "Exact/near duplicate audit and full derived-sample accounting are deferred to T-A6.",
            "A future independent holdout is required before an unbiased T-B LOCKED_TEST claim.",
        ],
        "downstream": {"T-A6": "full canonical conversion and integrity audit", "T-B": "conditional development evaluation; no final unbiased locked-test claim", "T-C": "Thermal-44 hardware semantics", "T-D": "gap-driven expansion if needed"},
        "large_download_authorization_required": True,
        "t_a6_full_completion_requires_placeholder_hydration": True,
        "t_a1_authorization_inherited": True,
    }


def _report(data: dict[str, Any]) -> str:
    chosen = data["selected_split_policy.json"]["selected_candidate_id"]
    return f"""# Thermal T-A5 — Grouping, Leakage-Resistant Split, and Immutable Assignment Policy

## Decision

T-A5 selects `{chosen}` under `{SPLIT_POLICY_ID}`.  The official SDT train, validation, and test boundaries are preserved; the real test partition is assigned only to `REAL_EVAL_DEVELOPMENT` because it was used for T-A2 geometry selection and subsequent T-A3/T-A4 development evidence.  No pristine Thermal `LOCKED_TEST` currently exists.

The selected source is the [SDT Dataset official documentation]({OFFICIAL_DOCUMENTATION_URL}) and [Zenodo distribution record]({OFFICIAL_SOURCE_URL}) ({DATASET_DOI}).  Official documentation distinguishes 32,000 synthetic train images, 8,000 synthetic validation images, and 8,000 real test images.  The local T-A1 archive inventory independently verifies the materialized real test members and labels.

## Grouping and access

No authoritative subject, session, recording, sequence, event, timestamp, scene, or per-frame camera identifier is available.  Frame index and label are provenance fields only and are never used as groups.  Consequently subject/session/event generalization is `NOT_VERIFIABLE`; a frame-random or frame-hash resplit is rejected.  The strongest verified unit is the official source partition.

T-A0 and T-A1 established source identity and bounded reader evidence.  T-A2 used a 48-frame real test pilot to compare and select geometry; T-A3 reused that pilot for temporal capability analysis; T-A4 reused the pilot and audited all 8,000 labels for semantic policy.  This history disqualifies the real test partition from pristine locked-test status even though it remains useful for development evaluation.

## Assignment roles

| Official partition | Domain | SafeNest role | Materialization | Count |
|---|---|---|---|---:|
| train | SYNTHETIC | TRAIN (planned) | LOCAL_CLOUD_PLACEHOLDER | 32,000 |
| validation | SYNTHETIC | VALIDATION (planned) | LOCAL_CLOUD_PLACEHOLDER | 8,000 |
| test | REAL | REAL_EVAL_DEVELOPMENT | LOCALLY_MATERIALIZED | 8,000 |
| independent holdout | — | LOCKED_TEST | NOT_AVAILABLE | 0 |

Every real test assignment preserves its source member/frame identity and inherits the T-A4 semantic policy.  No random seed, hash assignment, canonical tensor, augmentation, or model metric is introduced.  Derived artifacts must inherit the parent assignment; TRAIN-only augmentation is deferred to later phases.

## Limitations and gate

The T-A5 contract is `PASS_WITH_LIMITATIONS`.  T-A6 is authorized for policy/integrity work, but its full completion requires explicit authorization to hydrate the multi-gigabyte train/validation placeholders.  T-A6 does not create an unbiased final holdout.  Until an independent holdout exists, T-B may use the real test only conditionally for development evaluation.

Machine-readable evidence is under `{EVIDENCE_REL}/`; its checksum registry covers every JSON artifact.  The standalone validator independently recomputes candidate admissibility and selection, rechecks predecessor gates, verifies all 8,000 real assignments, and rejects tampering, absolute paths, frame-random/hash splitting, and retroactive locked-test claims.
"""


def build_artifacts(root: Path = ROOT) -> dict[str, Any]:
    global ROOT, EVIDENCE_DIR
    ROOT = root
    EVIDENCE_DIR = root / EVIDENCE_REL
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    assignments = _build_assignments()
    candidates = candidate_policy_definitions()
    selection_policy = selection_policy_definition()
    evaluated = evaluate_candidates(candidates, selection_policy)
    selected = selected_candidate(evaluated)
    history = _access_history()
    artifacts: dict[str, Any] = {
        "grouping_evidence_registry.json": grouping_evidence_definition(),
        "split_policy_candidates.json": {"phase": "T-A5", "schema_version": "1.0", "policy_id": SPLIT_SELECTION_POLICY_ID, "candidates": evaluated, "model_metrics_used": False},
        "split_selection_policy.json": selection_policy,
        "selected_split_policy.json": {**selected_split_policy_profile(), "selected_candidate": selected, "selection_derivation": "Independent candidate admissibility and lexicographic ranking; winner is not predeclared."},
        "source_partition_contract.json": _source_contract(),
        "data_access_history.json": history,
        "locked_test_eligibility.json": _locked_test(history),
        "assignment_rule_contract.json": assignment_rule_contract(),
        "real_test_assignment_inventory.json": {"phase": "T-A5", "schema_version": "1.0", "dataset_id": DATASET_ID, "source_split": "test", "source_domain": "REAL", "assignment_rule_id": ASSIGNMENT_RULE_ID, "split_policy_id": SPLIT_POLICY_ID, "record_count": len(assignments), "records": assignments},
        "augmentation_inheritance_policy.json": _inheritance(),
        "leakage_policy.json": _leakage(),
        "split_distribution_summary.json": _distribution(assignments),
        "limitations.json": _limitations(),
    }
    for name in JSON_NAMES:
        if name == "validation_result.json":
            continue
        (EVIDENCE_DIR / name).write_text(canonical_json(artifacts[name]), encoding="utf-8")
    from scripts.validate_thermal_t_a5 import validate_evidence
    result = validate_evidence(repo_root=root, evidence_dir=EVIDENCE_DIR, check_checksums=False, verify_real_payload=False)
    (EVIDENCE_DIR / "validation_result.json").write_text(canonical_json(result), encoding="utf-8")
    artifacts["validation_result.json"] = result
    checksum_lines = []
    for name in sorted(JSON_NAMES):
        path = EVIDENCE_DIR / name
        checksum_lines.append(f"{sha256_bytes(path.read_bytes())}  {EVIDENCE_REL}/{name}")
    (EVIDENCE_DIR / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    (root / REPORT_REL).write_text(_report(artifacts), encoding="utf-8")
    return artifacts


if __name__ == "__main__":
    build_artifacts()
    print(json.dumps({"phase": "T-A5", "evidence_dir": EVIDENCE_REL, "report": REPORT_REL}, sort_keys=True))
