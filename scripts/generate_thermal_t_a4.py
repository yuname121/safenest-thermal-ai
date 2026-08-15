#!/usr/bin/env python3
"""Generate deterministic Thermal T-A4 semantic and proxy evidence."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.thermal.label_semantics import (  # noqa: E402
    DATASET_DOI,
    DATASET_ID,
    RUNTIME_CLASS_MAP,
    SEMANTIC_POLICY_ID,
    SOURCE_ARCHIVE_PATH,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_LABELS,
    SOURCE_SPLIT,
    T_A2_PROFILE_ID,
    T_A3_POLICY_ID,
    candidate_policy_definitions,
    evaluate_candidates,
    map_source_label,
    selected_candidate,
    selection_policy_definition,
    semantic_policy_profile,
)
from datasets.thermal.raw_reader import SDTThermalRawReader  # noqa: E402


EVIDENCE_REL = "datasets/thermal/manifests/T-A4_label_semantics_ambiguity"
EVIDENCE_DIR = ROOT / EVIDENCE_REL
REPORT_REL = "docs/reports/20260810_Codex_T-A4_Thermal_Label_Semantics_Proxy_Mapping_Ambiguity_01.md"
JSON_NAMES = [
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
    "validation_result.json",
]
ACCESS_DATE = "2026-08-10"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _runtime_model_audit(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "models/model_manifest.json").read_text(encoding="utf-8"))
    thermal = manifest["models"]["thermal"]
    model_path = root / thermal["path"]
    measured_sha = sha256_file(model_path)
    measured_size = model_path.stat().st_size
    if measured_sha != thermal["sha256"] or measured_size != thermal["size_bytes"]:
        raise RuntimeError("current Thermal model artifact does not match its manifest")
    class_map = {int(key): value for key, value in thermal["class_map"].items()}
    return {
        "model_id": thermal["model_id"],
        "version": thermal["version"],
        "path": thermal["path"],
        "sha256": measured_sha,
        "size_bytes": measured_size,
        "class_map": {str(key): value for key, value in sorted(class_map.items())},
        "role": thermal.get("role"),
        "status": "LEGACY_OR_CURRENT_RUNTIME_CLASS_MAP_NOT_SOURCE_GROUND_TRUTH",
        "model_metrics_used": False,
    }


def _activity_registry(source_counts: dict[str, int]) -> dict[str, Any]:
    represented = {
        "LYING": "REPRESENTED_BY_VERIFIED_SOURCE_LABEL",
        "SITTING": "REPRESENTED_BY_VERIFIED_SOURCE_LABEL",
        "STANDING": "REPRESENTED_BY_VERIFIED_SOURCE_LABEL",
        "EMPTY_ROOM": "REPRESENTED_BY_VERIFIED_SOURCE_LABEL",
    }
    unsupported = {
        "BENDING": "NOT_REPRESENTED_BY_SELECTED_SOURCE",
        "KNEELING": "NOT_REPRESENTED_BY_SELECTED_SOURCE",
        "ENTERING": "NOT_REPRESENTED_BY_SELECTED_SOURCE",
        "EXITING": "NOT_REPRESENTED_BY_SELECTED_SOURCE",
        "WALKING": "NOT_REPRESENTED_BY_SELECTED_SOURCE",
        "PARTIAL_BODY": "NOT_VERIFIABLE_FROM_SELECTED_SOURCE_LABELS",
        "FALL_TRANSITION": "NOT_VERIFIABLE_NO_TEMPORAL_EVENT",
        "FALL_IMPACT": "NOT_VERIFIABLE_NO_TEMPORAL_EVENT",
        "POST_FALL_INTERVAL": "NOT_VERIFIABLE_NO_TEMPORAL_EVENT",
        "RECOVERY": "NOT_VERIFIABLE_NO_TEMPORAL_EVENT",
        "AMBIGUOUS_BOUNDARY_FRAME": "NOT_VERIFIABLE_NO_TEMPORAL_EVENT",
    }
    entries = {}
    for name, status in sorted({**represented, **unsupported}.items()):
        entries[name] = {
            "status": status,
            "source_count": source_counts.get(str({"LYING": 0, "SITTING": 1, "STANDING": 2, "EMPTY_ROOM": 3}.get(name, -1)), 0),
            "treated_as_negative": False,
            "claim_scope": "SOURCE_LABEL_OR_TEMPORAL_EVIDENCE_ONLY",
        }
    return {
        "phase": "T-A4",
        "schema_version": "1.0",
        "source_dataset_id": DATASET_ID,
        "entries": entries,
        "natural_fall_event_status": "NOT_REPRESENTED_OR_NOT_VERIFIABLE",
        "staged_fall_event_status": "NOT_REPRESENTED_OR_NOT_VERIFIABLE",
        "unsupported_activity_policy": "ABSENCE_OF_A_LABEL_IS_NOT_A_NEGATIVE_EXAMPLE",
        "hard_negative_coverage": "NOT_ESTABLISHED_FOR_UNSUPPORTED_ACTIVITIES",
    }


def _mapping_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    def count(field: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for record in records:
            key = str(record[field])
            result[key] = result.get(key, 0) + 1
        return {key: result[key] for key in sorted(result)}

    target_counts: dict[str, int] = {}
    for record in records:
        target = str(record["compatibility_target"])
        target_counts[target] = target_counts.get(target, 0) + 1
    return {
        "source_label_counts": count("original_label_name"),
        "mapping_type_counts": count("mapping_type"),
        "mapping_rule_counts": count("mapping_rule_id"),
        "fall_evidence_strength_counts": count("fall_evidence_strength"),
        "ambiguity_status_counts": count("ambiguity_status"),
        "assignment_status_counts": count("assignment_status"),
        "compatibility_target_counts": {key: target_counts[key] for key in sorted(target_counts)},
        "ambiguous_target_mapping_count": sum(record["mapping_type"] == "AMBIGUOUS_TARGET" for record in records),
        "unsupported_mapping_count": sum(record["mapping_type"] in {"UNSUPPORTED_MAPPING", "NOT_APPLICABLE"} for record in records),
        "direct_verified_fall_mapping_count": sum(
            record["mapping_type"] == "DIRECT_SOURCE_EQUIVALENT" and record["compatibility_target"] == "HUMAN_FALL"
            for record in records
        ),
        "assignment_exclusion_count": sum(record["assignment_status"] != "ELIGIBLE_FOR_LATER_PROXY_LABEL_CONSIDERATION" for record in records),
        "claim_scope_counts": {
            scope: sum(scope in record["claim_scope"] for record in records)
            for scope in sorted({scope for record in records for scope in record["claim_scope"]})
        },
        "source_label_modified_count": sum(record["source_label_modified"] is not False for record in records),
        "worker_safety_ground_truth_count": sum(record["worker_safety_ground_truth"] is not False for record in records),
        "fall_event_ground_truth_count": sum(record["fall_event_semantic_status"] != "NOT_VERIFIABLE" for record in records),
    }


def build_artifacts(root: Path = ROOT) -> dict[str, Any]:
    reader = SDTThermalRawReader(repo_root=root)
    inventory = reader.inspect_archive()
    labels = getattr(reader, "_labels", None)
    if labels is None or len(labels) != 8000:
        raise RuntimeError("SDT labels.txt was not deterministically loaded")
    class_counts = {str(key): int(value) for key, value in sorted(inventory["class_counts"].items())}
    if class_counts != {"0": 2000, "1": 2000, "2": 2000, "3": 2000}:
        raise RuntimeError(f"unexpected SDT label counts: {class_counts}")
    mapping_records: list[dict[str, Any]] = []
    for label in labels:
        frame_index = label.source_frame_index
        member_index = reader._thermal_info[frame_index][0]
        mapping_records.append(
            map_source_label(
                {
                    "dataset_id": DATASET_ID,
                    "source_doi": DATASET_DOI,
                    "source_split": SOURCE_SPLIT,
                    "source_archive_path": SOURCE_ARCHIVE_PATH,
                    "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                    "source_member": f"test/image_t_{frame_index}.png",
                    "source_member_index": member_index,
                    "source_frame_index": frame_index,
                    "original_label_id": label.source_pose_label,
                    "original_label_name": label.source_pose_name,
                    "original_bbox": list(label.source_bbox),
                }
            )
        )
    mapping_records.sort(key=lambda item: item["source_frame_index"])
    selection_policy = selection_policy_definition()
    candidates = candidate_policy_definitions()
    evaluated = evaluate_candidates(candidates, selection_policy)
    selected = selected_candidate(evaluated)
    policy_profile = semantic_policy_profile(selected["candidate_id"])
    policy_content_sha = sha256_bytes(canonical_json(policy_profile).encode("utf-8"))
    selection_policy_sha = sha256_bytes(canonical_json(selection_policy).encode("utf-8"))
    runtime_audit = _runtime_model_audit(root)
    t_a3 = json.loads((root / "datasets/thermal/manifests/T-A3_sequence_window_event_policy/pilot_temporal_summary.json").read_text(encoding="utf-8"))
    t_a3_records = {int(record["source_frame_index"]): record for record in t_a3["records"]}
    pilot_indices = sorted(t_a3_records)
    pilot_records = []
    for index in pilot_indices:
        mapping = mapping_records[index]
        pilot = {
            **mapping,
            "t_a1_raw_encoded_frame_sha256": t_a3_records[index]["t_a1_raw_encoded_frame_sha256"],
            "t_a2_canonical_frame_hash": t_a3_records[index]["canonical_frame_hash"],
            "t_a2_geometry_profile_id": t_a3_records[index]["t_a2_geometry_profile_id"],
            "t_a3_temporal_policy_id": T_A3_POLICY_ID,
            "source_timestamp_status": t_a3_records[index]["source_timestamp_status"],
            "source_fps_status": t_a3_records[index]["source_fps_status"],
            "sequence_id_status": t_a3_records[index]["sequence_id_status"],
            "event_id_status": t_a3_records[index]["event_id_status"],
        }
        pilot_records.append(pilot)
    source_contract = {
        "phase": "T-A4",
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "doi": DATASET_DOI,
        "source_split": SOURCE_SPLIT,
        "archive_path": SOURCE_ARCHIVE_PATH,
        "archive_sha256": SOURCE_ARCHIVE_SHA256,
        "original_labels": {str(key): value for key, value in sorted(SOURCE_LABELS.items())},
        "source_label_counts": class_counts,
        "source_label_status": "VERIFIED_IMMUTABLE_SOURCE_ANNOTATION",
        "source_label_fields": ["original_label_id", "original_label_name", "original_bbox", "source_frame_index", "source_member"],
        "bbox_semantics": "labels.txt source bbox preserved as provenance/geometry evidence; never used to redefine posture or fall semantics",
        "source_labels_modified": False,
        "unknown_source_label_behavior": "FAIL_CLOSED_SOURCE_LABEL_UNKNOWN",
    }
    frame_contract = {
        "phase": "T-A4",
        "schema_version": "1.0",
        "layer": "B_FRAME_LEVEL_EVIDENCE_SEMANTIC",
        "source_annotation_remains_separate": True,
        "evidence_labels": {
            "LYING": "HUMAN_LYING_POSTURE",
            "SITTING": "HUMAN_SITTING_POSTURE",
            "STANDING": "HUMAN_STANDING_POSTURE",
            "EMPTY_ROOM": "NO_ANNOTATED_HUMAN_IN_REPRESENTED_FRAME",
        },
        "claim_scope": ["FRAME_LEVEL_ONLY", "NOT_TEMPORAL_EVENT_GROUND_TRUTH", "NOT_SAFETY_GROUND_TRUTH"],
        "fall_event_ground_truth": "NOT_VERIFIABLE",
        "temporal_event_status": "NOT_VERIFIABLE",
        "worker_safety_ground_truth": "NOT_SUPPORTED",
        "bbox_relabeling": "PROHIBITED",
    }
    compatibility_contract = {
        "phase": "T-A4",
        "schema_version": "1.0",
        "layer": "C_OPTIONAL_RUNTIME_COMPATIBILITY_PROXY",
        "enabled": True,
        "runtime_model_reference": runtime_audit,
        "source_ground_truth_warning": "Runtime class names are not source-label ground truth.",
        "mapping_rule_ids": {
            "LYING": "THERMAL_MAP_LYING_TO_FALL_COMPAT_PROXY_001",
            "SITTING": "THERMAL_MAP_SITTING_TO_NON_LYING_PROXY_001",
            "STANDING": "THERMAL_MAP_STANDING_TO_NON_LYING_PROXY_001",
            "EMPTY_ROOM": "THERMAL_MAP_EMPTY_ROOM_TO_NO_HUMAN_001",
        },
        "proxy_claim_limits": {
            "LYING": "FRAME_LEVEL_FALL_COMPATIBILITY_PROXY_ONLY",
            "SITTING": "NON_LYING_POSTURE_COMPATIBILITY_ONLY",
            "STANDING": "NON_LYING_POSTURE_COMPATIBILITY_ONLY",
            "EMPTY_ROOM": "FRAME_LEVEL_PRESENCE_ONLY",
        },
        "safety_interpretation": "No compatibility target proves general worker safety or emergency state.",
        "model_metrics_used": False,
    }
    ambiguity = {
        "phase": "T-A4",
        "schema_version": "1.0",
        "source_label_ambiguity": {name: False for name in SOURCE_LABELS.values()},
        "fall_interpretation": {
            "LYING": "AMBIGUOUS_OR_NOT_VERIFIABLE_FOR_FALL_EVENT",
            "SITTING": "NOT_VERIFIABLE_FOR_FALL_EVENT",
            "STANDING": "NOT_VERIFIABLE_FOR_FALL_EVENT",
            "EMPTY_ROOM": "NOT_VERIFIABLE_FOR_FALL_EVENT",
        },
        "transition_frame_assignment": "NOT_APPLICABLE_NO_VERIFIED_TEMPORAL_EVENT",
        "boundary_frame_assignment": "NOT_APPLICABLE_NO_VERIFIED_TEMPORAL_EVENT",
        "unknown_label_policy": "FAIL_CLOSED; NEVER_COERCE_TO_NORMAL",
        "unsupported_target_policy": "COMPATIBILITY_TARGET_NULL_AND_UNSUPPORTED_MAPPING_IF_NO_DEFENSIBLE_RULE",
        "missing_mapping_default_behavior": "FAIL_CLOSED; NO_DEFAULT_NORMAL",
        "ambiguity_dimension_rule": "Keep clear source annotation VERIFIED while marking fall interpretation independently ambiguous/not verifiable.",
    }
    claim_scopes = {
        "phase": "T-A4",
        "schema_version": "1.0",
        "scopes": {
            "SOURCE_POSTURE_ONLY": "Original SDT posture/presence annotation only.",
            "FRAME_LEVEL_PRESENCE_ONLY": "No annotated human in the represented frame; no outside-FOV claim.",
            "FRAME_LEVEL_POSTURE_PROXY": "Frame-level posture compatibility only; not safety state.",
            "FRAME_LEVEL_FALL_COMPATIBILITY_PROXY_ONLY": "Frame-level compatibility with historical fall output; not a fall event.",
            "NOT_TEMPORAL_EVENT_GROUND_TRUTH": "Cannot support transition, onset, impact, end, or post-fall timing.",
            "NOT_SAFETY_GROUND_TRUTH": "Cannot establish worker safety, injury, unconsciousness, or emergency.",
        },
        "derived_target_requires_scope": True,
        "scope_escalation": "FAIL_CLOSED",
    }
    limitations = {
        "phase": "T-A4",
        "schema_version": "1.0",
        "status": "PASS_WITH_LIMITATIONS",
        "limitations": [
            "SDT source labels are posture/presence annotations, not fall-event annotations.",
            "LYING may map only to a DERIVED_POSTURE_PROXY compatibility target with explicit frame-only claim limits.",
            "SITTING/STANDING may map only to a non-lying posture proxy and never general worker safety.",
            "EMPTY_ROOM is scoped to the represented frame and does not prove no person exists outside the field of view.",
            "BENDING, KNEELING, WALKING, transitions, impacts, post-fall intervals, and recovery are not represented or not verifiable.",
            "No temporal event, sequence, split, model metric, risk threshold, hardware, clinical, or medical claim is created.",
            "Subject/session/event generalization remains NOT_VERIFIABLE from T-A3.",
        ],
        "t_a3_inheritance": {
            "frame_level": "SUPPORTED",
            "sequence_level": "NOT_VERIFIABLE",
            "event_level": "NOT_VERIFIABLE",
            "window_level": "NOT_APPLICABLE",
        },
        "generalization": {
            "subject_generalization": "NOT_VERIFIABLE",
            "session_generalization": "NOT_VERIFIABLE",
            "event_generalization": "NOT_VERIFIABLE",
            "temporal_fall_event_performance": "NOT_VERIFIABLE",
        },
        "downstream": {"T-A5": "grouping and split policy", "T-A6": "full canonical conversion/integrity", "T-B": "proxy-vs-runtime evaluation", "T-C": "Thermal-44 device semantics", "T-D": "gap-driven activity/hard-negative expansion"},
    }
    return {
        "label_semantic_policy.json": {**policy_profile, "selection_policy_id": selection_policy["policy_id"], "selection_policy_content_sha256": selection_policy_sha, "semantic_policy_content_sha256": policy_content_sha},
        "semantic_policy_candidates.json": {
            "phase": "T-A4",
            "schema_version": "1.0",
            "selection_policy": selection_policy,
            "selection_policy_content_sha256": selection_policy_sha,
            "predeclared_candidates": candidates,
            "evaluated_candidates": evaluated,
            "selected_candidate_id": selected["candidate_id"],
            "winner_derived_from_declared_policy": True,
            "model_metrics_used": False,
        },
        "selected_semantic_policy.json": {
            **policy_profile,
            "selection_policy_id": selection_policy["policy_id"],
            "selection_policy_content_sha256": selection_policy_sha,
            "semantic_policy_content_sha256": policy_content_sha,
            "selection_derivation": "Independent lexicographic admissibility/ranking over the predeclared candidate set; no model metric.",
            "selection_status": "SEMANTIC_POLICY_SELECTED_WITH_EXPLICIT_PROXY_LIMITATIONS",
        },
        "original_label_contract.json": source_contract,
        "frame_evidence_contract.json": frame_contract,
        "compatibility_mapping_contract.json": compatibility_contract,
        "activity_coverage_registry.json": _activity_registry(class_counts),
        "ambiguity_policy.json": ambiguity,
        "claim_scope_contract.json": claim_scopes,
        "label_mapping_inventory.json": {
            "phase": "T-A4",
            "schema_version": "1.0",
            "dataset_id": DATASET_ID,
            "source_split": SOURCE_SPLIT,
            "source_archive_path": SOURCE_ARCHIVE_PATH,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "semantic_policy_id": SEMANTIC_POLICY_ID,
            "row_count": len(mapping_records),
            "records": mapping_records,
            "counts": _mapping_counts(mapping_records),
            "split_assignment_created": False,
            "model_metrics_used": False,
        },
        "pilot_semantic_summary.json": {
            "phase": "T-A4",
            "schema_version": "1.0",
            "source_dataset_id": DATASET_ID,
            "source_split": SOURCE_SPLIT,
            "source_archive_path": SOURCE_ARCHIVE_PATH,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "selection_rule": "Reuse the deterministic 48-frame T-A3 pilot; sort by source frame index; no temporal inference.",
            "pilot_frame_count": len(pilot_records),
            "source_classes_represented": ["LYING", "SITTING", "STANDING", "EMPTY_ROOM"],
            "records": pilot_records,
            "fall_event_escalation": False,
            "t_a3_temporal_policy_id": T_A3_POLICY_ID,
            "t_a2_geometry_profile_id": T_A2_PROFILE_ID,
        },
        "limitations.json": limitations,
    }


def report_text(artifacts: dict[str, Any], validation: dict[str, Any]) -> str:
    selected = artifacts["selected_semantic_policy.json"]
    counts = artifacts["label_mapping_inventory.json"]["counts"]
    activity = artifacts["activity_coverage_registry.json"]["entries"]
    return f"""# Thermal T-A4 — Label Semantics, Proxy Mapping, and Ambiguity Contract

Date: {ACCESS_DATE}\n\nPhase: `T-A4`\n\nOutcome: `{validation.get('overall_outcome', 'PASS_WITH_LIMITATIONS')}`\n\nT-A5 authorized: `{'YES' if validation.get('t_a5_authorized') else 'NO'}`\n\n## Source truth\n\nThe selected source is the SDT real `test` split (`{DATASET_ID}`, {DATASET_DOI}) with archive SHA-256 `{SOURCE_ARCHIVE_SHA256}`. The official [Zenodo SDT record](https://zenodo.org/records/4124309) and [TU Wien documentation](https://cvl.tuwien.ac.at/research/cvl-databases/sdt-icip/) describe pose/presence labels: `LYING`, `SITTING`, `STANDING`, and `EMPTY_ROOM`. The source distribution is 2,000 rows per label, independently measured from `labels.txt`.\n\nThe original label remains immutable. Its source meaning is separate from what it may defensibly support in SafeNest. A `LYING` source annotation is verified as a lying posture, while fall-event interpretation remains ambiguous/not verifiable because T-A3 provides no timestamp, sequence, transition, onset, impact, or end evidence.\n\n## Selected semantic policy\n\nT-A4 evaluated L0 source-only, L1 dual-layer source-plus-proxy, and L2 direct legacy three-class collapse. The declared gates reject L2 because it rewrites source truth and creates unsupported semantic escalation. `{selected['selected_candidate_id']}` was selected by the declared deterministic ranking because it preserves source truth while retaining an explicitly qualified compatibility layer.\n\nLayer B frame evidence is `HUMAN_LYING_POSTURE`, `HUMAN_SITTING_POSTURE`, `HUMAN_STANDING_POSTURE`, and `NO_ANNOTATED_HUMAN_IN_REPRESENTED_FRAME`. Layer C is optional compatibility only: `LYING`→`HUMAN_FALL` is `DERIVED_POSTURE_PROXY`, `SITTING/STANDING`→`HUMAN_NORMAL` are non-lying posture proxies, and `EMPTY_ROOM`→`NOT_HUMAN` is frame-scoped presence equivalence. None is source ground truth, temporal event ground truth, or general worker-safety ground truth.\n\n## Coverage and ambiguity\n\nThe inventory contains {artifacts['label_mapping_inventory.json']['row_count']} deterministic label rows. Mapping types are `{counts['mapping_type_counts']}` and compatibility targets are `{counts['compatibility_target_counts']}`. Unsupported activity categories are explicitly marked not represented or not verifiable; their absence is never turned into a negative example. Bending, kneeling, walking, entering, exiting, transitions, impacts, post-fall intervals, recovery, and natural/staged fall events are not established by this source.\n\nSource-label ambiguity is false for known labels. Fall-interpretation ambiguity is represented independently. Unknown labels, unsupported targets, fake pre/post-fall fields, safety claims, and temporal escalation fail closed. Bounding boxes remain provenance/geometry evidence and do not redefine labels.\n\n## Boundaries\n\nT-A4 does not train, convert all frames, create splits, construct events, modify the runtime model, modify risk/fusion, or make Thermal-44, clinical, medical, or hardware claims. T-A3 remains inherited: frame-level supported; sequence/event not verifiable; window not applicable. T-A5 must preserve this semantic contract while solving grouping and splits.\n"""


def write_artifacts(root: Path = ROOT) -> dict[str, Any]:
    evidence_dir = root / EVIDENCE_REL
    evidence_dir.mkdir(parents=True, exist_ok=True)
    artifacts = build_artifacts(root)
    for name, data in sorted(artifacts.items()):
        (evidence_dir / name).write_text(canonical_json(data), encoding="utf-8")
    from scripts.validate_thermal_t_a4 import validate_evidence

    validation = validate_evidence(repo_root=root, evidence_dir=evidence_dir, check_checksums=False, verify_real_payload=True)
    (evidence_dir / "validation_result.json").write_text(canonical_json(validation), encoding="utf-8")
    checksum_lines = []
    for name in sorted(JSON_NAMES):
        path = evidence_dir / name
        checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (evidence_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    report_path = root / REPORT_REL
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text(artifacts, validation), encoding="utf-8")
    return validation


if __name__ == "__main__":
    result = write_artifacts()
    print(canonical_json(result), end="")
    raise SystemExit(0 if result.get("evidence_validation") == "PASS" else 1)
