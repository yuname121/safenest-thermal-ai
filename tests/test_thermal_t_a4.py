"""Focused fail-closed tests for Thermal T-A4 label semantics."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.thermal.label_semantics import (  # noqa: E402
    DATASET_DOI,
    DATASET_ID,
    SEMANTIC_POLICY_ID,
    SOURCE_ARCHIVE_PATH,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_LABELS,
    SOURCE_SPLIT,
    SafetyStateInferenceError,
    SemanticEscalationError,
    SemanticPolicyMismatchError,
    UnknownSourceLabelError,
    UnsupportedLabelMappingError,
    candidate_policy_definitions,
    evaluate_candidates,
    map_source_label,
    reject_semantic_escalation,
    validate_mapping_record,
)


def _source_record(label_id: int = 0, frame_index: int = 7, **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "source_doi": DATASET_DOI,
        "source_split": SOURCE_SPLIT,
        "source_archive_path": SOURCE_ARCHIVE_PATH,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_member": f"test/image_t_{frame_index}.png",
        "source_frame_index": frame_index,
        "original_label_id": label_id,
        "original_label_name": SOURCE_LABELS.get(label_id, "UNKNOWN"),
        "original_bbox": [1.0, 2.0, 30.0, 40.0],
    }
    value.update(overrides)
    return value


def test_all_four_source_labels_preserve_truth_and_separate_proxy() -> None:
    expected = {
        "LYING": ("HUMAN_LYING_POSTURE", "HUMAN_FALL", "DERIVED_POSTURE_PROXY"),
        "SITTING": ("HUMAN_SITTING_POSTURE", "HUMAN_NORMAL", "DERIVED_POSTURE_PROXY"),
        "STANDING": ("HUMAN_STANDING_POSTURE", "HUMAN_NORMAL", "DERIVED_POSTURE_PROXY"),
        "EMPTY_ROOM": ("NO_ANNOTATED_HUMAN_IN_REPRESENTED_FRAME", "NOT_HUMAN", "DIRECT_SOURCE_EQUIVALENT"),
    }
    for label_id, label_name in SOURCE_LABELS.items():
        result = map_source_label(_source_record(label_id))
        assert (result["frame_evidence_label"], result["compatibility_target"], result["mapping_type"]) == expected[label_name]
        assert result["original_label_id"] == label_id
        assert result["original_label_name"] == label_name
        assert result["original_bbox"] == [1.0, 2.0, 30.0, 40.0]
        assert result["source_label_modified"] is False
        assert result["fall_event_semantic_status"] == "NOT_VERIFIABLE"
        validate_mapping_record(result)


def test_lying_is_explicit_posture_proxy_not_verified_fall_event() -> None:
    result = map_source_label(_source_record(0))
    assert result["compatibility_target"] == "HUMAN_FALL"
    assert result["mapping_type"] == "DERIVED_POSTURE_PROXY"
    assert "FRAME_LEVEL_FALL_COMPATIBILITY_PROXY_ONLY" in result["claim_scope"]
    assert "NOT_TEMPORAL_EVENT_GROUND_TRUTH" in result["claim_scope"]
    with pytest.raises(SemanticEscalationError):
        reject_semantic_escalation({"verified_fall": True})
    with pytest.raises(SemanticEscalationError):
        reject_semantic_escalation({"fall_event_ground_truth": "VERIFIED"})
    with pytest.raises(SemanticEscalationError):
        reject_semantic_escalation({"event_start": 1, "event_end": 2})


def test_sitting_and_standing_are_not_safety_ground_truth() -> None:
    for label_id in (1, 2):
        result = map_source_label(_source_record(label_id))
        assert result["compatibility_target"] == "HUMAN_NORMAL"
        assert "NOT_SAFETY_GROUND_TRUTH" in result["claim_scope"]
    with pytest.raises(SafetyStateInferenceError):
        reject_semantic_escalation({"worker_safe": True})
    with pytest.raises(SafetyStateInferenceError):
        reject_semantic_escalation({"medical_diagnosis": True})


def test_empty_room_is_frame_scoped_presence_only() -> None:
    result = map_source_label(_source_record(3))
    assert result["compatibility_target"] == "NOT_HUMAN"
    assert result["mapping_type"] == "DIRECT_SOURCE_EQUIVALENT"
    assert result["claim_scope"] == ["FRAME_LEVEL_PRESENCE_ONLY", "NOT_SAFETY_GROUND_TRUTH"]
    assert "outside" not in result["frame_evidence_label"].lower()


@pytest.mark.parametrize("field", ["timestamp", "fps", "sequence_id", "session_id", "event_id", "pre_fall", "post_fall", "transition_frame"])
def test_temporal_fields_cannot_be_added_to_label_mapping(field: str) -> None:
    with pytest.raises(SemanticEscalationError):
        map_source_label(_source_record(0, **{field: 1}))


def test_unknown_labels_and_missing_proxy_rules_fail_closed() -> None:
    with pytest.raises(UnknownSourceLabelError):
        map_source_label(_source_record(9, original_label_name="UNKNOWN"))
    mapped = map_source_label(_source_record(0))
    mapped.pop("mapping_rule_id")
    with pytest.raises(SemanticPolicyMismatchError):
        validate_mapping_record(mapped)
    mapped = map_source_label(_source_record(0))
    mapped["mapping_type"] = "NOT_A_MAPPING"
    mapped["compatibility_target"] = None
    with pytest.raises(UnsupportedLabelMappingError):
        validate_mapping_record(mapped)
    with pytest.raises(UnknownSourceLabelError):
        reject_semantic_escalation({"original_label_id": 99})


def test_proxy_claim_scope_and_source_annotation_are_mandatory() -> None:
    mapped = map_source_label(_source_record(0))
    mapped["claim_scope"] = ["FRAME_LEVEL_ONLY"]
    with pytest.raises(SemanticPolicyMismatchError):
        validate_mapping_record(mapped)


def test_mapping_rule_claim_scope_and_event_status_tampering_fail_closed() -> None:
    mapped = map_source_label(_source_record(0))
    mapped["mapping_rule_id"] = "THERMAL_MAP_LYING_DIRECT_FALL_999"
    with pytest.raises(SemanticPolicyMismatchError):
        validate_mapping_record(mapped)
    mapped = map_source_label(_source_record(0))
    mapped["fall_event_semantic_status"] = "VERIFIED_FALL_EVENT"
    with pytest.raises(SemanticEscalationError):
        validate_mapping_record(mapped)
    mapped = map_source_label(_source_record(0))
    mapped["pre_fall_event"] = True
    with pytest.raises(SemanticEscalationError):
        validate_mapping_record(mapped)
    mapped = map_source_label(_source_record(0))
    mapped["claim_scope"] = ["FRAME_LEVEL_FALL_COMPATIBILITY_PROXY_ONLY"]
    with pytest.raises(SemanticPolicyMismatchError):
        validate_mapping_record(mapped)


@pytest.mark.parametrize("label", ["PRE_FALL", "POST_FALL", "FALL_TRANSITION", "FALL_IMPACT"])
def test_fake_temporal_label_names_are_rejected(label: str) -> None:
    with pytest.raises(SemanticEscalationError):
        reject_semantic_escalation({"label": label})
    mapped = map_source_label(_source_record(0))
    mapped["source_label_modified"] = True
    with pytest.raises(SemanticPolicyMismatchError):
        validate_mapping_record(mapped)


def test_candidate_selection_is_order_independent_and_rejects_direct_collapse() -> None:
    candidates = candidate_policy_definitions()
    first = evaluate_candidates(candidates)
    second = evaluate_candidates(list(reversed(candidates)))
    assert [item["candidate_id"] for item in first] == [item["candidate_id"] for item in second]
    assert [item["candidate_id"] for item in first if item["selected"]] == ["L1_DUAL_LAYER_SOURCE_PLUS_PROXY"]
    rejected = next(item for item in first if item["candidate_id"] == "L2_DIRECT_LEGACY_3_CLASS_COLLAPSE")
    assert rejected["admissible"] is False
    assert "preserves_original_source_labels" in rejected["rejection_reasons"]


def test_repeated_mapping_generation_is_byte_stable() -> None:
    import json

    first = [map_source_label(_source_record(label_id, frame_index=label_id)) for label_id in SOURCE_LABELS for _ in [0]]
    second = [map_source_label(_source_record(label_id, frame_index=label_id)) for label_id in SOURCE_LABELS for _ in [0]]
    encode = lambda value: json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    assert encode(first) == encode(second)


def test_t_a3_temporal_inheritance_is_preserved_in_pilot_artifact() -> None:
    pilot = json.loads((ROOT / "datasets/thermal/manifests/T-A4_label_semantics_ambiguity/pilot_semantic_summary.json").read_text(encoding="utf-8"))
    assert pilot["pilot_frame_count"] == 48
    assert pilot["fall_event_escalation"] is False
    assert all(record["t_a3_temporal_policy_id"] == "THERMAL_TEMPORAL_POLICY_001" for record in pilot["records"])
    assert all(record["event_id_status"] == "ABSENT" for record in pilot["records"])


def test_activity_coverage_does_not_turn_unsupported_activities_into_negatives() -> None:
    registry = json.loads((ROOT / "datasets/thermal/manifests/T-A4_label_semantics_ambiguity/activity_coverage_registry.json").read_text(encoding="utf-8"))
    assert registry["unsupported_activity_policy"] == "ABSENCE_OF_A_LABEL_IS_NOT_A_NEGATIVE_EXAMPLE"
    for name in ("BENDING", "KNEELING", "WALKING", "FALL_TRANSITION", "FALL_IMPACT", "RECOVERY"):
        assert registry["entries"][name]["source_count"] == 0
        assert registry["entries"][name]["treated_as_negative"] is False


def test_validator_rejects_policy_claim_scope_and_checksum_tampering(tmp_path: Path) -> None:
    source = ROOT / "datasets/thermal/manifests/T-A4_label_semantics_ambiguity"
    copied = tmp_path / "evidence"
    shutil.copytree(source, copied)
    policy_path = copied / "selected_semantic_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["selected_candidate_id"] = "L2_DIRECT_LEGACY_3_CLASS_COLLAPSE"
    policy_path.write_text(json.dumps(policy, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    from scripts.validate_thermal_t_a4 import validate_evidence

    result = validate_evidence(repo_root=ROOT, evidence_dir=copied, check_checksums=True, verify_real_payload=False)
    codes = {item["code"] for item in result["errors"]}
    assert result["evidence_validation"] == "FAIL"
    assert "CHECKSUM_MISMATCH" in codes
    assert "EXPECTED_SELECTED_POLICY_NOT_DUAL_LAYER" in codes or "SELECTED_POLICY_WINNER_MISMATCH" in codes


def test_validator_rejects_absolute_paths_and_unsupported_hardware_assertion(tmp_path: Path) -> None:
    source = ROOT / "datasets/thermal/manifests/T-A4_label_semantics_ambiguity"
    copied = tmp_path / "evidence"
    shutil.copytree(source, copied)
    path = copied / "limitations.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["limitations"].append("/Users/example/private payload")
    data["limitations"].append("Thermal-44 VERIFIED physical unit")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    from scripts.validate_thermal_t_a4 import validate_evidence

    result = validate_evidence(repo_root=ROOT, evidence_dir=copied, check_checksums=False, verify_real_payload=False)
    codes = {item["code"] for item in result["errors"]}
    assert "NONPORTABLE_PATH" in codes
    assert "UNSUPPORTED_THERMAL44_ASSERTION" in codes


def test_artifacts_are_canonical_portable_and_checksum_covered() -> None:
    evidence = ROOT / "datasets/thermal/manifests/T-A4_label_semantics_ambiguity"
    checksum_lines = (evidence / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    expected_names = {line.split("  ", 1)[1].split("/")[-1] for line in checksum_lines}
    for path in sorted(evidence.glob("*.json")):
        text = path.read_text(encoding="utf-8")
        assert text == json.dumps(json.loads(text), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        assert "/Users/" not in text
        assert "file://" not in text
        assert path.name in expected_names
    assert hashlib.sha256((evidence / "checksums.sha256").read_bytes()).hexdigest()


def test_t_a4_implementation_has_no_model_training_or_inference_coupling() -> None:
    for relative in ("datasets/thermal/label_semantics.py", "scripts/generate_thermal_t_a4.py"):
        source = (ROOT / relative).read_text(encoding="utf-8").lower()
        assert "thermalinterpreter" not in source
        assert "tensorflow" not in source
        assert ".tflite" not in source
        assert "train_and_quantize" not in source
