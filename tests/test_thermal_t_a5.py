"""Focused fail-closed tests for the Thermal T-A5 assignment contract."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.thermal.split_policy import (  # noqa: E402
    ASSIGNMENT_RULE_ID,
    DATASET_ID,
    FrameHashSplitError,
    FrameRandomSplitError,
    GroupingProvenanceUnavailableError,
    LockedTestAccessError,
    SEMANTIC_POLICY_ID,
    SPLIT_POLICY_ID,
    SourceDomainMismatchError,
    assignment_for_real_test_frame,
    candidate_policy_definitions,
    evaluate_candidates,
    grouping_evidence_definition,
    reject_unsupported_split_request,
    selected_candidate,
    selected_split_policy_profile,
    selection_policy_definition,
    validate_assignment_inventory,
    validate_assignment_record,
    validate_derived_assignment,
)


def _record(index: int = 0, **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "source_split": "test",
        "source_member": f"test/image_t_{index}.png",
        "source_frame_index": index,
        "semantic_policy_id": SEMANTIC_POLICY_ID,
        "original_label_id": 0,
        "original_label_name": "LYING",
        "compatibility_target": "HUMAN_FALL",
        "mapping_type": "DERIVED_POSTURE_PROXY",
        "mapping_rule_id": "THERMAL_MAP_LYING_TO_FALL_COMPAT_PROXY_001",
        "claim_scope": ["FRAME_LEVEL_FALL_COMPATIBILITY_PROXY_ONLY"],
    }
    value.update(overrides)
    return value


def test_selected_policy_preserves_official_partitions_without_random_or_hash() -> None:
    selected = selected_split_policy_profile()
    assert selected["selected_candidate_id"] == "S0_OFFICIAL_SOURCE_PARTITION_PRESERVATION"
    assert selected["source_partition_preservation"] is True
    assert selected["random_or_hash_resplit"] is False
    assert selected["real_test_role"] == "REAL_EVAL_DEVELOPMENT"


def test_candidate_selection_is_order_independent() -> None:
    policy = selection_policy_definition()
    first = evaluate_candidates(candidate_policy_definitions(), policy)
    second = evaluate_candidates(list(reversed(candidate_policy_definitions())), policy)
    assert [row["candidate_id"] for row in first] == [row["candidate_id"] for row in second]
    assert selected_candidate(first)["candidate_id"] == selected_candidate(second)["candidate_id"]


def test_missing_subject_session_event_are_explicitly_not_verifiable() -> None:
    profile = grouping_evidence_definition()
    for dimension in ("subject", "session", "event", "sequence", "scene", "camera"):
        assert profile["dimensions"][dimension]["status"] == "NOT_VERIFIABLE"
        assert profile["dimensions"][dimension]["usable_for_split"] is False
    assert profile["generalization_performance"] == "NOT_VERIFIABLE"


@pytest.mark.parametrize("method", ["RANDOM", "STRATIFIED_RANDOM", "FRAME_RANDOM", "SHUFFLE_RATIO"])
def test_frame_random_split_is_rejected(method: str) -> None:
    with pytest.raises(FrameRandomSplitError):
        reject_unsupported_split_request({"method": method})


@pytest.mark.parametrize("method", ["HASH", "FRAME_HASH", "MEMBER_HASH"])
def test_frame_hash_split_is_rejected(method: str) -> None:
    with pytest.raises(FrameHashSplitError):
        reject_unsupported_split_request({"method": method})


@pytest.mark.parametrize("group", ["FRAME_INDEX", "LABEL", "FRAME_BLOCK"])
def test_frame_index_label_and_arbitrary_blocks_are_not_groups(group: str) -> None:
    with pytest.raises(GroupingProvenanceUnavailableError):
        reject_unsupported_split_request({"grouping_unit_type": group})


def test_real_test_frame_assignment_inherits_t_a4_and_is_development_only() -> None:
    result = assignment_for_real_test_frame(_record(17))
    assert result["source_domain"] == "REAL"
    assert result["safenest_assignment_role"] == "REAL_EVAL_DEVELOPMENT"
    assert result["locked_test_eligibility"] is False
    assert result["assignment_rule_id"] == ASSIGNMENT_RULE_ID
    assert result["split_policy_id"] == SPLIT_POLICY_ID
    validate_assignment_record(result)


def test_source_domain_mismatch_is_rejected() -> None:
    with pytest.raises(SourceDomainMismatchError):
        assignment_for_real_test_frame(_record(source_split="train"))


def test_invalid_member_frame_linkage_is_rejected() -> None:
    with pytest.raises(Exception):
        assignment_for_real_test_frame(_record(3, source_member="test/image_t_4.png"))


def test_renaming_accessed_test_to_locked_test_is_rejected() -> None:
    with pytest.raises(LockedTestAccessError):
        reject_unsupported_split_request({"safenest_assignment_role": "LOCKED_TEST", "prior_access_status": "USED_FOR_T_A0_T_A4_DEVELOPMENT_AND_GEOMETRY_SELECTION"})


def test_tampered_assignment_role_or_lock_fails_closed() -> None:
    result = assignment_for_real_test_frame(_record())
    result["safenest_assignment_role"] = "LOCKED_TEST"
    with pytest.raises(Exception):
        validate_assignment_record(result)
    result = assignment_for_real_test_frame(_record())
    result["locked_test_eligibility"] = True
    with pytest.raises(Exception):
        validate_assignment_record(result)


def test_tampered_rule_and_semantic_policy_fail_closed() -> None:
    result = assignment_for_real_test_frame(_record())
    result["assignment_rule_id"] = "THERMAL_ASSIGNMENT_RULE_999"
    with pytest.raises(Exception):
        validate_assignment_record(result)
    result = assignment_for_real_test_frame(_record())
    result["t_a4_semantic_policy_id"] = "THERMAL_LABEL_SEMANTIC_POLICY_999"
    with pytest.raises(Exception):
        validate_assignment_record(result)


def test_duplicate_or_incomplete_inventory_fails_closed() -> None:
    one = assignment_for_real_test_frame(_record())
    with pytest.raises(Exception):
        validate_assignment_inventory([one])
    rows = [assignment_for_real_test_frame(_record(i)) for i in range(8000)]
    rows[-1]["source_member"] = rows[0]["source_member"]
    with pytest.raises(Exception):
        validate_assignment_inventory(rows)


def test_group_cross_role_is_rejected_by_record_validator() -> None:
    result = assignment_for_real_test_frame(_record())
    result["grouping_unit_id"] = "other:test"
    with pytest.raises(Exception):
        validate_assignment_record(result)


def test_derived_artifact_must_inherit_assignment() -> None:
    parent = assignment_for_real_test_frame(_record(9))
    child = dict(parent, artifact_kind="CANONICAL")
    validate_derived_assignment(parent, child)
    child["source_frame_index"] = 10
    with pytest.raises(Exception):
        validate_derived_assignment(parent, child)


def test_augmentation_is_train_only_and_real_test_cannot_be_augmented() -> None:
    parent = assignment_for_real_test_frame(_record(9))
    child = dict(parent, augmentation=True)
    with pytest.raises(Exception):
        validate_derived_assignment(parent, child)


def test_unknown_group_dependence_is_not_reported_as_zero() -> None:
    profile = grouping_evidence_definition()
    assert profile["generalization_performance"] == "NOT_VERIFIABLE"
    assert profile["dimensions"]["subject"]["cardinality"] == "NOT_VERIFIABLE"


def test_exact_member_inventory_can_prove_zero_overlap() -> None:
    rows = [assignment_for_real_test_frame(_record(i)) for i in range(8000)]
    summary = validate_assignment_inventory(rows)
    assert summary["unique_frame_count"] == 8000
    assert summary["unique_member_count"] == 8000


def test_policy_does_not_conflate_synthetic_and_real_domains() -> None:
    rows = {row["source_split"]: row for row in __import__("datasets.thermal.split_policy", fromlist=["source_partition_definitions"]).source_partition_definitions()}
    assert rows["train"]["source_domain"] == "SYNTHETIC"
    assert rows["validation"]["source_domain"] == "SYNTHETIC"
    assert rows["test"]["source_domain"] == "REAL"


def test_placeholder_partitions_are_not_sample_level_audited() -> None:
    rows = __import__("datasets.thermal.split_policy", fromlist=["source_partition_definitions"]).source_partition_definitions()
    for row in rows[:2]:
        assert row["materialization_status"] == "LOCAL_CLOUD_PLACEHOLDER"
        assert row["sample_inventory_status"] == "SAMPLE_LEVEL_INVENTORY_PENDING_MATERIALIZATION"


def test_legacy_npz_is_not_split_authority() -> None:
    limitations = json.loads((ROOT / "datasets/thermal/manifests/T-A5_grouping_immutable_split/limitations.json").read_text())
    assert any("processed_thermal_80x62.npz" in item and "not split authority" in item.lower() for item in limitations["limitations"])


def test_accessed_real_test_is_not_pristine_locked_test() -> None:
    history = __import__("datasets.thermal.split_policy", fromlist=["access_history_definition"]).access_history_definition()
    assert history["pristine_locked_test_available"] == "NO"
    assert any(row["access_type"] == "GEOMETRY_SELECTION" for row in history["entries"] if row["source_split"] == "test")


def test_assignment_is_deterministic() -> None:
    first = assignment_for_real_test_frame(_record(123))
    second = assignment_for_real_test_frame(_record(123))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_no_absolute_paths_in_t_a5_json() -> None:
    directory = ROOT / "datasets/thermal/manifests/T-A5_grouping_immutable_split"
    for path in directory.glob("*.json"):
        text = path.read_text()
        assert "/Users/" not in text
        assert "file://" not in text


def test_t_a5_validator_passes_with_limitations() -> None:
    from scripts.validate_thermal_t_a5 import validate_evidence
    result = validate_evidence(repo_root=ROOT, evidence_dir=ROOT / "datasets/thermal/manifests/T-A5_grouping_immutable_split")
    assert result["evidence_validation"] == "PASS"
    assert result["overall_outcome"] == "PASS_WITH_LIMITATIONS"
    assert result["t_a6_authorized"] is True


def test_tamper_selected_policy_is_detected(tmp_path: Path) -> None:
    source = ROOT / "datasets/thermal/manifests/T-A5_grouping_immutable_split"
    target = tmp_path / "evidence"
    shutil.copytree(source, target)
    path = target / "selected_split_policy.json"
    value = json.loads(path.read_text())
    value["selected_candidate_id"] = "S1_REAL_TEST_FRAME_RANDOM_RESPLIT"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    from scripts.validate_thermal_t_a5 import validate_evidence
    result = validate_evidence(repo_root=ROOT, evidence_dir=target, check_checksums=False)
    assert result["evidence_validation"] == "FAIL"


def test_tamper_access_history_is_detected(tmp_path: Path) -> None:
    source = ROOT / "datasets/thermal/manifests/T-A5_grouping_immutable_split"
    target = tmp_path / "evidence"
    shutil.copytree(source, target)
    path = target / "data_access_history.json"
    value = json.loads(path.read_text())
    value["entries"] = [row for row in value["entries"] if not (row["source_split"] == "test" and row["phase"] == "T-A2")]
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    from scripts.validate_thermal_t_a5 import validate_evidence
    result = validate_evidence(repo_root=ROOT, evidence_dir=target, check_checksums=False)
    assert result["evidence_validation"] == "FAIL"


def test_tamper_checksum_is_detected(tmp_path: Path) -> None:
    source = ROOT / "datasets/thermal/manifests/T-A5_grouping_immutable_split"
    target = tmp_path / "evidence"
    shutil.copytree(source, target)
    path = target / "grouping_evidence_registry.json"
    value = json.loads(path.read_text())
    value["generalization_performance"] = "SUBJECT_WISE_PASS"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    from scripts.validate_thermal_t_a5 import validate_evidence
    result = validate_evidence(repo_root=ROOT, evidence_dir=target, check_checksums=True)
    assert result["evidence_validation"] == "FAIL"


def test_tampered_t_a2_evidence_is_revalidated_even_with_stale_pass_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale T-A2 validation_result.json must not hide current evidence tampering."""
    import scripts.validate_thermal_t_a2 as t_a2

    geometry_path = ROOT / "datasets/thermal/manifests/T-A2_geometry_calibration_canonical_frame/selected_geometry_profile.json"
    stale_result = json.loads((geometry_path.parent / "validation_result.json").read_text(encoding="utf-8"))
    assert stale_result["evidence_validation"] == "PASS"
    original = geometry_path.read_text(encoding="utf-8")
    try:
        value = json.loads(original)
        value["profile"]["profile_id"] = "G1_TAMPERED_PROFILE"
        geometry_path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        calls: list[dict[str, object]] = []
        real_validate = t_a2.validate_evidence

        def spy_validate(**kwargs: object) -> dict[str, object]:
            calls.append(dict(kwargs))
            return real_validate(**kwargs)

        monkeypatch.setattr(t_a2, "validate_evidence", spy_validate)
        from scripts.validate_thermal_t_a5 import validate_evidence
        result = validate_evidence(repo_root=ROOT, evidence_dir=ROOT / "datasets/thermal/manifests/T-A5_grouping_immutable_split", check_checksums=False)
        assert calls and calls[0]["verify_real_payload"] is False
        assert result["evidence_validation"] == "FAIL"
        assert any(error["code"] == "T_A2_VALIDATION_FAILED" for error in result["errors"])
    finally:
        geometry_path.write_text(original, encoding="utf-8", newline="\n")


def test_no_t_a6_payload_conversion_or_model_metric_is_present() -> None:
    text = (ROOT / "datasets/thermal/split_policy.py").read_text()
    assert "train_test_split" not in text
    assert "np.random.permutation" not in text
    assert "tensorflow" not in text
