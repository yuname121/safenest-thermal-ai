"""Focused anti-hallucination tests for Thermal T-A3 temporal policy."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.thermal.temporal_policy import (  # noqa: E402
    FabricatedTemporalMetadataError,
    TemporalEventUnavailableError,
    TemporalSequenceUnavailableError,
    TemporalWindowUnavailableError,
    construct_event,
    construct_sequence,
    construct_window,
    evaluate_event_construction,
    evaluate_sequence_construction,
    evaluate_window_construction,
    frame_sample_from_provenance,
    temporal_policy_profile,
    validate_frame_sample,
)


def _provenance(**overrides):
    value = {
        "source_dataset_id": "local_sdt_zenodo_4124309",
        "source_doi": "10.5281/zenodo.4124309",
        "source_split": "test",
        "source_archive_path": "datasets/raw_archives/thermal_split_zips/test.zip",
        "source_archive_size_bytes": 1740348425,
        "source_archive_md5": "d59a739f3b5ecf373c94046fb94cd94f",
        "source_archive_sha256": "3a838bd70835e579ecfaa820a6c0b4cbc6ba7b76729417c73845f0c959281449",
        "source_member_name": "test/image_t_7.png",
        "source_member_index": 8008,
        "source_member_crc32": "deadbeef",
        "source_member_sha256": "a" * 64,
        "source_frame_index": 7,
        "source_pose_label": 0,
        "source_pose_name": "LYING",
        "source_bbox": [1.0, 2.0, 3.0, 4.0],
        "source_timestamp_status": "ABSENT",
        "source_subject_status": "ABSENT",
        "source_session_status": "ABSENT",
        "source_sequence_status": "ABSENT",
        "source_event_status": "ABSENT",
        "raw_encoded_frame_sha256": "b" * 64,
    }
    value.update(overrides)
    return value


def test_valid_static_frame_sample_preserves_source_and_t_a2_identity() -> None:
    record = frame_sample_from_provenance(_provenance(), canonical_frame_hash="c" * 64)
    validate_frame_sample(record)
    assert record["source_split"] == "test"
    assert record["source_frame_index_role"] == "PROVENANCE_IDENTIFIER_ONLY_NOT_TIMESTAMP"
    assert record["t_a2_geometry_profile_id"] == "G1_FIXED_ASPECT_CROP_BILINEAR"
    assert record["safe_nest_label_status"] == "NOT_ASSIGNED_T_A3"


@pytest.mark.parametrize(
    "field",
    [
        "timestamp", "timestamp_s", "timestamp_ms", "fps", "frame_rate", "sequence_id", "session_id",
        "recording_id", "event_id", "event_start", "event_end", "window_start", "window_end", "window_duration_s",
        "window_frame_count",
    ],
)
def test_temporal_fields_are_rejected_from_frame_requests(field: str) -> None:
    with pytest.raises(FabricatedTemporalMetadataError):
        frame_sample_from_provenance(_provenance(**{field: 1}), canonical_frame_hash="c" * 64)


def test_frame_index_and_filename_order_cannot_be_promoted_to_time() -> None:
    with pytest.raises(FabricatedTemporalMetadataError):
        evaluate_sequence_construction({"frame_index_as_timestamp": True})
    with pytest.raises(FabricatedTemporalMetadataError):
        evaluate_sequence_construction({"filename_order_is_temporal_order": True})
    assert temporal_policy_profile()["source_frame_semantics"]["index_is_timestamp"] is False
    assert temporal_policy_profile()["source_frame_semantics"]["filename_order_is_temporal_order"] is False


def test_neighboring_indices_do_not_establish_one_sequence() -> None:
    result = evaluate_sequence_construction({"frame_indices": [7, 8]})
    assert result["eligible"] is False
    assert result["status"] == "TEMPORAL_SEQUENCE_NOT_VERIFIABLE"
    with pytest.raises(TemporalSequenceUnavailableError):
        construct_sequence([{"source_frame_index": 7}, {"source_frame_index": 8}])


def test_lying_is_posture_not_fall_event_and_fake_boundaries_fail_closed() -> None:
    with pytest.raises(FabricatedTemporalMetadataError):
        evaluate_event_construction({"lying_is_fall_event": True})
    with pytest.raises(FabricatedTemporalMetadataError):
        construct_event([], event_start=1, event_end=2)
    result = evaluate_event_construction({"source_pose_label": "LYING"})
    assert result["eligible"] is False
    assert "LYING_IS_POSTURE_ONLY" in result["reasons"]
    with pytest.raises(TemporalEventUnavailableError):
        construct_event([{"source_frame_index": 7}])


def test_arbitrary_window_parameters_and_index_to_seconds_fail_closed() -> None:
    assert evaluate_window_construction({"frame_count": 16, "stride": 4})["eligible"] is False
    with pytest.raises(FabricatedTemporalMetadataError):
        evaluate_window_construction({"window_duration_s": 2.0})
    with pytest.raises(TemporalWindowUnavailableError):
        construct_window([{"source_frame_index": 7}] * 16, frame_count=16, stride=4)


def test_gap_duplicate_policy_distinguishes_structural_and_temporal_meaning() -> None:
    policy_path = ROOT / "datasets/thermal/manifests/T-A3_sequence_window_event_policy/gap_duplicate_policy.json"
    data = json.loads(policy_path.read_text(encoding="utf-8"))
    assert data["index_gap_status"] == "SOURCE_MEMBER_INDEX_GAP"
    assert data["index_gap_interpretation"].endswith("NOT_A_DROPPED_ACQUISITION_FRAME")
    assert data["temporal_dropped_frame_status"] == "TEMPORAL_DROPPED_FRAME_NOT_VERIFIABLE"
    assert data["exact_duplicate_content_semantics"].startswith("PRESERVE_BOTH")


def test_unknown_and_absent_temporal_statuses_are_not_collapsed() -> None:
    record = frame_sample_from_provenance(_provenance(), canonical_frame_hash="c" * 64)
    assert record["source_timestamp_status"] == "ABSENT"
    assert record["temporal_predecessor_status"] == "UNKNOWN_NOT_VERIFIABLE"
    assert record["sequence_id_status"] == "ABSENT"


def test_source_member_index_is_zip_provenance_not_frame_index() -> None:
    record = frame_sample_from_provenance(_provenance(source_member_index=8008), canonical_frame_hash="c" * 64)
    assert record["source_member_index"] != record["source_frame_index"]
    assert record["source_member_name"] == "test/image_t_7.png"


def test_policy_artifacts_are_canonical_and_portable() -> None:
    evidence = ROOT / "datasets/thermal/manifests/T-A3_sequence_window_event_policy"
    for path in sorted(evidence.glob("*.json")):
        text = path.read_text(encoding="utf-8")
        assert text == json.dumps(json.loads(text), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        assert "/Users/" not in text
        assert "file://" not in text


def test_validator_rejects_policy_tamper_and_checksum_tamper(tmp_path: Path) -> None:
    source = ROOT / "datasets/thermal/manifests/T-A3_sequence_window_event_policy"
    copied = tmp_path / "evidence"
    shutil.copytree(source, copied)
    policy_path = copied / "temporal_capability_contract.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["policy"]["capabilities"]["SEQUENCE_LEVEL"]["supported"] = True
    policy_path.write_text(json.dumps(policy, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    from scripts.validate_thermal_t_a3 import validate_evidence

    result = validate_evidence(repo_root=ROOT, evidence_dir=copied, check_checksums=True, verify_real_payload=False)
    codes = {item["code"] for item in result["errors"]}
    assert result["evidence_validation"] == "FAIL"
    assert "TEMPORAL_POLICY_INVALID" in codes or "UNSUPPORTED_CAPABILITY_OPEN" in codes
    assert "CHECKSUM_MISMATCH" in codes


def test_validator_accepts_static_frame_evidence_without_payload(tmp_path: Path) -> None:
    source = ROOT / "datasets/thermal/manifests/T-A3_sequence_window_event_policy"
    copied = tmp_path / "evidence"
    shutil.copytree(source, copied)
    from scripts.validate_thermal_t_a3 import validate_evidence

    result = validate_evidence(repo_root=ROOT, evidence_dir=copied, check_checksums=False, verify_real_payload=False)
    assert result["evidence_validation"] == "PASS"
    assert result["t_a4_authorized"] is True


def test_t_a3_implementation_has_no_model_coupling() -> None:
    for relative in ("datasets/thermal/temporal_policy.py", "scripts/generate_thermal_t_a3.py"):
        source = (ROOT / relative).read_text(encoding="utf-8").lower()
        assert "thermalinterpreter" not in source
        assert "tflite" not in source
        assert "tensorflow" not in source
