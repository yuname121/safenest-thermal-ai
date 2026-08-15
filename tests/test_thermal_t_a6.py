"""Focused non-hydrating tests for the Thermal T-A6 Stage-1 contract."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.thermal.canonical_converter import (  # noqa: E402
    CANONICAL_DTYPE,
    CANONICAL_SHAPE,
    COLAB_STAGE2_MODE,
    ConversionConfig,
    SyntheticPayloadAccessProhibitedError,
    STAGE1_MODE,
    audit_near_duplicates,
    canonical_json,
    convert_partition,
    near_duplicate_profile,
)
from datasets.thermal.canonical_geometry import GeometryShapeError, canonicalize_physical_frame, profile_for_id  # noqa: E402
from scripts.run_thermal_t_a6_colab import (  # noqa: E402
    ColabExecutionError,
    SyntheticMacAccessError,
    check_output_writable,
    check_storage,
    inspect_multipart_format,
    load_resume_ledger,
    reconstruct_raw_byte_split,
    run,
    save_resume_ledger,
)
from scripts.validate_thermal_t_a6 import validate_evidence  # noqa: E402
from tests.test_thermal_t_a6_stage2 import _write_valid_bundle  # noqa: E402


def test_source_to_canonical_is_deterministic_and_source_unchanged() -> None:
    profile = profile_for_id("G1_FIXED_ASPECT_CROP_BILINEAR")
    source = np.arange(480 * 640, dtype=np.float32).reshape(480, 640)
    before = source.copy()
    first = canonicalize_physical_frame(source, profile, source_frame_hash="a" * 64)
    second = canonicalize_physical_frame(source, profile, source_frame_hash="a" * 64)
    assert np.array_equal(source, before)
    assert first.canonical_frame_hash == second.canonical_frame_hash
    assert np.array_equal(first.physical_frame, second.physical_frame)
    assert first.physical_frame.shape == CANONICAL_SHAPE
    assert first.physical_frame.dtype == CANONICAL_DTYPE


def test_wrong_geometry_profile_shape_is_rejected() -> None:
    profile = profile_for_id("G1_FIXED_ASPECT_CROP_BILINEAR")
    object.__setattr__(profile, "canonical_shape", (63, 80))
    with pytest.raises(GeometryShapeError):
        canonicalize_physical_frame(np.zeros((480, 640), dtype=np.float32), profile)


def test_nonfinite_source_is_rejected() -> None:
    profile = profile_for_id("G1_FIXED_ASPECT_CROP_BILINEAR")
    source = np.zeros((480, 640), dtype=np.float32)
    source[0, 0] = np.nan
    with pytest.raises(Exception):
        canonicalize_physical_frame(source, profile)


def test_constant_physical_frame_is_preserved_without_normalization() -> None:
    profile = profile_for_id("G1_FIXED_ASPECT_CROP_BILINEAR")
    source = np.full((480, 640), 21.5, dtype=np.float32)
    result = canonicalize_physical_frame(source, profile)
    assert np.all(result.physical_frame == pytest.approx(21.5))


def test_mac_synthetic_payload_access_is_prohibited_before_path_read() -> None:
    config = ConversionConfig(mode=STAGE1_MODE, source_split="train", source_domain="SYNTHETIC", safenest_role="TRAIN")
    with pytest.raises(SyntheticPayloadAccessProhibitedError):
        config.validate()
    with pytest.raises(SyntheticPayloadAccessProhibitedError):
        convert_partition(config=config, repo_root=ROOT, source_archive=Path("/never/read/train.zip"), artifact_dir=Path("/tmp/no"))


def test_colab_synthetic_configuration_is_explicit() -> None:
    config = ConversionConfig(mode=COLAB_STAGE2_MODE, source_split="validation", source_domain="SYNTHETIC", safenest_role="VALIDATION")
    config.validate()


def _write_frames(path: Path, frames: np.ndarray) -> None:
    np.save(path, np.asarray(frames, dtype=CANONICAL_DTYPE, order="C"))


def test_exact_and_near_duplicate_controlled_fixture(tmp_path: Path) -> None:
    base = np.full((62, 80), 20.0, dtype=np.float32)
    near = base.copy(); near[0, 0] += 0.05
    different = np.full((62, 80), 30.0, dtype=np.float32)
    artifact = tmp_path / "frames.npy"
    _write_frames(artifact, np.stack([base, base, near, different]))
    audit = audit_near_duplicates(artifact, 4)
    assert audit["profile"]["profile_id"] == "THERMAL_T_A6_NEAR_DUPLICATE_SCREEN_V1"
    assert audit["confirmed_pair_count"] >= 2
    assert any(set(cluster["sample_indices"]) >= {0, 1} for cluster in audit["confirmed_clusters"])


def test_near_duplicate_screen_rejects_clearly_different_fixture(tmp_path: Path) -> None:
    artifact = tmp_path / "frames.npy"
    _write_frames(artifact, np.stack([np.zeros((62, 80)), np.full((62, 80), 50.0)]))
    audit = audit_near_duplicates(artifact, 2)
    assert audit["confirmed_pair_count"] == 0


def test_near_duplicate_profile_is_frozen_and_label_independent() -> None:
    profile = near_duplicate_profile()
    assert profile["label_independent"] is True
    assert profile["model_independent"] is True
    assert profile["exhaustiveness_claim"] == "DETERMINISTIC_SCREENING_NOT_EXHAUSTIVE"
    assert profile["confirmation_thresholds"]["mae_celsius_max"] == 0.20


def test_colab_missing_source_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ColabExecutionError) as exc:
        run(mode="COLAB_STAGE2", drive_raw_root=tmp_path / "raw", work_root=tmp_path / "work", drive_output_root=tmp_path / "out", dry_run=True)
    assert exc.value.code == "SOURCE_PAYLOAD_INCOMPLETE"


def test_colab_mac_guard_does_not_inspect_synthetic_paths(tmp_path: Path) -> None:
    with pytest.raises(SyntheticMacAccessError) as exc:
        run(mode="MAC_STAGE1", drive_raw_root=tmp_path / "raw", work_root=tmp_path / "work", drive_output_root=tmp_path / "out")
    assert exc.value.code == "MAC_SYNTHETIC_PAYLOAD_ACCESS_PROHIBITED"


def test_storage_gate_rejects_insufficient_space(tmp_path: Path) -> None:
    with pytest.raises(ColabExecutionError) as exc:
        check_storage(tmp_path, 10**18)
    assert exc.value.code == "COLAB_STORAGE_INSUFFICIENT"


def test_output_writable_probe_is_removed(tmp_path: Path) -> None:
    result = check_output_writable(tmp_path / "out")
    assert result["writable"] is True
    assert not list((tmp_path / "out").glob("*.partial"))


def _zip_bytes(path: Path, payload: bytes) -> None:
    import zipfile

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("member", payload)


def test_multipart_independent_archives_are_not_reconstructed(tmp_path: Path) -> None:
    parts = []
    for index in range(4):
        part = tmp_path / f"train.zip.00{index + 1}"
        _zip_bytes(part, bytes([index]))
        parts.append(part)
    report = inspect_multipart_format(parts)
    assert report["format"] == "INDEPENDENT_ZIPS"
    with pytest.raises(ColabExecutionError):
        reconstruct_raw_byte_split(parts, tmp_path / "joined.zip", format_report=report)


def test_multipart_unknown_format_is_rejected(tmp_path: Path) -> None:
    parts = []
    for index in range(4):
        part = tmp_path / f"part{index}"
        part.write_bytes(b"not-a-zip" + bytes([index]))
        parts.append(part)
    report = inspect_multipart_format(parts)
    assert report["format"] == "UNKNOWN"


def test_resume_ledger_atomic_finalization_guard(tmp_path: Path) -> None:
    ledger = tmp_path / "resume.json"
    save_resume_ledger(ledger, {"finalized": False, "partitions": {"train": {"status": "PARTIAL"}}})
    assert load_resume_ledger(ledger)["finalized"] is False
    save_resume_ledger(ledger, {"finalized": True, "execution_result_bundle": "T-A6_execution_result"})
    assert load_resume_ledger(ledger)["finalized"] is True
    ledger.write_text('{"finalized": true}', encoding="utf-8")
    with pytest.raises(ColabExecutionError):
        load_resume_ledger(ledger)


def _copy_evidence(tmp_path: Path) -> Path:
    source = ROOT / "datasets/thermal/manifests/T-A6_full_conversion_integrity"
    target = tmp_path / "evidence"
    shutil.copytree(source, target)
    return target


def test_stage1_validator_does_not_claim_full_t_a6_or_t_b() -> None:
    evidence = ROOT / "datasets/thermal/manifests/T-A6_full_conversion_integrity"
    result = validate_evidence(repo_root=ROOT, evidence_dir=evidence, mode="REAL_STAGE1", check_checksums=True)
    assert result["full_t_a6_gate"] == "NOT_YET_COMPLETE"
    assert result["t_b_authorized"] is False


def test_full_dataset_mode_requires_stage2_bundle() -> None:
    evidence = ROOT / "datasets/thermal/manifests/T-A6_full_conversion_integrity"
    result = validate_evidence(repo_root=ROOT, evidence_dir=evidence, mode="FULL_DATASET", check_checksums=True)
    assert result["overall_outcome"] == "NOT_VERIFIABLE"
    assert result["stage1_gate"] == "NOT_YET_COMPLETE"


def test_full_dataset_mode_live_validates_stage2_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "T-A6_execution_result"
    _write_valid_bundle(bundle)
    result = validate_evidence(repo_root=ROOT, evidence_dir=bundle, mode="FULL_DATASET", check_checksums=True)
    assert result["evidence_validation"] == "PASS"
    assert result["overall_outcome"] == "PASS_WITH_LIMITATIONS"
    assert result["full_t_a6_gate"] == "T_A6_FULL_COMPLETE_WITH_LIMITATIONS"
    assert result["t_b_authorized"] is False
    assert result["stage2_validation"]["evidence_validation"] == "PASS"

    execution = bundle / "execution_summary.json"
    value = json.loads(execution.read_text(encoding="utf-8"))
    value["t_b_authorized"] = True
    execution.write_text(canonical_json(value), encoding="utf-8")
    result = validate_evidence(repo_root=ROOT, evidence_dir=bundle, mode="FULL_DATASET", check_checksums=False)
    assert result["evidence_validation"] == "FAIL"
    assert any(item["code"] == "STAGE2_DOWNSTREAM_GATE_ESCALATION" for item in result["errors"])


def test_validator_rejects_absolute_path(tmp_path: Path) -> None:
    evidence = _copy_evidence(tmp_path)
    path = evidence / "limitations.json"
    value = json.loads(path.read_text(encoding="utf-8")); value["bad_path"] = "/Users/example/private"
    path.write_text(canonical_json(value), encoding="utf-8")
    result = validate_evidence(repo_root=ROOT, evidence_dir=evidence, mode="REAL_STAGE1", check_checksums=False)
    assert any(item["code"] == "NONPORTABLE_PATH" for item in result["errors"])


def test_validator_rejects_tampered_near_duplicate_profile(tmp_path: Path) -> None:
    evidence = _copy_evidence(tmp_path)
    path = evidence / "near_duplicate_profile.json"
    value = json.loads(path.read_text(encoding="utf-8")); value["confirmation_thresholds"]["mae_celsius_max"] = 999.0
    path.write_text(canonical_json(value), encoding="utf-8")
    result = validate_evidence(repo_root=ROOT, evidence_dir=evidence, mode="REAL_STAGE1", check_checksums=False)
    assert any(item["code"] in {"CHECKSUM_MISMATCH", "NEAR_AUDIT_SCOPE_INVALID", "NEAR_DUPLICATE_PROFILE_TAMPERED"} for item in result["errors"])


def test_validator_rejects_locked_test_escalation(tmp_path: Path) -> None:
    evidence = _copy_evidence(tmp_path)
    path = evidence / "stage1_status.json"
    value = json.loads(path.read_text(encoding="utf-8")); value["t_b_authorized"] = True
    path.write_text(canonical_json(value), encoding="utf-8")
    result = validate_evidence(repo_root=ROOT, evidence_dir=evidence, mode="REAL_STAGE1", check_checksums=False)
    assert any(item["code"] == "DOWNSTREAM_GATE_ESCALATION" for item in result["errors"])


def test_validator_rejects_partial_or_unfinalized_real_artifact(tmp_path: Path) -> None:
    evidence = _copy_evidence(tmp_path)
    path = evidence / "stage1_status.json"
    value = json.loads(path.read_text(encoding="utf-8")); value["stage1_gate"] = "T_A6_STAGE1_COMPLETE"
    path.write_text(canonical_json(value), encoding="utf-8")
    registry = evidence / "real_canonical_artifact_registry.json"
    registry_value = json.loads(registry.read_text(encoding="utf-8")); registry_value["finalized_status"] = "PARTIAL"
    registry.write_text(canonical_json(registry_value), encoding="utf-8")
    result = validate_evidence(repo_root=ROOT, evidence_dir=evidence, mode="REAL_STAGE1", check_checksums=False)
    assert any(item["code"] == "REAL_ARTIFACT_NOT_FINALIZED" for item in result["errors"])


def test_validator_requires_pending_cross_role_audits(tmp_path: Path) -> None:
    evidence = _copy_evidence(tmp_path)
    path = evidence / "real_leakage_audit.json"
    value = json.loads(path.read_text(encoding="utf-8")); value["train_to_real_exact"] = 0
    path.write_text(canonical_json(value), encoding="utf-8")
    result = validate_evidence(repo_root=ROOT, evidence_dir=evidence, mode="REAL_STAGE1", check_checksums=False)
    assert any(item["code"] == "CROSS_ROLE_PENDING_STATE_INVALID" for item in result["errors"])


def test_legacy_npz_is_not_a_canonical_source() -> None:
    evidence = ROOT / "datasets/thermal/manifests/T-A6_full_conversion_integrity"
    limitations = json.loads((evidence / "limitations.json").read_text(encoding="utf-8"))
    assert limitations["legacy_npz"] == "LEGACY_NON_AUTHORITATIVE_NOT_USED"


def test_no_model_metrics_in_stage1_evidence() -> None:
    evidence = ROOT / "datasets/thermal/manifests/T-A6_full_conversion_integrity"
    for path in evidence.glob("*.json"):
        text = path.read_text(encoding="utf-8").lower()
        assert "macro_f1" not in text
        assert "prediction_distribution" not in text


def test_stage1_role_is_real_development_not_locked_test() -> None:
    evidence = ROOT / "datasets/thermal/manifests/T-A6_full_conversion_integrity"
    contract = json.loads((evidence / "canonical_dataset_contract.json").read_text(encoding="utf-8"))
    assert contract["safenest_role"] == "REAL_EVAL_DEVELOPMENT"
    assert contract["source_partition"] == "test"


def test_provenance_schema_requires_one_to_one_rows() -> None:
    evidence = ROOT / "datasets/thermal/manifests/T-A6_full_conversion_integrity"
    schema = json.loads((evidence / "sample_provenance_schema.json").read_text(encoding="utf-8"))
    assert schema["one_row_per"].startswith("canonical sample")
    assert schema["no_object_array"] is True
