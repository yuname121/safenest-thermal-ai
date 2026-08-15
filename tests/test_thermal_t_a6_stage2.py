"""Focused non-hydrating tests for the Thermal T-A6 Stage 2 evidence contract."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from datasets.thermal.canonical_converter import (
    CANONICAL_DTYPE,
    CANONICAL_SHAPE,
    COLAB_STAGE2_MODE,
    ConversionConfig,
    ConversionIncompleteError,
    canonical_json,
    convert_sdt_partition,
    sha256_bytes,
)
from datasets.thermal.canonical_geometry import _canonical_hash  # type: ignore[attr-defined]
from datasets.thermal.canonical_converter import near_duplicate_profile
from datasets.thermal.t_a6_stage2 import (
    BUNDLE_JSON_FILES,
    ROLE_ORDER,
    audit_cross_role_leakage,
    audit_exact_duplicates,
    audit_near_duplicates_cross_role,
    build_checksums,
    validate_role_artifact,
    validate_stage2_bundle,
    verify_synthetic_source_contract,
)
from scripts.run_thermal_t_a6_colab import ROLE_ORDER as RUNNER_ROLE_ORDER
from scripts.run_thermal_t_a6_colab import ColabExecutionError, _reuse_or_convert, startup_checks


def test_colab_runner_uses_canonical_role_order() -> None:
    assert RUNNER_ROLE_ORDER == ROLE_ORDER


def _png(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint16), mode="I;16").save(stream, format="PNG")
    return stream.getvalue()


def test_synthetic_source_contract_checks_schema_unit_and_labels(tmp_path: Path) -> None:
    archive_path = tmp_path / "validation.zip"
    thermal = np.full((480, 640), 29_315, dtype=np.uint16)
    depth = np.full((480, 640), 1_500, dtype=np.uint16)
    labels = "0,0,0,640,480\n1,0,0,640,480\n"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("validation/labels.txt", labels)
        for index in range(2):
            archive.writestr(f"validation/image_t_{index}.png", _png(thermal + index))
            archive.writestr(f"validation/image_d_{index}.png", _png(depth + index))
    result = verify_synthetic_source_contract(archive_path, source_split="validation", expected_count=2)
    assert result["physical_contract"] == "PASS"
    assert result["thermal_member_count"] == 2
    assert result["depth_unit"] == "MILLIMETRES"


def test_synthetic_conversion_failure_is_explicit_and_not_finalized(tmp_path: Path) -> None:
    archive_path = tmp_path / "validation.zip"
    thermal = np.full((480, 640), 29_315, dtype=np.uint16)
    labels = "0,0,0,640,480\n1,0,0,640,480\n"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("validation/labels.txt", labels)
        archive.writestr("validation/image_t_0.png", _png(thermal))
        archive.writestr("validation/image_t_1.png", b"not-a-png")
    with pytest.raises(ConversionIncompleteError):
        convert_sdt_partition(
            config=ConversionConfig(mode=COLAB_STAGE2_MODE, source_split="validation", source_domain="SYNTHETIC", safenest_role="VALIDATION"),
            source_archive=archive_path,
            artifact_dir=tmp_path / "artifacts",
            expected_count=2,
        )
    assert not (tmp_path / "artifacts" / "validation_canonical.npy").exists()


def _role_data(tmp_path: Path) -> dict[str, dict[str, object]]:
    roles: dict[str, dict[str, object]] = {}
    for role, value in (("TRAIN", 20.0), ("VALIDATION", 20.05), ("REAL_EVAL_DEVELOPMENT", 40.0)):
        artifact = tmp_path / f"{role}.npy"
        frame = np.full(CANONICAL_SHAPE, value, dtype=CANONICAL_DTYPE)
        np.save(artifact, frame[None, ...])
        source_hash = sha256_bytes(f"{role}-source".encode())
        decoded_hash = sha256_bytes(f"{role}-decoded".encode())
        row = {
            "canonical_sample_index": 0,
            "source_dataset_id": "local_sdt_zenodo_4124309",
            "source_split": role.lower() if role != "REAL_EVAL_DEVELOPMENT" else "test",
            "source_member": f"{role.lower()}/image_t_0.png",
            "source_frame_index": 0,
            "source_archive_sha256": sha256_bytes(b"archive"),
            "source_member_sha256": source_hash,
            "source_frame_sha256": decoded_hash,
            "canonical_frame_hash": _canonical_hash(frame, np.ones(CANONICAL_SHAPE, dtype=bool)),
            "canonical_tensor_row_sha256": sha256_bytes(frame.tobytes(order="C")),
            "source_subject_status": "ABSENT",
            "source_session_status": "ABSENT",
            "source_sequence_status": "ABSENT",
            "source_event_status": "ABSENT",
            "safenest_assignment": {"safenest_assignment_role": role},
        }
        provenance = tmp_path / f"{role}.jsonl"
        provenance.write_text(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
        roles[role] = validate_role_artifact(role=role, artifact_path=artifact, provenance_path=provenance, expected_count=1)
    # Deliberately create one exact/near overlap between TRAIN and VALIDATION.
    roles["VALIDATION"]["rows"][0]["source_member_sha256"] = roles["TRAIN"]["rows"][0]["source_member_sha256"]
    roles["VALIDATION"]["rows"][0]["source_frame_sha256"] = roles["TRAIN"]["rows"][0]["source_frame_sha256"]
    return roles


def test_cross_role_exact_and_near_audits_are_deterministic(tmp_path: Path) -> None:
    roles = _role_data(tmp_path)
    exact = audit_exact_duplicates(roles)
    pair = exact["cross_role"]["source_member_byte_hashes"]["TRAIN__VALIDATION"]
    assert pair["overlap_cluster_count"] == 1
    near = audit_near_duplicates_cross_role(roles)
    assert near["profile"]["profile_id"] == "THERMAL_T_A6_NEAR_DUPLICATE_SCREEN_V1"
    assert near["cross_role_confirmed_pair_count"] >= 1
    leakage = audit_cross_role_leakage(roles, near)
    assert leakage["subject_leakage"]["status"] == "NOT_VERIFIABLE"
    assert leakage["near_duplicate_screening"]["status"] == "MEASURED"


def _write_valid_bundle(bundle: Path) -> None:
    bundle.mkdir(parents=True, exist_ok=True)
    roles = {
        role: {
            "source_split": "test" if role == "REAL_EVAL_DEVELOPMENT" else role.lower(),
            "source_domain": "REAL" if role == "REAL_EVAL_DEVELOPMENT" else "SYNTHETIC",
            "expected_count": count,
            "source_frames_measured": count,
            "canonical_rows": count,
            "canonical_shape": [62, 80],
            "canonical_dtype": "float32_little_endian",
            "canonical_unit": "CELSIUS",
            "artifact_sha256": "a" * 64,
            "provenance_sha256": "b" * 64,
            "artifact_size_bytes": 1,
            "provenance_size_bytes": 1,
            "artifact_path": f"canonical/{role}.npy",
            "provenance_path": f"canonical/{role}.jsonl",
        }
        for role, count in (("TRAIN", 32000), ("VALIDATION", 8000), ("REAL_EVAL_DEVELOPMENT", 8000))
    }
    status_roles = {
        role: {
            "expected_source_frames": record["expected_count"],
            "source_frames_measured": record["expected_count"],
            "status_counts": {"SUCCESS": record["expected_count"], "SUCCESS_WITH_WARNING": 0, "EXCLUDED": 0, "FAILED": 0},
            "canonical_rows": record["expected_count"],
            "finalized_status": "FINALIZED",
        }
        for role, record in roles.items()
    }
    docs = {
        "execution_summary.json": {"phase": "T-A6_COLAB_STAGE2", "status": "FULL_AUDIT_COMPLETE_WITH_LIMITATIONS", "t_b_authorized": False},
        "source_identity.json": {"phase": "T-A6_COLAB_STAGE2", "synthetic_physical_contract": {"train": {"physical_contract": "PASS"}, "validation": {"physical_contract": "PASS"}}},
        "canonical_artifact_registry.json": {"phase": "T-A6_COLAB_STAGE2", "locked_test_available": False, "roles": roles},
        "conversion_status_summary.json": {"phase": "T-A6_COLAB_STAGE2", "roles": status_roles},
        "output_checksums.json": {"phase": "T-A6_COLAB_STAGE2", "roles": {role: {"artifact_sha256": record["artifact_sha256"], "provenance_sha256": record["provenance_sha256"]} for role, record in roles.items()}},
        "quality_audit_summary.json": {"phase": "T-A6_COLAB_STAGE2", "overall_status": "PASS"},
        "exact_duplicate_audit.json": {"phase": "T-A6_COLAB_STAGE2", "audit_scope": "WITHIN_ROLE_AND_CROSS_ROLE", "layers": {"source_member_byte_hashes": "source_member_sha256", "decoded_frame_hashes": "source_frame_sha256", "canonical_frame_hashes": "canonical_frame_hash"}},
        "near_duplicate_audit.json": {"phase": "T-A6_COLAB_STAGE2", "profile": near_duplicate_profile(), "exhaustiveness_claim": "DETERMINISTIC_SCREENING_NOT_EXHAUSTIVE"},
        "cross_role_leakage_audit.json": {"phase": "T-A6_COLAB_STAGE2", "source_identity_overlap": {}, "source_member_identity_overlap": {}, "source_frame_id_overlap": {}, "source_member_leakage": {}, "exact_content_leakage": {}, "canonical_content_leakage": {}, "near_duplicate_screening": {}, "subject_leakage": {"status": "NOT_VERIFIABLE"}, "session_leakage": {"status": "NOT_VERIFIABLE"}, "sequence_leakage": {"status": "NOT_VERIFIABLE"}, "event_leakage": {"status": "NOT_VERIFIABLE"}},
        "determinism_summary.json": {"phase": "T-A6_COLAB_STAGE2", "status": "PASS", "full_second_conversion": True, "artifact_checksum_match": True, "provenance_checksum_match": True},
        "execution_environment.json": {"phase": "T-A6_COLAB_STAGE2", "gpu_required": False},
    }
    for name in ("exact_duplicate_audit.json", "near_duplicate_audit.json", "cross_role_leakage_audit.json"):
        docs[name]["audit_sha256"] = sha256_bytes(canonical_json(docs[name]).encode("utf-8"))
    for name, value in docs.items():
        (bundle / name).write_text(canonical_json(value), encoding="utf-8")
    preliminary = validate_stage2_bundle(bundle, require_validation_result=False)
    (bundle / "validation_result.json").write_text(canonical_json(preliminary), encoding="utf-8")
    build_checksums(bundle)


def test_stage2_compact_bundle_passes_without_bulk_payload(tmp_path: Path) -> None:
    bundle = tmp_path / "T-A6_execution_result"
    _write_valid_bundle(bundle)
    result = validate_stage2_bundle(bundle)
    assert result["evidence_validation"] == "PASS"
    assert result["overall_outcome"] == "PASS_WITH_LIMITATIONS"
    assert result["t_b_authorized"] is False


def test_stage2_validator_rejects_absolute_path_and_locked_test(tmp_path: Path) -> None:
    bundle = tmp_path / "T-A6_execution_result"
    _write_valid_bundle(bundle)
    source = json.loads((bundle / "source_identity.json").read_text(encoding="utf-8"))
    source["machine_path"] = "/Users/example/SafeNest"
    (bundle / "source_identity.json").write_text(canonical_json(source), encoding="utf-8")
    registry = json.loads((bundle / "canonical_artifact_registry.json").read_text(encoding="utf-8"))
    registry["locked_test_available"] = True
    (bundle / "canonical_artifact_registry.json").write_text(canonical_json(registry), encoding="utf-8")
    build_checksums(bundle)
    result = validate_stage2_bundle(bundle)
    codes = {item["code"] for item in result["errors"]}
    assert "NONPORTABLE_PATH" in codes
    assert "LOCKED_TEST_ESCALATION" in codes


def test_partition_resume_requires_finalized_checksum_backed_outputs(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifact = artifact_root / "train_canonical.npy"
    provenance = artifact_root / "train_provenance.jsonl"
    ledger = artifact_root / "train_conversion_ledger.json"
    np.save(artifact, np.zeros((1, 62, 80), dtype=CANONICAL_DTYPE))
    provenance.write_text("{}\n", encoding="utf-8")
    ledger.write_text(canonical_json({"finalized_status": "PARTIAL"}), encoding="utf-8")
    try:
        _reuse_or_convert(artifact_root=artifact_root, stem="train", converter=lambda: {"unexpected": True})
    except ColabExecutionError as exc:
        assert exc.code == "PARTITION_NOT_FINALIZED"
    else:
        raise AssertionError("partial partition must not resume")


def test_startup_rejects_expected_size_mismatch_before_storage_gate(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    for name in ("train.zip.001", "train.zip.002", "train.zip.003", "train.zip.004", "validation.zip", "test.zip"):
        (raw / name).parent.mkdir(parents=True, exist_ok=True)
        (raw / name).write_bytes(b"placeholder")
    try:
        startup_checks(drive_raw_root=raw, work_root=tmp_path / "work", drive_output_root=out, include_real_test=True)
    except ColabExecutionError as exc:
        assert exc.code == "SOURCE_PAYLOAD_SIZE_MISMATCH"
    else:
        raise AssertionError("incorrect logical source size must fail closed")
