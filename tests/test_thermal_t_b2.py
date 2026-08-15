"""Focused, payload-free tests for the Thermal T-B2 contract."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

import scripts.validate_thermal_t_b2 as validator
from datasets.thermal.t_b2_model import (
    ARCHITECTURE_IDS,
    DEPTHWISE_ID,
    SMALL_CNN_ID,
    architecture_contract,
    architecture_fingerprint,
    initial_weights,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / validator.EVIDENCE_REL


def _copy_bundle(tmp_path: Path) -> Path:
    target = tmp_path / "T-B2"
    shutil.copytree(EVIDENCE, target, dirs_exist_ok=True)
    return target


def _refresh_checksums(bundle: Path) -> None:
    rows = []
    for path in sorted(bundle.iterdir()):
        if path.is_file() and path.name != "checksums.sha256":
            rows.append(f"{validator.sha256_file(path)}  {path.name}")
    (bundle / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _mutate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str, mutate, *, fresh_checksums: bool = True) -> dict:
    bundle = _copy_bundle(tmp_path)
    path = bundle / filename
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    path.write_text(validator.canonical_json(document), encoding="utf-8")
    if fresh_checksums:
        _refresh_checksums(bundle)
    monkeypatch.setattr(validator, "_validate_predecessors", lambda repo_root, errors: {
        "T-A6": {"evidence_validation": "PASS", "overall_outcome": "PASS_WITH_LIMITATIONS"},
        "T-B0": {"evidence_validation": "PASS", "overall_outcome": "PASS_WITH_LIMITATIONS"},
        "T-B1": {"evidence_validation": "PASS", "overall_outcome": "T_B1_FULL_COMPLETE_WITH_LIMITATIONS", "t_b2_authorized": "YES_WITH_LIMITATIONS"},
    })
    return validator.validate_evidence(repo_root=ROOT, evidence_dir=bundle, mode="FULL_EXPERIMENT", check_checksums=True)


def test_clean_selected_comparison_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validator, "_validate_predecessors", lambda repo_root, errors: {
        "T-A6": {"evidence_validation": "PASS", "overall_outcome": "PASS_WITH_LIMITATIONS"},
        "T-B0": {"evidence_validation": "PASS", "overall_outcome": "PASS_WITH_LIMITATIONS"},
        "T-B1": {"evidence_validation": "PASS", "overall_outcome": "T_B1_FULL_COMPLETE_WITH_LIMITATIONS", "t_b2_authorized": "YES_WITH_LIMITATIONS"},
    })
    result = validator.validate_evidence(repo_root=ROOT, evidence_dir=EVIDENCE, mode="FULL_EXPERIMENT", check_checksums=True)
    assert result["evidence_validation"] == "PASS"
    assert result["overall_outcome"] == "T_B2_COMPLETE_WITH_LIMITATIONS"


def test_invalid_b1_predecessor_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validator, "_validate_predecessors", lambda repo_root, errors: {
        "T-A6": {"evidence_validation": "PASS"}, "T-B0": {"evidence_validation": "PASS"}, "T-B1": {"evidence_validation": "FAIL"}
    })
    result = validator.validate_evidence(repo_root=ROOT, evidence_dir=EVIDENCE, mode="FULL_EXPERIMENT", check_checksums=False)
    assert result["evidence_validation"] == "FAIL"
    assert result["predecessors"]["T-B1"]["evidence_validation"] == "FAIL"


@pytest.mark.parametrize("candidate_id", ["P0_CANONICAL_CELSIUS_DIRECT", "P2_LEGACY_PER_FRAME_MINMAX"])
def test_p0_and_p2_are_not_architecture_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, candidate_id: str) -> None:
    result = _mutate(tmp_path, monkeypatch, "architecture_candidate_registry.json", lambda d: d["candidates"].__setitem__(0, {**d["candidates"][0], "candidate_id": candidate_id}))
    assert result["evidence_validation"] == "FAIL"
    assert any(error["code"] in {"ARCHITECTURE_REGISTRY_INVALID", "ARCHITECTURE_CONTRACT_MISMATCH"} for error in result["errors"])


def test_p1_tamper_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _mutate(tmp_path, monkeypatch, "p1_lock.json", lambda d: d.update(mean=float(d["mean"]) + 1.0))
    assert result["evidence_validation"] == "FAIL"
    assert any(error["code"] in {"P1_CHECKSUM_INVALID", "P1_STATISTICS_RECOMPUTE_MISMATCH"} for error in result["errors"])


def test_canonical_identity_mismatch_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _mutate(tmp_path, monkeypatch, "dataset_lock.json", lambda d: d["roles"]["TRAIN"].update(sha256="0" * 64))
    assert result["evidence_validation"] == "FAIL"
    assert any(error["code"] == "CANONICAL_IDENTITY_INVALID" for error in result["errors"])


def test_unregistered_architecture_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _mutate(tmp_path, monkeypatch, "architecture_candidate_registry.json", lambda d: d["unregistered_candidates"].append("THIRD_CNN"))
    assert result["evidence_validation"] == "FAIL"
    assert any(error["code"] == "UNREGISTERED_ARCHITECTURE" for error in result["errors"])


def test_depthwise_parameter_violation_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _mutate(tmp_path, monkeypatch, "depthwise_architecture_contract.json", lambda d: d.update(parameter_count=30001))
    assert result["evidence_validation"] == "FAIL"
    assert any(error["code"] == "DEPTHWISE_PARAMETER_VIOLATION" for error in result["errors"])


def test_post_metric_architecture_modification_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _mutate(tmp_path, monkeypatch, "depthwise_architecture_contract.json", lambda d: d.update(frozen_before_metrics=False))
    assert result["evidence_validation"] == "FAIL"
    assert any(error["code"] == "POST_METRIC_ARCHITECTURE_TUNING" for error in result["errors"])


def test_invalid_and_valid_baseline_reuse(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = _mutate(tmp_path, monkeypatch, "small_cnn_baseline_reuse_assessment.json", lambda d: d.update(eligible=False))
    assert result["evidence_validation"] == "FAIL"
    assert any(error["code"] == "BASELINE_REUSE_INVALID" for error in result["errors"])


def test_different_architectures_have_distinct_initial_fingerprints() -> None:
    assert architecture_fingerprint(SMALL_CNN_ID) != architecture_fingerprint(DEPTHWISE_ID)
    assert initial_weights(SMALL_CNN_ID)[1] != initial_weights(DEPTHWISE_ID)[1]
    assert architecture_contract(DEPTHWISE_ID)["parameter_count"] <= 30000
    assert ARCHITECTURE_IDS == (SMALL_CNN_ID, DEPTHWISE_ID)


def test_winner_accepts_validation_only_and_rejects_real_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _mutate(tmp_path, monkeypatch, "winner_selection.json", lambda d: d["selection_input_metrics"][0].update(real_metrics={"macro_f1": 1.0}))
    assert result["evidence_validation"] == "FAIL"
    assert any(error["code"] in {"REAL_METRICS_IN_SELECTION", "REAL_SELECTION_CONTAMINATION"} for error in result["errors"])


def test_real_tie_break_and_losing_candidate_evaluation_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = _mutate(tmp_path, monkeypatch, "winner_selection.json", lambda d: d.update(tie_break_level="REAL_MACRO_F1"))
    assert any(error["code"] == "REAL_TIE_BREAK_FORBIDDEN" for error in first["errors"])
    second = _mutate(tmp_path, monkeypatch, "real_eval_development.json", lambda d: d.update(losing_candidate_new_real_evaluation=True))
    assert any(error["code"] == "LOSING_REAL_EVALUATED" for error in second["errors"])


def test_forbidden_augmentation_class_weight_and_refit_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _mutate(tmp_path, monkeypatch, "training_result.json", lambda d: d["training_contract"]["augmentation"].update(baseline="ENABLED"))
    assert any(error["code"] == "FORBIDDEN_TRAINING_STRATEGY" for error in result["errors"])
    result = _mutate(tmp_path, monkeypatch, "p1_lock.json", lambda d: d.update(validation_fit=True))
    assert any(error["code"] == "P1_REFIT_POLICY_INVALID" for error in result["errors"])
    result = _mutate(tmp_path, monkeypatch, "dataset_lock.json", lambda d: d.update(legacy_npz_used=True))
    assert any(error["code"] == "DATASET_SCOPE_ESCALATION" for error in result["errors"])


def test_absolute_archive_and_nondeterministic_paths_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _mutate(tmp_path, monkeypatch, "environment.json", lambda d: d.update(output_root="/Volumes/SafeNestssd/T-B2"))
    assert any(error["code"] == "NONPORTABLE_PATH" for error in result["errors"])
    result = _mutate(tmp_path, monkeypatch, "dataset_lock.json", lambda d: d["roles"]["TRAIN"].update(array_path="archive/train.npy"))
    assert any(error["code"] == "ARCHIVE_TREATED_AS_ACTIVE" for error in result["errors"])
    nondeterministic_root = tmp_path / "nondeterministic"
    nondeterministic_root.mkdir()
    bundle = _copy_bundle(nondeterministic_root)
    path = bundle / "t_b2_protocol.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    reversed_value = {key: value[key] for key in reversed(list(value))}
    path.write_text(json.dumps(reversed_value, sort_keys=False, indent=2) + "\n", encoding="utf-8")
    result = validator.validate_evidence(repo_root=ROOT, evidence_dir=bundle, mode="FULL_EXPERIMENT", check_checksums=False)
    assert any(error["code"] == "NONDETERMINISTIC_JSON" for error in result["errors"])


def test_candidate_failing_mandatory_selection_requirement_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _mutate(tmp_path, monkeypatch, "architecture_candidate_registry.json", lambda d: d["candidates"][1].update(winner_eligible=False))
    assert any(error["code"] == "CANDIDATE_NOT_WINNER_ELIGIBLE" for error in result["errors"])


def test_next_phase_not_started(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _mutate(tmp_path, monkeypatch, "t_b2_protocol.json", lambda d: d.update(next_phase_started=True))
    assert any(error["code"] == "SCOPE_ESCALATION" for error in result["errors"])
