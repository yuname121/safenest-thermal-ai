"""Focused, payload-free tests for the Thermal T-B1 Stage-1 contract."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

import datasets.thermal.t_b1_runner as runner
import scripts.validate_thermal_t_b1 as validator
from datasets.thermal.t_b1_model import (
    EXPECTED_PARAMETER_COUNT,
    PRIMARY_SEED,
    architecture_fingerprint,
    backend_info,
    create_small_cnn_baseline,
    fingerprint_weight_arrays,
    initial_weights,
)
from datasets.thermal.t_b1_preprocessing import (
    CLASS_ORDER,
    PreprocessingContractError,
    apply_p0,
    apply_p1,
    apply_p2,
    canonical_batch,
    compare_validation_rows,
    fit_p1_statistics,
    select_validation_winner,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / validator.EVIDENCE_REL


def _copy_evidence(tmp_path: Path) -> Path:
    target = tmp_path / "T-B1"
    shutil.copytree(EVIDENCE, target)
    return target


def _mutated_evidence(tmp_path: Path, filename: str, mutate) -> dict:
    target = _copy_evidence(tmp_path)
    path = target / filename
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # The stale checksum is intentional: a changed contract must fail closed.
    monkey = pytest.MonkeyPatch()
    monkey.setattr(validator, "_validate_predecessors", lambda repo_root, errors: {
        "T-A6": {"evidence_validation": "PASS"},
        "T-B0": {"evidence_validation": "PASS", "t_b1_authorized": "YES_WITH_LIMITATIONS"},
    })
    try:
        return validator.validate_evidence(repo_root=ROOT, evidence_dir=target)
    finally:
        monkey.undo()


def test_clean_stage1_validator_passes() -> None:
    result = validator.validate_evidence(repo_root=ROOT, evidence_dir=EVIDENCE)
    assert result["evidence_validation"] == "PASS"
    assert result["overall_outcome"] == "T_B1_STAGE1_IMPLEMENTATION_READY_WITH_LIMITATIONS"
    assert result["t_b2_authorized"] is False


@pytest.mark.parametrize(
    ("filename", "mutate"),
    [
        ("dataset_input_contract.json", lambda d: d["roles"]["TRAIN"].update(sha256="0" * 64)),
        ("dataset_input_contract.json", lambda d: d["roles"]["VALIDATION"].update(sha256="0" * 64)),
        ("dataset_input_contract.json", lambda d: d["roles"]["REAL_EVAL_DEVELOPMENT"].update(sha256="0" * 64)),
        ("dataset_input_contract.json", lambda d: d.update(legacy_npz_authority="ALLOWED")),
        ("dataset_input_contract.json", lambda d: d.update(random_resplit="ALLOWED")),
        ("dataset_input_contract.json", lambda d: d.update(hash_resplit="ALLOWED")),
        ("dataset_input_contract.json", lambda d: d["roles"]["REAL_EVAL_DEVELOPMENT"].update(winner_selection=True)),
        ("preprocessing_implementation_registry.json", lambda d: d["profiles"][1].update(fit_role="VALIDATION")),
        ("preprocessing_implementation_registry.json", lambda d: d["profiles"][1].update(real_fit_allowed=True)),
        ("training_runtime_contract.json", lambda d: d["budget"].update(augmentation="ENABLED")),
        ("training_runtime_contract.json", lambda d: d["budget"].update(class_weighting="ENABLED")),
        ("training_runtime_contract.json", lambda d: d["budget"].update(oversampling="ENABLED")),
        ("training_runtime_contract.json", lambda d: d["budget"].update(focal_loss="ENABLED")),
        ("initialization_contract.json", lambda d: d["seed_bindings"].update(numpy=7)),
        ("limitations.json", lambda d: d.update(near_duplicate_pairs=0)),
        ("limitations.json", lambda d: d.update(sensitivity_subset="FABRICATED_CLEAN_SUBSET")),
        ("expected_result_schema.json", lambda d: d.update(stage1_status="FINALIZED")),
        ("stage1_validation_result.json", lambda d: d.update(performance_winner_selected=True)),
        ("baseline_model_contract.json", lambda d: d.update(legacy_model_replacement=True)),
        ("t_b1_execution_contract.json", lambda d: d["runner"].update(canonical_root="/Users/owner/thermal")),
        ("external_storage_contract.json", lambda d: d["canonical_role_layouts"].append("archive/thermal.npy")),
    ],
)
def test_contract_tamper_is_rejected(tmp_path: Path, filename: str, mutate) -> None:
    result = _mutated_evidence(tmp_path, filename, mutate)
    assert result["evidence_validation"] == "FAIL"
    assert result["error_count"] > 0


def test_t_a6_predecessor_tamper_is_live_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_a6(**kwargs):
        calls.append("T-A6")
        return {"evidence_validation": "FAIL", "overall_outcome": "TAMPERED"}

    def fake_b0(**kwargs):
        calls.append("T-B0")
        return {"evidence_validation": "PASS", "t_b1_authorized": "YES_WITH_LIMITATIONS"}

    def fake_predecessors(repo_root, errors):
        a6 = fake_a6()
        b0 = fake_b0()
        errors.append({"code": "T_A6_PREDECESSOR_INVALID", "location": "T-A6", "message": "tampered"})
        return {"T-A6": a6, "T-B0": b0}

    monkeypatch.setattr(validator, "_validate_predecessors", fake_predecessors)
    result = validator.validate_evidence(repo_root=ROOT, evidence_dir=EVIDENCE, check_checksums=False)
    assert calls == ["T-A6", "T-B0"]
    assert result["evidence_validation"] == "FAIL"
    assert any(error["code"] == "T_A6_PREDECESSOR_INVALID" for error in result["errors"])


def test_p0_preserves_canonical_values_and_only_adds_channel() -> None:
    full = np.linspace(20.0, 24.25, 62 * 80, dtype="<f4").reshape(1, 62, 80)
    result = apply_p0(full)
    assert result.shape == (1, 62, 80, 1)
    np.testing.assert_array_equal(result[..., 0], full)
    assert result.dtype == np.dtype("<f4")


def test_p1_is_train_only_and_deterministic() -> None:
    train = np.arange(2 * 62 * 80, dtype="<f4").reshape(2, 62, 80) + 20.0
    first = fit_p1_statistics(train, train_artifact_sha256="train")
    second = fit_p1_statistics(train, train_artifact_sha256="train")
    assert first.to_dict() == second.to_dict()
    assert first.fit_role == "TRAIN"
    assert first.fit_sample_count == 2
    with pytest.raises(PreprocessingContractError):
        fit_p1_statistics(train, fit_role="VALIDATION")
    with pytest.raises(PreprocessingContractError):
        fit_p1_statistics(train, fit_role="REAL_EVAL_DEVELOPMENT")
    applied = apply_p1(train[:1], first)
    assert applied.shape == (1, 62, 80, 1)


def test_p2_matches_legacy_preparation_and_constant_behavior() -> None:
    from inference.thermal_interpreter import ThermalInterpreter

    varied = np.linspace(20.0, 30.0, 62 * 80, dtype="<f4").reshape(1, 62, 80)
    expected = ThermalInterpreter._prepare_float_frame(varied[0])
    actual = apply_p2(varied)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    constant = np.full((1, 62, 80), 42.0, dtype="<f4")
    expected_constant = ThermalInterpreter._prepare_float_frame(constant[0])
    np.testing.assert_allclose(apply_p2(constant), expected_constant, rtol=0, atol=0)


@pytest.mark.parametrize("bad", [
    np.zeros((61, 80), dtype="<f4"),
    np.zeros((62, 80), dtype="<i2"),
    np.full((62, 80), np.nan, dtype="<f4"),
    np.full((62, 80), np.inf, dtype="<f4"),
])
def test_canonical_shape_dtype_and_finite_checks(bad: np.ndarray) -> None:
    with pytest.raises(PreprocessingContractError):
        canonical_batch(bad)


def test_baseline_parameter_count_and_architecture_identity() -> None:
    model = create_small_cnn_baseline()
    assert int(model.count_params()) == EXPECTED_PARAMETER_COUNT
    assert architecture_fingerprint(model) == "937fceb88900a779d67a8e407cebc3362cc21346721f92cea5ae8de1413aba2a"
    _, fingerprint, _ = initial_weights(PRIMARY_SEED)
    assert fingerprint == "72a9805e1772aa0ee0a414fb7d41bc17ea0d183fa29e0e43049b22e0d5d35058"


def test_same_initial_weights_are_reproducible() -> None:
    first, first_fp, _ = initial_weights(PRIMARY_SEED)
    second, second_fp, _ = initial_weights(PRIMARY_SEED)
    assert first_fp == second_fp
    assert fingerprint_weight_arrays(first) == fingerprint_weight_arrays(second)


def test_winner_rule_is_validation_only_and_tie_deterministic() -> None:
    base = {"validation_metrics": {"macro_f1": 0.5, "balanced_accuracy": 0.5, "h_fall_posture_proxy_recall": 0.5}, "parameter_count": 10, "tflite_artifact_size_bytes": 100}
    left = {**base, "profile_id": "P0"}
    right = {**base, "profile_id": "P1"}
    assert compare_validation_rows(left, right) < 0
    tie_left = {**left, "profile_id": "SAME", "validation_metrics": {**base["validation_metrics"], "macro_f1": 0.500005}}
    tie_right = {**right, "profile_id": "SAME"}
    assert compare_validation_rows(tie_left, tie_right) == 0
    winner = select_validation_winner([right, left])
    assert winner["profile_id"] == "P0"
    with pytest.raises(PreprocessingContractError):
        select_validation_winner([{**left, "real_metrics": {"macro_f1": 1.0}}])


def test_stage1_does_not_require_external_ssd() -> None:
    result = runner.run(
        mode=runner.STAGE1_MODE,
        canonical_root="/path/that/is/not/required/in_stage1",
        work_root=ROOT / ".tmp-t-b1-test-work",
        output_root=ROOT / ".tmp-t-b1-test-output",
        repo_root=ROOT,
        dry_run=True,
    )
    assert result["status"] == "STAGE1_IMPLEMENTATION_READY"


def test_full_dry_run_missing_ssd_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(runner.RunnerContractError, match="EXTERNAL_CANONICAL_ROOT_UNAVAILABLE"):
        runner.run(
            mode=runner.FULL_MODE,
            canonical_root=tmp_path / "missing-canonical",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            repo_root=ROOT,
            dry_run=True,
        )


@pytest.mark.parametrize("wrong_role", ["TRAIN", "VALIDATION", "REAL_EVAL_DEVELOPMENT"])
def test_full_canonical_preflight_rejects_wrong_role_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wrong_role: str) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    for role, original in runner.EXPECTED_ROLES.items():
        role_dir = canonical / role
        role_dir.mkdir()
        array_path = role_dir / original["filename"]
        np.save(array_path, np.zeros((1, 62, 80), dtype="<f4"), allow_pickle=False)
        provenance = role_dir / original["provenance_filename"]
        provenance.write_text(
            json.dumps({"original_label_name": "LYING", "compatibility_target": "HUMAN_FALL"}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        spec = dict(original)
        spec["rows"] = 1
        spec["size_bytes"] = array_path.stat().st_size
        spec["sha256"] = "0" * 64 if role == wrong_role else runner.sha256_file(array_path)
        monkeypatch.setitem(runner.EXPECTED_ROLES, role, spec)
    with pytest.raises(runner.RunnerContractError, match="SHA-256 mismatch"):
        runner.validate_canonical_root(canonical, full_hash=True)


def test_full_mode_requires_owner_authorization(tmp_path: Path) -> None:
    with pytest.raises(runner.RunnerContractError, match="OWNER_AUTHORIZATION"):
        runner.run_full_experiment(
            canonical_root=tmp_path / "unused",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            repo_root=ROOT,
            dry_run=False,
            owner_authorized=False,
        )


def _write_full_fixture(root: Path) -> Path:
    """Create a compact, payload-free FULL_EXPERIMENT fixture."""

    bundle = root / "T-B1-full"
    bundle.mkdir()
    expected_roles = {
        "TRAIN": (32000, "749c847fc9ab50ea5eee8827f0d47b5ebaa48165732a382c59f8b96c565b9d93", "SYNTHETIC", "train_canonical.npy", "train_provenance.jsonl", "train-prov"),
        "VALIDATION": (8000, "5d16451702c1bccfa945d9188d9b29a26ce11c8b33bf7a0dfbb25bfa86d74610", "SYNTHETIC", "validation_canonical.npy", "validation_provenance.jsonl", "validation-prov"),
        "REAL_EVAL_DEVELOPMENT": (8000, "cd696e68aeec063cbc8185719b4f4dad3d038cb3d28eec0d3701b8311e4ad8f1", "REAL", "real_eval_development_canonical.npy", "real_eval_development_provenance.jsonl", "real-prov"),
    }
    dataset_roles = {
        role: {
            "role": role,
            "rows": rows,
            "sha256": digest,
            "provenance_sha256": provenance,
            "shape": [rows, 62, 80],
            "dtype": "float32_little_endian",
            "unit": "CELSIUS",
            "source_domain": domain,
        }
        for role, (rows, digest, domain, _array, _prov, provenance) in expected_roles.items()
    }
    source = np.repeat(np.asarray([0, 1, 2]), [2000, 4000, 2000]).astype(np.int32)
    metrics = runner.compute_metrics(source, source)
    _, initial_fp, architecture = initial_weights(PRIMARY_SEED)
    profile_rows = []
    checkpoint_rows = []
    for index, profile in enumerate(runner.PROFILE_IDS):
        checkpoint_name = f"{profile}.weights.h5"
        checkpoint_dir = bundle / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path = checkpoint_dir / checkpoint_name
        checkpoint_path.write_bytes(f"checkpoint-{index}".encode("ascii"))
        checkpoint_info = {"logical_path": f"checkpoints/{checkpoint_name}", "sha256": runner.sha256_file(checkpoint_path), "size_bytes": checkpoint_path.stat().st_size, "materialization": "PERSISTENT_EXTERNAL_OUTPUT"}
        checkpoint_rows.append({**checkpoint_info, "profile_id": profile})
        profile_rows.append({"profile_id": profile, "candidate_id": "SMALL_CNN_BASELINE_V1", "seed": PRIMARY_SEED, "status": "VALIDATION_COMPLETE", "initial_weight_fingerprint": initial_fp, "architecture_fingerprint": architecture, "parameter_count": 312131, "best_epoch": index + 1, "validation_metrics": metrics, "epoch_metrics": [], "checkpoint": checkpoint_info, "preprocessing_statistics": None})
    p1_stats = runner.P1Statistics(mean=1.0, std=2.0, fit_sample_count=32000, fit_pixel_count=32000 * 62 * 80, fit_role="TRAIN", train_artifact_sha256=expected_roles["TRAIN"][1])
    winner = dict(profile_rows[0])
    winner.update({"selection_role": "VALIDATION", "rule_id": "THERMAL_T_B0_WINNER_RULE_001", "tie_tolerance": 1e-5})
    documents = {
        "environment.json": {"schema_version": "1.0", "phase": "T-B1", "backend": {"physical_devices": [{"name": "/physical_device:CPU:0", "device_type": "CPU"}]}, "canonical_root": "CONFIGURABLE_EXTERNAL_STORAGE_ROOT", "work_root": "CONFIGURABLE_LOCAL_SCRATCH_ROOT", "output_root": "CONFIGURABLE_EXTERNAL_OUTPUT_ROOT"},
        "dataset_identity.json": {"canonical_root_configured": True, "roles": dataset_roles},
        "target_identity.json": {"target_class_order": list(CLASS_ORDER), "mapping": {"EMPTY_ROOM": "NOT_HUMAN", "SITTING": "HUMAN_NORMAL", "STANDING": "HUMAN_NORMAL", "LYING": "HUMAN_FALL"}, "lying_semantics": "DERIVED_POSTURE_PROXY_NOT_EVENT_GROUND_TRUTH"},
        "initialization_registry.json": {"schema_version": "1.0", "seed": PRIMARY_SEED, "candidate_id": "SMALL_CNN_BASELINE_V1", "initial_weight_fingerprint": initial_fp, "architecture_fingerprint": architecture, "parameter_count": 312131, "same_initial_weights_for_all_profiles": True},
        "p0_preprocessing.json": {"profile_id": runner.PROFILE_IDS[0]},
        "p1_preprocessing.json": {"profile_id": runner.PROFILE_IDS[1], **p1_stats.to_dict(), "statistics_checksum": p1_stats.checksum()},
        "p2_preprocessing.json": {"profile_id": runner.PROFILE_IDS[2]},
        "p0_training_summary.json": profile_rows[0],
        "p1_training_summary.json": profile_rows[1],
        "p2_training_summary.json": profile_rows[2],
        "validation_comparison.json": {"schema_version": "1.0", "selection_role": "VALIDATION", "primary_metric": "macro_f1", "tie_tolerance": 1e-5, "rule_id": "THERMAL_T_B0_WINNER_RULE_001", "candidates": [{"profile_id": row["profile_id"], "candidate_id": row["candidate_id"], "best_epoch": row["best_epoch"], "parameter_count": row["parameter_count"], "validation_metrics": row["validation_metrics"]} for row in profile_rows], "winner_profile_id": runner.PROFILE_IDS[0]},
        "winner_selection.json": winner,
        "real_eval_development.json": {"schema_version": "1.0", "role": "REAL_EVAL_DEVELOPMENT", "reporting_view": "POST_SELECTION_REAL_DOMAIN_DEVELOPMENT_CHARACTERIZATION", "profile_id": runner.PROFILE_IDS[0], "checkpoint": winner["checkpoint"], "metrics": metrics, "used_for_winner_selection": False, "used_for_preprocessing_fit": False, "locked_test": False},
        "checkpoint_registry.json": {"schema_version": "1.0", "storage_scope": "SSD_EXTERNAL_PERSISTENT", "checkpoint_count": 3, "checkpoints": checkpoint_rows, "winner_checkpoint": winner["checkpoint"], "bulk_checkpoints_tracked_in_git": False},
        "metrics_registry.json": {"schema_version": "1.0", "validation": [], "real_eval_development": {"profile_id": runner.PROFILE_IDS[0], "metrics": metrics}},
        "limitations.json": {"schema_version": "1.0", "locked_test_available": False, "subject_generalization": "NOT_VERIFIABLE", "near_duplicate_pairs": 14514, "sensitivity_subset": "SENSITIVITY_SUBSET_NOT_MATERIALIZABLE_FROM_CURRENT_COMPACT_EVIDENCE", "synthetic_real_domain_gap": "LARGE_SYNTHETIC_TO_REAL_DOMAIN_GAP_OBSERVED_NOT_DEPLOYMENT_VALIDATION"},
        "execution_summary.json": {"schema_version": "1.0", "status": "FINALIZED", "phase": "T-B1", "mode": "FULL_EXPERIMENT", "profile_order": list(runner.PROFILE_IDS), "selected_profile_id": runner.PROFILE_IDS[0], "full_training_performed": True, "new_trained_model_generated": True, "t_b2_authorized": "YES_WITH_LIMITATIONS"},
    }
    for name, value in documents.items():
        (bundle / name).write_text(runner.canonical_json(value), encoding="utf-8")
    _write_fixture_checksums(bundle)
    return bundle


def _write_fixture_checksums(bundle: Path) -> None:
    entries = []
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            entries.append(f"{runner.sha256_file(path)}  {path.relative_to(bundle).as_posix()}")
    (bundle / "checksums.sha256").write_text("\n".join(entries) + "\n", encoding="utf-8")


def test_full_validator_passes_materialized_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fixture = _write_full_fixture(tmp_path)
    monkeypatch.setattr(validator, "_validate_predecessors", lambda repo_root, errors: {"T-A6": {"evidence_validation": "PASS"}, "T-B0": {"evidence_validation": "PASS", "t_b1_authorized": "YES_WITH_LIMITATIONS"}})
    result = validator.validate_evidence(repo_root=ROOT, evidence_dir=fixture, mode="FULL_EXPERIMENT")
    assert result["evidence_validation"] == "PASS"
    assert result["t_b2_authorized"] == "YES_WITH_LIMITATIONS"


def test_full_validator_rejects_semantic_tamper_with_fresh_checksums(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fixture = _write_full_fixture(tmp_path)
    real_path = fixture / "real_eval_development.json"
    real = json.loads(real_path.read_text(encoding="utf-8"))
    real["used_for_winner_selection"] = True
    real_path.write_text(runner.canonical_json(real), encoding="utf-8")
    _write_fixture_checksums(fixture)
    monkeypatch.setattr(validator, "_validate_predecessors", lambda repo_root, errors: {"T-A6": {"evidence_validation": "PASS"}, "T-B0": {"evidence_validation": "PASS", "t_b1_authorized": "YES_WITH_LIMITATIONS"}})
    result = validator.validate_evidence(repo_root=ROOT, evidence_dir=fixture, mode="FULL_EXPERIMENT")
    assert result["evidence_validation"] == "FAIL"
    assert any(error["code"] == "REAL_EVALUATION_ORDER_INVALID" for error in result["errors"])


def test_cpu_backend_policy_allows_missing_gpu() -> None:
    info = backend_info()
    assert info["gpu_optional"] is True
    assert info["backend_selected"] in {"CPU", "APPLE_METAL"}


def test_fixture_result_cannot_be_finalized_in_stage1() -> None:
    schema = json.loads((EVIDENCE / "expected_result_schema.json").read_text(encoding="utf-8"))
    assert schema["stage1_status"] == "PENDING_FULL_EXPERIMENT"
    assert "full_performance_metrics" in schema["stage1_forbidden"]


def test_legacy_tflite_is_not_replaced() -> None:
    contract = json.loads((EVIDENCE / "baseline_model_contract.json").read_text(encoding="utf-8"))
    assert contract["legacy_model_replacement"] is False
    assert contract["tflite_conversion_in_stage1"] is False


def test_runner_role_order_is_frozen() -> None:
    assert runner.ROLE_ORDER == ("TRAIN", "VALIDATION", "REAL_EVAL_DEVELOPMENT")
    assert list(CLASS_ORDER) == ["NOT_HUMAN", "HUMAN_NORMAL", "HUMAN_FALL"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
