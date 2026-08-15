"""Payload-free focused tests for the Thermal T-B3 evidence contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from datasets.thermal.t_b1_preprocessing import canonical_json
from scripts import validate_thermal_t_b3 as validator


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / validator.EVIDENCE_REL


def _json(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def _errors(function, document, *args):
    errors: list[dict[str, str]] = []
    function(document, *args, errors)
    return errors


def test_protocol_rejects_fourth_seed_and_temporal_work():
    document = _json("t_b3_protocol.json")
    document["seeds"] = [20260813, 20260814, 20260815, 20260816]
    document["temporal_training"] = "ALLOWED"
    errors = _errors(validator._validate_protocol, document)
    assert {item["code"] for item in errors} >= {"SEED_SET_INVALID", "SCOPE_ESCALATION"}


def test_protocol_rejects_real_selection_and_best_seed_cherry_picking():
    document = _json("t_b3_protocol.json")
    document["real_policy"] = "REAL_SELECTS_WINNER"
    document["best_seed_cherry_picking"] = "ALLOWED"
    errors = _errors(validator._validate_protocol, document)
    assert {item["code"] for item in errors} >= {"REAL_SELECTION_POLICY_INVALID", "CHECKPOINT_SELECTION_POLICY_INVALID"}


def test_dataset_rejects_legacy_npz_and_resplit():
    document = _json("dataset_lock.json")
    document["legacy_npz_used"] = True
    document["new_split_created"] = True
    errors: list[dict[str, str]] = []
    validator._validate_dataset(document, errors, [])
    assert any(item["code"] == "DATASET_SCOPE_ESCALATION" for item in errors)


def test_dataset_rejects_changed_canonical_role_checksum():
    document = _json("dataset_lock.json")
    document["roles"]["TRAIN"]["sha256"] = "0" * 64
    errors: list[dict[str, str]] = []
    validator._validate_dataset(document, errors, [])
    assert any(item["code"] == "CANONICAL_IDENTITY_INVALID" for item in errors)


def test_p1_rejects_refit_and_statistic_drift():
    document = _json("p1_lock.json")
    document["mean"] += 0.25
    document["refit"] = True
    errors = _errors(validator._validate_p1, document)
    assert {item["code"] for item in errors} >= {"P1_STATISTICS_MISMATCH", "P1_REFIT_POLICY_INVALID"}


def test_architecture_lock_rejects_depthwise_or_modified_candidate():
    document = _json("architecture_lock.json")
    document["candidate_id"] = "DEPTHWISE_SEPARABLE_CNN"
    document["modified"] = True
    errors = _errors(validator._validate_architecture, document)
    assert any(item["code"] == "ARCHITECTURE_LOCK_INVALID" for item in errors)


def test_reuse_assessment_rejects_checkpoint_or_contract_drift():
    document = _json("seed_20260813_reuse_assessment.json")
    document["checkpoint"]["sha256"] = "f" * 64
    document["contract_comparison"]["p1_statistics"] = False
    errors = _errors(validator._validate_reuse, document)
    assert {item["code"] for item in errors} >= {"SEED_REUSE_CONTRACT_MISMATCH", "SEED_REUSE_CHECKPOINT_MISMATCH"}


def test_seed_registry_rejects_unexpected_seed():
    document = _json("seed_registry.json")
    document["extra_seeds"] = [20260816]
    errors = _errors(validator._validate_seed_registry, document, False)
    assert any(item["code"] == "SEED_REGISTRY_INVALID" for item in errors)


def test_metrics_reject_class_order_and_support_corruption():
    source = json.loads((ROOT / validator.TB1_REL / "validation_comparison.json").read_text(encoding="utf-8"))
    metrics = next(item for item in source["candidates"] if item["profile_id"] == validator.P1_PROFILE)["validation_metrics"]
    metrics = copy.deepcopy(metrics)
    metrics["class_order"] = ["HUMAN_FALL", "HUMAN_NORMAL", "NOT_HUMAN"]
    metrics["per_class"]["HUMAN_FALL"]["support"] = 1
    errors: list[dict[str, str]] = []
    validator._validate_metric_consistency(metrics, "metrics", errors)
    assert any(item["code"] == "METRIC_CONTRACT_INVALID" for item in errors)


def test_aggregate_recomputes_mean_std_and_rejects_tampering():
    source = json.loads((ROOT / validator.TB1_REL / "validation_comparison.json").read_text(encoding="utf-8"))
    metrics = next(item for item in source["candidates"] if item["profile_id"] == validator.P1_PROFILE)["validation_metrics"]
    summaries = [{"seed": seed, "validation_metrics": copy.deepcopy(metrics)} for seed in validator.SEEDS]
    aggregate = {
        "phase": validator.PHASE,
        "seeds": list(validator.SEEDS),
        "seed_count": 3,
        "metric": "VALIDATION",
        "real_used_for_selection": False,
        "macro_f1": {"mean": metrics["macro_f1"], "std": 0.0, "minimum": metrics["macro_f1"], "maximum": metrics["macro_f1"], "range": 0.0},
        "balanced_accuracy": {"mean": metrics["balanced_accuracy"], "worst": metrics["balanced_accuracy"]},
        "human_fall_posture_proxy_recall": {"mean": metrics["h_fall_posture_proxy_recall"], "worst": metrics["h_fall_posture_proxy_recall"]},
        "stability_threshold_predefined": False,
        "best_seed_cherry_picking": "PROHIBITED",
    }
    errors: list[dict[str, str]] = []
    validator._validate_aggregate(aggregate, summaries, errors)
    assert not errors
    aggregate["macro_f1"]["mean"] = 0.0
    errors = []
    validator._validate_aggregate(aggregate, summaries, errors)
    assert any(item["code"] == "AGGREGATE_RECOMPUTE_MISMATCH" for item in errors)


def test_limitations_retain_proxy_gap_and_grouping_warnings():
    document = {
        "near_duplicate_pairs_train_validation": validator.EXPECTED_NEAR_DUPLICATES,
        "locked_test_available": False,
        "human_fall_semantics": "LYING_TO_HUMAN_FALL_DERIVED_POSTURE_PROXY_NOT_TEMPORAL_EVENT_GROUND_TRUTH",
        "subject_independent": "NOT_VERIFIABLE", "session_independent": "NOT_VERIFIABLE", "event_independent": "NOT_VERIFIABLE",
        "temporal_fall": "NOT_VERIFIED", "synthetic_real_gap": validator.EXPECTED_VAL_F1 - validator.EXPECTED_REAL_F1,
        "multi_seed_real_evaluation": "NOT_PERFORMED", "best_seed_cherry_picking": "PROHIBITED", "next_phase_started": False,
        "device_domain_validation": "NOT_PERFORMED_DEFERRED_TO_T-C",
    }
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    validator._validate_limitations(document, errors, warnings)
    assert not errors
    assert {item["code"] for item in warnings} == {"GROUPING_NOT_VERIFIABLE", "NEAR_DUPLICATE_OVERLAP", "NO_PRISTINE_LOCKED_TEST"}


def test_limitations_reject_real_multiseed_or_temporal_claim():
    document = {
        "near_duplicate_pairs_train_validation": 0,
        "locked_test_available": True,
        "human_fall_semantics": "TEMPORAL_GROUND_TRUTH",
        "subject_independent": "PASS", "session_independent": "PASS", "event_independent": "PASS",
        "temporal_fall": "VERIFIED", "synthetic_real_gap": 0.0,
        "multi_seed_real_evaluation": "PERFORMED", "best_seed_cherry_picking": "BEST", "next_phase_started": True,
        "device_domain_validation": "PASS",
    }
    errors: list[dict[str, str]] = []
    validator._validate_limitations(document, errors, [])
    assert len(errors) >= 8


def test_candidate_policy_rejects_best_seed_checkpoint():
    document = _json("candidate_checkpoint_policy.json")
    document["best_seed_cherry_picking"] = "SELECTED_BEST"
    errors = _errors(validator._validate_candidate_policy, document)
    assert any(item["code"] == "CHECKPOINT_POLICY_INVALID" for item in errors)


def test_portable_path_rejects_absolute_and_archive_paths():
    assert not validator._portable("/Users/junwoo/raw.npy")
    errors: list[dict[str, str]] = []
    # The compact evidence walker is exercised through an isolated temporary file.
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad.json"
        path.write_text(canonical_json({"path": "/Users/junwoo/raw.npy", "archive": "archive/old.json"}), encoding="utf-8")
        validator._read_documents(Path(directory), ["bad.json"], errors)
    assert {item["code"] for item in errors} >= {"NONPORTABLE_PATH", "ARCHIVE_TREATED_AS_ACTIVE"}


def test_execution_rejects_real_evaluation_and_t_b4():
    documents = {
        "execution_environment.json": {"phase": validator.PHASE, "temporal_training": False, "gpu_required": False},
        "execution_summary.json": {"phase": validator.PHASE, "status": "FINALIZED", "mode": "FULL_EXPERIMENT", "seed_count": 3, "seeds": list(validator.SEEDS), "real_evaluations": 1, "next_phase_started": False, "t_b4_started": True, "candidate_changed": False},
    }
    errors: list[dict[str, str]] = []
    validator._validate_execution(documents, errors)
    assert {item["code"] for item in errors} >= {"REAL_EVALUATION_PERFORMED", "SCOPE_ESCALATION"}


def test_live_predecessor_failure_is_not_masked(monkeypatch):
    monkeypatch.setattr(validator, "_validate_predecessors", lambda repo_root, errors: {"T-A6": {"evidence_validation": "FAIL"}})
    result = validator.validate_evidence(repo_root=ROOT, evidence_dir=EVIDENCE, mode="READINESS", check_checksums=True)
    assert result["evidence_validation"] == "FAIL"
    assert any(item["code"] == "PREDECESSOR_LIVE_INVALID" for item in result["errors"])


def test_final_evidence_passes_with_all_predecessors(monkeypatch):
    passing = {phase: {"evidence_validation": "PASS", "overall_outcome": "PASS_WITH_LIMITATIONS"} for phase in ("T-A6", "T-B0", "T-B1", "T-B2")}
    monkeypatch.setattr(validator, "_validate_predecessors", lambda repo_root, errors: passing)
    result = validator.validate_evidence(repo_root=ROOT, evidence_dir=EVIDENCE, mode="FULL_EXPERIMENT", check_checksums=True)
    assert result["evidence_validation"] == "PASS"
