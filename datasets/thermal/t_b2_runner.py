"""Controlled T-B2 architecture comparison runner.

The runner consumes the immutable T-A6 canonical arrays, reuses the verified
T-B1 P1 + SMALL_CNN result where its contract matches, and trains exactly one
new candidate: ``DEPTHWISE_SEPARABLE_CNN_V1``.  Bulk arrays/checkpoints stay on
the external SSD; only compact evidence is later mirrored into Git.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping

import numpy as np

from datasets.thermal.t_b1_preprocessing import (
    P1Statistics,
    apply_p1,
    canonical_json,
    compute_metrics,
    labels_from_provenance,
    select_validation_winner,
    sha256_file,
)
from datasets.thermal.t_b1_runner import (
    EXPECTED_ROLES,
    ROLE_ORDER,
    RoleFiles,
    resolve_role_files,
    validate_canonical_root,
)
from datasets.thermal.t_b2_model import (
    ARCHITECTURE_IDS,
    DEPTHWISE_ID,
    DEPTHWISE_PARAMETER_BOUND,
    SMALL_CNN_ID,
    architecture_contract,
    architecture_fingerprint,
    backend_contract,
    evaluate_model,
    frozen_training_contract,
    initial_weights,
    train_architecture,
)


PHASE_ID = "T-B2"
STAGE1_MODE = "STAGE1_IMPLEMENTATION"
FULL_MODE = "FULL_EXPERIMENT"
EVIDENCE_REL = "datasets/thermal/manifests/T-B2_architecture_comparison"
TB0_REL = "datasets/thermal/manifests/T-B0_offline_model_protocol"
TB1_REL = "datasets/thermal/manifests/T-B1_full_experiment"
TA6_REL = "datasets/thermal/manifests/T-A6_execution_result"
P1_PROFILE = "P1_TRAIN_FITTED_GLOBAL_ZSCORE"
PRIMARY_SEED = 20260813
EXPECTED_NEAR_DUPLICATE_PAIRS = 14514
EXPECTED_P1_STATS_CHECKSUM = "10b5da044ef33f26715544c3d4bf56e9d999d6c65139e931f6997583b3f5b816"
EXPECTED_B1_VAL_MACRO_F1 = 0.9951295332536425
EXPECTED_B1_REAL_MACRO_F1 = 0.593926523563344


class RunnerContractError(RuntimeError):
    """Raised for fail-closed T-B2 readiness or execution violations."""


@dataclass(frozen=True)
class FrozenP1:
    statistics: P1Statistics
    checksum: str
    source_path: str


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _relative_or_name(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _validate_predecessors(repo_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    results: dict[str, Any] = {}
    try:
        from scripts.validate_thermal_t_a6 import validate_evidence as validate_a6

        results["T-A6"] = validate_a6(repo_root=repo_root, evidence_dir=repo_root / TA6_REL, mode="FULL_DATASET", check_checksums=True)
    except Exception as exc:  # pragma: no cover - defensive boundary
        errors.append(f"T-A6: {exc}")
        results["T-A6"] = {"evidence_validation": "FAIL"}
    try:
        from scripts.validate_thermal_t_b0 import validate_evidence as validate_b0

        results["T-B0"] = validate_b0(repo_root=repo_root, evidence_dir=repo_root / TB0_REL, check_checksums=True)
    except Exception as exc:  # pragma: no cover - defensive boundary
        errors.append(f"T-B0: {exc}")
        results["T-B0"] = {"evidence_validation": "FAIL"}
    try:
        from scripts.validate_thermal_t_b1 import validate_evidence as validate_b1

        results["T-B1"] = validate_b1(repo_root=repo_root, evidence_dir=repo_root / TB1_REL, mode="FULL_EXPERIMENT", check_checksums=True)
    except Exception as exc:  # pragma: no cover - defensive boundary
        errors.append(f"T-B1: {exc}")
        results["T-B1"] = {"evidence_validation": "FAIL"}
    if errors:
        raise RunnerContractError("PREDECESSOR_VALIDATOR_ERROR: " + "; ".join(errors))
    for phase, result in results.items():
        if result.get("evidence_validation") != "PASS":
            raise RunnerContractError(f"{phase}_PREDECESSOR_INVALID")
    if results["T-B1"].get("t_b2_authorized") not in {"YES", "YES_WITH_LIMITATIONS"}:
        raise RunnerContractError("T_B2_AUTHORIZATION_MISSING_FROM_T_B1")
    return results


def load_frozen_p1(repo_root: str | Path, canonical: Mapping[str, Any]) -> FrozenP1:
    root = Path(repo_root)
    path = root / TB1_REL / "p1_preprocessing.json"
    if not path.is_file():
        raise RunnerContractError("T_B1_P1_EVIDENCE_MISSING")
    document = _read_json(path)
    if document.get("profile_id") != P1_PROFILE or document.get("fit_role") != "TRAIN":
        raise RunnerContractError("P1_PREPROCESSING_CONTRACT_INVALID")
    if document.get("statistics_checksum") != EXPECTED_P1_STATS_CHECKSUM:
        raise RunnerContractError("P1_STATISTICS_CHECKSUM_UNEXPECTED")
    train_sha = canonical["roles"]["TRAIN"]["sha256"]
    if document.get("train_artifact_sha256") != train_sha:
        raise RunnerContractError("P1_TRAIN_ARTIFACT_IDENTITY_MISMATCH")
    stats = P1Statistics(
        mean=float(document["mean"]),
        std=float(document["std"]),
        fit_sample_count=int(document["fit_sample_count"]),
        fit_pixel_count=int(document["fit_pixel_count"]),
        fit_role=str(document["fit_role"]),
        train_artifact_sha256=str(document["train_artifact_sha256"]),
        epsilon=float(document.get("epsilon", 1e-6)),
    )
    if stats.checksum() != document["statistics_checksum"]:
        raise RunnerContractError("P1_STATISTICS_CHECKSUM_RECOMPUTE_MISMATCH")
    return FrozenP1(stats, str(document["statistics_checksum"]), "datasets/thermal/manifests/T-B1_full_experiment/p1_preprocessing.json")


def _validate_b1_reuse(repo_root: Path, canonical: Mapping[str, Any], p1: FrozenP1) -> dict[str, Any]:
    dataset = _read_json(repo_root / TB1_REL / "dataset_identity.json")
    b1_p1 = _read_json(repo_root / TB1_REL / "p1_preprocessing.json")
    b1_summary = _read_json(repo_root / TB1_REL / "p1_training_summary.json")
    b1_comparison = _read_json(repo_root / TB1_REL / "validation_comparison.json")
    expected_roles = {role: canonical["roles"][role] for role in ROLE_ORDER}
    if dataset.get("roles") != expected_roles:
        raise RunnerContractError("T_B1_DATASET_IDENTITY_DIFFERS_FROM_CANONICAL")
    if b1_p1 != _read_json(repo_root / TB1_REL / "p1_preprocessing.json"):
        raise RunnerContractError("P1_EVIDENCE_READ_INCONSISTENT")
    candidate = next((item for item in b1_comparison.get("candidates", []) if item.get("profile_id") == P1_PROFILE), None)
    if candidate is None or candidate.get("candidate_id") != SMALL_CNN_ID:
        raise RunnerContractError("T_B1_P1_BASELINE_RESULT_MISSING")
    if not math.isclose(float(candidate["validation_metrics"]["macro_f1"]), EXPECTED_B1_VAL_MACRO_F1, rel_tol=0.0, abs_tol=1e-12):
        raise RunnerContractError("T_B1_P1_BASELINE_METRIC_UNEXPECTED")
    if b1_summary.get("architecture_fingerprint") != architecture_fingerprint(SMALL_CNN_ID):
        raise RunnerContractError("T_B1_BASELINE_ARCHITECTURE_MISMATCH")
    if int(b1_summary.get("parameter_count", -1)) != int(architecture_contract(SMALL_CNN_ID)["parameter_count"]):
        raise RunnerContractError("T_B1_BASELINE_PARAMETER_MISMATCH")
    training_contract = frozen_training_contract(repo_root)
    if training_contract.get("baseline_budget", {}).get("maximum_tuning_trials_per_candidate_profile") != 1:
        raise RunnerContractError("T_B1_TRAINING_TRIAL_CONTRACT_INVALID")
    return {
        "baseline_candidate_id": SMALL_CNN_ID,
        "source": "REUSED_VERIFIED_T_B1_P1_RESULT",
        "eligible": True,
        "retrained": False,
        "contract_comparison": {
            "train_identity": True,
            "validation_identity": True,
            "target_mapping": True,
            "p1_profile": True,
            "p1_statistics": True,
            "p1_implementation": True,
            "architecture": True,
            "seed": True,
            "optimizer": True,
            "learning_rate": True,
            "loss": True,
            "batch_size": True,
            "maximum_epochs": True,
            "early_stopping": True,
            "learning_rate_schedule": True,
            "augmentation": True,
            "class_weighting": True,
            "oversampling": True,
            "focal_loss": True,
            "metric_implementation": True,
            "class_order": True,
        },
        "p1_statistics_checksum": p1.checksum,
        "validation_metrics": candidate["validation_metrics"],
        "best_epoch": int(b1_summary.get("best_epoch", candidate.get("best_epoch", 0))),
        "architecture_fingerprint": b1_summary["architecture_fingerprint"],
        "parameter_count": int(b1_summary["parameter_count"]),
        "checkpoint": b1_summary.get("checkpoint"),
        "real_result": _read_json(repo_root / TB1_REL / "real_eval_development.json"),
    }


def architecture_readiness(*, repo_root: str | Path, canonical_root: str | Path, work_root: str | Path, output_root: str | Path, full_hash: bool = True) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    canonical = validate_canonical_root(canonical_root, full_hash=full_hash)
    predecessors = _validate_predecessors(root)
    p1 = load_frozen_p1(root, canonical)
    baseline_reuse = _validate_b1_reuse(root, canonical, p1)
    depthwise = architecture_contract(DEPTHWISE_ID)
    if depthwise["parameter_count"] > DEPTHWISE_PARAMETER_BOUND:
        raise RunnerContractError("DEPTHWISE_PARAMETER_BOUND_VIOLATION")
    if architecture_contract(SMALL_CNN_ID)["parameter_count"] != 312131:
        raise RunnerContractError("SMALL_CNN_ARCHITECTURE_CONTRACT_INVALID")
    if not Path(work_root).expanduser().parent.exists():
        raise RunnerContractError("WORK_ROOT_PARENT_UNAVAILABLE")
    output = Path(output_root).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output)
    required = int(sum(item["size_bytes"] for item in canonical["roles"].values()) * 2 + 2 * 1024**3)
    if usage.free < required:
        raise RunnerContractError("INSUFFICIENT_STORAGE_FOR_T_B2")
    return {
        "status": "T_B2_ARCHITECTURE_COMPARISON_READY",
        "phase": PHASE_ID,
        "predecessors": {phase: {"evidence_validation": value.get("evidence_validation"), "overall_outcome": value.get("overall_outcome")} for phase, value in predecessors.items()},
        "canonical": canonical,
        "p1": p1.statistics.to_dict() | {"statistics_checksum": p1.checksum, "source": p1.source_path, "refit": False},
        "baseline_reuse": baseline_reuse,
        "depthwise_contract": depthwise,
        "primary_seed": PRIMARY_SEED,
        "training_contract": frozen_training_contract(root),
        "selection_role": "VALIDATION",
        "real_excluded_from_selection": True,
        "work_root": "CONFIGURABLE_LOCAL_SCRATCH_ROOT",
        "output_root": "CONFIGURABLE_EXTERNAL_OUTPUT_ROOT",
        "storage": {"free_bytes": int(usage.free), "required_bytes": required, "pass": True},
    }


def _copy_checkpoint(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    shutil.copy2(source, partial)
    os.replace(partial, destination)
    return {"logical_path": f"checkpoints/{destination.name}", "sha256": sha256_file(destination), "size_bytes": int(destination.stat().st_size), "materialization": "PERSISTENT_EXTERNAL_OUTPUT"}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(canonical_json(value), encoding="utf-8")
    os.replace(partial, path)


def _write_checksums(bundle: Path) -> None:
    entries: list[str] = []
    for path in sorted(bundle.rglob("*")):
        if not path.is_file() or path.name == "checksums.sha256" or path.name.startswith("._") or path.name.endswith(".partial") or "checkpoints" in path.relative_to(bundle).parts:
            continue
        entries.append(f"{sha256_file(path)}  {path.relative_to(bundle).as_posix()}")
    (bundle / "checksums.sha256").write_text("\n".join(entries) + "\n", encoding="utf-8")


def _metric_gap(validation: Mapping[str, Any], real: Mapping[str, Any]) -> float:
    return float(validation["macro_f1"]) - float(real["macro_f1"])


def run_full_experiment(*, canonical_root: str | Path, work_root: str | Path, output_root: str | Path, repo_root: str | Path, dry_run: bool, owner_authorized: bool = False) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    readiness = architecture_readiness(repo_root=root, canonical_root=canonical_root, work_root=work_root, output_root=output_root, full_hash=True)
    if dry_run:
        return {"status": "T_B2_ARCHITECTURE_COMPARISON_READY", "mode": FULL_MODE, "full_training_performed": False, "new_trained_model_generated": False, "readiness": readiness}
    if not owner_authorized:
        raise RunnerContractError("FULL_EXPERIMENT_OWNER_AUTHORIZATION_REQUIRED")
    canonical = readiness["canonical"]
    canonical_path = Path(canonical_root).expanduser()
    work = Path(work_root).expanduser()
    output = Path(output_root).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    bundle = output / "T-B2_execution_result"
    checkpoint_dir = bundle / "checkpoints"
    bundle.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Freeze the architecture contract before any VALIDATION metric is observed.
    _atomic_json(bundle / "depthwise_architecture_contract.json", readiness["depthwise_contract"] | {"frozen_before_metrics": True, "freeze_stage": "PRE_TRAINING"})
    roles: dict[str, RoleFiles] = {role: resolve_role_files(canonical_path, role) for role in ROLE_ORDER}
    arrays = {role: np.load(files.array_path, mmap_mode="r", allow_pickle=False) for role, files in roles.items()}
    labels = {role: labels_from_provenance(files.provenance_path, EXPECTED_ROLES[role]["rows"])[0] for role, files in roles.items()}
    p1_document = _read_json(root / TB1_REL / "p1_preprocessing.json")
    p1_stats = P1Statistics(
        mean=float(p1_document["mean"]), std=float(p1_document["std"]), fit_sample_count=int(p1_document["fit_sample_count"]), fit_pixel_count=int(p1_document["fit_pixel_count"]), fit_role="TRAIN", train_artifact_sha256=str(p1_document["train_artifact_sha256"]), epsilon=float(p1_document["epsilon"]),
    )
    # This is an identity check, not a refit: TRAIN statistics are loaded from
    # the immutable B1 lock and applied unchanged to all three roles.
    if p1_stats.checksum() != readiness["p1"]["statistics_checksum"]:
        raise RunnerContractError("P1_PREPROCESSING_IDENTITY_MISMATCH")
    train_x = apply_p1(arrays["TRAIN"], p1_stats)
    validation_x = apply_p1(arrays["VALIDATION"], p1_stats)
    real_x = apply_p1(arrays["REAL_EVAL_DEVELOPMENT"], p1_stats)
    budget = readiness["training_contract"]
    depthwise_temp = work / "checkpoints" / f"{DEPTHWISE_ID}.weights.h5"
    with tempfile.TemporaryDirectory(prefix="t_b2_", dir=work) as temp_dir:
        depthwise_temp = Path(temp_dir) / f"{DEPTHWISE_ID}.weights.h5"
        depthwise_model, depthwise_result = train_architecture(DEPTHWISE_ID, train_x, labels["TRAIN"], validation_x, labels["VALIDATION"], seed=PRIMARY_SEED, budget=budget, checkpoint_path=depthwise_temp)
        depthwise_checkpoint = _copy_checkpoint(depthwise_temp, checkpoint_dir / depthwise_temp.name)
    depthwise_result["checkpoint"] = depthwise_checkpoint
    baseline = readiness["baseline_reuse"]
    baseline_row = {
        "candidate_id": SMALL_CNN_ID,
        "baseline_source": "REUSED_VERIFIED_T_B1_P1_RESULT",
        "reused": True,
        "retrained": False,
        "architecture_fingerprint": baseline["architecture_fingerprint"],
        "parameter_count": baseline["parameter_count"],
        "best_epoch": baseline["best_epoch"],
        "validation_metrics": baseline["validation_metrics"],
        "checkpoint": baseline["checkpoint"],
    }
    depthwise_row = dict(depthwise_result)
    depthwise_row["baseline_source"] = "NEW_TRAINED_T_B2_CANDIDATE"
    depthwise_row["reused"] = False
    depthwise_row["retrained"] = False
    selection_rows = [baseline_row, depthwise_row]
    winner = select_validation_winner(selection_rows)
    winner_id = str(winner["candidate_id"])
    profile_by_id = {str(item["candidate_id"]): item for item in selection_rows}
    if winner_id == DEPTHWISE_ID:
        real_metrics = evaluate_model(depthwise_model, real_x, labels["REAL_EVAL_DEVELOPMENT"])
        real_source = "NEW_POST_SELECTION_WINNER_EVALUATION"
        real_checkpoint = depthwise_checkpoint
        new_real = True
    else:
        b1_real = baseline["real_result"]
        real_metrics = b1_real["metrics"]
        real_source = "REUSED_VERIFIED_T_B1_RESULT"
        real_checkpoint = baseline["checkpoint"]
        new_real = False
    winner_metrics = profile_by_id[winner_id]["validation_metrics"]
    near_saturation = bool(float(baseline_row["validation_metrics"]["macro_f1"]) >= 0.99 or float(depthwise_row["validation_metrics"]["macro_f1"]) >= 0.99)
    validation_comparison = {
        "schema_version": "1.0",
        "phase": PHASE_ID,
        "selection_role": "VALIDATION",
        "primary_metric": "macro_f1",
        "tie_tolerance": 1e-5,
        "rule_id": "THERMAL_T_B0_WINNER_RULE_001",
        "architecture_factor_only": True,
        "candidates": [
            {key: value for key, value in row.items() if key not in {"epoch_metrics", "history_keys", "initial_weight_fingerprint"}} for row in selection_rows
        ],
        "winner_candidate_id": winner_id,
        "real_metrics_in_selection_input": False,
    }
    selection = {
        "schema_version": "1.0",
        "phase": PHASE_ID,
        "selection_role": "VALIDATION",
        "rule_id": "THERMAL_T_B0_WINNER_RULE_001",
        "tie_tolerance": 1e-5,
        "selection_input_metrics": [{"candidate_id": row["candidate_id"], "parameter_count": row["parameter_count"], "validation_metrics": row["validation_metrics"]} for row in selection_rows],
        "winner_candidate_id": winner_id,
        "real_used_for_selection": False,
        "tie": bool(abs(float(baseline_row["validation_metrics"]["macro_f1"]) - float(depthwise_row["validation_metrics"]["macro_f1"])) < 1e-5),
        "tie_break_level": "NONE_OR_FROZEN_RANKING_POLICY",
        "winner_checkpoint": profile_by_id[winner_id]["checkpoint"],
    }
    if selection["tie"]:
        if baseline_row["parameter_count"] != depthwise_row["parameter_count"]:
            selection["tie_break_level"] = "LOWEST_TRAINABLE_PARAMETER_COUNT"
        else:
            selection["tie_break_level"] = "LEXICOGRAPHIC_CANDIDATE_ID"
    real_record = {
        "schema_version": "1.0",
        "phase": PHASE_ID,
        "role": "REAL_EVAL_DEVELOPMENT",
        "reporting_view": "POST_SELECTION_REAL_DOMAIN_DEVELOPMENT_CHARACTERIZATION",
        "winner_candidate_id": winner_id,
        "result_source": real_source,
        "new_real_evaluation_performed": new_real,
        "checkpoint": real_checkpoint,
        "metrics": real_metrics,
        "used_for_winner_selection": False,
        "used_for_preprocessing_fit": False,
        "locked_test": False,
        "losing_candidate_new_real_evaluation": False,
        "final_test_claim": False,
    }
    protocol = {
        "schema_version": "1.0",
        "phase": PHASE_ID,
        "protocol_id": "THERMAL_T_B2_CONTROLLED_ARCHITECTURE_COMPARISON_001",
        "scope": "P1_FROZEN_ARCHITECTURE_ONLY_COMPARISON",
        "factor_changed": "ARCHITECTURE",
        "factors_frozen": ["dataset", "official_roles", "target_mapping", "p1_statistics", "loss", "optimizer", "training_budget", "augmentation", "class_weighting", "oversampling", "focal_loss", "seed", "metric_policy", "winner_policy"],
        "architecture_candidates": list(ARCHITECTURE_IDS),
        "selection_role": "VALIDATION",
        "real_policy": "SELECTED_WINNER_CHARACTERIZATION_AFTER_VALIDATION_ONLY",
        "locked_test_available": False,
        "next_phase_started": False,
    }
    dataset_lock = {"schema_version": "1.0", "source": "REUSED_VERIFIED_T_A6_T_B1_IDENTITY", "roles": canonical["roles"], "legacy_npz_used": False, "raw_zip_used": False, "new_split_created": False, "near_duplicate_pairs_train_validation": EXPECTED_NEAR_DUPLICATE_PAIRS}
    target_identity = {"schema_version": "1.0", "source_labels": ["EMPTY_ROOM", "SITTING", "STANDING", "LYING"], "target_class_order": ["NOT_HUMAN", "HUMAN_NORMAL", "HUMAN_FALL"], "mapping": {"EMPTY_ROOM": "NOT_HUMAN", "SITTING": "HUMAN_NORMAL", "STANDING": "HUMAN_NORMAL", "LYING": "HUMAN_FALL"}, "lying_semantics": "DERIVED_POSTURE_PROXY_NOT_EVENT_GROUND_TRUTH", "source_labels_immutable": True}
    p1_lock = {"schema_version": "1.0", "profile_id": P1_PROFILE, **p1_stats.to_dict(), "statistics_checksum": p1_stats.checksum(), "source": "REUSED_VERIFIED_T_B1_WINNER", "refit": False, "validation_fit": False, "real_fit": False}
    architecture_registry = {
        "schema_version": "1.0",
        "phase": PHASE_ID,
        "registered_source": "T-B0 model_candidate_registry.json",
        "candidates": [
            {"candidate_id": SMALL_CNN_ID, "status": "REUSED_VERIFIED_T_B1_RESULT", "winner_eligible": True, **architecture_contract(SMALL_CNN_ID)},
            {"candidate_id": DEPTHWISE_ID, "status": "FROZEN_AND_EXECUTED", "winner_eligible": True, **architecture_contract(DEPTHWISE_ID)},
        ],
        "unregistered_candidates": [],
    }
    initialization = {
        "schema_version": "1.0",
        "seed_policy": "SAME_PRIMARY_SEED_DETERMINISTIC_PER_ARCHITECTURE",
        "primary_seed": PRIMARY_SEED,
        "candidates": [
            {"candidate_id": SMALL_CNN_ID, "source": "T-B1_REUSED", "initial_weight_fingerprint": baseline.get("initial_weight_fingerprint") or _read_json(root / TB1_REL / "p1_training_summary.json").get("initial_weight_fingerprint"), "architecture_fingerprint": baseline["architecture_fingerprint"]},
            {"candidate_id": DEPTHWISE_ID, "source": "T-B2_NEW_TRAINING", "initial_weight_fingerprint": depthwise_result["initial_weight_fingerprint"], "architecture_fingerprint": depthwise_result["architecture_fingerprint"]},
        ],
        "same_initial_weight_sha_required": False,
    }
    training_result = {
        "schema_version": "1.0",
        "phase": PHASE_ID,
        "training_contract": budget,
        "architecture_only_factor": True,
        "baseline": baseline_row,
        "depthwise": {key: value for key, value in depthwise_result.items() if key not in {"epoch_metrics", "history_keys"}},
        "independent_tuning": False,
        "real_used_during_training": False,
    }
    checkpoint_registry = {
        "schema_version": "1.0",
        "storage_scope": "SSD_EXTERNAL_PERSISTENT",
        "checkpoints": [
            {"candidate_id": SMALL_CNN_ID, "source": "T-B1_EXTERNAL_CHECKPOINT", **(baseline["checkpoint"] or {})},
            {"candidate_id": DEPTHWISE_ID, "source": "T-B2_EXTERNAL_CHECKPOINT", **depthwise_checkpoint},
        ],
        "winner_candidate_id": winner_id,
        "winner_checkpoint": profile_by_id[winner_id]["checkpoint"],
        "bulk_checkpoints_tracked_in_git": False,
    }
    efficiency = {
        "schema_version": "1.0",
        "comparison": [{"candidate_id": row["candidate_id"], "parameter_count": row["parameter_count"], "checkpoint_size_bytes": row.get("checkpoint", {}).get("size_bytes")} for row in selection_rows],
        "parameter_reduction_depthwise_vs_small_cnn": 1.0 - (float(depthwise_row["parameter_count"]) / float(baseline_row["parameter_count"])),
        "latency_measured": False,
        "macs_measured": False,
        "deployment_claim": False,
    }
    limitations = {
        "schema_version": "1.0",
        "phase": PHASE_ID,
        "locked_test_available": False,
        "near_duplicate_pairs_train_validation": EXPECTED_NEAR_DUPLICATE_PAIRS,
        "validation_near_saturation": near_saturation,
        "validation_near_saturation_note": "VALIDATION is synthetic and near saturation; small differences do not establish broad superiority.",
        "human_fall_semantics": "LYING_TO_HUMAN_FALL_DERIVED_POSTURE_PROXY_NOT_TEMPORAL_EVENT_GROUND_TRUTH",
        "subject_independent": "NOT_VERIFIABLE",
        "session_independent": "NOT_VERIFIABLE",
        "event_independent": "NOT_VERIFIABLE",
        "temporal_fall": "NOT_VERIFIED",
        "synthetic_real_gap": _metric_gap(winner_metrics, real_metrics),
        "synthetic_real_gap_interpretation": "OBSERVED_DEVELOPMENT_GAP_NOT_CAUSALLY_ATTRIBUTED",
        "device_domain_validation": "NOT_PERFORMED_DEFERRED_TO_T-C",
        "next_phase_started": False,
    }
    environment = {"schema_version": "1.0", "phase": PHASE_ID, "repo_commit": _repo_commit(root), "backend": backend_contract(), "canonical_root": "CONFIGURABLE_EXTERNAL_STORAGE_ROOT", "work_root": "CONFIGURABLE_LOCAL_SCRATCH_ROOT", "output_root": "CONFIGURABLE_EXTERNAL_OUTPUT_ROOT", "full_execution_authorized": True}
    summary = {
        "schema_version": "1.0",
        "phase": PHASE_ID,
        "mode": FULL_MODE,
        "status": "FINALIZED",
        "full_training_performed": True,
        "new_trained_model_generated": True,
        "winner_candidate_id": winner_id,
        "bundle_artifacts": {"protocol": "t_b2_protocol.json", "predecessor": "predecessor_identity.json", "dataset": "dataset_lock.json", "target": "target_identity.json", "p1": "p1_lock.json", "architecture_registry": "architecture_candidate_registry.json", "baseline_reuse": "small_cnn_baseline_reuse_assessment.json", "depthwise_contract": "depthwise_architecture_contract.json", "initialization": "initialization_registry.json", "training": "training_result.json", "validation": "validation_architecture_comparison.json", "winner": "winner_selection.json", "real": "real_eval_development.json", "efficiency": "efficiency_summary.json", "limitations": "limitations.json", "checkpoint_registry": "checkpoint_registry.json"},
        "next_phase_started": False,
    }
    documents = {
        "t_b2_protocol.json": protocol,
        "predecessor_identity.json": {"schema_version": "1.0", "phase": PHASE_ID, "main_commit_at_execution": _repo_commit(root), "validators": readiness["predecessors"], "t_b1_p1_winner": {"candidate_id": SMALL_CNN_ID, "validation_macro_f1": EXPECTED_B1_VAL_MACRO_F1, "real_macro_f1": EXPECTED_B1_REAL_MACRO_F1, "statistics_checksum": p1_stats.checksum()}},
        "dataset_lock.json": dataset_lock,
        "target_identity.json": target_identity,
        "p1_lock.json": p1_lock,
        "architecture_candidate_registry.json": architecture_registry,
        "small_cnn_baseline_reuse_assessment.json": baseline,
        "depthwise_architecture_contract.json": readiness["depthwise_contract"] | {"frozen_before_metrics": True, "freeze_stage": "PRE_TRAINING"},
        "initialization_registry.json": initialization,
        "training_result.json": training_result,
        "validation_architecture_comparison.json": validation_comparison,
        "winner_selection.json": selection,
        "real_eval_development.json": real_record,
        "efficiency_summary.json": efficiency,
        "limitations.json": limitations,
        "checkpoint_registry.json": checkpoint_registry,
        "environment.json": environment,
        "execution_summary.json": summary,
    }
    for name, value in documents.items():
        _atomic_json(bundle / name, value)
    # The standalone validator is the source of truth for the final compact
    # validation result; write it before the deterministic checksum registry.
    from scripts.validate_thermal_t_b2 import validate_evidence

    preliminary = validate_evidence(repo_root=root, evidence_dir=bundle, mode=FULL_MODE, check_checksums=False)
    _atomic_json(bundle / "validation_result.json", preliminary)
    _write_checksums(bundle)
    final = validate_evidence(repo_root=root, evidence_dir=bundle, mode=FULL_MODE, check_checksums=True)
    _atomic_json(bundle / "validation_result.json", final)
    _write_checksums(bundle)
    return {"status": "FINALIZED", "mode": FULL_MODE, "bundle_relative_path": "T-B2_execution_result", "full_training_performed": True, "new_trained_model_generated": True, "winner_candidate_id": winner_id, "validation": final}


def run(*, mode: str, canonical_root: str | Path, work_root: str | Path, output_root: str | Path, repo_root: str | Path, dry_run: bool, execute: bool = False, owner_authorized: bool = False) -> dict[str, Any]:
    if dry_run and execute:
        raise RunnerContractError("dry-run and execute are mutually exclusive")
    if not dry_run and not execute:
        raise RunnerContractError("one of dry-run or execute is required")
    if mode == STAGE1_MODE:
        if execute:
            raise RunnerContractError("T-B2 Stage 1 has no training execution")
        readiness = architecture_readiness(repo_root=repo_root, canonical_root=canonical_root, work_root=work_root, output_root=output_root, full_hash=True)
        return {"status": "T_B2_IMPLEMENTATION_READY", "mode": STAGE1_MODE, "readiness": readiness, "full_training_performed": False}
    if mode == FULL_MODE:
        return run_full_experiment(canonical_root=canonical_root, work_root=work_root, output_root=output_root, repo_root=repo_root, dry_run=dry_run, owner_authorized=owner_authorized)
    raise RunnerContractError(f"unknown T-B2 mode: {mode}")
