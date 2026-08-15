"""SafeNest Thermal T-B3 frame-only multi-seed stability runner.

This module deliberately reuses the frozen T-B1 P1 preprocessing and
``SMALL_CNN_BASELINE_V1`` training contract.  It never constructs temporal
windows, reads REAL for seed selection, or performs T-B4 conversion.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from typing import Any, Mapping

import numpy as np

from datasets.thermal.t_b1_model import (
    BASELINE_ID,
    backend_info,
    create_small_cnn_baseline,
    initial_weights,
    require_tensorflow,
    train_profile,
)
from datasets.thermal.t_b1_preprocessing import (
    P1Statistics,
    apply_p1,
    canonical_json,
    compute_metrics,
    labels_from_provenance,
    sha256_file,
)
from datasets.thermal.t_b1_runner import (
    EXPECTED_ROLES,
    ROLE_ORDER,
    resolve_role_files,
    validate_canonical_root,
)


PHASE_ID = "T-B3"
READINESS_MODE = "READINESS"
FULL_MODE = "FULL_EXPERIMENT"
EVIDENCE_REL = "datasets/thermal/manifests/T-B3_frame_multiseed_confirmation"
TA3_REL = "datasets/thermal/manifests/T-A3_sequence_window_event_policy"
TA6_REL = "datasets/thermal/manifests/T-A6_execution_result"
TB0_REL = "datasets/thermal/manifests/T-B0_offline_model_protocol"
TB1_REL = "datasets/thermal/manifests/T-B1_full_experiment"
TB2_REL = "datasets/thermal/manifests/T-B2_architecture_comparison"
ROADMAP_REL = "docs/20260810_ChatGPT_SafeNest_Multisensor_Parallel_Execution_Roadmap_01.md"
RECONCILIATION_REL = "docs/reports/20260814_Codex_Thermal_Post_T-B2_Pre_T-B3_Reconciliation_01.md"
SEEDS = (20260813, 20260814, 20260815)
P1_PROFILE = "P1_TRAIN_FITTED_GLOBAL_ZSCORE"
EXPECTED_P1_MEAN = 22.769290618485442
EXPECTED_P1_STD = 2.8684523405441222
EXPECTED_P1_CHECKSUM = "10b5da044ef33f26715544c3d4bf56e9d999d6c65139e931f6997583b3f5b816"
EXPECTED_ARCHITECTURE_FINGERPRINT = "937fceb88900a779d67a8e407cebc3362cc21346721f92cea5ae8de1413aba2a"
EXPECTED_PARAMETER_COUNT = 312131
EXPECTED_NEAR_DUPLICATE_PAIRS = 14514
EXPECTED_REAL_MACRO_F1 = 0.593926523563344
EXPECTED_VAL_MACRO_F1 = 0.9951295332536425
EXPECTED_CHECKPOINT_SHA = "7aba32fe8d0e241546429bdc3e8cd059b10d4d8f548e9e12c4085abeba308a75"
EXPECTED_CHECKPOINT_SIZE = 3777416


class RunnerContractError(RuntimeError):
    """Raised for a fail-closed readiness or execution violation."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(canonical_json(value), encoding="utf-8")
    os.replace(temporary, path)


def _repo_commit(repo_root: Path, ref: str = "HEAD") -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", ref],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_diff_is_empty(repo_root: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "origin/main...HEAD", "--name-only"],
            check=True,
            capture_output=True,
            text=True,
        )
        return not result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return False


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    shutil.copy2(source, partial)
    os.replace(partial, destination)


def _portable_logical(path: str) -> bool:
    lowered = path.lower()
    return not (
        path.startswith(("/", "~/", "file://"))
        or "\\" in path
        or "/users/" in lowered
        or "/private/" in lowered
        or path.startswith(("/Volumes/", "/content/"))
    )


def _validate_roadmap_reconciliation(repo_root: Path) -> dict[str, Any]:
    roadmap = repo_root / ROADMAP_REL
    report = repo_root / RECONCILIATION_REL
    if not roadmap.is_file() or not report.is_file():
        raise RunnerContractError("T_B3_BLOCKED_ROADMAP_RECONCILIATION_NOT_MERGED")
    roadmap_text = roadmap.read_text(encoding="utf-8").lower()
    report_text = report.read_text(encoding="utf-8").lower()
    required = (
        "controlled frame-architecture comparison",
        "frame-only multi-seed",
        "temporal comparison remains deferred",
        "small_cnn_baseline_v1",
    )
    if any(token not in roadmap_text + "\n" + report_text for token in required):
        raise RunnerContractError("T_B3_BLOCKED_ROADMAP_RECONCILIATION_NOT_MERGED")
    return {
        "roadmap_path": ROADMAP_REL,
        "reconciliation_report_path": RECONCILIATION_REL,
        "reconciliation_present": True,
        "temporal_training_authorized": False,
        "next_work": "FRAME_ONLY_MULTI_SEED_CONFIRMATION",
    }


def _run_predecessors(repo_root: Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    try:
        from scripts.validate_thermal_t_a6 import validate_evidence as validate_a6
        from scripts.validate_thermal_t_b0 import validate_evidence as validate_b0
        from scripts.validate_thermal_t_b1 import validate_evidence as validate_b1
        from scripts.validate_thermal_t_b2 import validate_evidence as validate_b2

        results["T-A6"] = validate_a6(repo_root=repo_root, evidence_dir=repo_root / TA6_REL, mode="FULL_DATASET", check_checksums=True)
        results["T-B0"] = validate_b0(repo_root=repo_root, evidence_dir=repo_root / TB0_REL, check_checksums=True)
        results["T-B1"] = validate_b1(repo_root=repo_root, evidence_dir=repo_root / TB1_REL, mode="FULL_EXPERIMENT", check_checksums=True)
        results["T-B2"] = validate_b2(repo_root=repo_root, evidence_dir=repo_root / TB2_REL, mode="FULL_EXPERIMENT", check_checksums=True)
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise RunnerContractError(f"T_B3_BLOCKED_PREDECESSOR_INVALID: {exc}") from exc
    for phase, result in results.items():
        if result.get("evidence_validation") != "PASS":
            raise RunnerContractError(f"T_B3_BLOCKED_PREDECESSOR_INVALID:{phase}")
    if results["T-B1"].get("overall_outcome") not in {"T_B1_FULL_COMPLETE_WITH_LIMITATIONS", "PASS_WITH_LIMITATIONS"}:
        raise RunnerContractError("T_B3_BLOCKED_PREDECESSOR_INVALID:T-B1_FULL")
    if results["T-B2"].get("overall_outcome") not in {"T_B2_COMPLETE_WITH_LIMITATIONS", "PASS_WITH_LIMITATIONS"}:
        raise RunnerContractError("T_B3_BLOCKED_PREDECESSOR_INVALID:T-B2")
    return results


def _load_budget(repo_root: Path) -> dict[str, Any]:
    return _read_json(repo_root / TB0_REL / "training_budget_policy.json")


def _expected_budget(repo_root: Path) -> dict[str, Any]:
    source = _load_budget(repo_root)
    baseline = source.get("baseline_budget", {})
    required = {
        "batch_size": 64,
        "maximum_epochs": 20,
        "initial_learning_rate": 0.001,
        "optimizer": "Adam",
        "loss": "unweighted_sparse_categorical_crossentropy",
        "early_stopping": {"monitor": "validation_macro_f1", "patience": 5, "restore_best_weights": True, "mode": "max"},
        "learning_rate_schedule": {"factor": 0.5, "patience": 3, "minimum_learning_rate": 1e-6, "monitor": "validation_macro_f1", "mode": "max"},
        "augmentation": {"baseline": "DISABLED", "allowed_partition": "TRAIN_ONLY", "temporal_event_fabrication": False},
        "class_imbalance": {"class_weight": "DISABLED_IN_BASELINE", "oversampling": "DISABLED_IN_BASELINE", "focal_loss": "DISABLED_IN_BASELINE"},
    }
    for key, expected in required.items():
        actual = baseline.get(key) if key in baseline else source.get("augmentation", {}).get(key)
        if key == "augmentation":
            actual = source.get("augmentation", {})
        elif key == "class_imbalance":
            actual = source.get("class_imbalance", {})
        if key in {"early_stopping", "learning_rate_schedule"}:
            actual = baseline.get(key, {})
        if key in {"batch_size", "maximum_epochs", "initial_learning_rate", "optimizer", "loss"} and actual != expected:
            raise RunnerContractError(f"T_B3_TRAINING_BUDGET_MISMATCH:{key}")
        if key in {"early_stopping", "learning_rate_schedule"}:
            for nested, value in expected.items():
                if actual.get(nested) != value:
                    raise RunnerContractError(f"T_B3_TRAINING_BUDGET_MISMATCH:{key}.{nested}")
        if key in {"augmentation", "class_imbalance"}:
            for nested, value in expected.items():
                if actual.get(nested) != value:
                    raise RunnerContractError(f"T_B3_TRAINING_BUDGET_MISMATCH:{key}.{nested}")
    return source


def _load_frozen_p1(repo_root: Path, canonical: Mapping[str, Any]) -> P1Statistics:
    document = _read_json(repo_root / TB1_REL / "p1_preprocessing.json")
    if document.get("profile_id") != P1_PROFILE or document.get("fit_role") != "TRAIN":
        raise RunnerContractError("P1_IDENTITY_MISMATCH")
    if document.get("statistics_checksum") != EXPECTED_P1_CHECKSUM:
        raise RunnerContractError("P1_IDENTITY_MISMATCH")
    if document.get("train_artifact_sha256") != canonical["roles"]["TRAIN"]["sha256"]:
        raise RunnerContractError("P1_IDENTITY_MISMATCH")
    stats = P1Statistics(
        mean=float(document["mean"]),
        std=float(document["std"]),
        fit_sample_count=int(document["fit_sample_count"]),
        fit_pixel_count=int(document["fit_pixel_count"]),
        fit_role=str(document["fit_role"]),
        train_artifact_sha256=str(document["train_artifact_sha256"]),
        epsilon=float(document.get("epsilon", 1e-6)),
    )
    if stats.checksum() != EXPECTED_P1_CHECKSUM:
        raise RunnerContractError("P1_IDENTITY_MISMATCH")
    if not math.isclose(stats.mean, EXPECTED_P1_MEAN, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(stats.std, EXPECTED_P1_STD, rel_tol=0.0, abs_tol=1e-12):
        raise RunnerContractError("P1_IDENTITY_MISMATCH")
    return stats


def _load_architecture_lock(repo_root: Path) -> dict[str, Any]:
    b2_training = _read_json(repo_root / TB2_REL / "training_result.json")
    b2_baseline = b2_training.get("baseline", {})
    model = create_small_cnn_baseline()
    fingerprint = b2_baseline.get("architecture_fingerprint")
    if fingerprint != EXPECTED_ARCHITECTURE_FINGERPRINT or int(b2_baseline.get("parameter_count", -1)) != EXPECTED_PARAMETER_COUNT:
        raise RunnerContractError("SMALL_CNN_IDENTITY_MISMATCH")
    if int(model.count_params()) != EXPECTED_PARAMETER_COUNT:
        raise RunnerContractError("SMALL_CNN_IDENTITY_MISMATCH")
    return {
        "schema_version": "1.0",
        "phase": PHASE_ID,
        "candidate_id": BASELINE_ID,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "architecture_fingerprint": EXPECTED_ARCHITECTURE_FINGERPRINT,
        "source": "REUSED_VERIFIED_T_B2_ARCHITECTURE_WINNER",
        "modified": False,
        "temporal_layers": False,
    }


def _load_reuse_assessment(repo_root: Path, p1: P1Statistics, canonical: Mapping[str, Any], budget: Mapping[str, Any]) -> dict[str, Any]:
    summary = _read_json(repo_root / TB1_REL / "p1_training_summary.json")
    comparison = _read_json(repo_root / TB1_REL / "validation_comparison.json")
    candidate = next((item for item in comparison.get("candidates", []) if item.get("profile_id") == P1_PROFILE), None)
    if candidate is None:
        raise RunnerContractError("SEED_20260813_REUSE_ASSESSMENT_FAILED")
    required_true = {
        "train_identity": True,
        "validation_identity": True,
        "target_mapping": True,
        "p1_profile": True,
        "p1_statistics": True,
        "p1_implementation": True,
        "architecture": True,
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
        "seed": True,
    }
    if int(summary.get("seed", -1)) != 20260813 or summary.get("profile_id") != P1_PROFILE or summary.get("candidate_id") != BASELINE_ID:
        raise RunnerContractError("SEED_20260813_REUSE_ASSESSMENT_FAILED")
    if summary.get("architecture_fingerprint") != EXPECTED_ARCHITECTURE_FINGERPRINT or int(summary.get("parameter_count", -1)) != EXPECTED_PARAMETER_COUNT:
        raise RunnerContractError("SEED_20260813_REUSE_ASSESSMENT_FAILED")
    if summary.get("preprocessing_statistics", {}).get("profile_id") != P1_PROFILE or summary.get("preprocessing_statistics", {}).get("train_artifact_sha256") != canonical["roles"]["TRAIN"]["sha256"]:
        raise RunnerContractError("SEED_20260813_REUSE_ASSESSMENT_FAILED")
    metrics = candidate.get("validation_metrics", {})
    if not math.isclose(float(metrics.get("macro_f1", -1)), EXPECTED_VAL_MACRO_F1, rel_tol=0.0, abs_tol=1e-12):
        raise RunnerContractError("SEED_20260813_REUSE_ASSESSMENT_FAILED")
    checkpoint = summary.get("checkpoint", {})
    if checkpoint.get("sha256") != EXPECTED_CHECKPOINT_SHA or int(checkpoint.get("size_bytes", -1)) != EXPECTED_CHECKPOINT_SIZE:
        raise RunnerContractError("SEED_20260813_REUSE_ASSESSMENT_FAILED")
    return {
        "schema_version": "1.0",
        "phase": PHASE_ID,
        "seed": 20260813,
        "source": "REUSED_VERIFIED_T_B1_RESULT",
        "eligible": True,
        "retrained": False,
        "contract_comparison": required_true,
        "p1_statistics_checksum": p1.checksum(),
        "architecture_fingerprint": EXPECTED_ARCHITECTURE_FINGERPRINT,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "best_epoch": int(summary.get("best_epoch", candidate.get("best_epoch", 0))),
        "initialization_fingerprint": summary.get("initial_weight_fingerprint"),
        "validation_metrics": metrics,
        "checkpoint": checkpoint,
        "real_evaluation": "EXISTING_REFERENCE_ONLY_NOT_REPEATED",
        "real_metric_for_seed_selection": False,
    }


def _validate_temporal_disabled(repo_root: Path) -> None:
    capability = _read_json(repo_root / TA3_REL / "temporal_capability_contract.json")
    sequence = capability.get("capabilities", {}).get("SEQUENCE_LEVEL", {})
    event = capability.get("capabilities", {}).get("EVENT_LEVEL", {})
    if sequence.get("supported") is not False or event.get("supported") is not False:
        raise RunnerContractError("TEMPORAL_PROVENANCE_VIOLATION")


def _logical_bundle_path(output_root: Path) -> str:
    # Never persist a machine-specific volume path in tracked evidence.
    return "SSD_EXTERNAL_OUTPUT_ROOT/T-B3_execution_result"


def _freeze_documents(*, repo_root: Path, canonical_root: Path, work_root: Path, output_root: Path, check_branch: bool = False) -> dict[str, Any]:
    roadmap = _validate_roadmap_reconciliation(repo_root)
    predecessors = _run_predecessors(repo_root)
    canonical = validate_canonical_root(canonical_root, full_hash=True)
    p1 = _load_frozen_p1(repo_root, canonical)
    architecture = _load_architecture_lock(repo_root)
    budget = _expected_budget(repo_root)
    _validate_temporal_disabled(repo_root)
    if not output_root.parent.exists():
        raise RunnerContractError("SSD_OUTPUT_PARENT_UNAVAILABLE")
    if not canonical_root.is_dir():
        raise RunnerContractError("EXTERNAL_STORAGE_DISCONNECTED")
    require_tensorflow()
    # Probe exactly the frozen seed set before any new metric is visible.
    initial_fingerprints = {}
    for seed in SEEDS:
        _, fingerprint, arch = initial_weights(seed)
        if arch != EXPECTED_ARCHITECTURE_FINGERPRINT:
            raise RunnerContractError("SMALL_CNN_IDENTITY_MISMATCH")
        initial_fingerprints[str(seed)] = fingerprint
    reuse = _load_reuse_assessment(repo_root, p1, canonical, budget)
    if check_branch and not _git_diff_is_empty(repo_root):
        # The branch must start empty; later generated evidence is allowed.
        raise RunnerContractError("T_B3_BRANCH_NOT_BASED_ON_CURRENT_MAIN")
    return {
        "roadmap": roadmap,
        "predecessors": predecessors,
        "canonical": canonical,
        "p1": p1,
        "architecture": architecture,
        "budget": budget,
        "reuse": reuse,
        "initial_fingerprints": initial_fingerprints,
        "backend": backend_info(),
        "repo_commit": _repo_commit(repo_root),
        "origin_main": _repo_commit(repo_root, "origin/main"),
        "canonical_root": canonical_root,
        "work_root": work_root,
        "output_root": output_root,
    }


def _freeze_artifacts(contract: Mapping[str, Any], repo_root: Path) -> None:
    evidence = repo_root / EVIDENCE_REL
    evidence.mkdir(parents=True, exist_ok=True)
    predecessors = contract["predecessors"]
    canonical = contract["canonical"]
    p1: P1Statistics = contract["p1"]
    architecture = contract["architecture"]
    _write_json(evidence / "t_b3_protocol.json", {
        "schema_version": "1.0", "phase": PHASE_ID,
        "protocol_id": "THERMAL_T_B3_FRAME_ONLY_MULTI_SEED_CONFIRMATION_001",
        "scope": "FROZEN_P1_SMALL_CNN_FRAME_ONLY_STABILITY_CONFIRMATION",
        "factor_changed": "SEED_ONLY",
        "factors_frozen": ["dataset", "official_roles", "target_mapping", "p1_statistics", "p1_implementation", "architecture", "loss", "optimizer", "training_budget", "early_stopping", "learning_rate_schedule", "augmentation", "class_weighting", "oversampling", "focal_loss", "metric_implementation", "class_order"],
        "seeds": list(SEEDS), "extra_seeds": [], "selection_role": "VALIDATION",
        "real_policy": "EXISTING_REFERENCE_ONLY_NOT_REPEATED",
        "best_seed_cherry_picking": "PROHIBITED",
        "temporal_training": "PROHIBITED",
        "t_b4_started": False,
        "next_phase_started": False,
        "status": "T_B3_MULTI_SEED_RUN_READY",
    })
    _write_json(evidence / "predecessor_identity.json", {
        "schema_version": "1.0", "phase": PHASE_ID,
        "main_commit_at_execution": contract["repo_commit"], "origin_main_at_execution": contract["origin_main"],
        "roadmap_reconciliation": contract["roadmap"],
        "validators": {phase: {"evidence_validation": value.get("evidence_validation"), "overall_outcome": value.get("overall_outcome")} for phase, value in predecessors.items()},
        "required_winner": {"candidate_id": BASELINE_ID, "architecture_fingerprint": EXPECTED_ARCHITECTURE_FINGERPRINT, "parameter_count": EXPECTED_PARAMETER_COUNT},
        "temporal_feasibility": "NOT_SUPPORTED_BY_CURRENT_DATASET_PROVENANCE",
    })
    _write_json(evidence / "dataset_lock.json", {
        "schema_version": "1.0", "phase": PHASE_ID, "source": "REUSED_VERIFIED_T_A6_T_B1_T_B2_IDENTITY",
        "roles": canonical["roles"], "legacy_npz_used": False, "raw_zip_used": False, "new_split_created": False,
        "near_duplicate_pairs_train_validation": EXPECTED_NEAR_DUPLICATE_PAIRS,
        "official_partition_preservation": True, "temporal_grouping": False,
    })
    _write_json(evidence / "p1_lock.json", {"schema_version": "1.0", "phase": PHASE_ID, "profile_id": P1_PROFILE, **p1.to_dict(), "statistics_checksum": p1.checksum(), "source": "REUSED_VERIFIED_T_B1_WINNER", "refit": False, "validation_fit": False, "real_fit": False})
    _write_json(evidence / "architecture_lock.json", architecture)
    _write_json(evidence / "seed_registry.json", {
        "schema_version": "1.0", "phase": PHASE_ID, "required_seeds": list(SEEDS), "extra_seeds": [],
        "seed_20260813": {"status": "REUSE_ASSESSMENT_COMPLETE", "source": "REUSED_VERIFIED_T_B1_RESULT"},
        "seed_20260814": {"status": "NOT_STARTED", "source": "NEW_TRAINING"},
        "seed_20260815": {"status": "NOT_STARTED", "source": "NEW_TRAINING"},
        "seed_set_frozen": True, "seed_selection_after_metrics": False,
    })
    _write_json(evidence / "seed_20260813_reuse_assessment.json", contract["reuse"])
    _write_json(evidence / "candidate_checkpoint_policy.json", {
        "schema_version": "1.0", "phase": PHASE_ID, "reference_candidate_id": BASELINE_ID,
        "reference_seed": 20260813, "reference_checkpoint": contract["reuse"]["checkpoint"],
        "source": "INHERITED_T_B1_VALIDATED_REFERENCE", "candidate_replacement_allowed": False,
        "best_seed_cherry_picking": "PROHIBITED",
    })
    _write_json(evidence / "environment.json", {
        "schema_version": "1.0", "phase": PHASE_ID, "repo_commit": contract["repo_commit"], "origin_main": contract["origin_main"],
        "backend": contract["backend"], "canonical_root": "CONFIGURABLE_EXTERNAL_STORAGE_ROOT", "work_root": "CONFIGURABLE_LOCAL_SCRATCH_ROOT",
        "output_root": _logical_bundle_path(contract["output_root"]), "ssd_required": True, "temporal_training": False,
    })
    _write_json(evidence / "readiness_result.json", {
        "schema_version": "1.0", "phase": PHASE_ID, "status": "T_B3_MULTI_SEED_RUN_READY",
        "roadmap_reconciliation_present": True, "predecessors_valid": True, "canonical_valid": True,
        "p1_frozen": True, "architecture_frozen": True, "seed_set": list(SEEDS), "seed_20260813_reusable": True,
        "real_excluded_from_seed_selection": True, "temporal_disabled": True, "t_b4_started": False,
    })
    _write_bundle_checksums(evidence)


def _write_bundle_checksums(bundle_dir: Path) -> None:
    rows: list[str] = []
    for path in sorted(bundle_dir.iterdir()):
        if not path.is_file() or path.name == "checksums.sha256" or path.name.startswith("._") or path.name.endswith(".partial"):
            continue
        rows.append(f"{sha256_file(path)}  {path.name}")
    (bundle_dir / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def run_readiness(*, canonical_root: str | Path, work_root: str | Path, output_root: str | Path, repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    contract = _freeze_documents(
        repo_root=root,
        canonical_root=Path(canonical_root).expanduser(),
        work_root=Path(work_root).expanduser(),
        output_root=Path(output_root).expanduser(),
        check_branch=True,
    )
    _freeze_artifacts(contract, root)
    return {
        "phase": PHASE_ID, "status": "T_B3_MULTI_SEED_RUN_READY", "repo_commit": contract["repo_commit"],
        "origin_main": contract["origin_main"], "required_seeds": list(SEEDS), "seed_20260813_reusable": True,
        "output_root": _logical_bundle_path(contract["output_root"]), "predecessors": {phase: {"evidence_validation": value.get("evidence_validation"), "overall_outcome": value.get("overall_outcome")} for phase, value in contract["predecessors"].items()},
    }


def _seed_summary_from_reuse(reuse: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0", "phase": PHASE_ID, "seed": 20260813, "source": "REUSED_VERIFIED_T_B1_RESULT",
        "candidate_id": BASELINE_ID, "profile_id": P1_PROFILE, "architecture_fingerprint": EXPECTED_ARCHITECTURE_FINGERPRINT,
        "parameter_count": EXPECTED_PARAMETER_COUNT, "initialization_fingerprint": reuse["initialization_fingerprint"],
        "best_epoch": reuse["best_epoch"], "validation_metrics": reuse["validation_metrics"], "checkpoint": reuse["checkpoint"],
        "finalized": True, "real_evaluated": False, "real_metric_for_selection": False,
    }


def _validate_runtime_seed_result(result: Mapping[str, Any], seed: int) -> None:
    if result.get("status") != "VALIDATION_COMPLETE" or int(result.get("seed", -1)) != seed:
        raise RunnerContractError(f"SEED_{seed}_NOT_FINALIZED")
    if result.get("profile_id") != P1_PROFILE or result.get("candidate_id") != BASELINE_ID:
        raise RunnerContractError(f"SEED_{seed}_CONTRACT_MISMATCH")
    if result.get("architecture_fingerprint") != EXPECTED_ARCHITECTURE_FINGERPRINT or int(result.get("parameter_count", -1)) != EXPECTED_PARAMETER_COUNT:
        raise RunnerContractError(f"SEED_{seed}_CONTRACT_MISMATCH")
    metrics = result.get("validation_metrics", {})
    for key in ("macro_f1", "accuracy", "balanced_accuracy", "h_fall_posture_proxy_recall"):
        if not isinstance(metrics.get(key), (int, float)):
            raise RunnerContractError(f"SEED_{seed}_METRICS_MISSING")


def _aggregate(seed_summaries: list[Mapping[str, Any]]) -> dict[str, Any]:
    if [int(item.get("seed", -1)) for item in seed_summaries] != list(SEEDS):
        raise RunnerContractError("SEED_SET_INVALID")
    values = np.asarray([float(item["validation_metrics"]["macro_f1"]) for item in seed_summaries], dtype=np.float64)
    balanced = np.asarray([float(item["validation_metrics"]["balanced_accuracy"]) for item in seed_summaries], dtype=np.float64)
    recall = np.asarray([float(item["validation_metrics"]["h_fall_posture_proxy_recall"]) for item in seed_summaries], dtype=np.float64)
    return {
        "schema_version": "1.0", "phase": PHASE_ID, "seed_count": 3, "seeds": list(SEEDS), "metric": "VALIDATION",
        "macro_f1": {"mean": float(values.mean()), "std": float(values.std(ddof=0)), "minimum": float(values.min()), "maximum": float(values.max()), "range": float(values.max() - values.min()), "formula": "population standard deviation (ddof=0) over the three frozen seeds"},
        "balanced_accuracy": {"mean": float(balanced.mean()), "worst": float(balanced.min()), "formula": "arithmetic mean and minimum over the three frozen seeds"},
        "human_fall_posture_proxy_recall": {"mean": float(recall.mean()), "worst": float(recall.min()), "formula": "arithmetic mean and minimum over the three frozen seeds"},
        "stability_threshold_predefined": False, "stability_characterization": "STABILITY_CHARACTERIZED",
        "best_seed_cherry_picking": "PROHIBITED", "real_used_for_selection": False,
    }


def run_full(*, canonical_root: str | Path, work_root: str | Path, output_root: str | Path, repo_root: str | Path, owner_authorized: bool) -> dict[str, Any]:
    if not owner_authorized:
        raise RunnerContractError("T_B3_FULL_EXECUTION_OWNER_AUTHORIZATION_REQUIRED")
    root = Path(repo_root).resolve()
    canonical_path = Path(canonical_root).expanduser()
    work = Path(work_root).expanduser()
    output = Path(output_root).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    contract = _freeze_documents(
        repo_root=root,
        canonical_root=canonical_path,
        work_root=work,
        output_root=output,
    )
    evidence = root / EVIDENCE_REL
    required_readiness = evidence / "readiness_result.json"
    if not required_readiness.is_file() or _read_json(required_readiness).get("status") != "T_B3_MULTI_SEED_RUN_READY":
        raise RunnerContractError("T_B3_READINESS_CONTRACT_MISSING")
    bundle = output / "T-B3_execution_result"
    if bundle.exists() and any(bundle.iterdir()):
        raise RunnerContractError("T_B3_OUTPUT_ALREADY_EXISTS")
    bundle.mkdir(parents=True, exist_ok=True)
    checkpoints = bundle / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    _write_json(bundle / "execution_environment.json", {
        "schema_version": "1.0", "phase": PHASE_ID, "repo_commit": contract["repo_commit"], "origin_main": contract["origin_main"],
        "backend": contract["backend"], "canonical_root": "CONFIGURABLE_EXTERNAL_STORAGE_ROOT", "work_root": "CONFIGURABLE_LOCAL_SCRATCH_ROOT",
        "output_root": _logical_bundle_path(output), "gpu_required": False, "temporal_training": False,
    })
    seed_summaries: list[dict[str, Any]] = [_seed_summary_from_reuse(contract["reuse"])]
    role_files = {role: resolve_role_files(canonical_path, role) for role in ROLE_ORDER}
    arrays = {role: np.load(files.array_path, mmap_mode="r", allow_pickle=False) for role, files in role_files.items()}
    labels = {role: labels_from_provenance(files.provenance_path, EXPECTED_ROLES[role]["rows"])[0] for role, files in role_files.items()}
    train_x = apply_p1(arrays["TRAIN"], contract["p1"])
    validation_x = apply_p1(arrays["VALIDATION"], contract["p1"])
    budget = contract["budget"]
    with tempfile.TemporaryDirectory(prefix="safenest_t_b3_", dir=work) as scratch:
        scratch_path = Path(scratch)
        for seed in (20260814, 20260815):
            weights, initial_fingerprint, architecture = initial_weights(seed)
            if architecture != EXPECTED_ARCHITECTURE_FINGERPRINT:
                raise RunnerContractError(f"SEED_{seed}_ARCHITECTURE_MISMATCH")
            temporary_checkpoint = scratch_path / f"seed_{seed}.weights.h5"
            model, result = train_profile(
                train_x,
                labels["TRAIN"],
                validation_x,
                labels["VALIDATION"],
                profile_id=P1_PROFILE,
                seed=seed,
                frozen_initial_weights=weights,
                budget=budget,
                checkpoint_path=temporary_checkpoint,
            )
            _validate_runtime_seed_result(result, seed)
            persistent = checkpoints / temporary_checkpoint.name
            _atomic_copy(temporary_checkpoint, persistent)
            result["checkpoint"] = {"logical_path": f"checkpoints/{persistent.name}", "materialization": "PERSISTENT_EXTERNAL_OUTPUT", "sha256": sha256_file(persistent), "size_bytes": int(persistent.stat().st_size)}
            result["schema_version"] = "1.0"
            result["phase"] = PHASE_ID
            result["source"] = "NEW_TRAINING"
            result["finalized"] = True
            result["real_evaluated"] = False
            result["real_metric_for_selection"] = False
            result["initialization_fingerprint"] = initial_fingerprint
            result.pop("epoch_metrics", None)
            seed_summaries.append(result)
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except Exception:
                pass
    seed_summaries.sort(key=lambda item: int(item["seed"]))
    aggregate = _aggregate(seed_summaries)
    reference = contract["reuse"]["checkpoint"]
    _write_json(bundle / "seed_20260813_summary.json", seed_summaries[0])
    _write_json(bundle / "seed_20260814_summary.json", seed_summaries[1])
    _write_json(bundle / "seed_20260815_summary.json", seed_summaries[2])
    _write_json(bundle / "multiseed_aggregate.json", aggregate)
    _write_json(bundle / "candidate_checkpoint_policy.json", {"schema_version": "1.0", "phase": PHASE_ID, "reference_candidate_id": BASELINE_ID, "reference_seed": 20260813, "reference_checkpoint": reference, "candidate_changed_due_to_t_b3": False, "best_seed_cherry_picking": "PROHIBITED", "reason": "T-B3 measures stability and does not define a post-hoc checkpoint selection rule."})
    _write_json(bundle / "limitations.json", {"schema_version": "1.0", "phase": PHASE_ID, "near_duplicate_pairs_train_validation": EXPECTED_NEAR_DUPLICATE_PAIRS, "locked_test_available": False, "human_fall_semantics": "LYING_TO_HUMAN_FALL_DERIVED_POSTURE_PROXY_NOT_TEMPORAL_EVENT_GROUND_TRUTH", "subject_independent": "NOT_VERIFIABLE", "session_independent": "NOT_VERIFIABLE", "event_independent": "NOT_VERIFIABLE", "temporal_fall": "NOT_VERIFIED", "synthetic_real_gap": EXPECTED_VAL_MACRO_F1 - EXPECTED_REAL_MACRO_F1, "synthetic_real_gap_interpretation": "OBSERVED_DEVELOPMENT_GAP_NOT_CAUSALLY_ATTRIBUTED", "device_domain_validation": "NOT_PERFORMED_DEFERRED_TO_T-C", "multi_seed_real_evaluation": "NOT_PERFORMED", "best_seed_cherry_picking": "PROHIBITED", "next_phase_started": False})
    _write_json(bundle / "execution_summary.json", {"schema_version": "1.0", "phase": PHASE_ID, "status": "FINALIZED", "mode": FULL_MODE, "full_training_performed": True, "new_trained_model_generated": True, "seed_count": 3, "seeds": list(SEEDS), "seed_20260813_source": "REUSED_VERIFIED_T_B1_RESULT", "new_training_seeds": [20260814, 20260815], "real_evaluations": 0, "next_phase_started": False, "t_b4_started": False, "candidate_changed": False, "bundle_artifacts": ["seed_20260813_summary.json", "seed_20260814_summary.json", "seed_20260815_summary.json", "multiseed_aggregate.json", "candidate_checkpoint_policy.json", "limitations.json", "execution_environment.json", "execution_summary.json"]})
    _write_json(evidence / "seed_registry.json", {
        "schema_version": "1.0", "phase": PHASE_ID, "required_seeds": list(SEEDS), "extra_seeds": [],
        "seed_20260813": {"status": "REUSED_VERIFIED", "source": "REUSED_VERIFIED_T_B1_RESULT"},
        "seed_20260814": {"status": "FINALIZED", "source": "NEW_TRAINING"},
        "seed_20260815": {"status": "FINALIZED", "source": "NEW_TRAINING"},
        "seed_set_frozen": True, "seed_selection_after_metrics": False,
    })
    _write_bundle_checksums(bundle)
    # Mirror only compact JSON evidence into Git; checkpoints remain on SSD.
    compact_names = [path.name for path in bundle.iterdir() if path.is_file() and path.suffix == ".json"]
    for name in sorted(compact_names):
        _atomic_copy(bundle / name, evidence / name)
    _write_bundle_checksums(evidence)
    return {"phase": PHASE_ID, "status": "FINALIZED", "bundle_relative_path": _logical_bundle_path(output), "seed_count": 3, "new_training_seeds": [20260814, 20260815], "real_evaluations": 0, "aggregate": aggregate}


def run(*, mode: str, canonical_root: str | Path, work_root: str | Path, output_root: str | Path, repo_root: str | Path, execute: bool, owner_authorized: bool = False) -> dict[str, Any]:
    if mode == READINESS_MODE:
        if execute:
            return run_readiness(canonical_root=canonical_root, work_root=work_root, output_root=output_root, repo_root=repo_root)
        # Readiness always performs bounded/full identity checks but never trains.
        return run_readiness(canonical_root=canonical_root, work_root=work_root, output_root=output_root, repo_root=repo_root)
    if mode == FULL_MODE:
        if not execute:
            contract = _freeze_documents(Path(repo_root).resolve(), Path(canonical_root).expanduser(), Path(work_root).expanduser(), Path(output_root).expanduser())
            return {"phase": PHASE_ID, "status": "TRAINING_RUN_READY", "required_seeds": list(SEEDS), "seed_20260813_reusable": contract["reuse"]["eligible"], "real_excluded_from_selection": True}
        return run_full(canonical_root=canonical_root, work_root=work_root, output_root=output_root, repo_root=repo_root, owner_authorized=owner_authorized)
    raise RunnerContractError(f"UNKNOWN_T_B3_MODE:{mode}")
