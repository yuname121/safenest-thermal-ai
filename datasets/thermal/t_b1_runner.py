"""Platform-neutral T-B1 Stage-1 runner and later full-experiment orchestration.

The default and Stage-1 paths are metadata/fixture-only.  Full canonical
training is explicitly gated behind ``mode=FULL_EXPERIMENT`` plus an owner
authorization flag and is not invoked by this phase.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from datasets.thermal.t_b1_model import (
    BASELINE_ID,
    PRIMARY_SEED,
    backend_info,
    create_small_cnn_baseline,
    initial_weights,
    model_contract,
    train_profile,
)
from datasets.thermal.t_b1_preprocessing import (
    PROFILE_IDS,
    P1Statistics,
    apply_profile,
    canonical_json,
    compute_metrics,
    fit_p1_statistics,
    labels_from_provenance,
    select_validation_winner,
    sha256_file,
)


PHASE_ID = "T-B1"
STAGE1_MODE = "STAGE1_IMPLEMENTATION"
FULL_MODE = "FULL_EXPERIMENT"
ROLE_ORDER = ("TRAIN", "VALIDATION", "REAL_EVAL_DEVELOPMENT")
EXPECTED_ROLES: dict[str, dict[str, Any]] = {
    "TRAIN": {
        "rows": 32000,
        "sha256": "749c847fc9ab50ea5eee8827f0d47b5ebaa48165732a382c59f8b96c565b9d93",
        "size_bytes": 634880128,
        "filename": "train_canonical.npy",
        "provenance_filename": "train_provenance.jsonl",
        "source_domain": "SYNTHETIC",
    },
    "VALIDATION": {
        "rows": 8000,
        "sha256": "5d16451702c1bccfa945d9188d9b29a26ce11c8b33bf7a0dfbb25bfa86d74610",
        "size_bytes": 158720128,
        "filename": "validation_canonical.npy",
        "provenance_filename": "validation_provenance.jsonl",
        "source_domain": "SYNTHETIC",
    },
    "REAL_EVAL_DEVELOPMENT": {
        "rows": 8000,
        "sha256": "cd696e68aeec063cbc8185719b4f4dad3d038cb3d28eec0d3701b8311e4ad8f1",
        "size_bytes": 158720128,
        "filename": "real_eval_development_canonical.npy",
        "provenance_filename": "real_eval_development_provenance.jsonl",
        "source_domain": "REAL",
    },
}
T_B0_REL = "datasets/thermal/manifests/T-B0_offline_model_protocol"
TA6_REL = "datasets/thermal/manifests/T-A6_execution_result"


class RunnerContractError(RuntimeError):
    """Raised for fail-closed runner/preflight violations."""


@dataclass(frozen=True)
class RoleFiles:
    role: str
    array_path: Path
    provenance_path: Path


def _candidate_role_files(canonical_root: Path, role: str) -> list[RoleFiles]:
    spec = EXPECTED_ROLES[role]
    filename = spec["filename"]
    provenance = spec["provenance_filename"]
    role_dir = role
    candidates = [
        (canonical_root / role_dir / filename, canonical_root / role_dir / provenance),
        (canonical_root / filename, canonical_root / provenance),
        (canonical_root / "T-A6_real_and_synthetic_canonical" / filename, canonical_root / "T-A6_real_and_synthetic_canonical" / provenance),
        (canonical_root / "T-A6_real_eval_development" / filename, canonical_root / "T-A6_real_eval_development" / provenance),
    ]
    # The real artifact is sometimes stored under a lower-case role directory.
    if role == "REAL_EVAL_DEVELOPMENT":
        candidates.append((canonical_root / "real_eval_development" / filename, canonical_root / "real_eval_development" / provenance))
    return [RoleFiles(role, array_path, provenance_path) for array_path, provenance_path in candidates]


def resolve_role_files(canonical_root: str | Path, role: str) -> RoleFiles:
    root = Path(canonical_root).expanduser()
    if role not in EXPECTED_ROLES:
        raise RunnerContractError(f"unknown canonical role: {role}")
    for candidate in _candidate_role_files(root, role):
        if candidate.array_path.is_file() and candidate.provenance_path.is_file():
            return candidate
    expected = [str(item.array_path.relative_to(root)) for item in _candidate_role_files(root, role)]
    raise RunnerContractError(f"canonical role {role} is unavailable; accepted relative layouts: {expected}")


def _available_memory_bytes() -> int | None:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return page_size * pages
    except (AttributeError, OSError, ValueError):
        return None


def _relative_or_role_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def verify_role_files(role_files: RoleFiles, *, canonical_root: Path, full_hash: bool = True) -> dict[str, Any]:
    role = role_files.role
    spec = EXPECTED_ROLES[role]
    if not role_files.array_path.is_file() or not role_files.provenance_path.is_file():
        raise RunnerContractError(f"{role} canonical/provenance file is missing")
    stat = role_files.array_path.stat()
    if stat.st_size != spec["size_bytes"]:
        raise RunnerContractError(f"{role} size mismatch: {stat.st_size} != {spec['size_bytes']}")
    array = np.load(role_files.array_path, mmap_mode="r", allow_pickle=False)
    if tuple(array.shape) != (spec["rows"], 62, 80):
        raise RunnerContractError(f"{role} shape mismatch: {array.shape}")
    if np.dtype(array.dtype) != np.dtype("<f4"):
        raise RunnerContractError(f"{role} dtype mismatch: {array.dtype}")
    # A bounded sample catches disconnected/short reads before the full hash.
    sample = np.asarray(array[[0, spec["rows"] // 2, spec["rows"] - 1]])
    if not np.all(np.isfinite(sample)):
        raise RunnerContractError(f"{role} sample contains non-finite values")
    measured_sha = sha256_file(role_files.array_path) if full_hash else None
    if full_hash and measured_sha != spec["sha256"]:
        raise RunnerContractError(f"{role} SHA-256 mismatch: {measured_sha} != {spec['sha256']}")
    provenance_sha = sha256_file(role_files.provenance_path) if full_hash else None
    labels, source_labels = labels_from_provenance(role_files.provenance_path, spec["rows"])
    if labels.shape[0] != spec["rows"]:
        raise RunnerContractError(f"{role} target row count mismatch")
    return {
        "role": role,
        "array_path": _relative_or_role_path(role_files.array_path, canonical_root),
        "provenance_path": _relative_or_role_path(role_files.provenance_path, canonical_root),
        "rows": spec["rows"],
        "size_bytes": stat.st_size,
        "sha256": measured_sha,
        "provenance_sha256": provenance_sha,
        "shape": list(array.shape),
        "dtype": "float32_little_endian",
        "unit": "CELSIUS",
        "source_domain": spec["source_domain"],
        "target_distribution": {name: int(np.sum(labels == index)) for index, name in enumerate(("NOT_HUMAN", "HUMAN_NORMAL", "HUMAN_FALL"))},
        "source_label_counts": {name: int(source_labels.count(name)) for name in sorted(set(source_labels))},
    }


def validate_canonical_root(canonical_root: str | Path, *, full_hash: bool = True) -> dict[str, Any]:
    root = Path(canonical_root).expanduser()
    if not root.is_dir():
        raise RunnerContractError("EXTERNAL_CANONICAL_ROOT_UNAVAILABLE")
    records: dict[str, Any] = {}
    for role in ROLE_ORDER:
        role_files = resolve_role_files(root, role)
        records[role] = verify_role_files(role_files, canonical_root=root, full_hash=full_hash)
    if records["REAL_EVAL_DEVELOPMENT"]["source_domain"] != "REAL":
        raise RunnerContractError("REAL role source domain is invalid")
    return {"canonical_root_configured": True, "roles": records}


def _storage_info(work_root: Path, output_root: Path, required_bytes: int) -> dict[str, Any]:
    work_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    work_usage = shutil.disk_usage(work_root)
    output_usage = shutil.disk_usage(output_root)
    return {
        "work_root_free_bytes": int(work_usage.free),
        "output_root_free_bytes": int(output_usage.free),
        "required_bytes": int(required_bytes),
        "work_capacity_pass": bool(work_usage.free >= required_bytes),
        "output_capacity_pass": bool(output_usage.free >= required_bytes),
        "available_memory_bytes": _available_memory_bytes(),
    }


def stage1_contract_check(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if not (root / "scripts/validate_thermal_t_b0.py").is_file():
        raise RunnerContractError("T-B0 predecessor validator is missing")
    weights, fingerprint, architecture = initial_weights(PRIMARY_SEED)
    model = create_small_cnn_baseline()
    if model.count_params() != 312131:
        raise RunnerContractError("SMALL_CNN_BASELINE_V1 parameter contract mismatch")
    return {
        "stage1_external_storage_required": False,
        "baseline": {"candidate_id": BASELINE_ID, "parameter_count": int(model.count_params()), "architecture_fingerprint": architecture, "initial_weight_fingerprint": fingerprint, "weight_tensor_count": len(weights)},
        "backend": backend_info(),
        "profiles": list(PROFILE_IDS),
        "full_experiment_status": "PENDING_EXTERNAL_SSD_EXECUTION",
        "full_training_performed": False,
        "new_trained_model_generated": False,
    }


def run_fixture_smoke(*, repo_root: str | Path, work_root: str | Path) -> dict[str, Any]:
    """Run one tiny epoch per profile; results are fixture-only and ephemeral."""

    from datasets.thermal.t_b1_model import seed_everything

    seed_everything(PRIMARY_SEED)
    rng = np.random.default_rng(PRIMARY_SEED)
    train_frames = rng.normal(23.0, 2.0, size=(12, 62, 80)).astype("<f4")
    validation_frames = rng.normal(23.0, 2.0, size=(6, 62, 80)).astype("<f4")
    train_y = np.asarray([0, 1, 2] * 4, dtype=np.int32)
    validation_y = np.asarray([0, 1, 2] * 2, dtype=np.int32)
    p1 = fit_p1_statistics(train_frames)
    frozen, fingerprint, _ = initial_weights(PRIMARY_SEED)
    budget = {"baseline_budget": {"batch_size": 4, "maximum_epochs": 1, "initial_learning_rate": 0.001, "optimizer": "Adam", "early_stopping": {"mode": "max", "monitor": "validation_macro_f1", "patience": 5, "restore_best_weights": True}, "learning_rate_schedule": {"factor": 0.5, "minimum_learning_rate": 1e-6, "mode": "max", "monitor": "validation_macro_f1", "patience": 3}}}
    rows = []
    for profile in PROFILE_IDS:
        train_x = apply_profile(profile, train_frames, p1_statistics=p1)
        val_x = apply_profile(profile, validation_frames, p1_statistics=p1)
        _, result = train_profile(train_x, train_y, val_x, validation_y, profile_id=profile, seed=PRIMARY_SEED, frozen_initial_weights=frozen, budget=budget)
        rows.append({"profile_id": profile, "status": "FIXTURE_ONLY", "initial_weight_fingerprint": fingerprint, "validation_metrics": result["validation_metrics"]})
    return {"status": "FIXTURE_ONLY", "full_training_performed": False, "new_trained_model_generated": False, "profiles": rows, "work_root_used": True}


def run_full_experiment(*, canonical_root: str | Path, work_root: str | Path, output_root: str | Path, repo_root: str | Path, dry_run: bool, owner_authorized: bool = False) -> dict[str, Any]:
    """Run the owner-authorized comparison and persist compact evidence."""

    if dry_run:
        records = validate_canonical_root(canonical_root, full_hash=True)
        required = sum(int(item["size_bytes"]) for item in records["roles"].values()) * 2 + 2 * 1024**3
        storage = _storage_info(Path(work_root), Path(output_root), required)
        contract = stage1_contract_check(repo_root)
        if not storage["work_capacity_pass"] or not storage["output_capacity_pass"]:
            raise RunnerContractError("INSUFFICIENT_STORAGE_FOR_FULL_EXPERIMENT")
        return {"status": "TRAINING_RUN_READY", "mode": FULL_MODE, "canonical": records, "storage": storage, "runtime": contract, "full_training_performed": False, "new_trained_model_generated": False}
    if not owner_authorized:
        raise RunnerContractError("FULL_EXPERIMENT_OWNER_AUTHORIZATION_REQUIRED")
    # Full execution is intentionally available only behind the explicit owner
    # flag.  Canonical tensors stay external/read-only; checkpoints are copied
    # to the persistent output before the local temporary directory is removed.
    canonical = validate_canonical_root(canonical_root, full_hash=True)
    root = Path(canonical_root).expanduser()
    work = Path(work_root).expanduser()
    output = Path(output_root).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    bundle_dir = output / "T-B1_execution_result"
    checkpoint_output = bundle_dir / "checkpoints"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_output.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    files: dict[str, RoleFiles] = {}
    for role in ROLE_ORDER:
        files[role] = resolve_role_files(root, role)
        arrays[role] = np.load(files[role].array_path, mmap_mode="r", allow_pickle=False)
        labels[role], _ = labels_from_provenance(files[role].provenance_path, EXPECTED_ROLES[role]["rows"])
    p1_stats = fit_p1_statistics(arrays["TRAIN"], train_artifact_sha256=canonical["roles"]["TRAIN"]["sha256"])
    frozen_weights, init_fingerprint, _ = initial_weights(PRIMARY_SEED)
    budget = _load_t_b0_budget(Path(repo_root))
    runtime = backend_info()
    repo_commit = _repo_commit(Path(repo_root))
    with tempfile.TemporaryDirectory(prefix="t_b1_", dir=work) as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        profile_results: list[dict[str, Any]] = []
        models: dict[str, Any] = {}
        for profile in PROFILE_IDS:
            train_x = apply_profile(profile, arrays["TRAIN"], p1_statistics=p1_stats)
            validation_x = apply_profile(profile, arrays["VALIDATION"], p1_statistics=p1_stats)
            temporary_checkpoint = checkpoint_dir / f"{profile}.weights.h5"
            model, result = train_profile(train_x, labels["TRAIN"], validation_x, labels["VALIDATION"], profile_id=profile, seed=PRIMARY_SEED, frozen_initial_weights=frozen_weights, budget=budget, checkpoint_path=temporary_checkpoint)
            persistent_checkpoint = checkpoint_output / temporary_checkpoint.name
            partial_checkpoint = persistent_checkpoint.with_name(persistent_checkpoint.name + ".partial")
            shutil.copy2(temporary_checkpoint, partial_checkpoint)
            os.replace(partial_checkpoint, persistent_checkpoint)
            result["checkpoint"] = {
                "logical_path": f"checkpoints/{persistent_checkpoint.name}",
                "sha256": sha256_file(persistent_checkpoint),
                "size_bytes": int(persistent_checkpoint.stat().st_size),
                "materialization": "PERSISTENT_EXTERNAL_OUTPUT",
            }
            result["preprocessing_statistics"] = p1_stats.to_dict() if profile == "P1_TRAIN_FITTED_GLOBAL_ZSCORE" else None
            profile_results.append(result)
            models[profile] = model
        winner = select_validation_winner(profile_results)
        selected_profile = str(winner["profile_id"])
        real_x = apply_profile(selected_profile, arrays["REAL_EVAL_DEVELOPMENT"], p1_statistics=p1_stats)
        real_probabilities = models[selected_profile].predict(real_x, batch_size=64, verbose=0)
        real_metrics = compute_metrics(labels["REAL_EVAL_DEVELOPMENT"], np.argmax(real_probabilities, axis=1))
        profile_by_id = {str(item["profile_id"]): item for item in profile_results}
        validation_comparison = {
            "schema_version": "1.0",
            "selection_role": "VALIDATION",
            "primary_metric": "macro_f1",
            "tie_tolerance": 1e-5,
            "rule_id": "THERMAL_T_B0_WINNER_RULE_001",
            "candidates": [
                {"profile_id": item["profile_id"], "candidate_id": item["candidate_id"], "best_epoch": item["best_epoch"], "parameter_count": item["parameter_count"], "validation_metrics": item["validation_metrics"]}
                for item in profile_results
            ],
            "winner_profile_id": selected_profile,
        }
        checkpoint_registry = {
            "schema_version": "1.0",
            "storage_scope": "SSD_EXTERNAL_PERSISTENT",
            "checkpoint_count": len(profile_results),
            "checkpoints": [item["checkpoint"] | {"profile_id": item["profile_id"]} for item in profile_results],
            "winner_checkpoint": dict(winner["checkpoint"]),
            "bulk_checkpoints_tracked_in_git": False,
        }
        real_record = {
            "schema_version": "1.0",
            "role": "REAL_EVAL_DEVELOPMENT",
            "reporting_view": "POST_SELECTION_REAL_DOMAIN_DEVELOPMENT_CHARACTERIZATION",
            "profile_id": selected_profile,
            "checkpoint": dict(profile_by_id[selected_profile]["checkpoint"]),
            "metrics": real_metrics,
            "used_for_winner_selection": False,
            "used_for_preprocessing_fit": False,
            "locked_test": False,
        }
        limitations = {
            "schema_version": "1.0",
            "phase": "T-B1",
            "locked_test_available": False,
            "subject_generalization": "NOT_VERIFIABLE",
            "near_duplicate_pairs": 14514,
            "sensitivity_subset": "SENSITIVITY_SUBSET_NOT_MATERIALIZABLE_FROM_CURRENT_COMPACT_EVIDENCE",
            "posture_proxy": "LYING_TO_HUMAN_FALL_DERIVED_POSTURE_PROXY",
            "thermal44_domain": "THERMAL44_DOMAIN_UNVERIFIED_DEFERRED_TO_T-C",
            "license": "LICENSE_NONCOMMERCIAL_RESTRICTION_MANUAL_REVIEW_REQUIRED_FOR_RELEASE",
            "synthetic_real_domain_gap": "LARGE_SYNTHETIC_TO_REAL_DOMAIN_GAP_OBSERVED_NOT_DEPLOYMENT_VALIDATION",
            "t_b2_authorized": "YES_WITH_LIMITATIONS",
        }
        result_bundle = {
            "schema_version": "1.0",
            "status": "FINALIZED",
            "phase": "T-B1",
            "mode": FULL_MODE,
            "profile_order": list(PROFILE_IDS),
            "selected_profile_id": selected_profile,
            "full_training_performed": True,
            "new_trained_model_generated": True,
            "t_b2_authorized": "YES_WITH_LIMITATIONS",
            "bundle_artifacts": {
                "environment": "environment.json", "dataset_identity": "dataset_identity.json", "target_identity": "target_identity.json", "initialization": "initialization_registry.json", "validation_comparison": "validation_comparison.json", "winner_selection": "winner_selection.json", "real_eval_development": "real_eval_development.json", "checkpoint_registry": "checkpoint_registry.json", "metrics_registry": "metrics_registry.json", "limitations": "limitations.json",
            },
        }
        _atomic_json(bundle_dir / "environment.json", {"schema_version": "1.0", "phase": "T-B1", "backend": runtime, "repo_commit": repo_commit, "canonical_root": "CONFIGURABLE_EXTERNAL_STORAGE_ROOT", "work_root": "CONFIGURABLE_LOCAL_SCRATCH_ROOT", "output_root": "CONFIGURABLE_EXTERNAL_OUTPUT_ROOT", "full_execution_authorized": True})
        _atomic_json(bundle_dir / "dataset_identity.json", canonical)
        _atomic_json(bundle_dir / "target_identity.json", {"schema_version": "1.0", "source_labels": ["EMPTY_ROOM", "SITTING", "STANDING", "LYING"], "target_class_order": ["NOT_HUMAN", "HUMAN_NORMAL", "HUMAN_FALL"], "mapping": {"EMPTY_ROOM": "NOT_HUMAN", "SITTING": "HUMAN_NORMAL", "STANDING": "HUMAN_NORMAL", "LYING": "HUMAN_FALL"}, "lying_semantics": "DERIVED_POSTURE_PROXY_NOT_EVENT_GROUND_TRUTH"})
        _atomic_json(bundle_dir / "initialization_registry.json", {"schema_version": "1.0", "seed": PRIMARY_SEED, "candidate_id": BASELINE_ID, "initial_weight_fingerprint": init_fingerprint, "architecture_fingerprint": profile_results[0]["architecture_fingerprint"], "parameter_count": profile_results[0]["parameter_count"], "same_initial_weights_for_all_profiles": True})
        _atomic_json(bundle_dir / "p0_preprocessing.json", {"schema_version": "1.0", "profile_id": PROFILE_IDS[0], "unit": "CELSIUS", "fit_role": None, "operation": "DIRECT_CANONICAL_VALUES_PLUS_CHANNEL"})
        _atomic_json(bundle_dir / "p1_preprocessing.json", {"schema_version": "1.0", "profile_id": PROFILE_IDS[1], **p1_stats.to_dict(), "statistics_checksum": p1_stats.checksum()})
        _atomic_json(bundle_dir / "p2_preprocessing.json", {"schema_version": "1.0", "profile_id": PROFILE_IDS[2], "operation": "LEGACY_PER_FRAME_MINMAX", "fit_role": None, "source_contract": "ThermalInterpreter._prepare_float_frame"})
        for profile in PROFILE_IDS:
            _atomic_json(bundle_dir / f"{profile[:2].lower()}_training_summary.json", profile_by_id[profile])
        _atomic_json(bundle_dir / "validation_comparison.json", validation_comparison)
        _atomic_json(bundle_dir / "winner_selection.json", winner)
        _atomic_json(bundle_dir / "real_eval_development.json", real_record)
        _atomic_json(bundle_dir / "checkpoint_registry.json", checkpoint_registry)
        _atomic_json(bundle_dir / "metrics_registry.json", {"schema_version": "1.0", "validation": validation_comparison["candidates"], "real_eval_development": {"profile_id": selected_profile, "metrics": real_metrics}})
        _atomic_json(bundle_dir / "limitations.json", limitations)
        _atomic_json(bundle_dir / "execution_summary.json", result_bundle)
        _write_bundle_checksums(bundle_dir)
    return {"status": "FINALIZED", "mode": FULL_MODE, "output_root_configured": True, "bundle_relative_path": "T-B1_execution_result", "full_training_performed": True, "new_trained_model_generated": True, "selected_profile_id": selected_profile, "checkpoint_count": len(profile_results)}


def _load_t_b0_budget(repo_root: Path) -> dict[str, Any]:
    path = repo_root / T_B0_REL / "training_budget_policy.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_bundle_checksums(bundle_dir: Path) -> None:
    """Write deterministic checksums for all finalized bundle files."""

    entries: list[str] = []
    for path in sorted(bundle_dir.rglob("*")):
        if not path.is_file() or path.name == "checksums.sha256" or path.name.startswith("._") or path.name.endswith(".partial"):
            continue
        relative = path.relative_to(bundle_dir).as_posix()
        entries.append(f"{sha256_file(path)}  {relative}")
    (bundle_dir / "checksums.sha256").write_text("\n".join(entries) + "\n", encoding="utf-8")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(canonical_json(value), encoding="utf-8")
    os.replace(temporary, path)


def run(*, mode: str, canonical_root: str | Path, work_root: str | Path, output_root: str | Path, repo_root: str | Path, dry_run: bool, execute: bool = False, owner_authorized: bool = False, fixture_smoke: bool = False) -> dict[str, Any]:
    if dry_run and execute:
        raise RunnerContractError("dry-run and execute are mutually exclusive")
    if not dry_run and not execute:
        raise RunnerContractError("one of dry-run or execute is required")
    if mode == STAGE1_MODE:
        if fixture_smoke:
            if not execute:
                raise RunnerContractError("fixture smoke requires --execute")
            return run_fixture_smoke(repo_root=repo_root, work_root=work_root)
        if execute:
            raise RunnerContractError("T-B1 Stage-1 does not execute full training; use --fixture-smoke or later FULL_EXPERIMENT")
        return {"status": "STAGE1_IMPLEMENTATION_READY", "mode": STAGE1_MODE, "contract": stage1_contract_check(repo_root)}
    if mode == FULL_MODE:
        return run_full_experiment(canonical_root=canonical_root, work_root=work_root, output_root=output_root, repo_root=repo_root, dry_run=dry_run, owner_authorized=owner_authorized)
    raise RunnerContractError(f"unknown runner mode: {mode}")
