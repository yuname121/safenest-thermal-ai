#!/usr/bin/env python3
"""Standalone validator for Thermal T-B1 Stage-1 and full-experiment evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.thermal.t_b1_model import (  # noqa: E402
    BASELINE_ID,
    EXPECTED_PARAMETER_COUNT,
    PRIMARY_SEED,
    architecture_fingerprint,
    backend_info,
    create_small_cnn_baseline,
    initial_weights,
)
from datasets.thermal.t_b1_preprocessing import (  # noqa: E402
    CLASS_ORDER,
    PROFILE_IDS,
    PreprocessingContractError,
    P1Statistics,
    apply_p0,
    apply_p1,
    apply_p2,
    canonical_json,
    compute_metrics,
    fit_p1_statistics,
    select_validation_winner,
)

EVIDENCE_REL = "datasets/thermal/manifests/T-B1_preprocessing_comparison"
TA6_REL = "datasets/thermal/manifests/T-A6_execution_result"
TB0_REL = "datasets/thermal/manifests/T-B0_offline_model_protocol"
REQUIRED_JSON = [
    "t_b1_execution_contract.json",
    "dataset_input_contract.json",
    "preprocessing_implementation_registry.json",
    "baseline_model_contract.json",
    "training_runtime_contract.json",
    "external_storage_contract.json",
    "expected_result_schema.json",
    "initialization_contract.json",
    "limitations.json",
    "stage1_validation_result.json",
]
CHECKSUMS_NAME = "checksums.sha256"
FULL_REQUIRED_JSON = [
    "environment.json",
    "dataset_identity.json",
    "target_identity.json",
    "initialization_registry.json",
    "p0_preprocessing.json",
    "p1_preprocessing.json",
    "p2_preprocessing.json",
    "p0_training_summary.json",
    "p1_training_summary.json",
    "p2_training_summary.json",
    "validation_comparison.json",
    "winner_selection.json",
    "real_eval_development.json",
    "checkpoint_registry.json",
    "metrics_registry.json",
    "limitations.json",
    "execution_summary.json",
]
FULL_VALIDATION_RESULT = "validation_result.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _error(errors: list[dict[str, str]], code: str, location: str, message: str) -> None:
    errors.append({"code": code, "location": location, "message": message})


def _warning(warnings: list[dict[str, str]], code: str, location: str, message: str) -> None:
    warnings.append({"code": code, "location": location, "message": message})


def _walk(value: Any, location: str = "$") -> Iterable[tuple[str, Any]]:
    yield location, value
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _walk(value[key], f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{location}[{index}]")


def _portable(value: str) -> bool:
    # TensorFlow exposes physical device identifiers such as
    # ``/physical_device:CPU:0``.  They are runtime identifiers, not persisted
    # filesystem paths, and are normalized as such in the environment record.
    if value.startswith("/physical_device:"):
        return True
    if value.startswith(("/", "~/", "file://")) or "\\" in value:
        return False
    if "/Users/" in value or "/private/" in value or value.startswith("/Volumes/") or value.startswith("/content/"):
        return False
    return True


def _load_documents(evidence_dir: Path, errors: list[dict[str, str]]) -> dict[str, Any]:
    documents: dict[str, Any] = {}
    for name in REQUIRED_JSON:
        path = evidence_dir / name
        if not path.is_file():
            _error(errors, "REQUIRED_ARTIFACT_MISSING", name, "Required T-B1 Stage-1 JSON artifact is missing.")
            continue
        try:
            text = path.read_text(encoding="utf-8")
            value = json.loads(text)
        except (OSError, json.JSONDecodeError) as exc:
            _error(errors, "JSON_READ_FAILED", name, str(exc))
            continue
        documents[name] = value
        if text != canonical_json(value):
            _error(errors, "NONDETERMINISTIC_JSON", name, "JSON must use canonical sorted-key formatting.")
        for location, item in _walk(value, name):
            if isinstance(item, str) and not _portable(item):
                _error(errors, "NONPORTABLE_PATH", location, item)
            if isinstance(item, str) and item.startswith("archive/"):
                _error(errors, "ARCHIVE_TREATED_AS_ACTIVE", location, item)
            if isinstance(item, str) and any(token in item.lower() for token in ("/co2/", "/mmwave/", "integration/")):
                _error(errors, "CROSS_TRACK_REFERENCE", location, item)
    return documents


def _validate_predecessors(repo_root: Path, errors: list[dict[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        from scripts.validate_thermal_t_a6 import validate_evidence as validate_a6

        result["T-A6"] = validate_a6(repo_root=repo_root, evidence_dir=repo_root / TA6_REL, mode="FULL_DATASET", check_checksums=True)
    except Exception as exc:  # pragma: no cover - defensive boundary
        _error(errors, "T_A6_VALIDATOR_ERROR", TA6_REL, str(exc))
        result["T-A6"] = {"evidence_validation": "FAIL"}
    if result["T-A6"].get("evidence_validation") != "PASS":
        _error(errors, "T_A6_PREDECESSOR_INVALID", TA6_REL, str(result["T-A6"].get("overall_outcome")))
    try:
        from scripts.validate_thermal_t_b0 import validate_evidence as validate_b0

        result["T-B0"] = validate_b0(repo_root=repo_root, evidence_dir=repo_root / TB0_REL, check_checksums=True)
    except Exception as exc:  # pragma: no cover - defensive boundary
        _error(errors, "T_B0_VALIDATOR_ERROR", TB0_REL, str(exc))
        result["T-B0"] = {"evidence_validation": "FAIL"}
    if result["T-B0"].get("evidence_validation") != "PASS":
        _error(errors, "T_B0_PREDECESSOR_INVALID", TB0_REL, str(result["T-B0"].get("overall_outcome")))
    if result["T-B0"].get("t_b1_authorized") not in {"YES", "YES_WITH_LIMITATIONS"}:
        _error(errors, "T_B1_AUTHORIZATION_MISSING", TB0_REL, "T-B0 did not authorize T-B1.")
    return result


def _validate_execution_contract(documents: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    doc = documents.get("t_b1_execution_contract.json", {})
    if doc.get("phase") != "T-B1" or doc.get("stage") != "STAGE1_IMPLEMENTATION":
        _error(errors, "PHASE_ID_INVALID", "t_b1_execution_contract.json", "Stage-1 identity must be T-B1.")
    if doc.get("full_experiment_status") != "PENDING_EXTERNAL_SSD_EXECUTION":
        _error(errors, "FULL_EXPERIMENT_CLAIM", "t_b1_execution_contract.json", "Stage 1 must not claim full experiment execution.")
    if doc.get("t_b2_authorized") is not False:
        _error(errors, "T_B2_EARLY_AUTHORIZATION", "t_b1_execution_contract.json", "T-B2 must remain unauthorized.")
    if doc.get("full_training_performed") is not False or doc.get("new_trained_model_generated") is not False:
        _error(errors, "TRAINING_SCOPE_INVALID", "t_b1_execution_contract.json", "Stage 1 cannot claim full training or a new trained model.")
    runner = doc.get("runner", {})
    for key in ("canonical_root", "work_root", "output_root"):
        if runner.get(key) != "CONFIGURABLE_ARGUMENT":
            _error(errors, "PATH_NOT_CONFIGURABLE", f"t_b1_execution_contract.json:runner.{key}", "Runner roots must be configurable.")
    if doc.get("no_colab_dependency") is not True:
        _error(errors, "COLAB_DEPENDENCY", "t_b1_execution_contract.json", "Stage 1 must not depend on Colab.")


def _validate_dataset_contract(documents: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    doc = documents.get("dataset_input_contract.json", {})
    if doc.get("canonical_shape") != [62, 80] or doc.get("canonical_dtype") != "float32_little_endian" or doc.get("canonical_unit") != "CELSIUS":
        _error(errors, "CANONICAL_CONTRACT_INVALID", "dataset_input_contract.json", "T-A6 physical contract changed.")
    if doc.get("legacy_npz_authority") != "PROHIBITED":
        _error(errors, "LEGACY_NPZ_AUTHORITY", "dataset_input_contract.json", "Legacy NPZ cannot become T-B1 authority.")
    if doc.get("raw_sdt_zip_required") is not False:
        _error(errors, "RAW_ZIP_REQUIRED", "dataset_input_contract.json", "T-B1 consumes canonical artifacts, not raw ZIPs.")
    expected = {
        "TRAIN": (32000, "749c847fc9ab50ea5eee8827f0d47b5ebaa48165732a382c59f8b96c565b9d93"),
        "VALIDATION": (8000, "5d16451702c1bccfa945d9188d9b29a26ce11c8b33bf7a0dfbb25bfa86d74610"),
        "REAL_EVAL_DEVELOPMENT": (8000, "cd696e68aeec063cbc8185719b4f4dad3d038cb3d28eec0d3701b8311e4ad8f1"),
    }
    roles = doc.get("roles", {})
    if set(roles) != set(expected):
        _error(errors, "ROLE_SET_INVALID", "dataset_input_contract.json:roles", "TRAIN/VALIDATION/REAL roles are required.")
    for role, (rows, sha) in expected.items():
        item = roles.get(role, {})
        if item.get("rows") != rows or item.get("sha256") != sha or item.get("source_split") not in {"train", "validation", "test"}:
            _error(errors, "ROLE_IDENTITY_INVALID", f"dataset_input_contract.json:roles.{role}", "T-A6 role identity is not pinned.")
        if role == "REAL_EVAL_DEVELOPMENT" and (item.get("winner_selection") is not False or item.get("fit_allowed") is not False):
            _error(errors, "REAL_ROLE_ESCALATION", f"dataset_input_contract.json:roles.{role}", "REAL cannot fit or select.")


def _validate_preprocessing_contract(documents: Mapping[str, Any], repo_root: Path, errors: list[dict[str, str]]) -> None:
    doc = documents.get("preprocessing_implementation_registry.json", {})
    profiles = doc.get("profiles", [])
    if [item.get("profile_id") for item in profiles] != list(PROFILE_IDS):
        _error(errors, "PROFILE_SET_INVALID", "preprocessing_implementation_registry.json:profiles", "Exactly frozen P0/P1/P2 profiles are required.")
    p1 = next((item for item in profiles if item.get("profile_id") == "P1_TRAIN_FITTED_GLOBAL_ZSCORE"), {})
    if p1.get("fit_role") != "TRAIN" or p1.get("validation_fit_allowed") is not False or p1.get("real_fit_allowed") is not False:
        _error(errors, "P1_FIT_SCOPE_INVALID", "preprocessing_implementation_registry.json:P1", "P1 must fit TRAIN only.")
    if doc.get("canonical_artifacts_mutated") is not False:
        _error(errors, "CANONICAL_MUTATION_POLICY", "preprocessing_implementation_registry.json", "Canonical artifacts must remain immutable.")
    module = repo_root / "datasets/thermal/t_b1_preprocessing.py"
    if not module.is_file():
        _error(errors, "PREPROCESSING_MODULE_MISSING", str(module), "Reusable preprocessing module is required.")
        return
    source = module.read_text(encoding="utf-8")
    for token in ("def apply_p0", "def apply_p1", "def apply_p2", "fit_p1_statistics", "TRAIN", "P2_LEGACY_PER_FRAME_MINMAX"):
        if token not in source:
            _error(errors, "PREPROCESSING_IMPLEMENTATION_MISSING", "datasets/thermal/t_b1_preprocessing.py", token)
    try:
        fixture = np.asarray([np.arange(62 * 80, dtype="<f4").reshape(62, 80) + 20.0], dtype="<f4")
        p1_stats = fit_p1_statistics(fixture)
        p0 = apply_p0(fixture)
        p1_applied = apply_p1(fixture, p1_stats)
        p2 = apply_p2(fixture)
        if p0.shape != (1, 62, 80, 1) or p1_applied.shape != p0.shape or p2.shape != p0.shape:
            raise PreprocessingContractError("profile output shape mismatch")
        if not np.array_equal(p0[..., 0], fixture):
            raise PreprocessingContractError("P0 changed canonical values")
        if not np.isclose(float(p1_stats.mean), float(fixture.mean()), atol=1e-6):
            raise PreprocessingContractError("P1 mean is not TRAIN-derived")
    except (PreprocessingContractError, ValueError) as exc:
        _error(errors, "PREPROCESSING_SMOKE_FAILED", "preprocessing_implementation_registry.json", str(exc))


def _validate_model_contract(documents: Mapping[str, Any], repo_root: Path, errors: list[dict[str, str]]) -> None:
    doc = documents.get("baseline_model_contract.json", {})
    if doc.get("candidate_id") != BASELINE_ID or doc.get("parameter_count_target") != EXPECTED_PARAMETER_COUNT:
        _error(errors, "BASELINE_CONTRACT_INVALID", "baseline_model_contract.json", "SMALL_CNN baseline identity/parameter target changed.")
    if doc.get("input_shape") != [1, 62, 80, 1] or doc.get("output_shape") != [1, 3] or doc.get("class_order") != list(CLASS_ORDER):
        _error(errors, "BASELINE_TENSOR_CONTRACT_INVALID", "baseline_model_contract.json", "Baseline tensor/class contract changed.")
    if doc.get("legacy_model_replacement") is not False or doc.get("tflite_conversion_in_stage1") is not False:
        _error(errors, "LEGACY_MODEL_REPLACED", "baseline_model_contract.json", "Stage 1 cannot replace/convert the legacy model.")
    module = repo_root / "datasets/thermal/t_b1_model.py"
    if not module.is_file():
        _error(errors, "MODEL_MODULE_MISSING", str(module), "Reusable baseline model module is required.")
        return
    try:
        model = create_small_cnn_baseline()
        if int(model.count_params()) != EXPECTED_PARAMETER_COUNT:
            _error(errors, "BASELINE_PARAMETER_MISMATCH", "baseline_model_contract.json", str(model.count_params()))
        if doc.get("architecture_fingerprint") != architecture_fingerprint(model):
            _error(errors, "ARCHITECTURE_FINGERPRINT_MISMATCH", "baseline_model_contract.json", "Architecture fingerprint mismatch.")
        _, fingerprint, _ = initial_weights(PRIMARY_SEED)
        if doc.get("initial_weight_fingerprint") != fingerprint:
            _error(errors, "INITIAL_WEIGHT_FINGERPRINT_MISMATCH", "baseline_model_contract.json", "Frozen initial-weight fingerprint mismatch.")
    except Exception as exc:
        _error(errors, "BASELINE_BUILD_FAILED", "baseline_model_contract.json", str(exc))


def _validate_runtime_contract(documents: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    doc = documents.get("training_runtime_contract.json", {})
    seeds = doc.get("seed_bindings", {})
    for key in ("python", "numpy", "tensorflow", "data_shuffle", "weight_initialization"):
        if seeds.get(key) != PRIMARY_SEED:
            _error(errors, "SEED_POLICY_INVALID", f"training_runtime_contract.json:seed_bindings.{key}", "All primary seed bindings must be 20260813.")
    budget = doc.get("budget", {})
    expected = {"maximum_epochs": 20, "batch_size": 64, "optimizer": "Adam", "initial_learning_rate": 0.001, "loss": "unweighted_sparse_categorical_crossentropy", "maximum_tuning_trials_per_profile": 1}
    for key, value in expected.items():
        if budget.get(key) != value:
            _error(errors, "TRAINING_BUDGET_INVALID", f"training_runtime_contract.json:budget.{key}", f"expected {value!r}")
    if budget.get("augmentation") != "DISABLED" or budget.get("class_weighting") != "DISABLED" or budget.get("oversampling") != "DISABLED" or budget.get("focal_loss") != "DISABLED":
        _error(errors, "BASELINE_EXPERIMENT_CONTAMINATION", "training_runtime_contract.json:budget", "T-B1 preprocessing comparison must keep later factors disabled.")
    if doc.get("cpu_supported") is not True or doc.get("gpu_optional") is not True or doc.get("full_execute_requires_owner_authorization") is not True:
        _error(errors, "RUNTIME_BACKEND_POLICY_INVALID", "training_runtime_contract.json", "CPU fallback/GPU optional/owner gate are required.")
    if doc.get("early_stopping", {}).get("monitor") != "validation_macro_f1" or doc.get("early_stopping", {}).get("patience") != 5:
        _error(errors, "EARLY_STOPPING_INVALID", "training_runtime_contract.json:early_stopping", "Validation Macro F1/patience 5 are required.")


def _validate_role_and_results(documents: Mapping[str, Any], errors: list[dict[str, str]], warnings: list[dict[str, str]]) -> None:
    schema = documents.get("expected_result_schema.json", {})
    if schema.get("stage1_status") != "PENDING_FULL_EXPERIMENT" or schema.get("full_experiment_status") != "RESERVED_NOT_EXECUTED":
        _error(errors, "RESULT_SCHEMA_EARLY_RESULT", "expected_result_schema.json", "Stage 1 cannot contain full experiment results.")
    limitation = documents.get("limitations.json", {})
    if limitation.get("near_duplicate_pairs") != 14514 or limitation.get("sensitivity_subset") != "SENSITIVITY_SUBSET_NOT_MATERIALIZABLE_FROM_CURRENT_COMPACT_EVIDENCE":
        _error(errors, "NEAR_DUPLICATE_LIMITATION_REMOVED", "limitations.json", "T-A6 near-duplicate limitation must be preserved.")
    if limitation.get("locked_test_available") is not False or limitation.get("subject_generalization") != "NOT_VERIFIABLE":
        _error(errors, "CLAIM_LIMITATION_REMOVED", "limitations.json", "A-stage limitations must remain explicit.")
    result = documents.get("stage1_validation_result.json", {})
    if result.get("phase") != "T-B1" or result.get("stage") != "STAGE1_IMPLEMENTATION":
        _error(errors, "RESULT_ID_INVALID", "stage1_validation_result.json", "Result must be T-B1 Stage 1.")
    if result.get("full_training_performed") is not False or result.get("new_trained_model_generated") is not False or result.get("performance_winner_selected") is not False:
        _error(errors, "EARLY_PERFORMANCE_CLAIM", "stage1_validation_result.json", "No full performance/winner claim is allowed.")
    _warning(warnings, "FULL_EXPERIMENT_PENDING_EXTERNAL_SSD", "limitations.json", "Full P0/P1/P2 execution is intentionally pending the external canonical artifacts.")
    _warning(warnings, "NO_FULL_TRAINING", "stage1_validation_result.json", "Stage 1 proves infrastructure only; no Thermal performance metrics exist.")


def _validate_checksums(evidence_dir: Path, errors: list[dict[str, str]]) -> None:
    path = evidence_dir / CHECKSUMS_NAME
    if not path.is_file():
        _error(errors, "CHECKSUM_REGISTRY_MISSING", CHECKSUMS_NAME, "T-B1 checksum registry is missing.")
        return
    entries: dict[str, str] = {}
    previous = ""
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", line)
        if not match:
            _error(errors, "CHECKSUM_LINE_INVALID", f"{CHECKSUMS_NAME}:{number}", line)
            continue
        digest, relative = match.groups()
        if relative <= previous:
            _error(errors, "CHECKSUM_ORDER_NONDETERMINISTIC", f"{CHECKSUMS_NAME}:{number}", relative)
        previous = relative
        if relative != f"{EVIDENCE_REL}/{PurePosixPath(relative).name}":
            _error(errors, "CHECKSUM_PATH_INVALID", f"{CHECKSUMS_NAME}:{number}", relative)
        entries[relative] = digest
    expected = {f"{EVIDENCE_REL}/{name}" for name in REQUIRED_JSON}
    if set(entries) != expected:
        _error(errors, "CHECKSUM_ARTIFACT_SET_INVALID", CHECKSUMS_NAME, "Checksums must cover exactly all Stage-1 JSON artifacts.")
    for name in REQUIRED_JSON:
        relative = f"{EVIDENCE_REL}/{name}"
        path = evidence_dir / name
        if path.is_file() and entries.get(relative) != sha256_file(path):
            _error(errors, "CHECKSUM_MISMATCH", relative, "Checksum is stale or incorrect.")


def _load_full_documents(evidence_dir: Path, errors: list[dict[str, str]]) -> dict[str, Any]:
    documents: dict[str, Any] = {}
    for name in FULL_REQUIRED_JSON + [FULL_VALIDATION_RESULT]:
        path = evidence_dir / name
        if not path.is_file():
            if name == FULL_VALIDATION_RESULT:
                continue
            _error(errors, "FULL_REQUIRED_ARTIFACT_MISSING", name, "Required T-B1 full-experiment JSON artifact is missing.")
            continue
        try:
            text = path.read_text(encoding="utf-8")
            value = json.loads(text)
        except (OSError, json.JSONDecodeError) as exc:
            _error(errors, "JSON_READ_FAILED", name, str(exc))
            continue
        documents[name] = value
        if text != canonical_json(value):
            _error(errors, "NONDETERMINISTIC_JSON", name, "JSON must use canonical sorted-key formatting.")
        for location, item in _walk(value, name):
            if isinstance(item, str) and not _portable(item):
                _error(errors, "NONPORTABLE_PATH", location, item)
            if isinstance(item, str) and item.startswith("archive/"):
                _error(errors, "ARCHIVE_TREATED_AS_ACTIVE", location, item)
            if isinstance(item, str) and any(token in item.lower() for token in ("/co2/", "/mmwave/", "integration/")):
                _error(errors, "CROSS_TRACK_REFERENCE", location, item)
    return documents


def _validate_full_dataset_identity(documents: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    dataset = documents.get("dataset_identity.json", {})
    if dataset.get("canonical_root_configured") is not True:
        _error(errors, "CANONICAL_ROOT_INVALID", "dataset_identity.json", "Canonical root was not validated.")
    roles = dataset.get("roles", {})
    expected = {
        "TRAIN": (32000, "749c847fc9ab50ea5eee8827f0d47b5ebaa48165732a382c59f8b96c565b9d93", "SYNTHETIC"),
        "VALIDATION": (8000, "5d16451702c1bccfa945d9188d9b29a26ce11c8b33bf7a0dfbb25bfa86d74610", "SYNTHETIC"),
        "REAL_EVAL_DEVELOPMENT": (8000, "cd696e68aeec063cbc8185719b4f4dad3d038cb3d28eec0d3701b8311e4ad8f1", "REAL"),
    }
    if set(roles) != set(expected):
        _error(errors, "ROLE_SET_INVALID", "dataset_identity.json:roles", "All three frozen T-B1 roles are required.")
    for role, (rows, digest, domain) in expected.items():
        item = roles.get(role, {})
        if item.get("rows") != rows or item.get("sha256") != digest or item.get("source_domain") != domain:
            _error(errors, "ROLE_IDENTITY_INVALID", f"dataset_identity.json:roles.{role}", "Canonical role identity does not match T-A6.")
        if item.get("shape") != [rows, 62, 80] or item.get("dtype") != "float32_little_endian" or item.get("unit") != "CELSIUS":
            _error(errors, "ROLE_TENSOR_CONTRACT_INVALID", f"dataset_identity.json:roles.{role}", "Shape/dtype/unit contract is invalid.")
        if not item.get("provenance_sha256"):
            _error(errors, "PROVENANCE_IDENTITY_MISSING", f"dataset_identity.json:roles.{role}", "Provenance checksum is required.")


def _validate_full_training_contract(documents: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    summary = documents.get("execution_summary.json", {})
    if summary.get("phase") != "T-B1" or summary.get("mode") != "FULL_EXPERIMENT" or summary.get("status") != "FINALIZED":
        _error(errors, "FULL_RESULT_ID_INVALID", "execution_summary.json", "Final result must identify T-B1 FULL_EXPERIMENT.")
    if summary.get("full_training_performed") is not True or summary.get("new_trained_model_generated") is not True:
        _error(errors, "FULL_TRAINING_FLAG_INVALID", "execution_summary.json", "Full execution flags are not true.")
    if list(summary.get("profile_order", [])) != list(PROFILE_IDS):
        _error(errors, "PROFILE_ORDER_INVALID", "execution_summary.json:profile_order", "P0/P1/P2 order is not frozen.")
    if summary.get("t_b2_authorized") not in {"YES_WITH_LIMITATIONS", "NO"}:
        _error(errors, "T_B2_AUTHORIZATION_INVALID", "execution_summary.json:t_b2_authorized", "T-B2 authorization must be explicit.")
    target = documents.get("target_identity.json", {})
    if target.get("target_class_order") != list(CLASS_ORDER) or target.get("mapping") != {"EMPTY_ROOM": "NOT_HUMAN", "SITTING": "HUMAN_NORMAL", "STANDING": "HUMAN_NORMAL", "LYING": "HUMAN_FALL"}:
        _error(errors, "TARGET_MAPPING_INVALID", "target_identity.json", "Frozen source-to-target mapping is invalid.")
    if target.get("lying_semantics") != "DERIVED_POSTURE_PROXY_NOT_EVENT_GROUND_TRUTH":
        _error(errors, "POSTURE_PROXY_SEMANTICS_MISSING", "target_identity.json:lying_semantics", "LYING must remain a derived posture proxy.")
    model = documents.get("initialization_registry.json", {})
    create_small_cnn_baseline()
    _, fingerprint, architecture = initial_weights(PRIMARY_SEED)
    if model.get("seed") != PRIMARY_SEED or model.get("initial_weight_fingerprint") != fingerprint or model.get("architecture_fingerprint") != architecture or model.get("parameter_count") != EXPECTED_PARAMETER_COUNT:
        _error(errors, "INITIALIZATION_CONTRACT_INVALID", "initialization_registry.json", "Seed, initial weights, architecture, or parameter count mismatch.")
    if model.get("same_initial_weights_for_all_profiles") is not True:
        _error(errors, "INITIALIZATION_NOT_SHARED", "initialization_registry.json", "Profiles must share frozen initial weights.")
    for profile in PROFILE_IDS:
        name = f"{profile[:2].lower()}_training_summary.json"
        item = documents.get(name, {})
        if item.get("profile_id") != profile or item.get("candidate_id") != BASELINE_ID or item.get("seed") != PRIMARY_SEED:
            _error(errors, "PROFILE_IDENTITY_INVALID", name, "Profile/candidate/seed identity mismatch.")
        if item.get("status") != "VALIDATION_COMPLETE" or item.get("initial_weight_fingerprint") != fingerprint or item.get("architecture_fingerprint") != architecture or item.get("parameter_count") != EXPECTED_PARAMETER_COUNT:
            _error(errors, "PROFILE_TRAINING_CONTRACT_INVALID", name, "Training output does not match the frozen baseline contract.")
        metrics = item.get("validation_metrics", {})
        if metrics.get("class_order") != list(CLASS_ORDER) or metrics.get("sample_count") != 8000:
            _error(errors, "VALIDATION_METRICS_INVALID", name, "Validation metric support/class order is invalid.")
        checkpoint = item.get("checkpoint", {})
        if not checkpoint.get("logical_path") or not re.fullmatch(r"checkpoints/[^/]+\.weights\.h5", str(checkpoint.get("logical_path"))):
            _error(errors, "CHECKPOINT_IDENTITY_INVALID", name, "Persistent checkpoint logical identity is missing.")
    p1 = documents.get("p1_preprocessing.json", {})
    if p1.get("profile_id") != "P1_TRAIN_FITTED_GLOBAL_ZSCORE" or p1.get("fit_role") != "TRAIN" or p1.get("fit_sample_count") != 32000 or p1.get("fit_pixel_count") != 32000 * 62 * 80:
        _error(errors, "P1_FIT_SCOPE_INVALID", "p1_preprocessing.json", "P1 statistics must be fitted from TRAIN only.")
    try:
        stats = P1Statistics(mean=float(p1["mean"]), std=float(p1["std"]), fit_sample_count=int(p1["fit_sample_count"]), fit_pixel_count=int(p1["fit_pixel_count"]), fit_role=str(p1["fit_role"]), train_artifact_sha256=str(p1["train_artifact_sha256"]), epsilon=float(p1["epsilon"]))
        if p1.get("statistics_checksum") != stats.checksum():
            _error(errors, "P1_STATISTICS_CHECKSUM_INVALID", "p1_preprocessing.json", "P1 statistics checksum is stale.")
    except (KeyError, TypeError, ValueError, PreprocessingContractError) as exc:
        _error(errors, "P1_STATISTICS_INVALID", "p1_preprocessing.json", str(exc))
    comparison = documents.get("validation_comparison.json", {})
    rows = comparison.get("candidates", [])
    if [row.get("profile_id") for row in rows] != list(PROFILE_IDS):
        _error(errors, "VALIDATION_CANDIDATE_SET_INVALID", "validation_comparison.json:candidates", "All profiles must be compared in frozen order.")
    if comparison.get("selection_role") != "VALIDATION" or comparison.get("primary_metric") != "macro_f1" or comparison.get("tie_tolerance") != 1e-5:
        _error(errors, "VALIDATION_POLICY_INVALID", "validation_comparison.json", "Winner policy is not frozen.")
    try:
        recomputed = select_validation_winner(rows)
        if comparison.get("winner_profile_id") != recomputed.get("profile_id") or summary.get("selected_profile_id") != recomputed.get("profile_id"):
            _error(errors, "WINNER_RECOMPUTATION_MISMATCH", "validation_comparison.json", "Winner does not recompute from VALIDATION metrics.")
    except Exception as exc:
        _error(errors, "WINNER_RECOMPUTATION_FAILED", "validation_comparison.json", str(exc))
    winner = documents.get("winner_selection.json", {})
    if winner.get("selection_role") != "VALIDATION" or winner.get("rule_id") != "THERMAL_T_B0_WINNER_RULE_001" or winner.get("real_metrics") is not None:
        _error(errors, "WINNER_ROLE_INVALID", "winner_selection.json", "Winner must be selected on VALIDATION only.")
    if winner.get("profile_id") != summary.get("selected_profile_id"):
        _error(errors, "WINNER_ID_INVALID", "winner_selection.json", "Winner identity disagrees with summary.")
    real = documents.get("real_eval_development.json", {})
    if real.get("role") != "REAL_EVAL_DEVELOPMENT" or real.get("profile_id") != summary.get("selected_profile_id") or real.get("reporting_view") != "POST_SELECTION_REAL_DOMAIN_DEVELOPMENT_CHARACTERIZATION" or real.get("used_for_winner_selection") is not False or real.get("used_for_preprocessing_fit") is not False or real.get("locked_test") is not False:
        _error(errors, "REAL_EVALUATION_ORDER_INVALID", "real_eval_development.json", "REAL must be winner-only, post-selection development characterization.")
    if real.get("metrics", {}).get("sample_count") != 8000 or real.get("metrics", {}).get("class_order") != list(CLASS_ORDER):
        _error(errors, "REAL_METRICS_INVALID", "real_eval_development.json:metrics", "REAL metric support/class order is invalid.")
    limitations = documents.get("limitations.json", {})
    if limitations.get("near_duplicate_pairs") != 14514 or limitations.get("sensitivity_subset") != "SENSITIVITY_SUBSET_NOT_MATERIALIZABLE_FROM_CURRENT_COMPACT_EVIDENCE" or limitations.get("locked_test_available") is not False or limitations.get("subject_generalization") != "NOT_VERIFIABLE" or limitations.get("synthetic_real_domain_gap") != "LARGE_SYNTHETIC_TO_REAL_DOMAIN_GAP_OBSERVED_NOT_DEPLOYMENT_VALIDATION":
        _error(errors, "LIMITATIONS_REMOVED", "limitations.json", "T-A6/T-B0 limitations must remain explicit.")


def _validate_full_checkpoints_and_checksums(evidence_dir: Path, documents: Mapping[str, Any], errors: list[dict[str, str]], warnings: list[dict[str, str]], *, check_checksums: bool) -> None:
    registry = documents.get("checkpoint_registry.json", {})
    scope = registry.get("storage_scope")
    if scope not in {"SSD_EXTERNAL_PERSISTENT", "EXTERNAL_SSD_NOT_TRACKED"}:
        _error(errors, "CHECKPOINT_SCOPE_INVALID", "checkpoint_registry.json:storage_scope", "Checkpoint storage scope must be explicit.")
    for item in registry.get("checkpoints", []):
        logical = str(item.get("logical_path", ""))
        if not logical or not _portable(logical) or logical.startswith("../"):
            _error(errors, "CHECKPOINT_PATH_INVALID", "checkpoint_registry.json", logical)
            continue
        path = evidence_dir / PurePosixPath(logical)
        if path.is_file() and check_checksums:
            if item.get("size_bytes") != path.stat().st_size or item.get("sha256") != sha256_file(path):
                _error(errors, "CHECKPOINT_CHECKSUM_MISMATCH", logical, "Persistent checkpoint identity is stale.")
        elif scope == "SSD_EXTERNAL_PERSISTENT":
            _error(errors, "CHECKPOINT_MISSING", logical, "SSD checkpoint is missing from a materialized full bundle.")
        else:
            _warning(warnings, "CHECKPOINT_EXTERNAL_NOT_TRACKED", logical, "Compact Git evidence preserves the checkpoint SHA but not bulk checkpoint bytes.")
    if not check_checksums:
        return
    checksum_path = evidence_dir / CHECKSUMS_NAME
    if not checksum_path.is_file():
        _error(errors, "CHECKSUM_REGISTRY_MISSING", CHECKSUMS_NAME, "Full bundle checksum registry is missing.")
        return
    entries: dict[str, str] = {}
    previous = ""
    for number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", line)
        if not match:
            _error(errors, "CHECKSUM_LINE_INVALID", f"{CHECKSUMS_NAME}:{number}", line)
            continue
        digest, relative = match.groups()
        if relative <= previous:
            _error(errors, "CHECKSUM_ORDER_NONDETERMINISTIC", f"{CHECKSUMS_NAME}:{number}", relative)
        previous = relative
        if not _portable(relative) or relative.startswith("../"):
            _error(errors, "CHECKSUM_PATH_INVALID", f"{CHECKSUMS_NAME}:{number}", relative)
        entries[relative] = digest
    actual = {path.relative_to(evidence_dir).as_posix() for path in evidence_dir.rglob("*") if path.is_file() and path.name != CHECKSUMS_NAME and not path.name.startswith("._") and not path.name.endswith(".partial")}
    if set(entries) != actual:
        _error(errors, "CHECKSUM_ARTIFACT_SET_INVALID", CHECKSUMS_NAME, "Checksums must cover every materialized full-bundle artifact exactly once.")
    for relative, digest in entries.items():
        path = evidence_dir / PurePosixPath(relative)
        if path.is_file() and sha256_file(path) != digest:
            _error(errors, "CHECKSUM_MISMATCH", relative, "Checksum is stale or incorrect.")


def _validate_full_evidence(*, repo_root: Path, evidence_dir: Path, check_checksums: bool) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    documents = _load_full_documents(evidence_dir, errors)
    predecessors = _validate_predecessors(repo_root, errors)
    if all(name in documents for name in FULL_REQUIRED_JSON):
        _validate_full_dataset_identity(documents, errors)
        _validate_full_training_contract(documents, errors)
        _validate_full_checkpoints_and_checksums(evidence_dir, documents, errors, warnings, check_checksums=check_checksums)
    elif check_checksums:
        _validate_full_checkpoints_and_checksums(evidence_dir, documents, errors, warnings, check_checksums=True)
    if FULL_VALIDATION_RESULT in documents:
        result_doc = documents[FULL_VALIDATION_RESULT]
        if result_doc.get("phase") != "T-B1" or result_doc.get("stage") != "FULL_EXPERIMENT":
            _error(errors, "VALIDATION_RESULT_ID_INVALID", FULL_VALIDATION_RESULT, "Full validation result identity is invalid.")
    _warning(warnings, "NO_PRISTINE_LOCKED_TEST", "limitations.json", "REAL_EVAL_DEVELOPMENT is not an untouched final test.")
    _warning(warnings, "GROUPING_NOT_VERIFIABLE", "limitations.json", "Subject/session/event generalization remains unavailable.")
    _warning(warnings, "NEAR_DUPLICATE_OVERLAP", "limitations.json", "14,514 TRAIN-VALIDATION near-duplicate pairs remain disclosed.")
    errors.sort(key=lambda item: (item["code"], item["location"], item["message"]))
    warnings.sort(key=lambda item: (item["code"], item["location"], item["message"]))
    predecessors_pass = predecessors.get("T-A6", {}).get("evidence_validation") == "PASS" and predecessors.get("T-B0", {}).get("evidence_validation") == "PASS"
    passed = not errors and predecessors_pass
    return {
        "phase": "T-B1",
        "stage": "FULL_EXPERIMENT",
        "schema_version": "1.0",
        "evidence_validation": "PASS" if passed else "FAIL",
        "overall_outcome": "T_B1_FULL_COMPLETE_WITH_LIMITATIONS" if passed else "T_B1_FULL_BLOCKED",
        "full_experiment": "FINALIZED" if passed else "BLOCKED",
        "t_b2_authorized": "YES_WITH_LIMITATIONS" if passed else False,
        "full_training_performed": bool(documents.get("execution_summary.json", {}).get("full_training_performed") is True),
        "new_trained_model_generated": bool(documents.get("execution_summary.json", {}).get("new_trained_model_generated") is True),
        "performance_winner_selected": bool(documents.get("execution_summary.json", {}).get("selected_profile_id")),
        "winner_profile_id": documents.get("execution_summary.json", {}).get("selected_profile_id"),
        "real_role": "REAL_EVAL_DEVELOPMENT",
        "predecessors": predecessors,
        "error_count": len(errors),
        "errors": errors,
        "warning_count": len(warnings),
        "warnings": warnings,
    }


def write_full_result_and_checksums(evidence_dir: Path, result: Mapping[str, Any]) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / FULL_VALIDATION_RESULT).write_text(canonical_json(dict(result)), encoding="utf-8")
    entries: list[str] = []
    for path in sorted(evidence_dir.rglob("*")):
        if not path.is_file() or path.name == CHECKSUMS_NAME or path.name.startswith("._") or path.name.endswith(".partial"):
            continue
        entries.append(f"{sha256_file(path)}  {path.relative_to(evidence_dir).as_posix()}")
    (evidence_dir / CHECKSUMS_NAME).write_text("\n".join(entries) + "\n", encoding="utf-8")


def validate_evidence(*, repo_root: Path = ROOT, evidence_dir: Path | None = None, mode: str = "STAGE1_IMPLEMENTATION", check_checksums: bool = True) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    evidence_dir = Path(evidence_dir or repo_root / EVIDENCE_REL).resolve()
    if mode == "FULL_EXPERIMENT":
        return _validate_full_evidence(repo_root=repo_root, evidence_dir=evidence_dir, check_checksums=check_checksums)
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    documents = _load_documents(evidence_dir, errors)
    predecessors = _validate_predecessors(repo_root, errors)
    if all(name in documents for name in REQUIRED_JSON):
        _validate_execution_contract(documents, errors)
        _validate_dataset_contract(documents, errors)
        _validate_preprocessing_contract(documents, repo_root, errors)
        _validate_model_contract(documents, repo_root, errors)
        _validate_runtime_contract(documents, errors)
        _validate_role_and_results(documents, errors, warnings)
    if check_checksums:
        _validate_checksums(evidence_dir, errors)
    errors.sort(key=lambda item: (item["code"], item["location"], item["message"]))
    warnings.sort(key=lambda item: (item["code"], item["location"], item["message"]))
    passed = not errors and predecessors.get("T-A6", {}).get("evidence_validation") == "PASS" and predecessors.get("T-B0", {}).get("evidence_validation") == "PASS"
    return {
        "phase": "T-B1",
        "stage": "STAGE1_IMPLEMENTATION",
        "schema_version": "1.0",
        "evidence_validation": "PASS" if passed else "FAIL",
        "overall_outcome": "T_B1_STAGE1_IMPLEMENTATION_READY_WITH_LIMITATIONS" if passed else "T_B1_STAGE1_BLOCKED",
        "stage1_gate": "T_B1_STAGE1_IMPLEMENTATION_READY_WITH_LIMITATIONS" if passed else "T_B1_STAGE1_BLOCKED",
        "full_experiment": "PENDING_EXTERNAL_SSD_EXECUTION",
        "t_b2_authorized": False,
        "full_training_performed": False,
        "new_trained_model_generated": False,
        "performance_winner_selected": False,
        "predecessors": predecessors,
        "error_count": len(errors),
        "errors": errors,
        "warning_count": len(warnings),
        "warnings": warnings,
    }


def write_result_and_checksums(evidence_dir: Path, result: Mapping[str, Any]) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    result_path = evidence_dir / "stage1_validation_result.json"
    result_path.write_text(canonical_json(dict(result)), encoding="utf-8")
    entries = []
    for name in REQUIRED_JSON:
        path = evidence_dir / name
        if not path.is_file():
            continue
        entries.append(f"{sha256_file(path)}  {EVIDENCE_REL}/{name}")
    (evidence_dir / CHECKSUMS_NAME).write_text(
        "\n".join(sorted(entries, key=lambda line: line.split("  ", 1)[1])) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Thermal T-B1 Stage-1 or FULL_EXPERIMENT evidence")
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--evidence-dir", default=str(ROOT / EVIDENCE_REL))
    parser.add_argument("--mode", choices=("STAGE1_IMPLEMENTATION", "FULL_EXPERIMENT"), default="STAGE1_IMPLEMENTATION")
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--write-result", action="store_true")
    args = parser.parse_args()
    result = validate_evidence(repo_root=Path(args.repo_root), evidence_dir=Path(args.evidence_dir), mode=args.mode, check_checksums=not args.skip_checksums)
    if args.write_result:
        if args.mode == "FULL_EXPERIMENT":
            write_full_result_and_checksums(Path(args.evidence_dir), result)
        else:
            write_result_and_checksums(Path(args.evidence_dir), result)
        result = validate_evidence(repo_root=Path(args.repo_root), evidence_dir=Path(args.evidence_dir), mode=args.mode, check_checksums=True)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["evidence_validation"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
