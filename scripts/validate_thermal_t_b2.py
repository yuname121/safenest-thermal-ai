#!/usr/bin/env python3
"""Standalone validator for the Thermal T-B2 controlled architecture comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.thermal.t_b1_preprocessing import P1Statistics, canonical_json, compare_validation_rows, select_validation_winner  # noqa: E402
from datasets.thermal.t_b2_model import (  # noqa: E402
    ARCHITECTURE_IDS,
    DEPTHWISE_ID,
    DEPTHWISE_PARAMETER_BOUND,
    SMALL_CNN_ID,
    architecture_contract,
    architecture_fingerprint,
)


EVIDENCE_REL = "datasets/thermal/manifests/T-B2_architecture_comparison"
TA6_REL = "datasets/thermal/manifests/T-A6_execution_result"
TB0_REL = "datasets/thermal/manifests/T-B0_offline_model_protocol"
TB1_REL = "datasets/thermal/manifests/T-B1_full_experiment"
CHECKSUMS_NAME = "checksums.sha256"
REQUIRED_JSON = [
    "t_b2_protocol.json",
    "predecessor_identity.json",
    "dataset_lock.json",
    "target_identity.json",
    "p1_lock.json",
    "architecture_candidate_registry.json",
    "small_cnn_baseline_reuse_assessment.json",
    "depthwise_architecture_contract.json",
    "initialization_registry.json",
    "training_result.json",
    "validation_architecture_comparison.json",
    "winner_selection.json",
    "real_eval_development.json",
    "efficiency_summary.json",
    "limitations.json",
    "checkpoint_registry.json",
    "environment.json",
    "execution_summary.json",
]
FULL_REQUIRED_JSON = REQUIRED_JSON + ["validation_result.json"]
EXPECTED_TRAIN = (32000, "749c847fc9ab50ea5eee8827f0d47b5ebaa48165732a382c59f8b96c565b9d93")
EXPECTED_VALIDATION = (8000, "5d16451702c1bccfa945d9188d9b29a26ce11c8b33bf7a0dfbb25bfa86d74610")
EXPECTED_REAL = (8000, "cd696e68aeec063cbc8185719b4f4dad3d038cb3d28eec0d3701b8311e4ad8f1")
EXPECTED_P1_STATS_CHECKSUM = "10b5da044ef33f26715544c3d4bf56e9d999d6c65139e931f6997583b3f5b816"
EXPECTED_B1_VAL_MACRO_F1 = 0.9951295332536425
EXPECTED_B1_REAL_MACRO_F1 = 0.593926523563344


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
    if value.startswith("/physical_device:"):
        return True
    return not (
        value.startswith(("/", "~/", "file://"))
        or "\\" in value
        or "/Users/" in value
        or "/private/" in value
        or value.startswith(("/Volumes/", "/content/"))
    )


def _load_documents(evidence_dir: Path, errors: list[dict[str, str]]) -> dict[str, Any]:
    documents: dict[str, Any] = {}
    for name in FULL_REQUIRED_JSON:
        path = evidence_dir / name
        if not path.is_file():
            _error(errors, "REQUIRED_ARTIFACT_MISSING", name, "Required compact T-B2 artifact is missing.")
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


def _validate_checksums(evidence_dir: Path, errors: list[dict[str, str]]) -> None:
    checksum_path = evidence_dir / CHECKSUMS_NAME
    if not checksum_path.is_file():
        _error(errors, "CHECKSUM_REGISTRY_MISSING", CHECKSUMS_NAME, "T-B2 checksum registry is required.")
        return
    entries: dict[str, str] = {}
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            _error(errors, "CHECKSUM_FORMAT_INVALID", f"{CHECKSUMS_NAME}:{line_number}", line)
            continue
        digest, relative = parts
        if relative.startswith(("/", "~/", "file://")) or "\\" in relative or relative.startswith("archive/"):
            _error(errors, "CHECKSUM_PATH_INVALID", f"{CHECKSUMS_NAME}:{line_number}", relative)
            continue
        if relative in entries:
            _error(errors, "CHECKSUM_DUPLICATE", relative, "Duplicate checksum path.")
        entries[relative] = digest
        path = evidence_dir / relative
        if not path.is_file():
            _error(errors, "CHECKSUM_TARGET_MISSING", relative, "Checksum target is missing.")
        elif sha256_file(path) != digest:
            _error(errors, "CHECKSUM_MISMATCH", relative, "Checksum is stale or incorrect.")
    required = {name for name in FULL_REQUIRED_JSON}
    if not required.issubset(entries):
        _error(errors, "CHECKSUM_COVERAGE_INCOMPLETE", CHECKSUMS_NAME, f"Missing: {sorted(required - set(entries))}")
    if list(entries) != sorted(entries):
        _error(errors, "CHECKSUM_ORDER_NONDETERMINISTIC", CHECKSUMS_NAME, "Checksum paths must be sorted.")


def _validate_predecessors(repo_root: Path, errors: list[dict[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        from scripts.validate_thermal_t_a6 import validate_evidence as validate_a6

        result["T-A6"] = validate_a6(repo_root=repo_root, evidence_dir=repo_root / TA6_REL, mode="FULL_DATASET", check_checksums=True)
    except Exception as exc:  # pragma: no cover
        _error(errors, "T_A6_VALIDATOR_ERROR", TA6_REL, str(exc))
        result["T-A6"] = {"evidence_validation": "FAIL"}
    try:
        from scripts.validate_thermal_t_b0 import validate_evidence as validate_b0

        result["T-B0"] = validate_b0(repo_root=repo_root, evidence_dir=repo_root / TB0_REL, check_checksums=True)
    except Exception as exc:  # pragma: no cover
        _error(errors, "T_B0_VALIDATOR_ERROR", TB0_REL, str(exc))
        result["T-B0"] = {"evidence_validation": "FAIL"}
    try:
        from scripts.validate_thermal_t_b1 import validate_evidence as validate_b1

        result["T-B1"] = validate_b1(repo_root=repo_root, evidence_dir=repo_root / TB1_REL, mode="FULL_EXPERIMENT", check_checksums=True)
    except Exception as exc:  # pragma: no cover
        _error(errors, "T_B1_VALIDATOR_ERROR", TB1_REL, str(exc))
        result["T-B1"] = {"evidence_validation": "FAIL"}
    for phase, item in result.items():
        if item.get("evidence_validation") != "PASS":
            _error(errors, f"{phase.replace('-', '_')}_PREDECESSOR_INVALID", phase, str(item.get("overall_outcome")))
    if result.get("T-B1", {}).get("t_b2_authorized") not in {"YES", "YES_WITH_LIMITATIONS"}:
        _error(errors, "T_B2_AUTHORIZATION_MISSING", TB1_REL, "T-B1 did not authorize T-B2.")
    return result


def _validate_protocol(doc: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    if doc.get("phase") != "T-B2" or doc.get("protocol_id") != "THERMAL_T_B2_CONTROLLED_ARCHITECTURE_COMPARISON_001":
        _error(errors, "PROTOCOL_ID_INVALID", "t_b2_protocol.json", "T-B2 protocol identity is invalid.")
    if doc.get("factor_changed") != "ARCHITECTURE" or doc.get("scope") != "P1_FROZEN_ARCHITECTURE_ONLY_COMPARISON":
        _error(errors, "EXPERIMENT_SCOPE_INVALID", "t_b2_protocol.json", "Architecture must be the only experimental factor.")
    if doc.get("architecture_candidates") != list(ARCHITECTURE_IDS):
        _error(errors, "CANDIDATE_SET_INVALID", "t_b2_protocol.json", "Only the two T-B0 architecture candidates are permitted.")
    if doc.get("selection_role") != "VALIDATION" or doc.get("real_policy") != "SELECTED_WINNER_CHARACTERIZATION_AFTER_VALIDATION_ONLY":
        _error(errors, "SELECTION_POLICY_INVALID", "t_b2_protocol.json", "Selection/REAL policy is invalid.")
    if doc.get("locked_test_available") is not False or doc.get("next_phase_started") is not False:
        _error(errors, "SCOPE_ESCALATION", "t_b2_protocol.json", "LOCKED_TEST and later phases must remain unavailable.")
    forbidden = {"augmentation", "class_weighting", "oversampling", "focal_loss", "dataset", "target_mapping", "p1_statistics", "loss", "optimizer", "training_budget", "seed", "metric_policy", "winner_policy"}
    if not forbidden.issubset(set(doc.get("factors_frozen", []))):
        _error(errors, "FROZEN_FACTOR_SET_INCOMPLETE", "t_b2_protocol.json", "The B1 training and data factors are not all frozen.")


def _validate_predecessor_identity(doc: Mapping[str, Any], predecessors: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    if doc.get("phase") != "T-B2":
        _error(errors, "PREDECESSOR_PHASE_INVALID", "predecessor_identity.json", "Evidence must identify T-B2.")
    validators = doc.get("validators", {})
    for phase in ("T-A6", "T-B0", "T-B1"):
        if validators.get(phase, {}).get("evidence_validation") != "PASS":
            _error(errors, "PREDECESSOR_SNAPSHOT_INVALID", f"predecessor_identity.json:validators.{phase}", "Snapshot is not PASS.")
        if predecessors.get(phase, {}).get("evidence_validation") != "PASS":
            _error(errors, "PREDECESSOR_LIVE_INVALID", phase, "Live predecessor validator is not PASS.")
    winner = doc.get("t_b1_p1_winner", {})
    if winner.get("candidate_id") != SMALL_CNN_ID or not math.isclose(float(winner.get("validation_macro_f1", -1)), 0.9951295332536425, rel_tol=0, abs_tol=1e-12):
        _error(errors, "B1_WINNER_IDENTITY_INVALID", "predecessor_identity.json:t_b1_p1_winner", "T-B1 P1 baseline identity/metric is not authoritative.")
    if not math.isclose(float(winner.get("real_macro_f1", -1)), 0.593926523563344, rel_tol=0, abs_tol=1e-12):
        _error(errors, "B1_REAL_IDENTITY_INVALID", "predecessor_identity.json:t_b1_p1_winner", "Known B1 REAL characterization is not preserved.")


def _validate_dataset(doc: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    roles = doc.get("roles", {})
    expected = {"TRAIN": EXPECTED_TRAIN, "VALIDATION": EXPECTED_VALIDATION, "REAL_EVAL_DEVELOPMENT": EXPECTED_REAL}
    if set(roles) != set(expected):
        _error(errors, "DATASET_ROLE_SET_INVALID", "dataset_lock.json:roles", "Exactly TRAIN/VALIDATION/REAL_EVAL_DEVELOPMENT are required.")
    for role, (rows, digest) in expected.items():
        item = roles.get(role, {})
        if item.get("rows") != rows or item.get("sha256") != digest or item.get("dtype") != "float32_little_endian" or item.get("unit") != "CELSIUS":
            _error(errors, "CANONICAL_IDENTITY_INVALID", f"dataset_lock.json:roles.{role}", "T-A6 canonical identity changed.")
        if role == "REAL_EVAL_DEVELOPMENT" and item.get("source_domain") != "REAL":
            _error(errors, "REAL_DOMAIN_INVALID", f"dataset_lock.json:roles.{role}", "REAL role must remain REAL.")
    if doc.get("legacy_npz_used") is not False or doc.get("raw_zip_used") is not False or doc.get("new_split_created") is not False:
        _error(errors, "DATASET_SCOPE_ESCALATION", "dataset_lock.json", "Legacy NPZ/raw ZIP/new split cannot be used.")
    if doc.get("near_duplicate_pairs_train_validation") != 14514:
        _error(errors, "NEAR_DUPLICATE_LIMITATION_LOST", "dataset_lock.json", "14,514 TRAIN-VALIDATION pairs must remain disclosed.")


def _validate_target(doc: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    expected_mapping = {"EMPTY_ROOM": "NOT_HUMAN", "SITTING": "HUMAN_NORMAL", "STANDING": "HUMAN_NORMAL", "LYING": "HUMAN_FALL"}
    if doc.get("source_labels") != ["EMPTY_ROOM", "SITTING", "STANDING", "LYING"] or doc.get("target_class_order") != ["NOT_HUMAN", "HUMAN_NORMAL", "HUMAN_FALL"]:
        _error(errors, "TARGET_CLASS_ORDER_INVALID", "target_identity.json", "Frozen source/target class order changed.")
    if doc.get("mapping") != expected_mapping or doc.get("source_labels_immutable") is not True:
        _error(errors, "TARGET_MAPPING_INVALID", "target_identity.json", "T-B1 target mapping is not preserved.")
    if doc.get("lying_semantics") != "DERIVED_POSTURE_PROXY_NOT_EVENT_GROUND_TRUTH":
        _error(errors, "TEMPORAL_FALL_CLAIM", "target_identity.json", "LYING must remain a derived posture proxy.")


def _validate_p1(doc: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    if doc.get("profile_id") != "P1_TRAIN_FITTED_GLOBAL_ZSCORE" or doc.get("fit_role") != "TRAIN":
        _error(errors, "P1_IDENTITY_INVALID", "p1_lock.json", "P1 profile or fit role is invalid.")
    if doc.get("statistics_checksum") != EXPECTED_P1_STATS_CHECKSUM:
        _error(errors, "P1_CHECKSUM_INVALID", "p1_lock.json", "P1 checksum differs from T-B1.")
    try:
        stats = P1Statistics(
            mean=float(doc["mean"]),
            std=float(doc["std"]),
            fit_sample_count=int(doc["fit_sample_count"]),
            fit_pixel_count=int(doc["fit_pixel_count"]),
            fit_role=str(doc["fit_role"]),
            train_artifact_sha256=str(doc["train_artifact_sha256"]),
            epsilon=float(doc.get("epsilon", 1e-6)),
        )
        if stats.checksum() != doc.get("statistics_checksum"):
            _error(errors, "P1_STATISTICS_RECOMPUTE_MISMATCH", "p1_lock.json", "Serialized P1 statistics do not match their checksum.")
    except (KeyError, TypeError, ValueError) as exc:
        _error(errors, "P1_STATISTICS_INVALID", "p1_lock.json", str(exc))
    if doc.get("source") != "REUSED_VERIFIED_T_B1_WINNER" or doc.get("refit") is not False or doc.get("validation_fit") is not False or doc.get("real_fit") is not False:
        _error(errors, "P1_REFIT_POLICY_INVALID", "p1_lock.json", "P1 must be reused without refitting.")


def _validate_architectures(registry: Mapping[str, Any], depthwise: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    candidates = registry.get("candidates", [])
    if [item.get("candidate_id") for item in candidates] != list(ARCHITECTURE_IDS):
        _error(errors, "ARCHITECTURE_REGISTRY_INVALID", "architecture_candidate_registry.json", "Candidate ordering/set is not frozen.")
    if registry.get("unregistered_candidates") != []:
        _error(errors, "UNREGISTERED_ARCHITECTURE", "architecture_candidate_registry.json", "Unregistered architecture entered comparison.")
    for candidate_id in ARCHITECTURE_IDS:
        item = next((row for row in candidates if row.get("candidate_id") == candidate_id), {})
        expected = architecture_contract(candidate_id)
        if item.get("winner_eligible") is not True:
            _error(errors, "CANDIDATE_NOT_WINNER_ELIGIBLE", candidate_id, "Both registered T-B0 candidates must remain eligible.")
        if item.get("architecture_fingerprint") != expected["architecture_fingerprint"] or item.get("parameter_count") != expected["parameter_count"]:
            _error(errors, "ARCHITECTURE_CONTRACT_MISMATCH", candidate_id, "Executable architecture differs from frozen contract.")
    expected_depthwise = architecture_contract(DEPTHWISE_ID)
    if depthwise.get("candidate_id") != DEPTHWISE_ID or depthwise.get("architecture_fingerprint") != expected_depthwise["architecture_fingerprint"]:
        _error(errors, "DEPTHWISE_CONTRACT_INVALID", "depthwise_architecture_contract.json", "Depthwise architecture fingerprint differs.")
    if depthwise.get("parameter_count") != expected_depthwise["parameter_count"] or int(depthwise.get("parameter_count", 10**9)) > DEPTHWISE_PARAMETER_BOUND:
        _error(errors, "DEPTHWISE_PARAMETER_VIOLATION", "depthwise_architecture_contract.json", "Depthwise parameter bound is violated.")
    if depthwise.get("frozen_before_metrics") is not True or depthwise.get("freeze_stage") != "PRE_TRAINING":
        _error(errors, "POST_METRIC_ARCHITECTURE_TUNING", "depthwise_architecture_contract.json", "Architecture was not frozen before metrics.")


def _validate_baseline_reuse(doc: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    if doc.get("baseline_candidate_id") != SMALL_CNN_ID or doc.get("source") != "REUSED_VERIFIED_T_B1_P1_RESULT" or doc.get("eligible") is not True or doc.get("retrained") is not False:
        _error(errors, "BASELINE_REUSE_INVALID", "small_cnn_baseline_reuse_assessment.json", "Baseline reuse claim is invalid.")
    comparison = doc.get("contract_comparison", {})
    required = ("train_identity", "validation_identity", "target_mapping", "p1_profile", "p1_statistics", "p1_implementation", "architecture", "seed", "optimizer", "learning_rate", "loss", "batch_size", "maximum_epochs", "early_stopping", "learning_rate_schedule", "augmentation", "class_weighting", "oversampling", "focal_loss", "metric_implementation", "class_order")
    if any(comparison.get(key) is not True for key in required):
        _error(errors, "BASELINE_REUSE_CONTRACT_MISMATCH", "small_cnn_baseline_reuse_assessment.json:contract_comparison", "A material T-B1/T-B2 contract differs.")
    if doc.get("p1_statistics_checksum") != EXPECTED_P1_STATS_CHECKSUM:
        _error(errors, "BASELINE_P1_IDENTITY_INVALID", "small_cnn_baseline_reuse_assessment.json", "Baseline is not tied to frozen P1.")


def _validate_initialization(doc: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    if doc.get("primary_seed") != 20260813 or doc.get("same_initial_weight_sha_required") is not False:
        _error(errors, "INITIALIZATION_POLICY_INVALID", "initialization_registry.json", "Seed/fingerprint policy is invalid.")
    rows = doc.get("candidates", [])
    if [row.get("candidate_id") for row in rows] != list(ARCHITECTURE_IDS):
        _error(errors, "INITIALIZATION_REGISTRY_INVALID", "initialization_registry.json", "Both architecture fingerprints are required.")
    if len({row.get("initial_weight_fingerprint") for row in rows}) != 2:
        _error(errors, "INITIALIZATION_FINGERPRINT_COLLISION", "initialization_registry.json", "Different architectures must retain separate initialization fingerprints.")


def _validate_training(doc: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    if doc.get("architecture_only_factor") is not True or doc.get("independent_tuning") is not False or doc.get("real_used_during_training") is not False:
        _error(errors, "TRAINING_FAIRNESS_INVALID", "training_result.json", "Training contract changed beyond architecture.")
    contract = doc.get("training_contract", {}).get("baseline_budget", {})
    expected = {"batch_size": 64, "maximum_epochs": 20, "initial_learning_rate": 0.001, "optimizer": "Adam", "loss": "unweighted_sparse_categorical_crossentropy"}
    for key, value in expected.items():
        if contract.get(key) != value:
            _error(errors, "TRAINING_BUDGET_INVALID", f"training_result.json:training_contract.baseline_budget.{key}", "Frozen B1 budget changed.")
    full_contract = doc.get("training_contract", {})
    if full_contract.get("augmentation", {}).get("baseline") != "DISABLED":
        _error(errors, "FORBIDDEN_TRAINING_STRATEGY", "training_result.json:training_contract.augmentation", "Augmentation must remain disabled.")
    imbalance = full_contract.get("class_imbalance", {})
    for key in ("class_weight", "focal_loss", "oversampling"):
        if imbalance.get(key) not in {"DISABLED_IN_BASELINE", "UNWEIGHTED_CROSS_ENTROPY"}:
            _error(errors, "FORBIDDEN_TRAINING_STRATEGY", f"training_result.json:training_contract.class_imbalance.{key}", "Class-imbalance strategy must remain disabled.")


def _validate_selection(comparison: Mapping[str, Any], selection: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    rows = comparison.get("candidates", [])
    if comparison.get("selection_role") != "VALIDATION" or comparison.get("real_metrics_in_selection_input") is not False or [row.get("candidate_id") for row in rows] != list(ARCHITECTURE_IDS):
        _error(errors, "VALIDATION_SELECTION_INPUT_INVALID", "validation_architecture_comparison.json", "Selection input must be exactly the two VALIDATION candidates.")
    if selection.get("selection_role") != "VALIDATION" or selection.get("real_used_for_selection") is not False:
        _error(errors, "REAL_SELECTION_CONTAMINATION", "winner_selection.json", "REAL may not select or tie-break the winner.")
    if "REAL" in str(selection.get("tie_break_level", "")).upper():
        _error(errors, "REAL_TIE_BREAK_FORBIDDEN", "winner_selection.json", "REAL may not participate in tie-breaking.")
    if any("real_metrics" in row for row in selection.get("selection_input_metrics", [])):
        _error(errors, "REAL_METRICS_IN_SELECTION", "winner_selection.json", "REAL metrics entered winner input.")
    try:
        expected_winner = select_validation_winner(selection.get("selection_input_metrics", [])) if selection.get("selection_input_metrics") else {}
    except Exception as exc:
        expected_winner = {}
        _error(errors, "REAL_METRICS_IN_SELECTION", "winner_selection.json", str(exc))
    if expected_winner and selection.get("winner_candidate_id") != expected_winner.get("candidate_id"):
        _error(errors, "WINNER_RULE_MISMATCH", "winner_selection.json", "Winner does not follow frozen VALIDATION ranking.")
    if selection.get("winner_checkpoint") is None:
        _error(errors, "WINNER_CHECKPOINT_MISSING", "winner_selection.json", "Winner checkpoint identity is required.")


def _validate_real(doc: Mapping[str, Any], selection: Mapping[str, Any], errors: list[dict[str, str]], warnings: list[dict[str, str]]) -> None:
    if doc.get("role") != "REAL_EVAL_DEVELOPMENT" or doc.get("used_for_winner_selection") is not False or doc.get("used_for_preprocessing_fit") is not False or doc.get("locked_test") is not False:
        _error(errors, "REAL_EVALUATION_ORDER_INVALID", "real_eval_development.json", "REAL must be post-selection development characterization only.")
    if doc.get("winner_candidate_id") != selection.get("winner_candidate_id"):
        _error(errors, "REAL_WINNER_MISMATCH", "real_eval_development.json", "REAL result is not for the frozen winner.")
    if doc.get("losing_candidate_new_real_evaluation") is not False:
        _error(errors, "LOSING_REAL_EVALUATED", "real_eval_development.json", "Losing architecture must not be newly evaluated on REAL.")
    if doc.get("final_test_claim") is not False or doc.get("reporting_view") != "POST_SELECTION_REAL_DOMAIN_DEVELOPMENT_CHARACTERIZATION":
        _error(errors, "REAL_CLAIM_SCOPE_INVALID", "real_eval_development.json", "REAL cannot be called a final/locked test.")
    metrics = doc.get("metrics", {})
    for key in ("macro_f1", "accuracy", "balanced_accuracy"):
        if not isinstance(metrics.get(key), (float, int)):
            _error(errors, "REAL_METRICS_MISSING", f"real_eval_development.json:metrics.{key}", "Selected-winner REAL metric is required.")
    if doc.get("result_source") == "REUSED_VERIFIED_T_B1_RESULT" and not math.isclose(float(metrics.get("macro_f1", -1)), EXPECTED_B1_REAL_MACRO_F1, rel_tol=0, abs_tol=1e-12):
        _error(errors, "REUSED_REAL_METRIC_MISMATCH", "real_eval_development.json", "Reused B1 REAL metric changed.")
    if doc.get("result_source") not in {"REUSED_VERIFIED_T_B1_RESULT", "NEW_POST_SELECTION_WINNER_EVALUATION"}:
        _error(errors, "REAL_RESULT_SOURCE_INVALID", "real_eval_development.json", "Unsupported REAL result source.")
    _warning(warnings, "NO_PRISTINE_LOCKED_TEST", "real_eval_development.json", "REAL_EVAL_DEVELOPMENT is not an untouched final test.")


def _validate_limitations(doc: Mapping[str, Any], errors: list[dict[str, str]], warnings: list[dict[str, str]]) -> None:
    if doc.get("near_duplicate_pairs_train_validation") != 14514:
        _error(errors, "NEAR_DUPLICATE_LIMITATION_LOST", "limitations.json", "14,514 TRAIN-VALIDATION near-duplicate pairs must remain disclosed.")
    required = {"NOT_VERIFIABLE", "NOT_PERFORMED_DEFERRED_TO_T-C"}
    if doc.get("human_fall_semantics") != "LYING_TO_HUMAN_FALL_DERIVED_POSTURE_PROXY_NOT_TEMPORAL_EVENT_GROUND_TRUTH":
        _error(errors, "TEMPORAL_FALL_CLAIM", "limitations.json", "HUMAN_FALL must remain a posture proxy.")
    if doc.get("device_domain_validation") != "NOT_PERFORMED_DEFERRED_TO_T-C" or doc.get("next_phase_started") is not False:
        _error(errors, "LATER_PHASE_CLAIM", "limitations.json", "Device-domain/later phase work is out of scope.")
    for key in ("subject_independent", "session_independent", "event_independent"):
        if doc.get(key) != "NOT_VERIFIABLE":
            _error(errors, "GROUPING_CLAIM_UNSUPPORTED", f"limitations.json:{key}", "Generalization provenance is unavailable.")
    if not isinstance(doc.get("synthetic_real_gap"), (float, int)):
        _error(errors, "REAL_GAP_MISSING", "limitations.json", "Observed synthetic-to-real gap is required.")
    _warning(warnings, "NEAR_DUPLICATE_OVERLAP", "limitations.json", "14,514 TRAIN-VALIDATION near-duplicate pairs remain disclosed.")
    _warning(warnings, "GROUPING_NOT_VERIFIABLE", "limitations.json", "Subject/session/event generalization remains unavailable.")


def _validate_result(doc: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    if doc.get("phase") != "T-B2" or doc.get("mode") != "FULL_EXPERIMENT" or doc.get("status") != "FINALIZED" or doc.get("next_phase_started") is not False:
        _error(errors, "EXECUTION_RESULT_INVALID", "execution_summary.json", "T-B2 execution scope/status is invalid.")
    if doc.get("winner_candidate_id") not in ARCHITECTURE_IDS:
        _error(errors, "WINNER_MISSING", "execution_summary.json", "A registered architecture winner is required.")


def validate_evidence(*, repo_root: Path = ROOT, evidence_dir: Path | None = None, mode: str = "FULL_EXPERIMENT", check_checksums: bool = True) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    evidence_dir = Path(evidence_dir or repo_root / EVIDENCE_REL).resolve()
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    documents = _load_documents(evidence_dir, errors)
    predecessors = _validate_predecessors(repo_root, errors)
    if all(name in documents for name in FULL_REQUIRED_JSON):
        _validate_protocol(documents["t_b2_protocol.json"], errors)
        _validate_predecessor_identity(documents["predecessor_identity.json"], predecessors, errors)
        _validate_dataset(documents["dataset_lock.json"], errors)
        _validate_target(documents["target_identity.json"], errors)
        _validate_p1(documents["p1_lock.json"], errors)
        _validate_architectures(documents["architecture_candidate_registry.json"], documents["depthwise_architecture_contract.json"], errors)
        _validate_baseline_reuse(documents["small_cnn_baseline_reuse_assessment.json"], errors)
        _validate_initialization(documents["initialization_registry.json"], errors)
        _validate_training(documents["training_result.json"], errors)
        _validate_selection(documents["validation_architecture_comparison.json"], documents["winner_selection.json"], errors)
        _validate_real(documents["real_eval_development.json"], documents["winner_selection.json"], errors, warnings)
        _validate_limitations(documents["limitations.json"], errors, warnings)
        _validate_result(documents["execution_summary.json"], errors)
        if documents["checkpoint_registry.json"].get("bulk_checkpoints_tracked_in_git") is not False:
            _error(errors, "BULK_CHECKPOINT_TRACKING", "checkpoint_registry.json", "Bulk checkpoints must remain external.")
        if documents["efficiency_summary.json"].get("deployment_claim") is not False:
            _error(errors, "DEPLOYMENT_CLAIM", "efficiency_summary.json", "Parameter count is not hardware validation.")
    if check_checksums:
        _validate_checksums(evidence_dir, errors)
    errors.sort(key=lambda item: (item["code"], item["location"], item["message"]))
    warnings.sort(key=lambda item: (item["code"], item["location"], item["message"]))
    passed = not errors and all(predecessors.get(phase, {}).get("evidence_validation") == "PASS" for phase in ("T-A6", "T-B0", "T-B1"))
    return {
        "phase": "T-B2",
        "mode": mode,
        "schema_version": "1.0",
        "evidence_validation": "PASS" if passed else "FAIL",
        "overall_outcome": "T_B2_COMPLETE_WITH_LIMITATIONS" if passed else "T_B2_BLOCKED",
        "full_experiment": "FINALIZED" if passed else "BLOCKED",
        "t_b3_authorized": False,
        "predecessors": predecessors,
        "error_count": len(errors),
        "errors": errors,
        "warning_count": len(warnings),
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Thermal T-B2 architecture comparison evidence")
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--evidence-dir", default=str(ROOT / EVIDENCE_REL))
    parser.add_argument("--mode", choices=("FULL_EXPERIMENT",), default="FULL_EXPERIMENT")
    parser.add_argument("--skip-checksums", action="store_true")
    args = parser.parse_args()
    result = validate_evidence(repo_root=Path(args.repo_root), evidence_dir=Path(args.evidence_dir), mode=args.mode, check_checksums=not args.skip_checksums)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["evidence_validation"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
