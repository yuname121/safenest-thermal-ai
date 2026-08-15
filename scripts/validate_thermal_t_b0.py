#!/usr/bin/env python3
"""Standalone, non-training validator for the Thermal T-B0 protocol.

The validator consumes compact T-B0 metadata, the tracked T-A6 compact bundle,
the tracked legacy TFLite artifact, and repository source text.  It never opens
or hydrates the large SDT payloads and never trains or converts a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EVIDENCE_REL = "datasets/thermal/manifests/T-B0_offline_model_protocol"
TA6_REL = "datasets/thermal/manifests/T-A6_execution_result"
REQUIRED_JSON = [
    "t_b0_protocol.json",
    "dataset_authority.json",
    "target_contract.json",
    "existing_model_inventory.json",
    "existing_runtime_preprocessing_audit.json",
    "preprocessing_candidate_registry.json",
    "model_candidate_registry.json",
    "evaluation_role_policy.json",
    "metric_policy.json",
    "near_duplicate_sensitivity_policy.json",
    "winner_selection_policy.json",
    "randomness_policy.json",
    "training_budget_policy.json",
    "deployment_measurement_policy.json",
    "license_boundary.json",
    "limitations.json",
    "validation_result.json",
]
CONTRACT_JSON = REQUIRED_JSON[:-1]
CHECKSUMS_NAME = "checksums.sha256"
EXPECTED_ROLES = {
    "TRAIN": {
        "rows": 32000,
        "source_domain": "SYNTHETIC",
        "source_split": "train",
        "sha256": "749c847fc9ab50ea5eee8827f0d47b5ebaa48165732a382c59f8b96c565b9d93",
        "provenance_sha256": "b4ec8228e35703bac3319ca218a69fdef43013ea44b23376da4929b274d24888",
    },
    "VALIDATION": {
        "rows": 8000,
        "source_domain": "SYNTHETIC",
        "source_split": "validation",
        "sha256": "5d16451702c1bccfa945d9188d9b29a26ce11c8b33bf7a0dfbb25bfa86d74610",
        "provenance_sha256": "48ebd03ca6f8d738ad7048aa72d4c454fd821140aa887971c27c5b49c1d7ec63",
    },
    "REAL_EVAL_DEVELOPMENT": {
        "rows": 8000,
        "source_domain": "REAL",
        "source_split": "test",
        "sha256": "cd696e68aeec063cbc8185719b4f4dad3d038cb3d28eec0d3701b8311e4ad8f1",
        "provenance_sha256": "c9d12f12d845d218e5636dad84a4a094e869faa29d95feb4a6f69603c195e550",
    },
}
EXPECTED_MODEL_SHA = "5b56da8d127ccef85f30b6459cc0cfe2d86490e41f3caa5bd2a7b70bbc46ae84"
MODEL_PATH = "models/thermal/thermal_fall_int8_v0.1.0.tflite"
_TA6_CACHE: dict[str, dict[str, Any]] = {}
MODEL_METRIC_KEYS = {
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
    "prediction_distribution",
    "loss",
    "auc",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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


def _portable_string(value: str) -> bool:
    if value.startswith(("/", "~/", "file://")) or "\\" in value:
        return False
    if "/Users/" in value or "/private/" in value or value.startswith("/content/") or "iCloud" in value:
        return False
    return True


def _load_documents(evidence_dir: Path, errors: list[dict[str, str]]) -> dict[str, Any]:
    documents: dict[str, Any] = {}
    for name in REQUIRED_JSON:
        path = evidence_dir / name
        if not path.is_file():
            _error(errors, "REQUIRED_ARTIFACT_MISSING", name, "Required T-B0 JSON artifact is missing.")
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
            if isinstance(item, str) and not _portable_string(item):
                _error(errors, "NONPORTABLE_PATH", location, item)
            if isinstance(item, str) and item.startswith("archive/"):
                _error(errors, "ARCHIVE_TREATED_AS_ACTIVE", location, item)
            if isinstance(item, dict):
                for key in item:
                    lowered = str(key).lower()
                    if lowered in {"performance_result", "model_performance_result", "training_result", "trained_model_artifact"}:
                        _error(errors, "T_B1_RESULT_PRESENT", f"{location}.{key}", "T-B0 cannot contain later training or performance results.")
    return documents


def _validate_a6(repo_root: Path, errors: list[dict[str, str]]) -> dict[str, Any]:
    cache_key = str(repo_root)
    if cache_key in _TA6_CACHE:
        result = _TA6_CACHE[cache_key]
        if result.get("evidence_validation") != "PASS":
            _error(errors, "T_A6_VALIDATION_FAILED", TA6_REL, str(result.get("overall_outcome")))
        return result
    try:
        from scripts.validate_thermal_t_a6 import validate_evidence

        result = validate_evidence(
            repo_root=repo_root,
            evidence_dir=repo_root / TA6_REL,
            mode="FULL_DATASET",
            check_checksums=True,
        )
    except Exception as exc:  # pragma: no cover - defensive contract boundary
        _error(errors, "T_A6_VALIDATOR_ERROR", TA6_REL, str(exc))
        return {"evidence_validation": "FAIL", "full_t_a6_gate": "NOT_VERIFIABLE"}
    if result.get("evidence_validation") != "PASS":
        _error(errors, "T_A6_VALIDATION_FAILED", TA6_REL, str(result.get("overall_outcome")))
    if result.get("full_t_a6_gate") != "T_A6_FULL_COMPLETE_WITH_LIMITATIONS":
        _error(errors, "T_A6_GATE_INVALID", TA6_REL, str(result.get("full_t_a6_gate")))
    _TA6_CACHE[cache_key] = result
    return result


def _validate_dataset_authority(documents: Mapping[str, Any], repo_root: Path, errors: list[dict[str, str]]) -> None:
    authority = documents.get("dataset_authority.json", {})
    if authority.get("total_canonical_rows") != 48000:
        _error(errors, "DATASET_TOTAL_INVALID", "dataset_authority.json", "T-A6 total must remain 48,000 rows.")
    if authority.get("official_partition_preservation") is not True:
        _error(errors, "PARTITION_PRESERVATION_MISSING", "dataset_authority.json", "Official source partitions must be preserved.")
    if authority.get("random_resplit") != "PROHIBITED" or authority.get("hash_resplit") != "PROHIBITED":
        _error(errors, "RESPLIT_POLICY_INVALID", "dataset_authority.json", "Random/hash resplits must be prohibited.")
    locked = authority.get("locked_test", {})
    if locked.get("available") is not False or locked.get("creation_in_t_b0") is not False:
        _error(errors, "LOCKED_TEST_ESCALATION", "dataset_authority.json:locked_test", "T-B0 cannot create or claim a LOCKED_TEST.")
    legacy = authority.get("legacy_npz", {})
    if legacy.get("authority") != "PROHIBITED":
        _error(errors, "LEGACY_NPZ_AUTHORITY_ESCALATED", "dataset_authority.json:legacy_npz", "Legacy NPZ must remain non-authoritative.")
    roles = authority.get("roles", {})
    if set(roles) != set(EXPECTED_ROLES):
        _error(errors, "ROLE_SET_CHANGED", "dataset_authority.json:roles", "Only TRAIN, VALIDATION and REAL_EVAL_DEVELOPMENT are allowed.")
    registry_path = repo_root / TA6_REL / "canonical_artifact_registry.json"
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _error(errors, "T_A6_REGISTRY_UNREADABLE", TA6_REL, str(exc))
        return
    for role, expected in EXPECTED_ROLES.items():
        item = roles.get(role, {})
        reg = registry.get("roles", {}).get(role, {})
        checks = {
            "canonical_rows": expected["rows"],
            "canonical_shape": [62, 80],
            "canonical_dtype": "float32_little_endian",
            "canonical_unit": "CELSIUS",
            "artifact_sha256": expected["sha256"],
            "provenance_sha256": expected["provenance_sha256"],
        }
        for key, value in checks.items():
            if item.get(key) != value or reg.get(key) != value:
                _error(errors, "DATASET_IDENTITY_MISMATCH", f"dataset_authority.json:roles.{role}.{key}", f"expected {value!r}")
        for key in ("source_domain", "source_split"):
            if item.get(key) != expected[key] or reg.get(key) != expected[key]:
                _error(errors, "DATASET_ROLE_MISMATCH", f"dataset_authority.json:roles.{role}.{key}", f"expected {expected[key]!r}")
    if not (repo_root / MODEL_PATH).is_file():
        _error(errors, "LEGACY_MODEL_MISSING", MODEL_PATH, "Tracked legacy model artifact is required for T-B0 audit.")


def _validate_target(documents: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    target = documents.get("target_contract.json", {})
    if target.get("class_order") != ["NOT_HUMAN", "HUMAN_NORMAL", "HUMAN_FALL"]:
        _error(errors, "CLASS_ORDER_INVALID", "target_contract.json:class_order", "Frozen compatibility class order changed.")
    labels = target.get("label_layers", {})
    if labels.get("source_labels_immutable") is not True or labels.get("proxy_labels_separate") is not True or labels.get("proxy_overwrites_source") is not False:
        _error(errors, "SOURCE_PROXY_SEPARATION_INVALID", "target_contract.json:label_layers", "Source labels and compatibility proxies must remain separate.")
    lying = target.get("compatibility_proxy", {}).get("LYING", {})
    if lying.get("derived_target") != "HUMAN_FALL" or lying.get("mapping_type") != "DERIVED_POSTURE_PROXY":
        _error(errors, "FALL_PROXY_SEMANTICS_INVALID", "target_contract.json:compatibility_proxy.LYING", "LYING must remain an explicit derived posture proxy.")
    if target.get("temporal_event_ground_truth") != "NOT_VERIFIABLE":
        _error(errors, "TEMPORAL_CLAIM_ESCALATED", "target_contract.json:temporal_event_ground_truth", "T-B0 cannot establish temporal fall ground truth.")


def _validate_runtime_and_model(documents: Mapping[str, Any], repo_root: Path, errors: list[dict[str, str]]) -> None:
    inventory = documents.get("existing_model_inventory.json", {})
    if inventory.get("t_b0_classification") != "LEGACY_DEPLOYED_REFERENCE":
        _error(errors, "LEGACY_MODEL_MISCLASSIFIED", "existing_model_inventory.json:t_b0_classification", "Existing model must remain a legacy reference.")
    if inventory.get("canonical_t_a6_trained_claim_allowed") is not False:
        _error(errors, "LEGACY_LINEAGE_ESCALATED", "existing_model_inventory.json", "T-A6 canonical training provenance cannot be assigned retroactively.")
    artifact = inventory.get("artifact", {})
    model_path = repo_root / MODEL_PATH
    if artifact.get("path") != MODEL_PATH or artifact.get("sha256") != EXPECTED_MODEL_SHA:
        _error(errors, "MODEL_IDENTITY_DECLARATION_INVALID", "existing_model_inventory.json:artifact", "Legacy model path/SHA declaration is invalid.")
    if model_path.is_file():
        if sha256_file(model_path) != EXPECTED_MODEL_SHA:
            _error(errors, "MODEL_SHA_MISMATCH", MODEL_PATH, "Measured legacy model SHA differs from pinned identity.")
        if model_path.stat().st_size != 318184:
            _error(errors, "MODEL_SIZE_MISMATCH", MODEL_PATH, "Measured legacy model size differs from pinned identity.")
    audit = documents.get("existing_runtime_preprocessing_audit.json", {})
    source = (repo_root / "inference/thermal_interpreter.py").read_text(encoding="utf-8")
    required_tokens = ["_prepare_float_frame", "array.min()", "array.max()", "(array - min_value) / range_val", "np.rint", "np.clip"]
    for token in required_tokens:
        if token not in source:
            _error(errors, "RUNTIME_AUDIT_NOT_REPRODUCIBLE", "inference/thermal_interpreter.py", f"Missing audited token: {token}")
    if audit.get("audit_status") != "COMPLETE_WITH_COMPATIBILITY_LIMITATION" or audit.get("per_frame") is not True:
        _error(errors, "RUNTIME_AUDIT_STATUS_INVALID", "existing_runtime_preprocessing_audit.json", "Runtime per-frame behavior must be explicitly audited.")
    if audit.get("absolute_temperature_preserved") is not False or audit.get("canonical_celsius_directly_consumed") is not False:
        _error(errors, "RUNTIME_CELSIUS_CLAIM_INVALID", "existing_runtime_preprocessing_audit.json", "Existing runtime cannot be reported as preserving Celsius.")


def _validate_preprocessing(documents: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    registry = documents.get("preprocessing_candidate_registry.json", {})
    profiles = registry.get("candidate_profiles", [])
    ids = [item.get("profile_id") for item in profiles]
    expected = ["P0_CANONICAL_CELSIUS_DIRECT", "P1_TRAIN_FITTED_GLOBAL_ZSCORE", "P2_LEGACY_PER_FRAME_MINMAX"]
    if ids != expected:
        _error(errors, "PREPROCESSING_PROFILE_SET_INVALID", "preprocessing_candidate_registry.json:candidate_profiles", f"expected ordered profiles {expected!r}")
    by_id = {item.get("profile_id"): item for item in profiles}
    if by_id.get("P0_CANONICAL_CELSIUS_DIRECT", {}).get("fit_required") is not False or by_id.get("P0_CANONICAL_CELSIUS_DIRECT", {}).get("absolute_temperature_preservation") is not True:
        _error(errors, "P0_PROFILE_INVALID", "preprocessing_candidate_registry.json:P0_CANONICAL_CELSIUS_DIRECT", "P0 must be direct physical Celsius with no fit.")
    p1 = by_id.get("P1_TRAIN_FITTED_GLOBAL_ZSCORE", {})
    if p1.get("fit_required") is not True or p1.get("fit_partition") != "TRAIN_ONLY":
        _error(errors, "TRAIN_ONLY_FIT_RULE_INVALID", "preprocessing_candidate_registry.json:P1_TRAIN_FITTED_GLOBAL_ZSCORE", "P1 statistics must fit on TRAIN only.")
    p2 = by_id.get("P2_LEGACY_PER_FRAME_MINMAX", {})
    if p2.get("fit_required") is not False or p2.get("status") != "REGISTERED_COMPATIBILITY_PROFILE_ONLY" or p2.get("absolute_temperature_preservation") is not False:
        _error(errors, "P2_PROFILE_INVALID", "preprocessing_candidate_registry.json:P2_LEGACY_PER_FRAME_MINMAX", "P2 must remain compatibility-only and relative.")
    rule = registry.get("fit_data_rule", {})
    if rule.get("validation_fit_prohibited") is not True or rule.get("real_eval_development_fit_prohibited") is not True or rule.get("locked_test_fit_prohibited") is not True:
        _error(errors, "FIT_PARTITION_GUARD_INVALID", "preprocessing_candidate_registry.json:fit_data_rule", "Non-TRAIN fitting must be prohibited.")


def _validate_roles_metrics_and_near_duplicates(documents: Mapping[str, Any], errors: list[dict[str, str]], warnings: list[dict[str, str]]) -> None:
    roles = documents.get("evaluation_role_policy.json", {}).get("roles", {})
    if roles.get("TRAIN", {}).get("preprocessing_fit") is not True or roles.get("TRAIN", {}).get("weight_fitting") is not True:
        _error(errors, "TRAIN_ROLE_INVALID", "evaluation_role_policy.json:roles.TRAIN", "TRAIN must authorize fitting.")
    if roles.get("VALIDATION", {}).get("preprocessing_fit") is not False or roles.get("VALIDATION", {}).get("winner_selection") is not True:
        _error(errors, "VALIDATION_ROLE_INVALID", "evaluation_role_policy.json:roles.VALIDATION", "VALIDATION must be selection-only and never fit preprocessing.")
    real = roles.get("REAL_EVAL_DEVELOPMENT", {})
    if real.get("preprocessing_fit") is not False or real.get("model_selection") is not False or real.get("winner_selection") is not False or real.get("final_test") is not False:
        _error(errors, "REAL_ROLE_ESCALATION", "evaluation_role_policy.json:roles.REAL_EVAL_DEVELOPMENT", "REAL must not fit, tune, select or become final test.")
    near = documents.get("near_duplicate_sensitivity_policy.json", {})
    counts = near.get("measured_counts_in_t_a6", {})
    if counts.get("train_validation_confirmed_pairs") != 14514:
        _error(errors, "NEAR_DUPLICATE_COUNT_ERASED", "near_duplicate_sensitivity_policy.json:measured_counts_in_t_a6", "TRAIN-VALIDATION near-duplicate count must remain 14,514.")
    frozen = near.get("frozen_source_profile", {})
    if frozen.get("profile_id") != "THERMAL_T_A6_NEAR_DUPLICATE_SCREEN_V1" or frozen.get("exhaustiveness_claim") != "DETERMINISTIC_SCREENING_NOT_EXHAUSTIVE":
        _error(errors, "NEAR_DUPLICATE_PROFILE_ESCALATED", "near_duplicate_sensitivity_policy.json:frozen_source_profile", "Frozen T-A6 near-duplicate limitation must be preserved.")
    v1 = near.get("evaluation_views", {}).get("V1_NEAR_DUPLICATE_SENSITIVITY", {})
    if v1.get("split_role_changed") is not False or v1.get("winner_selection_allowed") is not False or v1.get("materialization_status") != "SENSITIVITY_SUBSET_NOT_MATERIALIZABLE_FROM_CURRENT_COMPACT_EVIDENCE":
        _error(errors, "NEAR_DUPLICATE_SENSITIVITY_POLICY_INVALID", "near_duplicate_sensitivity_policy.json:evaluation_views.V1_NEAR_DUPLICATE_SENSITIVITY", "Sensitivity view must remain bounded and non-selective.")
    _warning(warnings, "NO_PRISTINE_LOCKED_TEST", "evaluation_role_policy.json", "REAL_EVAL_DEVELOPMENT is not an untouched final test.")
    _warning(warnings, "GROUPING_NOT_VERIFIABLE", "dataset_authority.json", "Subject/session/event generalization remains unavailable from source provenance.")
    _warning(warnings, "NEAR_DUPLICATE_OVERLAP", "near_duplicate_sensitivity_policy.json", "14,514 TRAIN-VALIDATION near-duplicate pairs require disclosure.")
    metric = documents.get("metric_policy.json", {})
    if metric.get("class_order") != ["NOT_HUMAN", "HUMAN_NORMAL", "HUMAN_FALL"] or metric.get("sample_weighting") != "NONE_UNWEIGHTED" or metric.get("metric_set_frozen_before_training") is not True:
        _error(errors, "METRIC_POLICY_INVALID", "metric_policy.json", "Metric order, weighting or freeze status is invalid.")
    required_metrics = {"macro_f1", "accuracy", "balanced_accuracy", "per_class_precision", "per_class_recall", "per_class_f1", "confusion_matrix"}
    if set(metric.get("metrics", {})) != required_metrics:
        _error(errors, "METRIC_SET_INVALID", "metric_policy.json:metrics", "Required metric set changed.")


def _validate_winner_repro_budget(documents: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    winner = documents.get("winner_selection_policy.json", {})
    if winner.get("evaluation_role") != "VALIDATION" or winner.get("primary_metric") != "macro_f1" or winner.get("rule_frozen_before_training") is not True:
        _error(errors, "WINNER_RULE_INVALID", "winner_selection_policy.json", "Winner selection must be validation macro-F1 and preregistered.")
    if winner.get("real_evaluation", {}).get("used_for_winner_selection") is not False or winner.get("real_evaluation", {}).get("used_as_tie_breaker") is not False:
        _error(errors, "REAL_WINNER_SELECTION_ESCALATION", "winner_selection_policy.json:real_evaluation", "REAL cannot select or tie-break a winner.")
    if winner.get("tie_tolerance") != 1e-5:
        _error(errors, "TIE_RULE_INVALID", "winner_selection_policy.json:tie_tolerance", "Tie tolerance must remain 1e-5.")
    randomness = documents.get("randomness_policy.json", {})
    if randomness.get("primary_seed") != 20260813:
        _error(errors, "PRIMARY_SEED_INVALID", "randomness_policy.json:primary_seed", "Primary seed changed.")
    bindings = randomness.get("seed_bindings", {})
    if set(bindings) != {"python", "numpy", "tensorflow", "data_shuffle", "weight_initialization"} or len(set(bindings.values())) != 1 or next(iter(bindings.values()), None) != 20260813:
        _error(errors, "SEED_BINDING_INCOMPLETE", "randomness_policy.json:seed_bindings", "All primary randomness bindings must use the canonical seed.")
    multi = randomness.get("multi_seed_policy", {})
    if multi.get("minimum_seed_count") != 3 or multi.get("executed_in_t_b0") is not False:
        _error(errors, "MULTI_SEED_POLICY_INVALID", "randomness_policy.json:multi_seed_policy", "Later multi-seed policy is incomplete or was executed early.")
    budget = documents.get("training_budget_policy.json", {})
    baseline = budget.get("baseline_budget", {})
    if baseline.get("maximum_epochs") != 20 or baseline.get("batch_size") != 64 or baseline.get("optimizer") != "Adam" or baseline.get("maximum_tuning_trials_per_candidate_profile") != 1:
        _error(errors, "TRAINING_BUDGET_INVALID", "training_budget_policy.json:baseline_budget", "Fair baseline budget changed.")
    if budget.get("training_executed_in_t_b0") is not False or budget.get("augmentation", {}).get("baseline") != "DISABLED" or budget.get("class_imbalance", {}).get("baseline") != "UNWEIGHTED_CROSS_ENTROPY":
        _error(errors, "T_B0_TRAINING_SCOPE_INVALID", "training_budget_policy.json", "T-B0 must not train or silently add augmentation/imbalance handling.")
    candidates = documents.get("model_candidate_registry.json", {})
    if candidates.get("all_new_artifacts_absent") is not True:
        _error(errors, "NEW_MODEL_ARTIFACT_CLAIM", "model_candidate_registry.json", "T-B0 must not generate a new trained model.")
    for item in candidates.get("candidates", []):
        if item.get("role") != "LEGACY_REFERENCE" and item.get("artifact_path") is not None:
            _error(errors, "UNTRAINED_CANDIDATE_ARTIFACT_PRESENT", f"model_candidate_registry.json:{item.get('candidate_id')}", "Trainable candidates must have no artifact in T-B0.")


def _validate_deployment_license_and_isolation(documents: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    deployment = documents.get("deployment_measurement_policy.json", {})
    if deployment.get("measurements_available_in_t_b0") is not False or deployment.get("tflite_readiness", {}).get("conversion_in_t_b0") is not False:
        _error(errors, "DEPLOYMENT_EARLY_CLAIM", "deployment_measurement_policy.json", "T-B0 cannot claim deployment measurements or conversions.")
    license_doc = documents.get("license_boundary.json", {})
    if license_doc.get("license_status") != "VERIFIED_ACCEPTABLE_WITH_NONCOMMERCIAL_RESEARCH_RESTRICTION" or license_doc.get("required_attribution") is not True:
        _error(errors, "LICENSE_BOUNDARY_INVALID", "license_boundary.json", "Non-commercial attribution boundary must remain explicit.")
    for name, value in documents.items():
        for location, item in _walk(value, name):
            if isinstance(item, str) and any(token in item.lower() for token in ("co2/", "mmwave/", "integration/")):
                _error(errors, "CROSS_TRACK_REFERENCE", location, item)
            if isinstance(item, str) and item.lower() in {"pass", "selected", "winner"} and "winner" in location.lower():
                _error(errors, "WINNER_RESULT_PRESENT", location, "T-B0 cannot declare a winner.")


def _validate_checksums(evidence_dir: Path, errors: list[dict[str, str]], check_validation_result: bool) -> None:
    checksum_path = evidence_dir / CHECKSUMS_NAME
    if not checksum_path.is_file():
        _error(errors, "CHECKSUM_REGISTRY_MISSING", CHECKSUMS_NAME, "T-B0 checksum registry is missing.")
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
        expected_prefix = f"{EVIDENCE_REL}/"
        if not relative.startswith(expected_prefix):
            _error(errors, "CHECKSUM_SCOPE_INVALID", f"{CHECKSUMS_NAME}:{number}", relative)
            continue
        if not _portable_string(relative) or ".." in PurePosixPath(relative).parts:
            _error(errors, "CHECKSUM_PATH_NOT_PORTABLE", f"{CHECKSUMS_NAME}:{number}", relative)
        entries[relative] = digest
    for name in REQUIRED_JSON:
        path = evidence_dir / name
        relative = f"{EVIDENCE_REL}/{name}"
        if not path.is_file():
            continue
        if relative not in entries:
            _error(errors, "CHECKSUM_COVERAGE_MISSING", relative, "Required JSON artifact is absent from checksum registry.")
        elif name != "validation_result.json" or check_validation_result:
            if entries[relative] != sha256_file(path):
                _error(errors, "CHECKSUM_MISMATCH", relative, "Checksum is stale or incorrect.")
    if set(entries) != {f"{EVIDENCE_REL}/{name}" for name in REQUIRED_JSON}:
        _error(errors, "CHECKSUM_ARTIFACT_SET_INVALID", CHECKSUMS_NAME, "Checksum registry must cover exactly all T-B0 JSON artifacts.")


def validate_evidence(*, repo_root: Path = ROOT, evidence_dir: Path | None = None, check_checksums: bool = True) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    evidence_dir = (evidence_dir or repo_root / EVIDENCE_REL).resolve()
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    documents = _load_documents(evidence_dir, errors)
    ta6 = _validate_a6(repo_root, errors)
    if all(name in documents for name in REQUIRED_JSON):
        _validate_dataset_authority(documents, repo_root, errors)
        _validate_target(documents, errors)
        _validate_runtime_and_model(documents, repo_root, errors)
        _validate_preprocessing(documents, errors)
        _validate_roles_metrics_and_near_duplicates(documents, errors, warnings)
        _validate_winner_repro_budget(documents, errors)
        _validate_deployment_license_and_isolation(documents, errors)
    if check_checksums:
        stored = documents.get("validation_result.json", {})
        _validate_checksums(evidence_dir, errors, check_validation_result=stored.get("evidence_validation") == "PASS")
    errors = sorted(errors, key=lambda item: (item["code"], item["location"], item["message"]))
    warnings = sorted(warnings, key=lambda item: (item["code"], item["location"], item["message"]))
    gate = not errors and ta6.get("evidence_validation") == "PASS"
    return {
        "evidence_validation": "PASS" if gate else "FAIL",
        "error_count": len(errors),
        "errors": errors,
        "full_training_performed": False,
        "new_trained_model_generated": False,
        "overall_outcome": "PASS_WITH_LIMITATIONS" if gate else "T_B0_BLOCKED",
        "phase": "T-B0",
        "schema_version": "1.0",
        "t_a6_gate": ta6.get("full_t_a6_gate", "NOT_VERIFIABLE"),
        "t_b0_gate": "T_B0_COMPLETE_WITH_LIMITATIONS" if gate else "T_B0_BLOCKED",
        "t_b1_authorized": "YES_WITH_LIMITATIONS" if gate else "NO",
        "warning_count": len(warnings),
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--evidence-dir", type=Path, default=None)
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--no-write-result", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    evidence_dir = (args.evidence_dir or repo_root / EVIDENCE_REL).resolve()
    result = validate_evidence(repo_root=repo_root, evidence_dir=evidence_dir, check_checksums=not args.skip_checksums)
    if not args.no_write_result:
        (evidence_dir / "validation_result.json").write_text(canonical_json(result), encoding="utf-8")
    print(canonical_json(result), end="")
    return 0 if result["evidence_validation"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
