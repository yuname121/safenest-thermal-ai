#!/usr/bin/env python3
"""Standalone validator for the Thermal T-B3 frame-only multi-seed contract."""

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

from datasets.thermal.t_b1_model import BASELINE_ID  # noqa: E402
from datasets.thermal.t_b1_preprocessing import CLASS_ORDER, P1Statistics, canonical_json  # noqa: E402


PHASE = "T-B3"
EVIDENCE_REL = "datasets/thermal/manifests/T-B3_frame_multiseed_confirmation"
TA3_REL = "datasets/thermal/manifests/T-A3_sequence_window_event_policy"
TA6_REL = "datasets/thermal/manifests/T-A6_execution_result"
TB0_REL = "datasets/thermal/manifests/T-B0_offline_model_protocol"
TB1_REL = "datasets/thermal/manifests/T-B1_full_experiment"
TB2_REL = "datasets/thermal/manifests/T-B2_architecture_comparison"
ROADMAP_REL = "docs/20260810_ChatGPT_SafeNest_Multisensor_Parallel_Execution_Roadmap_01.md"
RECONCILIATION_REL = "docs/reports/20260814_Codex_Thermal_Post_T-B2_Pre_T-B3_Reconciliation_01.md"
CHECKSUMS = "checksums.sha256"
SEEDS = (20260813, 20260814, 20260815)
CLASS_ORDER_LIST = list(CLASS_ORDER)
P1_PROFILE = "P1_TRAIN_FITTED_GLOBAL_ZSCORE"
EXPECTED_P1_MEAN = 22.769290618485442
EXPECTED_P1_STD = 2.8684523405441222
EXPECTED_P1_CHECKSUM = "10b5da044ef33f26715544c3d4bf56e9d999d6c65139e931f6997583b3f5b816"
EXPECTED_ARCH = "937fceb88900a779d67a8e407cebc3362cc21346721f92cea5ae8de1413aba2a"
EXPECTED_PARAMS = 312131
EXPECTED_NEAR_DUPLICATES = 14514
EXPECTED_REAL_F1 = 0.593926523563344
EXPECTED_VAL_F1 = 0.9951295332536425
EXPECTED_TRAIN = (32000, "749c847fc9ab50ea5eee8827f0d47b5ebaa48165732a382c59f8b96c565b9d93")
EXPECTED_VALIDATION = (8000, "5d16451702c1bccfa945d9188d9b29a26ce11c8b33bf7a0dfbb25bfa86d74610")
EXPECTED_REAL = (8000, "cd696e68aeec063cbc8185719b4f4dad3d038cb3d28eec0d3701b8311e4ad8f1")
EXPECTED_CHECKPOINT = "7aba32fe8d0e241546429bdc3e8cd059b10d4d8f548e9e12c4085abeba308a75"
EXPECTED_CHECKPOINT_SIZE = 3777416
BASE_JSON = (
    "t_b3_protocol.json", "predecessor_identity.json", "dataset_lock.json",
    "p1_lock.json", "architecture_lock.json", "seed_registry.json",
    "seed_20260813_reuse_assessment.json", "candidate_checkpoint_policy.json",
    "environment.json", "readiness_result.json",
)
FULL_JSON = BASE_JSON + (
    "execution_environment.json", "seed_20260813_summary.json",
    "seed_20260814_summary.json", "seed_20260815_summary.json",
    "multiseed_aggregate.json", "limitations.json", "execution_summary.json",
)


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
    lower = value.lower()
    if value.startswith("/physical_device:"):
        return True
    return not (
        value.startswith(("/", "~/", "file://"))
        or "\\" in value
        or "/users/" in lower
        or "/private/" in lower
        or value.startswith(("/volumes/", "/content/"))
    )


def _read_documents(evidence: Path, names: Iterable[str], errors: list[dict[str, str]]) -> dict[str, Any]:
    documents: dict[str, Any] = {}
    for name in names:
        path = evidence / name
        if not path.is_file():
            _error(errors, "REQUIRED_ARTIFACT_MISSING", name, "Required compact T-B3 artifact is missing.")
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
            if not isinstance(item, str):
                continue
            if not _portable(item):
                _error(errors, "NONPORTABLE_PATH", location, item)
            if item.startswith("archive/") or "/archive/" in item:
                _error(errors, "ARCHIVE_TREATED_AS_ACTIVE", location, item)
            if any(token in item.lower() for token in ("/co2/", "/mmwave/", "/integration/", "ondevice_ai/")):
                _error(errors, "CROSS_TRACK_REFERENCE", location, item)
    return documents


def _validate_checksums(evidence: Path, required: set[str], errors: list[dict[str, str]]) -> None:
    path = evidence / CHECKSUMS
    if not path.is_file():
        _error(errors, "CHECKSUM_REGISTRY_MISSING", CHECKSUMS, "T-B3 checksum registry is required.")
        return
    entries: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            _error(errors, "CHECKSUM_FORMAT_INVALID", f"{CHECKSUMS}:{number}", line)
            continue
        digest, relative = parts
        if relative.startswith(("/", "~/", "file://")) or "\\" in relative or relative.startswith("archive/"):
            _error(errors, "CHECKSUM_PATH_INVALID", f"{CHECKSUMS}:{number}", relative)
            continue
        if relative in entries:
            _error(errors, "CHECKSUM_DUPLICATE", relative, "Duplicate checksum path.")
        entries[relative] = digest
        target = evidence / relative
        if not target.is_file():
            _error(errors, "CHECKSUM_TARGET_MISSING", relative, "Checksum target is missing.")
        elif sha256_file(target) != digest:
            _error(errors, "CHECKSUM_MISMATCH", relative, "Checksum is stale or incorrect.")
    if not required.issubset(entries):
        _error(errors, "CHECKSUM_COVERAGE_INCOMPLETE", CHECKSUMS, f"Missing: {sorted(required - set(entries))}")
    if list(entries) != sorted(entries):
        _error(errors, "CHECKSUM_ORDER_NONDETERMINISTIC", CHECKSUMS, "Checksum paths must be sorted.")
    allowed = required | {"validation_result.json"}
    extra = sorted(set(entries) - allowed)
    if extra:
        _error(errors, "CHECKSUM_EXTRA_ARTIFACT", CHECKSUMS, f"Unexpected entries: {extra}")


def _validate_predecessors(repo_root: Path, errors: list[dict[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        from scripts.validate_thermal_t_a6 import validate_evidence as a6
        from scripts.validate_thermal_t_b0 import validate_evidence as b0
        from scripts.validate_thermal_t_b1 import validate_evidence as b1
        from scripts.validate_thermal_t_b2 import validate_evidence as b2
        result["T-A6"] = a6(repo_root=repo_root, evidence_dir=repo_root / TA6_REL, mode="FULL_DATASET", check_checksums=True)
        result["T-B0"] = b0(repo_root=repo_root, evidence_dir=repo_root / TB0_REL, check_checksums=True)
        result["T-B1"] = b1(repo_root=repo_root, evidence_dir=repo_root / TB1_REL, mode="FULL_EXPERIMENT", check_checksums=True)
        result["T-B2"] = b2(repo_root=repo_root, evidence_dir=repo_root / TB2_REL, mode="FULL_EXPERIMENT", check_checksums=True)
    except Exception as exc:  # pragma: no cover - defensive boundary
        _error(errors, "PREDECESSOR_VALIDATOR_ERROR", PHASE, str(exc))
        return result
    for phase, item in result.items():
        if item.get("evidence_validation") != "PASS":
            _error(errors, "PREDECESSOR_LIVE_INVALID", phase, str(item.get("overall_outcome")))
    return result


def _validate_roadmap(repo_root: Path, errors: list[dict[str, str]], warnings: list[dict[str, str]]) -> None:
    roadmap = repo_root / ROADMAP_REL
    reconciliation = repo_root / RECONCILIATION_REL
    if not reconciliation.is_file():
        _error(errors, "ROADMAP_RECONCILIATION_MISSING", PHASE, "Merged T-B3 reconciliation evidence is required.")
        return
    # The focused derivative intentionally carries phase evidence and reports,
    # not every historical team planning document.  The reconciliation report
    # is authoritative here and already records the relevant roadmap decision.
    text = reconciliation.read_text(encoding="utf-8")
    if roadmap.is_file():
        text = roadmap.read_text(encoding="utf-8") + "\n" + text
    else:
        _warning(warnings, "HISTORICAL_ROADMAP_NOT_CURATED", ROADMAP_REL, "Focused repository retains the merged reconciliation report but excludes this historical planning document.")
    text = text.lower()
    for phrase in ("controlled frame-architecture comparison", "frame-only multi-seed", "temporal comparison remains deferred", "small_cnn_baseline_v1"):
        if phrase not in text:
            _error(errors, "ROADMAP_RECONCILIATION_INCOMPLETE", phrase, "Required post-T-B2 reconciliation phrase is absent.")


def _validate_predecessor_identity(doc: Mapping[str, Any], live: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    if doc.get("phase") != PHASE or not doc.get("roadmap_reconciliation", {}).get("reconciliation_present"):
        _error(errors, "PREDECESSOR_IDENTITY_INVALID", "predecessor_identity.json", "T-B3 predecessor identity is not frozen.")
    for phase in ("T-A6", "T-B0", "T-B1", "T-B2"):
        snap = doc.get("validators", {}).get(phase, {})
        if snap.get("evidence_validation") != "PASS" or live.get(phase, {}).get("evidence_validation") != "PASS":
            _error(errors, "PREDECESSOR_SNAPSHOT_INVALID", phase, "Predecessor snapshot/live validation is not PASS.")
    winner = doc.get("required_winner", {})
    if winner.get("candidate_id") != BASELINE_ID or winner.get("architecture_fingerprint") != EXPECTED_ARCH or winner.get("parameter_count") != EXPECTED_PARAMS:
        _error(errors, "WINNER_IDENTITY_INVALID", "predecessor_identity.json:required_winner", "T-B2 winner identity changed.")
    if doc.get("temporal_feasibility") != "NOT_SUPPORTED_BY_CURRENT_DATASET_PROVENANCE":
        _error(errors, "TEMPORAL_FEASIBILITY_CLAIM", "predecessor_identity.json", "Temporal provenance must remain unsupported.")


def _validate_dataset(doc: Mapping[str, Any], errors: list[dict[str, str]], warnings: list[dict[str, str]]) -> None:
    expected = {"TRAIN": EXPECTED_TRAIN, "VALIDATION": EXPECTED_VALIDATION, "REAL_EVAL_DEVELOPMENT": EXPECTED_REAL}
    roles = doc.get("roles", {})
    if set(roles) != set(expected):
        _error(errors, "DATASET_ROLE_SET_INVALID", "dataset_lock.json:roles", "Exactly the three canonical roles are required.")
    for role, (rows, digest) in expected.items():
        item = roles.get(role, {})
        if item.get("rows") != rows or item.get("sha256") != digest or item.get("dtype") != "float32_little_endian" or item.get("unit") != "CELSIUS":
            _error(errors, "CANONICAL_IDENTITY_INVALID", f"dataset_lock.json:roles.{role}", "Canonical role identity changed.")
    if roles.get("REAL_EVAL_DEVELOPMENT", {}).get("source_domain") != "REAL":
        _error(errors, "REAL_DOMAIN_INVALID", "dataset_lock.json:roles.REAL_EVAL_DEVELOPMENT", "REAL role must remain REAL.")
    for key in ("legacy_npz_used", "raw_zip_used", "new_split_created", "temporal_grouping"):
        if doc.get(key) is not False:
            _error(errors, "DATASET_SCOPE_ESCALATION", f"dataset_lock.json:{key}", "Legacy/raw/resplit/temporal grouping is prohibited.")
    if doc.get("official_partition_preservation") is not True or doc.get("near_duplicate_pairs_train_validation") != EXPECTED_NEAR_DUPLICATES:
        _error(errors, "DATASET_LIMITATION_LOST", "dataset_lock.json", "Partition/near-duplicate limitation changed.")
    _warning(warnings, "NEAR_DUPLICATE_OVERLAP", "dataset_lock.json", "14,514 TRAIN-VALIDATION near-duplicate pairs remain disclosed.")


def _validate_p1(doc: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    if doc.get("profile_id") != P1_PROFILE or doc.get("fit_role") != "TRAIN" or doc.get("statistics_checksum") != EXPECTED_P1_CHECKSUM:
        _error(errors, "P1_IDENTITY_INVALID", "p1_lock.json", "P1 identity or checksum differs from T-B1.")
    try:
        stats = P1Statistics(mean=float(doc["mean"]), std=float(doc["std"]), fit_sample_count=int(doc["fit_sample_count"]), fit_pixel_count=int(doc["fit_pixel_count"]), fit_role=str(doc["fit_role"]), train_artifact_sha256=str(doc["train_artifact_sha256"]), epsilon=float(doc.get("epsilon", 1e-6)))
        if stats.checksum() != EXPECTED_P1_CHECKSUM or not math.isclose(stats.mean, EXPECTED_P1_MEAN, abs_tol=1e-12) or not math.isclose(stats.std, EXPECTED_P1_STD, abs_tol=1e-12):
            _error(errors, "P1_STATISTICS_MISMATCH", "p1_lock.json", "Serialized P1 statistics do not match the frozen T-B1 values.")
    except (KeyError, TypeError, ValueError) as exc:
        _error(errors, "P1_STATISTICS_INVALID", "p1_lock.json", str(exc))
    for key in ("refit", "validation_fit", "real_fit"):
        if doc.get(key) is not False:
            _error(errors, "P1_REFIT_POLICY_INVALID", f"p1_lock.json:{key}", "P1 statistics may not be refit.")


def _validate_protocol(doc: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    if doc.get("phase") != PHASE or doc.get("protocol_id") != "THERMAL_T_B3_FRAME_ONLY_MULTI_SEED_CONFIRMATION_001":
        _error(errors, "PROTOCOL_ID_INVALID", "t_b3_protocol.json", "T-B3 protocol identity is invalid.")
    if doc.get("factor_changed") != "SEED_ONLY" or doc.get("scope") != "FROZEN_P1_SMALL_CNN_FRAME_ONLY_STABILITY_CONFIRMATION":
        _error(errors, "EXPERIMENT_SCOPE_INVALID", "t_b3_protocol.json", "Seed must be the only changed factor.")
    if doc.get("seeds") != list(SEEDS) or doc.get("extra_seeds") != [] or doc.get("selection_role") != "VALIDATION":
        _error(errors, "SEED_SET_INVALID", "t_b3_protocol.json", "The frozen seed set/order is invalid.")
    if doc.get("real_policy") != "EXISTING_REFERENCE_ONLY_NOT_REPEATED" or doc.get("real_used_for_selection") not in (None, False):
        _error(errors, "REAL_SELECTION_POLICY_INVALID", "t_b3_protocol.json", "REAL must not be used for new seed selection.")
    for key in ("best_seed_cherry_picking",):
        if doc.get(key) != "PROHIBITED":
            _error(errors, "CHECKPOINT_SELECTION_POLICY_INVALID", f"t_b3_protocol.json:{key}", "Post-hoc best-seed selection is prohibited.")
    if doc.get("temporal_training") != "PROHIBITED" or doc.get("t_b4_started") is not False or doc.get("next_phase_started") is not False:
        _error(errors, "SCOPE_ESCALATION", "t_b3_protocol.json", "Temporal/T-B4 work entered the T-B3 run.")


def _validate_architecture(doc: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    if doc.get("candidate_id") != BASELINE_ID or doc.get("architecture_fingerprint") != EXPECTED_ARCH or doc.get("parameter_count") != EXPECTED_PARAMS or doc.get("modified") is not False or doc.get("temporal_layers") is not False:
        _error(errors, "ARCHITECTURE_LOCK_INVALID", "architecture_lock.json", "SMALL_CNN baseline lock changed.")


def _validate_reuse(doc: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    if doc.get("seed") != 20260813 or doc.get("source") != "REUSED_VERIFIED_T_B1_RESULT" or doc.get("eligible") is not True or doc.get("retrained") is not False:
        _error(errors, "SEED_REUSE_INVALID", "seed_20260813_reuse_assessment.json", "Seed 20260813 reuse is not proven.")
    if doc.get("architecture_fingerprint") != EXPECTED_ARCH or doc.get("parameter_count") != EXPECTED_PARAMS or doc.get("p1_statistics_checksum") != EXPECTED_P1_CHECKSUM:
        _error(errors, "SEED_REUSE_CONTRACT_MISMATCH", "seed_20260813_reuse_assessment.json", "Reusable seed contract differs.")
    if any(value is not True for value in doc.get("contract_comparison", {}).values()) or len(doc.get("contract_comparison", {})) < 20:
        _error(errors, "SEED_REUSE_CONTRACT_MISMATCH", "seed_20260813_reuse_assessment.json:contract_comparison", "All frozen contract dimensions must match.")
    metrics = doc.get("validation_metrics", {})
    if not math.isclose(float(metrics.get("macro_f1", -1)), EXPECTED_VAL_F1, abs_tol=1e-12):
        _error(errors, "SEED_REUSE_METRIC_MISMATCH", "seed_20260813_reuse_assessment.json", "Inherited VALIDATION metric changed.")
    checkpoint = doc.get("checkpoint", {})
    if checkpoint.get("sha256") != EXPECTED_CHECKPOINT or checkpoint.get("size_bytes") != EXPECTED_CHECKPOINT_SIZE:
        _error(errors, "SEED_REUSE_CHECKPOINT_MISMATCH", "seed_20260813_reuse_assessment.json", "Inherited checkpoint identity changed.")


def _validate_seed_registry(doc: Mapping[str, Any], full: bool, errors: list[dict[str, str]]) -> None:
    if doc.get("required_seeds") != list(SEEDS) or doc.get("extra_seeds") != [] or doc.get("seed_set_frozen") is not True or doc.get("seed_selection_after_metrics") is not False:
        _error(errors, "SEED_REGISTRY_INVALID", "seed_registry.json", "Seed registry is not frozen.")
    expected_status = {"20260813": "REUSED_VERIFIED", "20260814": "FINALIZED", "20260815": "FINALIZED"} if full else {"20260813": "REUSE_ASSESSMENT_COMPLETE", "20260814": "NOT_STARTED", "20260815": "NOT_STARTED"}
    for seed, status in expected_status.items():
        if doc.get(f"seed_{seed}", {}).get("status") != status:
            _error(errors, "SEED_REGISTRY_STATUS_INVALID", f"seed_registry.json:seed_{seed}", "Seed status does not match the selected mode.")


def _validate_metric_consistency(metrics: Mapping[str, Any], location: str, errors: list[dict[str, str]]) -> None:
    if metrics.get("class_order") != CLASS_ORDER_LIST or metrics.get("sample_count") != 8000:
        _error(errors, "METRIC_CONTRACT_INVALID", location, "Class order or validation sample count changed.")
        return
    matrix = metrics.get("confusion_matrix")
    if not isinstance(matrix, list) or len(matrix) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in matrix):
        _error(errors, "CONFUSION_MATRIX_INVALID", location, "3x3 confusion matrix is required.")
        return
    matrix = [[int(value) for value in row] for row in matrix]
    if any(value < 0 for row in matrix for value in row) or sum(sum(row) for row in matrix) != 8000:
        _error(errors, "CONFUSION_MATRIX_INVALID", location, "Confusion matrix counts are invalid.")
    prediction = metrics.get("prediction_distribution", {})
    if sum(int(prediction.get(name, -1)) for name in CLASS_ORDER) != 8000:
        _error(errors, "PREDICTION_DISTRIBUTION_INVALID", location, "Prediction distribution must sum to 8000.")
    per_class = metrics.get("per_class", {})
    for index, name in enumerate(CLASS_ORDER):
        item = per_class.get(name, {})
        support = sum(matrix[index])
        if item.get("support") != support or int(prediction.get(name, -1)) != sum(matrix[row][index] for row in range(3)):
            _error(errors, "PER_CLASS_SUPPORT_INVALID", f"{location}:{name}", "Per-class support/prediction count disagrees with confusion matrix.")
        for key in ("precision", "recall", "f1"):
            if not isinstance(item.get(key), (float, int)) or not math.isfinite(float(item[key])):
                _error(errors, "PER_CLASS_METRIC_INVALID", f"{location}:{name}.{key}", "Per-class metric is missing or non-finite.")
    for key in ("macro_f1", "accuracy", "balanced_accuracy", "h_fall_posture_proxy_recall"):
        if not isinstance(metrics.get(key), (float, int)) or not math.isfinite(float(metrics[key])):
            _error(errors, "METRIC_MISSING", f"{location}:{key}", "Required validation metric is missing.")


def _validate_seed_summary(doc: Mapping[str, Any], seed: int, full: bool, errors: list[dict[str, str]]) -> None:
    location = f"seed_{seed}_summary.json"
    if doc.get("phase") != PHASE or doc.get("seed") != seed or doc.get("candidate_id") != BASELINE_ID or doc.get("profile_id") != P1_PROFILE or doc.get("architecture_fingerprint") != EXPECTED_ARCH or doc.get("parameter_count") != EXPECTED_PARAMS:
        _error(errors, "SEED_SUMMARY_CONTRACT_INVALID", location, "Seed summary does not use the frozen candidate contract.")
    if doc.get("finalized") is not True or doc.get("real_evaluated") is not False or doc.get("real_metric_for_selection") is not False:
        _error(errors, "SEED_SCOPE_INVALID", location, "Seed must be finalized on VALIDATION only; REAL is excluded.")
    if full:
        source = "REUSED_VERIFIED_T_B1_RESULT" if seed == 20260813 else "NEW_TRAINING"
        if doc.get("source") != source:
            _error(errors, "SEED_SOURCE_INVALID", location, "Seed source/reuse policy is invalid.")
        if not isinstance(doc.get("initialization_fingerprint"), str) or len(doc["initialization_fingerprint"]) != 64:
            _error(errors, "INITIALIZATION_FINGERPRINT_INVALID", location, "Initialization fingerprint is required.")
        checkpoint = doc.get("checkpoint", {})
        if not isinstance(checkpoint.get("sha256"), str) or len(checkpoint["sha256"]) != 64 or int(checkpoint.get("size_bytes", 0)) <= 0:
            _error(errors, "CHECKPOINT_IDENTITY_INVALID", location, "Checkpoint SHA/size identity is required.")
        if seed == 20260813 and (checkpoint.get("sha256") != EXPECTED_CHECKPOINT or checkpoint.get("size_bytes") != EXPECTED_CHECKPOINT_SIZE):
            _error(errors, "CHECKPOINT_IDENTITY_INVALID", location, "Inherited checkpoint identity changed.")
    _validate_metric_consistency(doc.get("validation_metrics", {}), f"{location}:validation_metrics", errors)


def _validate_aggregate(doc: Mapping[str, Any], summaries: list[Mapping[str, Any]], errors: list[dict[str, str]]) -> None:
    if doc.get("phase") != PHASE or doc.get("seeds") != list(SEEDS) or doc.get("seed_count") != 3 or doc.get("metric") != "VALIDATION" or doc.get("real_used_for_selection") is not False:
        _error(errors, "AGGREGATE_CONTRACT_INVALID", "multiseed_aggregate.json", "Aggregate contract is invalid.")
        return
    for metric_key, field in (("macro_f1", "macro_f1"), ("balanced_accuracy", "balanced_accuracy"), ("human_fall_posture_proxy_recall", "h_fall_posture_proxy_recall")):
        values = [float(row["validation_metrics"][field]) for row in summaries]
        block = doc.get(metric_key, {})
        if metric_key == "macro_f1":
            expected = {"mean": float(sum(values) / 3), "std": float(__import__("numpy").std(values, ddof=0)), "minimum": min(values), "maximum": max(values), "range": max(values) - min(values)}
        else:
            expected = {"mean": float(sum(values) / 3), "worst": min(values)}
        for key, value in expected.items():
            if not math.isclose(float(block.get(key, float("nan"))), value, rel_tol=0.0, abs_tol=1e-12):
                _error(errors, "AGGREGATE_RECOMPUTE_MISMATCH", f"multiseed_aggregate.json:{metric_key}.{key}", "Aggregate does not recompute from the three frozen seeds.")
    if doc.get("stability_threshold_predefined") is not False or doc.get("best_seed_cherry_picking") != "PROHIBITED":
        _error(errors, "AGGREGATE_POLICY_INVALID", "multiseed_aggregate.json", "No post-hoc threshold or best-seed selection is permitted.")


def _validate_limitations(doc: Mapping[str, Any], errors: list[dict[str, str]], warnings: list[dict[str, str]]) -> None:
    expected = {
        "near_duplicate_pairs_train_validation": EXPECTED_NEAR_DUPLICATES,
        "locked_test_available": False,
        "subject_independent": "NOT_VERIFIABLE",
        "session_independent": "NOT_VERIFIABLE",
        "event_independent": "NOT_VERIFIABLE",
        "temporal_fall": "NOT_VERIFIED",
        "multi_seed_real_evaluation": "NOT_PERFORMED",
        "best_seed_cherry_picking": "PROHIBITED",
        "next_phase_started": False,
        "device_domain_validation": "NOT_PERFORMED_DEFERRED_TO_T-C",
    }
    for key, value in expected.items():
        if doc.get(key) != value:
            _error(errors, "LIMITATION_LOST", f"limitations.json:{key}", f"Expected {value!r}.")
    if doc.get("human_fall_semantics") != "LYING_TO_HUMAN_FALL_DERIVED_POSTURE_PROXY_NOT_TEMPORAL_EVENT_GROUND_TRUTH":
        _error(errors, "TEMPORAL_FALL_CLAIM", "limitations.json:human_fall_semantics", "HUMAN_FALL remains a posture proxy.")
    if not math.isclose(float(doc.get("synthetic_real_gap", float("nan"))), EXPECTED_VAL_F1 - EXPECTED_REAL_F1, abs_tol=1e-12):
        _error(errors, "REAL_GAP_INVALID", "limitations.json:synthetic_real_gap", "Inherited development gap changed.")
    for code, message in (("NEAR_DUPLICATE_OVERLAP", "14,514 TRAIN-VALIDATION near-duplicate pairs remain disclosed."), ("GROUPING_NOT_VERIFIABLE", "Subject/session/event generalization remains unavailable."), ("NO_PRISTINE_LOCKED_TEST", "REAL_EVAL_DEVELOPMENT is not an untouched final test.")):
        _warning(warnings, code, "limitations.json", message)


def _validate_execution(documents: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    env = documents.get("execution_environment.json", {})
    summary = documents.get("execution_summary.json", {})
    if env.get("phase") != PHASE or env.get("temporal_training") is not False or env.get("gpu_required") is not False:
        _error(errors, "EXECUTION_ENVIRONMENT_INVALID", "execution_environment.json", "Execution environment/scope is invalid.")
    if summary.get("phase") != PHASE or summary.get("status") != "FINALIZED" or summary.get("mode") != "FULL_EXPERIMENT" or summary.get("seed_count") != 3 or summary.get("seeds") != list(SEEDS):
        _error(errors, "EXECUTION_SUMMARY_INVALID", "execution_summary.json", "Full experiment summary is invalid.")
    for key in ("real_evaluations",):
        if summary.get(key) != 0:
            _error(errors, "REAL_EVALUATION_PERFORMED", f"execution_summary.json:{key}", "T-B3 must not evaluate REAL for new seeds.")
    for key in ("next_phase_started", "t_b4_started", "candidate_changed"):
        if summary.get(key) is not False:
            _error(errors, "SCOPE_ESCALATION", f"execution_summary.json:{key}", "Later phase/candidate changes are prohibited.")


def _validate_self_report(path: Path, errors: list[dict[str, str]]) -> None:
    if not path.is_file():
        return
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _error(errors, "VALIDATION_RESULT_INVALID", "validation_result.json", str(exc))
        return
    if document.get("phase") != PHASE or document.get("evidence_validation") != "PASS" or document.get("overall_outcome") != "T_B3_COMPLETE_WITH_LIMITATIONS" or document.get("error_count") != 0 or document.get("full_experiment") != "FINALIZED":
        _error(errors, "VALIDATION_RESULT_INVALID", "validation_result.json", "Self-report is not a passing finalized T-B3 result.")


def _validate_candidate_policy(doc: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    if doc.get("phase") != PHASE or doc.get("reference_candidate_id") != BASELINE_ID or doc.get("reference_seed") != 20260813 or doc.get("best_seed_cherry_picking") != "PROHIBITED":
        _error(errors, "CHECKPOINT_POLICY_INVALID", "candidate_checkpoint_policy.json", "Reference candidate/checkpoint policy is invalid.")
    if doc.get("candidate_changed_due_to_t_b3") not in (None, False) or doc.get("candidate_replacement_allowed") not in (None, False):
        _error(errors, "CANDIDATE_REPLACEMENT", "candidate_checkpoint_policy.json", "T-B3 may not replace the candidate.")


def validate_evidence(*, repo_root: Path = ROOT, evidence_dir: Path | None = None, mode: str = "FULL_EXPERIMENT", check_checksums: bool = True) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    evidence = Path(evidence_dir or repo_root / EVIDENCE_REL).resolve()
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    full = mode == "FULL_EXPERIMENT"
    names = FULL_JSON if full else BASE_JSON
    documents = _read_documents(evidence, names, errors)
    live = _validate_predecessors(repo_root, errors)
    for phase in ("T-A6", "T-B0", "T-B1", "T-B2"):
        if live.get(phase, {}).get("evidence_validation") != "PASS":
            _error(errors, "PREDECESSOR_LIVE_INVALID", phase, "Live predecessor validator is not PASS.")
    _validate_roadmap(repo_root, errors, warnings)
    if all(name in documents for name in BASE_JSON):
        _validate_protocol(documents["t_b3_protocol.json"], errors)
        _validate_predecessor_identity(documents["predecessor_identity.json"], live, errors)
        _validate_dataset(documents["dataset_lock.json"], errors, warnings)
        _validate_p1(documents["p1_lock.json"], errors)
        _validate_architecture(documents["architecture_lock.json"], errors)
        _validate_reuse(documents["seed_20260813_reuse_assessment.json"], errors)
        _validate_seed_registry(documents["seed_registry.json"], full, errors)
        _validate_candidate_policy(documents["candidate_checkpoint_policy.json"], errors)
        readiness = documents["readiness_result.json"]
        if readiness.get("status") != "T_B3_MULTI_SEED_RUN_READY" or readiness.get("t_b4_started") is not False or readiness.get("temporal_disabled") is not True:
            _error(errors, "READINESS_CONTRACT_INVALID", "readiness_result.json", "Readiness gate is invalid.")
        if full:
            for seed in SEEDS:
                _validate_seed_summary(documents[f"seed_{seed}_summary.json"], seed, True, errors)
            _validate_aggregate(documents["multiseed_aggregate.json"], [documents[f"seed_{seed}_summary.json"] for seed in SEEDS], errors)
            _validate_limitations(documents["limitations.json"], errors, warnings)
            _validate_execution(documents, errors)
    if check_checksums:
        # validation_result.json is produced after the first successful pass and is
        # accepted as an optional self-report; every other compact artifact is mandatory.
        _validate_checksums(evidence, set(names), errors)
    _validate_self_report(evidence / "validation_result.json", errors)
    errors.sort(key=lambda item: (item["code"], item["location"], item["message"]))
    warnings.sort(key=lambda item: (item["code"], item["location"], item["message"]))
    passed = not errors and all(live.get(phase, {}).get("evidence_validation") == "PASS" for phase in ("T-A6", "T-B0", "T-B1", "T-B2"))
    return {
        "phase": PHASE,
        "mode": mode,
        "schema_version": "1.0",
        "evidence_validation": "PASS" if passed else "FAIL",
        "overall_outcome": "T_B3_COMPLETE_WITH_LIMITATIONS" if passed else "T_B3_BLOCKED",
        "full_experiment": "FINALIZED" if full and passed else ("READY" if not full and passed else "BLOCKED"),
        "t_b4_authorized": "YES_WITH_LIMITATIONS" if full and passed else False,
        "predecessors": {phase: {"evidence_validation": value.get("evidence_validation"), "overall_outcome": value.get("overall_outcome")} for phase, value in sorted(live.items())},
        "error_count": len(errors), "errors": errors,
        "warning_count": len(warnings), "warnings": warnings,
    }


def _write_result(evidence: Path, result: Mapping[str, Any]) -> None:
    target = evidence / "validation_result.json"
    temporary = target.with_name(target.name + ".partial")
    temporary.write_text(canonical_json(result), encoding="utf-8")
    temporary.replace(target)
    rows: list[str] = []
    for path in sorted(evidence.iterdir()):
        if path.is_file() and path.name != CHECKSUMS and not path.name.startswith("._"):
            rows.append(f"{sha256_file(path)}  {path.name}")
    (evidence / CHECKSUMS).write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Thermal T-B3 frame-only multi-seed evidence")
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--evidence-dir", default=str(ROOT / EVIDENCE_REL))
    parser.add_argument("--mode", choices=("READINESS", "FULL_EXPERIMENT"), default="FULL_EXPERIMENT")
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--write-result", action="store_true", help="Write compact validation_result.json and refresh checksums")
    args = parser.parse_args()
    evidence = Path(args.evidence_dir)
    result = validate_evidence(repo_root=Path(args.repo_root), evidence_dir=evidence, mode=args.mode, check_checksums=not args.skip_checksums)
    if args.write_result:
        _write_result(evidence, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["evidence_validation"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
