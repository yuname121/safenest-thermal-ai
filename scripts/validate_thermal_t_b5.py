#!/usr/bin/env python3
"""Standalone compact-evidence validator for Thermal T-B5."""

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

from datasets.thermal.t_b1_preprocessing import canonical_json, sha256_file
from datasets.thermal.t_b5_runner import (
    EVIDENCE_REL,
    EXPECTED_ARCHITECTURE_FINGERPRINT,
    EXPECTED_CHECKPOINT_SHA,
    EXPECTED_FP32_SHA,
    EXPECTED_FP32_SIZE,
    EXPECTED_INT8_SHA,
    EXPECTED_INT8_SIZE,
    FORMER_DYNAMIC_RANGE_SHA,
    FORMER_DYNAMIC_RANGE_SIZE,
    FULL_MODE,
    P1_PROFILE,
    PROFILE_ID,
    REPORT_NAME,
    READINESS_MODE,
    ROBUSTNESS_SAMPLE_COUNT,
    _profile,
    _run_predecessors,
    _sha256_text,
)


PHASE_ID = "T-B5"
BASE_JSON = (
    "t_b5_protocol.json", "predecessor_identity.json", "dataset_lock.json", "candidate_set.json",
    "artifact_registry.json", "robustness_profile.json", "latency_protocol.json", "candidate_lock_policy.json", "readiness_result.json",
)
FULL_JSON = BASE_JSON + (
    "robustness_results.json", "parity_summary.json", "latency_results.json", "candidate_lock.json",
    "real_diagnostic_summary.json", "execution_summary.json", "evidence_handoff.json", "limitation_registry.json", "t_b5_execution_result.json",
)


def _error(errors: list[dict[str, str]], code: str, location: str, message: str) -> None:
    errors.append({"code": code, "location": location, "message": message})


def _warning(warnings: list[dict[str, str]], code: str, location: str, message: str) -> None:
    warnings.append({"code": code, "location": location, "message": message})


def _read_documents(evidence: Path, names: Iterable[str], errors: list[dict[str, str]]) -> dict[str, Any]:
    documents: dict[str, Any] = {}
    for name in names:
        path = evidence / name
        if not path.is_file():
            _error(errors, "MISSING_EVIDENCE", name, "Required compact T-B5 evidence is missing.")
            continue
        try:
            documents[name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _error(errors, "INVALID_JSON", name, str(exc))
    return documents


def _walk_strings(value: Any, location: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield location, value
    elif isinstance(value, Mapping):
        for key in sorted(value):
            yield from _walk_strings(value[key], f"{location}.{key}" if location else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{location}[{index}]")


def _portable(value: str) -> bool:
    lower = value.lower()
    return not (value.startswith(("/", "~/", "file://")) or "\\" in value or "/users/" in lower or "/private/" in lower or value.startswith(("/volumes/", "/content/")))


def _validate_portability(documents: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    for name, document in documents.items():
        for location, value in _walk_strings(document):
            if not _portable(value):
                _error(errors, "ABSOLUTE_PATH_LEAK", f"{name}:{location}", "Tracked T-B5 evidence must contain repository-relative or symbolic portable paths.")
            if "archive/" in value.lower() and "diagnostic" not in value.lower() and "historical" not in value.lower():
                _error(errors, "ARCHIVE_TREATED_AS_ACTIVE", f"{name}:{location}", "Historical/archive paths cannot be active T-B5 sources.")


def _validate_checksums(evidence: Path, required: set[str], errors: list[dict[str, str]]) -> None:
    checksum_path = evidence / "checksums.sha256"
    if not checksum_path.is_file():
        _error(errors, "CHECKSUM_REGISTRY_MISSING", "checksums.sha256", "Checksum registry is required.")
        return
    measured: dict[str, str] = {}
    for line_no, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split("  ", 1)
        if len(parts) != 2:
            _error(errors, "CHECKSUM_LINE_INVALID", f"checksums.sha256:{line_no}", "Expected '<sha256>  <filename>'.")
            continue
        measured[parts[1]] = parts[0]
    for name in sorted(required):
        if name not in measured:
            _error(errors, "CHECKSUM_COVERAGE_MISSING", name, "Required evidence has no checksum coverage.")
            continue
        path = evidence / name
        if path.is_file() and sha256_file(path) != measured[name]:
            _error(errors, "CHECKSUM_MISMATCH", name, "Evidence checksum does not match the current file.")


def _validate_predecessor_identity(doc: Mapping[str, Any], live: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    expected = ("T-A6", "T-B0", "T-B1", "T-B2", "T-B3", "T-B4")
    if doc.get("phase") != PHASE_ID or set(doc.get("predecessors", {})) != set(expected):
        _error(errors, "PREDECESSOR_IDENTITY_INVALID", "predecessor_identity.json", "T-A6 through T-B4 live predecessor chain is incomplete.")
    for phase in expected:
        if live.get(phase, {}).get("evidence_validation") != "PASS":
            _error(errors, "PREDECESSOR_LIVE_INVALID", phase, "Live predecessor validator did not return PASS.")
        recorded = doc.get("predecessors", {}).get(phase, {})
        if recorded.get("evidence_validation") != "PASS":
            _error(errors, "PREDECESSOR_RECORDED_INVALID", phase, "Recorded predecessor status is not PASS.")


def _validate_base(documents: Mapping[str, Any], errors: list[dict[str, str]], warnings: list[dict[str, str]]) -> None:
    protocol = documents.get("t_b5_protocol.json", {})
    if protocol.get("phase") != PHASE_ID or protocol.get("protocol_id") != "THERMAL_T_B5_OFFLINE_ROBUSTNESS_LATENCY_CANDIDATE_LOCK_001" or any(protocol.get(key) is not False for key in ("training_performed", "recalibration_performed", "conversion_performed", "production_model_changed", "t_c_started")):
        _error(errors, "SCOPE_INVALID", "t_b5_protocol.json", "T-B5 must be an offline, non-training, non-conversion phase.")
    profile = documents.get("robustness_profile.json", {})
    profile_without_checksum = dict(profile); supplied = profile_without_checksum.pop("profile_checksum", None)
    if profile.get("profile_id") != PROFILE_ID or profile.get("selection_role") != "VALIDATION" or profile.get("sample_count") != ROBUSTNESS_SAMPLE_COUNT or supplied != _sha256_text(canonical_json(profile_without_checksum)):
        _error(errors, "PROFILE_CHECKSUM_INVALID", "robustness_profile.json", "Robustness profile is not the frozen pre-registered profile.")
    expected_families = {"AMBIENT_OFFSET", "DEAD_PIXEL", "PARTIAL_OCCLUSION", "HOT_OBJECT", "MISSING_FRAME", "ORIENTATION_ERROR"}
    actual_families = {str(item.get("family_id")) for item in profile.get("families", [])}
    if actual_families != expected_families:
        _error(errors, "ROBUSTNESS_FAMILY_SET_INVALID", "robustness_profile.json", "All six required robustness concerns must be present.")
    if profile.get("real_role_used") is not False or profile.get("locked_test_available") is not False or profile.get("production_runtime_changed") is not False:
        _error(errors, "ROLE_OR_PRODUCTION_SCOPE_INVALID", "robustness_profile.json", "REAL/LOCKED_TEST selection and production changes are forbidden.")
    dataset = documents.get("dataset_lock.json", {})
    if dataset.get("selection_role") != "VALIDATION" or dataset.get("real_role") != "REAL_EVAL_DEVELOPMENT" or dataset.get("validation_sample_count") != ROBUSTNESS_SAMPLE_COUNT or dataset.get("real_used_for_selection") is not False or dataset.get("locked_test_available") is not False:
        _error(errors, "DATASET_ROLE_INVALID", "dataset_lock.json", "Robustness must use VALIDATION and exclude REAL from selection.")
    indices = dataset.get("validation_sample_indices", [])
    if len(indices) != ROBUSTNESS_SAMPLE_COUNT or len(set(indices)) != ROBUSTNESS_SAMPLE_COUNT or any(not isinstance(item, int) or item < 0 or item >= 8000 for item in indices):
        _error(errors, "VALIDATION_SAMPLE_MANIFEST_INVALID", "dataset_lock.json", "Deterministic VALIDATION sample manifest is invalid.")
    candidates = documents.get("candidate_set.json", {})
    ids = {str(item.get("candidate_id")) for item in candidates.get("candidates", [])}
    if not {"FLOAT_KERAS", "TFLITE_FP32", "FULL_INT8", "TFLITE_DYNAMIC_RANGE"}.issubset(ids):
        _error(errors, "CANDIDATE_SET_INCOMPLETE", "candidate_set.json", "Frozen F0/F1/F2 and dynamic-range diagnostic candidate are required.")
    for item in candidates.get("candidates", []):
        if item.get("candidate_id") == "TFLITE_DYNAMIC_RANGE" and (item.get("eligible") is not False or item.get("reason") != "DYNAMIC_RANGE_DIAGNOSTIC_ONLY"):
            _error(errors, "DYNAMIC_RANGE_ELIGIBLE", "candidate_set.json", "Dynamic-range artifact must remain diagnostic-only.")
        if item.get("candidate_id") == "TFLITE_FP32":
            policy = item.get("conversion_policy", {})
            if policy.get("optimizations") != [] or policy.get("representative_dataset_attached") is not False or policy.get("float16_enabled") is not False or policy.get("dynamic_range_quantization") is not False or policy.get("quantization_mode") != "NONE" or policy.get("builtin_only") is not True:
                _error(errors, "FP32_CONVERSION_POLICY_INVALID", "candidate_set.json:TFLITE_FP32", "Float32 I/O alone is insufficient; true FP32 requires an explicit no-optimization conversion policy.")
    frozen = candidates.get("frozen_model_contract", {})
    if frozen.get("architecture_fingerprint") != EXPECTED_ARCHITECTURE_FINGERPRINT or frozen.get("architecture") != "SMALL_CNN_BASELINE_V1" or frozen.get("seed") != 20260813 or frozen.get("p1_profile") != P1_PROFILE or frozen.get("retraining") is not False or frozen.get("recalibration") is not False or frozen.get("float_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA or frozen.get("float_checkpoint_size_bytes") != 3777416:
        _error(errors, "FROZEN_MODEL_CONTRACT_DRIFT", "candidate_set.json", "T-B1/T-B4 model and preprocessing locks changed.")
    registry = documents.get("artifact_registry.json", {})
    registry_by_id = {str(item.get("candidate_id")): item for item in registry.get("artifacts", [])}
    for candidate, digest, size in (("FLOAT_KERAS", EXPECTED_CHECKPOINT_SHA, 3777416), ("TFLITE_FP32", EXPECTED_FP32_SHA, EXPECTED_FP32_SIZE), ("FULL_INT8", EXPECTED_INT8_SHA, EXPECTED_INT8_SIZE)):
        item = registry_by_id.get(candidate, {})
        if item.get("sha256") != digest or item.get("size_bytes") != size or item.get("eligible") is not True:
            _error(errors, "ARTIFACT_IDENTITY_INVALID", f"artifact_registry.json:{candidate}", "Frozen candidate identity/eligibility does not match T-B4.")
    dynamic = registry_by_id.get("TFLITE_DYNAMIC_RANGE", {})
    if dynamic.get("sha256") != FORMER_DYNAMIC_RANGE_SHA or dynamic.get("size_bytes") != FORMER_DYNAMIC_RANGE_SIZE or dynamic.get("eligible") is not False or dynamic.get("diagnostic_only") is not True:
        _error(errors, "DYNAMIC_RANGE_REGISTRY_INVALID", "artifact_registry.json:TFLITE_DYNAMIC_RANGE", "Dynamic-range artifact must remain diagnostic-only.")
    latency_protocol = documents.get("latency_protocol.json", {})
    if latency_protocol.get("host_scope") != "MAC_HOST_ONLY" or latency_protocol.get("pi_latency") != "NOT_MEASURED" or latency_protocol.get("thermal44_end_to_end") != "NOT_MEASURED_DEFERRED_TO_T-C" or latency_protocol.get("threads") != 1 or latency_protocol.get("measured_iterations") != 200:
        _error(errors, "LATENCY_PROTOCOL_INVALID", "latency_protocol.json", "Latency protocol is not the frozen Mac-only protocol.")
    policy = documents.get("candidate_lock_policy.json", {})
    if policy.get("frozen_before_results") is not True or policy.get("selection_role") != "VALIDATION" or policy.get("real_used_for_selection") is not False or "TFLITE_DYNAMIC_RANGE" not in policy.get("ineligible_candidates", {}):
        _error(errors, "CANDIDATE_POLICY_INVALID", "candidate_lock_policy.json", "Candidate policy was not frozen before results or admits REAL/dynamic-range selection.")
    readiness = documents.get("readiness_result.json", {})
    if readiness.get("status") != "T_B5_FULL_EXPERIMENT_READY" or readiness.get("predecessors_pass") is not True or readiness.get("profile_checksum") != profile.get("profile_checksum") or readiness.get("t_b6_started") is not False or readiness.get("t_c_started") is not False:
        _error(errors, "READINESS_INVALID", "readiness_result.json", "T-B5 readiness gate is invalid.")
    _warning(warnings, "NO_PRISTINE_LOCKED_TEST", "dataset_lock.json", "No pristine LOCKED_TEST exists; REAL_EVAL_DEVELOPMENT is diagnostic only.")
    _warning(warnings, "POSTURE_PROXY", "limitation_registry.json", "HUMAN_FALL remains a Lying-derived posture proxy, not temporal fall ground truth.")


def _validate_full(documents: Mapping[str, Any], errors: list[dict[str, str]], warnings: list[dict[str, str]]) -> None:
    results = documents.get("robustness_results.json", {})
    if results.get("profile_checksum") != documents.get("robustness_profile.json", {}).get("profile_checksum") or results.get("selection_role") != "VALIDATION" or results.get("real_used_for_selection") is not False or results.get("locked_test_available") is not False:
        _error(errors, "ROBUSTNESS_RESULT_SCOPE_INVALID", "robustness_results.json", "Robustness result role/scope is invalid.")
    cases = results.get("cases", [])
    required = {"AMBIENT_OFFSET", "DEAD_PIXEL", "PARTIAL_OCCLUSION", "HOT_OBJECT", "MISSING_FRAME", "ORIENTATION_ERROR"}
    if {str(item.get("family_id")) for item in cases} != required:
        _error(errors, "ROBUSTNESS_RESULT_FAMILIES_INVALID", "robustness_results.json", "Measured results do not cover all required families.")
    for item in cases:
        if item.get("family_id") == "MISSING_FRAME":
            if item.get("model_inference_performed") is not False or item.get("status") != "PIPELINE_CONTRACT_FAULT_FAIL_CLOSED" or item.get("replacement") != "NONE" or item.get("predictions") is not None:
                _error(errors, "MISSING_FRAME_NOT_FAIL_CLOSED", "robustness_results.json", "Missing-frame fault must not be imputed or inferred.")
        else:
            if item.get("status") != "MEASURED" or item.get("source_role") != "VALIDATION" or item.get("model_inference_performed") is not True or item.get("real_used_for_selection") is not False:
                _error(errors, "ROBUSTNESS_CASE_INVALID", "robustness_results.json", "Robustness case is not a VALIDATION-only measurement.")
            models = item.get("metrics", {})
            for candidate in ("FLOAT_KERAS", "TFLITE_FP32", "FULL_INT8"):
                metrics = models.get(candidate)
                if not isinstance(metrics, Mapping) or not all(math.isfinite(float(metrics.get(key, float("nan")))) for key in ("macro_f1", "accuracy", "balanced_accuracy", "h_fall_posture_proxy_recall")):
                    _error(errors, "ROBUSTNESS_METRICS_INVALID", "robustness_results.json", f"Missing finite metrics for {candidate}.")
    parity = documents.get("parity_summary.json", {})
    if parity.get("real_used_for_selection") is not False or set(parity.get("clean_cross_artifact", {})) != {"FLOAT_KERAS__TFLITE_FP32", "TFLITE_FP32__FULL_INT8", "FLOAT_KERAS__FULL_INT8"}:
        _error(errors, "PARITY_SCOPE_INVALID", "parity_summary.json", "Clean three-stage parity is incomplete or REAL contaminated selection.")
    real = documents.get("real_diagnostic_summary.json", {})
    if real.get("role") != "REAL_EVAL_DEVELOPMENT" or real.get("source_phase") != "T-B4" or real.get("used_for_selection") is not False or real.get("locked_test") is not False or real.get("int8_sensitivity", {}).get("diagnostic_only") is not True:
        _error(errors, "REAL_DIAGNOSTIC_SCOPE_INVALID", "real_diagnostic_summary.json", "REAL_EVAL_DEVELOPMENT must remain diagnostic-only and not LOCKED_TEST.")
    latency = documents.get("latency_results.json", {})
    if latency.get("role") != "VALIDATION" or latency.get("environment", {}).get("pi_measured") is not False or set(latency.get("candidates", {})) != {"TFLITE_FP32", "FULL_INT8"}:
        _error(errors, "LATENCY_RESULT_SCOPE_INVALID", "latency_results.json", "Mac-only latency evidence is incomplete or claims Pi.")
    for candidate, data in latency.get("candidates", {}).items():
        for view in ("invoke_only", "preprocess_plus_invoke"):
            stats = data.get(view, {})
            if stats.get("sample_count") != 200 or not all(math.isfinite(float(stats.get(key, float("nan")))) for key in ("mean_us", "median_us", "std_us", "min_us", "p90_us", "p95_us", "p99_us", "max_us")):
                _error(errors, "LATENCY_STATISTICS_INVALID", f"latency_results.json:{candidate}:{view}", "Latency statistics are incomplete or non-finite.")
    lock = documents.get("candidate_lock.json", {})
    if lock.get("status") != "OFFLINE_INT8_CANDIDATE_LOCKED_WITH_LIMITATIONS" or lock.get("selected_candidate_id") != "FULL_INT8" or lock.get("selection_role") != "VALIDATION" or lock.get("real_used_for_selection") is not False or lock.get("locked_test_used") is not False or lock.get("candidate_sha256") != EXPECTED_INT8_SHA or lock.get("candidate_size_bytes") != EXPECTED_INT8_SIZE or lock.get("thermal44_deployment_validated") is not False or lock.get("pi_latency_validated") is not False:
        _error(errors, "CANDIDATE_LOCK_INVALID", "candidate_lock.json", "Offline INT8 candidate lock is invalid or overclaims deployment.")
    summary = documents.get("execution_summary.json", {})
    if summary.get("status") != "FINALIZED_WITH_LIMITATIONS" or any(summary.get(key) is not False for key in ("training_performed", "recalibration_performed", "conversion_performed", "real_used_for_selection", "locked_test_available", "production_model_changed", "t_c_started", "t_b6_started")):
        _error(errors, "EXECUTION_SCOPE_INVALID", "execution_summary.json", "T-B5 execution scope or phase boundary is invalid.")
    handoff = documents.get("evidence_handoff.json", {})
    if handoff.get("thermal44_status") != "NOT_VALIDATED_DEFERRED_TO_T-C" or handoff.get("pi_status") != "NOT_MEASURED" or handoff.get("real_status") != "REAL_EVAL_DEVELOPMENT_DIAGNOSTIC_ONLY" or handoff.get("no_pristine_locked_test") is not True:
        _error(errors, "HANDOFF_OVERCLAIM", "evidence_handoff.json", "Device/deployment or locked-test status was overstated.")
    limitations = documents.get("limitation_registry.json", {})
    # The limitation registry intentionally names unsupported claims in a
    # negative list.  Reject only a positive boolean/status field, not the
    # disclosure text itself.
    forbidden_fields = {"thermal44_validated", "pi_latency_measured", "real_world_fall_ground_truth", "final_unbiased_test"}
    def positive_claims(value: Any, location: str = "") -> Iterable[tuple[str, Any]]:
        if isinstance(value, Mapping):
            for key, item in value.items():
                key_location = f"{location}.{key}" if location else str(key)
                if str(key).lower() in forbidden_fields:
                    yield key_location, item
                yield from positive_claims(item, key_location)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from positive_claims(item, f"{location}[{index}]")
    for location, value in positive_claims(limitations):
        if value is True or (isinstance(value, str) and value.upper() in {"TRUE", "PASS", "VALIDATED", "MEASURED"}):
            _error(errors, "UNSUPPORTED_CLAIM_PRESENT", f"limitation_registry.json:{location}", "Unsupported Thermal-44/Pi/ground-truth/final-test claim is positive.")
    execution = documents.get("t_b5_execution_result.json", {})
    if execution.get("phase") != PHASE_ID or execution.get("status") != "PASS_WITH_LIMITATIONS_PENDING_VALIDATOR":
        _error(errors, "EXECUTION_RESULT_INVALID", "t_b5_execution_result.json", "Execution result status is invalid.")
    _warning(warnings, "MAC_NOT_PI", "latency_results.json", "Mac host latency is not Raspberry Pi or sensor-to-alarm latency.")
    _warning(warnings, "SYNTHETIC_PERTURBATIONS", "robustness_results.json", "Perturbations are controlled offline diagnostics, not Thermal-44 validation.")


def validate_evidence(*, repo_root: Path = ROOT, evidence_dir: Path | None = None, mode: str = FULL_MODE, check_checksums: bool = True, verify_predecessors: bool = True) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); evidence = Path(evidence_dir or repo / EVIDENCE_REL).resolve(); errors: list[dict[str, str]] = []; warnings: list[dict[str, str]] = []
    names = FULL_JSON if mode == FULL_MODE else BASE_JSON
    documents = _read_documents(evidence, names, errors)
    if verify_predecessors:
        try:
            live = _run_predecessors(repo)
        except Exception as exc:  # pragma: no cover - defensive validator boundary
            live = {}
            _error(errors, "PREDECESSOR_VALIDATION_ERROR", "predecessor_identity.json", str(exc))
    else:
        live = {phase: {"evidence_validation": "PASS", "overall_outcome": "TEST_FIXTURE"} for phase in ("T-A6", "T-B0", "T-B1", "T-B2", "T-B3", "T-B4")}
    if all(name in documents for name in BASE_JSON): _validate_base(documents, errors, warnings); _validate_predecessor_identity(documents["predecessor_identity.json"], live, errors)
    if mode == FULL_MODE and all(name in documents for name in FULL_JSON): _validate_full(documents, errors, warnings)
    if mode == FULL_MODE:
        report = evidence / REPORT_NAME
        if not report.is_file():
            _error(errors, "REPORT_MISSING", REPORT_NAME, "Human-readable T-B5 comparison report is required.")
        elif any(not _portable(line.strip()) for line in report.read_text(encoding="utf-8").splitlines() if line.strip().startswith(("/", "file://", "~/"))):
            _error(errors, "ABSOLUTE_PATH_LEAK", REPORT_NAME, "T-B5 report contains an absolute path.")
    _validate_portability(documents, errors)
    if check_checksums: _validate_checksums(evidence, set(names), errors)
    errors.sort(key=lambda item: (item["code"], item["location"], item["message"])); warnings.sort(key=lambda item: (item["code"], item["location"], item["message"]))
    passed = not errors
    return {"phase": PHASE_ID, "mode": mode, "schema_version": "1.0", "evidence_validation": "PASS" if passed else "FAIL", "overall_outcome": "T_B5_COMPLETE_WITH_LIMITATIONS" if passed else "T_B5_BLOCKED", "t_c_authorized": "YES_WITH_LIMITATIONS" if passed else False, "error_count": len(errors), "errors": errors, "warning_count": len(warnings), "warnings": warnings, "predecessors": {phase: {"evidence_validation": value.get("evidence_validation"), "overall_outcome": value.get("overall_outcome")} for phase, value in sorted(live.items())}}


def _write_result(evidence: Path, result: Mapping[str, Any]) -> None:
    target = evidence / "validation_result.json"; target.write_text(canonical_json(result), encoding="utf-8")
    rows: list[str] = []
    for path in sorted(evidence.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name not in {"checksums.sha256", "validation_result.json"} and not path.name.startswith("._"):
            rows.append(f"{sha256_file(path)}  {path.name}")
    (evidence / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SafeNest Thermal T-B5 compact evidence")
    parser.add_argument("--repo-root", default=str(ROOT)); parser.add_argument("--evidence-dir", default=str(ROOT / EVIDENCE_REL)); parser.add_argument("--mode", choices=(READINESS_MODE, FULL_MODE), default=FULL_MODE); parser.add_argument("--skip-checksums", action="store_true"); parser.add_argument("--write-result", action="store_true")
    args = parser.parse_args(); evidence = Path(args.evidence_dir); result = validate_evidence(repo_root=Path(args.repo_root), evidence_dir=evidence, mode=args.mode, check_checksums=not args.skip_checksums)
    if args.write_result: _write_result(evidence, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)); return 0 if result["evidence_validation"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
