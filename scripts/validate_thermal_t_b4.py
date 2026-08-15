#!/usr/bin/env python3
"""Standalone, payload-free validator for the Thermal T-B4 contract."""

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

from datasets.thermal.t_b1_preprocessing import CLASS_ORDER, P1Statistics, canonical_json  # noqa: E402
from datasets.thermal.t_b4_runner import (  # noqa: E402
    CALIBRATION_COUNT,
    CALIBRATION_LABEL_ORDER,
    EVIDENCE_REL,
    EXPECTED_ARCHITECTURE_FINGERPRINT,
    EXPECTED_CHECKPOINT_SHA,
    EXPECTED_CHECKPOINT_SIZE,
    EXPECTED_NEAR_DUPLICATE_PAIRS,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_P1_CHECKSUM,
    EXPECTED_P1_MEAN,
    EXPECTED_P1_STD,
    FULL_MODE,
    P1_PROFILE,
    PHASE_ID,
    PROTOCOL_ID,
    READINESS_MODE,
)


TB0_REL = "datasets/thermal/manifests/T-B0_offline_model_protocol"
TB1_REL = "datasets/thermal/manifests/T-B1_full_experiment"
TB2_REL = "datasets/thermal/manifests/T-B2_architecture_comparison"
TB3_REL = "datasets/thermal/manifests/T-B3_frame_multiseed_confirmation"
TA6_REL = "datasets/thermal/manifests/T-A6_execution_result"
CHECKSUMS = "checksums.sha256"
FORMER_DYNAMIC_RANGE_SHA = "297de231e26ecf2d4cd4010bd10c08d4df3b6b0a531c69693daea353afb8127d"
FORMER_DYNAMIC_RANGE_SIZE = 317344

BASE_JSON = (
    "t_b4_protocol.json", "predecessor_identity.json", "float_candidate_lock.json", "p1_lock.json",
    "dataset_lock.json", "environment.json", "artifact_storage_policy.json",
    "representative_calibration_policy.json", "representative_sample_manifest.json", "temperature_range_policy.json",
    "readiness_result.json",
)
FULL_JSON = BASE_JSON + (
    "execution_environment.json", "float_source_integrity.json", "tflite_fp32_artifact.json", "tflite_int8_artifact.json",
    "tensor_quantization_contract.json", "op_inventory.json", "float_vs_tflite_fp32_parity.json", "float_vs_int8_parity.json",
    "fp32_vs_int8_parity.json", "input_saturation_audit.json", "output_saturation_audit.json", "temperature_range_error.json",
    "real_development_parity.json", "limitations.json", "artifact_registry.json", "execution_summary.json",
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
    return not (value.startswith(("/", "~/", "file://")) or "\\" in value or "/users/" in lower or "/private/" in lower or value.startswith(("/volumes/", "/content/")))


def _read_documents(evidence: Path, names: Iterable[str], errors: list[dict[str, str]]) -> dict[str, Any]:
    documents: dict[str, Any] = {}
    for name in names:
        path = evidence / name
        if not path.is_file():
            _error(errors, "REQUIRED_ARTIFACT_MISSING", name, "Required compact T-B4 artifact is missing.")
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
            if isinstance(item, str):
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
        _error(errors, "CHECKSUM_REGISTRY_MISSING", CHECKSUMS, "T-B4 checksum registry is required.")
        return
    entries: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64 or any(ch not in "0123456789abcdef" for ch in parts[0].lower()):
            _error(errors, "CHECKSUM_FORMAT_INVALID", f"{CHECKSUMS}:{number}", line)
            continue
        digest, relative = parts
        if not _portable(relative) or relative.startswith("archive/"):
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
    try:
        from scripts.validate_thermal_t_a6 import validate_evidence as a6
        from scripts.validate_thermal_t_b0 import validate_evidence as b0
        from scripts.validate_thermal_t_b1 import validate_evidence as b1
        from scripts.validate_thermal_t_b2 import validate_evidence as b2
        from scripts.validate_thermal_t_b3 import validate_evidence as b3
        result = {
            "T-A6": a6(repo_root=repo_root, evidence_dir=repo_root / TA6_REL, mode="FULL_DATASET", check_checksums=True),
            "T-B0": b0(repo_root=repo_root, evidence_dir=repo_root / TB0_REL, check_checksums=True),
            "T-B1": b1(repo_root=repo_root, evidence_dir=repo_root / TB1_REL, mode="FULL_EXPERIMENT", check_checksums=True),
            "T-B2": b2(repo_root=repo_root, evidence_dir=repo_root / TB2_REL, mode="FULL_EXPERIMENT", check_checksums=True),
            "T-B3": b3(repo_root=repo_root, evidence_dir=repo_root / TB3_REL, mode="FULL_EXPERIMENT", check_checksums=True),
        }
    except Exception as exc:  # pragma: no cover
        _error(errors, "PREDECESSOR_VALIDATOR_ERROR", PHASE_ID, str(exc))
        return {}
    for phase, item in result.items():
        if item.get("evidence_validation") != "PASS":
            _error(errors, "PREDECESSOR_LIVE_INVALID", phase, str(item.get("overall_outcome")))
    return result


def _validate_protocol(doc: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    exact = {"phase": PHASE_ID, "protocol_id": PROTOCOL_ID, "factor_changed": "EXPORT_AND_QUANTIZATION_ONLY", "training": "PROHIBITED", "retraining": False, "reference_candidate_id": "SMALL_CNN_BASELINE_V1", "reference_seed": 20260813, "p1_profile": P1_PROFILE, "real_used_for_calibration": False, "validation_used_for_calibration": False, "new_split_created": False, "t_b5_started": False, "t_c_started": False, "legacy_model_replacement": False, "runtime_manifest_modified": False, "equivalence_contract": "EQUIVALENCE_MEASURED_NO_PREEXISTING_ABSOLUTE_ACCEPTANCE_THRESHOLD", "acceptance_threshold_status": "NO_PREEXISTING_ABSOLUTE_ACCEPTANCE_THRESHOLD"}
    for key, value in exact.items():
        if doc.get(key) != value:
            _error(errors, "PROTOCOL_INVALID", f"t_b4_protocol.json:{key}", f"Expected {value!r}.")


def _validate_predecessor_identity(doc: Mapping[str, Any], live: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    if doc.get("phase") != PHASE_ID or doc.get("t_b3_merge_verified") is not True:
        _error(errors, "PREDECESSOR_IDENTITY_INVALID", "predecessor_identity.json", "T-B3 merge/live identity is not verified.")
    for phase in ("T-A6", "T-B0", "T-B1", "T-B2", "T-B3"):
        if live.get(phase, {}).get("evidence_validation") != "PASS" or doc.get("predecessors", {}).get(phase, {}).get("evidence_validation") != "PASS":
            _error(errors, "PREDECESSOR_IDENTITY_INVALID", phase, "Predecessor must be live validator PASS.")


def _validate_dataset(doc: Mapping[str, Any], errors: list[dict[str, str]], warnings: list[dict[str, str]]) -> None:
    expected = {"TRAIN": (32000, "749c847fc9ab50ea5eee8827f0d47b5ebaa48165732a382c59f8b96c565b9d93"), "VALIDATION": (8000, "5d16451702c1bccfa945d9188d9b29a26ce11c8b33bf7a0dfbb25bfa86d74610"), "REAL_EVAL_DEVELOPMENT": (8000, "cd696e68aeec063cbc8185719b4f4dad3d038cb3d28eec0d3701b8311e4ad8f1")}
    roles = doc.get("roles", {})
    if set(roles) != set(expected):
        _error(errors, "DATASET_ROLE_SET_INVALID", "dataset_lock.json:roles", "Exactly TRAIN, VALIDATION, REAL_EVAL_DEVELOPMENT are required.")
    for role, (rows, digest) in expected.items():
        item = roles.get(role, {})
        for key, value in (("rows", rows), ("sha256", digest), ("dtype", "float32_little_endian"), ("unit", "CELSIUS")):
            if item.get(key) != value:
                _error(errors, "CANONICAL_IDENTITY_INVALID", f"dataset_lock.json:roles.{role}.{key}", f"Expected {value!r}.")
    real = roles.get("REAL_EVAL_DEVELOPMENT", {})
    if real.get("source_domain") != "REAL" or real.get("locked_test") is not False:
        _error(errors, "REAL_ROLE_INVALID", "dataset_lock.json:roles.REAL_EVAL_DEVELOPMENT", "REAL must remain development-only and unlocked.")
    for key in ("legacy_npz_used", "raw_zip_used", "official_partition_preservation"):
        expected_value = False if key in ("legacy_npz_used", "raw_zip_used") else True
        if doc.get(key) is not expected_value:
            _error(errors, "DATASET_SCOPE_INVALID", f"dataset_lock.json:{key}", f"Expected {expected_value!r}.")
    if doc.get("near_duplicate_pairs_train_validation") != EXPECTED_NEAR_DUPLICATE_PAIRS:
        _error(errors, "DATASET_LIMITATION_LOST", "dataset_lock.json:near_duplicate_pairs_train_validation", "Inherited near-duplicate count changed.")
    _warning(warnings, "NEAR_DUPLICATE_OVERLAP", "dataset_lock.json", "14,514 TRAIN-VALIDATION near-duplicate pairs remain disclosed.")


def _validate_p1(doc: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    if doc.get("phase") != PHASE_ID or doc.get("profile_id") != P1_PROFILE or doc.get("fit_role") != "TRAIN" or doc.get("statistics_checksum") != EXPECTED_P1_CHECKSUM:
        _error(errors, "P1_IDENTITY_INVALID", "p1_lock.json", "P1 identity/checksum differs from T-B1.")
    try:
        stats = P1Statistics(mean=float(doc["mean"]), std=float(doc["std"]), fit_sample_count=int(doc["fit_sample_count"]), fit_pixel_count=int(doc["fit_pixel_count"]), fit_role=str(doc["fit_role"]), train_artifact_sha256=str(doc["train_artifact_sha256"]), epsilon=float(doc.get("epsilon", 1e-6)))
        if stats.checksum() != EXPECTED_P1_CHECKSUM or not math.isclose(stats.mean, EXPECTED_P1_MEAN, abs_tol=1e-12) or not math.isclose(stats.std, EXPECTED_P1_STD, abs_tol=1e-12):
            _error(errors, "P1_STATISTICS_MISMATCH", "p1_lock.json", "P1 statistics drifted.")
    except (KeyError, TypeError, ValueError) as exc:
        _error(errors, "P1_STATISTICS_INVALID", "p1_lock.json", str(exc))
    for key in ("refit", "validation_fit", "real_fit"):
        if doc.get(key) is not False:
            _error(errors, "P1_REFIT_POLICY_INVALID", f"p1_lock.json:{key}", "P1 may not be refit.")


def _validate_candidate(doc: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    if doc.get("phase") != PHASE_ID or doc.get("candidate_id") != "SMALL_CNN_BASELINE_V1" or doc.get("reference_seed") != 20260813 or doc.get("architecture_fingerprint") != EXPECTED_ARCHITECTURE_FINGERPRINT or doc.get("parameter_count") != EXPECTED_PARAMETER_COUNT or doc.get("retraining") is not False or doc.get("candidate_changed") is not False:
        _error(errors, "FLOAT_CANDIDATE_LOCK_INVALID", "float_candidate_lock.json", "Frozen candidate identity changed.")
    checkpoint = doc.get("checkpoint", {})
    if checkpoint.get("sha256") != EXPECTED_CHECKPOINT_SHA or checkpoint.get("size_bytes") != EXPECTED_CHECKPOINT_SIZE:
        _error(errors, "CHECKPOINT_IDENTITY_INVALID", "float_candidate_lock.json:checkpoint", "Reference checkpoint hash/size changed.")


def _validate_storage(doc: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    expected = {"raw_payloads_tracked": False, "canonical_payloads_tracked": False, "checkpoint_tracked": False, "legacy_model_overwrite": False}
    if doc.get("phase") != PHASE_ID:
        _error(errors, "STORAGE_POLICY_INVALID", "artifact_storage_policy.json", "Wrong phase.")
    for key, value in expected.items():
        if doc.get(key) is not value:
            _error(errors, "STORAGE_POLICY_INVALID", f"artifact_storage_policy.json:{key}", f"Expected {value!r}.")


def _validate_calibration(policy: Mapping[str, Any], manifest: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    if policy.get("phase") != PHASE_ID or policy.get("policy_id") != "T-B4_TRAIN_ONLY_STRATIFIED_CALIBRATION_512" or policy.get("source_role") != "TRAIN" or policy.get("sample_count") != CALIBRATION_COUNT or policy.get("validation_samples_used") != 0 or policy.get("real_samples_used") != 0 or policy.get("p1_profile") != P1_PROFILE:
        _error(errors, "REPRESENTATIVE_POLICY_INVALID", "representative_calibration_policy.json", "Calibration must be frozen TRAIN-only P1.")
    if manifest.get("phase") != PHASE_ID or manifest.get("role") != "TRAIN" or manifest.get("row_count") != CALIBRATION_COUNT or len(manifest.get("rows", [])) != CALIBRATION_COUNT:
        _error(errors, "REPRESENTATIVE_MANIFEST_INVALID", "representative_sample_manifest.json", "Representative manifest count/role invalid.")
    rows = manifest.get("rows", [])
    indices = [row.get("canonical_sample_index") for row in rows]
    if len(set(indices)) != CALIBRATION_COUNT or any(not isinstance(value, int) or value < 0 or value >= 32000 for value in indices):
        _error(errors, "REPRESENTATIVE_MANIFEST_INVALID", "representative_sample_manifest.json:rows", "Indices must be unique TRAIN indices.")
    for row in rows:
        if row.get("role") != "TRAIN" or row.get("p1_applied") is not True or row.get("source_label") not in CALIBRATION_LABEL_ORDER or not isinstance(row.get("temperature_band"), int) or not 0 <= row.get("temperature_band") < 8:
            _error(errors, "REPRESENTATIVE_ROW_INVALID", "representative_sample_manifest.json:rows", "Representative row is outside the frozen policy.")
            break
    manifest_copy = dict(manifest); manifest_checksum = manifest_copy.pop("manifest_checksum", None)
    if manifest_checksum != hashlib.sha256(canonical_json(manifest_copy).encode("utf-8")).hexdigest() or policy.get("manifest_checksum") != manifest_checksum:
        _error(errors, "REPRESENTATIVE_MANIFEST_CHECKSUM_INVALID", "representative_sample_manifest.json", "Manifest checksum is stale.")
    policy_copy = dict(policy); policy_checksum = policy_copy.pop("policy_checksum", None)
    if policy_checksum != hashlib.sha256(canonical_json(policy_copy).encode("utf-8")).hexdigest():
        _error(errors, "REPRESENTATIVE_POLICY_CHECKSUM_INVALID", "representative_calibration_policy.json", "Policy checksum is stale.")
    if "VALIDATION" in json.dumps(manifest, sort_keys=True) or "REAL_EVAL_DEVELOPMENT" in json.dumps(manifest, sort_keys=True):
        _error(errors, "VALIDATION_OR_REAL_USED_FOR_CALIBRATION", "representative_sample_manifest.json", "Only TRAIN samples are allowed.")


def _validate_temperature(doc: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    if doc.get("phase") != PHASE_ID or doc.get("policy_id") != "T-B4_TRAIN_MEAN_QUANTILE_BANDS_8" or doc.get("fit_role") != "TRAIN" or doc.get("band_count") != 8 or doc.get("validation_or_real_used") is not False:
        _error(errors, "TEMPERATURE_POLICY_INVALID", "temperature_range_policy.json", "Temperature bands must be frozen from TRAIN only.")
    boundaries = doc.get("boundaries_celsius", [])
    if len(boundaries) != 9 or any(not isinstance(value, (float, int)) or not math.isfinite(float(value)) for value in boundaries) or any(float(left) > float(right) for left, right in zip(boundaries, boundaries[1:])):
        _error(errors, "TEMPERATURE_POLICY_INVALID", "temperature_range_policy.json:boundaries_celsius", "Eight ordered finite bands are required.")


def _validate_readiness(doc: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    if doc.get("phase") != PHASE_ID or doc.get("status") != "T_B4_CONVERSION_READY" or doc.get("predecessors_pass") is not True or doc.get("checkpoint_verified") is not True or doc.get("calibration_frozen") is not True or doc.get("temperature_policy_frozen") is not True or doc.get("t_b5_started") is not False or doc.get("t_c_started") is not False:
        _error(errors, "READINESS_GATE_INVALID", "readiness_result.json", "T-B4 readiness gate is not satisfied.")


def _validate_shape_dtype(document: Mapping[str, Any], location: str, dtype: str, errors: list[dict[str, str]]) -> None:
    item = document.get("input", {})
    output = document.get("output", {})
    if item.get("shape") != [1, 62, 80, 1] or output.get("shape") != [1, 3] or item.get("dtype") != dtype or output.get("dtype") != dtype:
        _error(errors, "TENSOR_CONTRACT_INVALID", location, "Input/output shape and dtype do not match the frozen candidate contract.")


def _validate_artifacts(documents: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    fp32 = documents.get("tflite_fp32_artifact.json", {}); int8 = documents.get("tflite_int8_artifact.json", {})
    _validate_shape_dtype(fp32, "tflite_fp32_artifact.json", "float32", errors)
    _validate_shape_dtype(int8, "tflite_int8_artifact.json", "int8", errors)
    for name, doc in (("tflite_fp32_artifact.json", fp32), ("tflite_int8_artifact.json", int8)):
        if doc.get("phase") != PHASE_ID or not isinstance(doc.get("sha256"), str) or len(doc.get("sha256", "")) != 64 or int(doc.get("size_bytes", 0)) <= 0 or doc.get("select_tf_ops_used") is not False or doc.get("builtin_only") is not True:
            _error(errors, "TFLITE_ARTIFACT_INVALID", name, "Artifact identity or builtin-only contract invalid.")
    fp32_conversion = fp32.get("conversion", {})
    if fp32.get("artifact_id") != "TFLITE_FP32":
        _error(errors, "FP32_ARTIFACT_ID_INVALID", "tflite_fp32_artifact.json:artifact_id", "The official equivalence artifact must be labeled TFLITE_FP32.")
    if fp32_conversion.get("supported_ops") != ["TFLITE_BUILTINS"]:
        _error(errors, "FP32_CONVERSION_POLICY_INVALID", "tflite_fp32_artifact.json:conversion.supported_ops", "FP32 candidate must use builtin ops only.")
    if fp32_conversion.get("optimizations") != []:
        _error(errors, "FP32_QUANTIZATION_POLICY_INVALID", "tflite_fp32_artifact.json:conversion.optimizations", "True FP32 conversion must not enable Optimize.DEFAULT or another optimization.")
    for key, expected in (("representative_dataset_attached", False), ("float16_enabled", False), ("dynamic_range_quantization", False), ("quantization_mode", "NONE")):
        if fp32_conversion.get(key) != expected:
            _error(errors, "FP32_QUANTIZATION_POLICY_INVALID", f"tflite_fp32_artifact.json:conversion.{key}", f"Expected {expected!r} for an unquantized FP32 artifact.")
    internal_dtypes = fp32.get("internal_dtype_counts")
    if not isinstance(internal_dtypes, dict) or any(str(dtype) in internal_dtypes for dtype in ("int8", "uint8", "int16", "uint16")):
        _error(errors, "FP32_INTERNAL_QUANTIZATION_INVALID", "tflite_fp32_artifact.json:internal_dtype_counts", "Internal tensor dtypes contradict a true unquantized FP32 graph.")
    for key in ("quantized_tensor_count", "quantized_parameter_tensor_count", "nonzero_quantization_tensor_count"):
        if fp32.get(key) != 0:
            _error(errors, "FP32_INTERNAL_QUANTIZATION_INVALID", f"tflite_fp32_artifact.json:{key}", "FP32 artifact contains quantized tensor evidence.")
    int8_conversion = int8.get("conversion", {})
    if int8_conversion.get("supported_ops") != ["TFLITE_BUILTINS_INT8"] or int8_conversion.get("representative_policy_checksum") != documents.get("representative_calibration_policy.json", {}).get("policy_checksum") or int8_conversion.get("representative_dataset_attached") is not True or int8_conversion.get("quantization_mode") != "FULL_INT8":
        _error(errors, "FULL_INT8_REQUIREMENT_NOT_MET", "tflite_int8_artifact.json", "INT8 conversion policy/frozen representative checksum invalid.")
    quant = documents.get("tensor_quantization_contract.json", {})
    if quant.get("phase") != PHASE_ID or quant.get("full_integer") is not True or quant.get("input", {}).get("dtype") != "int8" or quant.get("output", {}).get("dtype") != "int8" or float(quant.get("input", {}).get("scale", 0.0)) <= 0 or float(quant.get("output", {}).get("scale", 0.0)) <= 0:
        _error(errors, "TENSOR_QUANTIZATION_INVALID", "tensor_quantization_contract.json", "Actual INT8 quantization metadata is invalid.")
    inventory = documents.get("op_inventory.json", {})
    if not inventory.get("float32", {}).get("builtin_only") or inventory.get("float32", {}).get("quantized_tensor_count") != 0 or any(str(dtype) in inventory.get("float32", {}).get("internal_dtype_counts", {}) for dtype in ("int8", "uint8", "int16", "uint16")):
        _error(errors, "FP32_INTERNAL_QUANTIZATION_INVALID", "op_inventory.json:float32", "FP32 operation inventory contains quantized internal dtype evidence.")
    if not inventory.get("full_int8", {}).get("builtin_only") or any("FLOAT" in str(op).upper() or "FLEX" in str(op).upper() for op in inventory.get("full_int8", {}).get("ops", [])):
        _error(errors, "FLOAT_FALLBACK_PRESENT", "op_inventory.json:full_int8", "Full INT8 graph contains an unauthorized fallback.")
    registry = documents.get("artifact_registry.json", {})
    if registry.get("phase") != PHASE_ID or registry.get("legacy_model_overwrite") is not False or registry.get("production_manifest_changed") is not False:
        _error(errors, "ARTIFACT_REGISTRY_INVALID", "artifact_registry.json", "Legacy/runtime model replacement is prohibited.")
    for item in registry.get("artifacts", []):
        if item.get("tracked_in_git") is not False or not str(item.get("logical_path", "")).startswith(("artifacts/", "parity/")):
            _error(errors, "BINARY_GIT_SCOPE_INVALID", "artifact_registry.json", "Binary artifacts must remain external.")
        if item.get("id") in {"TFLITE_FP32", "FULL_INT8"} and item.get("official_equivalence_stage") is not True:
            _error(errors, "ARTIFACT_STAGE_CLASSIFICATION_INVALID", f"artifact_registry.json:{item.get('id')}", "Official Float/FP32/INT8 equivalence stages must be explicitly marked.")
    dynamic = next((item for item in registry.get("artifacts", []) if item.get("id") == "TFLITE_DYNAMIC_RANGE"), None)
    official_fp32 = next((item for item in registry.get("artifacts", []) if item.get("id") == "TFLITE_FP32"), None)
    if not isinstance(official_fp32, dict) or official_fp32.get("logical_path") != "artifacts/SMALL_CNN_BASELINE_V1_P1_float32.tflite" or official_fp32.get("conversion", {}).get("optimizations") != [] or official_fp32.get("conversion", {}).get("dynamic_range_quantization") is not False or official_fp32.get("quantized_tensor_count") != 0:
        _error(errors, "FP32_REGISTRY_POLICY_INVALID", "artifact_registry.json:TFLITE_FP32", "Registry metadata must identify the no-optimization, unquantized FP32 artifact.")
    if not isinstance(dynamic, dict):
        _error(errors, "DYNAMIC_RANGE_RECLASSIFICATION_MISSING", "artifact_registry.json", "Former FP32 artifact must be retained only as a diagnostic dynamic-range entry.")
    else:
        if dynamic.get("logical_path") != "artifacts/SMALL_CNN_BASELINE_V1_P1_dynamic_range.tflite" or dynamic.get("sha256") != FORMER_DYNAMIC_RANGE_SHA or dynamic.get("size_bytes") != FORMER_DYNAMIC_RANGE_SIZE or dynamic.get("official_equivalence_stage") is not False or dynamic.get("diagnostic_only") is not True or dynamic.get("reclassified_from") != "TFLITE_FP32":
            _error(errors, "DYNAMIC_RANGE_RECLASSIFICATION_INVALID", "artifact_registry.json:TFLITE_DYNAMIC_RANGE", "Former artifact identity/classification is not preserved as diagnostic-only.")
        if dynamic.get("conversion", {}).get("optimizations") != ["DEFAULT"] or dynamic.get("conversion", {}).get("dynamic_range_quantization") is not True or int(dynamic.get("quantized_parameter_tensor_count", 0)) <= 0:
            _error(errors, "DYNAMIC_RANGE_EVIDENCE_INVALID", "artifact_registry.json:TFLITE_DYNAMIC_RANGE", "Dynamic-range conversion evidence is incomplete.")


def _validate_parity(doc: Mapping[str, Any], location: str, errors: list[dict[str, str]]) -> None:
    if doc.get("phase") != PHASE_ID or doc.get("role") != "VALIDATION" or doc.get("sample_count") != 8000:
        _error(errors, "PARITY_EVIDENCE_INVALID", location, "VALIDATION parity sample count/role invalid.")
        return
    for key in ("argmax_agreement", "probability_mae", "probability_max_absolute_error"):
        if not isinstance(doc.get(key), (int, float)) or not math.isfinite(float(doc[key])) or float(doc[key]) < 0:
            _error(errors, "PARITY_EVIDENCE_INVALID", f"{location}:{key}", "Parity statistic must be finite and non-negative.")
    if int(doc.get("disagreement_count", -1)) < 0 or int(doc.get("disagreement_count", -1)) > 8000:
        _error(errors, "PARITY_EVIDENCE_INVALID", location, "Disagreement count is invalid.")
    for side in ("float_metrics", "other_metrics"):
        metrics = doc.get(side, {})
        if metrics.get("class_order") != list(CLASS_ORDER) or metrics.get("sample_count") != 8000:
            _error(errors, "METRIC_CONTRACT_INVALID", f"{location}:{side}", "Metric class/sample contract invalid.")
        matrix = metrics.get("confusion_matrix")
        if not isinstance(matrix, list) or len(matrix) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in matrix) or sum(sum(int(v) for v in row) for row in matrix) != 8000:
            _error(errors, "METRIC_CONTRACT_INVALID", f"{location}:{side}", "Confusion matrix must be 3x3 and sum to 8000.")


def _validate_saturation(documents: Mapping[str, Any], errors: list[dict[str, str]], warnings: list[dict[str, str]]) -> None:
    input_doc = documents.get("input_saturation_audit.json", {}); output_doc = documents.get("output_saturation_audit.json", {})
    if input_doc.get("phase") != PHASE_ID or input_doc.get("role") != "VALIDATION" or int(input_doc.get("total_elements", 0)) != 8000 * 62 * 80:
        _error(errors, "SATURATION_AUDIT_INVALID", "input_saturation_audit.json", "Input saturation total must cover all VALIDATION elements.")
    if int(input_doc.get("required_clipping_count", -1)) < 0 or float(input_doc.get("required_clipping_fraction", -1.0)) < 0 or float(input_doc.get("required_clipping_fraction", 2.0)) > 1:
        _error(errors, "SATURATION_AUDIT_INVALID", "input_saturation_audit.json", "Clipping statistics invalid.")
    if output_doc.get("phase") != PHASE_ID or int(output_doc.get("total_elements", 0)) != 8000 * 3:
        _error(errors, "SATURATION_AUDIT_INVALID", "output_saturation_audit.json", "Output saturation total must cover VALIDATION outputs.")
    _warning(warnings, "SATURATION_IS_DESCRIPTIVE", "input_saturation_audit.json", "No post-hoc absolute saturation acceptance threshold was invented.")


def _validate_temperature_error(doc: Mapping[str, Any], policy: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    rows = doc.get("bands", [])
    if doc.get("phase") != PHASE_ID or len(rows) != 8 or sum(int(row.get("sample_count", 0)) for row in rows) != 8000:
        _error(errors, "TEMPERATURE_AUDIT_INVALID", "temperature_range_error.json", "Temperature audit must contain eight VALIDATION bands totaling 8000.")
    for row in rows:
        if not 0 <= int(row.get("band", -1)) < 8 or not math.isfinite(float(row.get("argmax_agreement", float("nan")))) or not math.isfinite(float(row.get("probability_mae", float("nan")))):
            _error(errors, "TEMPERATURE_AUDIT_INVALID", "temperature_range_error.json:bands", "Temperature-band metric invalid.")
    policy_copy = dict(policy)
    expected_checksum = hashlib.sha256(canonical_json(policy_copy).encode("utf-8")).hexdigest()
    if doc.get("temperature_policy_checksum") != expected_checksum:
        _error(errors, "TEMPERATURE_POLICY_CHECKSUM_MISMATCH", "temperature_range_error.json:temperature_policy_checksum", "Temperature-band results do not identify the frozen policy.")


def _validate_real(doc: Mapping[str, Any], errors: list[dict[str, str]], warnings: list[dict[str, str]]) -> None:
    if doc.get("phase") != PHASE_ID or doc.get("role") != "REAL_EVAL_DEVELOPMENT" or doc.get("diagnostic_performed") is not True or doc.get("frozen_before_real") is not True or doc.get("used_for_calibration") is not False or doc.get("used_for_selection") is not False or doc.get("sample_count") != 8000:
        _error(errors, "REAL_DIAGNOSTIC_INVALID", "real_development_parity.json", "REAL must be a fixed post-artifact development diagnostic.")
    for key in ("float_vs_tflite_fp32_parity", "fp32_vs_int8_parity", "float_vs_int8_parity"):
        parity = doc.get(key)
        if not isinstance(parity, dict) or parity.get("sample_count") != 8000 or not isinstance(parity.get("argmax_agreement"), (int, float)) or not isinstance(parity.get("probability_mae"), (int, float)) or not isinstance(parity.get("probability_max_absolute_error"), (int, float)):
            _error(errors, "REAL_PARITY_INVALID", f"real_development_parity.json:{key}", "REAL must retain all three fixed Float/true-FP32/INT8 parity comparisons.")
        elif any(not math.isfinite(float(parity[field])) or float(parity[field]) < 0 for field in ("argmax_agreement", "probability_mae", "probability_max_absolute_error")):
            _error(errors, "REAL_PARITY_INVALID", f"real_development_parity.json:{key}", "REAL parity statistics must be finite and non-negative.")
    _warning(warnings, "REAL_NOT_LOCKED_TEST", "real_development_parity.json", "REAL_EVAL_DEVELOPMENT is not pristine LOCKED_TEST.")


def _validate_limitations(doc: Mapping[str, Any], errors: list[dict[str, str]], warnings: list[dict[str, str]]) -> None:
    expected = {"phase": PHASE_ID, "near_duplicate_pairs_train_validation": EXPECTED_NEAR_DUPLICATE_PAIRS, "locked_test_available": False, "real_role": "REAL_EVAL_DEVELOPMENT_NOT_LOCKED_TEST", "human_fall_semantics": "LYING_TO_HUMAN_FALL_DERIVED_POSTURE_PROXY_NOT_TEMPORAL_EVENT_GROUND_TRUTH", "temporal_event_parity": "NOT_EVALUATED_FRAME_LEVEL_ONLY", "subject_session_event_generalization": "NOT_VERIFIABLE", "thermal44_validation": "NOT_PERFORMED_DEFERRED_TO_T-C", "pi_latency": "NOT_MEASURED", "absolute_equivalence_threshold": "NOT_PREEXISTING", "next_phase_started": False, "t_b5_started": False, "t_c_started": False}
    for key, value in expected.items():
        if doc.get(key) != value:
            _error(errors, "LIMITATION_LOST", f"limitations.json:{key}", f"Expected {value!r}.")
    if doc.get("former_fp32_artifact") != "TFLITE_DYNAMIC_RANGE_DIAGNOSTIC_ONLY" or doc.get("former_fp32_size_anomaly") != "OPTIMIZE_DEFAULT_DYNAMIC_RANGE_QUANTIZED_FLOAT_IO" or doc.get("true_fp32_equivalence_stage") != "UNQUANTIZED_NO_OPTIMIZATIONS":
        _error(errors, "FP32_RECLASSIFICATION_LIMITATION_LOST", "limitations.json", "The former dynamic-range size anomaly and corrected FP32 policy must remain explicit.")
    if not math.isclose(float(doc.get("synthetic_real_gap", float("nan"))), 0.9951295332536425 - 0.593926523563344, abs_tol=1e-12):
        _error(errors, "REAL_GAP_INVALID", "limitations.json:synthetic_real_gap", "Inherited synthetic-REAL gap changed.")
    _warning(warnings, "POSTURE_PROXY", "limitations.json", "HUMAN_FALL remains a Lying-derived posture proxy, not temporal fall ground truth.")
    _warning(warnings, "GROUPING_NOT_VERIFIABLE", "limitations.json", "Subject/session/event generalization is not verifiable.")
    _warning(warnings, "THERMAL44_DEFERRED", "limitations.json", "Thermal-44 validation remains deferred to T-C.")


def _validate_execution(documents: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    env = documents.get("execution_environment.json", {}); summary = documents.get("execution_summary.json", {})
    if env.get("phase") != PHASE_ID or env.get("gpu_required") is not False:
        _error(errors, "EXECUTION_ENVIRONMENT_INVALID", "execution_environment.json", "Execution environment is invalid.")
    if summary.get("phase") != PHASE_ID or summary.get("status") != "FINALIZED" or summary.get("mode") != FULL_MODE or summary.get("conversion_performed") is not True or summary.get("retraining_performed") is not False or summary.get("calibration_samples") != CALIBRATION_COUNT or summary.get("t_b5_started") is not False or summary.get("t_c_started") is not False or summary.get("candidate_changed") is not False:
        _error(errors, "EXECUTION_SCOPE_INVALID", "execution_summary.json", "Full T-B4 scope/status is invalid.")
    if summary.get("correction_performed") is not True or summary.get("true_fp32_generated") is not True or summary.get("former_fp32_reclassified") != "TFLITE_DYNAMIC_RANGE" or summary.get("full_int8_preserved") is not True or summary.get("former_fp32_sha256") != FORMER_DYNAMIC_RANGE_SHA or summary.get("former_fp32_size_bytes") != FORMER_DYNAMIC_RANGE_SIZE:
        _error(errors, "CORRECTION_SCOPE_INVALID", "execution_summary.json", "T-B4 correction identity/preservation evidence is missing.")


def validate_evidence(*, repo_root: Path = ROOT, evidence_dir: Path | None = None, mode: str = FULL_MODE, check_checksums: bool = True) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve(); evidence = Path(evidence_dir or repo_root / EVIDENCE_REL).resolve()
    errors: list[dict[str, str]] = []; warnings: list[dict[str, str]] = []
    full = mode == FULL_MODE; names = FULL_JSON if full else BASE_JSON
    documents = _read_documents(evidence, names, errors)
    live = _validate_predecessors(repo_root, errors)
    if all(name in documents for name in BASE_JSON):
        _validate_protocol(documents["t_b4_protocol.json"], errors)
        _validate_predecessor_identity(documents["predecessor_identity.json"], live, errors)
        _validate_candidate(documents["float_candidate_lock.json"], errors)
        _validate_p1(documents["p1_lock.json"], errors)
        _validate_dataset(documents["dataset_lock.json"], errors, warnings)
        _validate_storage(documents["artifact_storage_policy.json"], errors)
        _validate_calibration(documents["representative_calibration_policy.json"], documents["representative_sample_manifest.json"], errors)
        _validate_temperature(documents["temperature_range_policy.json"], errors)
        _validate_readiness(documents["readiness_result.json"], errors)
    if full and all(name in documents for name in FULL_JSON):
        _validate_artifacts(documents, errors)
        for name in ("float_vs_tflite_fp32_parity.json", "float_vs_int8_parity.json", "fp32_vs_int8_parity.json"):
            _validate_parity(documents[name], name, errors)
        _validate_saturation(documents, errors, warnings)
        _validate_temperature_error(documents["temperature_range_error.json"], documents["temperature_range_policy.json"], errors)
        _validate_real(documents["real_development_parity.json"], errors, warnings)
        _validate_limitations(documents["limitations.json"], errors, warnings)
        _validate_execution(documents, errors)
    if check_checksums:
        _validate_checksums(evidence, set(names), errors)
    errors.sort(key=lambda item: (item["code"], item["location"], item["message"])); warnings.sort(key=lambda item: (item["code"], item["location"], item["message"]))
    passed = not errors and all(live.get(phase, {}).get("evidence_validation") == "PASS" for phase in ("T-A6", "T-B0", "T-B1", "T-B2", "T-B3"))
    correction = {}
    if full and "tflite_fp32_artifact.json" in documents:
        fp32 = documents["tflite_fp32_artifact.json"]
        correction = {"true_fp32_artifact_id": "TFLITE_FP32", "true_fp32_sha256": fp32.get("sha256"), "true_fp32_size_bytes": fp32.get("size_bytes"), "former_artifact_id": "TFLITE_DYNAMIC_RANGE", "former_artifact_sha256": FORMER_DYNAMIC_RANGE_SHA, "former_artifact_size_bytes": FORMER_DYNAMIC_RANGE_SIZE, "official_equivalence_chain": ["FLOAT_KERAS", "TFLITE_FP32", "FULL_INT8"]}
    return {"phase": PHASE_ID, "mode": mode, "schema_version": "1.0", "evidence_validation": "PASS" if passed else "FAIL", "overall_outcome": "T_B4_COMPLETE_WITH_LIMITATIONS" if passed else "T_B4_BLOCKED", "t_b5_authorized": "YES_WITH_LIMITATIONS" if passed else False, "error_count": len(errors), "errors": errors, "warning_count": len(warnings), "warnings": warnings, "correction": correction, "predecessors": {phase: {"evidence_validation": item.get("evidence_validation"), "overall_outcome": item.get("overall_outcome")} for phase, item in sorted(live.items())}}


def _write_result(evidence: Path, result: Mapping[str, Any], required: Iterable[str]) -> None:
    target = evidence / "validation_result.json"; target.write_text(canonical_json(result), encoding="utf-8")
    _entries = []
    for path in sorted(evidence.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != CHECKSUMS and not path.name.startswith("._"):
            _entries.append(f"{sha256_file(path)}  {path.name}")
    (evidence / CHECKSUMS).write_text("\n".join(_entries) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Thermal T-B4 Float/TFLite/INT8 evidence")
    parser.add_argument("--repo-root", default=str(ROOT)); parser.add_argument("--evidence-dir", default=str(ROOT / EVIDENCE_REL)); parser.add_argument("--mode", choices=(READINESS_MODE, FULL_MODE), default=FULL_MODE); parser.add_argument("--skip-checksums", action="store_true"); parser.add_argument("--write-result", action="store_true")
    args = parser.parse_args(); evidence = Path(args.evidence_dir); result = validate_evidence(repo_root=Path(args.repo_root), evidence_dir=evidence, mode=args.mode, check_checksums=not args.skip_checksums)
    if args.write_result:
        _write_result(evidence, result, FULL_JSON if args.mode == FULL_MODE else BASE_JSON)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)); return 0 if result["evidence_validation"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
