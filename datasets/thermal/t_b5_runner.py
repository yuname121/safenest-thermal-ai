"""SafeNest Thermal T-B5 offline robustness, latency, and candidate lock.

T-B5 consumes the immutable T-B1/T-B4 chain.  It never trains, recalibrates,
converts, changes P1, or touches the production model.  The full experiment
uses only the official VALIDATION role for selection/robustness and records
REAL_EVAL_DEVELOPMENT as diagnostic context only.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np

from datasets.thermal.t_b1_model import create_small_cnn_baseline, require_tensorflow
from datasets.thermal.t_b1_preprocessing import (
    CLASS_ORDER,
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
from datasets.thermal.t_b4_runner import (
    _keras_predict,
    _load_p1,
    _parity,
    _quantize_input,
    _tflite_predict,
)


PHASE_ID = "T-B5"
READINESS_MODE = "READINESS"
FULL_MODE = "FULL_EXPERIMENT"
EVIDENCE_REL = "datasets/thermal/manifests/T-B5_robustness_latency_candidate_lock"
TA6_REL = "datasets/thermal/manifests/T-A6_execution_result"
TB0_REL = "datasets/thermal/manifests/T-B0_offline_model_protocol"
TB1_REL = "datasets/thermal/manifests/T-B1_full_experiment"
TB2_REL = "datasets/thermal/manifests/T-B2_architecture_comparison"
TB3_REL = "datasets/thermal/manifests/T-B3_frame_multiseed_confirmation"
TB4_REL = "datasets/thermal/manifests/T-B4_tflite_int8_equivalence"
P1_PROFILE = "P1_TRAIN_FITTED_GLOBAL_ZSCORE"
EXPECTED_P1_CHECKSUM = "10b5da044ef33f26715544c3d4bf56e9d999d6c65139e931f6997583b3f5b816"
EXPECTED_ARCHITECTURE_FINGERPRINT = "937fceb88900a779d67a8e407cebc3362cc21346721f92cea5ae8de1413aba2a"
EXPECTED_CHECKPOINT_SHA = "7aba32fe8d0e241546429bdc3e8cd059b10d4d8f548e9e12c4085abeba308a75"
EXPECTED_CHECKPOINT_SIZE = 3777416
EXPECTED_FP32_SHA = "fbe891520f07e0534b1a7074dc819d8ed44bca58b27e35c78916c3ddae73a779"
EXPECTED_FP32_SIZE = 1252048
EXPECTED_INT8_SHA = "fa9730c29535477a3994c11e664474a0ca0116afaaa172889f47446ab2ac46be"
EXPECTED_INT8_SIZE = 318280
FORMER_DYNAMIC_RANGE_SHA = "297de231e26ecf2d4cd4010bd10c08d4df3b6b0a531c69693daea353afb8127d"
FORMER_DYNAMIC_RANGE_SIZE = 317344
ROBUSTNESS_SEED = 20260815
ROBUSTNESS_SAMPLE_COUNT = 512
LATENCY_SAMPLE_COUNT = 64
LATENCY_WARMUP = 20
LATENCY_ITERATIONS = 200
PROTOCOL_ID = "THERMAL_T_B5_OFFLINE_ROBUSTNESS_LATENCY_CANDIDATE_LOCK_001"
PROFILE_ID = "THERMAL_T_B5_ROBUSTNESS_PROFILE_001"
REPORT_NAME = "T-B5_offline_report.md"


class RunnerContractError(RuntimeError):
    """Raised for fail-closed T-B5 contract violations."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(canonical_json(value), encoding="utf-8")
    os.replace(temporary, path)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _repo_commit(repo_root: Path, ref: str = "HEAD") -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(repo_root), "rev-parse", ref], check=True, capture_output=True, text=True)
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _portable(value: str) -> bool:
    lower = value.lower()
    return not (value.startswith(("/", "~/", "file://")) or "\\" in value or "/users/" in lower or "/private/" in lower or value.startswith(("/volumes/", "/content/")))


def _write_checksums(directory: Path) -> None:
    rows: list[str] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "checksums.sha256" and not path.name.startswith("._") and not path.name.endswith(".partial"):
            rows.append(f"{sha256_file(path)}  {path.name}")
    (directory / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _run_predecessors(repo_root: Path) -> dict[str, Any]:
    from scripts.validate_thermal_t_a6 import validate_evidence as a6
    from scripts.validate_thermal_t_b0 import validate_evidence as b0
    from scripts.validate_thermal_t_b1 import validate_evidence as b1
    from scripts.validate_thermal_t_b2 import validate_evidence as b2
    from scripts.validate_thermal_t_b3 import validate_evidence as b3
    from scripts.validate_thermal_t_b4 import validate_evidence as b4

    results = {
        "T-A6": a6(repo_root=repo_root, evidence_dir=repo_root / TA6_REL, mode="FULL_DATASET", check_checksums=True),
        "T-B0": b0(repo_root=repo_root, evidence_dir=repo_root / TB0_REL, check_checksums=True),
        "T-B1": b1(repo_root=repo_root, evidence_dir=repo_root / TB1_REL, mode="FULL_EXPERIMENT", check_checksums=True),
        "T-B2": b2(repo_root=repo_root, evidence_dir=repo_root / TB2_REL, mode="FULL_EXPERIMENT", check_checksums=True),
        "T-B3": b3(repo_root=repo_root, evidence_dir=repo_root / TB3_REL, mode="FULL_EXPERIMENT", check_checksums=True),
        "T-B4": b4(repo_root=repo_root, evidence_dir=repo_root / TB4_REL, mode="FULL_EXPERIMENT", check_checksums=True),
    }
    for phase, result in results.items():
        if result.get("evidence_validation") != "PASS":
            raise RunnerContractError(f"T_B5_BLOCKED_PREDECESSOR_INVALID:{phase}")
    return results


def _verify_branch(repo_root: Path) -> dict[str, Any]:
    branch = subprocess.run(["git", "-C", str(repo_root), "branch", "--show-current"], check=True, capture_output=True, text=True).stdout.strip()
    if "t-b5" not in branch.lower() or not branch.startswith(("feature/", "codex/")):
        raise RunnerContractError("T_B5_BRANCH_IDENTITY_INVALID")
    status = subprocess.run(["git", "-C", str(repo_root), "status", "--short"], check=True, capture_output=True, text=True).stdout.strip()
    return {"branch": branch, "head": _repo_commit(repo_root), "origin_main": _repo_commit(repo_root, "origin/main"), "clean": not bool(status)}


def _sample_indices(rows: int = 8000, count: int = ROBUSTNESS_SAMPLE_COUNT) -> list[int]:
    if rows < count:
        raise RunnerContractError("T_B5_VALIDATION_SAMPLE_COUNT_INVALID")
    values = np.linspace(0, rows - 1, count, dtype=np.int64)
    if len(set(int(item) for item in values)) != count:
        raise RunnerContractError("T_B5_VALIDATION_SAMPLE_INDICES_NOT_UNIQUE")
    return [int(item) for item in values]


def _profile() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "phase": PHASE_ID,
        "profile_id": PROFILE_ID,
        "selection_role": "VALIDATION",
        "sample_count": ROBUSTNESS_SAMPLE_COUNT,
        "sample_index_algorithm": "numpy.linspace(0, VALIDATION_rows-1, 512, dtype=int64)",
        "seed": ROBUSTNESS_SEED,
        "families": [
            {"family_id": "AMBIENT_OFFSET", "classification": "MODEL_INPUT_PERTURBATION", "levels": [-2.0, 2.0], "unit": "Celsius", "injection": "canonical_celsius_before_P1", "device_claim": "NOT_THERMAL44_VALIDATION"},
            {"family_id": "DEAD_PIXEL", "classification": "MODEL_INPUT_PERTURBATION", "levels": [0.01, 0.05], "unit": "fraction", "replacement": "per_frame_mean_celsius", "mask_seed": ROBUSTNESS_SEED, "device_claim": "SYNTHETIC_MASK_ONLY"},
            {"family_id": "PARTIAL_OCCLUSION", "classification": "MODEL_INPUT_PERTURBATION", "levels": [0.05, 0.15], "unit": "frame_area_fraction", "mask": "deterministic_centered_rectangle_with_seeded_offset", "fill": "per_frame_mean_celsius", "device_claim": "SYNTHETIC_MASK_ONLY"},
            {"family_id": "HOT_OBJECT", "classification": "MODEL_INPUT_PERTURBATION", "levels": [5.0, 10.0], "unit": "Celsius_delta", "region": "deterministic_centered_rectangle_with_seeded_offset", "device_claim": "SYNTHETIC_HOT_REGION_ONLY"},
            {"family_id": "MISSING_FRAME", "classification": "PIPELINE_CONTRACT_FAULT", "levels": ["single_frame"], "inference": "FORBIDDEN", "policy": "FAIL_CLOSED_NO_ZERO_LAST_MEAN_REPLACEMENT", "device_claim": "RUNTIME_CONTRACT_ONLY"},
            {"family_id": "ORIENTATION_ERROR", "classification": "MODEL_INPUT_PERTURBATION", "levels": ["horizontal_flip", "vertical_flip", "rotate_180"], "transform": "numpy_deterministic_geometric_transform", "device_claim": "THERMAL44_ORIENTATION_DEFERRED_TO_T-C"},
        ],
        "unsupported_temporal_faults": ["timestamp_jitter", "sequence_reorder", "temporal_dropout", "motion_burst"],
        "absolute_pass_threshold": "NO_PREEXISTING_ABSOLUTE_PASS_THRESHOLD",
        "real_role_used": False,
        "locked_test_available": False,
        "production_runtime_changed": False,
    }


def _candidate_documents() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "phase": PHASE_ID,
        "selection_role": "VALIDATION",
        "candidates": [
            {"candidate_id": "FLOAT_KERAS", "stage": "F0", "artifact": "P1_TRAIN_FITTED_GLOBAL_ZSCORE_FLOAT_CHECKPOINT", "eligible": True, "real_used_for_selection": False},
            {"candidate_id": "TFLITE_FP32", "stage": "F1", "artifact_id": "TFLITE_FP32", "sha256": EXPECTED_FP32_SHA, "size_bytes": EXPECTED_FP32_SIZE, "eligible": True, "real_used_for_selection": False, "conversion_policy": {"optimizations": [], "representative_dataset_attached": False, "float16_enabled": False, "dynamic_range_quantization": False, "quantization_mode": "NONE", "builtin_only": True}},
            {"candidate_id": "FULL_INT8", "stage": "F2", "artifact_id": "FULL_INT8", "sha256": EXPECTED_INT8_SHA, "size_bytes": EXPECTED_INT8_SIZE, "eligible": True, "real_used_for_selection": False, "conversion_policy": {"optimizations": ["DEFAULT"], "representative_dataset_attached": True, "float16_enabled": False, "dynamic_range_quantization": False, "quantization_mode": "FULL_INT8", "builtin_only": True}},
            {"candidate_id": "TFLITE_DYNAMIC_RANGE", "stage": "DIAGNOSTIC_ONLY", "sha256": FORMER_DYNAMIC_RANGE_SHA, "size_bytes": FORMER_DYNAMIC_RANGE_SIZE, "eligible": False, "reason": "DYNAMIC_RANGE_DIAGNOSTIC_ONLY"},
        ],
        "frozen_model_contract": {"architecture": "SMALL_CNN_BASELINE_V1", "architecture_fingerprint": EXPECTED_ARCHITECTURE_FINGERPRINT, "seed": 20260813, "p1_profile": P1_PROFILE, "representative_calibration_fit_role": "TRAIN", "retraining": False, "recalibration": False, "float_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA, "float_checkpoint_size_bytes": EXPECTED_CHECKPOINT_SIZE},
    }


def _artifact_registry() -> dict[str, Any]:
    return {
        "schema_version": "1.0", "phase": PHASE_ID, "binary_storage": "EXTERNAL_SSD_ONLY", "binaries_tracked_in_git": False,
        "artifacts": [
            {"candidate_id": "FLOAT_KERAS", "logical_path": "T-B1/checkpoints/P1_TRAIN_FITTED_GLOBAL_ZSCORE.weights.h5", "sha256": EXPECTED_CHECKPOINT_SHA, "size_bytes": EXPECTED_CHECKPOINT_SIZE, "eligible": True},
            {"candidate_id": "TFLITE_FP32", "logical_path": "T-B4/artifacts/SMALL_CNN_BASELINE_V1_P1_float32.tflite", "sha256": EXPECTED_FP32_SHA, "size_bytes": EXPECTED_FP32_SIZE, "eligible": True},
            {"candidate_id": "FULL_INT8", "logical_path": "T-B4/artifacts/SMALL_CNN_BASELINE_V1_P1_full_int8.tflite", "sha256": EXPECTED_INT8_SHA, "size_bytes": EXPECTED_INT8_SIZE, "eligible": True},
            {"candidate_id": "TFLITE_DYNAMIC_RANGE", "logical_path": "T-B4/artifacts/SMALL_CNN_BASELINE_V1_P1_dynamic_range.tflite", "sha256": FORMER_DYNAMIC_RANGE_SHA, "size_bytes": FORMER_DYNAMIC_RANGE_SIZE, "eligible": False, "diagnostic_only": True},
        ],
    }


def _latency_protocol() -> dict[str, Any]:
    return {
        "schema_version": "1.0", "phase": PHASE_ID, "protocol_id": "THERMAL_T_B5_MAC_LATENCY_001", "host_scope": "MAC_HOST_ONLY",
        "input_role": "VALIDATION", "sample_count": LATENCY_SAMPLE_COUNT, "warmup_iterations": LATENCY_WARMUP, "measured_iterations": LATENCY_ITERATIONS,
        "threads": 1, "batch_size": 1, "delegate": "CPU_XNNPACK_IF_AVAILABLE", "gpu_delegate": False,
        "timers": {"invoke_only": "interpreter.set_tensor+invoke+get_tensor", "preprocess_plus_invoke": "canonical_frame_to_P1_or_INT8_input_plus_invoke"},
        "units": "microseconds", "pi_latency": "NOT_MEASURED", "thermal44_end_to_end": "NOT_MEASURED_DEFERRED_TO_T-C", "posthoc_tuning": False,
    }


def _candidate_lock_policy() -> dict[str, Any]:
    return {
        "schema_version": "1.0", "phase": PHASE_ID, "policy_id": "THERMAL_T_B5_OFFLINE_CANDIDATE_LOCK_001", "frozen_before_results": True,
        "selection_role": "VALIDATION", "real_role": "REAL_EVAL_DEVELOPMENT", "real_used_for_selection": False, "locked_test_required": False,
        "eligible_candidates": ["FLOAT_KERAS", "TFLITE_FP32", "FULL_INT8"], "ineligible_candidates": {"TFLITE_DYNAMIC_RANGE": "DIAGNOSTIC_ONLY"},
        "rule": "Prefer FULL_INT8 when artifact identity, VALIDATION robustness execution, and Mac latency evidence are complete; otherwise prefer TRUE TFLITE_FP32; never select on REAL or deploy claim.",
        "absolute_robustness_threshold": "NO_PREEXISTING_ABSOLUTE_PASS_THRESHOLD",
        "lock_status_allowed": ["OFFLINE_INT8_CANDIDATE_LOCKED_WITH_LIMITATIONS", "OFFLINE_FP32_CANDIDATE_LOCKED_WITH_LIMITATIONS", "BLOCKED"],
        "deployment_status": "NOT_THERMAL44_VALIDATED",
    }


def run_readiness(*, repo_root: str | Path, canonical_root: str | Path, work_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(); canonical = Path(canonical_root).expanduser(); output = Path(output_root).expanduser()
    branch = _verify_branch(root)
    predecessors = _run_predecessors(root)
    if not canonical.is_dir():
        raise RunnerContractError("T_B5_BLOCKED_CANONICAL_ROOT_UNAVAILABLE")
    dataset = validate_canonical_root(canonical, full_hash=False)
    profile = _profile(); profile_checksum = _sha256_text(canonical_json(profile))
    evidence = root / EVIDENCE_REL; evidence.mkdir(parents=True, exist_ok=True)
    documents = {
        "t_b5_protocol.json": {"schema_version": "1.0", "phase": PHASE_ID, "protocol_id": PROTOCOL_ID, "mode": "READINESS", "training_performed": False, "recalibration_performed": False, "conversion_performed": False, "production_model_changed": False, "t_c_started": False},
        "predecessor_identity.json": {"schema_version": "1.0", "phase": PHASE_ID, "predecessors": {key: {"evidence_validation": value.get("evidence_validation"), "overall_outcome": value.get("overall_outcome")} for key, value in sorted(predecessors.items())}},
        "dataset_lock.json": {"schema_version": "1.0", "phase": PHASE_ID, "roles": dataset, "selection_role": "VALIDATION", "real_role": "REAL_EVAL_DEVELOPMENT", "validation_sample_indices": _sample_indices(), "validation_sample_count": ROBUSTNESS_SAMPLE_COUNT, "real_used_for_selection": False, "locked_test_available": False, "p1_statistics_checksum": EXPECTED_P1_CHECKSUM},
        "candidate_set.json": _candidate_documents(),
        "artifact_registry.json": _artifact_registry(),
        "robustness_profile.json": profile | {"profile_checksum": profile_checksum},
        "latency_protocol.json": _latency_protocol(),
        "candidate_lock_policy.json": _candidate_lock_policy(),
        "readiness_result.json": {"schema_version": "1.0", "phase": PHASE_ID, "status": "T_B5_FULL_EXPERIMENT_READY", "repo_commit": branch["head"], "origin_main": branch["origin_main"], "predecessors_pass": True, "profile_checksum": profile_checksum, "canonical_payloads_external": True, "output_root": "EXTERNAL_SSD_SAFE_NEST_AI_THERMAL_EXPERIMENTS_T_B5", "full_execution_authorized": True, "t_b6_started": False, "t_c_started": False},
    }
    for name, value in documents.items():
        _write_json(evidence / name, value)
    _write_checksums(evidence)
    return {"phase": PHASE_ID, "mode": READINESS_MODE, "status": "T_B5_FULL_EXPERIMENT_READY", "evidence_dir": EVIDENCE_REL, "profile_checksum": profile_checksum, "predecessors": {key: {"evidence_validation": value.get("evidence_validation"), "overall_outcome": value.get("overall_outcome")} for key, value in sorted(predecessors.items())}}


def _rect_bounds(height: int, width: int, fraction: float, index: int, seed: int) -> tuple[int, int, int, int]:
    area = max(1, int(round(height * width * fraction)))
    rh = max(1, int(round(np.sqrt(area * height / width))))
    rw = max(1, int(round(area / rh)))
    rh, rw = min(height, rh), min(width, rw)
    rng = np.random.default_rng(seed + index * 7919)
    top = int(rng.integers(0, max(1, height - rh + 1))); left = int(rng.integers(0, max(1, width - rw + 1)))
    return top, left, rh, rw


def _perturb(frames: np.ndarray, family: str, level: Any, *, seed: int) -> np.ndarray:
    result = np.asarray(frames, dtype=np.float32).copy()
    height, width = result.shape[1:]
    if family == "AMBIENT_OFFSET":
        return result + np.float32(level)
    if family == "DEAD_PIXEL":
        for index, row in enumerate(result):
            rng = np.random.default_rng(seed + index * 7919 + int(round(float(level) * 1000)))
            mask = rng.random(row.shape) < float(level)
            row[mask] = np.float32(row.mean())
        return result
    if family == "PARTIAL_OCCLUSION":
        for index, row in enumerate(result):
            top, left, rh, rw = _rect_bounds(height, width, float(level), index, seed)
            row[top : top + rh, left : left + rw] = np.float32(row.mean())
        return result
    if family == "HOT_OBJECT":
        for index, row in enumerate(result):
            top, left, rh, rw = _rect_bounds(height, width, 0.10, index, seed)
            row[top : top + rh, left : left + rw] += np.float32(level)
        return result
    if family == "ORIENTATION_ERROR":
        if level == "horizontal_flip": return np.flip(result, axis=2).copy()
        if level == "vertical_flip": return np.flip(result, axis=1).copy()
        if level == "rotate_180": return np.rot90(result, 2, axes=(1, 2)).copy()
    raise RunnerContractError(f"T_B5_UNKNOWN_PERTURBATION:{family}:{level}")


def _metrics_record(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    return compute_metrics(y_true, np.argmax(probabilities, axis=1))


def _prediction_digest(probabilities: np.ndarray) -> str:
    return _sha256_bytes(np.asarray(probabilities, dtype="<f4").tobytes(order="C"))


def _load_interpreter(model_bytes: bytes) -> tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    tf = require_tensorflow(); interpreter = tf.lite.Interpreter(model_content=model_bytes, num_threads=1); interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details()[0], interpreter.get_output_details()[0]


def _invoke_one(interpreter: Any, input_detail: Mapping[str, Any], output_detail: Mapping[str, Any], row: np.ndarray, *, quantized: bool) -> np.ndarray:
    if quantized:
        values, _ = _quantize_input(row, input_detail); interpreter.set_tensor(input_detail["index"], values)
    else:
        interpreter.set_tensor(input_detail["index"], np.asarray(row, dtype=np.float32))
    interpreter.invoke(); raw = np.asarray(interpreter.get_tensor(output_detail["index"])).copy()
    if quantized:
        scale, zero = output_detail.get("quantization", (0.0, 0));
        if float(scale) <= 0: raise RunnerContractError("T_B5_INT8_OUTPUT_QUANTIZATION_INVALID")
        return ((raw.astype(np.float32) - float(zero)) * float(scale)).reshape(-1)
    return raw.astype(np.float32).reshape(-1)


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    percentiles = np.percentile(array, [90, 95, 99])
    return {"sample_count": int(array.size), "mean_us": float(array.mean()), "median_us": float(np.median(array)), "std_us": float(array.std()), "min_us": float(array.min()), "p90_us": float(percentiles[0]), "p95_us": float(percentiles[1]), "p99_us": float(percentiles[2]), "max_us": float(array.max())}


def _latency_candidate(model_bytes: bytes, frames: np.ndarray, *, quantized: bool, p1: P1Statistics) -> dict[str, Any]:
    interpreter, input_detail, output_detail = _load_interpreter(model_bytes)
    transformed = apply_p1(frames, p1)
    for index in range(LATENCY_WARMUP):
        row = transformed[index % transformed.shape[0] : index % transformed.shape[0] + 1]
        _invoke_one(interpreter, input_detail, output_detail, row, quantized=quantized)
    invoke_values: list[float] = []; end_to_end_values: list[float] = []
    for index in range(LATENCY_ITERATIONS):
        row = transformed[index % transformed.shape[0] : index % transformed.shape[0] + 1]
        start = time.perf_counter_ns(); _invoke_one(interpreter, input_detail, output_detail, row, quantized=quantized); invoke_values.append((time.perf_counter_ns() - start) / 1000.0)
        raw_frame = frames[index % frames.shape[0] : index % frames.shape[0] + 1]
        start = time.perf_counter_ns(); prepared = apply_p1(raw_frame, p1); _invoke_one(interpreter, input_detail, output_detail, prepared, quantized=quantized); end_to_end_values.append((time.perf_counter_ns() - start) / 1000.0)
    return {"invoke_only": _summary(invoke_values), "preprocess_plus_invoke": _summary(end_to_end_values), "input_dtype": np.dtype(input_detail["dtype"]).name, "output_dtype": np.dtype(output_detail["dtype"]).name, "threads": 1, "delegate": "CPU_XNNPACK_IF_AVAILABLE"}


def _artifact_paths(canonical: Path) -> tuple[Path, Path]:
    bundle = canonical.parent / "experiments" / "T-B4" / "T-B4_execution_result" / "artifacts"
    return bundle / "SMALL_CNN_BASELINE_V1_P1_float32.tflite", bundle / "SMALL_CNN_BASELINE_V1_P1_full_int8.tflite"


def _checkpoint_path(canonical: Path) -> Path:
    return canonical.parent / "experiments" / "T-B1" / "T-B1_execution_result" / "checkpoints" / "P1_TRAIN_FITTED_GLOBAL_ZSCORE.weights.h5"


def run_full(*, repo_root: str | Path, canonical_root: str | Path, work_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(); canonical = Path(canonical_root).expanduser(); output = Path(output_root).expanduser(); evidence = root / EVIDENCE_REL
    readiness = _read_json(evidence / "readiness_result.json")
    profile_doc = _read_json(evidence / "robustness_profile.json"); profile = dict(profile_doc); supplied_checksum = profile.pop("profile_checksum", None)
    if supplied_checksum != _sha256_text(canonical_json(profile)):
        raise RunnerContractError("T_B5_PROFILE_CHECKSUM_MISMATCH")
    predecessors = _run_predecessors(root)
    if readiness.get("status") != "T_B5_FULL_EXPERIMENT_READY": raise RunnerContractError("T_B5_READINESS_MISSING")
    role = resolve_role_files(canonical, "VALIDATION"); validation = np.load(role.array_path, mmap_mode="r", allow_pickle=False); labels, _ = labels_from_provenance(role.provenance_path, EXPECTED_ROLES["VALIDATION"]["rows"])
    indices = np.asarray(_sample_indices(), dtype=np.int64); frames = np.asarray(validation[indices], dtype=np.float32); y_true = labels[indices]
    p1 = _load_p1(root); checkpoint = _checkpoint_path(canonical)
    if not checkpoint.is_file() or checkpoint.stat().st_size != EXPECTED_CHECKPOINT_SIZE or sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA: raise RunnerContractError("T_B5_CHECKPOINT_IDENTITY_MISMATCH")
    model = create_small_cnn_baseline(); model.load_weights(str(checkpoint)); clean_input = apply_p1(frames, p1); clean_probs = {"FLOAT_KERAS": _keras_predict(model, clean_input)}
    fp32_path, int8_path = _artifact_paths(canonical)
    for path, size, digest in ((fp32_path, EXPECTED_FP32_SIZE, EXPECTED_FP32_SHA), (int8_path, EXPECTED_INT8_SIZE, EXPECTED_INT8_SHA)):
        if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest: raise RunnerContractError(f"T_B5_ARTIFACT_IDENTITY_MISMATCH:{path.name}")
    fp32_bytes, int8_bytes = fp32_path.read_bytes(), int8_path.read_bytes(); clean_probs["TFLITE_FP32"], _ = _tflite_predict(fp32_bytes, clean_input, quantized=False); clean_probs["FULL_INT8"], _ = _tflite_predict(int8_bytes, clean_input, quantized=True)
    clean = {candidate: _metrics_record(y_true, probs) | {"prediction_digest": _prediction_digest(probs)} for candidate, probs in clean_probs.items()}
    robustness_rows: list[dict[str, Any]] = []
    for family in profile["families"]:
        family_id = str(family["family_id"])
        for level in family["levels"]:
            if family_id == "MISSING_FRAME":
                robustness_rows.append({"family_id": family_id, "classification": family["classification"], "level": level, "source_role": "VALIDATION", "status": "PIPELINE_CONTRACT_FAULT_FAIL_CLOSED", "model_inference_performed": False, "replacement": "NONE", "predictions": None, "real_used_for_selection": False})
                continue
            perturbed = _perturb(frames, family_id, level, seed=ROBUSTNESS_SEED); prepared = apply_p1(perturbed, p1)
            probabilities = {"FLOAT_KERAS": _keras_predict(model, prepared)}
            probabilities["TFLITE_FP32"], _ = _tflite_predict(fp32_bytes, prepared, quantized=False); probabilities["FULL_INT8"], _ = _tflite_predict(int8_bytes, prepared, quantized=True)
            metrics = {candidate: _metrics_record(y_true, probs) | {"prediction_digest": _prediction_digest(probs)} for candidate, probs in probabilities.items()}
            deltas = {candidate: {key: float(metrics[candidate][key] - clean[candidate][key]) for key in ("macro_f1", "accuracy", "balanced_accuracy", "h_fall_posture_proxy_recall")} for candidate in metrics}
            parity = {"FLOAT_KERAS__TFLITE_FP32": _parity(probabilities["FLOAT_KERAS"], probabilities["TFLITE_FP32"], y_true), "TFLITE_FP32__FULL_INT8": _parity(probabilities["TFLITE_FP32"], probabilities["FULL_INT8"], y_true), "FLOAT_KERAS__FULL_INT8": _parity(probabilities["FLOAT_KERAS"], probabilities["FULL_INT8"], y_true)}
            robustness_rows.append({"family_id": family_id, "classification": family["classification"], "level": level, "source_role": "VALIDATION", "status": "MEASURED", "sample_count": ROBUSTNESS_SAMPLE_COUNT, "model_inference_performed": True, "metrics": metrics, "delta_vs_clean": deltas, "cross_artifact_parity": parity, "real_used_for_selection": False})
    latency_frames = frames[:LATENCY_SAMPLE_COUNT]
    latency = {"schema_version": "1.0", "phase": PHASE_ID, "role": "VALIDATION", "environment": {"platform": platform.platform(), "system": platform.system(), "machine": platform.machine(), "python": platform.python_version(), "tensorflow": str(getattr(require_tensorflow(), "__version__", "unknown")), "threads": 1, "delegate": "CPU_XNNPACK_IF_AVAILABLE", "gpu_delegate": False, "pi_measured": False}, "candidates": {"TFLITE_FP32": _latency_candidate(fp32_bytes, latency_frames, quantized=False, p1=p1), "FULL_INT8": _latency_candidate(int8_bytes, latency_frames, quantized=True, p1=p1)}, "protocol": _latency_protocol()}
    measured = {"robustness_rows": robustness_rows, "clean": clean, "sample_indices": [int(item) for item in indices], "profile_checksum": supplied_checksum}
    lock = {"schema_version": "1.0", "phase": PHASE_ID, "status": "OFFLINE_INT8_CANDIDATE_LOCKED_WITH_LIMITATIONS", "selected_candidate_id": "FULL_INT8", "selection_role": "VALIDATION", "real_used_for_selection": False, "locked_test_used": False, "candidate_sha256": EXPECTED_INT8_SHA, "candidate_size_bytes": EXPECTED_INT8_SIZE, "candidate_stage": "F2", "candidate_lock_policy_id": "THERMAL_T_B5_OFFLINE_CANDIDATE_LOCK_001", "offline_only": True, "thermal44_deployment_validated": False, "pi_latency_validated": False, "rationale": ["full INT8 artifact identity is inherited and verified", "robustness protocol completed on deterministic VALIDATION samples", "Mac host latency completed under frozen protocol", "REAL_EVAL_DEVELOPMENT excluded from selection"], "real_domain_sensitivity": "CARRIED_FORWARD_DIAGNOSTIC_ONLY", "next_phase": "T-C_DEVICE_DOMAIN_PROTOCOL", "t_c_execution_ready": "NO_REAL_CAPTURE_INTAKE_YET"}
    summary = {"schema_version": "1.0", "phase": PHASE_ID, "mode": FULL_MODE, "status": "FINALIZED_WITH_LIMITATIONS", "training_performed": False, "recalibration_performed": False, "conversion_performed": False, "selection_role": "VALIDATION", "real_role": "REAL_EVAL_DEVELOPMENT", "real_used_for_selection": False, "locked_test_available": False, "production_model_changed": False, "t_c_started": False, "t_b6_started": False, "profile_checksum": supplied_checksum, "robustness_case_count": len(robustness_rows), "candidate_lock_status": lock["status"]}
    predecessor_identity = {"schema_version": "1.0", "phase": PHASE_ID, "predecessors": {key: {"evidence_validation": value.get("evidence_validation"), "overall_outcome": value.get("overall_outcome")} for key, value in sorted(predecessors.items())}}
    real_t_b4 = _read_json(root / TB4_REL / "real_development_parity.json")
    documents = {"predecessor_identity.json": predecessor_identity, "artifact_registry.json": _artifact_registry(), "robustness_results.json": {"schema_version": "1.0", "phase": PHASE_ID, "profile_checksum": supplied_checksum, "selection_role": "VALIDATION", "real_used_for_selection": False, "locked_test_available": False, "clean": clean, "cases": robustness_rows}, "parity_summary.json": {"schema_version": "1.0", "phase": PHASE_ID, "clean_cross_artifact": {"FLOAT_KERAS__TFLITE_FP32": _parity(clean_probs["FLOAT_KERAS"], clean_probs["TFLITE_FP32"], y_true), "TFLITE_FP32__FULL_INT8": _parity(clean_probs["TFLITE_FP32"], clean_probs["FULL_INT8"], y_true), "FLOAT_KERAS__FULL_INT8": _parity(clean_probs["FLOAT_KERAS"], clean_probs["FULL_INT8"], y_true)}, "perturbation_cases": len(robustness_rows), "real_used_for_selection": False}, "real_diagnostic_summary.json": {"schema_version": "1.0", "phase": PHASE_ID, "source_phase": "T-B4", "role": "REAL_EVAL_DEVELOPMENT", "used_for_selection": False, "locked_test": False, "int8_sensitivity": {"float_vs_int8_argmax_agreement": real_t_b4.get("float_vs_int8_parity", {}).get("argmax_agreement"), "true_fp32_vs_int8_argmax_agreement": real_t_b4.get("fp32_vs_int8_parity", {}).get("argmax_agreement"), "diagnostic_only": True}}, "latency_results.json": latency, "candidate_lock.json": lock, "execution_summary.json": summary, "evidence_handoff.json": {"schema_version": "1.0", "phase": PHASE_ID, "offline_candidate": "FULL_INT8", "offline_candidate_status": lock["status"], "thermal44_status": "NOT_VALIDATED_DEFERRED_TO_T-C", "pi_status": "NOT_MEASURED", "real_status": "REAL_EVAL_DEVELOPMENT_DIAGNOSTIC_ONLY", "human_fall_semantics": "LYING_DERIVED_POSTURE_PROXY_NOT_TEMPORAL_FALL_GROUND_TRUTH", "no_pristine_locked_test": True}, "limitation_registry.json": {"schema_version": "1.0", "phase": PHASE_ID, "limitations": ["NO_PRISTINE_LOCKED_TEST", "REAL_EVAL_DEVELOPMENT_NOT_SELECTION", "TRAIN_VALIDATION_NEAR_DUPLICATE_PAIRS_14514", "SUBJECT_SESSION_EVENT_GENERALIZATION_NOT_VERIFIABLE", "HUMAN_FALL_POSTURE_PROXY", "SYNTHETIC_PERTURBATIONS_NOT_THERMAL44_VALIDATION", "MAC_LATENCY_NOT_PI_OR_END_TO_END", "MISSING_FRAME_IS_PIPELINE_FAIL_CLOSED_ONLY"], "unsupported_claims": ["THERMAL44_VALIDATED", "PI_LATENCY_MEASURED", "REAL_WORLD_FALL_GROUND_TRUTH", "FINAL_UNBIASED_TEST"]}, "t_b5_execution_result.json": {"schema_version": "1.0", "phase": PHASE_ID, "status": "PASS_WITH_LIMITATIONS_PENDING_VALIDATOR", "profile_checksum": supplied_checksum, "output_root": "EXTERNAL_SSD_SAFE_NEST_AI_THERMAL_EXPERIMENTS_T_B5", "canonical_payloads_tracked": False}}
    for name, value in documents.items(): _write_json(evidence / name, value)
    report_lines = [
        "# Thermal T-B5 — Offline Robustness, Mac Latency, Candidate Lock",
        "",
        "- Status: `T_B5_COMPLETE_WITH_LIMITATIONS` (compact validator required)",
        "- Selection role: `VALIDATION` only; `REAL_EVAL_DEVELOPMENT` was diagnostic only.",
        "- Candidate lock: `FULL_INT8` / `OFFLINE_INT8_CANDIDATE_LOCKED_WITH_LIMITATIONS`.",
        "- LOCKED_TEST: unavailable; no final unbiased test claim.",
        "",
        "## Frozen contract",
        "",
        f"- Profile: `{PROFILE_ID}`; checksum `{supplied_checksum}`; deterministic VALIDATION sample count: {ROBUSTNESS_SAMPLE_COUNT}.",
        "- Architecture: `SMALL_CNN_BASELINE_V1`; P1 is TRAIN-fitted; seed `20260813`; retraining/recalibration/conversion: none.",
        "- Dynamic-range TFLite is diagnostic-only and ineligible for the equivalence chain.",
        "",
        "## Clean VALIDATION metrics (same deterministic subset)",
        "",
    ]
    for candidate in ("FLOAT_KERAS", "TFLITE_FP32", "FULL_INT8"):
        row = clean[candidate]; report_lines.append(f"- `{candidate}`: macro F1 `{row['macro_f1']:.9f}`, accuracy `{row['accuracy']:.9f}`, balanced accuracy `{row['balanced_accuracy']:.9f}`, HUMAN_FALL posture-proxy recall `{row['h_fall_posture_proxy_recall']:.9f}`.")
    report_lines.extend(["", "## Robustness", "", "Perturbations are synthetic offline diagnostics, not Thermal-44 validation. Missing-frame handling is fail-closed with no zero/last/mean imputation.", ""])
    for row in robustness_rows:
        if row["family_id"] == "MISSING_FRAME": report_lines.append(f"- `{row['family_id']}` `{row['level']}`: `{row['status']}`; model inference performed: `false`.")
        else: report_lines.append(f"- `{row['family_id']}` `{row['level']}`: FULL_INT8 macro F1 `{row['metrics']['FULL_INT8']['macro_f1']:.9f}`; delta vs clean `{row['delta_vs_clean']['FULL_INT8']['macro_f1']:.9f}`.")
    report_lines.extend(["", "## Mac latency (microseconds)", "", "CPU/XNNPACK-if-available, one thread, batch one, 20 warmups and 200 measured iterations. This is not Raspberry Pi or sensor-to-alarm latency.", ""])
    for candidate, values in latency["candidates"].items(): report_lines.append(f"- `{candidate}` invoke-only mean/median/p95/p99: `{values['invoke_only']['mean_us']:.3f}` / `{values['invoke_only']['median_us']:.3f}` / `{values['invoke_only']['p95_us']:.3f}` / `{values['invoke_only']['p99_us']:.3f}` us; preprocess+invoke mean: `{values['preprocess_plus_invoke']['mean_us']:.3f}` us.")
    report_lines.extend(["", "## Limitations and handoff", "", "- REAL_EVAL_DEVELOPMENT is not LOCKED_TEST; its T-B4 INT8 sensitivity is diagnostic only.", "- HUMAN_FALL is a Lying-derived posture proxy, not temporal fall ground truth.", "- Subject/session/event generalization is not verifiable; TRAIN-VALIDATION near-duplicate overlap (14,514 pairs) remains disclosed.", "- Thermal-44 device-domain validation and Raspberry Pi latency are deferred to T-C.", ""])
    (evidence / REPORT_NAME).write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    _write_checksums(evidence)
    return {"phase": PHASE_ID, "mode": FULL_MODE, "status": "FINALIZED_WITH_LIMITATIONS", "evidence_dir": EVIDENCE_REL, "selected_candidate_id": "FULL_INT8", "robustness_case_count": len(robustness_rows), "profile_checksum": supplied_checksum, "latency_candidates": list(latency["candidates"])}


def run(*, mode: str, repo_root: str | Path, canonical_root: str | Path, work_root: str | Path, output_root: str | Path, execute: bool = False) -> dict[str, Any]:
    if mode == READINESS_MODE:
        return run_readiness(repo_root=repo_root, canonical_root=canonical_root, work_root=work_root, output_root=output_root)
    if mode == FULL_MODE:
        if not execute: return {"phase": PHASE_ID, "mode": FULL_MODE, "status": "DRY_RUN_READY", "evidence_dir": EVIDENCE_REL}
        return run_full(repo_root=repo_root, canonical_root=canonical_root, work_root=work_root, output_root=output_root)
    raise RunnerContractError(f"unsupported mode: {mode}")
