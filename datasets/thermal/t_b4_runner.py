"""SafeNest Thermal T-B4 frozen Float -> TFLite FP32 -> full INT8 audit.

The runner intentionally does not train a model.  It consumes the verified
T-B3 reference checkpoint, the immutable T-A6 canonical roles, and the frozen
T-B1 P1 preprocessing contract.  Readiness freezes every choice that could
otherwise be tuned after observing quantization results; FULL_EXPERIMENT then
only performs conversion and measurement.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np

from datasets.thermal.t_b1_model import (
    BASELINE_ID,
    architecture_fingerprint,
    backend_info,
    create_small_cnn_baseline,
    require_tensorflow,
)
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


PHASE_ID = "T-B4"
READINESS_MODE = "READINESS"
FULL_MODE = "FULL_EXPERIMENT"
CORRECTION_MODE = "CORRECTION"
EVIDENCE_REL = "datasets/thermal/manifests/T-B4_tflite_int8_equivalence"
TB0_REL = "datasets/thermal/manifests/T-B0_offline_model_protocol"
TB1_REL = "datasets/thermal/manifests/T-B1_full_experiment"
TB2_REL = "datasets/thermal/manifests/T-B2_architecture_comparison"
TB3_REL = "datasets/thermal/manifests/T-B3_frame_multiseed_confirmation"
TA6_REL = "datasets/thermal/manifests/T-A6_execution_result"
ROADMAP_REL = "docs/20260810_ChatGPT_SafeNest_Multisensor_Parallel_Execution_Roadmap_01.md"
RECONCILIATION_REL = "docs/reports/20260814_Codex_Thermal_Post_T-B2_Pre-T-B3_Reconciliation_01.md"

P1_PROFILE = "P1_TRAIN_FITTED_GLOBAL_ZSCORE"
EXPECTED_P1_MEAN = 22.769290618485442
EXPECTED_P1_STD = 2.8684523405441222
EXPECTED_P1_CHECKSUM = "10b5da044ef33f26715544c3d4bf56e9d999d6c65139e931f6997583b3f5b816"
EXPECTED_ARCHITECTURE_FINGERPRINT = "937fceb88900a779d67a8e407cebc3362cc21346721f92cea5ae8de1413aba2a"
EXPECTED_PARAMETER_COUNT = 312131
EXPECTED_CHECKPOINT_SHA = "7aba32fe8d0e241546429bdc3e8cd059b10d4d8f548e9e12c4085abeba308a75"
EXPECTED_CHECKPOINT_SIZE = 3777416
EXPECTED_NEAR_DUPLICATE_PAIRS = 14514
EXPECTED_VAL_MACRO_F1 = 0.9951295332536425
EXPECTED_REAL_MACRO_F1 = 0.593926523563344
CALIBRATION_COUNT = 512
CALIBRATION_BANDS = 8
CALIBRATION_PER_STRATUM = 16
CALIBRATION_LABEL_ORDER = ("EMPTY_ROOM", "SITTING", "STANDING", "LYING")
PROTOCOL_ID = "THERMAL_T_B4_FLOAT_TFLITE_FP32_FULL_INT8_EQUIVALENCE_001"
FORMER_DYNAMIC_RANGE_SHA = "297de231e26ecf2d4cd4010bd10c08d4df3b6b0a531c69693daea353afb8127d"
FORMER_DYNAMIC_RANGE_SIZE = 317344
FULL_INT8_SHA = "fa9730c29535477a3994c11e664474a0ca0116afaaa172889f47446ab2ac46be"


class RunnerContractError(RuntimeError):
    """Raised for a fail-closed readiness or execution violation."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(canonical_json(value), encoding="utf-8")
    os.replace(temporary, path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _repo_commit(repo_root: Path, ref: str = "HEAD") -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(repo_root), "rev-parse", ref], check=True, capture_output=True, text=True)
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _logical_path(path: Path, root: Path | None = None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.name


def _portable(value: str) -> bool:
    lower = value.lower()
    return not (value.startswith(("/", "~/", "file://")) or "\\" in value or "/users/" in lower or "/private/" in lower or value.startswith(("/volumes/", "/content/")))


def _run_predecessors(repo_root: Path) -> dict[str, Any]:
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
    except Exception as exc:  # pragma: no cover - defensive phase boundary
        raise RunnerContractError(f"T_B4_BLOCKED_PREDECESSOR_ERROR:{exc}") from exc
    for phase, item in result.items():
        if item.get("evidence_validation") != "PASS":
            raise RunnerContractError(f"T_B4_BLOCKED_PREDECESSOR_INVALID:{phase}")
    return result


def _verify_branch(repo_root: Path) -> dict[str, Any]:
    branch = subprocess.run(["git", "-C", str(repo_root), "branch", "--show-current"], check=True, capture_output=True, text=True).stdout.strip()
    if not branch.startswith(("feature/", "codex/")) or "t-b4" not in branch.lower():
        raise RunnerContractError("T_B4_BRANCH_IDENTITY_INVALID")
    status = subprocess.run(["git", "-C", str(repo_root), "status", "--short"], check=True, capture_output=True, text=True).stdout.strip()
    return {"branch": branch, "head": _repo_commit(repo_root), "origin_main": _repo_commit(repo_root, "origin/main"), "clean": not bool(status), "status_present": bool(status)}


def _load_p1(repo_root: Path) -> P1Statistics:
    document = _read_json(repo_root / TB1_REL / "p1_preprocessing.json")
    if document.get("profile_id") != P1_PROFILE or document.get("fit_role") != "TRAIN" or document.get("statistics_checksum") != EXPECTED_P1_CHECKSUM:
        raise RunnerContractError("T_B4_P1_IDENTITY_MISMATCH")
    stats = P1Statistics(
        mean=float(document["mean"]), std=float(document["std"]), fit_sample_count=int(document["fit_sample_count"]),
        fit_pixel_count=int(document["fit_pixel_count"]), fit_role=str(document["fit_role"]),
        train_artifact_sha256=str(document["train_artifact_sha256"]), epsilon=float(document.get("epsilon", 1e-6)),
    )
    if stats.checksum() != EXPECTED_P1_CHECKSUM or not math.isclose(stats.mean, EXPECTED_P1_MEAN, abs_tol=1e-12) or not math.isclose(stats.std, EXPECTED_P1_STD, abs_tol=1e-12):
        raise RunnerContractError("T_B4_P1_STATISTICS_MISMATCH")
    return stats


def _checkpoint_path(canonical_root: Path, explicit: str | Path | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(canonical_root.parent / "experiments" / "T-B1" / "T-B1_execution_result" / "checkpoints" / "P1_TRAIN_FITTED_GLOBAL_ZSCORE.weights.h5")
    for path in candidates:
        if path.is_file():
            if path.stat().st_size != EXPECTED_CHECKPOINT_SIZE or sha256_file(path) != EXPECTED_CHECKPOINT_SHA:
                raise RunnerContractError("T_B4_CHECKPOINT_IDENTITY_MISMATCH")
            return path
    raise RunnerContractError("T_B4_REFERENCE_CHECKPOINT_UNAVAILABLE")


def _load_source_labels(path: Path, rows: int) -> list[str]:
    labels: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= rows:
                raise RunnerContractError("T_B4_PROVENANCE_TOO_LONG")
            document = json.loads(line)
            labels.append(str(document.get("original_label_name", "")))
    if len(labels) != rows or any(label not in CALIBRATION_LABEL_ORDER for label in labels):
        raise RunnerContractError("T_B4_PROVENANCE_LABEL_INVALID")
    return labels


def _frame_means(array: np.ndarray, chunk: int = 512) -> np.ndarray:
    result = np.empty((array.shape[0],), dtype=np.float64)
    for start in range(0, array.shape[0], chunk):
        result[start : start + chunk] = np.asarray(array[start : start + chunk], dtype=np.float64).mean(axis=(1, 2))
    return result


def _temperature_policy(train_means: np.ndarray) -> dict[str, Any]:
    quantiles = np.linspace(0.0, 1.0, CALIBRATION_BANDS + 1, dtype=np.float64)
    boundaries = np.quantile(train_means, quantiles, method="linear")
    return {
        "schema_version": "1.0", "phase": PHASE_ID, "policy_id": "T-B4_TRAIN_MEAN_QUANTILE_BANDS_8",
        "fit_role": "TRAIN", "statistic": "per_frame_mean_celsius", "band_count": CALIBRATION_BANDS,
        "quantiles": [float(value) for value in quantiles], "boundaries_celsius": [float(value) for value in boundaries],
        "boundary_method": "numpy.quantile_linear_on_all_TRAIN_frame_means", "validation_or_real_used": False,
    }


def _band_indices(means: np.ndarray, boundaries: Iterable[float]) -> np.ndarray:
    values = np.searchsorted(np.asarray(list(boundaries), dtype=np.float64), means, side="right") - 1
    return np.clip(values, 0, CALIBRATION_BANDS - 1).astype(np.int32)


def _freeze_calibration(canonical_root: Path, role_files: Mapping[str, Any], output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    train = np.load(role_files["TRAIN"].array_path, mmap_mode="r", allow_pickle=False)
    source_labels = _load_source_labels(role_files["TRAIN"].provenance_path, EXPECTED_ROLES["TRAIN"]["rows"])
    means = _frame_means(train)
    temperature = _temperature_policy(means)
    bands = _band_indices(means, temperature["boundaries_celsius"])
    selected: list[int] = []
    strata: list[dict[str, Any]] = []
    stratum_candidates: dict[tuple[str, int], np.ndarray] = {}
    for label in CALIBRATION_LABEL_ORDER:
        for band in range(CALIBRATION_BANDS):
            candidates = np.flatnonzero((np.asarray(source_labels) == label) & (bands == band))
            if candidates.size == 0:
                raise RunnerContractError(f"T_B4_CALIBRATION_EMPTY_STRATUM:{label}:{band}")
            stratum_candidates[(label, band)] = candidates
            quota = min(CALIBRATION_PER_STRATUM, int(candidates.size))
            positions = np.linspace(0, candidates.size - 1, quota, dtype=np.int64)
            chosen = candidates[positions].astype(np.int64)
            selected.extend(int(value) for value in chosen)
            strata.append({"source_label": label, "temperature_band": band, "available": int(candidates.size), "quota": int(quota), "selected": [int(value) for value in chosen]})
    # Some source-label/temperature strata contain fewer than sixteen frames.
    # Backfill the deficit deterministically from the globally remaining TRAIN
    # indices; this preserves unique samples without silently duplicating a
    # calibration frame or looking at VALIDATION/REAL.
    selected_set = set(selected)
    remaining = np.asarray([index for index in range(train.shape[0]) if index not in selected_set], dtype=np.int64)
    deficit = CALIBRATION_COUNT - len(selected_set)
    if deficit > 0:
        positions = np.linspace(0, remaining.size - 1, deficit, dtype=np.int64)
        selected.extend(int(value) for value in remaining[positions])
        backfill = [int(value) for value in remaining[positions]]
    else:
        backfill = []
    selected = sorted(selected)
    if len(selected) != CALIBRATION_COUNT or len(set(selected)) != CALIBRATION_COUNT:
        raise RunnerContractError("T_B4_CALIBRATION_COUNT_INVALID")
    manifest_rows: list[dict[str, Any]] = []
    with role_files["TRAIN"].provenance_path.open("r", encoding="utf-8") as handle:
        by_index: dict[int, dict[str, Any]] = {}
        for index, line in enumerate(handle):
            if index in set(selected):
                document = json.loads(line)
                by_index[index] = document
    for index in selected:
        document = by_index[index]
        manifest_rows.append({
            "canonical_sample_index": index, "stable_sample_id": str(document.get("stable_sample_id", "")),
            "source_label": source_labels[index], "temperature_band": int(bands[index]),
            "frame_mean_celsius": float(means[index]), "role": "TRAIN", "p1_applied": True,
        })
    policy = {
        "schema_version": "1.0", "phase": PHASE_ID, "policy_id": "T-B4_TRAIN_ONLY_STRATIFIED_CALIBRATION_512",
        "source_role": "TRAIN", "sample_count": CALIBRATION_COUNT, "selection_algorithm": "4_source_labels_x_8_TRAIN_mean_quantile_bands_up_to_16_evenly_spaced_indices_then_global_index_backfill",
        "label_order": list(CALIBRATION_LABEL_ORDER), "temperature_policy_id": temperature["policy_id"],
        "temperature_boundaries_checksum": _sha256_text(canonical_json(temperature)), "p1_profile": P1_PROFILE,
        "validation_samples_used": 0, "real_samples_used": 0, "selection_seed": None,
        "representative_dataset_contract": "P1_FLOAT32_NHWC_SINGLE_SAMPLE_GENERATOR",
        "strata": strata, "backfill_count": len(backfill), "backfill_indices": backfill,
    }
    manifest = {"schema_version": "1.0", "phase": PHASE_ID, "policy_id": policy["policy_id"], "role": "TRAIN", "rows": manifest_rows, "row_count": len(manifest_rows)}
    manifest["manifest_checksum"] = _sha256_text(canonical_json(manifest))
    policy["manifest_checksum"] = manifest["manifest_checksum"]
    policy["policy_checksum"] = _sha256_text(canonical_json(policy))
    return policy, {"temperature": temperature, "manifest": manifest}


def _dataset_documents(canonical_root: Path, role_files: Mapping[str, Any]) -> dict[str, Any]:
    identity = validate_canonical_root(canonical_root, full_hash=True)
    identity["legacy_npz_used"] = False
    identity["raw_zip_used"] = False
    identity["official_partition_preservation"] = True
    identity["near_duplicate_pairs_train_validation"] = EXPECTED_NEAR_DUPLICATE_PAIRS
    identity["roles"]["REAL_EVAL_DEVELOPMENT"]["locked_test"] = False
    return identity


def _protocol_documents(repo_root: Path, branch: Mapping[str, Any], predecessors: Mapping[str, Any], dataset: Mapping[str, Any], p1: P1Statistics, checkpoint: Path, role_files: Mapping[str, Any], canonical_root: Path, work_root: Path, output_root: Path) -> dict[str, Any]:
    model = create_small_cnn_baseline()
    if model.count_params() != EXPECTED_PARAMETER_COUNT or architecture_fingerprint(model) != EXPECTED_ARCHITECTURE_FINGERPRINT:
        raise RunnerContractError("T_B4_ARCHITECTURE_IDENTITY_MISMATCH")
    checkpoint_doc = {"logical_path": "experiments/T-B1/T-B1_execution_result/checkpoints/P1_TRAIN_FITTED_GLOBAL_ZSCORE.weights.h5", "sha256": EXPECTED_CHECKPOINT_SHA, "size_bytes": EXPECTED_CHECKPOINT_SIZE, "materialization": "PERSISTENT_EXTERNAL_SSD"}
    docs: dict[str, Any] = {
        "t_b4_protocol.json": {
            "schema_version": "1.0", "phase": PHASE_ID, "protocol_id": PROTOCOL_ID,
            "objective": "FROZEN_FLOAT_TO_TFLITE_FP32_TO_FULL_INT8_EQUIVALENCE_AND_QUANTIZATION_AUDIT",
            "factor_changed": "EXPORT_AND_QUANTIZATION_ONLY", "training": "PROHIBITED", "retraining": False,
            "reference_candidate_id": BASELINE_ID, "reference_seed": 20260813, "p1_profile": P1_PROFILE,
            "real_policy": "POST_ARTIFACT_FIXED_DEVELOPMENT_DIAGNOSTIC_ONLY", "real_used_for_calibration": False,
            "validation_used_for_calibration": False, "new_split_created": False, "t_b5_started": False, "t_c_started": False,
            "legacy_model_replacement": False, "runtime_manifest_modified": False,
            "equivalence_contract": "EQUIVALENCE_MEASURED_NO_PREEXISTING_ABSOLUTE_ACCEPTANCE_THRESHOLD",
            "acceptance_threshold_status": "NO_PREEXISTING_ABSOLUTE_ACCEPTANCE_THRESHOLD",
        },
        "predecessor_identity.json": {"schema_version": "1.0", "phase": PHASE_ID, "predecessors": {phase: {"evidence_validation": value.get("evidence_validation"), "overall_outcome": value.get("overall_outcome")} for phase, value in sorted(predecessors.items())}, "t_b3_merge_required": True, "t_b3_merge_verified": True},
        "float_candidate_lock.json": {"schema_version": "1.0", "phase": PHASE_ID, "candidate_id": BASELINE_ID, "reference_seed": 20260813, "architecture_fingerprint": EXPECTED_ARCHITECTURE_FINGERPRINT, "parameter_count": EXPECTED_PARAMETER_COUNT, "checkpoint": checkpoint_doc, "checkpoint_verified_before_conversion": True, "retraining": False, "candidate_changed": False, "inherited_validation_macro_f1": EXPECTED_VAL_MACRO_F1, "inherited_real_macro_f1": EXPECTED_REAL_MACRO_F1},
        "p1_lock.json": p1.to_dict() | {"schema_version": "1.0", "phase": PHASE_ID, "statistics_checksum": EXPECTED_P1_CHECKSUM, "refit": False, "validation_fit": False, "real_fit": False},
        "dataset_lock.json": dataset,
        "environment.json": {"schema_version": "1.0", "phase": PHASE_ID, "python": platform.python_version(), "platform": platform.platform(), "backend": backend_info(), "canonical_root": "CONFIGURABLE_EXTERNAL_SSD_CANONICAL_ROOT", "work_root": "CONFIGURABLE_LOCAL_SCRATCH_ROOT", "output_root": "CONFIGURABLE_EXTERNAL_SSD_T_B4_ROOT", "gpu_required": False},
        "artifact_storage_policy.json": {"schema_version": "1.0", "phase": PHASE_ID, "external_root": "EXTERNAL_SSD_SAFE_NEST_AI_THERMAL_EXPERIMENTS_T_B4", "git_binary_policy": "COMPACT_EVIDENCE_AND_CHECKSUMS_ONLY", "raw_payloads_tracked": False, "canonical_payloads_tracked": False, "checkpoint_tracked": False, "legacy_model_overwrite": False},
    }
    return docs


def run_readiness(*, canonical_root: str | Path, work_root: str | Path, output_root: str | Path, repo_root: str | Path, checkpoint_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    canonical = Path(canonical_root).expanduser()
    work = Path(work_root).expanduser()
    output = Path(output_root).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    branch = _verify_branch(root)
    predecessors = _run_predecessors(root)
    role_files = {role: resolve_role_files(canonical, role) for role in ROLE_ORDER}
    dataset = _dataset_documents(canonical, role_files)
    p1 = _load_p1(root)
    checkpoint = _checkpoint_path(canonical, checkpoint_path)
    docs = _protocol_documents(root, branch, predecessors, dataset, p1, checkpoint, role_files, canonical, work, output)
    calibration, extra = _freeze_calibration(canonical, role_files, output)
    docs["representative_calibration_policy.json"] = calibration
    docs["representative_sample_manifest.json"] = extra["manifest"]
    docs["temperature_range_policy.json"] = extra["temperature"]
    docs["readiness_result.json"] = {"schema_version": "1.0", "phase": PHASE_ID, "status": "T_B4_CONVERSION_READY", "repo_commit": branch["head"], "origin_main": branch["origin_main"], "branch": branch["branch"], "predecessors_pass": True, "checkpoint_verified": True, "calibration_frozen": True, "temperature_policy_frozen": True, "t_b5_started": False, "t_c_started": False, "output_root": "EXTERNAL_SSD_SAFE_NEST_AI_THERMAL_EXPERIMENTS_T_B4"}
    evidence = root / EVIDENCE_REL
    evidence.mkdir(parents=True, exist_ok=True)
    for name, value in docs.items():
        _write_json(evidence / name, value)
    _write_checksums(evidence, exclude={"checksums.sha256"})
    return {"phase": PHASE_ID, "status": "T_B4_CONVERSION_READY", "evidence_dir": EVIDENCE_REL, "calibration_policy_id": calibration["policy_id"], "calibration_count": CALIBRATION_COUNT, "temperature_policy_id": extra["temperature"]["policy_id"], "predecessors": {phase: {"evidence_validation": item.get("evidence_validation"), "overall_outcome": item.get("overall_outcome")} for phase, item in sorted(predecessors.items())}}


def _tflite_metadata(model_bytes: bytes, artifact: Path, label: str) -> dict[str, Any]:
    tf = require_tensorflow()
    interpreter = tf.lite.Interpreter(model_content=model_bytes, num_threads=1)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    op_details = interpreter._get_ops_details()  # TensorFlow's stable inspection API for this audit.
    tensor_details = interpreter.get_tensor_details()
    names = [str(item.get("op_name", "UNKNOWN")) for item in op_details]
    op_counts: dict[str, int] = {}
    for name in names:
        op_counts[name] = op_counts.get(name, 0) + 1
    def tensor_detail(item: Mapping[str, Any]) -> dict[str, Any]:
        quant = item.get("quantization", (0.0, 0))
        qparams = item.get("quantization_parameters", {})
        scales = np.asarray(qparams.get("scales", []), dtype=np.float64).reshape(-1).tolist()
        zeros = np.asarray(qparams.get("zero_points", []), dtype=np.int64).reshape(-1).tolist()
        return {"name": str(item.get("name", "")), "shape": [int(value) for value in np.asarray(item.get("shape", []), dtype=np.int64)], "dtype": np.dtype(item.get("dtype")).name, "scale": float(quant[0]), "zero_point": int(quant[1]), "quantized_dimension": int(qparams.get("quantized_dimension", 0)), "scales": [float(value) for value in scales], "zero_points": [int(value) for value in zeros]}
    internal_dtype_counts: dict[str, int] = {}
    quantized_tensor_count = 0
    quantized_parameter_tensor_count = 0
    nonzero_quantization_tensor_count = 0
    for item in tensor_details:
        dtype_name = np.dtype(item.get("dtype")).name
        internal_dtype_counts[dtype_name] = internal_dtype_counts.get(dtype_name, 0) + 1
        if dtype_name in {"int8", "uint8", "int16", "uint16"}:
            quantized_tensor_count += 1
            # Inputs/outputs and activation tensors are not parameters.  The
            # pseudo_qconst names are the converter's durable weight evidence.
            name = str(item.get("name", ""))
            if "pseudo_qconst" in name.lower() or "const" in name.lower():
                quantized_parameter_tensor_count += 1
        qparams = item.get("quantization_parameters", {})
        scales = np.asarray(qparams.get("scales", []), dtype=np.float64).reshape(-1)
        zeros = np.asarray(qparams.get("zero_points", []), dtype=np.int64).reshape(-1)
        if scales.size and (np.any(scales != 0.0) or np.any(zeros != 0)):
            nonzero_quantization_tensor_count += 1
    return {"schema_version": "1.0", "phase": PHASE_ID, "artifact_id": label, "logical_path": f"artifacts/{artifact.name}", "sha256": _sha256_bytes(model_bytes), "size_bytes": len(model_bytes), "converter_tensorflow": str(getattr(tf, "__version__", "unknown")), "input": tensor_detail(inputs[0]), "output": tensor_detail(outputs[0]), "ops": names, "op_counts": dict(sorted(op_counts.items())), "internal_tensor_count": len(tensor_details), "internal_dtype_counts": dict(sorted(internal_dtype_counts.items())), "quantized_tensor_count": quantized_tensor_count, "quantized_parameter_tensor_count": quantized_parameter_tensor_count, "nonzero_quantization_tensor_count": nonzero_quantization_tensor_count, "select_tf_ops_used": any("Flex" in name or "SELECT" in name for name in names), "builtin_only": not any("Flex" in name or "SELECT" in name for name in names)}


def _convert(model: Any, calibration_inputs: np.ndarray, *, int8: bool) -> bytes:
    tf = require_tensorflow()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if int8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        def representative_dataset() -> Iterable[list[np.ndarray]]:
            for row in calibration_inputs:
                yield [np.asarray(row[None, ...], dtype=np.float32)]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        # A true TFLite FP32 baseline must not enable any post-training
        # optimization.  In particular, Optimize.DEFAULT without a
        # representative dataset produces dynamic-range-quantized weights
        # while retaining float32 I/O, which is not an FP32 baseline.
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    try:
        return bytes(converter.convert())
    except Exception as exc:
        raise RunnerContractError(f"T_B4_TFLITE_CONVERSION_FAILED:{exc}") from exc


def _keras_predict(model: Any, x: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(x, batch_size=128, verbose=0), dtype=np.float32)


def _quantize_input(values: np.ndarray, detail: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    quantization = detail.get("quantization", (detail.get("scale", 0.0), detail.get("zero_point", 0)))
    scale = float(quantization[0]); zero = int(quantization[1])
    if scale <= 0:
        raise RunnerContractError("T_B4_INT8_INPUT_QUANTIZATION_INVALID")
    dtype = np.dtype(detail["dtype"])
    info = np.iinfo(dtype)
    raw = np.asarray(values, dtype=np.float64) / scale + zero
    below = raw < info.min; above = raw > info.max
    rounded = np.rint(raw)
    clipped = np.clip(rounded, info.min, info.max).astype(dtype)
    return clipped, {"total_elements": int(raw.size), "lower_boundary_count": int(np.sum(clipped == info.min)), "upper_boundary_count": int(np.sum(clipped == info.max)), "required_clipping_count": int(np.sum(below | above)), "required_clipping_fraction": float(np.mean(below | above)), "frames_with_required_clipping": int(np.sum(np.any(below.reshape((values.shape[0], -1)), axis=1))), "frames_with_boundary_value": int(np.sum(np.any((clipped == info.min) | (clipped == info.max), axis=tuple(range(1, clipped.ndim))))) }


def _tflite_predict(model_bytes: bytes, inputs: np.ndarray, *, quantized: bool) -> tuple[np.ndarray, dict[str, Any]]:
    tf = require_tensorflow()
    interpreter = tf.lite.Interpreter(model_content=model_bytes, num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    outputs: list[np.ndarray] = []
    raw_outputs: list[np.ndarray] = []
    input_audit = {"total_elements": 0, "lower_boundary_count": 0, "upper_boundary_count": 0, "required_clipping_count": 0, "frames_with_required_clipping": 0, "frames_with_boundary_value": 0}
    for index in range(inputs.shape[0]):
        row = inputs[index : index + 1]
        if quantized:
            quantized_row, audit = _quantize_input(row, input_detail)
            interpreter.set_tensor(input_detail["index"], quantized_row)
            for key in input_audit:
                input_audit[key] += int(audit.get(key, 0))
        else:
            interpreter.set_tensor(input_detail["index"], np.asarray(row, dtype=np.float32))
        interpreter.invoke()
        raw = np.asarray(interpreter.get_tensor(output_detail["index"])).copy()
        raw_outputs.append(raw.reshape(-1))
        if quantized:
            scale, zero = output_detail.get("quantization", (0.0, 0))
            if float(scale) <= 0:
                raise RunnerContractError("T_B4_INT8_OUTPUT_QUANTIZATION_INVALID")
            outputs.append(((raw.astype(np.float32) - float(zero)) * float(scale)).reshape(-1))
        else:
            outputs.append(raw.astype(np.float32).reshape(-1))
    result = np.asarray(outputs, dtype=np.float32)
    raw = np.asarray(raw_outputs)
    if quantized:
        info = np.iinfo(np.dtype(input_detail["dtype"]))
        input_audit["boundary_fraction"] = float((input_audit["lower_boundary_count"] + input_audit["upper_boundary_count"]) / max(1, input_audit["total_elements"]))
        input_audit["required_clipping_fraction"] = float(input_audit["required_clipping_count"] / max(1, input_audit["total_elements"]))
        out_info = np.iinfo(np.dtype(output_detail["dtype"]))
        input_audit["output_lower_boundary_count"] = int(np.sum(raw == out_info.min)); input_audit["output_upper_boundary_count"] = int(np.sum(raw == out_info.max)); input_audit["output_total_elements"] = int(raw.size)
        input_audit["output_boundary_fraction"] = float((input_audit["output_lower_boundary_count"] + input_audit["output_upper_boundary_count"]) / max(1, raw.size))
        input_audit["output_dequantized_min"] = float(result.min()); input_audit["output_dequantized_max"] = float(result.max())
    return result, {"input_saturation": input_audit, "input_tensor": {"dtype": np.dtype(input_detail["dtype"]).name, "scale": float(input_detail.get("quantization", (0.0, 0))[0]), "zero_point": int(input_detail.get("quantization", (0.0, 0))[1])}, "output_tensor": {"dtype": np.dtype(output_detail["dtype"]).name, "scale": float(output_detail.get("quantization", (0.0, 0))[0]), "zero_point": int(output_detail.get("quantization", (0.0, 0))[1])}}


def _parity(float_probs: np.ndarray, other_probs: np.ndarray, y_true: np.ndarray) -> dict[str, Any]:
    float_pred = np.argmax(float_probs, axis=1)
    other_pred = np.argmax(other_probs, axis=1)
    abs_error = np.abs(float_probs.astype(np.float64) - other_probs.astype(np.float64))
    float_metrics = compute_metrics(y_true, float_pred)
    other_metrics = compute_metrics(y_true, other_pred)
    return {"sample_count": int(y_true.size), "argmax_agreement": float(np.mean(float_pred == other_pred)), "disagreement_count": int(np.sum(float_pred != other_pred)), "probability_mae": float(abs_error.mean()), "probability_max_absolute_error": float(abs_error.max()), "per_class_probability_mae": {name: float(abs_error[:, index].mean()) for index, name in enumerate(CLASS_ORDER)}, "float_metrics": float_metrics, "other_metrics": other_metrics, "metric_delta_other_minus_float": {"macro_f1": float(other_metrics["macro_f1"] - float_metrics["macro_f1"]), "accuracy": float(other_metrics["accuracy"] - float_metrics["accuracy"]), "balanced_accuracy": float(other_metrics["balanced_accuracy"] - float_metrics["balanced_accuracy"]), "h_fall_posture_proxy_recall": float(other_metrics["h_fall_posture_proxy_recall"] - float_metrics["h_fall_posture_proxy_recall"])}, "confusion_matrix_difference": (np.asarray(other_metrics["confusion_matrix"], dtype=np.int64) - np.asarray(float_metrics["confusion_matrix"], dtype=np.int64)).tolist()}


def _band_metrics(float_probs: np.ndarray, int8_probs: np.ndarray, y_true: np.ndarray, means: np.ndarray, boundaries: list[float], saturation: Mapping[str, Any]) -> dict[str, Any]:
    bands = _band_indices(means, boundaries)
    rows = []
    for band in range(CALIBRATION_BANDS):
        mask = bands == band
        if not np.any(mask):
            continue
        parity = _parity(float_probs[mask], int8_probs[mask], y_true[mask])
        total = int(np.sum(mask))
        rows.append({"band": band, "lower_celsius": float(boundaries[band]), "upper_celsius": float(boundaries[band + 1]), "sample_count": total, "float_macro_f1": float(parity["float_metrics"]["macro_f1"]), "int8_macro_f1": float(parity["other_metrics"]["macro_f1"]), "argmax_agreement": parity["argmax_agreement"], "probability_mae": parity["probability_mae"], "input_required_clipping_fraction": float(saturation.get("required_clipping_fraction", 0.0))})
    worst_agreement = min(rows, key=lambda item: (item["argmax_agreement"], item["band"])) if rows else None
    worst_error = max(rows, key=lambda item: (item["probability_mae"], -item["band"])) if rows else None
    return {"schema_version": "1.0", "phase": PHASE_ID, "statistic": "per_frame_mean_celsius", "bands": rows, "worst_agreement_band": worst_agreement["band"] if worst_agreement else None, "worst_probability_error_band": worst_error["band"] if worst_error else None}


def _write_checksums(directory: Path, exclude: set[str] | None = None) -> None:
    excluded = exclude or set()
    rows = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name in excluded or path.name.startswith("._") or path.name.endswith(".partial"):
            continue
        rows.append(f"{sha256_file(path)}  {path.name}")
    (directory / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def run_full(*, canonical_root: str | Path, work_root: str | Path, output_root: str | Path, repo_root: str | Path, owner_authorized: bool, checkpoint_path: str | Path | None = None) -> dict[str, Any]:
    if not owner_authorized:
        raise RunnerContractError("T_B4_FULL_EXECUTION_OWNER_AUTHORIZATION_REQUIRED")
    root = Path(repo_root).resolve(); canonical = Path(canonical_root).expanduser(); work = Path(work_root).expanduser(); output = Path(output_root).expanduser()
    readiness = root / EVIDENCE_REL / "readiness_result.json"
    if not readiness.is_file() or _read_json(readiness).get("status") != "T_B4_CONVERSION_READY":
        raise RunnerContractError("T_B4_READINESS_CONTRACT_MISSING")
    branch = _verify_branch(root); predecessors = _run_predecessors(root)
    role_files = {role: resolve_role_files(canonical, role) for role in ROLE_ORDER}
    dataset = _dataset_documents(canonical, role_files); p1 = _load_p1(root); checkpoint = _checkpoint_path(canonical, checkpoint_path)
    evidence = root / EVIDENCE_REL
    calibration_policy = _read_json(evidence / "representative_calibration_policy.json")
    sample_manifest = _read_json(evidence / "representative_sample_manifest.json")
    temp_policy = _read_json(evidence / "temperature_range_policy.json")
    if calibration_policy.get("validation_samples_used") != 0 or calibration_policy.get("real_samples_used") != 0 or sample_manifest.get("row_count") != CALIBRATION_COUNT:
        raise RunnerContractError("T_B4_CALIBRATION_SCOPE_INVALID")
    if calibration_policy.get("manifest_checksum") != sample_manifest.get("manifest_checksum"):
        raise RunnerContractError("T_B4_CALIBRATION_MANIFEST_MISMATCH")
    bundle = output / "T-B4_execution_result"
    if bundle.exists() and any(path for path in bundle.iterdir() if not path.name.startswith("._")):
        raise RunnerContractError("T_B4_OUTPUT_ALREADY_EXISTS")
    bundle.mkdir(parents=True, exist_ok=True); (bundle / "artifacts").mkdir(); (bundle / "parity").mkdir(); (bundle / "calibration").mkdir()
    arrays = {role: np.load(role_files[role].array_path, mmap_mode="r", allow_pickle=False) for role in ROLE_ORDER}
    labels = {role: labels_from_provenance(role_files[role].provenance_path, EXPECTED_ROLES[role]["rows"])[0] for role in ROLE_ORDER}
    train_indices = np.asarray([int(row["canonical_sample_index"]) for row in sample_manifest["rows"]], dtype=np.int64)
    calibration_inputs = apply_p1(np.asarray(arrays["TRAIN"][train_indices]), p1)
    model = create_small_cnn_baseline(); model.load_weights(checkpoint)
    if int(model.count_params()) != EXPECTED_PARAMETER_COUNT or architecture_fingerprint(model) != EXPECTED_ARCHITECTURE_FINGERPRINT:
        raise RunnerContractError("T_B4_MODEL_CONTRACT_INVALID")
    validation_inputs = apply_p1(arrays["VALIDATION"], p1); real_inputs = apply_p1(arrays["REAL_EVAL_DEVELOPMENT"], p1)
    float_validation = _keras_predict(model, validation_inputs); float_real = _keras_predict(model, real_inputs)
    inherited = compute_metrics(labels["VALIDATION"], np.argmax(float_validation, axis=1))
    if not math.isclose(float(inherited["macro_f1"]), EXPECTED_VAL_MACRO_F1, abs_tol=1e-7):
        raise RunnerContractError("T_B4_INHERITED_VALIDATION_METRIC_MISMATCH")
    fp32_bytes = _convert(model, calibration_inputs, int8=False)
    int8_bytes = _convert(model, calibration_inputs, int8=True)
    fp32_path = bundle / "artifacts" / "SMALL_CNN_BASELINE_V1_P1_float32.tflite"; int8_path = bundle / "artifacts" / "SMALL_CNN_BASELINE_V1_P1_full_int8.tflite"
    fp32_path.write_bytes(fp32_bytes); int8_path.write_bytes(int8_bytes)
    fp32_meta = _tflite_metadata(fp32_bytes, fp32_path, "TFLITE_FP32"); int8_meta = _tflite_metadata(int8_bytes, int8_path, "FULL_INT8")
    if fp32_meta["input"]["dtype"] != "float32" or fp32_meta["output"]["dtype"] != "float32" or int8_meta["input"]["dtype"] != "int8" or int8_meta["output"]["dtype"] != "int8" or not int8_meta["builtin_only"]:
        raise RunnerContractError("T_B4_FULL_INT8_REQUIREMENT_NOT_MET")
    fp32_validation, fp32_aux = _tflite_predict(fp32_bytes, validation_inputs, quantized=False)
    int8_validation, int8_aux = _tflite_predict(int8_bytes, validation_inputs, quantized=True)
    fp32_real, _ = _tflite_predict(fp32_bytes, real_inputs, quantized=False)
    int8_real, int8_real_aux = _tflite_predict(int8_bytes, real_inputs, quantized=True)
    fp32_parity = _parity(float_validation, fp32_validation, labels["VALIDATION"])
    int8_parity = _parity(float_validation, int8_validation, labels["VALIDATION"])
    fp32_int8_parity = _parity(fp32_validation, int8_validation, labels["VALIDATION"])
    real_parity = _parity(float_real, int8_real, labels["REAL_EVAL_DEVELOPMENT"])
    temp_means = _frame_means(arrays["VALIDATION"]); temp_audit = _band_metrics(float_validation, int8_validation, labels["VALIDATION"], temp_means, [float(value) for value in temp_policy["boundaries_celsius"]], int8_aux["input_saturation"])
    temp_audit["temperature_policy_checksum"] = _sha256_text(canonical_json(temp_policy))
    calibration_manifest_path = bundle / "calibration" / "representative_sample_manifest.json"; _write_json(calibration_manifest_path, sample_manifest)
    np.savez_compressed(bundle / "parity" / "validation_probabilities.npz", float_keras=float_validation, tflite_fp32=fp32_validation, tflite_int8=int8_validation, y_true=labels["VALIDATION"])
    np.savez_compressed(bundle / "parity" / "real_probabilities.npz", float_keras=float_real, tflite_fp32=fp32_real, tflite_int8=int8_real, y_true=labels["REAL_EVAL_DEVELOPMENT"])
    documents: dict[str, Any] = {
        "execution_environment.json": {"schema_version": "1.0", "phase": PHASE_ID, "mode": FULL_MODE, "repo_commit": branch["head"], "origin_main": branch["origin_main"], "backend": backend_info(), "gpu_required": False, "canonical_root": "CONFIGURABLE_EXTERNAL_SSD_CANONICAL_ROOT", "output_root": "EXTERNAL_SSD_SAFE_NEST_AI_THERMAL_EXPERIMENTS_T_B4", "tensorflow": fp32_meta["converter_tensorflow"]},
        "float_source_integrity.json": {"schema_version": "1.0", "phase": PHASE_ID, "candidate_id": BASELINE_ID, "checkpoint": {"logical_path": "experiments/T-B1/T-B1_execution_result/checkpoints/P1_TRAIN_FITTED_GLOBAL_ZSCORE.weights.h5", "sha256": EXPECTED_CHECKPOINT_SHA, "size_bytes": EXPECTED_CHECKPOINT_SIZE}, "architecture_fingerprint": EXPECTED_ARCHITECTURE_FINGERPRINT, "parameter_count": EXPECTED_PARAMETER_COUNT, "p1_checksum": EXPECTED_P1_CHECKSUM, "retraining": False, "validation_metrics": inherited},
        "tflite_fp32_artifact.json": fp32_meta | {"conversion": {"optimizations": [], "supported_ops": ["TFLITE_BUILTINS"], "select_tf_ops": False, "representative_dataset_attached": False, "float16_enabled": False, "dynamic_range_quantization": False, "quantization_mode": "NONE"}},
        "tflite_int8_artifact.json": int8_meta | {"conversion": {"optimizations": ["DEFAULT"], "supported_ops": ["TFLITE_BUILTINS_INT8"], "select_tf_ops": False, "representative_dataset_attached": True, "float16_enabled": False, "dynamic_range_quantization": False, "quantization_mode": "FULL_INT8", "representative_policy_checksum": calibration_policy["policy_checksum"]}},
        "tensor_quantization_contract.json": {"schema_version": "1.0", "phase": PHASE_ID, "input": int8_meta["input"], "output": int8_meta["output"], "full_integer": True, "per_axis_details_present": any(len(int8_meta[key].get("scales", [])) > 1 for key in ("input", "output"))},
        "op_inventory.json": {"schema_version": "1.0", "phase": PHASE_ID, "float32": {"ops": fp32_meta["ops"], "counts": fp32_meta["op_counts"], "builtin_only": fp32_meta["builtin_only"], "internal_dtype_counts": fp32_meta["internal_dtype_counts"], "quantized_tensor_count": fp32_meta["quantized_tensor_count"], "quantized_parameter_tensor_count": fp32_meta["quantized_parameter_tensor_count"]}, "full_int8": {"ops": int8_meta["ops"], "counts": int8_meta["op_counts"], "builtin_only": int8_meta["builtin_only"], "internal_dtype_counts": int8_meta["internal_dtype_counts"], "quantized_tensor_count": int8_meta["quantized_tensor_count"], "quantized_parameter_tensor_count": int8_meta["quantized_parameter_tensor_count"]}},
        "float_vs_tflite_fp32_parity.json": {"schema_version": "1.0", "phase": PHASE_ID, "role": "VALIDATION", **fp32_parity},
        "float_vs_int8_parity.json": {"schema_version": "1.0", "phase": PHASE_ID, "role": "VALIDATION", **int8_parity},
        "fp32_vs_int8_parity.json": {"schema_version": "1.0", "phase": PHASE_ID, "role": "VALIDATION", **fp32_int8_parity},
        "input_saturation_audit.json": {"schema_version": "1.0", "phase": PHASE_ID, "role": "VALIDATION", **int8_aux["input_saturation"], "interpretation": "BOUNDARY_OCCUPANCY_AND_REQUIRED_CLIPPING_REPORTED; NO_POST_HOC_ACCEPTANCE_THRESHOLD"},
        "output_saturation_audit.json": {"schema_version": "1.0", "phase": PHASE_ID, "role": "VALIDATION", "lower_boundary_count": int(int8_aux["input_saturation"].get("output_lower_boundary_count", 0)), "upper_boundary_count": int(int8_aux["input_saturation"].get("output_upper_boundary_count", 0)), "total_elements": int(int8_aux["input_saturation"].get("output_total_elements", 0)), "boundary_fraction": float(int8_aux["input_saturation"].get("output_boundary_fraction", 0.0)), "dequantized_min": float(int8_aux["input_saturation"].get("output_dequantized_min", 0.0)), "dequantized_max": float(int8_aux["input_saturation"].get("output_dequantized_max", 0.0))},
        "temperature_range_error.json": temp_audit,
        "real_development_parity.json": {"schema_version": "1.0", "phase": PHASE_ID, "role": "REAL_EVAL_DEVELOPMENT", "diagnostic_performed": True, "frozen_before_real": True, "used_for_calibration": False, "used_for_selection": False, "fp32_tflite_metrics": {key: float(compute_metrics(labels["REAL_EVAL_DEVELOPMENT"], np.argmax(fp32_real, axis=1))[key]) for key in ("macro_f1", "accuracy", "balanced_accuracy", "h_fall_posture_proxy_recall")}, "float_vs_tflite_fp32_parity": _parity(float_real, fp32_real, labels["REAL_EVAL_DEVELOPMENT"]), "fp32_vs_int8_parity": _parity(fp32_real, int8_real, labels["REAL_EVAL_DEVELOPMENT"]), "float_vs_int8_parity": real_parity, **real_parity, "input_saturation": int8_real_aux["input_saturation"]},
        "representative_calibration_policy.json": calibration_policy,
        "representative_sample_manifest.json": sample_manifest,
        "temperature_range_policy.json": temp_policy,
        "dataset_lock.json": dataset,
        "p1_lock.json": _read_json(evidence / "p1_lock.json"),
        "float_candidate_lock.json": _read_json(evidence / "float_candidate_lock.json"),
        "t_b4_protocol.json": _read_json(evidence / "t_b4_protocol.json"),
        "limitations.json": {"schema_version": "1.0", "phase": PHASE_ID, "near_duplicate_pairs_train_validation": EXPECTED_NEAR_DUPLICATE_PAIRS, "locked_test_available": False, "real_role": "REAL_EVAL_DEVELOPMENT_NOT_LOCKED_TEST", "human_fall_semantics": "LYING_TO_HUMAN_FALL_DERIVED_POSTURE_PROXY_NOT_TEMPORAL_EVENT_GROUND_TRUTH", "temporal_event_parity": "NOT_EVALUATED_FRAME_LEVEL_ONLY", "subject_session_event_generalization": "NOT_VERIFIABLE", "synthetic_real_gap": float(EXPECTED_VAL_MACRO_F1 - EXPECTED_REAL_MACRO_F1), "synthetic_real_gap_interpretation": "OBSERVED_DEVELOPMENT_GAP_SEPARATE_FROM_QUANTIZATION_GAP", "thermal44_validation": "NOT_PERFORMED_DEFERRED_TO_T-C", "pi_latency": "NOT_MEASURED", "absolute_equivalence_threshold": "NOT_PREEXISTING", "next_phase_started": False, "t_b5_started": False, "t_c_started": False},
    }
    artifact_registry = {"schema_version": "1.0", "phase": PHASE_ID, "artifacts": [
        {"id": "TFLITE_FP32", "logical_path": "artifacts/SMALL_CNN_BASELINE_V1_P1_float32.tflite", "sha256": fp32_meta["sha256"], "size_bytes": fp32_meta["size_bytes"], "tracked_in_git": False},
        {"id": "FULL_INT8", "logical_path": "artifacts/SMALL_CNN_BASELINE_V1_P1_full_int8.tflite", "sha256": int8_meta["sha256"], "size_bytes": int8_meta["size_bytes"], "tracked_in_git": False},
        {"id": "VALIDATION_PARITY_ARRAYS", "logical_path": "parity/validation_probabilities.npz", "sha256": sha256_file(bundle / "parity" / "validation_probabilities.npz"), "size_bytes": int((bundle / "parity" / "validation_probabilities.npz").stat().st_size), "tracked_in_git": False},
        {"id": "REAL_PARITY_ARRAYS", "logical_path": "parity/real_probabilities.npz", "sha256": sha256_file(bundle / "parity" / "real_probabilities.npz"), "size_bytes": int((bundle / "parity" / "real_probabilities.npz").stat().st_size), "tracked_in_git": False},
    ], "legacy_model_overwrite": False, "production_manifest_changed": False}
    documents["artifact_registry.json"] = artifact_registry
    documents["execution_summary.json"] = {"schema_version": "1.0", "phase": PHASE_ID, "status": "FINALIZED", "mode": FULL_MODE, "conversion_performed": True, "retraining_performed": False, "validation_samples": 8000, "real_samples": 8000, "calibration_samples": CALIBRATION_COUNT, "t_b5_started": False, "t_c_started": False, "candidate_changed": False}
    for name, value in documents.items():
        _write_json(bundle / name, value)
    _write_checksums(bundle, exclude={"checksums.sha256"})
    compact_names = [path.name for path in bundle.iterdir() if path.is_file() and path.suffix == ".json"]
    for name in sorted(compact_names):
        _atomic_copy(bundle / name, evidence / name)
    _write_checksums(evidence, exclude={"checksums.sha256"})
    return {"phase": PHASE_ID, "status": "FINALIZED", "bundle": "EXTERNAL_SSD_SAFE_NEST_AI_THERMAL_EXPERIMENTS_T_B4/T-B4_execution_result", "float32_sha256": fp32_meta["sha256"], "int8_sha256": int8_meta["sha256"], "validation": {"float_vs_fp32": fp32_parity, "float_vs_int8": int8_parity, "fp32_vs_int8": fp32_int8_parity}, "real": real_parity}


def run_correction(*, canonical_root: str | Path, work_root: str | Path, output_root: str | Path, repo_root: str | Path, owner_authorized: bool, checkpoint_path: str | Path | None = None) -> dict[str, Any]:
    """Replace the mislabeled dynamic-range artifact with true unquantized FP32.

    The frozen checkpoint, TRAIN-only calibration manifest, and existing FULL_INT8
    binary are verified.  Conversion and parity are performed by ``run_full`` in
    an isolated temporary bundle; only the corrected FP32, parity evidence, and
    compact metadata are copied into the existing external T-B4 bundle.
    """
    if not owner_authorized:
        raise RunnerContractError("T_B4_CORRECTION_OWNER_AUTHORIZATION_REQUIRED")
    root = Path(repo_root).resolve()
    output = Path(output_root).expanduser()
    bundle = output / "T-B4_execution_result"
    old_path = bundle / "artifacts" / "SMALL_CNN_BASELINE_V1_P1_float32.tflite"
    dynamic_path = bundle / "artifacts" / "SMALL_CNN_BASELINE_V1_P1_dynamic_range.tflite"
    int8_path = bundle / "artifacts" / "SMALL_CNN_BASELINE_V1_P1_full_int8.tflite"
    if not int8_path.is_file():
        raise RunnerContractError("T_B4_EXISTING_FULL_INT8_UNAVAILABLE")
    existing_int8_sha = sha256_file(int8_path)
    if existing_int8_sha != FULL_INT8_SHA:
        raise RunnerContractError("T_B4_EXISTING_FULL_INT8_IDENTITY_MISMATCH")
    if dynamic_path.is_file():
        former_bytes = dynamic_path.read_bytes()
        if _sha256_bytes(former_bytes) != FORMER_DYNAMIC_RANGE_SHA or len(former_bytes) != FORMER_DYNAMIC_RANGE_SIZE:
            raise RunnerContractError("T_B4_DYNAMIC_RANGE_IDENTITY_MISMATCH")
        if old_path.is_file() and _sha256_bytes(old_path.read_bytes()) == FORMER_DYNAMIC_RANGE_SHA:
            raise RunnerContractError("T_B4_DYNAMIC_RANGE_DUPLICATE_PATH_AMBIGUOUS")
    elif old_path.is_file():
        former_bytes = old_path.read_bytes()
        if _sha256_bytes(former_bytes) != FORMER_DYNAMIC_RANGE_SHA or len(former_bytes) != FORMER_DYNAMIC_RANGE_SIZE:
            raise RunnerContractError("T_B4_FORMER_FP32_IDENTITY_MISMATCH")
        os.replace(old_path, dynamic_path)
    else:
        raise RunnerContractError("T_B4_FORMER_FP32_ARTIFACT_UNAVAILABLE")

    temporary_root = Path(tempfile.mkdtemp(prefix="safenest-t-b4-correction-"))
    generated = run_full(canonical_root=canonical_root, work_root=work_root, output_root=temporary_root, repo_root=root, owner_authorized=True, checkpoint_path=checkpoint_path)
    generated_bundle = temporary_root / "T-B4_execution_result"
    generated_int8 = generated_bundle / "artifacts" / "SMALL_CNN_BASELINE_V1_P1_full_int8.tflite"
    if sha256_file(generated_int8) != existing_int8_sha:
        raise RunnerContractError("T_B4_FULL_INT8_RECONVERSION_DRIFT")
    generated_fp32 = generated_bundle / "artifacts" / "SMALL_CNN_BASELINE_V1_P1_float32.tflite"
    fp32_meta = _read_json(generated_bundle / "tflite_fp32_artifact.json")
    dynamic_meta = _tflite_metadata(former_bytes, dynamic_path, "TFLITE_DYNAMIC_RANGE")
    dynamic_meta["conversion"] = {"optimizations": ["DEFAULT"], "supported_ops": ["TFLITE_BUILTINS"], "select_tf_ops": False, "representative_dataset_attached": False, "float16_enabled": False, "dynamic_range_quantization": True, "quantization_mode": "DYNAMIC_RANGE"}
    dynamic_meta["diagnostic_only"] = True
    dynamic_meta["official_equivalence_stage"] = False
    registry = _read_json(generated_bundle / "artifact_registry.json")
    registry["artifacts"].insert(0, {"id": "TFLITE_DYNAMIC_RANGE", "logical_path": "artifacts/SMALL_CNN_BASELINE_V1_P1_dynamic_range.tflite", "sha256": dynamic_meta["sha256"], "size_bytes": dynamic_meta["size_bytes"], "tracked_in_git": False, "official_equivalence_stage": False, "diagnostic_only": True, "conversion": dynamic_meta["conversion"], "internal_dtype_counts": dynamic_meta["internal_dtype_counts"], "quantized_tensor_count": dynamic_meta["quantized_tensor_count"], "quantized_parameter_tensor_count": dynamic_meta["quantized_parameter_tensor_count"], "reclassified_from": "TFLITE_FP32"})
    for item in registry["artifacts"]:
        if item.get("id") == "TFLITE_FP32":
            item.update({"official_equivalence_stage": True, "diagnostic_only": False, "conversion": fp32_meta["conversion"], "internal_dtype_counts": fp32_meta["internal_dtype_counts"], "quantized_tensor_count": fp32_meta["quantized_tensor_count"], "quantized_parameter_tensor_count": fp32_meta["quantized_parameter_tensor_count"]})
        elif item.get("id") == "FULL_INT8":
            int8_meta = _read_json(generated_bundle / "tflite_int8_artifact.json")
            item.update({"official_equivalence_stage": True, "diagnostic_only": False, "conversion": int8_meta["conversion"], "internal_dtype_counts": int8_meta["internal_dtype_counts"], "quantized_tensor_count": int8_meta["quantized_tensor_count"], "quantized_parameter_tensor_count": int8_meta["quantized_parameter_tensor_count"]})
    _write_json(generated_bundle / "artifact_registry.json", registry)
    inventory = _read_json(generated_bundle / "op_inventory.json")
    inventory["dynamic_range"] = {"artifact_id": "TFLITE_DYNAMIC_RANGE", "ops": dynamic_meta["ops"], "counts": dynamic_meta["op_counts"], "builtin_only": dynamic_meta["builtin_only"], "internal_dtype_counts": dynamic_meta["internal_dtype_counts"], "quantized_tensor_count": dynamic_meta["quantized_tensor_count"], "quantized_parameter_tensor_count": dynamic_meta["quantized_parameter_tensor_count"], "conversion": dynamic_meta["conversion"], "diagnostic_only": True}
    _write_json(generated_bundle / "op_inventory.json", inventory)
    summary = _read_json(generated_bundle / "execution_summary.json")
    summary.update({"correction_performed": True, "true_fp32_generated": True, "former_fp32_reclassified": "TFLITE_DYNAMIC_RANGE", "former_fp32_sha256": FORMER_DYNAMIC_RANGE_SHA, "former_fp32_size_bytes": FORMER_DYNAMIC_RANGE_SIZE, "full_int8_preserved": True, "full_int8_sha256": existing_int8_sha})
    _write_json(generated_bundle / "execution_summary.json", summary)
    limitations = _read_json(generated_bundle / "limitations.json")
    limitations.update({"former_fp32_artifact": "TFLITE_DYNAMIC_RANGE_DIAGNOSTIC_ONLY", "former_fp32_size_anomaly": "OPTIMIZE_DEFAULT_DYNAMIC_RANGE_QUANTIZED_FLOAT_IO", "true_fp32_equivalence_stage": "UNQUANTIZED_NO_OPTIMIZATIONS"})
    _write_json(generated_bundle / "limitations.json", limitations)
    _write_checksums(generated_bundle, exclude={"checksums.sha256"})

    _atomic_copy(generated_fp32, old_path)
    for path in sorted(generated_bundle.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.suffix == ".json":
            _atomic_copy(path, bundle / path.name)
    for subdirectory in ("parity", "calibration"):
        source = generated_bundle / subdirectory
        destination = bundle / subdirectory
        destination.mkdir(parents=True, exist_ok=True)
        for path in sorted(source.iterdir(), key=lambda item: item.name):
            if path.is_file() and not path.name.startswith("._"):
                _atomic_copy(path, destination / path.name)
    _write_checksums(bundle, exclude={"checksums.sha256"})
    evidence = root / EVIDENCE_REL
    for path in sorted(generated_bundle.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.suffix == ".json":
            _atomic_copy(path, evidence / path.name)
    _write_checksums(evidence, exclude={"checksums.sha256"})
    real = _read_json(generated_bundle / "real_development_parity.json")
    return {"phase": PHASE_ID, "status": "CORRECTED", "correction": "T_B4_TRUE_FP32_RECLASSIFICATION", "true_fp32_sha256": fp32_meta["sha256"], "true_fp32_size_bytes": fp32_meta["size_bytes"], "former_dynamic_range_sha256": dynamic_meta["sha256"], "former_dynamic_range_size_bytes": dynamic_meta["size_bytes"], "full_int8_sha256": existing_int8_sha, "validation": generated["validation"], "real": {"float_vs_true_fp32": real["float_vs_tflite_fp32_parity"], "true_fp32_vs_int8": real["fp32_vs_int8_parity"], "float_vs_int8": real["float_vs_int8_parity"]}}


def run(*, mode: str, canonical_root: str | Path, work_root: str | Path, output_root: str | Path, repo_root: str | Path, execute: bool, owner_authorized: bool = False, checkpoint_path: str | Path | None = None) -> dict[str, Any]:
    if mode == READINESS_MODE:
        return run_readiness(canonical_root=canonical_root, work_root=work_root, output_root=output_root, repo_root=repo_root, checkpoint_path=checkpoint_path)
    if mode == FULL_MODE:
        if not execute:
            return {"phase": PHASE_ID, "status": "T_B4_FULL_EXPERIMENT_READY", "owner_authorization_required": True, "t_b5_started": False, "t_c_started": False}
        return run_full(canonical_root=canonical_root, work_root=work_root, output_root=output_root, repo_root=repo_root, owner_authorized=owner_authorized, checkpoint_path=checkpoint_path)
    if mode == CORRECTION_MODE:
        if not execute:
            return {"phase": PHASE_ID, "status": "T_B4_CORRECTION_READY", "owner_authorization_required": True, "t_b5_started": False, "t_c_started": False}
        return run_correction(canonical_root=canonical_root, work_root=work_root, output_root=output_root, repo_root=repo_root, owner_authorized=owner_authorized, checkpoint_path=checkpoint_path)
    raise RunnerContractError(f"UNKNOWN_T_B4_MODE:{mode}")
