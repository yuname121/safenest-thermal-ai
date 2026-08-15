#!/usr/bin/env python3
"""T-A6 bounded-memory Thermal canonical conversion and integrity audits.

The converter is deliberately independent of TFLite, normalization, model
metrics, and augmentation.  It consumes one SDT partition, writes a finalized
float32 Celsius ``.npy`` memmap plus JSONL provenance, and emits deterministic
quality/duplicate summaries.  Stage 1 invokes it only for the materialized
real ``test`` partition; the same API is reusable by the later Colab stage.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.lib.format import open_memmap

from datasets.thermal.canonical_geometry import (
    CANONICAL_SHAPE,
    canonicalize_physical_frame,
    profile_for_id,
)
from datasets.thermal.label_semantics import SOURCE_LABELS, map_source_label
from datasets.thermal.raw_reader import (
    DEFAULT_ARCHIVE_SHA256,
    DEFAULT_ARCHIVE_SIZE,
    DEFAULT_FRAME_COUNT,
    MAX_IMAGE_MEMBER_BYTES,
    SDTThermalRawReader,
    encoded_frame_sha256,
    encoded_to_celsius,
)
from datasets.thermal.split_policy import (
    ASSIGNMENT_RULE_ID,
    DATASET_DOI,
    DATASET_ID,
    SEMANTIC_POLICY_ID,
    SOURCE_ARCHIVE_PATH,
    SOURCE_ARCHIVE_SHA256,
    SPLIT_POLICY_ID,
    TEMPORAL_POLICY_ID,
    assignment_for_real_test_frame,
)


STAGE1_MODE = "MAC_STAGE1"
COLAB_STAGE2_MODE = "COLAB_STAGE2"
ALLOWED_PARTITIONS = {"train", "validation", "test"}
REAL_TEST_ROLE = "REAL_EVAL_DEVELOPMENT"
CANONICAL_DTYPE = np.dtype("<f4")
STORAGE_FORMAT_ID = "NPY_MEMMAP_V1_LITTLE_ENDIAN_FLOAT32"
PROVENANCE_FORMAT_ID = "JSONL_V1_ONE_ROW_PER_CANONICAL_SAMPLE"
NEAR_DUPLICATE_PROFILE_ID = "THERMAL_T_A6_NEAR_DUPLICATE_SCREEN_V1"
NEAR_DUPLICATE_PROFILE_VERSION = "1.0"


class CanonicalConversionError(RuntimeError):
    code = "THERMAL_T_A6_CONVERSION_ERROR"

    def __init__(self, message: str) -> None:
        self.detail = message
        super().__init__(f"{self.code}: {message}")


class SyntheticPayloadAccessProhibitedError(CanonicalConversionError):
    code = "MAC_SYNTHETIC_PAYLOAD_ACCESS_PROHIBITED"


class ConversionIncompleteError(CanonicalConversionError):
    code = "SOURCE_PAYLOAD_INCOMPLETE"


class FinalizationError(CanonicalConversionError):
    code = "CANONICAL_ARTIFACT_NOT_FINALIZED"


@dataclass(frozen=True)
class ConversionConfig:
    """Portable conversion configuration frozen into the compact contract."""

    mode: str
    source_split: str
    source_domain: str
    safenest_role: str
    dataset_id: str = DATASET_ID
    doi: str = DATASET_DOI
    geometry_profile_id: str = "G1_FIXED_ASPECT_CROP_BILINEAR"
    temporal_policy_id: str = TEMPORAL_POLICY_ID
    semantic_policy_id: str = SEMANTIC_POLICY_ID
    split_policy_id: str = SPLIT_POLICY_ID
    assignment_rule_id: str = ASSIGNMENT_RULE_ID
    storage_format: str = STORAGE_FORMAT_ID
    provenance_format: str = PROVENANCE_FORMAT_ID

    def validate(self) -> None:
        if self.mode not in {STAGE1_MODE, COLAB_STAGE2_MODE}:
            raise CanonicalConversionError(f"unsupported conversion mode: {self.mode}")
        if self.source_split not in ALLOWED_PARTITIONS:
            raise CanonicalConversionError(f"unsupported source split: {self.source_split}")
        if self.mode == STAGE1_MODE and self.source_split in {"train", "validation"}:
            raise SyntheticPayloadAccessProhibitedError(
                f"{self.mode} cannot access synthetic source split {self.source_split}"
            )
        if self.source_split == "test" and (self.source_domain, self.safenest_role) != ("REAL", REAL_TEST_ROLE):
            raise CanonicalConversionError("real test must remain REAL_EVAL_DEVELOPMENT")
        if self.source_split in {"train", "validation"} and self.source_domain != "SYNTHETIC":
            raise CanonicalConversionError("synthetic source split has non-synthetic domain")
        if self.storage_format != STORAGE_FORMAT_ID or self.provenance_format != PROVENANCE_FORMAT_ID:
            raise CanonicalConversionError("unsupported storage/provenance format")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def canonical_json_line(value: Any) -> str:
    """Deterministic single-line JSON for the JSONL provenance contract."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_repo_path(value: str) -> str:
    if not isinstance(value, str) or not value or value.startswith(("/", "~/", "file://")) or "\\" in value:
        raise CanonicalConversionError(f"repository-relative POSIX path required: {value!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise CanonicalConversionError(f"unsafe repository path: {value!r}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(canonical_json(value), encoding="utf-8")
    os.replace(temporary, path)


def _clusters(values: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        grouped[value].append(index)
    clusters = []
    for digest, indices in sorted(grouped.items()):
        if len(indices) > 1:
            clusters.append({"hash": digest, "sample_indices": indices, "size": len(indices)})
    return clusters


class _RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self.sum = 0.0
        self.sum_sq = 0.0
        self.frame_means: list[float] = []
        self.frame_mins: list[float] = []
        self.frame_maxs: list[float] = []

    def update(self, values: np.ndarray) -> None:
        finite = np.asarray(values, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(finite)):
            raise ValueError("non-finite values cannot enter quality statistics")
        self.count += int(finite.size)
        self.minimum = min(self.minimum, float(finite.min()))
        self.maximum = max(self.maximum, float(finite.max()))
        self.sum += float(finite.sum(dtype=np.float64))
        self.sum_sq += float(np.square(finite, dtype=np.float64).sum(dtype=np.float64))
        self.frame_means.append(float(finite.mean()))
        self.frame_mins.append(float(finite.min()))
        self.frame_maxs.append(float(finite.max()))

    def summary(self) -> dict[str, Any]:
        mean = self.sum / self.count if self.count else None
        variance = max(0.0, self.sum_sq / self.count - mean * mean) if self.count and mean is not None else None
        return {
            "finite_pixel_count": self.count,
            "minimum_celsius": self.minimum if self.count else None,
            "maximum_celsius": self.maximum if self.count else None,
            "mean_celsius": mean,
            "std_celsius": float(np.sqrt(variance)) if variance is not None else None,
            "frame_mean_p01": float(np.percentile(self.frame_means, 1)) if self.frame_means else None,
            "frame_mean_p50": float(np.percentile(self.frame_means, 50)) if self.frame_means else None,
            "frame_mean_p99": float(np.percentile(self.frame_means, 99)) if self.frame_means else None,
            "frame_minimum_p01": float(np.percentile(self.frame_mins, 1)) if self.frame_mins else None,
            "frame_maximum_p99": float(np.percentile(self.frame_maxs, 99)) if self.frame_maxs else None,
        }


def near_duplicate_profile() -> dict[str, Any]:
    return {
        "profile_id": NEAR_DUPLICATE_PROFILE_ID,
        "version": NEAR_DUPLICATE_PROFILE_VERSION,
        "label_independent": True,
        "model_independent": True,
        "blocking": {
            "sample_grid": "canonical_frame[::8,::8]",
            "quantization_step_celsius": 2.0,
            "fingerprint": "SHA256(little-endian int16 quantized coarse grid)",
            "candidate_pair_order": "lexicographic ascending (left_index,right_index)",
            "max_candidate_pairs": 200000,
        },
        "confirmation_thresholds": {
            "mae_celsius_max": 0.20,
            "rmse_celsius_max": 0.30,
            "max_abs_difference_celsius_max": 1.00,
            "mean_temperature_difference_celsius_max": 0.20,
        },
        "scope": "WITHIN_ROLE_ONLY_UNTIL_COLAB_STAGE_2",
        "exhaustiveness_claim": "DETERMINISTIC_SCREENING_NOT_EXHAUSTIVE",
        "no_label_or_model_tuning": True,
    }


def _coarse_fingerprint(frame: np.ndarray, step: float = 2.0) -> str:
    coarse = np.asarray(frame, dtype=np.float32)[::8, ::8]
    quantized = np.rint(coarse / np.float32(step)).astype("<i2", copy=False)
    return sha256_bytes(quantized.tobytes(order="C"))


def audit_near_duplicates(artifact_path: Path, sample_count: int) -> dict[str, Any]:
    profile = near_duplicate_profile()
    frames = np.load(artifact_path, mmap_mode="r")
    if tuple(frames.shape) != (sample_count, *CANONICAL_SHAPE) or frames.dtype != CANONICAL_DTYPE:
        raise CanonicalConversionError("near-duplicate input artifact shape/dtype mismatch")
    blocks: dict[str, list[int]] = defaultdict(list)
    for index in range(sample_count):
        blocks[_coarse_fingerprint(frames[index])].append(index)
    pairs: list[tuple[int, int]] = []
    for key in sorted(blocks):
        indices = blocks[key]
        pairs.extend(itertools.combinations(indices, 2))
    pairs.sort()
    complete = len(pairs) <= profile["blocking"]["max_candidate_pairs"]
    pairs = pairs[: profile["blocking"]["max_candidate_pairs"]]
    confirmed: list[dict[str, Any]] = []
    parent = list(range(sample_count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    thresholds = profile["confirmation_thresholds"]
    for left, right in pairs:
        delta = np.asarray(frames[left], dtype=np.float64) - np.asarray(frames[right], dtype=np.float64)
        mae = float(np.mean(np.abs(delta)))
        rmse = float(np.sqrt(np.mean(np.square(delta))))
        max_abs = float(np.max(np.abs(delta)))
        mean_delta = float(abs(np.mean(frames[left], dtype=np.float64) - np.mean(frames[right], dtype=np.float64)))
        if (
            mae <= thresholds["mae_celsius_max"]
            and rmse <= thresholds["rmse_celsius_max"]
            and max_abs <= thresholds["max_abs_difference_celsius_max"]
            and mean_delta <= thresholds["mean_temperature_difference_celsius_max"]
        ):
            union(left, right)
            confirmed.append({"left_index": left, "right_index": right, "mae_celsius": mae, "rmse_celsius": rmse, "max_abs_difference_celsius": max_abs, "mean_temperature_difference_celsius": mean_delta})
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(sample_count):
        groups[find(index)].append(index)
    clusters = [{"sample_indices": indices, "size": len(indices)} for indices in sorted(groups.values()) if len(indices) > 1]
    result = {
        "profile": profile,
        "audit_scope": "WITHIN_REAL_EVAL_DEVELOPMENT",
        "sample_count": sample_count,
        "blocking_key_count": len(blocks),
        "candidate_pair_count": len(pairs),
        "candidate_pairs_truncated": not complete,
        "confirmed_pair_count": len(confirmed),
        "confirmed_pairs": confirmed,
        "confirmed_clusters": clusters,
        "confirmed_cluster_sample_count": sum(item["size"] for item in clusters),
        "exhaustiveness_claim": profile["exhaustiveness_claim"],
    }
    result["audit_sha256"] = sha256_bytes(canonical_json(result).encode("utf-8"))
    return result


def _source_frame_record(
    *,
    reader: SDTThermalRawReader,
    identity: Mapping[str, Any],
    index: int,
    info: zipfile.ZipInfo,
    payload: bytes,
    encoded: np.ndarray,
    quality_flags: Sequence[str],
) -> dict[str, Any]:
    label = reader._labels[index]  # validated by inspect_archive before conversion
    return {
        "source_dataset_id": DATASET_ID,
        "source_doi": DATASET_DOI,
        "source_split": "test",
        "source_domain": "REAL",
        "source_archive_path": SOURCE_ARCHIVE_PATH,
        "source_archive_size_bytes": int(identity["size_bytes"]),
        "source_archive_md5": identity["md5"],
        "source_archive_sha256": identity["sha256"],
        "source_member": info.filename,
        "source_member_index": int(reader._thermal_info[index][0]),
        "source_member_crc32": f"{info.CRC:08x}",
        "source_member_sha256": sha256_bytes(payload),
        "source_frame_index": index,
        "source_pose_label": int(label.source_pose_label),
        "source_pose_name": label.source_pose_name,
        "source_bbox": list(label.source_bbox),
        "source_shape": [480, 640],
        "source_dtype": "uint16",
        "source_representation": "RADIOMETRIC_TEMPERATURE_ENCODED_UINT16",
        "source_temperature_encoding": "kelvin_centiunits; celsius=(raw-27315)/100",
        "source_timestamp_status": "ABSENT",
        "source_subject_status": "ABSENT",
        "source_session_status": "ABSENT",
        "source_sequence_status": "ABSENT",
        "source_event_status": "ABSENT",
        "source_frame_sha256": encoded_frame_sha256(encoded),
        "quality_flags": sorted(set(quality_flags)),
    }


def _status_summary(statuses: Sequence[str]) -> dict[str, int]:
    counter = Counter(statuses)
    return {key: int(counter[key]) for key in ("SUCCESS", "SUCCESS_WITH_WARNING", "EXCLUDED", "FAILED")}


def _error_code(exc: BaseException) -> str:
    return str(getattr(exc, "code", exc.__class__.__name__))


def convert_real_test(
    *,
    repo_root: Path,
    source_archive: Path | None = None,
    artifact_dir: Path,
    provenance_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert all 8,000 real test frames using bounded memory and atomic output."""

    config = ConversionConfig(mode=STAGE1_MODE, source_split="test", source_domain="REAL", safenest_role=REAL_TEST_ROLE)
    config.validate()
    archive = Path(source_archive) if source_archive is not None else (repo_root / SOURCE_ARCHIVE_PATH)
    if not archive.is_file():
        raise ConversionIncompleteError(f"real test archive not found: {SOURCE_ARCHIVE_PATH}")
    try:
        # Use lexical paths here: Stage-1 may expose the owner-local payload
        # through a temporary ignored symlink, while persisted provenance must
        # retain only the canonical repository-relative path.
        archive_rel_path = archive.absolute().relative_to(repo_root.absolute()).as_posix()
    except ValueError as exc:
        raise CanonicalConversionError("source archive must be inside the canonical repository root") from exc
    if archive_rel_path != SOURCE_ARCHIVE_PATH:
        raise CanonicalConversionError(
            f"real conversion source path must be {SOURCE_ARCHIVE_PATH}, got {archive_rel_path}"
        )
    artifact_dir = artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "real_eval_development_canonical.npy"
    provenance_path = provenance_path or artifact_dir / "real_eval_development_provenance.jsonl"
    ledger_path = artifact_dir / "real_eval_development_conversion_ledger.json"
    for path in (artifact_path, provenance_path, ledger_path):
        if path.exists() and not overwrite:
            raise FinalizationError(f"output already exists; use overwrite explicitly: {path.name}")
    reader = SDTThermalRawReader(repo_root=repo_root, archive_path=archive_rel_path)
    inventory = reader.inspect_archive()
    identity = inventory["archive_identity"]
    if identity["sha256"] != DEFAULT_ARCHIVE_SHA256 or int(identity["size_bytes"]) != DEFAULT_ARCHIVE_SIZE:
        raise ConversionIncompleteError("real test archive identity differs from locked T-A1 identity")
    profile = profile_for_id(config.geometry_profile_id)
    temp_artifact = artifact_path.with_name(artifact_path.name + ".partial")
    temp_provenance = provenance_path.with_name(provenance_path.name + ".partial")
    for path in (temp_artifact, temp_provenance):
        if path.exists():
            path.unlink()
    mmap = open_memmap(temp_artifact, mode="w+", dtype=CANONICAL_DTYPE, shape=(DEFAULT_FRAME_COUNT, *CANONICAL_SHAPE), fortran_order=False)
    statuses: list[str] = []
    status_rows: list[dict[str, Any]] = []
    source_hashes: list[str] = []
    decoded_hashes: list[str] = []
    canonical_hashes: list[str] = []
    provenance_rows: list[dict[str, Any]] = []
    quality = {
        "corrupt": 0, "truncated": 0, "shape_mismatch": 0, "dtype_mismatch": 0, "channel_mismatch": 0,
        "nonfinite": 0, "full_invalid": 0, "constant_source": 0, "constant_canonical": 0,
        "partial_extreme_warnings": 0, "full_extreme": 0, "warning_code_counts": {},
    }
    stats = _RunningStats()
    try:
        with provenance_path.with_name(provenance_path.name + ".partial").open("w", encoding="utf-8") as provenance_file, zipfile.ZipFile(reader.archive_path, "r") as archive:
            for index in range(DEFAULT_FRAME_COUNT):
                status = "SUCCESS"
                warning_codes: list[str] = []
                try:
                    info = reader._thermal_info[index][1]
                    payload = reader._read_bounded_member(archive, info, limit=MAX_IMAGE_MEMBER_BYTES)
                    encoded, flags = reader._decode_thermal_png(payload, info.filename)
                    source_hashes.append(sha256_bytes(payload))
                    decoded_hashes.append(encoded_frame_sha256(encoded))
                    if "CONSTANT_FRAME" in flags:
                        quality["constant_source"] += 1
                    if "CONTAINER_MIN_PRESENT" in flags or "CONTAINER_MAX_PRESENT" in flags:
                        quality["partial_extreme_warnings"] += 1
                    if int(encoded.min()) == int(encoded.max()) and int(encoded.min()) in (0, np.iinfo(np.uint16).max):
                        quality["full_extreme"] += 1
                    warning_codes.extend(flags)
                    physical = encoded_to_celsius(encoded)
                    if not np.all(np.isfinite(physical)):
                        quality["nonfinite"] += 1
                        raise ValueError("non-finite Celsius values")
                    canonical = canonicalize_physical_frame(physical, profile, source_frame_hash=encoded_frame_sha256(encoded))
                    frame = np.asarray(canonical.physical_frame, dtype=CANONICAL_DTYPE)
                    if frame.shape != CANONICAL_SHAPE:
                        quality["shape_mismatch"] += 1
                        raise ValueError("canonical shape mismatch")
                    if frame.dtype != CANONICAL_DTYPE or not np.all(np.isfinite(frame)):
                        quality["nonfinite"] += 1
                        raise ValueError("canonical dtype/nonfinite mismatch")
                    if not np.any(canonical.validity_mask):
                        quality["full_invalid"] += 1
                        raise ValueError("canonical frame is fully invalid")
                    if float(np.ptp(frame)) == 0.0:
                        quality["constant_canonical"] += 1
                    stats.update(frame)
                    mapped = map_source_label({
                        "dataset_id": DATASET_ID,
                        "source_doi": DATASET_DOI,
                        "source_split": "test",
                        "source_archive_path": SOURCE_ARCHIVE_PATH,
                        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                        "source_member": info.filename,
                        "source_frame_index": index,
                        "original_label_id": reader._labels[index].source_pose_label,
                        "original_label_name": reader._labels[index].source_pose_name,
                        "original_bbox": list(reader._labels[index].source_bbox),
                    })
                    assignment = assignment_for_real_test_frame(mapped)
                    source_record = _source_frame_record(reader=reader, identity=identity, index=index, info=info, payload=payload, encoded=encoded, quality_flags=warning_codes)
                    canonical_hashes.append(canonical.canonical_frame_hash)
                    if warning_codes:
                        status = "SUCCESS_WITH_WARNING"
                    for code in warning_codes:
                        quality["warning_code_counts"][code] = int(quality["warning_code_counts"].get(code, 0)) + 1
                    mmap[index] = frame
                    row = {
                        "canonical_sample_index": index,
                        "stable_sample_id": f"{DATASET_ID}:test:{index:04d}",
                        **source_record,
                        "source_frame_sha256": decoded_hashes[-1],
                        "t_a1_reader_contract": "T-A1_SOURCE_FRAME_PROVENANCE_CONTRACT",
                        "t_a2_geometry_profile_id": config.geometry_profile_id,
                        "t_a3_temporal_policy_id": config.temporal_policy_id,
                        "t_a4_semantic_policy_id": config.semantic_policy_id,
                        "t_a5_split_policy_id": config.split_policy_id,
                        "t_a5_assignment_rule_id": config.assignment_rule_id,
                        "safenest_assignment": assignment,
                        "original_label_id": mapped["original_label_id"],
                        "original_label_name": mapped["original_label_name"],
                        "original_bbox": mapped["original_bbox"],
                        "frame_evidence_label": mapped["frame_evidence_label"],
                        "compatibility_target": mapped["compatibility_target"],
                        "mapping_type": mapped["mapping_type"],
                        "mapping_rule_id": mapped["mapping_rule_id"],
                        "claim_scope": mapped["claim_scope"],
                        "canonical_shape": list(CANONICAL_SHAPE),
                        "canonical_dtype": "float32",
                        "canonical_unit": "CELSIUS",
                        "canonical_frame_hash": canonical.canonical_frame_hash,
                        "canonical_tensor_row_sha256": sha256_bytes(np.asarray(frame, dtype=CANONICAL_DTYPE, order="C").tobytes(order="C")),
                        "quality_status": status,
                        "quality_warning_codes": sorted(set(warning_codes)),
                        "conversion_status": status,
                    }
                    provenance_file.write(canonical_json_line(row))
                    provenance_rows.append(row)
                except Exception as exc:  # every source frame receives an explicit terminal status
                    status = "FAILED"
                    code = _error_code(exc)
                    if "TRUNCAT" in code:
                        quality["truncated"] += 1
                    elif "SHAPE" in code:
                        quality["shape_mismatch"] += 1
                    elif "DTYPE" in code:
                        quality["dtype_mismatch"] += 1
                    elif "CHANNEL" in code:
                        quality["channel_mismatch"] += 1
                    elif "FINITE" in code:
                        quality["nonfinite"] += 1
                    else:
                        quality["corrupt"] += 1
                    status_rows.append({"source_frame_index": index, "status": status, "error_code": code, "message": str(exc)})
                statuses.append(status)
    finally:
        mmap.flush()
        del mmap
    if len(statuses) != DEFAULT_FRAME_COUNT:
        raise ConversionIncompleteError(f"source accounting has {len(statuses)} rows, expected {DEFAULT_FRAME_COUNT}")
    os.replace(temp_artifact, artifact_path)
    os.replace(temp_provenance, provenance_path)
    summary = {
        "phase": "T-A6_STAGE1",
        "mode": config.mode,
        "source_split": config.source_split,
        "source_domain": config.source_domain,
        "safenest_role": config.safenest_role,
        "expected_source_frames": DEFAULT_FRAME_COUNT,
        "source_frames_measured": len(statuses),
        "status_counts": _status_summary(statuses),
        "canonical_rows": len(provenance_rows),
        "reconciliation": {
            "expected_equals_status_sum": len(statuses) == sum(_status_summary(statuses).values()),
            "canonical_rows_equals_success_plus_warning": len(provenance_rows) == _status_summary(statuses)["SUCCESS"] + _status_summary(statuses)["SUCCESS_WITH_WARNING"],
        },
        "canonical_shape": list(CANONICAL_SHAPE),
        "canonical_dtype": "float32",
        "canonical_unit": "CELSIUS",
        "geometry_profile_id": config.geometry_profile_id,
        "storage_format": config.storage_format,
        "provenance_format": config.provenance_format,
        "artifact_path": "datasets/thermal/artifacts/T-A6_real_eval_development/real_eval_development_canonical.npy",
        "provenance_path": "datasets/thermal/artifacts/T-A6_real_eval_development/real_eval_development_provenance.jsonl",
        "quality": {**quality, "temperature_distribution": stats.summary()},
        "failed_rows": status_rows,
        "source_member_byte_hashes": source_hashes,
        "decoded_frame_hashes": decoded_hashes,
        "canonical_frame_hashes": canonical_hashes,
        "provenance_rows": provenance_rows,
    }
    def rich_clusters(values: Sequence[str]) -> list[dict[str, Any]]:
        result = []
        for cluster in _clusters(values):
            indices = cluster["sample_indices"]
            result.append({
                **cluster,
                "source_members": [provenance_rows[index]["source_member"] for index in indices],
                "original_labels": [provenance_rows[index]["original_label_name"] for index in indices],
                "compatibility_targets": [provenance_rows[index]["compatibility_target"] for index in indices],
            })
        return result

    source_clusters = rich_clusters(source_hashes)
    decoded_clusters = rich_clusters(decoded_hashes)
    canonical_clusters = rich_clusters(canonical_hashes)
    exact = {
        "source_member_byte_hashes": {"cluster_count": len(source_clusters), "duplicate_sample_count": sum(item["size"] for item in source_clusters), "clusters": source_clusters},
        "decoded_frame_hashes": {"cluster_count": len(decoded_clusters), "duplicate_sample_count": sum(item["size"] for item in decoded_clusters), "clusters": decoded_clusters},
        "canonical_frame_hashes": {"cluster_count": len(canonical_clusters), "duplicate_sample_count": sum(item["size"] for item in canonical_clusters), "clusters": canonical_clusters},
        "audit_scope": "WITHIN_REAL_EVAL_DEVELOPMENT",
        "exclusions_caused_by_duplicates": 0,
        "cluster_metadata_fields": ["sample_indices", "source_members", "original_labels", "compatibility_targets"],
    }
    exact["audit_sha256"] = sha256_bytes(canonical_json(exact).encode("utf-8"))
    summary["exact_duplicate_audit"] = exact
    summary["quality"]["silent_skips"] = 0
    summary.pop("source_member_byte_hashes")
    summary.pop("decoded_frame_hashes")
    summary.pop("canonical_frame_hashes")
    summary.pop("provenance_rows")
    summary["artifact_sha256"] = sha256_file(artifact_path)
    summary["provenance_sha256"] = sha256_file(provenance_path)
    summary["artifact_size_bytes"] = artifact_path.stat().st_size
    summary["provenance_size_bytes"] = provenance_path.stat().st_size
    summary["finalized_status"] = "FINALIZED"
    _atomic_json(ledger_path, summary)
    return summary


def verify_provenance_alignment(artifact_path: Path, provenance_path: Path, sample_count: int) -> dict[str, Any]:
    frames = np.load(artifact_path, mmap_mode="r")
    if tuple(frames.shape) != (sample_count, *CANONICAL_SHAPE) or frames.dtype != CANONICAL_DTYPE:
        raise FinalizationError("canonical artifact shape/dtype does not match contract")
    rows: list[dict[str, Any]] = []
    with provenance_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            row = json.loads(line)
            if row.get("canonical_sample_index") != line_number:
                raise FinalizationError(f"provenance row index mismatch at {line_number}")
            if row.get("source_frame_index") != line_number:
                raise FinalizationError(f"source frame index mismatch at {line_number}")
            if row.get("source_member") != f"test/image_t_{line_number}.png":
                raise FinalizationError(f"source member mismatch at {line_number}")
            if row.get("safenest_assignment", {}).get("safenest_assignment_role") != REAL_TEST_ROLE:
                raise FinalizationError(f"assignment role mismatch at {line_number}")
            tensor_hash = sha256_bytes(np.asarray(frames[line_number], dtype=CANONICAL_DTYPE, order="C").tobytes(order="C"))
            if row.get("canonical_tensor_row_sha256") != tensor_hash:
                raise FinalizationError(f"tensor/provenance hash mismatch at {line_number}")
            rows.append(row)
    if len(rows) != sample_count:
        raise FinalizationError(f"provenance row count {len(rows)} != {sample_count}")
    label_counts = Counter(str(row["original_label_name"]) for row in rows)
    proxy_counts = Counter(str(row["compatibility_target"]) for row in rows)
    return {
        "tensor_row_count": int(frames.shape[0]),
        "provenance_row_count": len(rows),
        "label_row_count": len(rows),
        "assignment_row_count": len(rows),
        "tensor_provenance_1_to_1": True,
        "tensor_label_1_to_1": True,
        "tensor_assignment_1_to_1": True,
        "original_label_counts": {key: int(label_counts[key]) for key in sorted(label_counts)},
        "compatibility_target_counts": {key: int(proxy_counts[key]) for key in sorted(proxy_counts)},
        "provenance_sha256": sha256_file(provenance_path),
    }


def finalize_and_audit_real_artifact(artifact_dir: Path, *, sample_count: int = DEFAULT_FRAME_COUNT) -> dict[str, Any]:
    artifact_path = artifact_dir / "real_eval_development_canonical.npy"
    provenance_path = artifact_dir / "real_eval_development_provenance.jsonl"
    ledger_path = artifact_dir / "real_eval_development_conversion_ledger.json"
    if not artifact_path.is_file() or not provenance_path.is_file() or not ledger_path.is_file():
        raise FinalizationError("canonical artifact, provenance, and finalized ledger are all required")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("finalized_status") != "FINALIZED":
        raise FinalizationError("ledger is not finalized")
    alignment = verify_provenance_alignment(artifact_path, provenance_path, sample_count)
    if ledger.get("artifact_sha256") != sha256_file(artifact_path) or ledger.get("provenance_sha256") != sha256_file(provenance_path):
        raise FinalizationError("ledger checksum does not match finalized outputs")
    return alignment


def _synthetic_mapping(label_id: int, label_name: str) -> dict[str, Any]:
    """Apply the same source/proxy separation for a staged SDT partition.

    T-A4's selected mapping is defined over the already-audited real test
    partition.  For future synthetic conversion we preserve the identical
    posture semantics in a partition-neutral record and never promote the
    proxy to event ground truth.
    """
    targets = {"LYING": "HUMAN_FALL", "SITTING": "HUMAN_NORMAL", "STANDING": "HUMAN_NORMAL", "EMPTY_ROOM": "NOT_HUMAN"}
    evidence = {"LYING": "HUMAN_LYING_POSTURE", "SITTING": "HUMAN_SITTING_POSTURE", "STANDING": "HUMAN_STANDING_POSTURE", "EMPTY_ROOM": "NO_ANNOTATED_HUMAN_IN_REPRESENTED_FRAME"}
    mapping_type = "DIRECT_SOURCE_EQUIVALENT" if label_name == "EMPTY_ROOM" else "DERIVED_POSTURE_PROXY"
    rule = {
        "LYING": "THERMAL_MAP_LYING_TO_FALL_COMPAT_PROXY_001",
        "SITTING": "THERMAL_MAP_SITTING_TO_NON_LYING_PROXY_001",
        "STANDING": "THERMAL_MAP_STANDING_TO_NON_LYING_PROXY_001",
        "EMPTY_ROOM": "THERMAL_MAP_EMPTY_ROOM_TO_NO_HUMAN_001",
    }[label_name]
    scope = {
        "LYING": ["FRAME_LEVEL_FALL_COMPATIBILITY_PROXY_ONLY", "NOT_TEMPORAL_EVENT_GROUND_TRUTH", "NOT_SAFETY_GROUND_TRUTH"],
        "SITTING": ["FRAME_LEVEL_POSTURE_PROXY", "NOT_SAFETY_GROUND_TRUTH"],
        "STANDING": ["FRAME_LEVEL_POSTURE_PROXY", "NOT_SAFETY_GROUND_TRUTH"],
        "EMPTY_ROOM": ["FRAME_LEVEL_PRESENCE_ONLY", "NOT_SAFETY_GROUND_TRUTH"],
    }[label_name]
    return {
        "original_label_id": int(label_id),
        "original_label_name": label_name,
        "frame_evidence_label": evidence[label_name],
        "compatibility_target": targets[label_name],
        "mapping_type": mapping_type,
        "mapping_rule_id": rule,
        "claim_scope": scope,
        "source_label_modified": False,
        "fall_event_semantic_status": "NOT_VERIFIABLE",
        "temporal_event_status": "NOT_VERIFIABLE",
    }


def convert_sdt_partition(
    *,
    config: ConversionConfig,
    source_archive: Path,
    artifact_dir: Path,
    expected_count: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert one staged SDT synthetic partition with the frozen T-A6 path.

    This function is intentionally called only after the Colab runner has
    verified Drive completeness and, for multipart train data, reconstructed a
    validated logical ZIP.  It does not know or assume a personal Drive path.
    """
    config.validate()
    if config.mode != COLAB_STAGE2_MODE or config.source_split not in {"train", "validation"} or config.source_domain != "SYNTHETIC":
        raise CanonicalConversionError("convert_sdt_partition requires COLAB_STAGE2 synthetic train/validation configuration")
    if not source_archive.is_file():
        raise ConversionIncompleteError(f"staged {config.source_split} archive is unavailable")
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stem = config.source_split
    artifact_path = artifact_dir / f"{stem}_canonical.npy"
    provenance_path = artifact_dir / f"{stem}_provenance.jsonl"
    ledger_path = artifact_dir / f"{stem}_conversion_ledger.json"
    if any(path.exists() for path in (artifact_path, provenance_path, ledger_path)) and not overwrite:
        raise FinalizationError(f"synthetic output exists; use overwrite explicitly: {stem}")
    profile = profile_for_id(config.geometry_profile_id)
    temp_artifact = artifact_path.with_name(artifact_path.name + ".partial")
    temp_provenance = provenance_path.with_name(provenance_path.name + ".partial")
    for path in (temp_artifact, temp_provenance):
        path.unlink(missing_ok=True)
    identity = {"size_bytes": int(source_archive.stat().st_size), "sha256": sha256_file(source_archive), "path": f"COLAB_STAGED/{stem}.zip"}
    thermal_infos: dict[int, zipfile.ZipInfo] = {}
    labels = None
    with zipfile.ZipFile(source_archive, "r") as archive:
        names = {info.filename: info for info in archive.infolist()}
        label_name = f"{stem}/labels.txt"
        if label_name not in names:
            raise ConversionIncompleteError(f"{label_name} missing from staged archive")
        labels = SDTThermalRawReader._parse_labels(
            SDTThermalRawReader._read_bounded_member(archive, names[label_name], limit=MAX_IMAGE_MEMBER_BYTES), expected_count
        )
        for index in range(expected_count):
            member = f"{stem}/image_t_{index}.png"
            if member not in names:
                raise ConversionIncompleteError(f"{member} missing from staged archive")
            thermal_infos[index] = names[member]
    mmap = open_memmap(temp_artifact, mode="w+", dtype=CANONICAL_DTYPE, shape=(expected_count, *CANONICAL_SHAPE), fortran_order=False)
    statuses: list[str] = []
    source_hashes: list[str] = []
    decoded_hashes: list[str] = []
    canonical_hashes: list[str] = []
    failed_rows: list[dict[str, Any]] = []
    stats = _RunningStats()
    try:
        with zipfile.ZipFile(source_archive, "r") as archive, temp_provenance.open("w", encoding="utf-8") as provenance:
            for index in range(expected_count):
                try:
                    info = thermal_infos[index]
                    payload = SDTThermalRawReader._read_bounded_member(archive, info, limit=MAX_IMAGE_MEMBER_BYTES)
                    encoded, flags = SDTThermalRawReader._decode_thermal_png(payload, info.filename)
                    physical = encoded_to_celsius(encoded)
                    canonical = canonicalize_physical_frame(physical, profile, source_frame_hash=encoded_frame_sha256(encoded))
                    frame = np.asarray(canonical.physical_frame, dtype=CANONICAL_DTYPE, order="C")
                    stats.update(frame)
                    label = labels[index]
                    mapped = _synthetic_mapping(label.source_pose_label, label.source_pose_name)
                    row = {
                        "canonical_sample_index": index,
                        "stable_sample_id": f"{DATASET_ID}:{stem}:{index:05d}",
                        "source_dataset_id": DATASET_ID,
                        "source_doi": DATASET_DOI,
                        "source_split": stem,
                        "source_domain": "SYNTHETIC",
                        "source_archive_path": identity["path"],
                        "source_archive_size_bytes": identity["size_bytes"],
                        "source_archive_sha256": identity["sha256"],
                        "source_member": info.filename,
                        "source_member_index": int(index),
                        "source_member_crc32": f"{info.CRC:08x}",
                        "source_member_sha256": sha256_bytes(payload),
                        "source_frame_index": index,
                        "source_frame_sha256": encoded_frame_sha256(encoded),
                        "source_pose_label": int(label.source_pose_label),
                        "source_pose_name": label.source_pose_name,
                        "source_bbox": list(label.source_bbox),
                        "source_shape": [480, 640],
                        "source_dtype": "uint16",
                        "source_representation": "RADIOMETRIC_TEMPERATURE_ENCODED_UINT16",
                        "source_temperature_encoding": "kelvin_centiunits; celsius=(raw-27315)/100",
                        "source_subject_status": "ABSENT", "source_session_status": "ABSENT", "source_sequence_status": "ABSENT", "source_event_status": "ABSENT",
                        "t_a1_reader_contract": "T-A1_SOURCE_FRAME_PROVENANCE_CONTRACT",
                        "t_a2_geometry_profile_id": config.geometry_profile_id,
                        "t_a3_temporal_policy_id": config.temporal_policy_id,
                        "t_a4_semantic_policy_id": config.semantic_policy_id,
                        "t_a5_split_policy_id": config.split_policy_id,
                        "t_a5_assignment_rule_id": config.assignment_rule_id,
                        "safenest_assignment": {"safenest_assignment_role": config.safenest_role, "source_split": stem, "source_domain": "SYNTHETIC", "split_assignment_status": "ASSIGNED_T_A5_IMMUTABLE"},
                        **mapped,
                        "canonical_shape": list(CANONICAL_SHAPE), "canonical_dtype": "float32", "canonical_unit": "CELSIUS",
                        "canonical_frame_hash": canonical.canonical_frame_hash,
                        "canonical_tensor_row_sha256": sha256_bytes(frame.tobytes(order="C")),
                        "quality_status": "SUCCESS_WITH_WARNING" if flags else "SUCCESS",
                        "quality_warning_codes": sorted(flags), "conversion_status": "SUCCESS_WITH_WARNING" if flags else "SUCCESS",
                    }
                    mmap[index] = frame
                    provenance.write(canonical_json_line(row))
                    statuses.append(row["conversion_status"])
                    source_hashes.append(row["source_member_sha256"]); decoded_hashes.append(row["source_frame_sha256"]); canonical_hashes.append(row["canonical_frame_hash"])
                except Exception as exc:
                    failed_rows.append({"source_frame_index": index, "status": "FAILED", "error_code": _error_code(exc), "message": str(exc)})
                    statuses.append("FAILED")
    finally:
        mmap.flush(); del mmap
    if failed_rows:
        temp_artifact.unlink(missing_ok=True)
        temp_provenance.unlink(missing_ok=True)
        raise ConversionIncompleteError(
            f"{stem} conversion failed for {len(failed_rows)} source frames; no canonical artifact was finalized: {failed_rows[:3]}"
        )
    os.replace(temp_artifact, artifact_path)
    os.replace(temp_provenance, provenance_path)
    exact = {
        "audit_scope": f"WITHIN_{config.safenest_role}",
        "source_member_byte_hashes": {"cluster_count": len(_clusters(source_hashes)), "clusters": _clusters(source_hashes)},
        "decoded_frame_hashes": {"cluster_count": len(_clusters(decoded_hashes)), "clusters": _clusters(decoded_hashes)},
        "canonical_frame_hashes": {"cluster_count": len(_clusters(canonical_hashes)), "clusters": _clusters(canonical_hashes)},
    }
    summary = {
        "phase": "T-A6_COLAB_STAGE2", "mode": config.mode, "source_split": stem, "source_domain": "SYNTHETIC", "safenest_role": config.safenest_role,
        "expected_source_frames": expected_count, "source_frames_measured": len(statuses), "status_counts": _status_summary(statuses), "canonical_rows": len(statuses),
        "canonical_shape": list(CANONICAL_SHAPE), "canonical_dtype": "float32", "canonical_unit": "CELSIUS", "geometry_profile_id": config.geometry_profile_id,
        "artifact_path": f"{artifact_dir.name}/{artifact_path.name}", "provenance_path": f"{artifact_dir.name}/{provenance_path.name}",
        "artifact_sha256": sha256_file(artifact_path), "provenance_sha256": sha256_file(provenance_path), "artifact_size_bytes": artifact_path.stat().st_size, "provenance_size_bytes": provenance_path.stat().st_size,
        "quality": {"temperature_distribution": stats.summary(), "warning_count": sum(1 for status in statuses if status == "SUCCESS_WITH_WARNING"), "silent_skips": 0, "failed_rows": failed_rows}, "exact_duplicate_audit": exact,
        "finalized_status": "FINALIZED",
    }
    _atomic_json(ledger_path, summary)
    return summary


def convert_partition(
    *,
    config: ConversionConfig,
    repo_root: Path,
    source_archive: Path,
    artifact_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Partition-independent dispatch with an explicit Mac safety boundary.

    Stage 1 invokes only the real-test path.  Later Colab code performs the
    synthetic multipart checks and supplies a staged logical archive; keeping
    this dispatch explicit prevents a caller on the Mac from accidentally
    treating a cloud placeholder as a synthetic source.
    """
    config.validate()
    if config.mode == STAGE1_MODE:
        if config.source_split != "test":
            raise SyntheticPayloadAccessProhibitedError(
                f"{config.mode} cannot convert synthetic source split {config.source_split}"
            )
        return convert_real_test(
            repo_root=repo_root,
            source_archive=source_archive,
            artifact_dir=artifact_dir,
            overwrite=overwrite,
        )
    if config.mode == COLAB_STAGE2_MODE and config.source_split in {"train", "validation"}:
        expected = 32000 if config.source_split == "train" else 8000
        return convert_sdt_partition(
            config=config,
            source_archive=source_archive,
            artifact_dir=artifact_dir,
            expected_count=expected,
            overwrite=overwrite,
        )
    raise CanonicalConversionError("unsupported partition conversion request")


__all__ = [
    "CANONICAL_DTYPE", "CANONICAL_SHAPE", "COLAB_STAGE2_MODE", "ConversionConfig", "CanonicalConversionError", "ConversionIncompleteError", "FinalizationError", "NEAR_DUPLICATE_PROFILE_ID", "REAL_TEST_ROLE", "STAGE1_MODE", "SyntheticPayloadAccessProhibitedError", "audit_near_duplicates", "canonical_json", "canonical_json_line", "convert_partition", "convert_real_test", "convert_sdt_partition", "finalize_and_audit_real_artifact", "near_duplicate_profile", "sha256_file", "verify_provenance_alignment",
]
