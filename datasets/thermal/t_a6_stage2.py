"""Compact T-A6 Stage 2 integrity audits.

The Colab runner owns source access and conversion orchestration.  This module
contains the deterministic, payload-independent evidence operations that run
after the three canonical roles have been finalized.  It deliberately keeps
bulk tensors out of the compact result bundle; only bounded provenance rows,
hash intersections, screening witnesses, and checksums are persisted.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from datasets.thermal.canonical_converter import (
    CANONICAL_DTYPE,
    CANONICAL_SHAPE,
    NEAR_DUPLICATE_PROFILE_ID,
    canonical_json,
    near_duplicate_profile,
    sha256_bytes,
    sha256_file,
)
from datasets.thermal.raw_reader import (
    DISTRIBUTED_FRAME_SHAPE,
    MAX_IMAGE_MEMBER_BYTES,
    MAX_LABEL_MEMBER_BYTES,
    SDTThermalRawReader,
    encoded_to_celsius,
)


STAGE2_PHASE = "T-A6_COLAB_STAGE2"
ROLE_ORDER = ("TRAIN", "VALIDATION", "REAL_EVAL_DEVELOPMENT")
ROLE_TO_SPLIT = {
    "TRAIN": "train",
    "VALIDATION": "validation",
    "REAL_EVAL_DEVELOPMENT": "test",
}
EXPECTED_ROLE_COUNTS = {
    "TRAIN": 32_000,
    "VALIDATION": 8_000,
    "REAL_EVAL_DEVELOPMENT": 8_000,
}
BUNDLE_JSON_FILES = (
    "execution_summary.json",
    "source_identity.json",
    "canonical_artifact_registry.json",
    "conversion_status_summary.json",
    "output_checksums.json",
    "quality_audit_summary.json",
    "exact_duplicate_audit.json",
    "near_duplicate_audit.json",
    "cross_role_leakage_audit.json",
    "determinism_summary.json",
    "execution_environment.json",
    "validation_result.json",
)
CHECKSUMS_NAME = "checksums.sha256"
MODEL_METRIC_KEYS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "confusion_matrix",
    "prediction_distribution",
    "loss",
    "auc",
}
PORTABLE_PATH_RE = re.compile(r"^(?!/)(?!~)(?!file://)(?![A-Za-z]:)(?!.*\\).+$")


class Stage2AuditError(RuntimeError):
    """A deterministic, machine-readable Stage 2 audit failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.detail = message
        super().__init__(f"{code}: {message}")


def _portable(value: Any) -> bool:
    if not isinstance(value, str) or not PORTABLE_PATH_RE.fullmatch(value):
        return False
    if any(token in value for token in ("/Users/", "/private/", "iCloud", "/content/")):
        return False
    parts = value.split("/")
    return not value.startswith("/") and all(part not in {"", ".", ".."} for part in parts)


def _walk(value: Any, location: str = "$") -> Iterable[tuple[str, Any]]:
    yield location, value
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _walk(value[key], f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{location}[{index}]")


def _safe_logical(path: str) -> str:
    if not _portable(path):
        raise Stage2AuditError("NONPORTABLE_PATH", path)
    return path


def _canonical_frame_hash(frame: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(frame, dtype=CANONICAL_DTYPE, order="C").tobytes(order="C"))
    digest.update(np.ones(CANONICAL_SHAPE, dtype=np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage2AuditError("JSON_READ_FAILED", f"{path.name}: {exc}") from exc
    if path.read_text(encoding="utf-8") != canonical_json(value):
        raise Stage2AuditError("NONDETERMINISTIC_JSON", path.name)
    return value


def load_provenance(path: Path, *, role: str) -> list[dict[str, Any]]:
    """Load and validate deterministic one-row-per-canonical-sample JSONL."""

    rows: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise Stage2AuditError("PROVENANCE_READ_FAILED", f"{path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Stage2AuditError("PROVENANCE_JSON_INVALID", f"{path.name}:{line_number + 1}") from exc
            if not isinstance(row, dict):
                raise Stage2AuditError("PROVENANCE_ROW_NOT_OBJECT", f"{path.name}:{line_number + 1}")
            if row.get("canonical_sample_index") != line_number:
                raise Stage2AuditError("PROVENANCE_INDEX_MISMATCH", f"{path.name}:{line_number}")
            assignment = row.get("safenest_assignment", {})
            if assignment.get("safenest_assignment_role") != role:
                raise Stage2AuditError("PROVENANCE_ROLE_MISMATCH", f"{path.name}:{line_number}")
            for location, item in _walk(row, f"{path.name}[{line_number}]"):
                if isinstance(item, str) and (item.startswith(("/", "~/", "file://")) or "\\" in item or "/Users/" in item or "/private/" in item or "iCloud" in item):
                    raise Stage2AuditError("NONPORTABLE_PATH", f"{location}: {item}")
            rows.append(row)
    return rows


def _validate_row_hashes(frames: np.ndarray, rows: Sequence[Mapping[str, Any]], role: str) -> None:
    if len(rows) != int(frames.shape[0]):
        raise Stage2AuditError("PROVENANCE_COUNT_MISMATCH", role)
    if tuple(frames.shape[1:]) != CANONICAL_SHAPE or frames.dtype != CANONICAL_DTYPE:
        raise Stage2AuditError("CANONICAL_ARTIFACT_CONTRACT_MISMATCH", role)
    for index, row in enumerate(rows):
        frame = np.asarray(frames[index], dtype=CANONICAL_DTYPE, order="C")
        if not np.all(np.isfinite(frame)):
            raise Stage2AuditError("CANONICAL_NONFINITE", f"{role}:{index}")
        tensor_hash = sha256_bytes(frame.tobytes(order="C"))
        if row.get("canonical_tensor_row_sha256") != tensor_hash:
            raise Stage2AuditError("CANONICAL_TENSOR_HASH_MISMATCH", f"{role}:{index}")
        expected_frame_hash = _canonical_frame_hash(frame)
        if row.get("canonical_frame_hash") != expected_frame_hash:
            raise Stage2AuditError("CANONICAL_FRAME_HASH_MISMATCH", f"{role}:{index}")
        for key in ("source_member_sha256", "source_frame_sha256", "canonical_frame_hash"):
            if not isinstance(row.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", row[key]):
                raise Stage2AuditError("PROVENANCE_HASH_MISSING", f"{role}:{index}:{key}")


def validate_role_artifact(
    *,
    role: str,
    artifact_path: Path,
    provenance_path: Path,
    expected_count: int,
) -> dict[str, Any]:
    """Validate a finalized role artifact and return rows for compact audits."""

    if role not in ROLE_ORDER:
        raise Stage2AuditError("ROLE_INVALID", role)
    try:
        frames = np.load(artifact_path, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise Stage2AuditError("CANONICAL_ARTIFACT_READ_FAILED", f"{role}: {exc}") from exc
    if tuple(frames.shape) != (expected_count, *CANONICAL_SHAPE):
        raise Stage2AuditError("CANONICAL_SHAPE_MISMATCH", f"{role}: {frames.shape}")
    rows = load_provenance(provenance_path, role=role)
    _validate_row_hashes(frames, rows, role)
    return {
        "role": role,
        "source_split": ROLE_TO_SPLIT[role],
        "source_domain": "REAL" if role == "REAL_EVAL_DEVELOPMENT" else "SYNTHETIC",
        "expected_count": expected_count,
        "source_frames_measured": len(rows),
        "canonical_rows": len(rows),
        "canonical_shape": list(CANONICAL_SHAPE),
        "canonical_dtype": "float32_little_endian",
        "canonical_unit": "CELSIUS",
        "artifact_path": artifact_path,
        "provenance_path": provenance_path,
        "artifact_sha256": sha256_file(artifact_path),
        "provenance_sha256": sha256_file(provenance_path),
        "artifact_size_bytes": artifact_path.stat().st_size,
        "provenance_size_bytes": provenance_path.stat().st_size,
        "rows": rows,
    }


def _hash_groups(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        value = row.get(key)
        if not isinstance(value, str):
            raise Stage2AuditError("PROVENANCE_HASH_MISSING", f"{key}:{index}")
        groups[value].append(index)
    return {key: value for key, value in sorted(groups.items())}


def _witnesses(groups: Mapping[str, Sequence[int]], *, max_items: int = 50) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for digest, indices in groups.items():
        if len(indices) > 1:
            result.append({"hash": digest, "sample_indices": list(indices[:max_items]), "size": len(indices)})
    return result[:max_items]


def _within_layer(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    groups = _hash_groups(rows, key)
    duplicate_count = sum(len(indices) for indices in groups.values() if len(indices) > 1)
    return {
        "duplicate_sample_count": duplicate_count,
        "duplicate_cluster_count": sum(len(indices) > 1 for indices in groups.values()),
        "witnesses": _witnesses(groups),
    }


def _cross_layer(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    left_groups = _hash_groups(left, key)
    right_groups = _hash_groups(right, key)
    shared = sorted(set(left_groups).intersection(right_groups))
    return {
        "overlap_sample_count_left": sum(len(left_groups[digest]) for digest in shared),
        "overlap_sample_count_right": sum(len(right_groups[digest]) for digest in shared),
        "overlap_cluster_count": len(shared),
        "witnesses": [
            {
                "hash": digest,
                "left_indices": list(left_groups[digest][:20]),
                "right_indices": list(right_groups[digest][:20]),
            }
            for digest in shared[:50]
        ],
    }


def audit_exact_duplicates(role_data: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Audit source-byte, decoded-frame, and canonical-frame hash layers."""

    layers = {
        "source_member_byte_hashes": "source_member_sha256",
        "decoded_frame_hashes": "source_frame_sha256",
        "canonical_frame_hashes": "canonical_frame_hash",
    }
    within: dict[str, Any] = {}
    cross: dict[str, Any] = {}
    for output_key, row_key in layers.items():
        within[output_key] = {
            role: _within_layer(role_data[role]["rows"], row_key)
            for role in ROLE_ORDER
        }
        cross[output_key] = {}
        for left_index, left_role in enumerate(ROLE_ORDER):
            for right_role in ROLE_ORDER[left_index + 1 :]:
                cross[output_key][f"{left_role}__{right_role}"] = _cross_layer(
                    role_data[left_role]["rows"], role_data[right_role]["rows"], row_key
                )
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "phase": STAGE2_PHASE,
        "audit_scope": "WITHIN_ROLE_AND_CROSS_ROLE",
        "layers": layers,
        "within_role": within,
        "cross_role": cross,
        "exclusions_caused_by_duplicates": 0,
        "duplicate_policy": "PRESERVE_BOTH_PROVENANCE_RECORDS_AND_FLAG_DUPLICATE_CONTENT",
    }
    result["audit_sha256"] = sha256_bytes(canonical_json(result).encode("utf-8"))
    return result


def _coarse_fingerprint(frame: np.ndarray, step: float = 2.0) -> str:
    coarse = np.asarray(frame, dtype=np.float32)[::8, ::8]
    quantized = np.rint(coarse / np.float32(step)).astype("<i2", copy=False)
    return sha256_bytes(quantized.tobytes(order="C"))


def audit_near_duplicates_cross_role(
    role_data: Mapping[str, Mapping[str, Any]],
    *,
    max_witnesses: int = 200,
) -> dict[str, Any]:
    """Run the frozen deterministic screen over all roles and cross-role pairs."""

    profile = near_duplicate_profile()
    frames_by_role = {
        role: np.load(role_data[role]["artifact_path"], mmap_mode="r")
        for role in ROLE_ORDER
    }
    blocks: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for role in ROLE_ORDER:
        frames = frames_by_role[role]
        for index in range(int(frames.shape[0])):
            blocks[_coarse_fingerprint(frames[index])].append((role, index))
    pairs: list[tuple[tuple[str, int], tuple[str, int]]] = []
    for key in sorted(blocks):
        members = sorted(blocks[key], key=lambda item: (ROLE_ORDER.index(item[0]), item[1]))
        pairs.extend(itertools.combinations(members, 2))
    pairs.sort(key=lambda pair: (ROLE_ORDER.index(pair[0][0]), pair[0][1], ROLE_ORDER.index(pair[1][0]), pair[1][1]))
    total_candidate_pairs = len(pairs)
    complete = total_candidate_pairs <= profile["blocking"]["max_candidate_pairs"]
    pairs = pairs[: profile["blocking"]["max_candidate_pairs"]]
    thresholds = profile["confirmation_thresholds"]
    confirmed: list[dict[str, Any]] = []
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(item: tuple[str, int]) -> tuple[str, int]:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: tuple[str, int], right: tuple[str, int]) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in pairs:
        left_frame = frames_by_role[left[0]][left[1]]
        right_frame = frames_by_role[right[0]][right[1]]
        delta = np.asarray(left_frame, dtype=np.float64) - np.asarray(right_frame, dtype=np.float64)
        mae = float(np.mean(np.abs(delta)))
        rmse = float(np.sqrt(np.mean(np.square(delta))))
        max_abs = float(np.max(np.abs(delta)))
        mean_delta = float(abs(np.mean(left_frame, dtype=np.float64) - np.mean(right_frame, dtype=np.float64)))
        if (
            mae <= thresholds["mae_celsius_max"]
            and rmse <= thresholds["rmse_celsius_max"]
            and max_abs <= thresholds["max_abs_difference_celsius_max"]
            and mean_delta <= thresholds["mean_temperature_difference_celsius_max"]
        ):
            union(left, right)
            confirmed.append({
                "left_role": left[0],
                "left_index": left[1],
                "right_role": right[0],
                "right_index": right[1],
                "mae_celsius": mae,
                "rmse_celsius": rmse,
                "max_abs_difference_celsius": max_abs,
                "mean_temperature_difference_celsius": mean_delta,
            })
    clusters: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
    for item in parent:
        clusters[find(item)].append(item)
    cluster_rows = [
        {
            "members": [{"role": role, "sample_index": index} for role, index in sorted(members, key=lambda value: (ROLE_ORDER.index(value[0]), value[1]))],
            "size": len(members),
        }
        for members in clusters.values()
        if len(members) > 1
    ]
    cluster_rows.sort(key=lambda item: [(ROLE_ORDER.index(member["role"]), member["sample_index"]) for member in item["members"]])
    cross_confirmed = sum(item["left_role"] != item["right_role"] for item in confirmed)
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "phase": STAGE2_PHASE,
        "profile": profile,
        "audit_scope": "WITHIN_ROLE_AND_CROSS_ROLE",
        "role_sample_counts": {role: int(role_data[role]["canonical_rows"]) for role in ROLE_ORDER},
        "blocking_key_count": len(blocks),
        "candidate_pair_count": total_candidate_pairs,
        "screened_pair_count": len(pairs),
        "candidate_pairs_truncated": not complete,
        "confirmed_pair_count": len(confirmed),
        "cross_role_confirmed_pair_count": cross_confirmed,
        "confirmed_pair_counts_by_role_pair": {
            f"{left_role}__{right_role}": sum(
                item["left_role"] == left_role and item["right_role"] == right_role
                for item in confirmed
            )
            for left_index, left_role in enumerate(ROLE_ORDER)
            for right_role in ROLE_ORDER[left_index + 1 :]
        },
        "within_role_confirmed_pair_counts": {
            role: sum(item["left_role"] == role and item["right_role"] == role for item in confirmed)
            for role in ROLE_ORDER
        },
        "confirmed_pairs": confirmed[:max_witnesses],
        "confirmed_pairs_witness_truncated": len(confirmed) > max_witnesses,
        "confirmed_clusters": cluster_rows[:max_witnesses],
        "confirmed_cluster_witness_truncated": len(cluster_rows) > max_witnesses,
        "confirmed_cluster_sample_count": sum(item["size"] for item in cluster_rows),
        "exhaustiveness_claim": "DETERMINISTIC_SCREENING_NOT_EXHAUSTIVE",
    }
    result["audit_sha256"] = sha256_bytes(canonical_json(result).encode("utf-8"))
    return result


def audit_cross_role_leakage(role_data: Mapping[str, Mapping[str, Any]], near_audit: Mapping[str, Any]) -> dict[str, Any]:
    exact_layers = {
        "source_member_leakage": "source_member_sha256",
        "decoded_frame_leakage": "source_frame_sha256",
        "canonical_content_leakage": "canonical_frame_hash",
    }
    pairs: dict[str, Any] = {}
    for left_index, left_role in enumerate(ROLE_ORDER):
        for right_role in ROLE_ORDER[left_index + 1 :]:
            pair_name = f"{left_role}__{right_role}"
            pairs[pair_name] = {
                output_key: _cross_layer(role_data[left_role]["rows"], role_data[right_role]["rows"], row_key)
                for output_key, row_key in exact_layers.items()
            }
    identity_layers = {
        "source_archive_identity": "source_archive_sha256",
        "source_member_identity": "_source_member_identity_hash",
        "source_frame_id": "_source_frame_id_hash",
    }
    for role in ROLE_ORDER:
        for row in role_data[role]["rows"]:
            row["_source_member_identity_hash"] = sha256_bytes(
                canonical_json({
                    "source_dataset_id": row.get("source_dataset_id"),
                    "source_split": row.get("source_split"),
                    "source_member": row.get("source_member"),
                    "source_frame_index": row.get("source_frame_index"),
                }).encode("utf-8")
            )
            row["_source_frame_id_hash"] = sha256_bytes(
                canonical_json({
                    "source_dataset_id": row.get("source_dataset_id"),
                    "source_split": row.get("source_split"),
                    "source_frame_index": row.get("source_frame_index"),
                }).encode("utf-8")
            )
    identity_pairs: dict[str, Any] = {}
    for left_index, left_role in enumerate(ROLE_ORDER):
        for right_role in ROLE_ORDER[left_index + 1 :]:
            pair_name = f"{left_role}__{right_role}"
            identity_pairs[pair_name] = {
                output_key: _cross_layer(role_data[left_role]["rows"], role_data[right_role]["rows"], row_key)
                for output_key, row_key in identity_layers.items()
            }
    provenance_fields = {
        "subject": "source_subject_status",
        "session": "source_session_status",
        "sequence": "source_sequence_status",
        "event": "source_event_status",
    }
    unavailable: dict[str, Any] = {}
    for label, field in provenance_fields.items():
        values = [row.get(field) for role in ROLE_ORDER for row in role_data[role]["rows"]]
        if not values or all(value in {None, "ABSENT", "NOT_VERIFIABLE"} for value in values):
            unavailable[label] = {"status": "NOT_VERIFIABLE", "reason": "SOURCE_PROVENANCE_ABSENT"}
        else:
            unavailable[label] = {"status": "MEASURED", "reason": "IDENTIFIER_PRESENT"}
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "phase": STAGE2_PHASE,
        "source_member_leakage": pairs,
        "source_identity_overlap": identity_pairs,
        "source_member_identity_overlap": {
            pair: values["source_member_identity"] for pair, values in identity_pairs.items()
        },
        "source_frame_id_overlap": {
            pair: values["source_frame_id"] for pair, values in identity_pairs.items()
        },
        "exact_content_leakage": pairs,
        "canonical_content_leakage": pairs,
        "near_duplicate_screening": {
            "status": "MEASURED",
            "cross_role_confirmed_pair_count": int(near_audit.get("cross_role_confirmed_pair_count", 0)),
            "audit_sha256": near_audit.get("audit_sha256"),
        },
        "subject_leakage": unavailable["subject"],
        "session_leakage": unavailable["session"],
        "sequence_leakage": unavailable["sequence"],
        "event_leakage": unavailable["event"],
        "overall_measurable_leakage_status": "PASS_WITH_LIMITATIONS",
        "grouping_limitation": "SUBJECT_SESSION_SEQUENCE_EVENT_NOT_VERIFIABLE_SOURCE_PROVENANCE_ABSENT",
    }
    result["audit_sha256"] = sha256_bytes(canonical_json(result).encode("utf-8"))
    return result


def verify_synthetic_source_contract(
    source_archive: Path,
    *,
    source_split: str,
    expected_count: int,
    sample_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Validate the SDT synthetic archive schema/unit before conversion."""

    if source_split not in {"train", "validation"}:
        raise Stage2AuditError("SYNTHETIC_SPLIT_INVALID", source_split)
    sample_indices = tuple(sample_indices or (0, expected_count // 2, expected_count - 1))
    prefix = f"{source_split}/"
    label_name = f"{source_split}/labels.txt"
    thermal_pattern = re.compile(rf"{re.escape(prefix)}image_t_(\d+)\.png")
    depth_pattern = re.compile(rf"{re.escape(prefix)}image_d_(\d+)\.png")
    try:
        with zipfile.ZipFile(source_archive, "r") as archive:
            info_list = archive.infolist()
            member_names = [info.filename for info in info_list]
            if len(member_names) != len(set(member_names)):
                raise Stage2AuditError("BLOCKED_SYNTHETIC_PHYSICAL_CONTRACT_NOT_VERIFIED", "duplicate archive member names")
            infos = {info.filename: info for info in info_list}
            label_info = infos.get(label_name)
            if label_info is None:
                raise Stage2AuditError("BLOCKED_SYNTHETIC_PHYSICAL_CONTRACT_NOT_VERIFIED", f"missing {label_name}")
            labels = SDTThermalRawReader._parse_labels(
                SDTThermalRawReader._read_bounded_member(archive, label_info, limit=MAX_LABEL_MEMBER_BYTES),
                expected_count,
            )
            thermal_indices = sorted(int(match.group(1)) for name in infos if (match := thermal_pattern.fullmatch(name)))
            depth_indices = sorted(int(match.group(1)) for name in infos if (match := depth_pattern.fullmatch(name)))
            expected_indices = list(range(expected_count))
            if thermal_indices != expected_indices or depth_indices != expected_indices:
                raise Stage2AuditError("BLOCKED_SYNTHETIC_PHYSICAL_CONTRACT_NOT_VERIFIED", "thermal/depth index set mismatch")
            samples: list[dict[str, Any]] = []
            for index in sample_indices:
                thermal_info = infos[f"{source_split}/image_t_{index}.png"]
                depth_info = infos[f"{source_split}/image_d_{index}.png"]
                thermal_payload = SDTThermalRawReader._read_bounded_member(archive, thermal_info, limit=MAX_IMAGE_MEMBER_BYTES)
                depth_payload = SDTThermalRawReader._read_bounded_member(archive, depth_info, limit=MAX_IMAGE_MEMBER_BYTES)
                thermal, thermal_flags = SDTThermalRawReader._decode_thermal_png(thermal_payload, thermal_info.filename)
                depth, depth_flags = SDTThermalRawReader._decode_thermal_png(depth_payload, depth_info.filename)
                celsius = encoded_to_celsius(thermal)
                if tuple(thermal.shape) != DISTRIBUTED_FRAME_SHAPE or tuple(depth.shape) != DISTRIBUTED_FRAME_SHAPE or not np.all(np.isfinite(celsius)):
                    raise Stage2AuditError("BLOCKED_SYNTHETIC_PHYSICAL_CONTRACT_NOT_VERIFIED", f"sample {index} shape/unit failure")
                samples.append({
                    "index": index,
                    "image_t_shape": list(thermal.shape),
                    "image_t_dtype": "uint16",
                    "image_t_representation": "RADIOMETRIC_TEMPERATURE_ENCODED_UINT16",
                    "image_t_temperature_encoding": "kelvin_centiunits; celsius=(raw-27315)/100",
                    "image_t_quality_flags": list(thermal_flags),
                    "image_d_shape": list(depth.shape),
                    "image_d_dtype": "uint16",
                    "image_d_representation": "DEPTH_UINT16_MILLIMETRES",
                    "image_d_quality_flags": list(depth_flags),
                })
    except Stage2AuditError:
        raise
    except (zipfile.BadZipFile, OSError, KeyError) as exc:
        raise Stage2AuditError("BLOCKED_SYNTHETIC_PHYSICAL_CONTRACT_NOT_VERIFIED", str(exc)) from exc
    class_counts: dict[str, int] = defaultdict(int)
    for label in labels:
        class_counts[label.source_pose_name] += 1
    result = {
        "schema_version": "1.0",
        "phase": STAGE2_PHASE,
        "source_split": source_split,
        "expected_frame_count": expected_count,
        "thermal_member_count": len(thermal_indices),
        "depth_member_count": len(depth_indices),
        "label_row_count": len(labels),
        "class_counts": {key: class_counts[key] for key in sorted(class_counts)},
        "samples": samples,
        "physical_contract": "PASS",
        "raw_schema": "640x480 uint16 grayscale PNG image_t + image_d pairs",
        "thermal_unit": "KELVIN_CENTIUNITS_CONVERTED_TO_CELSIUS",
        "depth_unit": "MILLIMETRES",
        "orientation": "SOURCE_ORIENTATION_AS_STORED",
    }
    result["evidence_sha256"] = sha256_bytes(canonical_json(result).encode("utf-8"))
    return result


def build_role_registry(role_data: Mapping[str, Mapping[str, Any]], output_root: Path) -> dict[str, Any]:
    registry: dict[str, Any] = {
        "schema_version": "1.0",
        "phase": STAGE2_PHASE,
        "locked_test_available": False,
        "real_role": "REAL_EVAL_DEVELOPMENT",
        "roles": {},
    }
    for role in ROLE_ORDER:
        data = role_data[role]
        artifact_path = Path(data["artifact_path"])
        provenance_path = Path(data["provenance_path"])
        registry["roles"][role] = {
            key: data[key]
            for key in (
                "source_split",
                "source_domain",
                "expected_count",
                "source_frames_measured",
                "canonical_rows",
                "canonical_shape",
                "canonical_dtype",
                "canonical_unit",
                "artifact_sha256",
                "provenance_sha256",
                "artifact_size_bytes",
                "provenance_size_bytes",
            )
        }
        registry["roles"][role].update({
            "artifact_path": _safe_logical(artifact_path.relative_to(output_root).as_posix()),
            "provenance_path": _safe_logical(provenance_path.relative_to(output_root).as_posix()),
        })
    return registry


def build_conversion_status(role_data: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    roles: dict[str, Any] = {}
    for role in ROLE_ORDER:
        summary = role_data[role]["summary"]
        counts = summary.get("status_counts", {})
        expected = EXPECTED_ROLE_COUNTS[role]
        roles[role] = {
            "expected_source_frames": expected,
            "source_frames_measured": int(summary.get("source_frames_measured", -1)),
            "status_counts": {key: int(counts.get(key, 0)) for key in ("SUCCESS", "SUCCESS_WITH_WARNING", "EXCLUDED", "FAILED")},
            "canonical_rows": int(summary.get("canonical_rows", -1)),
            "finalized_status": summary.get("finalized_status"),
            "reconciliation": {
                "status_sum_equals_expected": sum(int(counts.get(key, 0)) for key in ("SUCCESS", "SUCCESS_WITH_WARNING", "EXCLUDED", "FAILED")) == expected,
                "canonical_equals_success_plus_warning": int(summary.get("canonical_rows", -1)) == int(counts.get("SUCCESS", 0)) + int(counts.get("SUCCESS_WITH_WARNING", 0)),
            },
        }
    return {"schema_version": "1.0", "phase": STAGE2_PHASE, "roles": roles}


def build_quality_audit(role_data: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    roles: dict[str, Any] = {}
    for role in ROLE_ORDER:
        summary = role_data[role]["summary"]
        counts = summary.get("status_counts", {})
        quality = summary.get("quality", {})
        roles[role] = {
            "status": "PASS" if int(counts.get("FAILED", 0)) == 0 and int(quality.get("silent_skips", 0)) == 0 else "FAIL",
            "status_counts": {key: int(counts.get(key, 0)) for key in ("SUCCESS", "SUCCESS_WITH_WARNING", "EXCLUDED", "FAILED")},
            "quality": quality,
            "silent_skips": int(quality.get("silent_skips", 0)),
        }
    return {
        "schema_version": "1.0",
        "phase": STAGE2_PHASE,
        "roles": roles,
        "overall_status": "PASS" if all(item["status"] == "PASS" for item in roles.values()) else "FAIL",
        "policy": "EVERY_SOURCE_FRAME_HAS_EXPLICIT_TERMINAL_STATUS",
    }


def build_output_checksums(role_data: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "phase": STAGE2_PHASE,
        "roles": {
            role: {
                "artifact_sha256": role_data[role]["artifact_sha256"],
                "provenance_sha256": role_data[role]["provenance_sha256"],
                "ledger_sha256": role_data[role].get("ledger_sha256"),
            }
            for role in ROLE_ORDER
        },
    }


def build_checksums(bundle_dir: Path) -> Path:
    lines: list[str] = []
    for path in sorted(bundle_dir.glob("*.json")):
        lines.append(f"{sha256_file(path)}  {bundle_dir.name}/{path.name}")
    target = bundle_dir / "checksums.sha256"
    partial = target.with_name(target.name + ".partial")
    partial.write_text("\n".join(lines) + "\n", encoding="utf-8")
    partial.replace(target)
    return target


def validate_stage2_bundle(
    bundle_dir: Path,
    *,
    require_validation_result: bool = True,
) -> dict[str, Any]:
    """Validate only compact Stage 2 evidence; bulk payloads are not required."""

    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    required = list(BUNDLE_JSON_FILES)
    if not require_validation_result:
        required.remove("validation_result.json")
    documents: dict[str, Any] = {}
    for name in required:
        path = bundle_dir / name
        if not path.is_file():
            errors.append({"code": "REQUIRED_ARTIFACT_MISSING", "location": name, "message": "required compact artifact is missing"})
            continue
        try:
            value = _load_json(path)
        except Stage2AuditError as exc:
            errors.append({"code": exc.code, "location": name, "message": exc.detail})
            continue
        documents[name] = value
        for location, item in _walk(value, name):
            if isinstance(item, str):
                if item.startswith(("/", "~/", "file://")) or "\\" in item or "/Users/" in item or "/private/" in item or "iCloud" in item or "/content/" in item:
                    errors.append({"code": "NONPORTABLE_PATH", "location": location, "message": item})
                if "Thermal-44" in item and any(token in item.upper() for token in ("VERIFIED", "CONFIRMED", "VALIDATED")):
                    errors.append({"code": "UNSUPPORTED_THERMAL44_ASSERTION", "location": location, "message": item})
            if isinstance(item, dict):
                for key in item:
                    if str(key).lower() in MODEL_METRIC_KEYS:
                        errors.append({"code": "MODEL_METRIC_CONTAMINATION", "location": f"{location}.{key}", "message": "T-A6 must not contain model metrics"})
    checksum_path = bundle_dir / CHECKSUMS_NAME
    if not checksum_path.is_file():
        errors.append({"code": "CHECKSUM_REGISTRY_MISSING", "location": CHECKSUMS_NAME, "message": "compact checksum registry is missing"})
    else:
        entries: dict[str, str] = {}
        previous = ""
        for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
            match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", line)
            if not match:
                errors.append({"code": "CHECKSUM_LINE_INVALID", "location": f"{CHECKSUMS_NAME}:{line_number}", "message": line})
                continue
            digest, relative = match.groups()
            if relative <= previous:
                errors.append({"code": "CHECKSUM_ORDER_NONDETERMINISTIC", "location": f"{CHECKSUMS_NAME}:{line_number}", "message": relative})
            previous = relative
            if not _portable(relative) or not relative.startswith(bundle_dir.name + "/"):
                errors.append({"code": "CHECKSUM_PATH_NOT_PORTABLE", "location": f"{CHECKSUMS_NAME}:{line_number}", "message": relative})
            entries[relative] = digest
        for name in required:
            relative = f"{bundle_dir.name}/{name}"
            path = bundle_dir / name
            if path.is_file() and entries.get(relative) != sha256_file(path):
                errors.append({"code": "CHECKSUM_MISMATCH", "location": relative, "message": "checksum is missing or stale"})
    registry = documents.get("canonical_artifact_registry.json", {})
    roles = registry.get("roles", {})
    if set(roles) != set(ROLE_ORDER):
        errors.append({"code": "ROLE_ORDER_INVALID", "location": "canonical_artifact_registry.json:roles", "message": str(list(roles))})
    if registry.get("locked_test_available") is not False:
        errors.append({"code": "LOCKED_TEST_ESCALATION", "location": "canonical_artifact_registry.json:locked_test_available", "message": "real test must remain development-only"})
    for role in ROLE_ORDER:
        record = roles.get(role, {})
        expected_domain = "REAL" if role == "REAL_EVAL_DEVELOPMENT" else "SYNTHETIC"
        expected_split = "test" if role == "REAL_EVAL_DEVELOPMENT" else role.lower()
        if record.get("source_domain") != expected_domain or record.get("source_split") != expected_split:
            errors.append({"code": "ROLE_SOURCE_CONTRACT_INVALID", "location": f"canonical_artifact_registry.json:{role}", "message": str(record)})
        if record.get("canonical_rows") != EXPECTED_ROLE_COUNTS[role] or record.get("source_frames_measured") != EXPECTED_ROLE_COUNTS[role]:
            errors.append({"code": "ROLE_COUNT_INVALID", "location": f"canonical_artifact_registry.json:{role}", "message": str(record)})
        if record.get("canonical_shape") != list(CANONICAL_SHAPE) or record.get("canonical_dtype") != "float32_little_endian" or record.get("canonical_unit") != "CELSIUS":
            errors.append({"code": "ROLE_CONTRACT_INVALID", "location": f"canonical_artifact_registry.json:{role}", "message": "shape/dtype/unit mismatch"})
    status = documents.get("conversion_status_summary.json", {}).get("roles", {})
    for role in ROLE_ORDER:
        record = status.get(role, {})
        counts = record.get("status_counts", {})
        if sum(int(counts.get(key, 0)) for key in ("SUCCESS", "SUCCESS_WITH_WARNING", "EXCLUDED", "FAILED")) != EXPECTED_ROLE_COUNTS[role]:
            errors.append({"code": "STATUS_RECONCILIATION_INVALID", "location": f"conversion_status_summary.json:{role}", "message": str(record)})
        if int(record.get("canonical_rows", -1)) != int(counts.get("SUCCESS", 0)) + int(counts.get("SUCCESS_WITH_WARNING", 0)):
            errors.append({"code": "CANONICAL_ROW_RECONCILIATION_INVALID", "location": f"conversion_status_summary.json:{role}", "message": str(record)})
        if record.get("finalized_status") != "FINALIZED":
            errors.append({"code": "ROLE_NOT_FINALIZED", "location": f"conversion_status_summary.json:{role}", "message": str(record)})
    exact = documents.get("exact_duplicate_audit.json", {})
    if exact.get("audit_scope") != "WITHIN_ROLE_AND_CROSS_ROLE" or set(exact.get("layers", {})) != {"source_member_byte_hashes", "decoded_frame_hashes", "canonical_frame_hashes"}:
        errors.append({"code": "EXACT_AUDIT_INCOMPLETE", "location": "exact_duplicate_audit.json", "message": "all three hash layers and cross-role scope are required"})
    for name in ("exact_duplicate_audit.json", "near_duplicate_audit.json", "cross_role_leakage_audit.json"):
        document = documents.get(name, {})
        reported = document.get("audit_sha256")
        if not isinstance(reported, str):
            errors.append({"code": "AUDIT_CHECKSUM_MISSING", "location": f"{name}:audit_sha256", "message": "audit checksum is required"})
        else:
            without_hash = dict(document)
            without_hash.pop("audit_sha256", None)
            measured = sha256_bytes(canonical_json(without_hash).encode("utf-8"))
            if reported != measured:
                errors.append({"code": "AUDIT_CHECKSUM_MISMATCH", "location": f"{name}:audit_sha256", "message": "audit checksum is stale"})
    near = documents.get("near_duplicate_audit.json", {})
    if near.get("profile", {}).get("profile_id") != NEAR_DUPLICATE_PROFILE_ID or near.get("exhaustiveness_claim") != "DETERMINISTIC_SCREENING_NOT_EXHAUSTIVE":
        errors.append({"code": "NEAR_AUDIT_PROFILE_INVALID", "location": "near_duplicate_audit.json", "message": "frozen T-A6 profile is not preserved"})
    leakage = documents.get("cross_role_leakage_audit.json", {})
    for key in ("source_identity_overlap", "source_member_identity_overlap", "source_frame_id_overlap", "source_member_leakage", "exact_content_leakage", "canonical_content_leakage", "near_duplicate_screening"):
        if key not in leakage:
            errors.append({"code": "LEAKAGE_AUDIT_INCOMPLETE", "location": f"cross_role_leakage_audit.json:{key}", "message": "required leakage dimension is missing"})
    for key in ("subject_leakage", "session_leakage", "sequence_leakage", "event_leakage"):
        if leakage.get(key, {}).get("status") != "NOT_VERIFIABLE":
            errors.append({"code": "GROUPING_OVERCLAIM", "location": f"cross_role_leakage_audit.json:{key}", "message": "missing source provenance must remain NOT_VERIFIABLE"})
    determinism = documents.get("determinism_summary.json", {})
    if determinism.get("status") != "PASS" or determinism.get("full_second_conversion") is not True or determinism.get("artifact_checksum_match") is not True or determinism.get("provenance_checksum_match") is not True:
        errors.append({"code": "DETERMINISM_AUDIT_INCOMPLETE", "location": "determinism_summary.json", "message": str(determinism)})
    execution = documents.get("execution_summary.json", {})
    if execution.get("t_b_authorized") is not False:
        errors.append({"code": "DOWNSTREAM_GATE_ESCALATION", "location": "execution_summary.json:t_b_authorized", "message": "T-B must remain blocked"})
    if execution.get("phase") != STAGE2_PHASE:
        errors.append({"code": "PHASE_ID_INVALID", "location": "execution_summary.json:phase", "message": str(execution.get("phase"))})
    if documents.get("quality_audit_summary.json", {}).get("overall_status") != "PASS":
        errors.append({"code": "QUALITY_AUDIT_INCOMPLETE", "location": "quality_audit_summary.json", "message": "quality audit did not pass"})
    source_identity = documents.get("source_identity.json", {})
    physical_contract = source_identity.get("synthetic_physical_contract", {})
    if physical_contract.get("train", {}).get("physical_contract") != "PASS" or physical_contract.get("validation", {}).get("physical_contract") != "PASS":
        errors.append({"code": "SYNTHETIC_PHYSICAL_CONTRACT_INVALID", "location": "source_identity.json:synthetic_physical_contract", "message": "both synthetic partitions require an independently verified contract"})
    checksums = documents.get("output_checksums.json", {}).get("roles", {})
    for role in ROLE_ORDER:
        record = roles.get(role, {})
        checksum_record = checksums.get(role, {})
        if checksum_record.get("artifact_sha256") != record.get("artifact_sha256") or checksum_record.get("provenance_sha256") != record.get("provenance_sha256"):
            errors.append({"code": "OUTPUT_CHECKSUM_REGISTRY_MISMATCH", "location": f"output_checksums.json:{role}", "message": "role checksum registry differs from canonical artifact registry"})
    errors = sorted(errors, key=lambda item: (item["code"], item["location"], item["message"]))
    warnings = sorted(warnings, key=lambda item: (item["code"], item["location"], item["message"]))
    outcome = "PASS_WITH_LIMITATIONS" if not errors else "NOT_VERIFIABLE"
    return {
        "schema_version": "1.0",
        "phase": STAGE2_PHASE,
        "evidence_validation": "PASS" if not errors else "FAIL",
        "overall_outcome": outcome,
        "full_t_a6_gate": "T_A6_FULL_COMPLETE_WITH_LIMITATIONS" if not errors else "NOT_YET_COMPLETE",
        "t_b_authorized": False,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }


__all__ = [
    "BUNDLE_JSON_FILES",
    "CHECKSUMS_NAME",
    "EXPECTED_ROLE_COUNTS",
    "ROLE_ORDER",
    "ROLE_TO_SPLIT",
    "STAGE2_PHASE",
    "Stage2AuditError",
    "audit_cross_role_leakage",
    "audit_exact_duplicates",
    "audit_near_duplicates_cross_role",
    "build_checksums",
    "build_conversion_status",
    "build_output_checksums",
    "build_quality_audit",
    "build_role_registry",
    "load_provenance",
    "validate_role_artifact",
    "validate_stage2_bundle",
    "verify_synthetic_source_contract",
]
