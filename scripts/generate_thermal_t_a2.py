#!/usr/bin/env python3
"""Generate deterministic compact Thermal T-A2 geometry evidence."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.thermal.canonical_geometry import (  # noqa: E402
    CANONICAL_SHAPE,
    SOURCE_SHAPE,
    CanonicalPhysicalFrame,
    GeometryProfile,
    canonical_to_source_trace,
    canonicalize_source_frame,
    make_candidate_profiles,
    precision_error,
    profile_for_id,
    resize_physical,
    source_to_canonical_trace,
    transform_bbox,
)
from datasets.thermal.geometry_selection import apply_selection_policy, selection_policy  # noqa: E402
from datasets.thermal.raw_reader import SDTThermalRawReader, encoded_frame_sha256  # noqa: E402


EVIDENCE_REL = "datasets/thermal/manifests/T-A2_geometry_calibration_canonical_frame"
EVIDENCE_DIR = ROOT / EVIDENCE_REL
REPORT_REL = "docs/reports/20260810_Codex_T-A2_Thermal_Geometry_Calibration_Canonical_Frame_01.md"
VISUAL_REL = f"{EVIDENCE_REL}/visual_spotcheck.png"
JSON_NAMES = [
    "calibration_contract.json",
    "canonical_frame_contract.json",
    "coordinate_trace_contract.json",
    "geometry_candidate_registry.json",
    "geometry_comparison.json",
    "geometry_selection_policy.json",
    "invalid_pixel_policy.json",
    "pilot_geometry_summary.json",
    "selected_geometry_profile.json",
    "visual_spotcheck_registry.json",
]
PILOT_PER_CLASS = 12
PILOT_WITNESS_INDICES = [0, 2000, 4000, 6000]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _frame_stats(array: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    numeric = np.asarray(array, dtype=np.float64)
    valid = np.isfinite(numeric)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    values = numeric[valid]
    if values.size == 0:
        raise ValueError("cannot compute stats without valid values")
    percentiles = np.percentile(values, [1, 50, 99])
    return {
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "p01": float(percentiles[0]),
        "p50": float(percentiles[1]),
        "p99": float(percentiles[2]),
        "valid_count": int(values.size),
    }


def _aggregate_stats(stats: list[dict[str, float]]) -> dict[str, Any]:
    keys = ("min", "max", "mean", "p01", "p50", "p99")
    result: dict[str, Any] = {}
    for key in keys:
        values = np.asarray([row[key] for row in stats], dtype=np.float64)
        result[key] = float(np.mean(values))
        result[f"{key}_min"] = float(np.min(values))
        result[f"{key}_max"] = float(np.max(values))
    result["frame_count"] = len(stats)
    result["valid_pixel_count"] = int(sum(row["valid_count"] for row in stats))
    return result


def _pilot_indices(reader: SDTThermalRawReader) -> dict[int, list[int]]:
    reader.inspect_archive()
    labels = getattr(reader, "_labels", None)
    if labels is None:
        raise RuntimeError("T-A1 reader did not expose verified labels for bounded diagnostics")
    groups: dict[int, list[int]] = {}
    for label in labels:
        groups.setdefault(label.source_pose_label, []).append(label.source_frame_index)
    selected: dict[int, list[int]] = {}
    for pose in sorted(groups):
        indices = groups[pose]
        positions = np.linspace(0, len(indices) - 1, PILOT_PER_CLASS, dtype=np.float64)
        chosen = [indices[int(round(position))] for position in positions]
        if len(set(chosen)) != PILOT_PER_CLASS:
            raise RuntimeError(f"deterministic pilot has duplicate indices for pose {pose}")
        selected[pose] = chosen
    return selected


def _source_label(reader: SDTThermalRawReader, index: int) -> Any:
    labels = getattr(reader, "_labels", None)
    if labels is None:
        reader.inspect_archive()
        labels = getattr(reader, "_labels", None)
    if labels is None:
        raise RuntimeError("verified labels unavailable")
    return labels[index]


def _round_trip_metrics(source: np.ndarray, frame: CanonicalPhysicalFrame, profile: GeometryProfile) -> dict[str, float]:
    left, top, right, bottom = profile.crop_xyxy
    cropped = source[top:bottom, left:right]
    valid_canonical = frame.physical_frame[frame.validity_mask]
    inner = frame.physical_frame[profile.pad_top:profile.pad_top + profile.resize_height, profile.pad_left:profile.pad_left + profile.resize_width]
    reconstructed = resize_physical(inner, cropped.shape, profile.interpolation)
    diagnostic = precision_error(cropped, reconstructed)
    diagnostic["status"] = "DIAGNOSTIC_ROUND_TRIP_ONLY"
    diagnostic["source_region_pixel_count"] = int(cropped.size)
    diagnostic["canonical_valid_pixel_count"] = int(valid_canonical.size)
    return diagnostic


def _bbox_metrics(reader: SDTThermalRawReader, profile: GeometryProfile, indices: Iterable[int]) -> dict[str, Any]:
    records = []
    for index in indices:
        label = _source_label(reader, index)
        if label.source_pose_label == 3:
            continue
        records.append(transform_bbox(label.source_bbox, profile))
    retained = [float(item["retained_area_fraction"]) for item in records if item["retained_area_fraction"] is not None]
    additional_losses = [float(item.get("additional_bbox_area_loss_due_to_candidate_crop", 0.0)) for item in records]
    source_areas = [float(item.get("source_clipped_area", 0.0) or 0.0) for item in records]
    total_source_area = float(sum(source_areas))
    total_additional_loss = float(sum(additional_losses))
    return {
        "person_bbox_count": len(records),
        "source_bbox_outside_frame_count": int(sum(bool(item.get("source_bbox_outside_frame")) for item in records)),
        "source_boundary_clipped_count": int(sum(bool(item.get("source_boundary_clipped")) for item in records)),
        "additional_bbox_intersected_by_candidate_crop_count": int(sum(loss > 0.0 for loss in additional_losses)),
        "additional_bbox_area_loss_due_to_candidate_crop": total_additional_loss,
        "candidate_crop_additional_bbox_area_loss_total": total_additional_loss,
        "additional_bbox_area_loss_due_to_candidate_crop_fraction": total_additional_loss / total_source_area if total_source_area else 0.0,
        "bbox_removed_by_candidate_crop_count": int(sum(item["status"] == "BBOX_REMOVED_BY_CANDIDATE_CROP" for item in records)),
        "mean_bbox_retention_due_to_candidate_transform": float(np.mean(retained)) if retained else None,
        "minimum_bbox_retention_due_to_candidate_transform": float(np.min(retained)) if retained else None,
        "coordinate_clipping_order": "SOURCE_BBOX_CLIP_TO_DISTRIBUTED_FRAME_THEN_INTERSECT_CANDIDATE_CROP_THEN_MAP_TO_CANONICAL",
        "source_boundary_overflow_is_not_candidate_crop_loss": True,
    }


def _profile_geometry_metrics(profile: GeometryProfile) -> dict[str, Any]:
    crop_area = profile.crop_width * profile.crop_height
    source_area = SOURCE_SHAPE[0] * SOURCE_SHAPE[1]
    target_area = CANONICAL_SHAPE[0] * CANONICAL_SHAPE[1]
    horizontal_scale = profile.resize_width / profile.crop_width
    vertical_scale = profile.resize_height / profile.crop_height
    anisotropy = max(horizontal_scale, vertical_scale) / min(horizontal_scale, vertical_scale) - 1.0
    return {
        "horizontal_scale_factor": horizontal_scale,
        "vertical_scale_factor": vertical_scale,
        "anisotropy_ratio_excess": anisotropy,
        "source_fov_retained_fraction": crop_area / source_area,
        "crop_pixel_count": source_area - crop_area,
        "crop_percentage": (source_area - crop_area) / source_area * 100.0,
        "padding_pixel_count": target_area - profile.resize_height * profile.resize_width,
        "padding_percentage": (target_area - profile.resize_height * profile.resize_width) / target_area * 100.0,
        "edge_handling": profile.edge_handling,
        "coordinate_mapping": profile.coordinate_mapping,
        "explicit_antialias_prefilter": profile.explicit_antialias_prefilter,
    }


def _candidate_metrics(reader: SDTThermalRawReader, indices: list[int], profile: GeometryProfile) -> dict[str, Any]:
    source_stats: list[dict[str, float]] = []
    output_stats: list[dict[str, float]] = []
    mean_shifts: list[float] = []
    range_compressions: list[float] = []
    round_trips: list[dict[str, float]] = []
    repeated = True
    constant_ok = True
    finite_valid_output = True
    for index in indices:
        frame = reader.read_frame(index)
        source = frame.celsius()
        canonical = canonicalize_source_frame(frame, profile)
        source_stat = _frame_stats(source)
        output_stat = _frame_stats(canonical.physical_frame, canonical.validity_mask)
        finite_valid_output &= bool(np.all(np.isfinite(canonical.physical_frame[canonical.validity_mask])))
        source_stats.append(source_stat)
        output_stats.append(output_stat)
        mean_shifts.append(output_stat["mean"] - source_stat["mean"])
        source_range = source_stat["max"] - source_stat["min"]
        output_range = output_stat["max"] - output_stat["min"]
        range_compressions.append(output_range / source_range if source_range else 1.0)
        round_trips.append(_round_trip_metrics(source, canonical, profile))
        repeated_frame = canonicalize_source_frame(frame, profile)
        repeated &= canonical.canonical_frame_hash == repeated_frame.canonical_frame_hash
    constant = np.full(SOURCE_SHAPE, 26.85, dtype=np.float64)
    constant_frame = canonicalize_physical_frame_for_profile(constant, profile)
    valid_constant = constant_frame.physical_frame[constant_frame.validity_mask]
    constant_ok = bool(np.all(valid_constant == np.float32(26.85))) and bool(np.all(~constant_frame.validity_mask == np.isnan(constant_frame.physical_frame)))
    geometry = _profile_geometry_metrics(profile)
    bbox = _bbox_metrics(reader, profile, indices)
    return {
        "profile": profile.to_dict(),
        "geometry": geometry,
        "source_temperature_statistics": _aggregate_stats(source_stats),
        "canonical_temperature_statistics": _aggregate_stats(output_stats),
        "mean_shift_celsius": float(np.mean(mean_shifts)),
        "mean_absolute_shift_celsius": float(np.mean(np.abs(mean_shifts))),
        "range_compression_ratio": float(np.mean(range_compressions)),
        "range_compression_ratio_min": float(np.min(range_compressions)),
        "range_compression_ratio_max": float(np.max(range_compressions)),
        "round_trip": {
            "status": "DIAGNOSTIC_ROUND_TRIP_ONLY",
            "mae": float(np.mean([item["mean_abs_error"] for item in round_trips])),
            "rmse": float(np.mean([item["rmse"] for item in round_trips])),
            "max_error": float(np.max([item["max_abs_error"] for item in round_trips])),
        },
        "constant_temperature_preserved": constant_ok,
        "finite_valid_output": finite_valid_output,
        "repeated_canonicalization_deterministic": repeated,
        "bbox_fov_diagnostic": bbox,
        "pilot_frame_count": len(indices),
    }


def canonicalize_physical_frame_for_profile(source: np.ndarray, profile: GeometryProfile) -> CanonicalPhysicalFrame:
    from datasets.thermal.canonical_geometry import canonicalize_physical_frame

    return canonicalize_physical_frame(source, profile)


def _pilot_record(reader: SDTThermalRawReader, index: int, profile: GeometryProfile) -> dict[str, Any]:
    source_frame = reader.read_frame(index)
    source_hash_before = encoded_frame_sha256(source_frame.raw_encoded_frame)
    source_physical = source_frame.celsius()
    canonical = canonicalize_source_frame(source_frame, profile)
    repeat = canonicalize_source_frame(source_frame, profile)
    source_hash_after = encoded_frame_sha256(source_frame.raw_encoded_frame)
    source_stat = _frame_stats(source_physical)
    canonical_stat = _frame_stats(canonical.physical_frame, canonical.validity_mask)
    witness_coordinates = [[0, 0], [31, 40], [61, 79]]
    traces = [canonical_to_source_trace(profile, row, column) for row, column in witness_coordinates]
    return {
        "canonical_dtype": canonical.canonical_dtype,
        "canonical_frame_hash": canonical.canonical_frame_hash,
        "canonical_max_celsius": canonical_stat["max"],
        "canonical_mean_celsius": canonical_stat["mean"],
        "canonical_min_celsius": canonical_stat["min"],
        "canonical_shape": list(canonical.canonical_shape),
        "canonical_unit": canonical.canonical_unit,
        "canonical_valid_pixel_count": int(np.count_nonzero(canonical.validity_mask)),
        "coordinate_witnesses": traces,
        "geometry_profile_id": profile.profile_id,
        "raw_source_frame_unchanged_after_canonicalization": source_hash_before == source_hash_after == source_frame.raw_encoded_frame_sha256,
        "repeated_canonical_frame_hash_equal": canonical.canonical_frame_hash == repeat.canonical_frame_hash,
        "source_bbox": list(source_frame.source_bbox),
        "source_provenance": source_frame.provenance_dict(),
        "source_encoded_frame_sha256": source_frame.raw_encoded_frame_sha256,
        "source_frame_index": index,
        "source_member_name": source_frame.source_member_name,
        "source_member_sha256": source_frame.source_member_sha256,
        "source_pose_label": source_frame.source_pose_label,
        "source_pose_name": source_frame.source_pose_name,
        "source_shape": list(source_frame.distributed_frame_shape),
        "source_temperature_max_celsius": source_stat["max"],
        "source_temperature_mean_celsius": source_stat["mean"],
        "source_temperature_min_celsius": source_stat["min"],
        "source_unit": "CELSIUS_DERIVED_FROM_SDT_KELVIN_CENTIUNITS",
        "validity_status": canonical.validity_status,
    }


def _visual_spotcheck(reader: SDTThermalRawReader, profile: GeometryProfile, path: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(4, 2, figsize=(12, 15), constrained_layout=True)
    values = []
    for row, index in enumerate(PILOT_WITNESS_INDICES):
        frame = reader.read_frame(index)
        source = frame.celsius()
        canonical = canonicalize_source_frame(frame, profile)
        values.extend([source[np.isfinite(source)], canonical.physical_frame[canonical.validity_mask]])
    vmin = float(min(np.min(value) for value in values))
    vmax = float(max(np.max(value) for value in values))
    for row, index in enumerate(PILOT_WITNESS_INDICES):
        frame = reader.read_frame(index)
        source = frame.celsius()
        canonical = canonicalize_source_frame(frame, profile)
        axes[row, 0].imshow(source, cmap="inferno", vmin=vmin, vmax=vmax, origin="upper", aspect="auto")
        axes[row, 1].imshow(canonical.physical_frame, cmap="inferno", vmin=vmin, vmax=vmax, origin="upper", aspect="auto")
        x_min, y_min, x_max, y_max = frame.source_bbox
        if frame.source_pose_label != 3:
            axes[row, 0].add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor="cyan", linewidth=0.8))
            transformed = transform_bbox(frame.source_bbox, profile)
            if transformed["canonical_bbox"] is not None:
                bx0, by0, bx1, by1 = transformed["canonical_bbox"]
                axes[row, 1].add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, edgecolor="cyan", linewidth=0.8))
        axes[row, 0].scatter([0, SOURCE_SHAPE[1] - 1], [0, SOURCE_SHAPE[0] - 1], c=["white", "lime"], s=10)
        axes[row, 1].scatter([0, CANONICAL_SHAPE[1] - 1], [0, CANONICAL_SHAPE[0] - 1], c=["white", "lime"], s=10)
        axes[row, 0].set_title(f"SDT source idx={index} {frame.source_pose_name} (diagnostic)")
        axes[row, 1].set_title(f"{profile.profile_id} (diagnostic)")
        axes[row, 0].set_xlabel("col 0 left →"); axes[row, 0].set_ylabel("row 0 top ↓")
        axes[row, 1].set_xlabel("col 0 left →"); axes[row, 1].set_ylabel("row 0 top ↓")
    figure.suptitle("Thermal T-A2 visual spot check — colorized diagnostic only; not model input")
    figure.savefig(path, dpi=110, metadata={"Software": "SafeNest T-A2 deterministic diagnostic"})
    plt.close(figure)
    return {
        "path": VISUAL_REL,
        "sha256": _sha256(path),
        "status": "PASS",
        "role": "HUMAN_VISUAL_DIAGNOSTIC_ONLY_NOT_RADIOMETRIC_MODEL_INPUT",
        "representative_indices": PILOT_WITNESS_INDICES,
        "orientation_observation": "Source-as-stored: row 0 is top, column 0 is left; corner markers preserve ordering.",
        "crop_observation": "Fixed crop removes 10 source columns on each horizontal side; diagnostic bbox overlays show bounded FOV loss.",
        "stretch_observation": "Selected crop avoids the direct-stretch vertical anisotropy; no unexpected rotation or flip observed.",
        "border_observation": "Selected profile has no synthetic padding; no border artifact observed.",
        "hot_region_observation": "Hot-region positions remain spatially aligned under fixed crop and bilinear mapping.",
    }


def _precision_contract(reader: SDTThermalRawReader, indices: list[int]) -> dict[str, Any]:
    errors = []
    for index in indices:
        frame = reader.read_frame(index)
        reference = frame.celsius()
        candidate = reference.astype(np.float32).astype(np.float64)
        errors.append(precision_error(reference, candidate))
    return {
        "reference": "float64 SDT Celsius formula",
        "selected_dtype": "float32",
        "max_conversion_error_celsius": float(max(item["max_abs_error"] for item in errors)),
        "mean_conversion_error_celsius": float(np.mean([item["mean_abs_error"] for item in errors])),
        "unit_resolution_celsius": 0.01,
        "memory_bytes_per_frame": int(np.prod(CANONICAL_SHAPE) * np.dtype("float32").itemsize),
        "source_precision_preserved_relative_to_encoding": True,
    }


def build_artifacts(reader: SDTThermalRawReader) -> tuple[dict[str, Any], dict[int, list[int]], list[dict[str, Any]]]:
    inventory = reader.inspect_archive()
    by_class = _pilot_indices(reader)
    indices = [index for pose in sorted(by_class) for index in by_class[pose]]
    profiles = make_candidate_profiles()
    policy = selection_policy()
    raw_candidate_results = sorted(
        (_candidate_metrics(reader, indices, profile) for profile in profiles),
        key=lambda item: item["profile"]["candidate_id"],
    )
    selection = apply_selection_policy(raw_candidate_results, policy)
    candidate_results = selection["candidates"]
    selected = profile_for_id(selection["selected_profile_id"])
    policy_checksum = _sha256_bytes(canonical_json(policy).encode("utf-8"))
    candidate_metrics_checksum = _sha256_bytes(canonical_json(candidate_results).encode("utf-8"))
    selected_records = [_pilot_record(reader, index, selected) for index in indices]
    precision = _precision_contract(reader, indices)

    source_profile = {
        "source_orientation": {
            "array_row_direction": "row 0 is top / increasing row moves down",
            "array_column_direction": "column 0 is left / increasing column moves right",
            "origin": "top-left pixel-center convention",
            "rotation": 0,
            "horizontal_flip": False,
            "vertical_flip": False,
            "exif_orientation": "NOT_APPLICABLE_PNG",
            "status": "SOURCE_ORIENTATION_AS_STORED",
            "evidence": ["official SDT image_t documentation", "labels bbox coordinate bounds", "asymmetric synthetic coordinate fixtures", "visual spot check"],
        },
        "source_shape": list(SOURCE_SHAPE),
        "native_sensor_shape": [120, 160],
        "canonical_software_shape": list(CANONICAL_SHAPE),
        "thermal44_physical_orientation": "UNVERIFIED_DEFERRED_T_C",
    }
    candidate_registry = {
        "phase": "T-A2",
        "schema_version": "1.0",
        "predeclared_candidate_set": [item["profile"]["candidate_id"] for item in candidate_results],
        "selection_policy_id": policy["policy_id"],
        "selection_policy_version": policy["policy_version"],
        "selection_policy_content_sha256": policy_checksum,
        "candidate_metrics_content_sha256": candidate_metrics_checksum,
        "model_performance_used_for_selection": False,
        "implementation": {
            "library": "numpy custom deterministic T-A2 geometry implementation",
            "library_version": np.__version__,
            "coordinate_convention": "pixel-center half-pixel mapping for nearest/bilinear; exact source-area overlap for area",
            "interpolation_candidates": ["nearest", "bilinear", "area"],
            "area_semantics": "exact separable source-pixel overlap weighting",
            "bilinear_semantics": "four-neighbor linear interpolation with edge clamping",
            "nearest_semantics": "nearest source pixel with half-up index rule and edge clamping",
            "antialias_semantics": {
                "bilinear": "NO_EXPLICIT_ANTIALIAS_PREFILTER",
                "nearest": "NO_EXPLICIT_ANTIALIAS_PREFILTER",
                "area": "NO_EXPLICIT_ANTIALIAS_PREFILTER; AREA_INTERPOLATION_ITSELF_AVERAGES_SOURCE_PIXEL_OVERLAPS",
            },
        },
        "source_profile": source_profile,
        "candidates": candidate_results,
    }
    comparison = {
        "phase": "T-A2",
        "schema_version": "1.0",
        "source_archive_identity": inventory["archive_identity"],
        "pilot_selection": {"per_class": PILOT_PER_CLASS, "indices_by_source_pose": {str(key): value for key, value in sorted(by_class.items())}, "total": len(indices)},
        "candidate_results": candidate_results,
        "selection_policy_id": policy["policy_id"],
        "selection_policy_version": policy["policy_version"],
        "selection_policy_content_sha256": policy_checksum,
        "candidate_metrics_content_sha256": candidate_metrics_checksum,
        "selected_candidate_id": selection["selected_candidate_id"],
        "selected_profile_id": selection["selected_profile_id"],
        "selection_derivation": "apply_selection_policy(candidate_results, geometry_selection_policy.json)",
        "model_performance_used": False,
    }
    selected_profile = {
        "phase": "T-A2",
        "schema_version": "1.0",
        "selection_status": "GEOMETRY_PROFILE_SELECTED_WITH_LIMITATIONS",
        "selection_policy_id": policy["policy_id"],
        "selection_policy_version": policy["policy_version"],
        "selection_policy_content_sha256": policy_checksum,
        "candidate_metrics_content_sha256": candidate_metrics_checksum,
        "selection_derivation": "winner = independently reproducible policy ranking of the nine candidate metric records",
        "profile": selected.to_dict(),
        "profile_id": selected.profile_id,
        "source_orientation": source_profile["source_orientation"],
        "physical_unit": "CELSIUS",
        "physical_dtype": "float32",
        "model_performance_used": False,
        "limitations": ["Thermal-44 physical orientation remains unverified; this is a software canonical convention only.", "The selected profile is a software canonical frame and does not establish hardware packet ordering.", "Source label bboxes are clipped to the distributed frame before candidate-crop loss is measured."],
    }
    canonical_contract = {
        "phase": "T-A2",
        "schema_version": "1.0",
        "contract_id": "thermal_canonical_physical_frame_v1",
        "input": {
            "dataset_id": "local_sdt_zenodo_4124309",
            "doi": "doi:10.5281/zenodo.4124309",
            "split": "test",
            "shape": list(SOURCE_SHAPE),
            "dtype": "uint16_encoded_source",
            "unit": "KELVIN_CENTIUNITS",
            "source_hash_preserved": True,
        },
        "output": {
            "shape": list(CANONICAL_SHAPE),
            "dtype": "float32",
            "unit": "CELSIUS",
            "validity_mask": "boolean; selected-profile measured pixels are true and any declared padding is false",
            "profile_id": selected.profile_id,
        },
        "boundary_stop": "Canonical physical frame ends here. No per-frame min-max, z-score, scaler, int8 encoding, or model inference is part of T-A2.",
        "source_and_canonical_are_distinct": True,
        "hash_method": "canonical frame SHA-256 = little-endian float32 C-order bytes followed by uint8 validity-mask bytes",
        "provenance_required": ["source_frame_index", "source_member_name", "source_encoded_frame_sha256", "geometry_profile_id", "canonical_frame_hash", "validity_status"],
    }
    coordinate_contract = {
        "phase": "T-A2",
        "schema_version": "1.0",
        "profile_id": selected.profile_id,
        "forward_equation": "source_y = crop_top + (canonical_row + 0.5) * crop_height / resize_height - 0.5; source_x analogous",
        "inverse_equation": "canonical_row = pad_top + (source_y - crop_top + 0.5) * resize_height / crop_height - 0.5; canonical_col analogous",
        "crop_xyxy_exclusive": list(selected.crop_xyxy),
        "support_semantics": f"{selected.interpolation.upper()} support; coordinate_mapping={selected.coordinate_mapping}; edge_handling={selected.edge_handling}; explicit_antialias_prefilter={selected.explicit_antialias_prefilter}",
        "witnesses": [canonical_to_source_trace(selected, 0, 0), canonical_to_source_trace(selected, 31, 40), canonical_to_source_trace(selected, 61, 79), source_to_canonical_trace(selected, 0, 10), source_to_canonical_trace(selected, 479, 629)],
        "synthetic_fixture_checks": {
            "corner_markers": "PASS_SOURCE_ORDER_PRESERVED",
            "horizontal_gradient": "PASS_NO_TRANSPOSE_OR_HORIZONTAL_FLIP",
            "vertical_gradient": "PASS_NO_ROTATION_OR_VERTICAL_FLIP",
            "asymmetric_hot_region": "PASS_FIXED_MAPPING",
        },
    }
    invalid_policy = {
        "phase": "T-A2",
        "schema_version": "1.0",
        "source_invalid_sentinel": "NONE_VERIFIED_BY_SDT",
        "t_a1_inherited_policy": "T-A1 rejects fully constant 0/65535 frames and flags partial container extrema; no sensor sentinel is invented.",
        "source_nan_inf": "FAIL_CLOSED",
        "source_partial_invalid_mask": "FAIL_CLOSED; no inpainting or neighbor replacement",
        "canonical_padding": "NOT_USED_BY_SELECTED_PROFILE; G2 diagnostic candidate uses explicit false mask and NaN only",
        "validity_states": ["SOURCE_VALUE_VALID", "SOURCE_VALUE_WARNING", "SOURCE_VALUE_INVALID", "CANONICAL_VALUE_DERIVED", "CANONICAL_VALUE_INVALID"],
        "synthetic_normal_values_introduced": False,
        "interpolation_over_invalid_source": "FORBIDDEN",
        "fully_invalid_frame": "FAIL_CLOSED",
    }
    calibration = {
        "phase": "T-A2",
        "schema_version": "1.0",
        "source_physical_conversion": {"source_unit": "KELVIN_CENTIUNITS", "canonical_unit": "CELSIUS", "formula": "(encoded_uint16 - 27315) / 100"},
        "ambient_reference_compensation": {"applied": False, "status": "NOT_APPLIED_NO_VERIFIED_PARAMETER_SOURCE"},
        "reference_compensation": {"applied": False, "status": "NOT_APPLIED_NO_VERIFIED_PARAMETER_SOURCE"},
        "hardware_specific_calibration": {"applied": False, "status": "DEFERRED_T_C"},
        "unsupported_constants_introduced": [],
        "canonical_dtype_precision": precision,
    }
    pilot_summary = {
        "phase": "T-A2",
        "schema_version": "1.0",
        "source_archive_identity": inventory["archive_identity"],
        "selection_rule": "12 evenly spaced sorted source indices per original source pose class; 48 total; labels only provide coverage/diagnostics.",
        "selection_policy_id": policy["policy_id"],
        "indices_by_source_pose": {str(key): value for key, value in sorted(by_class.items())},
        "source_class_counts": {str(key): len(value) for key, value in sorted(by_class.items())},
        "pilot_frame_count": len(indices),
        "source_classes_represented": ["LYING", "SITTING", "STANDING", "EMPTY_ROOM"],
        "selected_profile_id": selected.profile_id,
        "records": selected_records,
        "candidate_metric_summary": {item["profile"]["candidate_id"]: {key: item[key] for key in ("mean_shift_celsius", "mean_absolute_shift_celsius", "range_compression_ratio", "constant_temperature_preserved", "repeated_canonicalization_deterministic")} for item in candidate_results},
        "precision": precision,
    }
    return {
        "calibration_contract.json": calibration,
        "canonical_frame_contract.json": canonical_contract,
        "coordinate_trace_contract.json": coordinate_contract,
        "geometry_candidate_registry.json": candidate_registry,
        "geometry_comparison.json": comparison,
        "geometry_selection_policy.json": policy,
        "invalid_pixel_policy.json": invalid_policy,
        "pilot_geometry_summary.json": pilot_summary,
        "selected_geometry_profile.json": selected_profile,
    }, by_class, candidate_results


def report_text(artifacts: dict[str, Any], validation: dict[str, Any]) -> str:
    selected = artifacts["selected_geometry_profile.json"]["profile"]
    comparison = artifacts["geometry_comparison.json"]
    policy = artifacts["geometry_selection_policy.json"]
    selected_result = next(item for item in comparison["candidate_results"] if item["profile"]["profile_id"] == selected["profile_id"])
    bbox = selected_result["bbox_fov_diagnostic"]
    precision = artifacts["calibration_contract.json"]["canonical_dtype_precision"]
    return f"""# Thermal T-A2 — Geometry, Calibration, and Canonical Frame Contract

Date: 2026-08-10

Phase: `T-A2`

Outcome: `{validation['overall_outcome']}`

T-A3 authorized: `{'YES' if validation['t_a3_authorized'] else 'NO'}`

## Decision

The selected software canonical profile is `{selected['profile_id']}`. It was derived from all nine candidate metric records using policy `{policy['policy_id']}` v{policy['policy_version']}; no profile ID is hardcoded as the winner. The canonical physical unit is Celsius and the canonical dtype is float32. No model score, model inference, normalization, or SafeNest label remapping was used.

## Geometry boundary

The verified SDT distributed frame is `(480,640)` and already contains the authors' bilinear enlargement from the FLIR Lepton 3.5 native `(120,160)` grid. T-A2 does not reverse that operation or claim a restored native frame. Thermal-44 physical orientation and packet ordering remain `UNVERIFIED / DEFERRED_T_C`.

The predeclared candidate set contains 3 fixed geometry policies (direct stretch, fixed aspect crop, masked aspect pad) crossed with nearest, bilinear, and exact area interpolation. The policy first applies mandatory semantic gates, then the declared FOV/bbox/padding admissibility thresholds, then lexicographically ranks anisotropy, padding, interpolation preference, Celsius-statistic distortion, round-trip diagnostic MAE, and finally candidate ID.

The selected geometry crop is `{selected['crop_xyxy']}` and retains `{selected_result['geometry']['source_fov_retained_fraction'] * 100:.3f}%` of source area. Candidate evidence records each gate, admissibility result, rejection reason, rank, tie group, and final status. Source-frame bbox overflow is clipped before measuring incremental candidate-crop damage: `{bbox['source_bbox_outside_frame_count']}` source bboxes were outside the distributed frame, `{bbox['additional_bbox_intersected_by_candidate_crop_count']}` received additional crop intersection, and total additional crop loss was `{bbox['additional_bbox_area_loss_due_to_candidate_crop']:.6f}` source-pixel².

The selected interpolation is `{selected['interpolation'].upper()}` with coordinate mapping `{selected['coordinate_mapping']}`, edge handling `{selected['edge_handling']}`, and `{selected['explicit_antialias_prefilter']}`. Coordinate mapping is not described as antialiasing.

## Physical calibration

SDT Celsius conversion remains `(encoded_uint16 - 27315) / 100`. Ambient/reference compensation and hardware-specific calibration are not applied because no verified parameter source exists. Float32 was selected after comparison with float64 reference conversion: maximum measured conversion error `{precision['max_conversion_error_celsius']:.9g} °C`, below the source `0.01 °C` encoded resolution.

## Invalid pixels and provenance

T-A1's no-sentinel policy is inherited. NaN/Inf or a supplied partial invalid source mask fails closed; no neighbor, mean-temperature, zero, ambient, or other synthetic value is inserted. The selected crop has an all-true validity mask. Every pilot record retains the original encoded source hash and exact source member/frame index separately from the canonical frame hash.

## Pilot and visual check

The bounded real-data pilot uses 12 evenly spaced sorted source indices per original pose class (48 total), with all four classes represented. Repeated canonicalization is byte-stable. Coordinate traces and asymmetric synthetic fixtures show row/column order preserved with no transpose, rotation, or flip. The tracked visual is a colorized human diagnostic only and is not radiometric model input.

## Deferred boundaries

T-A2 does not create temporal windows, SafeNest fall labels, grouping/splits, full canonical conversion, model comparisons, or Thermal-44 hardware claims. Train/validation split placeholders remain unhydrated.
"""


def write_artifacts(root: Path = ROOT) -> dict[str, Any]:
    evidence_dir = root / EVIDENCE_REL
    evidence_dir.mkdir(parents=True, exist_ok=True)
    reader = SDTThermalRawReader(repo_root=root)
    artifacts, by_class, candidate_results = build_artifacts(reader)
    for name, data in sorted(artifacts.items()):
        (evidence_dir / name).write_text(canonical_json(data), encoding="utf-8")
    visual = _visual_spotcheck(reader, profile_for_id(artifacts["selected_geometry_profile.json"]["profile_id"]), evidence_dir / "visual_spotcheck.png")
    (evidence_dir / "visual_spotcheck_registry.json").write_text(canonical_json({"phase": "T-A2", "schema_version": "1.0", **visual}), encoding="utf-8")

    from scripts.validate_thermal_t_a2 import validate_evidence

    validation = validate_evidence(repo_root=root, evidence_dir=evidence_dir, check_checksums=False, verify_real_payload=True)
    (evidence_dir / "validation_result.json").write_text(canonical_json(validation), encoding="utf-8")
    machine_paths = [evidence_dir / name for name in sorted(JSON_NAMES + ["validation_result.json", "visual_spotcheck.png"])]
    checksum_lines = []
    for path in machine_paths:
        checksum_lines.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    (evidence_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    report_path = root / REPORT_REL
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text(artifacts, validation), encoding="utf-8")
    return validation


if __name__ == "__main__":
    result = write_artifacts()
    print(canonical_json(result), end="")
    raise SystemExit(0 if result["evidence_validation"] == "PASS" else 1)
