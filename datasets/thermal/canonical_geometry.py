#!/usr/bin/env python3
"""Pure deterministic geometry and physical-frame operations for Thermal T-A2.

This module stops at a canonical physical frame.  It deliberately has no
model, label, split, sensor-driver, or normalization dependency.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


SOURCE_SHAPE = (480, 640)
CANONICAL_SHAPE = (62, 80)
SOURCE_UNIT = "CELSIUS"
CANONICAL_UNIT = "CELSIUS"
CANONICAL_DTYPE = np.dtype("<f4")


class GeometryError(ValueError):
    """Base error for unsupported or unsafe geometry operations."""


class GeometryShapeError(GeometryError):
    pass


class InvalidPixelError(GeometryError):
    pass


class GeometryPathError(GeometryError):
    pass


@dataclass(frozen=True)
class GeometryProfile:
    """Immutable geometry profile; changing any field requires a new ID."""

    profile_id: str
    candidate_id: str
    geometry_policy: str
    source_shape: tuple[int, int]
    canonical_shape: tuple[int, int]
    crop_xyxy: tuple[int, int, int, int]
    resize_shape: tuple[int, int]
    padding_tblr: tuple[int, int, int, int]
    rotation: int
    horizontal_flip: bool
    vertical_flip: bool
    interpolation: str
    coordinate_convention: str
    invalid_pixel_policy: str
    coordinate_mapping: str = "HALF_PIXEL_CENTER"
    edge_handling: str = "EDGE_CLAMPING"
    explicit_antialias_prefilter: str = "NO_EXPLICIT_ANTIALIAS_PREFILTER"
    source_unit: str = SOURCE_UNIT
    canonical_unit: str = CANONICAL_UNIT
    canonical_dtype: str = "float32"

    @property
    def crop_left(self) -> int:
        return self.crop_xyxy[0]

    @property
    def crop_top(self) -> int:
        return self.crop_xyxy[1]

    @property
    def crop_right(self) -> int:
        return self.crop_xyxy[2]

    @property
    def crop_bottom(self) -> int:
        return self.crop_xyxy[3]

    @property
    def crop_width(self) -> int:
        return self.crop_right - self.crop_left

    @property
    def crop_height(self) -> int:
        return self.crop_bottom - self.crop_top

    @property
    def resize_height(self) -> int:
        return self.resize_shape[0]

    @property
    def resize_width(self) -> int:
        return self.resize_shape[1]

    @property
    def pad_top(self) -> int:
        return self.padding_tblr[0]

    @property
    def pad_bottom(self) -> int:
        return self.padding_tblr[1]

    @property
    def pad_left(self) -> int:
        return self.padding_tblr[2]

    @property
    def pad_right(self) -> int:
        return self.padding_tblr[3]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "canonical_dtype": self.canonical_dtype,
            "canonical_shape": list(self.canonical_shape),
            "canonical_unit": self.canonical_unit,
            "coordinate_convention": self.coordinate_convention,
            "coordinate_mapping": self.coordinate_mapping,
            "crop_xyxy": list(self.crop_xyxy),
            "edge_handling": self.edge_handling,
            "explicit_antialias_prefilter": self.explicit_antialias_prefilter,
            "geometry_policy": self.geometry_policy,
            "horizontal_flip": self.horizontal_flip,
            "interpolation": self.interpolation,
            "interpolation_semantics": self.interpolation.upper(),
            "invalid_pixel_policy": self.invalid_pixel_policy,
            "padding_tblr": list(self.padding_tblr),
            "profile_id": self.profile_id,
            "resize_shape": list(self.resize_shape),
            "rotation": self.rotation,
            "source_shape": list(self.source_shape),
            "source_unit": self.source_unit,
            "vertical_flip": self.vertical_flip,
        }


@dataclass(frozen=True)
class CanonicalPhysicalFrame:
    """Immutable canonical physical frame plus compact provenance."""

    source_provenance: Mapping[str, Any]
    geometry_profile_id: str
    source_shape: tuple[int, int]
    canonical_shape: tuple[int, int]
    source_unit: str
    canonical_unit: str
    source_dtype: str
    canonical_dtype: str
    orientation_transform: str
    crop_xyxy: tuple[int, int, int, int]
    padding_tblr: tuple[int, int, int, int]
    resize_shape: tuple[int, int]
    interpolation: str
    coordinate_mapping: str
    edge_handling: str
    explicit_antialias_prefilter: str
    validity_status: str
    source_frame_hash: str
    canonical_frame_hash: str
    physical_frame: np.ndarray
    validity_mask: np.ndarray

    def provenance_dict(self) -> dict[str, Any]:
        return {
            "canonical_dtype": self.canonical_dtype,
            "canonical_frame_hash": self.canonical_frame_hash,
            "canonical_shape": list(self.canonical_shape),
            "canonical_unit": self.canonical_unit,
            "crop_xyxy": list(self.crop_xyxy),
            "geometry_profile_id": self.geometry_profile_id,
            "coordinate_mapping": self.coordinate_mapping,
            "edge_handling": self.edge_handling,
            "explicit_antialias_prefilter": self.explicit_antialias_prefilter,
            "interpolation": self.interpolation,
            "orientation_transform": self.orientation_transform,
            "padding_tblr": list(self.padding_tblr),
            "resize_shape": list(self.resize_shape),
            "source_dtype": self.source_dtype,
            "source_frame_hash": self.source_frame_hash,
            "source_shape": list(self.source_shape),
            "source_unit": self.source_unit,
            "validity_status": self.validity_status,
        }


def _ensure_portable_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or value.startswith(("/", "~/", "file://")) or "\\" in value:
        raise GeometryPathError(f"repository-relative POSIX path required: {value!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise GeometryPathError(f"unsafe repository-relative path: {value!r}")
    return value


def _validate_profile(profile: GeometryProfile) -> None:
    if profile.source_shape != SOURCE_SHAPE or profile.canonical_shape != CANONICAL_SHAPE:
        raise GeometryShapeError("T-A2 profile shape differs from locked SDT/software shapes")
    left, top, right, bottom = profile.crop_xyxy
    if not (0 <= left < right <= SOURCE_SHAPE[1] and 0 <= top < bottom <= SOURCE_SHAPE[0]):
        raise GeometryShapeError(f"invalid fixed crop: {profile.crop_xyxy}")
    if profile.resize_height <= 0 or profile.resize_width <= 0:
        raise GeometryShapeError("resize shape must be positive")
    if any(int(value) != value or value < 0 for value in profile.padding_tblr):
        raise GeometryShapeError("padding values must be non-negative integers")
    if profile.pad_top + profile.resize_height + profile.pad_bottom != CANONICAL_SHAPE[0]:
        raise GeometryShapeError("vertical resize/padding does not produce canonical height")
    if profile.pad_left + profile.resize_width + profile.pad_right != CANONICAL_SHAPE[1]:
        raise GeometryShapeError("horizontal resize/padding does not produce canonical width")
    if profile.rotation != 0 or profile.horizontal_flip or profile.vertical_flip:
        raise GeometryError("T-A2 selected software orientation must be source-as-stored")
    if profile.interpolation not in {"nearest", "bilinear", "area"}:
        raise GeometryError(f"unsupported interpolation: {profile.interpolation}")
    if profile.canonical_dtype != "float32":
        raise GeometryError("T-A2 selected physical dtype must be explicit float32")


def make_candidate_profiles() -> tuple[GeometryProfile, ...]:
    """Return the predeclared 3 geometry x 3 interpolation candidate set."""
    candidates: list[GeometryProfile] = []
    definitions = (
        ("G0_DIRECT_STRETCH", "DIRECT_STRETCH", (0, 0, 640, 480), (62, 80), (0, 0, 0, 0), "EDGE_CLAMPING"),
        ("G1_FIXED_ASPECT_CROP", "FIXED_ASPECT_CROP", (10, 0, 630, 480), (62, 80), (0, 0, 0, 0), "EDGE_CLAMPING"),
        ("G2_ASPECT_PAD_MASKED", "ASPECT_PRESERVING_PAD", (0, 0, 640, 480), (60, 80), (1, 1, 0, 0), "MASKED_NAN_PADDING"),
    )
    for geometry_id, policy, crop, resize_shape, padding, edge_policy in definitions:
        for interpolation in ("nearest", "bilinear", "area"):
            profile_id = f"{geometry_id}_{interpolation.upper()}"
            if interpolation == "area":
                coord_map = "EXACT_SOURCE_AREA_INTEGRATION"
            else:
                coord_map = "HALF_PIXEL_CENTER"

            profile = GeometryProfile(
                profile_id=profile_id,
                candidate_id=profile_id,
                geometry_policy=policy,
                source_shape=SOURCE_SHAPE,
                canonical_shape=CANONICAL_SHAPE,
                crop_xyxy=crop,
                resize_shape=resize_shape,
                padding_tblr=padding,
                rotation=0,
                horizontal_flip=False,
                vertical_flip=False,
                interpolation=interpolation,
                coordinate_convention="row-major; pixel centers at integer source coordinates; output center maps with half-pixel formula",
                invalid_pixel_policy=f"FAIL_CLOSED_SOURCE_INVALID; {edge_policy}",
                coordinate_mapping=coord_map,
                edge_handling=edge_policy,
                explicit_antialias_prefilter="NO_EXPLICIT_ANTIALIAS_PREFILTER",
            )
            _validate_profile(profile)
            candidates.append(profile)
    return tuple(candidates)


def get_candidate_profile(profile_id: str) -> GeometryProfile:
    """Return candidate profile by profile_id or candidate_id."""
    for candidate in make_candidate_profiles():
        if candidate.profile_id == profile_id or candidate.candidate_id == profile_id:
            return candidate
    raise GeometryError(f"unknown geometry profile_id: {profile_id!r}")


def profile_for_id(profile_id: str) -> GeometryProfile:
    """Resolve a profile only from an explicitly derived candidate ID."""
    return get_candidate_profile(profile_id)


def legacy_profile_lookup(profile_id: str) -> GeometryProfile:
    """Resolve an explicitly supplied profile ID; no default winner exists."""
    return profile_for_id(profile_id)


SELECTION_RULE_VERSION = "T-A2_GEOMETRY_RANKING_RULE_V1"

SELECTION_RULE_DEFINITION = {
    "version": SELECTION_RULE_VERSION,
    "description": "Pre-declared 2-stage operational candidate ranking policy for Thermal T-A2 geometry selection.",
    "stage1_mandatory_eligibility_constraints": {
        "preserves_physical_semantics": "source_unit == CELSIUS and canonical_unit == CELSIUS and canonical_dtype == float32",
        "orientation_as_stored": "rotation == 0 and horizontal_flip == False and vertical_flip == False",
        "constant_temperature_preserved": "constant_temperature_preserved == True",
        "repeated_canonicalization_deterministic": "repeated_canonicalization_deterministic == True",
        "no_masked_padding_pixels": "padding_pixel_count == 0 (100% valid rectangular array coverage)",
        "max_source_fov_loss_percentage": 5.0,
        "no_bbox_removed_by_crop": "bbox_removed_by_candidate_crop_count == 0",
    },
    "stage2_lexicographic_ranking_criteria": [
        "1. Minimal anisotropy_ratio_excess (abs difference between horizontal and vertical scale factors)",
        "2. Interpolation preference: bilinear (1) > area (2) > nearest (3)",
        "3. Minimal mean_absolute_shift_celsius",
        "4. Minimal round_trip MAE (MAE of downsampled and reconstructed region)",
        "5. Profile ID lexicographical order",
    ],
}


def evaluate_candidate_eligibility_and_ranking(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the pre-declared operational ranking policy to candidate metrics."""
    evaluated = []
    interp_order = {"bilinear": 1, "area": 2, "nearest": 3}

    for candidate in candidates:
        prof = candidate["profile"]
        geom = candidate["geometry"]
        bbox = candidate.get("bbox_fov_diagnostic", {})

        failed = []
        if prof.get("source_unit") != SOURCE_UNIT or prof.get("canonical_unit") != CANONICAL_UNIT or prof.get("canonical_dtype") != "float32":
            failed.append("INVALID_PHYSICAL_SEMANTICS")
        if prof.get("rotation") != 0 or prof.get("horizontal_flip") or prof.get("vertical_flip"):
            failed.append("INVALID_SOFTWARE_ORIENTATION")
        if not candidate.get("constant_temperature_preserved", False):
            failed.append("CONSTANT_TEMPERATURE_NOT_PRESERVED")
        if not candidate.get("repeated_canonicalization_deterministic", False):
            failed.append("NONDETERMINISTIC_CANONICALIZATION")
        if geom.get("padding_pixel_count", 0) > 0:
            failed.append("MASKED_PADDING_PIXELS_PRESENT")
        crop_pct = float(geom.get("crop_percentage", 0.0))
        if crop_pct > 5.0:
            failed.append("EXCESSIVE_SOURCE_FOV_LOSS")
        if bbox.get("bbox_removed_by_candidate_crop_count", 0) > 0:
            failed.append("BBOX_REMOVED_BY_CROP")

        eligible = (len(failed) == 0)

        anisotropy = float(geom.get("anisotropy_ratio_excess", 0.0))
        mean_abs_shift = float(candidate.get("mean_absolute_shift_celsius", 0.0))
        round_trip_mae = float(candidate.get("round_trip", {}).get("mae", 0.0))
        interp_pref = interp_order.get(prof.get("interpolation"), 99)
        pid = str(prof.get("profile_id", ""))

        ranking_tuple = (
            0 if eligible else 1,
            anisotropy,
            interp_pref,
            mean_abs_shift,
            round_trip_mae,
            pid,
        )

        evaluated.append({
            "candidate_id": prof.get("candidate_id"),
            "profile_id": pid,
            "eligible": eligible,
            "failed_constraints": failed,
            "ranking_values": {
                "anisotropy_ratio_excess": anisotropy,
                "mean_absolute_shift_celsius": mean_abs_shift,
                "round_trip_mae": round_trip_mae,
                "interpolation_preference": interp_pref,
                "crop_percentage": crop_pct,
                "padding_pixel_count": geom.get("padding_pixel_count", 0),
            },
            "ranking_tuple": [float(x) if isinstance(x, (int, float)) else str(x) for x in ranking_tuple],
            "profile": prof,
        })

    sorted_candidates = sorted(evaluated, key=lambda c: c["ranking_tuple"])
    for idx, item in enumerate(sorted_candidates, 1):
        item["ranking_position"] = idx
        if item["eligible"]:
            item["selection_reason"] = f"Eligible candidate ranked position {idx} (anisotropy={item['ranking_values']['anisotropy_ratio_excess']:.6f})."
        else:
            item["selection_reason"] = f"Ineligible candidate (failed: {', '.join(item['failed_constraints'])})."

    winner = sorted_candidates[0]
    if not winner["eligible"]:
        raise ValueError("No candidate profile satisfied Stage 1 mandatory eligibility constraints!")

    return {
        "selection_rule_version": SELECTION_RULE_VERSION,
        "selection_rule_definition": SELECTION_RULE_DEFINITION,
        "selected_profile_id": winner["profile_id"],
        "selected_candidate_id": winner["candidate_id"],
        "ranking_trace": sorted_candidates,
        "winner": winner,
    }


def _validate_source(source: np.ndarray, expected_shape: tuple[int, int] = SOURCE_SHAPE) -> np.ndarray:
    array = np.asarray(source)
    if array.shape != expected_shape:
        raise GeometryShapeError(f"source shape {array.shape} != {expected_shape}")
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
        raise GeometryShapeError(f"source must be a numeric 2D array, got {array.dtype}")
    numeric = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise InvalidPixelError("source contains NaN or infinity")
    return numeric


def _resize_nearest(source: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    in_h, in_w = source.shape
    out_h, out_w = out_shape
    ys = (np.arange(out_h, dtype=np.float64) + 0.5) * in_h / out_h - 0.5
    xs = (np.arange(out_w, dtype=np.float64) + 0.5) * in_w / out_w - 0.5
    yi = np.clip(np.floor(ys + 0.5).astype(np.int64), 0, in_h - 1)
    xi = np.clip(np.floor(xs + 0.5).astype(np.int64), 0, in_w - 1)
    return source[np.ix_(yi, xi)]


def _resize_bilinear(source: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    in_h, in_w = source.shape
    out_h, out_w = out_shape
    ys = (np.arange(out_h, dtype=np.float64) + 0.5) * in_h / out_h - 0.5
    xs = (np.arange(out_w, dtype=np.float64) + 0.5) * in_w / out_w - 0.5
    y0 = np.clip(np.floor(ys).astype(np.int64), 0, in_h - 1)
    x0 = np.clip(np.floor(xs).astype(np.int64), 0, in_w - 1)
    y1 = np.clip(y0 + 1, 0, in_h - 1)
    x1 = np.clip(x0 + 1, 0, in_w - 1)
    wy = (ys - np.floor(ys)).clip(0.0, 1.0)
    wx = (xs - np.floor(xs)).clip(0.0, 1.0)
    top = source[y0][:, x0] * (1.0 - wx)[None, :] + source[y0][:, x1] * wx[None, :]
    bottom = source[y1][:, x0] * (1.0 - wx)[None, :] + source[y1][:, x1] * wx[None, :]
    return top * (1.0 - wy)[:, None] + bottom * wy[:, None]


def _resize_area(source: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Exact separable box-overlap averaging in source pixel-area coordinates."""
    in_h, in_w = source.shape
    out_h, out_w = out_shape

    def overlap_weights(input_size: int, output_size: int) -> np.ndarray:
        weights = np.zeros((output_size, input_size), dtype=np.float64)
        for output_index in range(output_size):
            start = output_index * input_size / output_size
            end = (output_index + 1) * input_size / output_size
            first = int(math.floor(start))
            last = int(math.ceil(end))
            for source_index in range(first, last):
                weights[output_index, source_index] = max(
                    0.0, min(end, source_index + 1.0) - max(start, source_index)
                )
            total = weights[output_index].sum()
            if total <= 0.0:
                raise GeometryError("area interpolation produced an empty support interval")
            weights[output_index] /= total
        return weights

    y_weights = overlap_weights(in_h, out_h)
    x_weights = overlap_weights(in_w, out_w)
    return y_weights @ source @ x_weights.T


def resize_physical(source: np.ndarray, out_shape: tuple[int, int], interpolation: str) -> np.ndarray:
    numeric = _validate_source(source, expected_shape=source.shape)
    if interpolation == "nearest":
        return _resize_nearest(numeric, out_shape)
    if interpolation == "bilinear":
        return _resize_bilinear(numeric, out_shape)
    if interpolation == "area":
        return _resize_area(numeric, out_shape)
    raise GeometryError(f"unsupported interpolation: {interpolation}")


def _canonical_hash(frame: np.ndarray, validity_mask: np.ndarray) -> str:
    data = np.asarray(frame, dtype=CANONICAL_DTYPE, order="C")
    mask = np.asarray(validity_mask, dtype=np.uint8, order="C")
    digest = hashlib.sha256()
    digest.update(data.tobytes(order="C"))
    digest.update(mask.tobytes(order="C"))
    return digest.hexdigest()


def canonicalize_physical_frame(
    source_physical: np.ndarray,
    profile: GeometryProfile,
    *,
    source_frame_hash: str = "UNKNOWN",
    source_provenance: Mapping[str, Any] | None = None,
    source_validity_mask: np.ndarray | None = None,
) -> CanonicalPhysicalFrame:
    _validate_profile(profile)
    source = _validate_source(source_physical, expected_shape=profile.source_shape)
    if source_validity_mask is not None:
        validity = np.asarray(source_validity_mask, dtype=bool)
        if validity.shape != profile.source_shape:
            raise GeometryShapeError("source validity mask shape mismatch")
        if not np.all(validity):
            raise InvalidPixelError("source invalid pixels require an approved mask-aware transform; no inpainting is allowed")

    left, top, right, bottom = profile.crop_xyxy
    crop = source[top:bottom, left:right]
    resized = resize_physical(crop, profile.resize_shape, profile.interpolation)
    output = np.full(profile.canonical_shape, np.nan, dtype=CANONICAL_DTYPE)
    output_mask = np.zeros(profile.canonical_shape, dtype=bool)
    row_end = profile.pad_top + profile.resize_height
    col_end = profile.pad_left + profile.resize_width
    output[profile.pad_top:row_end, profile.pad_left:col_end] = np.asarray(resized, dtype=CANONICAL_DTYPE)
    output_mask[profile.pad_top:row_end, profile.pad_left:col_end] = True
    if not np.all(np.isfinite(output[output_mask])):
        raise InvalidPixelError("canonicalization produced non-finite valid pixels")
    output.setflags(write=False)
    output_mask.setflags(write=False)
    return CanonicalPhysicalFrame(
        source_provenance=dict(source_provenance or {}),
        geometry_profile_id=profile.profile_id,
        source_shape=profile.source_shape,
        canonical_shape=profile.canonical_shape,
        source_unit=profile.source_unit,
        canonical_unit=profile.canonical_unit,
        source_dtype="float64_physical_input",
        canonical_dtype="float32",
        orientation_transform="SOURCE_ORIENTATION_AS_STORED; ROTATION_NONE; FLIP_NONE",
        crop_xyxy=profile.crop_xyxy,
        padding_tblr=profile.padding_tblr,
        resize_shape=profile.resize_shape,
        interpolation=profile.interpolation,
        coordinate_mapping=profile.coordinate_mapping,
        edge_handling=profile.edge_handling,
        explicit_antialias_prefilter=profile.explicit_antialias_prefilter,
        validity_status="VALID_MEASURED_PIXELS; MASKED_PADDING" if not np.all(output_mask) else "VALID_MEASURED_PIXELS",
        source_frame_hash=source_frame_hash,
        canonical_frame_hash=_canonical_hash(output, output_mask),
        physical_frame=output,
        validity_mask=output_mask,
    )


def canonicalize_source_frame(frame: Any, profile: GeometryProfile) -> CanonicalPhysicalFrame:
    """Canonicalize a T-A1 frame without modifying its encoded source array."""
    source = np.asarray(frame.raw_encoded_frame)
    before_hash = getattr(frame, "raw_encoded_frame_sha256", "UNKNOWN")
    physical = frame.celsius()
    record = frame.provenance_dict() if hasattr(frame, "provenance_dict") else {}
    return canonicalize_physical_frame(
        physical,
        profile,
        source_frame_hash=before_hash,
        source_provenance=record,
    )


def canonical_to_source_trace(profile: GeometryProfile, row: int, column: int) -> dict[str, Any]:
    _validate_profile(profile)
    if not (0 <= row < profile.canonical_shape[0] and 0 <= column < profile.canonical_shape[1]):
        raise GeometryShapeError(f"canonical coordinate out of bounds: {(row, column)}")
    inner_row = row - profile.pad_top
    inner_col = column - profile.pad_left
    if not (0 <= inner_row < profile.resize_height and 0 <= inner_col < profile.resize_width):
        return {
            "canonical_coordinate": [row, column],
            "source_coordinate": None,
            "source_support": [],
            "status": "CANONICAL_PADDING_INVALID",
        }
    source_y = profile.crop_top + (inner_row + 0.5) * profile.crop_height / profile.resize_height - 0.5
    source_x = profile.crop_left + (inner_col + 0.5) * profile.crop_width / profile.resize_width - 0.5
    if profile.interpolation == "nearest":
        support = [[int(np.clip(np.floor(source_y + 0.5), 0, SOURCE_SHAPE[0] - 1)), int(np.clip(np.floor(source_x + 0.5), 0, SOURCE_SHAPE[1] - 1))]]
    elif profile.interpolation == "area":
        y0 = profile.crop_top + inner_row * profile.crop_height / profile.resize_height
        y1 = profile.crop_top + (inner_row + 1) * profile.crop_height / profile.resize_height
        x0 = profile.crop_left + inner_col * profile.crop_width / profile.resize_width
        x1 = profile.crop_left + (inner_col + 1) * profile.crop_width / profile.resize_width
        support = [[r, c] for r in range(int(math.floor(y0)), int(math.ceil(y1))) for c in range(int(math.floor(x0)), int(math.ceil(x1)))]
    else:
        y0 = int(np.clip(math.floor(source_y), 0, SOURCE_SHAPE[0] - 1))
        x0 = int(np.clip(math.floor(source_x), 0, SOURCE_SHAPE[1] - 1))
        y1 = min(y0 + 1, SOURCE_SHAPE[0] - 1)
        x1 = min(x0 + 1, SOURCE_SHAPE[1] - 1)
        support = sorted({(y0, x0), (y0, x1), (y1, x0), (y1, x1)})
        support = [[int(r), int(c)] for r, c in support]
    return {
        "canonical_coordinate": [row, column],
        "source_coordinate": [source_y, source_x],
        "source_support": support,
        "status": "MEASURED_SOURCE_SUPPORT",
    }


def source_to_canonical_trace(profile: GeometryProfile, row: int, column: int) -> dict[str, Any]:
    _validate_profile(profile)
    if not (0 <= row < SOURCE_SHAPE[0] and 0 <= column < SOURCE_SHAPE[1]):
        raise GeometryShapeError(f"source coordinate out of bounds: {(row, column)}")
    if not (profile.crop_top <= row < profile.crop_bottom and profile.crop_left <= column < profile.crop_right):
        return {"source_coordinate": [row, column], "canonical_coordinate": None, "status": "SOURCE_CROPPED_OUT"}
    canonical_y = profile.pad_top + (row - profile.crop_top + 0.5) * profile.resize_height / profile.crop_height - 0.5
    canonical_x = profile.pad_left + (column - profile.crop_left + 0.5) * profile.resize_width / profile.crop_width - 0.5
    return {
        "source_coordinate": [row, column],
        "canonical_coordinate": [canonical_y, canonical_x],
        "status": "MEASURED_CANONICAL_COORDINATE",
    }


def clip_bbox_to_source_frame(bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    """Clip source labels to the distributed frame before candidate crop math."""
    x_min, y_min, x_max, y_max = (float(value) for value in bbox)
    if (x_min, y_min, x_max, y_max) == (-1.0, -1.0, -1.0, -1.0):
        return {
            "source_bbox": list(bbox),
            "source_clipped_bbox": None,
            "source_bbox_outside_frame": False,
            "source_boundary_clipped": False,
            "source_clipped_area": None,
        }
    clipped = (
        max(0.0, min(float(SOURCE_SHAPE[1]), x_min)),
        max(0.0, min(float(SOURCE_SHAPE[0]), y_min)),
        max(0.0, min(float(SOURCE_SHAPE[1]), x_max)),
        max(0.0, min(float(SOURCE_SHAPE[0]), y_max)),
    )
    clipped_area = max(0.0, clipped[2] - clipped[0]) * max(0.0, clipped[3] - clipped[1])
    outside = x_min < 0.0 or y_min < 0.0 or x_max > SOURCE_SHAPE[1] or y_max > SOURCE_SHAPE[0]
    return {
        "source_bbox": list(bbox),
        "source_clipped_bbox": list(clipped),
        "source_bbox_outside_frame": bool(outside),
        "source_boundary_clipped": bool(outside and tuple(clipped) != (x_min, y_min, x_max, y_max)),
        "source_clipped_area": clipped_area,
    }


def transform_bbox(bbox: tuple[float, float, float, float], profile: GeometryProfile) -> dict[str, Any]:
    """Transform a bbox after source-boundary clipping, for diagnostics only."""
    _validate_profile(profile)
    clipped = clip_bbox_to_source_frame(bbox)
    x_min_raw, y_min_raw, x_max_raw, y_max_raw = (float(v) for v in bbox)
    raw_area = max(0.0, x_max_raw - x_min_raw) * max(0.0, y_max_raw - y_min_raw)
    if clipped["source_clipped_bbox"] is None:
        return {
            **clipped,
            "source_bbox_outside_source_frame": clipped.get("source_bbox_outside_frame", False),
            "source_bbox_clipped_area_fraction": 1.0 if raw_area > 0.0 else 0.0,
            "candidate_crop_additional_bbox_area_loss": 0.0,
            "canonical_bbox": None,
            "candidate_crop_bbox_area": None,
            "additional_bbox_area_loss_due_to_candidate_crop": 0.0,
            "retained_area_fraction": None,
            "status": "EMPTY_ROOM",
        }
    x_min, y_min, x_max, y_max = (float(value) for value in clipped["source_clipped_bbox"])
    source_area = float(clipped["source_clipped_area"] or 0.0)
    clip_frac = (raw_area - source_area) / raw_area if raw_area > 0.0 else 0.0
    ix_min = max(x_min, float(profile.crop_left))
    iy_min = max(y_min, float(profile.crop_top))
    ix_max = min(x_max, float(profile.crop_right))
    iy_max = min(y_max, float(profile.crop_bottom))
    retained_area = max(0.0, ix_max - ix_min) * max(0.0, iy_max - iy_min)
    crop_loss = max(0.0, source_area - retained_area)
    if retained_area <= 0.0 or source_area <= 0.0:
        return {
            **clipped,
            "source_bbox_outside_source_frame": clipped.get("source_bbox_outside_frame", False),
            "source_bbox_clipped_area_fraction": clip_frac,
            "candidate_crop_additional_bbox_area_loss": crop_loss,
            "canonical_bbox": None,
            "candidate_crop_bbox_area": retained_area,
            "additional_bbox_area_loss_due_to_candidate_crop": crop_loss,
            "retained_area_fraction": 0.0,
            "status": "BBOX_REMOVED_BY_CANDIDATE_CROP",
        }
    canonical = [
        profile.pad_left + (ix_min - profile.crop_left) * profile.resize_width / profile.crop_width,
        profile.pad_top + (iy_min - profile.crop_top) * profile.resize_height / profile.crop_height,
        profile.pad_left + (ix_max - profile.crop_left) * profile.resize_width / profile.crop_width,
        profile.pad_top + (iy_max - profile.crop_top) * profile.resize_height / profile.crop_height,
    ]
    return {
        **clipped,
        "source_bbox_outside_source_frame": clipped.get("source_bbox_outside_frame", False),
        "source_bbox_clipped_area_fraction": clip_frac,
        "candidate_crop_additional_bbox_area_loss": crop_loss,
        "canonical_bbox": canonical,
        "candidate_crop_bbox_area": retained_area,
        "additional_bbox_area_loss_due_to_candidate_crop": crop_loss,
        "retained_area_fraction": retained_area / source_area,
        "status": "BBOX_FULLY_RETAINED" if crop_loss <= 0.0 else "BBOX_PARTIALLY_CROPPED",
    }


def precision_error(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape:
        raise GeometryShapeError("precision arrays have different shapes")
    valid = np.isfinite(ref) & np.isfinite(cand)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not np.any(valid):
        raise InvalidPixelError("no valid values for precision comparison")
    difference = np.abs(ref[valid] - cand[valid])
    return {
        "max_abs_error": float(np.max(difference)),
        "mean_abs_error": float(np.mean(difference)),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "valid_count": int(np.count_nonzero(valid)),
    }


__all__ = [
    "CANONICAL_DTYPE", "CANONICAL_SHAPE", "CANONICAL_UNIT", "CanonicalPhysicalFrame",
    "GeometryError", "GeometryPathError", "GeometryProfile", "GeometryShapeError",
    "InvalidPixelError", "SELECTION_RULE_DEFINITION", "SELECTION_RULE_VERSION",
    "SOURCE_SHAPE", "SOURCE_UNIT", "canonical_to_source_trace",
    "canonicalize_physical_frame", "canonicalize_source_frame",
    "evaluate_candidate_eligibility_and_ranking", "get_candidate_profile",
    "make_candidate_profiles", "precision_error", "profile_for_id", "legacy_profile_lookup",
    "resize_physical", "source_to_canonical_trace", "transform_bbox",
]
