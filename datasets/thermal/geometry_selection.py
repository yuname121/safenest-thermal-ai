#!/usr/bin/env python3
"""Deterministic, model-independent T-A2 geometry selection policy."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


POLICY_ID = "THERMAL_T_A2_GEOMETRY_SELECTION_POLICY_002"
POLICY_VERSION = "2.0"
NUMERIC_TIE_TOLERANCE = 1e-12


def selection_policy() -> dict[str, Any]:
    return deepcopy({
        "phase": "T-A2",
        "schema_version": "1.0",
        "policy_id": POLICY_ID,
        "policy_version": POLICY_VERSION,
        "selection_algorithm": "MANDATORY_GATES_THEN_ADMISSIBILITY_THEN_LEXICOGRAPHIC_RANKING",
        "model_performance_used": False,
        "metric_definitions": {
            "anisotropy_ratio_excess": "max(horizontal_scale_factor, vertical_scale_factor) / min(horizontal_scale_factor, vertical_scale_factor) - 1; dimensionless; lower is better",
            "source_fov_retained_fraction": "candidate crop area / distributed source-frame area; fraction; thresholded before ranking",
            "padding_percentage": "masked/non-measured canonical pixels / canonical pixel count * 100; percent; lower is better",
            "mean_absolute_shift_celsius": "mean absolute candidate-minus-source frame mean; degrees Celsius; lower is better",
            "range_compression_distance_from_one": "abs(mean range_compression_ratio - 1); dimensionless; lower is better",
            "round_trip_mae": "diagnostic downsample/reconstruction mean absolute error; degrees Celsius; lower is better",
            "additional_bbox_area_loss_due_to_candidate_crop_fraction": "candidate-crop area loss / source-frame-clipped person-bbox area; fraction; thresholded before ranking",
        },
        "mandatory_rejection_gates": [
            {"gate_id": "SOURCE_AND_CANONICAL_SHAPES", "condition": "profile.source_shape == [480,640] and profile.canonical_shape == [62,80]", "unit": "shape tuple"},
            {"gate_id": "PHYSICAL_FRAME_CONTRACT", "condition": "source_unit == CELSIUS and canonical_unit == CELSIUS and canonical_dtype == float32", "unit": "declared representation"},
            {"gate_id": "SOURCE_AS_STORED_ORIENTATION", "condition": "rotation == 0 and horizontal_flip == false and vertical_flip == false", "unit": "boolean/quarter-turn"},
            {"gate_id": "FINITE_VALID_OUTPUT", "condition": "finite_valid_output == true", "unit": "boolean"},
            {"gate_id": "CONSTANT_TEMPERATURE_PRESERVED", "condition": "constant_temperature_preserved == true", "unit": "boolean"},
            {"gate_id": "DETERMINISTIC_REPEATED_CANONICALIZATION", "condition": "repeated_canonicalization_deterministic == true", "unit": "boolean"},
        ],
        "admissibility_thresholds": {
            "source_fov_retained_fraction_min": {"value": 0.95, "inclusive": True, "unit": "fraction of source area", "justification": "At most five percent bounded source-area loss is admissible."},
            "minimum_bbox_retention_due_to_candidate_transform_min": {"value": 0.995, "inclusive": True, "unit": "fraction after source-frame clipping", "justification": "Candidate geometry must not materially remove a labeled person."},
            "padding_percentage_max": {"value": 0.0, "inclusive": True, "unit": "percent of canonical pixels", "justification": "Reject synthetic/masked pixels when a measured rectangular candidate exists."},
            "candidate_crop_additional_bbox_area_loss_fraction_max": {"value": 0.01, "inclusive": True, "unit": "fraction of source-clipped person-bbox area", "justification": "One percent permits annotation-boundary overlap while rejecting material crop damage."},
        },
        "ranking_order": [
            {"metric": "anisotropy_ratio_excess", "direction": "ascending", "unit": "ratio excess"},
            {"metric": "padding_percentage", "direction": "ascending", "unit": "percent"},
            {"metric": "interpolation_preference_rank", "direction": "ascending", "unit": "ordinal (lower is preferred)"},
            {"metric": "mean_absolute_shift_celsius", "direction": "ascending", "unit": "degrees Celsius"},
            {"metric": "range_compression_distance_from_one", "direction": "ascending", "unit": "absolute ratio distance"},
            {"metric": "round_trip_mae", "direction": "ascending", "unit": "degrees Celsius"},
            {"metric": "candidate_id", "direction": "ascending", "unit": "lexical identifier"},
        ],
        "interpolation_preference": {"bilinear": 0, "area": 1, "nearest": 2},
        "tie_definition": {"numeric_tolerance": NUMERIC_TIE_TOLERANCE, "fields": ["anisotropy_ratio_excess", "padding_percentage", "interpolation_preference_rank", "mean_absolute_shift_celsius", "range_compression_distance_from_one", "round_trip_mae"], "candidate_id_excluded_until_tie_break": True},
        "tie_break_rule": "candidate_id ascending after all declared numeric ranking fields are tied within numeric_tolerance",
        "unsupported_metrics": ["accuracy", "f1", "fall_recall", "model_score", "inference_latency", "Thermal-44 hardware performance"],
        "policy_justification": [
            "FOV and bbox thresholds define an admissible geometry set without model output.",
            "Within admissible geometry, anisotropy is minimized before physical-statistic distortion.",
            "Bilinear is the declared deterministic interpolation preference; this is not an antialias claim.",
            "Candidate ID is only the final neutral reproducibility tie-break.",
        ],
    })


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def evaluate_candidate(candidate: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    profile = candidate.get("profile", {})
    geometry = candidate.get("geometry", {})
    bbox = candidate.get("bbox_fov_diagnostic", {})
    gates = {
        "SOURCE_AND_CANONICAL_SHAPES": profile.get("source_shape") == [480, 640] and profile.get("canonical_shape") == [62, 80],
        "PHYSICAL_FRAME_CONTRACT": profile.get("source_unit") == "CELSIUS" and profile.get("canonical_unit") == "CELSIUS" and profile.get("canonical_dtype") == "float32",
        "SOURCE_AS_STORED_ORIENTATION": profile.get("rotation") == 0 and profile.get("horizontal_flip") is False and profile.get("vertical_flip") is False,
        "FINITE_VALID_OUTPUT": candidate.get("finite_valid_output") is True,
        "CONSTANT_TEMPERATURE_PRESERVED": candidate.get("constant_temperature_preserved") is True,
        "DETERMINISTIC_REPEATED_CANONICALIZATION": candidate.get("repeated_canonicalization_deterministic") is True,
    }
    failed_gates = [key for key, passed in gates.items() if not passed]
    thresholds = policy.get("admissibility_thresholds", {})
    fov = _number(geometry.get("source_fov_retained_fraction"))
    padding = _number(geometry.get("padding_percentage"))
    bbox_retention = _number(bbox.get("minimum_bbox_retention_due_to_candidate_transform"), 1.0)
    bbox_loss_fraction = _number(bbox.get("additional_bbox_area_loss_due_to_candidate_crop_fraction"), 0.0)
    checks = {
        "SOURCE_FOV_WITHIN_THRESHOLD": fov >= _number(thresholds.get("source_fov_retained_fraction_min", {}).get("value")),
        "BBOX_RETENTION_WITHIN_THRESHOLD": bbox_retention >= _number(thresholds.get("minimum_bbox_retention_due_to_candidate_transform_min", {}).get("value")),
        "NO_SYNTHETIC_PADDING": padding <= _number(thresholds.get("padding_percentage_max", {}).get("value")),
        "NO_MATERIAL_ADDITIONAL_CROP_BBOX_LOSS": bbox_loss_fraction <= _number(thresholds.get("candidate_crop_additional_bbox_area_loss_fraction_max", {}).get("value")),
    }
    failed_admissibility = [key for key, passed in checks.items() if not passed]
    interpolation = str(profile.get("interpolation", "")).lower()
    metrics = {
        "anisotropy_ratio_excess": _number(geometry.get("anisotropy_ratio_excess")),
        "padding_percentage": padding,
        "interpolation_preference_rank": int(policy.get("interpolation_preference", {}).get(interpolation, 999)),
        "mean_absolute_shift_celsius": _number(candidate.get("mean_absolute_shift_celsius")),
        "range_compression_distance_from_one": abs(_number(candidate.get("range_compression_ratio"), 1.0) - 1.0),
        "round_trip_mae": _number(candidate.get("round_trip", {}).get("mae")),
        "candidate_id": str(profile.get("candidate_id", "")),
    }
    reasons = [f"MANDATORY_{key}" for key in failed_gates] + [f"ADMISSIBILITY_{key}" for key in failed_admissibility]
    return {
        "mandatory_gates": {"all_pass": not failed_gates, "checks": gates, "failed": failed_gates},
        "admissibility": {"all_pass": not failed_gates and not failed_admissibility, "checks": checks, "failed": failed_admissibility},
        "admissible": not failed_gates and not failed_admissibility,
        "rejection_reason": reasons[0] if reasons else None,
        "ranking_metrics": metrics,
    }


def _sort_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = item["ranking_metrics"]
    return tuple(metrics[key] for key in ("anisotropy_ratio_excess", "padding_percentage", "interpolation_preference_rank", "mean_absolute_shift_celsius", "range_compression_distance_from_one", "round_trip_mae", "candidate_id"))


def apply_selection_policy(candidates: list[dict[str, Any]], policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    active = policy or selection_policy()
    evaluated = []
    for candidate in candidates:
        item = dict(candidate)
        item.update(evaluate_candidate(candidate, active))
        evaluated.append(item)
    admissible = sorted((item for item in evaluated if item["admissible"]), key=_sort_key)
    if not admissible:
        raise ValueError("No candidate satisfies the declared T-A2 policy")
    for item in evaluated:
        item["rank"] = None
        item["tie_group"] = None
        item["final_selection_status"] = "REJECTED_POLICY" if not item["admissible"] else "ADMISSIBLE_NOT_SELECTED"
    tolerance = _number(active.get("tie_definition", {}).get("numeric_tolerance"), NUMERIC_TIE_TOLERANCE)
    tie_fields = active.get("tie_definition", {}).get("fields", [])
    previous = None
    tie_number = 0
    for rank, item in enumerate(admissible, 1):
        item["rank"] = rank
        if previous is not None and all(abs(_number(item["ranking_metrics"].get(field)) - _number(previous["ranking_metrics"].get(field))) <= tolerance for field in tie_fields):
            item["tie_group"] = previous["tie_group"]
        else:
            tie_number += 1
            item["tie_group"] = f"TIE_GROUP_{tie_number:02d}"
        previous = item
    winner = admissible[0]
    winner["final_selection_status"] = "SELECTED"
    for item in evaluated:
        item["selection_reason"] = "SELECTED_BY_DECLARED_POLICY" if item is winner else ("ADMISSIBLE_NOT_SELECTED_BY_DECLARED_RANKING" if item["admissible"] else f"REJECTED_BY_DECLARED_POLICY:{item['rejection_reason']}")
    return {"policy_id": active.get("policy_id"), "policy_version": active.get("policy_version"), "selected_candidate_id": winner["profile"]["candidate_id"], "selected_profile_id": winner["profile"]["profile_id"], "candidates": sorted(evaluated, key=lambda item: str(item["profile"].get("candidate_id", ""))), "winner": winner}


__all__ = ["POLICY_ID", "POLICY_VERSION", "apply_selection_policy", "evaluate_candidate", "selection_policy"]
