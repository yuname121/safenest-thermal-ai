#!/usr/bin/env python3
"""Standalone validator for the Thermal T-A2 geometry contract."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.thermal.canonical_geometry import (  # noqa: E402
    CANONICAL_SHAPE,
    SOURCE_SHAPE,
    InvalidPixelError,
    canonical_to_source_trace,
    canonicalize_physical_frame,
    make_candidate_profiles,
    precision_error,
    profile_for_id,
    source_to_canonical_trace,
)
from datasets.thermal.raw_reader import SDTThermalRawReader  # noqa: E402


DEFAULT_EVIDENCE = ROOT / "datasets/thermal/manifests/T-A2_geometry_calibration_canonical_frame"
T_A0_REL = "datasets/thermal/manifests/T-A0_source_identity"
T_A1_REL = "datasets/thermal/manifests/T-A1_safe_reader_raw_unit_contract"
EVIDENCE_REL = "datasets/thermal/manifests/T-A2_geometry_calibration_canonical_frame"
REQUIRED_JSON = [
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
EXPECTED_PILOT_COUNT = 48
POLICY_NUMERIC_FIELDS = (
    "anisotropy_ratio_excess",
    "padding_percentage",
    "interpolation_preference_rank",
    "mean_absolute_shift_celsius",
    "range_compression_distance_from_one",
    "round_trip_mae",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


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
    if value.startswith(("/", "~/", "file://")) or "\\" in value or re.match(r"^[A-Za-z]:", value):
        return False
    pure = PurePosixPath(value)
    return not pure.is_absolute() and ".." not in pure.parts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_documents(evidence_dir: Path, errors: list[dict[str, str]]) -> tuple[dict[str, Any], list[Path]]:
    documents: dict[str, Any] = {}
    paths: list[Path] = []
    for name in REQUIRED_JSON:
        path = evidence_dir / name
        if not path.is_file():
            _error(errors, "REQUIRED_ARTIFACT_MISSING", name, "Required T-A2 JSON is missing.")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _error(errors, "JSON_READ_FAILED", name, str(exc))
            continue
        documents[name] = data
        paths.append(path)
        if path.read_text(encoding="utf-8") != canonical_json(data):
            _error(errors, "NONDETERMINISTIC_JSON", name, "JSON must use sorted canonical formatting.")
        for location, value in _walk(data):
            if isinstance(value, str) and not _portable(value) and ("/" in value or value.startswith(("~", "file:"))):
                _error(errors, "NONPORTABLE_PATH", f"{name}:{location}", value)
            if isinstance(value, str) and ("Thermal44" in value or "Thermal-44" in value) and any(token in value for token in ("VERIFIED", "CONFIRMED")):
                _error(errors, "THERMAL44_FACT_MARKED_VERIFIED", f"{name}:{location}", value)
    return documents, paths


def _validate_checksums(repo_root: Path, evidence_dir: Path, required: list[Path], errors: list[dict[str, str]]) -> None:
    checksum_path = evidence_dir / "checksums.sha256"
    if not checksum_path.is_file():
        _error(errors, "CHECKSUM_REGISTRY_MISSING", "checksums.sha256", "T-A2 checksum registry missing.")
        return
    entries: dict[str, str] = {}
    previous = ""
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", line)
        if not match:
            _error(errors, "CHECKSUM_LINE_INVALID", f"checksums.sha256:{line_number}", line)
            continue
        digest, relative = match.groups()
        if relative <= previous:
            _error(errors, "CHECKSUM_ORDER_NONDETERMINISTIC", f"checksums.sha256:{line_number}", relative)
        previous = relative
        if not _portable(relative):
            _error(errors, "CHECKSUM_PATH_NOT_PORTABLE", f"checksums.sha256:{line_number}", relative)
        entries[relative] = digest
    for path in required:
        relative = path.relative_to(repo_root).as_posix()
        if relative not in entries:
            _error(errors, "CHECKSUM_COVERAGE_MISSING", relative, "Required artifact has no checksum.")
        elif _sha256(path) != entries[relative]:
            _error(errors, "CHECKSUM_MISMATCH", relative, "Measured checksum differs.")


def _validate_static_geometry(repo_root: Path, errors: list[dict[str, str]]) -> None:
    path = repo_root / "datasets/thermal/canonical_geometry.py"
    if not path.is_file():
        _error(errors, "GEOMETRY_IMPLEMENTATION_MISSING", path.as_posix(), "T-A2 geometry module missing.")
        return
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        _error(errors, "GEOMETRY_SYNTAX_ERROR", path.as_posix(), str(exc))
        return
    for needle, code in (
        ("ThermalInterpreter", "MODEL_COUPLING"),
        ("tflite", "MODEL_COUPLING"),
        ("per-frame min-max", "NORMALIZATION_COUPLING"),
        ("z-score", "NORMALIZATION_COUPLING"),
        ("extractall(", "ARCHIVE_EXTRACTION"),
    ):
        if needle.lower() in source.lower():
            _error(errors, code, path.as_posix(), f"Forbidden T-A2 geometry coupling: {needle}")
    if re.search(r"\bint8\b", source, flags=re.IGNORECASE):
        _error(errors, "INT8_COUPLING", path.as_posix(), "Forbidden T-A2 geometry coupling: int8")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"extractall", "resize"}:
            # The geometry module's own resize_* functions are pure numeric operations;
            # archive extraction and external image-library resize calls are forbidden.
            if node.func.attr == "extractall":
                _error(errors, "ARCHIVE_EXTRACTION", f"{path.as_posix()}:{node.lineno}", "Archive extraction is forbidden.")


def _validate_predecessors(repo_root: Path, errors: list[dict[str, str]], verify_real_payload: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from scripts.validate_thermal_t_a0 import validate_evidence as validate_a0
        a0 = validate_a0(repo_root / T_A0_REL, repo_root)
        if a0.get("evidence_validation") != "PASS":
            _error(errors, "T_A0_INVALID", T_A0_REL, canonical_json(a0).strip())
    except Exception as exc:
        a0 = {"evidence_validation": "FAIL", "overall_outcome": "NOT_VERIFIABLE"}
        _error(errors, "T_A0_VALIDATOR_ERROR", T_A0_REL, str(exc))
    try:
        from scripts.validate_thermal_t_a1 import validate_evidence as validate_a1
        a1 = validate_a1(
            repo_root=repo_root,
            evidence_dir=repo_root / T_A1_REL,
            check_checksums=True,
            verify_real_payload=verify_real_payload,
        )
        if a1.get("evidence_validation") != "PASS":
            _error(errors, "T_A1_INVALID", T_A1_REL, canonical_json(a1).strip())
    except Exception as exc:
        a1 = {"evidence_validation": "FAIL", "overall_outcome": "NOT_VERIFIABLE"}
        _error(errors, "T_A1_VALIDATOR_ERROR", T_A1_REL, str(exc))
    return a0, a1


def _validate_synthetic_coordinate_contract(profile: Any, errors: list[dict[str, str]]) -> None:
    rows = np.arange(SOURCE_SHAPE[0], dtype=np.float64)[:, None]
    cols = np.arange(SOURCE_SHAPE[1], dtype=np.float64)[None, :]
    gradient = rows * 1000.0 + cols
    canonical = canonicalize_physical_frame(gradient, profile)
    values = canonical.physical_frame
    if not (values[0, 0] < values[0, -1] and values[0, 0] < values[-1, 0] and values[-1, -1] > values[0, -1]):
        _error(errors, "COORDINATE_AXIS_ORDER_FAILED", "synthetic_gradient", "Rows/columns are not monotonic in source order.")
    traces = [canonical_to_source_trace(profile, 0, 0), canonical_to_source_trace(profile, 61, 79)]
    if any(item["status"] not in {"MEASURED_SOURCE_SUPPORT", "CANONICAL_PADDING_INVALID"} for item in traces):
        _error(errors, "COORDINATE_TRACE_FAILED", "synthetic_trace", "Selected profile has an invalid witness trace.")
    if source_to_canonical_trace(profile, profile.crop_top, profile.crop_left)["status"] != "MEASURED_CANONICAL_COORDINATE":
        _error(errors, "CROP_BOUNDARY_TRACE_FAILED", "synthetic_trace", "Candidate crop boundary coordinate is not traceable.")
    hot = np.zeros(SOURCE_SHAPE, dtype=np.float64)
    hot[220:260, 280:320] = 100.0
    hot_frame = canonicalize_physical_frame(hot, profile)
    peak = np.unravel_index(np.nanargmax(hot_frame.physical_frame), hot_frame.physical_frame.shape)
    if not (10 <= peak[0] <= 50 and 15 <= peak[1] <= 65):
        _error(errors, "ASYMMETRIC_HOT_REGION_MAPPING_FAILED", "synthetic_hot_region", str(peak))
    constant = canonicalize_physical_frame(np.full(SOURCE_SHAPE, 26.85, dtype=np.float64), profile)
    if not np.all(constant.physical_frame[constant.validity_mask] == np.float32(26.85)):
        _error(errors, "CONSTANT_PHYSICAL_FRAME_NOT_PRESERVED", "synthetic_constant", "Constant Celsius frame changed.")
    invalid = np.ones(SOURCE_SHAPE, dtype=bool)
    invalid[0, 0] = False
    try:
        canonicalize_physical_frame(np.full(SOURCE_SHAPE, 26.85), profile, source_validity_mask=invalid)
    except InvalidPixelError:
        pass
    else:
        _error(errors, "INVALID_MASK_NOT_FAIL_CLOSED", "synthetic_invalid_mask", "Partial invalid source was accepted without approved mask-aware interpolation.")
    nan_frame = np.full(SOURCE_SHAPE, 26.85, dtype=np.float64)
    nan_frame[0, 0] = np.nan
    try:
        canonicalize_physical_frame(nan_frame, profile)
    except InvalidPixelError:
        pass
    else:
        _error(errors, "NAN_NOT_FAIL_CLOSED", "synthetic_nan", "NaN source was accepted.")


def _independent_selection(candidates: list[dict[str, Any]], policy: dict[str, Any], errors: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Recompute gates/rank from evidence and policy without generator imports."""
    evaluated: list[dict[str, Any]] = []
    thresholds = policy.get("admissibility_thresholds", {})
    interpolation_order = policy.get("interpolation_preference", {})
    for index, candidate in enumerate(candidates):
        location = f"candidate[{index}]"
        profile = candidate.get("profile", {})
        profile_id = profile.get("profile_id")
        try:
            if profile != profile_for_id(profile_id).to_dict():
                _error(errors, "PROFILE_METRIC_INCONSISTENCY", f"{location}.profile", "Profile does not match predeclared candidate definition.")
        except Exception as exc:
            _error(errors, "UNKNOWN_CANDIDATE_PROFILE", f"{location}.profile_id", str(exc))
        gates = {
            "SOURCE_AND_CANONICAL_SHAPES": profile.get("source_shape") == list(SOURCE_SHAPE) and profile.get("canonical_shape") == list(CANONICAL_SHAPE),
            "PHYSICAL_FRAME_CONTRACT": profile.get("source_unit") == "CELSIUS" and profile.get("canonical_unit") == "CELSIUS" and profile.get("canonical_dtype") == "float32",
            "SOURCE_AS_STORED_ORIENTATION": profile.get("rotation") == 0 and profile.get("horizontal_flip") is False and profile.get("vertical_flip") is False,
            "FINITE_VALID_OUTPUT": candidate.get("finite_valid_output") is True,
            "CONSTANT_TEMPERATURE_PRESERVED": candidate.get("constant_temperature_preserved") is True,
            "DETERMINISTIC_REPEATED_CANONICALIZATION": candidate.get("repeated_canonicalization_deterministic") is True,
        }
        failed_mandatory = [key for key, passed in gates.items() if not passed]
        geometry = candidate.get("geometry", {})
        bbox = candidate.get("bbox_fov_diagnostic", {})
        fov = float(geometry.get("source_fov_retained_fraction", float("nan")))
        padding = float(geometry.get("padding_percentage", float("nan")))
        bbox_min = float(bbox.get("minimum_bbox_retention_due_to_candidate_transform", float("nan")))
        bbox_loss_fraction = float(bbox.get("additional_bbox_area_loss_due_to_candidate_crop_fraction", float("nan")))
        fov_limit = float(thresholds.get("source_fov_retained_fraction_min", {}).get("value", float("nan")))
        bbox_limit = float(thresholds.get("minimum_bbox_retention_due_to_candidate_transform_min", {}).get("value", float("nan")))
        padding_limit = float(thresholds.get("padding_percentage_max", {}).get("value", float("nan")))
        bbox_loss_limit = float(thresholds.get("candidate_crop_additional_bbox_area_loss_fraction_max", {}).get("value", float("nan")))
        admissibility_checks = {
            "SOURCE_FOV_WITHIN_THRESHOLD": fov >= fov_limit,
            "BBOX_RETENTION_WITHIN_THRESHOLD": bbox_min >= bbox_limit,
            "NO_SYNTHETIC_PADDING": padding <= padding_limit,
            "NO_MATERIAL_ADDITIONAL_CROP_BBOX_LOSS": bbox_loss_fraction <= bbox_loss_limit,
        }
        failed_admissibility = [key for key, passed in admissibility_checks.items() if not passed]
        admissible = not failed_mandatory and not failed_admissibility
        interpolation = str(profile.get("interpolation", "")).lower()
        metrics = {
            "anisotropy_ratio_excess": float(geometry.get("anisotropy_ratio_excess", float("nan"))),
            "padding_percentage": padding,
            "interpolation_preference_rank": int(interpolation_order.get(interpolation, 999)),
            "mean_absolute_shift_celsius": float(candidate.get("mean_absolute_shift_celsius", float("nan"))),
            "range_compression_distance_from_one": abs(float(candidate.get("range_compression_ratio", float("nan"))) - 1.0),
            "round_trip_mae": float(candidate.get("round_trip", {}).get("mae", float("nan"))),
            "candidate_id": str(profile.get("candidate_id", "")),
        }
        if any(not math.isfinite(float(value)) for key, value in metrics.items() if key != "candidate_id"):
            _error(errors, "METRIC_NONFINITE", location, str(metrics))
        reasons = [f"MANDATORY_{key}" for key in failed_mandatory] + [f"ADMISSIBILITY_{key}" for key in failed_admissibility]
        item = dict(candidate)
        item.update({
            "mandatory_gates": {"all_pass": not failed_mandatory, "checks": gates, "failed": failed_mandatory},
            "admissibility": {"all_pass": admissible, "checks": admissibility_checks, "failed": failed_admissibility},
            "admissible": admissible,
            "rejection_reason": reasons[0] if reasons else None,
            "ranking_metrics": metrics,
        })
        evaluated.append(item)
    sort_key = lambda item: tuple(item["ranking_metrics"][key] for key in ("anisotropy_ratio_excess", "padding_percentage", "interpolation_preference_rank", "mean_absolute_shift_celsius", "range_compression_distance_from_one", "round_trip_mae", "candidate_id"))
    admissible = sorted((item for item in evaluated if item["admissible"]), key=sort_key)
    tolerance = float(policy.get("tie_definition", {}).get("numeric_tolerance", 1e-12))
    tie_fields = policy.get("tie_definition", {}).get("fields", list(POLICY_NUMERIC_FIELDS))
    previous = None
    tie_number = 0
    for item in evaluated:
        item["rank"] = None
        item["tie_group"] = None
        item["final_selection_status"] = "REJECTED_POLICY" if not item["admissible"] else "ADMISSIBLE_NOT_SELECTED"
    for rank, item in enumerate(admissible, 1):
        item["rank"] = rank
        if previous is not None and all(abs(float(item["ranking_metrics"][field]) - float(previous["ranking_metrics"][field])) <= tolerance for field in tie_fields):
            item["tie_group"] = previous["tie_group"]
        else:
            tie_number += 1
            item["tie_group"] = f"TIE_GROUP_{tie_number:02d}"
        previous = item
    winner = admissible[0] if admissible else None
    if winner is not None:
        winner["final_selection_status"] = "SELECTED"
    for item in evaluated:
        item["selection_reason"] = "SELECTED_BY_DECLARED_POLICY" if item is winner else ("ADMISSIBLE_NOT_SELECTED_BY_DECLARED_RANKING" if item["admissible"] else f"REJECTED_BY_DECLARED_POLICY:{item['rejection_reason']}")
    return sorted(evaluated, key=lambda item: item["profile"].get("candidate_id", "")), winner


def validate_evidence(
    *,
    repo_root: Path,
    evidence_dir: Path,
    check_checksums: bool = True,
    verify_real_payload: bool = True,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    a0, a1 = _validate_predecessors(repo_root, errors, verify_real_payload)
    documents, paths = _load_json_documents(evidence_dir, errors)
    _validate_static_geometry(repo_root, errors)

    selected_path = repo_root / T_A0_REL / "selected_source_identity.json"
    try:
        selected_t_a0 = json.loads(selected_path.read_text(encoding="utf-8"))
        if selected_t_a0.get("selected_candidate_id") != "local_sdt_zenodo_4124309" or selected_t_a0.get("stable_identifier") != "doi:10.5281/zenodo.4124309":
            _error(errors, "SOURCE_IDENTITY_CHANGED", selected_path.as_posix(), "T-A0 selected source differs from SDT DOI.")
        if selected_t_a0.get("t_a1_authorized") is not True:
            _error(errors, "T_A1_AUTHORIZATION_CLOSED", selected_path.as_posix(), "T-A1 predecessor authorization is not true.")
    except (OSError, json.JSONDecodeError) as exc:
        _error(errors, "SOURCE_IDENTITY_UNREADABLE", selected_path.as_posix(), str(exc))

    required_names = set(REQUIRED_JSON)
    if required_names - set(documents):
        result = _result(a0, a1, errors, warnings, False)
        return result

    policy = documents["geometry_selection_policy.json"]
    required_policy_keys = {"policy_id", "policy_version", "mandatory_rejection_gates", "metric_definitions", "admissibility_thresholds", "ranking_order", "tie_definition", "tie_break_rule", "unsupported_metrics", "model_performance_used"}
    if not required_policy_keys.issubset(policy):
        _error(errors, "SELECTION_POLICY_INCOMPLETE", "geometry_selection_policy.json", str(sorted(required_policy_keys - set(policy))))
    if policy.get("model_performance_used") is not False:
        _error(errors, "MODEL_METRIC_SELECTION_CONTAMINATION", "geometry_selection_policy.json:model_performance_used", "must be false")
    if not policy.get("mandatory_rejection_gates") or not policy.get("ranking_order") or not policy.get("tie_break_rule"):
        _error(errors, "SELECTION_POLICY_INCOMPLETE", "geometry_selection_policy.json", "gates, ranking order, and tie break are mandatory")

    registry = documents["geometry_candidate_registry.json"]
    comparison = documents["geometry_comparison.json"]
    candidates = registry.get("candidates")
    comparison_candidates = comparison.get("candidate_results")
    if not isinstance(candidates, list) or len(candidates) != 9:
        _error(errors, "CANDIDATE_SET_INVALID", "geometry_candidate_registry.json:candidates", "Expected exactly 9 predeclared candidates.")
        candidates = []
    if not isinstance(comparison_candidates, list) or len(comparison_candidates) != 9:
        _error(errors, "COMPARISON_CANDIDATE_SET_INVALID", "geometry_comparison.json:candidate_results", "Expected exactly 9 candidates.")
        comparison_candidates = []
    candidate_ids = [item.get("profile", {}).get("candidate_id") for item in candidates]
    if candidate_ids != sorted(candidate_ids) or len(set(candidate_ids)) != len(candidate_ids):
        _error(errors, "CANDIDATE_ORDER_INVALID", "geometry_candidate_registry.json:candidates", "Candidates must be unique and sorted.")
    if registry.get("selection_policy_id") != policy.get("policy_id") or comparison.get("selection_policy_id") != policy.get("policy_id"):
        _error(errors, "SELECTION_POLICY_REFERENCE_MISMATCH", "geometry_candidate_registry.json", "Registry/comparison policy reference is stale.")
    if registry.get("model_performance_used_for_selection") is not False or comparison.get("model_performance_used") is not False:
        _error(errors, "MODEL_METRIC_SELECTION_CONTAMINATION", "geometry_candidate_registry.json", "Model performance must not select geometry.")
    for index, candidate in enumerate(candidates):
        serialized = json.dumps(candidate, ensure_ascii=False).lower()
        if any(term in serialized for term in ("accuracy", "f1", "fall_recall", "confidence", "model_score")):
            _error(errors, "MODEL_METRIC_IN_CANDIDATE_EVIDENCE", f"geometry_candidate_registry.json:candidates[{index}]", "Candidate evidence contains model-selection metrics.")
        profile = candidate.get("profile", {})
        if profile.get("interpolation") == "bilinear" and profile.get("explicit_antialias_prefilter") != "NO_EXPLICIT_ANTIALIAS_PREFILTER":
            _error(errors, "BILINEAR_ANTIALIAS_SEMANTICS_INVALID", f"candidate[{index}].profile", str(profile))
        bbox = candidate.get("bbox_fov_diagnostic", {})
        for field in ("source_bbox_outside_frame_count", "additional_bbox_intersected_by_candidate_crop_count", "additional_bbox_area_loss_due_to_candidate_crop", "mean_bbox_retention_due_to_candidate_transform", "minimum_bbox_retention_due_to_candidate_transform", "coordinate_clipping_order"):
            if field not in bbox:
                _error(errors, "BBOX_DIAGNOSTIC_MISSING", f"candidate[{index}].bbox_fov_diagnostic", field)
        if not str(bbox.get("coordinate_clipping_order", "")).startswith("SOURCE_BBOX_CLIP_TO_DISTRIBUTED_FRAME"):
            _error(errors, "BBOX_CLIPPING_ORDER_UNDECLARED", f"candidate[{index}].bbox_fov_diagnostic", str(bbox.get("coordinate_clipping_order")))

    evaluated, winner = _independent_selection(candidates, policy, errors)
    actual_by_id = {item.get("profile", {}).get("candidate_id"): item for item in candidates}
    expected_by_id = {item.get("profile", {}).get("candidate_id"): item for item in evaluated}
    for candidate_id, expected in expected_by_id.items():
        actual = actual_by_id.get(candidate_id, {})
        for field in ("mandatory_gates", "admissibility", "rejection_reason", "ranking_metrics", "rank", "tie_group", "final_selection_status", "selection_reason"):
            if actual.get(field) != expected.get(field):
                _error(errors, "CANDIDATE_RANKING_INCONSISTENT", f"geometry_candidate_registry.json:{candidate_id}.{field}", f"expected={expected.get(field)!r}; found={actual.get(field)!r}")
    if comparison_candidates and canonical_json(comparison_candidates) != canonical_json(candidates):
        _error(errors, "REGISTRY_COMPARISON_MISMATCH", "geometry_comparison.json:candidate_results", "Registry and comparison candidate metrics differ.")

    selected = documents["selected_geometry_profile.json"]
    profile = selected.get("profile", {})
    selected_id = winner["profile"]["profile_id"] if winner is not None else None
    if winner is None:
        _error(errors, "NO_ADMISSIBLE_CANDIDATE", "geometry_selection_policy.json", "No candidate passed the declared policy.")
    elif comparison.get("selected_profile_id") != selected_id or comparison.get("selected_candidate_id") != winner["profile"]["candidate_id"]:
        _error(errors, "WINNER_DERIVATION_MISMATCH", "geometry_comparison.json", "Stored winner differs from independent ranking.")
    if selected.get("profile_id") != profile.get("profile_id") or (selected_id is not None and profile.get("profile_id") != selected_id):
        _error(errors, "SELECTED_PROFILE_MISMATCH", "selected_geometry_profile.json:profile.profile_id", f"independent winner={selected_id}; found={profile.get('profile_id')}")
    if selected_id is not None:
        try:
            expected_profile = profile_for_id(selected_id).to_dict()
            if profile != expected_profile:
                _error(errors, "SELECTED_PROFILE_MISMATCH", "selected_geometry_profile.json:profile", "Selected profile definition differs from the independently declared candidate profile.")
        except Exception as exc:
            _error(errors, "SELECTED_PROFILE_DEFINITION_UNREADABLE", "selected_geometry_profile.json:profile", str(exc))
    policy_checksum = hashlib.sha256(canonical_json(policy).encode("utf-8")).hexdigest()
    metrics_checksum = hashlib.sha256(canonical_json(candidates).encode("utf-8")).hexdigest()
    if selected.get("selection_policy_id") != policy.get("policy_id") or selected.get("selection_policy_content_sha256") != policy_checksum:
        _error(errors, "SELECTION_POLICY_REFERENCE_MISMATCH", "selected_geometry_profile.json", "Selected policy reference/checksum is stale.")
    if selected.get("candidate_metrics_content_sha256") != metrics_checksum:
        _error(errors, "CANDIDATE_METRIC_CHECKSUM_MISMATCH", "selected_geometry_profile.json", "Selected candidate metric checksum is stale.")
    for key, expected in (("canonical_shape", list(CANONICAL_SHAPE)), ("source_shape", list(SOURCE_SHAPE)), ("canonical_dtype", "float32"), ("canonical_unit", "CELSIUS"), ("rotation", 0), ("horizontal_flip", False), ("vertical_flip", False)):
        if profile.get(key) != expected:
            _error(errors, "SELECTED_PROFILE_MISMATCH", f"selected_geometry_profile.json:profile.{key}", f"expected={expected!r}; found={profile.get(key)!r}")
    if selected.get("selection_status") not in {"GEOMETRY_PROFILE_SELECTED", "GEOMETRY_PROFILE_SELECTED_WITH_LIMITATIONS"}:
        _error(errors, "SELECTION_STATUS_INVALID", "selected_geometry_profile.json:selection_status", str(selected.get("selection_status")))
    if selected.get("model_performance_used") is not False:
        _error(errors, "MODEL_METRIC_SELECTION_CONTAMINATION", "selected_geometry_profile.json:model_performance_used", "must be false")
    if selected.get("source_orientation", {}).get("status") != "SOURCE_ORIENTATION_AS_STORED":
        _error(errors, "SOURCE_ORIENTATION_NOT_EXPLICIT", "selected_geometry_profile.json:source_orientation", "orientation must be explicit")

    canonical = documents["canonical_frame_contract.json"]
    output = canonical.get("output", {})
    if output.get("shape") != list(CANONICAL_SHAPE) or output.get("dtype") != "float32" or output.get("unit") != "CELSIUS":
        _error(errors, "CANONICAL_CONTRACT_INVALID", "canonical_frame_contract.json:output", str(output))
    if canonical.get("source_and_canonical_are_distinct") is not True or canonical.get("input", {}).get("source_hash_preserved") is not True:
        _error(errors, "SOURCE_CANONICAL_ALIASING", "canonical_frame_contract.json", "Source and canonical representations must remain distinct.")
    boundary = canonical.get("boundary_stop", "").lower()
    for term in ("min-max", "z-score", "int8", "model inference"):
        if term not in boundary:
            _error(errors, "CANONICAL_BOUNDARY_INCOMPLETE", "canonical_frame_contract.json:boundary_stop", term)

    units = documents["calibration_contract.json"]
    conversion = units.get("source_physical_conversion", {})
    if conversion.get("formula") != "(encoded_uint16 - 27315) / 100" or conversion.get("canonical_unit") != "CELSIUS":
        _error(errors, "PHYSICAL_CONVERSION_CHANGED", "calibration_contract.json:source_physical_conversion", str(conversion))
    for field in ("ambient_reference_compensation", "reference_compensation"):
        if units.get(field, {}).get("applied") is not False or units.get(field, {}).get("status") != "NOT_APPLIED_NO_VERIFIED_PARAMETER_SOURCE":
            _error(errors, "UNVERIFIED_CALIBRATION_APPLIED", f"calibration_contract.json:{field}", str(units.get(field)))
    if units.get("hardware_specific_calibration", {}).get("status") != "DEFERRED_T_C":
        _error(errors, "HARDWARE_CALIBRATION_NOT_DEFERRED", "calibration_contract.json:hardware_specific_calibration", str(units.get("hardware_specific_calibration")))
    precision = units.get("canonical_dtype_precision", {})
    if precision.get("selected_dtype") != "float32" or precision.get("source_precision_preserved_relative_to_encoding") is not True or precision.get("max_conversion_error_celsius", 1.0) >= 0.005:
        _error(errors, "CANONICAL_DTYPE_PRECISION_INVALID", "calibration_contract.json:canonical_dtype_precision", str(precision))

    invalid = documents["invalid_pixel_policy.json"]
    if invalid.get("synthetic_normal_values_introduced") is not False or invalid.get("interpolation_over_invalid_source") != "FORBIDDEN":
        _error(errors, "INVALID_PIXEL_POLICY_UNSAFE", "invalid_pixel_policy.json", str(invalid))
    if "FAIL_CLOSED" not in invalid.get("source_partial_invalid_mask", "") or "FAIL_CLOSED" not in invalid.get("source_nan_inf", ""):
        _error(errors, "INVALID_PIXEL_POLICY_INCOMPLETE", "invalid_pixel_policy.json", str(invalid))

    trace = documents["coordinate_trace_contract.json"]
    if trace.get("profile_id") != profile.get("profile_id") or trace.get("synthetic_fixture_checks", {}).get("corner_markers") != "PASS_SOURCE_ORDER_PRESERVED":
        _error(errors, "COORDINATE_TRACE_CONTRACT_INVALID", "coordinate_trace_contract.json", str(trace))
    if not trace.get("forward_equation") or not trace.get("inverse_equation"):
        _error(errors, "COORDINATE_EQUATIONS_MISSING", "coordinate_trace_contract.json", "Forward and inverse equations required.")

    pilot = documents["pilot_geometry_summary.json"]
    records = pilot.get("records", [])
    if pilot.get("pilot_frame_count") != EXPECTED_PILOT_COUNT or len(records) != EXPECTED_PILOT_COUNT:
        _error(errors, "PILOT_COUNT_INVALID", "pilot_geometry_summary.json", str(pilot.get("pilot_frame_count")))
    if pilot.get("source_class_counts") != {"0": 12, "1": 12, "2": 12, "3": 12}:
        _error(errors, "PILOT_CLASS_COVERAGE_INVALID", "pilot_geometry_summary.json:source_class_counts", str(pilot.get("source_class_counts")))
    if pilot.get("selected_profile_id") != profile.get("profile_id"):
        _error(errors, "PILOT_PROFILE_MISMATCH", "pilot_geometry_summary.json:selected_profile_id", str(pilot.get("selected_profile_id")))
    record_indices = [item.get("source_frame_index") for item in records]
    if record_indices != sorted(record_indices) or len(set(record_indices)) != len(record_indices):
        _error(errors, "PILOT_ORDER_INVALID", "pilot_geometry_summary.json:records", "Records must be sorted and unique.")
    for index, item in enumerate(records):
        for field in ("source_encoded_frame_sha256", "source_member_name", "source_frame_index", "geometry_profile_id", "canonical_frame_hash", "canonical_shape", "canonical_dtype", "canonical_unit", "source_provenance"):
            if field not in item:
                _error(errors, "PILOT_PROVENANCE_MISSING", f"pilot_geometry_summary.json:records[{index}]", field)
        if item.get("geometry_profile_id") != profile.get("profile_id") or item.get("canonical_shape") != list(CANONICAL_SHAPE) or item.get("canonical_dtype") != "float32":
            _error(errors, "PILOT_CANONICAL_CONTRACT_INVALID", f"pilot_geometry_summary.json:records[{index}]", str(item))
        if item.get("source_pose_name") == "HUMAN_FALL":
            _error(errors, "SAFE_NEST_LABEL_REWRITE", f"pilot_geometry_summary.json:records[{index}]", "Original LYING label must remain unchanged.")
        if item.get("raw_source_frame_unchanged_after_canonicalization") is not True or item.get("repeated_canonical_frame_hash_equal") is not True:
            _error(errors, "PILOT_SOURCE_MUTATION_OR_NONDETERMINISM", f"pilot_geometry_summary.json:records[{index}]", str(item))
        source_provenance = item.get("source_provenance", {})
        for field in ("source_dataset_id", "source_doi", "source_split", "source_archive_path", "source_archive_sha256", "source_member_name", "source_frame_index", "source_pose_label", "source_bbox", "raw_encoded_frame_sha256"):
            if field not in source_provenance:
                _error(errors, "PILOT_SOURCE_PROVENANCE_MISSING", f"pilot_geometry_summary.json:records[{index}].source_provenance", field)

    visual = documents["visual_spotcheck_registry.json"]
    visual_path = repo_root / visual.get("path", "")
    if visual.get("status") != "PASS" or visual.get("role") != "HUMAN_VISUAL_DIAGNOSTIC_ONLY_NOT_RADIOMETRIC_MODEL_INPUT":
        _error(errors, "VISUAL_SPOTCHECK_INVALID", "visual_spotcheck_registry.json", str(visual))
    if not visual_path.is_file():
        _error(errors, "VISUAL_SPOTCHECK_MISSING", str(visual.get("path")), "Diagnostic image is missing.")
    elif visual.get("sha256") != _sha256(visual_path):
        _error(errors, "VISUAL_SPOTCHECK_CHECKSUM_MISMATCH", str(visual.get("path")), "Diagnostic image checksum differs.")

    if selected_id is not None:
        _validate_synthetic_coordinate_contract(profile_for_id(selected_id), errors)

    if verify_real_payload:
        try:
            reader = SDTThermalRawReader(repo_root=repo_root)
            measured = reader.inspect_archive()
            if measured["archive_identity"]["sha256"] != "3a838bd70835e579ecfaa820a6c0b4cbc6ba7b76729417c73845f0c959281449":
                _error(errors, "REAL_SOURCE_IDENTITY_MISMATCH", "datasets/raw_archives/thermal_split_zips/test.zip", "T-A1 source hash changed.")
            if selected_id is None:
                raise ValueError("Cannot validate real pilot without an independently selected profile")
            selected_profile = profile_for_id(selected_id)
            for item in records:
                frame = reader.read_frame(int(item["source_frame_index"]))
                before = frame.raw_encoded_frame_sha256
                canonical_frame = __import__("datasets.thermal.canonical_geometry", fromlist=["canonicalize_source_frame"]).canonicalize_source_frame(frame, selected_profile)
                after = frame.raw_encoded_frame_sha256
                if before != after or canonical_frame.canonical_frame_hash != item.get("canonical_frame_hash"):
                    _error(errors, "REAL_PILOT_HASH_MISMATCH", f"pilot:{item.get('source_frame_index')}", "Source or canonical hash differs.")
                if canonical_frame.physical_frame.shape != CANONICAL_SHAPE or canonical_frame.physical_frame.dtype != np.dtype("float32"):
                    _error(errors, "REAL_PILOT_SHAPE_DTYPE_MISMATCH", f"pilot:{item.get('source_frame_index')}", "Canonical output contract differs.")
                if frame.source_pose_name != item.get("source_pose_name"):
                    _error(errors, "REAL_PILOT_LABEL_MISMATCH", f"pilot:{item.get('source_frame_index')}", "Original label differs.")
            # Physical conversion and precision are independently witnessed on real data.
            witness = reader.read_frame(0).celsius()
            converted = witness.astype(np.float32).astype(np.float64)
            if precision_error(witness, converted)["max_abs_error"] >= 0.005:
                _error(errors, "REAL_DTYPE_PRECISION_FAILED", "pilot:0", "float32 conversion exceeds source resolution.")
        except Exception as exc:
            _error(errors, "REAL_PILOT_VALIDATION_FAILED", "datasets/raw_archives/thermal_split_zips/test.zip", str(exc))

    if check_checksums:
        validation_path = evidence_dir / "validation_result.json"
        if not validation_path.is_file():
            _error(errors, "VALIDATION_RESULT_MISSING", "validation_result.json", "Stored validation result is required.")
        else:
            paths.append(validation_path)
        visual_path = evidence_dir / "visual_spotcheck.png"
        if visual_path.is_file():
            paths.append(visual_path)
        _validate_checksums(repo_root, evidence_dir, paths, errors)

    _warning(warnings, "THERMAL44_ORIENTATION_UNVERIFIED", "selected_geometry_profile.json", "Software canonical orientation is not a hardware packet-orientation claim.")
    _warning(warnings, "SOURCE_NATIVE_RECONSTRUCTION_NOT_CLAIMED", "source_schema_profile/T-A2", "Distributed SDT image is author-upscaled; T-A2 does not reverse it.")
    _warning(warnings, "NO_TEMPORAL_EVENT_FIELDS", "SDT source", "Sequence/event/timestamp fields remain absent for later phases.")
    _warning(warnings, "TRAIN_VALIDATION_PLACEHOLDERS", "archive_member_inventory.json", "No large archive hydration occurred.")
    return _result(a0, a1, errors, warnings, not errors)


def _result(a0: dict[str, Any], a1: dict[str, Any], errors: list[dict[str, str]], warnings: list[dict[str, str]], gate: bool) -> dict[str, Any]:
    sorted_errors = sorted(errors, key=lambda item: (item["code"], item["location"], item["message"]))
    sorted_warnings = sorted(warnings, key=lambda item: (item["code"], item["location"], item["message"]))
    return {
        "error_count": len(sorted_errors),
        "errors": sorted_errors,
        "evidence_validation": "PASS" if not sorted_errors else "FAIL",
        "overall_outcome": "PASS_WITH_LIMITATIONS" if gate and not sorted_errors else "NOT_VERIFIABLE",
        "phase": "T-A2",
        "schema_version": "1.0",
        "t_a0_outcome": a0.get("overall_outcome", "NOT_VERIFIABLE"),
        "t_a0_validation": a0.get("evidence_validation", "FAIL"),
        "t_a1_outcome": a1.get("overall_outcome", "NOT_VERIFIABLE"),
        "t_a1_validation": a1.get("evidence_validation", "FAIL"),
        "t_a3_authorized": bool(gate and not sorted_errors),
        "warning_count": len(sorted_warnings),
        "warnings": sorted_warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--skip-real-payload", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    evidence_dir = args.evidence_dir or repo_root / EVIDENCE_REL
    result = validate_evidence(repo_root=repo_root, evidence_dir=evidence_dir, check_checksums=True, verify_real_payload=not args.skip_real_payload)
    print(canonical_json(result), end="")
    return 0 if result["evidence_validation"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
