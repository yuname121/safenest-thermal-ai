#!/usr/bin/env python3
"""Fail-closed temporal policy for the selected SDT Thermal source.

T-A3 deliberately treats the SDT archive as a static, frame-level source.  The
archive has deterministic member/index provenance, but no verified acquisition
clock, sequence grouping, or fall-event annotations.  This module keeps those
concepts separate so a later phase cannot silently turn an integer file index
or a posture label into temporal evidence.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping, Sequence


TEMPORAL_POLICY_ID = "THERMAL_TEMPORAL_POLICY_001"
TEMPORAL_POLICY_VERSION = "1.0"
SDT_DATASET_ID = "local_sdt_zenodo_4124309"
SDT_DOI = "doi:10.5281/zenodo.4124309"
SDT_SOURCE_SPLIT = "test"
SDT_ARCHIVE_PATH = "datasets/raw_archives/thermal_split_zips/test.zip"
SDT_ARCHIVE_SHA256 = "3a838bd70835e579ecfaa820a6c0b4cbc6ba7b76729417c73845f0c959281449"
T_A2_PROFILE_ID = "G1_FIXED_ASPECT_CROP_BILINEAR"
T_A2_CANONICAL_SHAPE = (62, 80)
T_A2_CANONICAL_DTYPE = "float32"
T_A2_CANONICAL_UNIT = "CELSIUS"
POSE_NAMES = {0: "LYING", 1: "SITTING", 2: "STANDING", 3: "EMPTY_ROOM"}

UNKNOWN_NOT_VERIFIABLE = "UNKNOWN_NOT_VERIFIABLE"
TEMPORAL_METADATA_FORBIDDEN_KEYS = frozenset(
    {
        "timestamp",
        "timestamp_s",
        "timestamp_ms",
        "fps",
        "frame_rate",
        "sample_period_s",
        "sequence_id",
        "session_id",
        "recording_id",
        "subject_id",
        "event_id",
        "event_start",
        "event_end",
        "window_start",
        "window_end",
        "window_duration_s",
        "window_frame_count",
    }
)


class TemporalPolicyError(ValueError):
    """Base error for invalid or unsupported T-A3 temporal requests."""

    code = "THERMAL_TEMPORAL_POLICY_ERROR"

    def __init__(self, message: str) -> None:
        self.detail = message
        super().__init__(f"{self.code}: {message}")


class FabricatedTemporalMetadataError(TemporalPolicyError):
    code = "FABRICATED_TEMPORAL_METADATA"


class TemporalSequenceUnavailableError(TemporalPolicyError):
    code = "TEMPORAL_SEQUENCE_UNAVAILABLE"


class TemporalEventUnavailableError(TemporalPolicyError):
    code = "SOURCE_EVENT_BOUNDARY_UNAVAILABLE"


class TemporalWindowUnavailableError(TemporalPolicyError):
    code = "WINDOW_CONSTRUCTION_NOT_ALLOWED"


def _copy(value: Any) -> Any:
    return deepcopy(value)


def temporal_policy_profile() -> dict[str, Any]:
    """Return the immutable, model-independent T-A3 policy profile."""

    return _copy(
        {
            "phase": "T-A3",
            "schema_version": "1.0",
            "policy_id": TEMPORAL_POLICY_ID,
            "policy_version": TEMPORAL_POLICY_VERSION,
            "source": {
                "dataset_id": SDT_DATASET_ID,
                "doi": SDT_DOI,
                "source_split": SDT_SOURCE_SPLIT,
                "archive_path": SDT_ARCHIVE_PATH,
                "archive_sha256": SDT_ARCHIVE_SHA256,
            },
            "source_frame_semantics": {
                "frame_index_role": "PROVENANCE_IDENTIFIER_ONLY",
                "member_name_role": "PROVENANCE_IDENTIFIER_ONLY",
                "index_is_timestamp": False,
                "filename_order_is_temporal_order": False,
                "original_labels": {
                    "0": "LYING",
                    "1": "SITTING",
                    "2": "STANDING",
                    "3": "EMPTY_ROOM",
                },
                "lying_is_fall_event": False,
            },
            "temporal_evidence": {
                "source_fps": {
                    "status": "SOURCE_FRAME_RATE_NOT_VERIFIABLE",
                    "value": "NOT_VERIFIABLE",
                    "evidence": "No SDT-specific distributed acquisition rate or timestamp cadence is documented or stored.",
                },
                "timestamps": {
                    "status": "SOURCE_TIMESTAMP_ABSENT",
                    "reliability": "NOT_APPLICABLE",
                    "fields": "ABSENT",
                },
                "temporal_ordering": {
                    "status": "TEMPORAL_ORDER_NOT_VERIFIABLE",
                    "structural_index_order": "MEASURED_ONLY_AS_ARCHIVE_PROVENANCE",
                },
                "sequence_identity": {
                    "status": "SOURCE_SEQUENCE_STATUS_ABSENT",
                    "identifiers": ["sequence_id", "recording_id", "clip_id", "trial_id"],
                },
                "session_identity": {
                    "status": "SOURCE_SESSION_STATUS_ABSENT",
                    "identifiers": ["session_id"],
                },
                "event_identity": {
                    "status": "SOURCE_EVENT_STATUS_ABSENT",
                    "identifiers": ["event_id"],
                },
                "fall_boundaries": {
                    "status": "FALL_EVENT_BOUNDARY_NOT_VERIFIABLE",
                    "onset": "NOT_VERIFIABLE",
                    "end": "NOT_VERIFIABLE",
                    "impact": "NOT_VERIFIABLE",
                },
            },
            "capabilities": {
                "FRAME_LEVEL": {
                    "supported": True,
                    "status": "SUPPORTED",
                    "source_evidence": ["T-A1 reader and frame provenance contract", "T-A2 canonical-frame pilot"],
                    "required_identifiers": [
                        "source_dataset_id",
                        "source_split",
                        "source_archive_path",
                        "source_archive_sha256",
                        "source_member_name",
                        "source_frame_index",
                        "source_pose_label",
                        "source_bbox",
                        "raw_encoded_frame_sha256",
                        "canonical_frame_hash",
                        "t_a2_geometry_profile_id",
                    ],
                    "actual_identifiers": [
                        "dataset_id",
                        "split",
                        "archive_path",
                        "archive_sha256",
                        "member_name",
                        "frame_index",
                        "pose_label",
                        "bbox",
                        "raw_hash",
                        "canonical_hash",
                        "geometry_profile_id",
                    ],
                    "temporal_ordering_available": False,
                    "timestamp_available": False,
                    "grouping_available": False,
                    "safe_to_construct": True,
                    "reason": "One verified image_t member is one frame-level sample; no temporal claim is attached.",
                },
                "SEQUENCE_LEVEL": {
                    "supported": False,
                    "status": "TEMPORAL_SEQUENCE_NOT_VERIFIABLE",
                    "source_evidence": ["T-A1 source schema has no sequence/recording identifier", "SDT official documentation does not document clips or cadence"],
                    "required_identifiers": ["sequence_id or recording_id", "verified_temporal_order", "timestamp_or_verified_cadence"],
                    "actual_identifiers": [],
                    "temporal_ordering_available": False,
                    "timestamp_available": False,
                    "grouping_available": False,
                    "safe_to_construct": False,
                    "reason": "Archive indices and neighboring filenames are provenance only, not verified sequence continuity.",
                },
                "EVENT_LEVEL": {
                    "supported": False,
                    "status": "TEMPORAL_EVENT_NOT_VERIFIABLE",
                    "source_evidence": ["T-A1 source schema has no event identifier or boundaries", "official SDT labels are posture labels"],
                    "required_identifiers": ["sequence_id", "event_id or defensible event annotation", "onset", "end", "pre_post_context"],
                    "actual_identifiers": [],
                    "temporal_ordering_available": False,
                    "timestamp_available": False,
                    "grouping_available": False,
                    "safe_to_construct": False,
                    "reason": "LYING is a posture label and does not identify a fall transition, onset, impact, or end.",
                },
                "WINDOW_LEVEL": {
                    "supported": False,
                    "status": "WINDOWING_NOT_APPLICABLE_TO_SOURCE",
                    "source_evidence": ["No verified SDT timeline, cadence, sequence, or window unit"],
                    "required_identifiers": ["verified_timeline", "window_duration_or_frame_count", "gap_policy", "duplicate_policy", "boundary_policy"],
                    "actual_identifiers": [],
                    "temporal_ordering_available": False,
                    "timestamp_available": False,
                    "grouping_available": False,
                    "safe_to_construct": False,
                    "reason": "A frame count or duration would have no verified temporal unit for this source.",
                },
            },
            "gap_drop_duplicate_policy": {
                "archive_index_gap": "SOURCE_MEMBER_INDEX_GAP",
                "temporal_dropped_frame": "TEMPORAL_DROPPED_FRAME_NOT_VERIFIABLE",
                "large_temporal_gap": "TEMPORAL_GAP_NOT_VERIFIABLE",
                "duplicate_member_index": "STRUCTURAL_SOURCE_FAILURE",
                "duplicate_member_name": "STRUCTURAL_SOURCE_FAILURE",
                "exact_duplicate_content": "PRESERVE_BOTH_PROVENANCE_RECORDS_AND_FLAG_DUPLICATE_CONTENT; DO_NOT_INFER_ADJACENCY",
                "near_duplicate_content": "DO_NOT_INFER_TEMPORAL_ADJACENCY; FULL_AUDIT_DEFERRED_T_A6",
            },
            "official_split_policy": {
                "source_split_preserved": True,
                "current_source_split": "test",
                "safenest_split_created": False,
                "safe_nest_split_owner": "T-A5",
            },
            "model_performance_used": False,
            "unsupported_inference": [
                "FRAME_INDEX_TO_TIMESTAMP",
                "FRAME_INDEX_TO_FPS",
                "FILENAME_ORDER_TO_TEMPORAL_ORDER",
                "POSE_RUN_TO_SEQUENCE",
                "LYING_TO_FALL_EVENT",
                "IMAGE_SIMILARITY_TO_SEQUENCE",
                "DEPTH_SIMILARITY_TO_SEQUENCE",
                "MODEL_OUTPUT_TO_TEMPORAL_ORDER",
                "SYNTHETIC_TIMESTAMP_OR_WINDOW",
            ],
            "downstream_boundary": "T-A3 freezes source temporal evidence only; T-A4 owns original-label ambiguity and T-A5 owns grouping/splits.",
        }
    )


def validate_temporal_policy_profile(profile: Mapping[str, Any]) -> None:
    """Validate the immutable capability profile without touching payloads."""

    if profile.get("phase") != "T-A3" or profile.get("policy_id") != TEMPORAL_POLICY_ID:
        raise TemporalPolicyError("unexpected T-A3 temporal policy identity")
    source = profile.get("source")
    if not isinstance(source, Mapping):
        raise TemporalPolicyError("temporal policy source block is missing")
    for key, expected in {
        "dataset_id": SDT_DATASET_ID,
        "doi": SDT_DOI,
        "source_split": SDT_SOURCE_SPLIT,
        "archive_path": SDT_ARCHIVE_PATH,
        "archive_sha256": SDT_ARCHIVE_SHA256,
    }.items():
        if source.get(key) != expected:
            raise TemporalPolicyError(f"temporal policy source {key} mismatch")
    capabilities = profile.get("capabilities")
    if not isinstance(capabilities, Mapping):
        raise TemporalPolicyError("temporal capability block is missing")
    expected_status = {
        "FRAME_LEVEL": (True, "SUPPORTED"),
        "SEQUENCE_LEVEL": (False, "TEMPORAL_SEQUENCE_NOT_VERIFIABLE"),
        "EVENT_LEVEL": (False, "TEMPORAL_EVENT_NOT_VERIFIABLE"),
        "WINDOW_LEVEL": (False, "WINDOWING_NOT_APPLICABLE_TO_SOURCE"),
    }
    for level, (supported, status) in expected_status.items():
        block = capabilities.get(level)
        if not isinstance(block, Mapping) or block.get("supported") is not supported or block.get("status") != status:
            raise TemporalPolicyError(f"invalid capability block for {level}")
        if block.get("safe_to_construct") is not supported:
            raise TemporalPolicyError(f"invalid safe_to_construct for {level}")
    semantics = profile.get("source_frame_semantics", {})
    if semantics.get("index_is_timestamp") is not False or semantics.get("filename_order_is_temporal_order") is not False:
        raise FabricatedTemporalMetadataError("source indices/order cannot be temporal evidence")
    if semantics.get("lying_is_fall_event") is not False:
        raise FabricatedTemporalMetadataError("LYING cannot be promoted to a fall event")


def validate_temporal_request(metadata: Mapping[str, Any]) -> None:
    """Reject fabricated temporal fields in a frame or constructor request."""

    unexpected = sorted(TEMPORAL_METADATA_FORBIDDEN_KEYS.intersection(metadata))
    if unexpected:
        raise FabricatedTemporalMetadataError(
            "unsupported temporal metadata fields: " + ", ".join(unexpected)
        )
    if metadata.get("frame_index_as_timestamp") is True:
        raise FabricatedTemporalMetadataError("frame index cannot be promoted to timestamp")
    if metadata.get("filename_order_is_temporal_order") is True:
        raise FabricatedTemporalMetadataError("filename order is not verified temporal order")
    if metadata.get("lying_is_fall_event") is True:
        raise FabricatedTemporalMetadataError("LYING is not a verified fall event")


def _unsupported_result(level: str, reasons: Sequence[str]) -> dict[str, Any]:
    return {
        "level": level,
        "eligible": False,
        "status": {
            "SEQUENCE_LEVEL": "TEMPORAL_SEQUENCE_NOT_VERIFIABLE",
            "EVENT_LEVEL": "TEMPORAL_EVENT_NOT_VERIFIABLE",
            "WINDOW_LEVEL": "WINDOWING_NOT_APPLICABLE_TO_SOURCE",
        }[level],
        "reasons": list(reasons),
        "safe_to_construct": False,
        "failure_code": {
            "SEQUENCE_LEVEL": TemporalSequenceUnavailableError.code,
            "EVENT_LEVEL": TemporalEventUnavailableError.code,
            "WINDOW_LEVEL": TemporalWindowUnavailableError.code,
        }[level],
    }


def evaluate_sequence_construction(metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if metadata is not None:
        validate_temporal_request(metadata)
    return _unsupported_result(
        "SEQUENCE_LEVEL",
        [
            "SOURCE_SEQUENCE_STATUS_ABSENT",
            "TEMPORAL_ORDER_NOT_VERIFIABLE",
            "SOURCE_TIMESTAMP_STATUS_ABSENT",
            "NO_VERIFIED_GAP_POLICY_FOR_ACQUISITION_TIMELINE",
        ],
    )


def evaluate_event_construction(metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if metadata is not None:
        validate_temporal_request(metadata)
    return _unsupported_result(
        "EVENT_LEVEL",
        [
            "SOURCE_EVENT_STATUS_ABSENT",
            "FALL_EVENT_BOUNDARY_NOT_VERIFIABLE",
            "LYING_IS_POSTURE_ONLY",
            "NO_PRE_DURING_POST_EVENT_CONTEXT",
        ],
    )


def evaluate_window_construction(metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if metadata is not None:
        validate_temporal_request(metadata)
    return _unsupported_result(
        "WINDOW_LEVEL",
        [
            "SOURCE_TIMESTAMP_STATUS_ABSENT",
            "SOURCE_SEQUENCE_STATUS_ABSENT",
            "SOURCE_FRAME_RATE_NOT_VERIFIABLE",
            "WINDOW_DURATION_AND_FRAME_COUNT_HAVE_NO_VERIFIED_TIME_UNIT",
        ],
    )


def construct_sequence(frames: Sequence[Mapping[str, Any]], **metadata: Any) -> None:
    validate_temporal_request(metadata)
    raise TemporalSequenceUnavailableError(
        f"SDT source sequence construction is not verifiable for {len(frames)} frame records"
    )


def construct_event(frames: Sequence[Mapping[str, Any]], **metadata: Any) -> None:
    validate_temporal_request(metadata)
    raise TemporalEventUnavailableError(
        f"SDT source event construction is not verifiable for {len(frames)} frame records"
    )


def construct_window(frames: Sequence[Mapping[str, Any]], **metadata: Any) -> None:
    validate_temporal_request(metadata)
    raise TemporalWindowUnavailableError(
        f"SDT source window construction is not allowed for {len(frames)} frame records"
    )


def frame_sample_from_provenance(
    provenance: Mapping[str, Any],
    *,
    canonical_frame_hash: str,
    geometry_profile_id: str = T_A2_PROFILE_ID,
) -> dict[str, Any]:
    """Build a frame-level record without adding a temporal identifier."""

    validate_temporal_request(provenance)
    required_source = {
        "source_dataset_id": SDT_DATASET_ID,
        "source_doi": SDT_DOI,
        "source_split": SDT_SOURCE_SPLIT,
        "source_archive_path": SDT_ARCHIVE_PATH,
        "source_archive_sha256": SDT_ARCHIVE_SHA256,
    }
    for key, expected in required_source.items():
        actual = provenance.get(key)
        if key == "source_doi" and actual == SDT_DOI.removeprefix("doi:"):
            continue
        if actual != expected:
            raise TemporalPolicyError(f"frame provenance mismatch for {key}: {actual!r}")
    for key in ("source_member_name", "source_member_index", "source_frame_index", "source_pose_label", "source_bbox", "raw_encoded_frame_sha256"):
        if key not in provenance:
            raise TemporalPolicyError(f"frame provenance missing {key}")
    if geometry_profile_id != T_A2_PROFILE_ID:
        raise TemporalPolicyError(f"unexpected T-A2 geometry profile: {geometry_profile_id}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(canonical_frame_hash)):
        raise TemporalPolicyError("canonical frame hash must be a lowercase SHA-256")
    pose_label = int(provenance["source_pose_label"])
    if pose_label not in POSE_NAMES:
        raise TemporalPolicyError(f"unknown original pose label: {pose_label}")
    record = {
        "sample_type": "ThermalFrameSample",
        "source_dataset_id": SDT_DATASET_ID,
        "source_doi": SDT_DOI,
        "source_split": SDT_SOURCE_SPLIT,
        "source_archive_path": SDT_ARCHIVE_PATH,
        "source_archive_sha256": SDT_ARCHIVE_SHA256,
        "source_member_name": str(provenance["source_member_name"]),
        "source_member_index": int(provenance["source_member_index"]),
        "source_frame_index": int(provenance["source_frame_index"]),
        "source_frame_index_role": "PROVENANCE_IDENTIFIER_ONLY_NOT_TIMESTAMP",
        "original_source_pose_label": pose_label,
        "original_source_pose_name": POSE_NAMES[pose_label],
        "original_source_bbox": list(provenance["source_bbox"]),
        "t_a1_raw_encoded_frame_sha256": str(provenance["raw_encoded_frame_sha256"]),
        "t_a2_geometry_profile_id": geometry_profile_id,
        "canonical_frame_hash": str(canonical_frame_hash),
        "canonical_shape": list(T_A2_CANONICAL_SHAPE),
        "canonical_dtype": T_A2_CANONICAL_DTYPE,
        "canonical_unit": T_A2_CANONICAL_UNIT,
        "source_timestamp_status": "ABSENT",
        "timestamp_reliability": "NOT_APPLICABLE",
        "source_fps_status": "NOT_VERIFIABLE",
        "sequence_id_status": "ABSENT",
        "session_id_status": "ABSENT",
        "event_id_status": "ABSENT",
        "temporal_predecessor_status": UNKNOWN_NOT_VERIFIABLE,
        "temporal_successor_status": UNKNOWN_NOT_VERIFIABLE,
        "frame_level_eligibility": "SUPPORTED",
        "sequence_level_eligibility": "NOT_VERIFIABLE",
        "event_level_eligibility": "NOT_VERIFIABLE",
        "window_level_eligibility": "NOT_APPLICABLE",
        "safe_nest_label_status": "NOT_ASSIGNED_T_A3",
    }
    validate_frame_sample(record)
    return record


def validate_frame_sample(record: Mapping[str, Any]) -> None:
    """Validate a frame-level record and reject temporal masquerading."""

    if record.get("sample_type") != "ThermalFrameSample":
        raise TemporalPolicyError("only ThermalFrameSample is source-supported in T-A3")
    unexpected = sorted(TEMPORAL_METADATA_FORBIDDEN_KEYS.intersection(record))
    if unexpected:
        raise FabricatedTemporalMetadataError(
            "frame sample contains forbidden temporal keys: " + ", ".join(unexpected)
        )
    required = (
        "source_dataset_id",
        "source_doi",
        "source_split",
        "source_archive_path",
        "source_archive_sha256",
        "source_member_name",
        "source_member_index",
        "source_frame_index",
        "source_frame_index_role",
        "original_source_pose_label",
        "original_source_pose_name",
        "original_source_bbox",
        "t_a1_raw_encoded_frame_sha256",
        "t_a2_geometry_profile_id",
        "canonical_frame_hash",
        "canonical_shape",
        "canonical_dtype",
        "canonical_unit",
        "source_timestamp_status",
        "timestamp_reliability",
        "source_fps_status",
        "sequence_id_status",
        "session_id_status",
        "event_id_status",
        "temporal_predecessor_status",
        "temporal_successor_status",
        "frame_level_eligibility",
        "sequence_level_eligibility",
        "event_level_eligibility",
        "window_level_eligibility",
        "safe_nest_label_status",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise TemporalPolicyError("frame sample missing fields: " + ", ".join(missing))
    expected = {
        "source_dataset_id": SDT_DATASET_ID,
        "source_doi": SDT_DOI,
        "source_split": SDT_SOURCE_SPLIT,
        "source_archive_path": SDT_ARCHIVE_PATH,
        "source_archive_sha256": SDT_ARCHIVE_SHA256,
        "source_frame_index_role": "PROVENANCE_IDENTIFIER_ONLY_NOT_TIMESTAMP",
        "t_a2_geometry_profile_id": T_A2_PROFILE_ID,
        "canonical_shape": list(T_A2_CANONICAL_SHAPE),
        "canonical_dtype": T_A2_CANONICAL_DTYPE,
        "canonical_unit": T_A2_CANONICAL_UNIT,
        "source_timestamp_status": "ABSENT",
        "timestamp_reliability": "NOT_APPLICABLE",
        "source_fps_status": "NOT_VERIFIABLE",
        "sequence_id_status": "ABSENT",
        "session_id_status": "ABSENT",
        "event_id_status": "ABSENT",
        "temporal_predecessor_status": UNKNOWN_NOT_VERIFIABLE,
        "temporal_successor_status": UNKNOWN_NOT_VERIFIABLE,
        "frame_level_eligibility": "SUPPORTED",
        "sequence_level_eligibility": "NOT_VERIFIABLE",
        "event_level_eligibility": "NOT_VERIFIABLE",
        "window_level_eligibility": "NOT_APPLICABLE",
        "safe_nest_label_status": "NOT_ASSIGNED_T_A3",
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise TemporalPolicyError(f"frame sample field {key} must be {value!r}, found {record.get(key)!r}")
    pose_label = record.get("original_source_pose_label")
    if pose_label not in POSE_NAMES or record.get("original_source_pose_name") != POSE_NAMES[pose_label]:
        raise TemporalPolicyError("original source pose label/name mismatch")
    for key in ("t_a1_raw_encoded_frame_sha256", "canonical_frame_hash"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(record.get(key))):
            raise TemporalPolicyError(f"{key} must be a lowercase SHA-256")
    if not isinstance(record.get("source_member_name"), str) or not re.fullmatch(r"test/image_t_[0-9]+\.png", record["source_member_name"]):
        raise TemporalPolicyError("source member name must be a portable SDT image_t member")
    for key in ("source_member_index", "source_frame_index"):
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TemporalPolicyError(f"{key} must be a non-negative integer source provenance index")
    if record["source_frame_index"] >= 8000:
        raise TemporalPolicyError("source_frame_index must be in [0, 7999]")
    expected_member = f"test/image_t_{record['source_frame_index']}.png"
    if record["source_member_name"] != expected_member:
        raise TemporalPolicyError("source member name and source frame index must agree structurally")
    if not isinstance(record.get("original_source_bbox"), list) or len(record["original_source_bbox"]) != 4:
        raise TemporalPolicyError("original source bbox must be preserved as four values")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in record["original_source_bbox"]):
        raise TemporalPolicyError("original source bbox values must be numeric")


__all__ = [
    "FabricatedTemporalMetadataError",
    "SDT_ARCHIVE_PATH",
    "SDT_ARCHIVE_SHA256",
    "SDT_DATASET_ID",
    "SDT_DOI",
    "SDT_SOURCE_SPLIT",
    "TEMPORAL_POLICY_ID",
    "TEMPORAL_POLICY_VERSION",
    "TemporalEventUnavailableError",
    "TemporalPolicyError",
    "TemporalSequenceUnavailableError",
    "TemporalWindowUnavailableError",
    "construct_event",
    "construct_sequence",
    "construct_window",
    "evaluate_event_construction",
    "evaluate_sequence_construction",
    "evaluate_window_construction",
    "frame_sample_from_provenance",
    "temporal_policy_profile",
    "validate_temporal_policy_profile",
    "validate_frame_sample",
    "validate_temporal_request",
]
