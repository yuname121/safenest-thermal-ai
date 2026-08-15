#!/usr/bin/env python3
"""Generate compact, deterministic Thermal T-A3 temporal evidence."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.thermal.canonical_geometry import canonicalize_source_frame, profile_for_id  # noqa: E402
from datasets.thermal.raw_reader import SDTThermalRawReader  # noqa: E402
from datasets.thermal.temporal_policy import (  # noqa: E402
    SDT_ARCHIVE_PATH,
    SDT_ARCHIVE_SHA256,
    SDT_DATASET_ID,
    SDT_DOI,
    SDT_SOURCE_SPLIT,
    T_A2_PROFILE_ID,
    frame_sample_from_provenance,
    temporal_policy_profile,
    validate_temporal_policy_profile,
)


EVIDENCE_REL = "datasets/thermal/manifests/T-A3_sequence_window_event_policy"
EVIDENCE_DIR = ROOT / EVIDENCE_REL
REPORT_REL = "docs/reports/20260810_Codex_T-A3_Thermal_Sequence_Window_Event_Evidence_Policy_01.md"
JSON_NAMES = [
    "event_policy.json",
    "frame_sample_contract.json",
    "gap_duplicate_policy.json",
    "limitations.json",
    "pilot_temporal_summary.json",
    "sequence_policy.json",
    "temporal_capability_contract.json",
    "temporal_evidence_registry.json",
    "validation_result.json",
    "window_policy.json",
]
PILOT_PER_CLASS = 12
ACCESS_DATE = "2026-08-10"
SDT_ZENODO_URL = "https://zenodo.org/records/4124309"
SDT_TUWIEN_URL = "https://cvl.tuwien.ac.at/research/cvl-databases/sdt-icip/"
SDT_PAPER_URL = "https://doi.org/10.1109/ICIP40778.2020.9191284"
TEMPORAL_FORBIDDEN_TERMS = (
    "timestamp",
    "fps",
    "frame_rate",
    "sequence_id",
    "session_id",
    "recording_id",
    "event_id",
    "event_start",
    "event_end",
    "window_start",
    "window_end",
    "window_duration",
    "window_frame_count",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _pilot_indices(reader: SDTThermalRawReader) -> dict[str, list[int]]:
    inventory = reader.inspect_archive()
    labels = getattr(reader, "_labels", None)
    if labels is None:
        raise RuntimeError("T-A1 labels were not loaded")
    grouped: dict[int, list[int]] = {}
    for label in labels:
        grouped.setdefault(label.source_pose_label, []).append(label.source_frame_index)
    selected: dict[str, list[int]] = {}
    for pose in sorted(grouped):
        indices = grouped[pose]
        positions = np.linspace(0, len(indices) - 1, PILOT_PER_CLASS, dtype=np.float64)
        chosen = [indices[int(round(position))] for position in positions]
        if len(chosen) != len(set(chosen)):
            raise RuntimeError(f"duplicate pilot index for source pose {pose}")
        selected[str(pose)] = chosen
    if sum(len(value) for value in selected.values()) != 48 or inventory["class_counts"] != {"0": 2000, "1": 2000, "2": 2000, "3": 2000}:
        raise RuntimeError("unexpected deterministic SDT pilot cardinality")
    return selected


def _bounded_png_metadata(reader: SDTThermalRawReader, indices: list[int]) -> dict[str, Any]:
    """Inspect bounded PNG headers/chunks without extracting archive members."""

    chunk_types: dict[str, int] = {}
    sampled_members: list[str] = []
    for index in indices:
        frame = reader.read_frame(index)
        sampled_members.append(frame.source_member_name)
        archive_path = reader.archive_path
        with zipfile.ZipFile(archive_path, "r") as archive:
            payload = archive.read(frame.source_member_name)
        cursor = 8
        while cursor + 12 <= len(payload):
            length = struct.unpack(">I", payload[cursor : cursor + 4])[0]
            chunk = payload[cursor + 4 : cursor + 8].decode("latin1")
            chunk_types[chunk] = chunk_types.get(chunk, 0) + 1
            cursor += 12 + length
            if chunk == "IEND" or cursor > len(payload):
                break
    temporal_chunks = {name: count for name, count in sorted(chunk_types.items()) if name in {"tIME", "eXIf", "tEXt", "zTXt", "iTXt"}}
    return {
        "method": "bounded direct member read; no extraction; deterministic pilot members only",
        "sampled_member_count": len(sampled_members),
        "sampled_members": sorted(sampled_members),
        "png_chunk_counts": dict(sorted(chunk_types.items())),
        "timestamp_or_metadata_chunk_counts": temporal_chunks,
        "timestamp_metadata_status": "NOT_FOUND_IN_BOUNDED_SAMPLE" if not temporal_chunks else "PRESENT_BUT_NOT_TREATED_AS_ACQUISITION_TIMELINE",
    }


def _official_evidence() -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": "OFFICIAL_ZENODO_SDT_4124309",
            "category": "OFFICIAL_EXTERNAL_SOURCE_VERIFIED",
            "source_url": SDT_ZENODO_URL,
            "stable_identifier": "doi:10.5281/zenodo.4124309",
            "access_date": ACCESS_DATE,
            "source_title": "SDT: A Dataset for Developing Fall Detection Algorithms using Depth and Thermal Images",
            "publisher": "TU Wien / Zenodo",
            "verified_claims": [
                "8,000 real depth/thermal image pairs are distributed for the test portion",
                "pose labels and bounding boxes are distributed",
                "non-commercial research use and citation/attribution conditions are stated",
            ],
            "temporal_schema_claim": "The official record describes image pairs and pose labels; no source FPS, timestamp, sequence, session, or event-boundary field is documented in the published record examined for T-A3.",
        },
        {
            "evidence_id": "OFFICIAL_TUWIEN_SDT_DOCUMENTATION",
            "category": "OFFICIAL_EXTERNAL_SOURCE_VERIFIED",
            "source_url": SDT_TUWIEN_URL,
            "stable_identifier": "SDT-ICIP-2020",
            "access_date": ACCESS_DATE,
            "source_title": "SDT dataset documentation",
            "publisher": "TU Wien Computer Vision Lab",
            "verified_claims": [
                "the dataset contains synthetic and real depth/thermal images",
                "the documented real capture uses an Orbbec Astra and FLIR Lepton 3.5",
                "the source classes are lying, sitting, standing, and empty-room control",
            ],
            "temporal_schema_claim": "The documentation examined is image/pose oriented and does not document a timestamped sequence or fall-event boundary schema.",
        },
        {
            "evidence_id": "OFFICIAL_SDT_PAPER",
            "category": "OFFICIAL_EXTERNAL_SOURCE_VERIFIED",
            "source_url": SDT_PAPER_URL,
            "stable_identifier": "doi:10.1109/ICIP40778.2020.9191284",
            "access_date": ACCESS_DATE,
            "source_title": "SDT: A Dataset for Developing Fall Detection Algorithms using Depth and Thermal Images",
            "publisher": "IEEE ICIP 2020",
            "verified_claims": ["the publication describes the dataset and pose-class image task"],
            "temporal_schema_claim": "No reproducible temporal acquisition fields or event onset/end annotations were identified in the source evidence used for this phase.",
        },
        {
            "evidence_id": "LOCAL_T_A1_SCHEMA_AND_READER",
            "category": "REPOSITORY_CODE_VERIFIED",
            "source_url": "datasets/thermal/raw_reader.py",
            "stable_identifier": SDT_DATASET_ID,
            "access_date": ACCESS_DATE,
            "verified_claims": [
                "image_t member and labels row are linked one-to-one by zero-based source frame index",
                "source timestamp/subject/session/sequence/event statuses are explicitly ABSENT",
                "archive index continuity is structural inventory evidence only",
            ],
            "temporal_schema_claim": "The active reader does not emit timestamp, FPS, sequence, session, recording, or event identifiers.",
        },
        {
            "evidence_id": "LOCAL_T_A2_SELECTED_PROFILE",
            "category": "VALIDATOR_INHERITED",
            "source_url": "datasets/thermal/manifests/T-A2_geometry_calibration_canonical_frame/selected_geometry_profile.json",
            "stable_identifier": T_A2_PROFILE_ID,
            "access_date": ACCESS_DATE,
            "verified_claims": ["T-A2 selected a deterministic 62x80 float32 Celsius canonical frame profile"],
            "temporal_schema_claim": "T-A2 geometry identity is retained; T-A3 does not create a split, window, label remap, or model input contract.",
        },
    ]


def _frame_sample_contract() -> dict[str, Any]:
    return {
        "phase": "T-A3",
        "schema_version": "1.0",
        "sample_type": "ThermalFrameSample",
        "status": "SUPPORTED",
        "definition": "One verified SDT test/image_t_<index>.png member plus its labels.txt row and T-A1/T-A2 hashes; source member index is ZIP directory provenance and source frame index is the filename/label-row link.",
        "required_fields": [
            "source_dataset_id", "source_doi", "source_split", "source_archive_path", "source_archive_sha256",
            "source_member_name", "source_member_index", "source_frame_index", "original_source_pose_label",
            "original_source_pose_name", "original_source_bbox", "t_a1_raw_encoded_frame_sha256",
            "t_a2_geometry_profile_id", "canonical_frame_hash", "canonical_shape", "canonical_dtype", "canonical_unit",
            "source_timestamp_status", "timestamp_reliability", "source_fps_status", "sequence_id_status",
            "session_id_status", "event_id_status", "temporal_predecessor_status", "temporal_successor_status",
        ],
        "source_identity_retained": True,
        "original_label_semantics_retained": True,
        "source_split": SDT_SOURCE_SPLIT,
        "safenest_split_created": False,
        "t_a1_raw_identity_retained": True,
        "t_a2_geometry_profile_retained": T_A2_PROFILE_ID,
        "canonical_physical_frame_identity_retained": True,
        "synthetic_temporal_metadata_added": False,
        "forbidden_fields": sorted(TEMPORAL_FORBIDDEN_TERMS),
        "original_labels": {"0": "LYING", "1": "SITTING", "2": "STANDING", "3": "EMPTY_ROOM"},
        "bbox_semantics": "Original labels.txt bbox is preserved; EMPTY_ROOM keeps -1 sentinel; bbox is not a temporal event annotation.",
    }


def _sequence_policy() -> dict[str, Any]:
    return {
        "phase": "T-A3", "schema_version": "1.0", "policy_id": "THERMAL_T_A3_SEQUENCE_POLICY_001",
        "eligible": False, "status": "TEMPORAL_SEQUENCE_NOT_VERIFIABLE",
        "required_evidence": ["sequence_id or recording_id", "verified acquisition order", "timestamp or documented cadence", "gap and duplicate semantics"],
        "available_evidence": ["structural member index", "one-to-one labels linkage"],
        "grouping_rule": "No sequence is constructed. Member filename/index and neighboring numbers are provenance identifiers only.",
        "forbidden_inferences": ["index adjacency is frame adjacency", "filename order is temporal order", "pose runs form a sequence", "image similarity forms a sequence"],
        "frame_adjacency_claim": "NOT_VERIFIABLE",
        "unsupported_request_behavior": {"error_code": "TEMPORAL_SEQUENCE_UNAVAILABLE", "behavior": "FAIL_CLOSED"},
        "safe_to_construct": False,
    }


def _event_policy() -> dict[str, Any]:
    return {
        "phase": "T-A3", "schema_version": "1.0", "policy_id": "THERMAL_T_A3_EVENT_POLICY_001",
        "eligible": False, "status": "TEMPORAL_EVENT_NOT_VERIFIABLE",
        "required_evidence": ["verified sequence", "event identifier or annotation", "fall transition", "onset", "end", "pre/during/post context"],
        "available_evidence": ["original posture label only"],
        "fall_transition": "NOT_VERIFIABLE", "fall_onset": "NOT_VERIFIABLE", "fall_end": "NOT_VERIFIABLE",
        "pre_fall_range": "NOT_VERIFIABLE", "during_fall_range": "NOT_VERIFIABLE", "post_fall_range": "NOT_VERIFIABLE",
        "lying_is_fall_event": False,
        "label_semantics": {"LYING": "POSTURE_ONLY_NOT_A_FALL_EVENT", "SITTING": "POSTURE", "STANDING": "POSTURE", "EMPTY_ROOM": "BACKGROUND_CONTROL"},
        "unsupported_request_behavior": {"error_code": "SOURCE_EVENT_BOUNDARY_UNAVAILABLE", "behavior": "FAIL_CLOSED"},
        "safe_to_construct": False,
    }


def _window_policy() -> dict[str, Any]:
    return {
        "phase": "T-A3", "schema_version": "1.0", "policy_id": "THERMAL_T_A3_WINDOW_POLICY_001",
        "eligible": False, "status": "WINDOWING_NOT_APPLICABLE_TO_SOURCE",
        "window_duration": "NOT_APPLICABLE_NO_VERIFIED_TIMELINE", "window_frame_count": "NOT_APPLICABLE_NO_VERIFIED_TIMELINE",
        "stride": "NOT_APPLICABLE_NO_VERIFIED_TIMELINE", "overlap": "NOT_APPLICABLE_NO_VERIFIED_TIMELINE", "temporal_unit": "NOT_APPLICABLE",
        "required_evidence": ["verified timeline", "sequence grouping", "duration or frame-count semantics", "gap/duplicate policy", "boundary policy"],
        "available_evidence": [],
        "arbitrary_values_allowed": False,
        "reason": "A frame count or duration has no verified time unit because SDT FPS/timestamps/sequence identity are unavailable.",
        "unsupported_request_behavior": {"error_code": "WINDOW_CONSTRUCTION_NOT_ALLOWED", "behavior": "FAIL_CLOSED"},
        "safe_to_construct": False,
    }


def build_artifacts(root: Path = ROOT) -> dict[str, Any]:
    reader = SDTThermalRawReader(repo_root=root)
    inventory = reader.inspect_archive()
    by_pose = _pilot_indices(reader)
    indices = [index for pose in sorted(by_pose, key=int) for index in by_pose[pose]]
    profile = profile_for_id(T_A2_PROFILE_ID)
    policy = temporal_policy_profile()
    validate_temporal_policy_profile(policy)
    pilot_records: list[dict[str, Any]] = []
    for index in indices:
        source = reader.read_frame(index)
        canonical = canonicalize_source_frame(source, profile)
        record = frame_sample_from_provenance(source.provenance_dict(), canonical_frame_hash=canonical.canonical_frame_hash)
        record.update({
            "source_member_crc32": source.source_member_crc32,
            "source_member_sha256": source.source_member_sha256,
            "source_representation": source.source_representation,
            "source_temperature_encoding": source.source_temperature_encoding,
            "source_dtype": source.source_dtype,
            "distributed_frame_shape": list(source.distributed_frame_shape),
            "native_sensor_shape": list(source.native_sensor_shape),
            "quality_flags": list(source.quality_flags),
        })
        pilot_records.append(record)
    pilot_records.sort(key=lambda item: item["source_frame_index"])
    source_counts = {str(key): len(value) for key, value in sorted(by_pose.items(), key=lambda item: int(item[0]))}
    temporal_statuses = {
        "source_fps": "NOT_VERIFIABLE",
        "source_timestamp": "ABSENT",
        "timestamp_reliability": "NOT_APPLICABLE",
        "sequence_id": "ABSENT",
        "session_id": "ABSENT",
        "event_id": "ABSENT",
        "fall_event_boundaries": "NOT_VERIFIABLE",
    }
    metadata = _bounded_png_metadata(reader, indices[:12])
    gap_policy = {
        "phase": "T-A3", "schema_version": "1.0",
        "archive_identity": inventory["archive_identity"],
        "structural_member_index": {
            "index_base": inventory["index_base"], "index_last": inventory["index_last"],
            "continuous": inventory["index_continuous"], "missing_indices": inventory["missing_thermal_indices"],
            "duplicate_indices": inventory["duplicate_thermal_indices"], "duplicate_member_names": inventory["duplicate_member_names"],
            "index_relationship": {
                "thermal_indices": "0..7999 continuous",
                "depth_indices": "0..7999 continuous",
                "label_indices": "labels.txt row 0..7999",
                "missing_intersections": [],
                "unexpected_indices": [],
            },
        },
        "index_gap_status": "SOURCE_MEMBER_INDEX_GAP",
        "index_gap_interpretation": "STRUCTURAL_ARCHIVE_PROVENANCE_ONLY; NOT_A_DROPPED_ACQUISITION_FRAME",
        "temporal_dropped_frame_status": "TEMPORAL_DROPPED_FRAME_NOT_VERIFIABLE",
        "temporal_large_gap_status": "TEMPORAL_GAP_NOT_VERIFIABLE",
        "duplicate_member_index_status": "STRUCTURAL_SOURCE_FAILURE",
        "duplicate_member_name_status": "STRUCTURAL_SOURCE_FAILURE",
        "exact_duplicate_content_semantics": "PRESERVE_BOTH_PROVENANCE_RECORDS_AND_FLAG_DUPLICATE_CONTENT; DO_NOT_INFER_ADJACENCY",
        "near_duplicate_semantics": "DO_NOT_INFER_TEMPORAL_ADJACENCY; FULL_AUDIT_DEFERRED_T_A6",
        "full_duplicate_audit_deferred_to": "T-A6",
    }
    temporal_contract = {
        "phase": "T-A3", "schema_version": "1.0", "policy": policy,
        "source_representation": "RADIOMETRIC_TEMPERATURE_ENCODED_UINT16",
        "source_frame_shape": [480, 640], "source_dtype": "uint16", "source_unit": "KELVIN_CENTIUNITS",
        "source_fps": {"status": "NOT_VERIFIABLE", "value": "NOT_VERIFIABLE"},
        "timestamps": {"availability": "ABSENT", "reliability": "NOT_APPLICABLE"},
        "sequence_identifiers": {"availability": "ABSENT"}, "session_identifiers": {"availability": "ABSENT"},
        "event_identifiers": {"availability": "ABSENT"}, "fall_event_boundaries": {"availability": "NOT_VERIFIABLE"},
        "temporal_ordering": {"availability": "NOT_VERIFIABLE", "structural_index_only": True},
        "capabilities": policy["capabilities"],
        "source_split_preserved": True, "safenest_split_created": False, "model_performance_used": False,
        "evidence_categories": ["LOCALLY_MEASURED", "VALIDATOR_INHERITED", "OFFICIAL_EXTERNAL_SOURCE_VERIFIED", "UNKNOWN", "NOT_APPLICABLE"],
    }
    evidence_registry = {
        "phase": "T-A3", "schema_version": "1.0", "access_date": ACCESS_DATE,
        "source_records": _official_evidence(),
        "local_measurements": {
            "archive_identity": inventory["archive_identity"], "thermal_member_count": inventory["thermal_member_count"],
            "depth_member_count": inventory["depth_member_count"], "label_row_count": inventory["label_row_count"],
            "index_continuity": inventory["index_continuous"], "duplicate_thermal_indices": inventory["duplicate_thermal_indices"],
            "duplicate_member_names": inventory["duplicate_member_names"], "label_class_counts": inventory["class_counts"],
            "bounded_png_metadata": metadata,
        },
        "temporal_fields": temporal_statuses,
        "source_schema_conclusion": "Frame-level identity is reproducible. Temporal sequence, event, and window construction remain NOT_VERIFIABLE/NOT_APPLICABLE because the active source schema and official docs do not provide the needed evidence.",
    }
    summary = {
        "phase": "T-A3", "schema_version": "1.0", "source_dataset_id": SDT_DATASET_ID, "source_doi": SDT_DOI,
        "source_archive_identity": inventory["archive_identity"], "source_split": SDT_SOURCE_SPLIT,
        "selection_rule": "12 evenly spaced sorted source frame indices per original class, inherited from T-A2 pilot; no temporal ordering claim",
        "pilot_frame_count": len(pilot_records), "source_class_counts": source_counts,
        "source_classes_represented": ["LYING", "SITTING", "STANDING", "EMPTY_ROOM"],
        "temporal_statuses": temporal_statuses,
        "frame_level_eligibility": "SUPPORTED", "sequence_level_eligibility": "NOT_VERIFIABLE",
        "event_level_eligibility": "NOT_VERIFIABLE", "window_level_eligibility": "NOT_APPLICABLE",
        "t_a2_geometry_profile_id": T_A2_PROFILE_ID, "records": pilot_records,
        "fabricated_temporal_metadata": {"timestamps": False, "sequences": False, "events": False, "windows": False},
        "model_performance_used": False,
    }
    limitations = {
        "phase": "T-A3", "schema_version": "1.0", "status": "PASS_WITH_LIMITATIONS",
        "limitations": [
            "SDT source FPS and acquisition timestamps are not verifiable from the distributed archive/schema.",
            "Source frame indices and filename order are provenance identifiers, not timestamps or temporal adjacency.",
            "No subject, session, recording, sequence, or event identifier is retained by the active source schema.",
            "LYING is an original posture label and is not converted into a fall event or fall boundary.",
            "No sequence, event, or temporal window is constructed; arbitrary frame counts/durations are prohibited.",
            "Archive index gaps, duplicate content, and near-duplicates cannot be mapped to acquisition drops without a timeline; full duplicate audit is deferred to T-A6.",
            "Historical thermal_prep.py posture-to-model mapping is legacy evidence only and is not used by T-A3.",
            "Train/validation split placeholders remain unhydrated; T-A3 does not hydrate or reconstruct them.",
            "T-A2 geometry identity is inherited; full conversion and SafeNest grouping/splits remain later-phase work.",
            "Thermal-44 temporal and hardware behavior remains deferred to T-C.",
        ],
        "not_verifiable": ["source_fps", "timestamps", "temporal_order", "sequence_boundaries", "fall_event_onset_end", "window_duration", "event_level_performance"],
        "downstream": {"T-A4": "original label semantics/ambiguity only", "T-A5": "subject/session/grouping and split policy", "T-A6": "full conversion, duplicate/integrity audit", "T-B": "temporal architecture/event metrics", "T-C": "Thermal-44 temporal hardware contract"},
    }
    return {
        "temporal_capability_contract.json": temporal_contract,
        "frame_sample_contract.json": _frame_sample_contract(),
        "sequence_policy.json": _sequence_policy(),
        "event_policy.json": _event_policy(),
        "window_policy.json": _window_policy(),
        "gap_duplicate_policy.json": gap_policy,
        "temporal_evidence_registry.json": evidence_registry,
        "pilot_temporal_summary.json": summary,
        "limitations.json": limitations,
    }


def report_text(artifacts: dict[str, Any], validation: dict[str, Any]) -> str:
    summary = artifacts["pilot_temporal_summary.json"]
    return f"""# Thermal T-A3 — Sequence, Window, and Event-Evidence Policy

Date: {ACCESS_DATE}\n\nPhase: `T-A3`\n\nOutcome: `{validation.get('overall_outcome', 'PASS_WITH_LIMITATIONS')}`\n\nT-A4 authorized: `{'YES' if validation.get('t_a4_authorized') else 'NO'}`\n\n## Source decision\n\nThe selected local source is the SDT test archive `{SDT_ARCHIVE_PATH}` with SHA-256 `{SDT_ARCHIVE_SHA256}`. The [Zenodo SDT record]({SDT_ZENODO_URL}) and [TU Wien SDT documentation]({SDT_TUWIEN_URL}) describe thermal/depth image pairs and pose labels (LYING, SITTING, STANDING, EMPTY_ROOM), not a timestamped fall-event stream. The active T-A1 reader independently measures one-to-one member/index/label linkage and explicitly records timestamp, subject, session, sequence, and event fields as absent.\n\n## Temporal boundary\n\nT-A3 freezes a supported `FRAME_LEVEL` contract for {summary['pilot_frame_count']} real frames ({summary['source_class_counts']} by original class). A member filename or integer index identifies source provenance only. It is not a timestamp, FPS proxy, or proof that neighboring indices belong to one sequence. An index gap is reported as structural archive evidence; it is not labeled a dropped acquisition frame.\n\n`SEQUENCE_LEVEL` is `NOT_VERIFIABLE`, `EVENT_LEVEL` is `NOT_VERIFIABLE`, and `WINDOW_LEVEL` is `NOT_APPLICABLE`. No FPS, timestamps, sequence IDs, event IDs, fall onset/end, pre/during/post ranges, duration, stride, or overlap values are fabricated.\n\n## Labels and events\n\nThe original `LYING` label is preserved as posture semantics. It is not silently relabeled as a fall event: without transition, onset, impact, end, and surrounding context, a lying frame cannot establish when or whether a fall occurred. A later event policy must use authoritative temporal evidence rather than a posture run.\n\n## Representation and geometry\n\nThe source remains radiometric temperature encoded as uint16 PNG values with the T-A1 Celsius conversion. Each pilot record retains source archive/member/index, raw encoded hash, original bbox and pose label, and the selected T-A2 geometry profile `{T_A2_PROFILE_ID}` with canonical frame hash. T-A3 does not train, infer, normalize, relabel, split, or perform full conversion.\n\n## Gap, duplicate, and downstream limits\n\nDuplicate member names/indices and missing structural indices are checked from the ZIP central directory. Exact duplicate content would retain both provenance records and be flagged; near-duplicate/complete acquisition-level audit is deferred to T-A6 because no timeline exists. Train/validation placeholders are not hydrated. Thermal-44 FPS, clock, drop semantics, buffering, and hardware validation remain deferred to T-C.\n\nEvidence categories used: `LOCALLY_MEASURED`, `VALIDATOR_INHERITED`, `OFFICIAL_EXTERNAL_SOURCE_VERIFIED`, `UNKNOWN`, and `NOT_APPLICABLE`.\n"""


def write_artifacts(root: Path = ROOT) -> dict[str, Any]:
    evidence_dir = root / EVIDENCE_REL
    evidence_dir.mkdir(parents=True, exist_ok=True)
    artifacts = build_artifacts(root)
    for name, data in sorted(artifacts.items()):
        (evidence_dir / name).write_text(canonical_json(data), encoding="utf-8")
    from scripts.validate_thermal_t_a3 import validate_evidence

    validation = validate_evidence(repo_root=root, evidence_dir=evidence_dir, check_checksums=False, verify_real_payload=True)
    (evidence_dir / "validation_result.json").write_text(canonical_json(validation), encoding="utf-8")
    checksum_paths = [evidence_dir / name for name in sorted(JSON_NAMES)]
    checksum_lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in checksum_paths]
    (evidence_dir / "checksums.sha256").write_text("\n".join(sorted(checksum_lines, key=lambda line: line.split("  ", 1)[1])) + "\n", encoding="utf-8")
    report_path = root / REPORT_REL
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text(artifacts, validation), encoding="utf-8")
    return validation


if __name__ == "__main__":
    result = write_artifacts()
    print(canonical_json(result), end="")
    raise SystemExit(0 if result.get("evidence_validation") == "PASS" else 1)
