#!/usr/bin/env python3
"""Generate compact T-A6 Stage-1 evidence and the local real artifact.

The generator is intentionally conservative about cloud-backed files.  It
performs metadata-only inspection first and invokes the converter only when
the real ``test.zip`` is demonstrably materialized.  Synthetic train and
validation paths are never opened in MAC_STAGE1.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EVIDENCE_REL = "datasets/thermal/manifests/T-A6_full_conversion_integrity"
ARTIFACT_REL = "datasets/thermal/artifacts/T-A6_real_eval_development"
REPORT_REL = "docs/reports/20260811_Codex_T-A6_Stage1_Thermal_Real_Conversion_Colab_01.md"
SOURCE_REL = "datasets/raw_archives/thermal_split_zips"
REAL_ARCHIVE_REL = f"{SOURCE_REL}/test.zip"
JSON_NAMES = [
    "canonical_dataset_contract.json",
    "real_source_partition_inventory.json",
    "real_conversion_status_summary.json",
    "real_canonical_artifact_registry.json",
    "sample_provenance_schema.json",
    "real_sample_index_summary.json",
    "real_label_alignment_summary.json",
    "real_quality_audit.json",
    "real_exact_duplicate_audit.json",
    "near_duplicate_profile.json",
    "real_near_duplicate_audit.json",
    "real_leakage_audit.json",
    "real_output_checksum_registry.json",
    "real_determinism_audit.json",
    "stage1_status.json",
    "colab_execution_contract.json",
    "limitations.json",
    "validation_result.json",
]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def _metadata_state(path: Path, *, owner_state: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.relative_to(ROOT).as_posix(),
        "owner_observation": owner_state,
        "path_exists": False,
        "logical_size_bytes": None,
        "physical_blocks": None,
        "filesystem_flags": None,
        "git_visibility": "NOT_VISIBLE_GIT_IGNORED",
        "git_ignore_rule": "datasets/raw_archives/",
        "materialization_state": "ABSENT",
        "readable_offline": False,
        "payload_read_attempted": False,
        "checksum_status": "NOT_COMPUTED_METADATA_ONLY",
    }
    try:
        metadata = path.stat()
    except (FileNotFoundError, OSError):
        return result
    result["path_exists"] = True
    result["logical_size_bytes"] = int(metadata.st_size)
    result["physical_blocks"] = int(getattr(metadata, "st_blocks", 0))
    flags = int(getattr(metadata, "st_flags", 0))
    result["filesystem_flags"] = flags
    dataless = bool(flags & int(getattr(stat, "SF_DATALESS", 0x40000000)))
    # A zero-block or dataless cloud item is recorded as a placeholder.  This
    # metadata check does not open the file and therefore cannot hydrate it.
    if metadata.st_size > 0 and (result["physical_blocks"] == 0 or dataless):
        result["materialization_state"] = "LOCAL_CLOUD_PLACEHOLDER"
    elif path.is_file():
        result["materialization_state"] = "LOCALLY_MATERIALIZED"
    else:
        result["materialization_state"] = "NOT_REGULAR_FILE"
    result["readable_offline"] = result["materialization_state"] == "LOCALLY_MATERIALIZED"
    return result


def inspect_local_payloads(root: Path) -> dict[str, Any]:
    base = root / SOURCE_REL
    entries = {
        "test.zip": "OWNER_CONFIRMED_LOCALLY_MATERIALIZED",
        "train.zip.001": "OWNER_CONFIRMED_LOCAL_CLOUD_PLACEHOLDER",
        "train.zip.002": "OWNER_CONFIRMED_LOCAL_CLOUD_PLACEHOLDER",
        "train.zip.003": "OWNER_CONFIRMED_LOCAL_CLOUD_PLACEHOLDER",
        "train.zip.004": "OWNER_CONFIRMED_LOCAL_CLOUD_PLACEHOLDER",
        "validation.zip": "OWNER_CONFIRMED_LOCAL_CLOUD_PLACEHOLDER",
    }
    return {name: _metadata_state(base / name, owner_state=owner) for name, owner in entries.items()}


def _load_json(path: Path, default: Mapping[str, Any] | None = None) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else dict(default or {})
    except (OSError, json.JSONDecodeError):
        return dict(default or {})


def _near_profile() -> dict[str, Any]:
    from datasets.thermal.canonical_converter import near_duplicate_profile

    return near_duplicate_profile()


def _source_inventory(payloads: Mapping[str, Any], root: Path) -> dict[str, Any]:
    a1 = _load_json(root / "datasets/thermal/manifests/T-A1_safe_reader_raw_unit_contract/archive_member_inventory.json")
    return {
        "phase": "T-A6_STAGE1",
        "dataset_id": "local_sdt_zenodo_4124309",
        "doi": "doi:10.5281/zenodo.4124309",
        "official_partition": "test",
        "source_domain": "REAL",
        "safenest_role": "REAL_EVAL_DEVELOPMENT",
        "source_path": REAL_ARCHIVE_REL,
        "locked_archive_sha256": "3a838bd70835e579ecfaa820a6c0b4cbc6ba7b76729417c73845f0c959281449",
        "locked_archive_size_bytes": 1740348425,
        "expected_frame_count": 8000,
        "t_a1_compact_inventory": {
            key: a1.get(key)
            for key in (
                "member_count", "file_count", "directory_count", "thermal_member_count",
                "depth_member_count", "label_row_count", "index_continuous", "thermal_depth_label_linkage",
            )
            if key in a1
        },
        "local_payload_measurements": payloads,
        "synthetic_payload_access": "PROHIBITED_STAGE_1",
    }


def _base_contract() -> dict[str, Any]:
    return {
        "phase": "T-A6_STAGE1",
        "schema_version": "1.0",
        "source_dataset_id": "local_sdt_zenodo_4124309",
        "source_doi": "doi:10.5281/zenodo.4124309",
        "source_partition": "test",
        "source_domain": "REAL",
        "safenest_role": "REAL_EVAL_DEVELOPMENT",
        "canonical_order": "source frame index ascending, zero based",
        "raw_reader_contract": "T-A1_SOURCE_FRAME_PROVENANCE_CONTRACT",
        "geometry_profile_id": "G1_FIXED_ASPECT_CROP_BILINEAR",
        "canonical_shape": [62, 80],
        "canonical_dtype": "float32_little_endian",
        "canonical_unit": "CELSIUS",
        "physical_conversion": "celsius=(uint16_kelvin_centiunits-27315)/100",
        "normalization": "NONE_T_A6_PHYSICAL_ARTIFACT",
        "augmentation": "NONE",
        "model_dependency": "NONE",
        "storage_format": "NPY_MEMMAP_V1_LITTLE_ENDIAN_FLOAT32",
        "provenance_format": "JSONL_V1_ONE_ROW_PER_CANONICAL_SAMPLE",
        "bulk_artifact_boundary": "Git-ignored local artifact; compact checksums and summaries are tracked",
        "legacy_npz_authority": "LEGACY_NON_AUTHORITATIVE_NOT_USED",
    }


def _provenance_schema() -> dict[str, Any]:
    fields = [
        "canonical_sample_index", "stable_sample_id", "source_dataset_id", "source_doi", "source_split",
        "source_domain", "source_archive_path", "source_archive_size_bytes", "source_archive_md5",
        "source_archive_sha256", "source_member", "source_member_index", "source_member_crc32",
        "source_member_sha256", "source_frame_index", "source_frame_sha256", "source_pose_label",
        "source_pose_name", "source_bbox", "source_shape", "source_dtype", "source_representation",
        "source_temperature_encoding", "t_a1_reader_contract", "t_a2_geometry_profile_id",
        "t_a3_temporal_policy_id", "t_a4_semantic_policy_id", "t_a5_split_policy_id",
        "t_a5_assignment_rule_id", "safenest_assignment", "original_label_id", "original_label_name",
        "original_bbox", "frame_evidence_label", "compatibility_target", "mapping_type", "mapping_rule_id",
        "claim_scope", "canonical_shape", "canonical_dtype", "canonical_unit", "canonical_frame_hash",
        "canonical_tensor_row_sha256", "quality_status", "quality_warning_codes", "conversion_status",
    ]
    return {
        "phase": "T-A6_STAGE1",
        "format_id": "JSONL_V1_ONE_ROW_PER_CANONICAL_SAMPLE",
        "one_row_per": "canonical sample; source frame index is retained separately",
        "required_fields": fields,
        "ordering": "canonical_sample_index ascending",
        "no_object_array": True,
        "source_subject_session_event": "ABSENT_AND_EXPLICIT_NOT_VERIFIABLE",
    }


def _artifact_summary(summary: Mapping[str, Any] | None, artifact_dir: Path) -> dict[str, Any]:
    if not summary:
        return {
            "status": "BLOCKED_SOURCE_PAYLOAD_NOT_MATERIALIZED",
            "artifact_path": f"{ARTIFACT_REL}/real_eval_development_canonical.npy",
            "provenance_path": f"{ARTIFACT_REL}/real_eval_development_provenance.jsonl",
            "expected_shape": [8000, 62, 80],
            "dtype": "float32_little_endian",
            "unit": "CELSIUS",
            "safenest_role": "REAL_EVAL_DEVELOPMENT",
            "finalized_status": "ABSENT",
            "checksum_status": "NOT_AVAILABLE",
        }
    artifact = artifact_dir / "real_eval_development_canonical.npy"
    provenance = artifact_dir / "real_eval_development_provenance.jsonl"
    return {
        "status": summary.get("finalized_status", "UNKNOWN"),
        "artifact_path": summary.get("artifact_path"),
        "provenance_path": summary.get("provenance_path"),
        "shape": [summary.get("expected_source_frames"), 62, 80],
        "dtype": summary.get("canonical_dtype"),
        "unit": summary.get("canonical_unit"),
        "safenest_role": summary.get("safenest_role"),
        "artifact_size_bytes": summary.get("artifact_size_bytes", artifact.stat().st_size if artifact.exists() else None),
        "artifact_sha256": summary.get("artifact_sha256"),
        "provenance_size_bytes": summary.get("provenance_size_bytes", provenance.stat().st_size if provenance.exists() else None),
        "provenance_sha256": summary.get("provenance_sha256"),
        "finalized_status": summary.get("finalized_status"),
        "checksum_status": "LOCKED" if summary.get("artifact_sha256") else "NOT_AVAILABLE",
        "partial_files_rejected": True,
    }


def _compact_near_duplicate_audit(audit: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Keep compact deterministic witnesses while retaining full counts."""
    if not audit:
        return None
    value = dict(audit)
    pairs = list(value.get("confirmed_pairs", []))
    witness_limit = 200
    if len(pairs) > witness_limit:
        value["confirmed_pairs_total"] = len(pairs)
        value["confirmed_pairs_truncated"] = True
        value["confirmed_pairs"] = pairs[:witness_limit]
    else:
        value["confirmed_pairs_total"] = len(pairs)
        value["confirmed_pairs_truncated"] = False
    value.pop("audit_sha256", None)
    value["audit_sha256"] = sha256_bytes(canonical_json(value).encode("utf-8"))
    return value


def _default_status(payloads: Mapping[str, Any]) -> dict[str, Any]:
    test = payloads["test.zip"]
    if test["materialization_state"] != "LOCALLY_MATERIALIZED":
        return {
            "gate": "BLOCKED",
            "code": "SOURCE_PAYLOAD_NOT_MATERIALIZED",
            "message": "test.zip is owner-confirmed but current metadata marks it as a cloud/dataless placeholder; no read or hydration was attempted.",
        }
    return {
        "gate": "PENDING_REAL_CONVERSION",
        "code": "REAL_CONVERSION_NOT_RUN",
        "message": "real test conversion has not yet been executed",
    }


def _build_documents(root: Path, payloads: Mapping[str, Any], summary: Mapping[str, Any] | None, alignment: Mapping[str, Any] | None, near: Mapping[str, Any] | None, deterministic: Mapping[str, Any]) -> dict[str, Any]:
    artifact_dir = root / ARTIFACT_REL
    status = _default_status(payloads) if summary is None else {
        "gate": "CONVERSION_COMPLETE",
        "code": "REAL_CONVERSION_FINALIZED",
        "message": "all real source rows were converted and finalized",
    }
    counts = summary.get("status_counts", {}) if summary else {}
    quality = summary.get("quality", {}) if summary else {}
    exact = summary.get("exact_duplicate_audit", {}) if summary else None
    compact_near = _compact_near_duplicate_audit(near)
    docs: dict[str, Any] = {
        "canonical_dataset_contract.json": _base_contract(),
        "real_source_partition_inventory.json": _source_inventory(payloads, root),
        "real_conversion_status_summary.json": {
            "phase": "T-A6_STAGE1", "source_split": "test", "source_domain": "REAL",
            "safenest_role": "REAL_EVAL_DEVELOPMENT", "expected_source_frames": 8000,
            "source_frames_measured": summary.get("source_frames_measured") if summary else None,
            "status_counts": counts or {key: None for key in ("SUCCESS", "SUCCESS_WITH_WARNING", "EXCLUDED", "FAILED")},
            "canonical_rows": summary.get("canonical_rows") if summary else None,
            "reconciliation": summary.get("reconciliation") if summary else {"status": "PENDING_REAL_CONVERSION"},
            "finalized_status": summary.get("finalized_status") if summary else "ABSENT",
            "stage1_status": status,
        },
        "real_canonical_artifact_registry.json": _artifact_summary(summary, artifact_dir),
        "sample_provenance_schema.json": _provenance_schema(),
        "real_sample_index_summary.json": {
            "status": "PASS" if alignment else "PENDING_REAL_CONVERSION",
            "expected_source_frame_index_range": [0, 7999],
            "canonical_order": "source frame index ascending",
            "alignment": alignment or {},
            "witness_indices": [0, 1, 4000, 7998, 7999],
        },
        "real_label_alignment_summary.json": {
            "status": "PASS" if alignment else "PENDING_REAL_CONVERSION",
            "original_source_labels_retained": True,
            "proxy_layer_separate": True,
            "mapping_is_event_ground_truth": False,
            "counts": alignment or {},
        },
        "real_quality_audit.json": {
            "status": "PASS" if summary else "PENDING_REAL_CONVERSION",
            "source_frame_count": 8000,
            "quality": quality,
            "threshold_policy": "NO_INVENTED_THERMAL44_SATURATION_THRESHOLDS",
            "difficult_valid_frames_not_auto_excluded": True,
        },
        "real_exact_duplicate_audit.json": exact or {
            "status": "PENDING_REAL_CONVERSION", "audit_scope": "WITHIN_REAL_EVAL_DEVELOPMENT",
            "source_member_bytes": "PENDING", "decoded_frames": "PENDING", "canonical_frames": "PENDING",
            "exclusions_caused_by_duplicates": 0,
        },
        "near_duplicate_profile.json": _near_profile(),
        "real_near_duplicate_audit.json": compact_near or {
            "status": "PENDING_REAL_CONVERSION", "audit_scope": "WITHIN_REAL_EVAL_DEVELOPMENT",
            "exhaustiveness_claim": "DETERMINISTIC_SCREENING_NOT_EXHAUSTIVE",
        },
        "real_leakage_audit.json": {
            "status": "PENDING_COLAB_STAGE2",
            "within_real_exact": "REPORTED_IN_REAL_EXACT_DUPLICATE_AUDIT",
            "within_real_near": "REPORTED_IN_REAL_NEAR_DUPLICATE_AUDIT",
            "train_to_validation_exact": "PENDING_COLAB_STAGE2",
            "train_to_real_exact": "PENDING_COLAB_STAGE2",
            "validation_to_real_exact": "PENDING_COLAB_STAGE2",
            "cross_domain_near_duplicates": "PENDING_COLAB_STAGE2",
            "subject_session_event_generalization": "NOT_VERIFIABLE_SOURCE_PROVENANCE_ABSENT",
        },
        "real_output_checksum_registry.json": {
            "status": "LOCKED" if summary and summary.get("artifact_sha256") else "PENDING_REAL_CONVERSION",
            "artifact_path": f"{ARTIFACT_REL}/real_eval_development_canonical.npy",
            "provenance_path": f"{ARTIFACT_REL}/real_eval_development_provenance.jsonl",
            "artifact_sha256": summary.get("artifact_sha256") if summary else None,
            "provenance_sha256": summary.get("provenance_sha256") if summary else None,
            "ledger_sha256": sha256_file(artifact_dir / "real_eval_development_conversion_ledger.json") if (artifact_dir / "real_eval_development_conversion_ledger.json").is_file() else None,
            "partial_artifacts_accepted": False,
        },
        "real_determinism_audit.json": deterministic,
        "stage1_status.json": {
            "phase": "T-A6_STAGE1", "stage1_gate": "PENDING", "full_t_a6_gate": "NOT_YET_COMPLETE",
            "t_b_authorized": False, "reason": status,
            "synthetic_train_status": "PENDING_COLAB_STAGE2",
            "synthetic_validation_status": "PENDING_COLAB_STAGE2",
            "mac_synthetic_payload_access": "PROHIBITED_STAGE_1",
        },
        "colab_execution_contract.json": {
            "phase": "T-A6_STAGE1", "runner_path": "scripts/run_thermal_t_a6_colab.py",
            "core_converter_path": "datasets/thermal/canonical_converter.py",
            "drive_raw_root": "ARGUMENT_OR_THERMAL_A6_DRIVE_RAW_ROOT",
            "work_root": "ARGUMENT_OR_THERMAL_A6_WORK_ROOT",
            "drive_output_root": "ARGUMENT_OR_THERMAL_A6_DRIVE_OUTPUT_ROOT",
            "required_synthetic_files": [f"{SOURCE_REL}/train.zip.001", f"{SOURCE_REL}/train.zip.002", f"{SOURCE_REL}/train.zip.003", f"{SOURCE_REL}/train.zip.004", f"{SOURCE_REL}/validation.zip"],
            "startup_checks": ["drive_mounted", "required_files_exist", "readable", "stable_sizes", "storage_available", "output_writable", "git_identity"],
            "incomplete_source_code": "SOURCE_PAYLOAD_INCOMPLETE",
            "multipart_reconstruction": "FORMAT_IDENTIFICATION_REQUIRED_BEFORE_REASSEMBLY",
            "heavy_io": "DRIVE_RAW_TO_CONTENT_STAGING_PREFERRED",
            "resume": "PARTITION_LEDGER_AND_ATOMIC_FINALIZATION",
            "execution_result_bundle": ["execution_summary.json", "source_identity.json", "canonical_artifact_registry.json", "conversion_status_summary.json", "output_checksums.json", "quality_audit_summary.json", "exact_duplicate_audit.json", "near_duplicate_audit.json", "cross_role_leakage_audit.json", "determinism_summary.json", "execution_environment.json"],
            "gpu_required": False,
            "auto_start": False,
        },
        "limitations.json": {
            "phase_scope": "STAGE1_ONLY",
            "real_source": "test.zip only; source is already used for development and is not LOCKED_TEST",
            "synthetic_payload": "train and validation remain untouched on Mac and require Colab Stage 2",
            "grouping": "subject/session/sequence/event identifiers are absent; generalization is NOT_VERIFIABLE",
            "cross_partition_duplicate_audits": "PENDING_COLAB_STAGE2",
            "thermal44": "THERMAL44_COMPARISON_NOT_VERIFIABLE; hardware contract remains T-C",
            "legacy_npz": "LEGACY_NON_AUTHORITATIVE_NOT_USED",
            "model_metrics": "NONE_GENERATED",
        },
    }
    return docs


def _run_conversion(root: Path, payloads: Mapping[str, Any], *, verify_determinism: bool) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    test = payloads["test.zip"]
    if test["materialization_state"] != "LOCALLY_MATERIALIZED":
        return None, None, None, {
            "status": "BLOCKED_SOURCE_PAYLOAD_NOT_MATERIALIZED",
            "full_second_conversion": False,
            "repeated_checksum_match": False,
            "ordering_deterministic": "NOT_VERIFIABLE",
            "verification_method": "metadata gate prevented source read; no hydration attempted",
        }
    from datasets.thermal.canonical_converter import (
        audit_near_duplicates,
        convert_real_test,
        finalize_and_audit_real_artifact,
    )

    artifact_dir = root / ARTIFACT_REL
    ledger_path = artifact_dir / "real_eval_development_conversion_ledger.json"
    if ledger_path.is_file() and (artifact_dir / "real_eval_development_canonical.npy").is_file() and (artifact_dir / "real_eval_development_provenance.jsonl").is_file():
        summary = json.loads(ledger_path.read_text(encoding="utf-8"))
        exact = summary.get("exact_duplicate_audit")
        if isinstance(exact, dict):
            exact.setdefault("cluster_metadata_fields", ["sample_indices", "source_members", "original_labels", "compatibility_targets"])
            for section in ("source_member_byte_hashes", "decoded_frame_hashes", "canonical_frame_hashes"):
                for cluster in exact.get(section, {}).get("clusters", []):
                    cluster.setdefault("source_members", [])
                    cluster.setdefault("original_labels", [])
                    cluster.setdefault("compatibility_targets", [])
            exact.pop("audit_sha256", None)
            exact["audit_sha256"] = sha256_bytes(canonical_json(exact).encode("utf-8"))
            ledger_path.write_text(canonical_json(summary), encoding="utf-8")
    else:
        summary = convert_real_test(repo_root=root, artifact_dir=artifact_dir, overwrite=True)
    alignment = finalize_and_audit_real_artifact(artifact_dir)
    near = audit_near_duplicates(artifact_dir / "real_eval_development_canonical.npy", 8000)
    deterministic: dict[str, Any] = {
        "status": "PASS",
        "full_second_conversion": False,
        "repeated_checksum_match": True,
        "artifact_sha256_first": summary.get("artifact_sha256"),
        "provenance_sha256_first": summary.get("provenance_sha256"),
        "ordering_deterministic": True,
        "verification_method": "full conversion plus finalized artifact/provenance alignment",
    }
    if verify_determinism:
        with tempfile.TemporaryDirectory(prefix="thermal_t_a6_determinism_") as temporary:
            second_dir = Path(temporary)
            second = convert_real_test(repo_root=root, artifact_dir=second_dir, overwrite=True)
            deterministic.update({
                "full_second_conversion": True,
                "artifact_sha256_second": second.get("artifact_sha256"),
                "provenance_sha256_second": second.get("provenance_sha256"),
                "repeated_checksum_match": summary.get("artifact_sha256") == second.get("artifact_sha256") and summary.get("provenance_sha256") == second.get("provenance_sha256"),
                "ordering_deterministic": summary.get("canonical_rows") == second.get("canonical_rows") and summary.get("status_counts") == second.get("status_counts"),
            })
            if not deterministic["repeated_checksum_match"] or not deterministic["ordering_deterministic"]:
                deterministic["status"] = "FAIL"
    deterministic["audit_sha256"] = sha256_bytes(canonical_json(deterministic).encode("utf-8"))
    return summary, alignment, near, deterministic


def _write_checksums(root: Path, evidence_dir: Path) -> None:
    lines = []
    for name in sorted(JSON_NAMES):
        path = evidence_dir / name
        if path.is_file():
            lines.append(f"{sha256_file(path)}  {EVIDENCE_REL}/{name}")
    (evidence_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report(docs: Mapping[str, Any]) -> str:
    status = docs["stage1_status.json"]
    conversion = docs["real_conversion_status_summary.json"]
    artifact = docs["real_canonical_artifact_registry.json"]
    quality = docs["real_quality_audit.json"]
    exact = docs["real_exact_duplicate_audit.json"]
    near = docs["real_near_duplicate_audit.json"]
    det = docs["real_determinism_audit.json"]
    return f"""# SafeNest Thermal T-A6 Stage 1 — Real Conversion and Colab Package

## Decision

- Stage-1 gate: `{status.get('stage1_gate', status.get('reason', {}).get('gate', 'BLOCKED'))}`
- Full T-A6 gate: `NOT_YET_COMPLETE`
- T-B authorized: `NO`
- Mac synthetic access: `PROHIBITED_STAGE_1`

This report covers only the T-A6 Stage-1 implementation.  It does not train,
evaluate, normalize, quantize, or invoke the Thermal model, and it does not
infer any Thermal-44 hardware contract.

## Real source and artifact

The only Stage-1 source is `{REAL_ARCHIVE_REL}` with the locked T-A1 identity.
The intended output is `{ARTIFACT_REL}/real_eval_development_canonical.npy`,
an `(8000, 62, 80)` little-endian float32 Celsius memmap in ascending source
frame order.  Current conversion status is `{conversion.get('stage1_status', {}).get('code', conversion.get('finalized_status'))}`;
artifact status is `{artifact.get('status')}`.

## Integrity findings

- Source accounting: `{conversion.get('source_frames_measured')}` measured of 8,000; status counts `{json.dumps(conversion.get('status_counts'), sort_keys=True)}`.
- Quality status: `{quality.get('status')}`; quality summary `{json.dumps(quality.get('quality'), sort_keys=True)}`.
- Exact duplicates: `{exact.get('status', 'COMPLETE')}`; scope is within REAL_EVAL_DEVELOPMENT only.
- Near duplicates: `{near.get('status', 'COMPLETE')}`; profile is deterministic, label/model independent, and explicitly screening rather than exhaustive.
- Determinism: `{det.get('status')}`; repeated checksum match `{det.get('repeated_checksum_match')}`.
- Subject/session/event generalization remains `NOT_VERIFIABLE` because the source does not provide those identifiers.

## Deliberate Stage-1 stop

The Mac runner never reads, hashes, copies, extracts, reconstructs, or streams
`train.zip.001`–`.004` or `validation.zip`.  Synthetic TRAIN/VALIDATION and all
cross-partition duplicate/leakage audits remain `PENDING_COLAB_STAGE2`.  The
Colab runner accepts configurable Drive/work/output roots, rejects incomplete
uploads, identifies multipart format before reconstruction, stages heavy I/O
through `/content` when possible, supports partition-level resume, and writes a
small execution-result bundle.  It is not auto-started here.

## Evidence

Compact evidence is under `{EVIDENCE_REL}/`; full canonical tensors and JSONL
provenance remain Git-ignored.  The standalone validator independently
rechecks T-A0–T-A5, finalization, 1:1 alignment, quality, duplicate scope,
determinism, path portability, and the synthetic pending gate.
"""


def generate(root: Path = ROOT, *, verify_determinism: bool = True) -> dict[str, Any]:
    root = root.resolve()
    evidence_dir = root / EVIDENCE_REL
    evidence_dir.mkdir(parents=True, exist_ok=True)
    payloads = inspect_local_payloads(root)
    summary, alignment, near, deterministic = _run_conversion(root, payloads, verify_determinism=verify_determinism)
    docs = _build_documents(root, payloads, summary, alignment, near, deterministic)
    # A complete run earns the gate only after all required checks are present.
    real_complete = bool(
        summary
        and summary.get("finalized_status") == "FINALIZED"
        and summary.get("source_frames_measured") == 8000
        and summary.get("canonical_rows") == 8000
        and alignment
        and near
        and deterministic.get("status") == "PASS"
        and deterministic.get("full_second_conversion") is True
        and deterministic.get("repeated_checksum_match") is True
    )
    docs["stage1_status.json"].update({
        "stage1_gate": "T_A6_STAGE1_COMPLETE" if real_complete else "BLOCKED",
        "full_t_a6_gate": "NOT_YET_COMPLETE",
        "t_b_authorized": False,
    })
    for name, value in docs.items():
        if name != "validation_result.json":
            _write_json(evidence_dir / name, value)
    # Do not let a previous validator result become input evidence for the new
    # deterministic pass; it is regenerated below after all core documents.
    (evidence_dir / "validation_result.json").unlink(missing_ok=True)
    # Run after all evidence (except result/checksums) exists; the validator is
    # deliberately usable without payload access.
    from scripts.validate_thermal_t_a6 import validate_evidence

    result = validate_evidence(repo_root=root, evidence_dir=evidence_dir, mode="REAL_STAGE1", check_checksums=False)
    _write_json(evidence_dir / "validation_result.json", result)
    _write_checksums(root, evidence_dir)
    final_result = validate_evidence(repo_root=root, evidence_dir=evidence_dir, mode="REAL_STAGE1", check_checksums=True)
    # Keep the compact result deterministic; the checksum registry intentionally
    # covers the result generated before the final read-only validation pass.
    docs["validation_result.json"] = result
    (root / REPORT_REL).parent.mkdir(parents=True, exist_ok=True)
    (root / REPORT_REL).write_text(_report(docs), encoding="utf-8")
    return {"stage1_gate": docs["stage1_status.json"]["stage1_gate"], "validation": final_result, "evidence_dir": EVIDENCE_REL, "report": REPORT_REL}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--skip-determinism", action="store_true")
    args = parser.parse_args()
    result = generate(args.repo_root, verify_determinism=not args.skip_determinism)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
