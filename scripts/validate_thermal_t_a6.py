#!/usr/bin/env python3
"""Standalone non-hydrating validator for Thermal T-A6.

``REAL_STAGE1`` validates the complete local real-test artifact when present,
or returns an explicit BLOCKED result when the owner-confirmed payload is not
currently materialized.  ``FULL_DATASET`` validates the compact Stage-2 bundle
emitted by the owner-started Colab runner without opening bulk payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EVIDENCE_REL = "datasets/thermal/manifests/T-A6_full_conversion_integrity"
ARTIFACT_REL = "datasets/thermal/artifacts/T-A6_real_eval_development"
REPORT_REL = "docs/reports/20260811_Codex_T-A6_Stage1_Thermal_Real_Conversion_Colab_01.md"
REQUIRED_JSON = [
    "canonical_dataset_contract.json", "real_source_partition_inventory.json",
    "real_conversion_status_summary.json", "real_canonical_artifact_registry.json",
    "sample_provenance_schema.json", "real_sample_index_summary.json",
    "real_label_alignment_summary.json", "real_quality_audit.json",
    "real_exact_duplicate_audit.json", "near_duplicate_profile.json",
    "real_near_duplicate_audit.json", "real_leakage_audit.json",
    "real_output_checksum_registry.json", "real_determinism_audit.json",
    "stage1_status.json", "colab_execution_contract.json", "limitations.json",
    "validation_result.json",
]
CORE_JSON = REQUIRED_JSON[:-1]
CHECKSUMS_NAME = "checksums.sha256"
PORTABLE_PATH_RE = re.compile(r"^(?!/)(?!~)(?!file://)(?![A-Za-z]:)(?!.*\\).+$")
MODEL_METRIC_KEYS = {"accuracy", "precision", "recall", "f1", "macro_f1", "confusion_matrix", "prediction_distribution", "loss", "auc"}
FULL_DATASET_PHASE = "T-A6_COLAB_STAGE2"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if not PORTABLE_PATH_RE.fullmatch(value):
        return False
    if "/Users/" in value or "/private/" in value or "iCloud" in value or value.startswith("/content/"):
        return False
    pure = PurePosixPath(value)
    return not pure.is_absolute() and ".." not in pure.parts and all(part not in {"", "."} for part in pure.parts)


def _load_documents(evidence_dir: Path, errors: list[dict[str, str]]) -> dict[str, Any]:
    docs: dict[str, Any] = {}
    for name in REQUIRED_JSON:
        path = evidence_dir / name
        if not path.is_file():
            if name == "validation_result.json":
                continue
            _error(errors, "REQUIRED_ARTIFACT_MISSING", name, "T-A6 compact evidence is missing.")
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _error(errors, "JSON_READ_FAILED", name, str(exc))
            continue
        docs[name] = value
        if path.read_text(encoding="utf-8") != canonical_json(value):
            _error(errors, "NONDETERMINISTIC_JSON", name, "JSON must use canonical sorted formatting.")
        for location, item in _walk(value, name):
            if isinstance(item, str):
                if item.startswith(("/", "~/", "file://")) or "/Users/" in item or "/private/" in item or "iCloud" in item or "\\" in item:
                    _error(errors, "NONPORTABLE_PATH", location, item)
                if "Thermal-44" in item and any(token in item.upper() for token in ("VERIFIED", "CONFIRMED", "VALIDATED")):
                    _error(errors, "UNSUPPORTED_THERMAL44_ASSERTION", location, item)
            if isinstance(item, dict):
                for key in item:
                    if str(key).lower() in MODEL_METRIC_KEYS:
                        _error(errors, "MODEL_METRIC_CONTAMINATION", f"{location}.{key}", "T-A6 Stage 1 must not contain model-performance metrics.")
    return docs


def _validate_checksums(evidence_dir: Path, errors: list[dict[str, str]]) -> None:
    path = evidence_dir / CHECKSUMS_NAME
    if not path.is_file():
        _error(errors, "CHECKSUM_REGISTRY_MISSING", CHECKSUMS_NAME, "Compact evidence checksum registry is missing.")
        return
    entries: dict[str, str] = {}
    previous = ""
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", line)
        if not match:
            _error(errors, "CHECKSUM_LINE_INVALID", f"{CHECKSUMS_NAME}:{number}", line)
            continue
        digest, relative = match.groups()
        if relative <= previous:
            _error(errors, "CHECKSUM_ORDER_NONDETERMINISTIC", f"{CHECKSUMS_NAME}:{number}", relative)
        previous = relative
        if not relative.startswith(EVIDENCE_REL + "/") or not _portable(relative):
            _error(errors, "CHECKSUM_PATH_NOT_PORTABLE", f"{CHECKSUMS_NAME}:{number}", relative)
        entries[relative] = digest
    for name in REQUIRED_JSON:
        path = evidence_dir / name
        if not path.is_file():
            continue
        relative = f"{EVIDENCE_REL}/{name}"
        if entries.get(relative) != sha256_file(path):
            _error(errors, "CHECKSUM_MISMATCH", relative, "Checksum coverage is missing or stale.")
    for relative in entries:
        if not relative.startswith(EVIDENCE_REL + "/"):
            _error(errors, "CHECKSUM_SCOPE_INVALID", relative, "Registry may cover only T-A6 evidence.")


def _run_predecessors(repo_root: Path, errors: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    calls = [
        ("T-A0", "scripts.validate_thermal_t_a0", "datasets/thermal/manifests/T-A0_source_identity", False),
        ("T-A1", "scripts.validate_thermal_t_a1", "datasets/thermal/manifests/T-A1_safe_reader_raw_unit_contract", False),
        ("T-A2", "scripts.validate_thermal_t_a2", "datasets/thermal/manifests/T-A2_geometry_calibration_canonical_frame", False),
        ("T-A3", "scripts.validate_thermal_t_a3", "datasets/thermal/manifests/T-A3_sequence_window_event_policy", False),
        ("T-A4", "scripts.validate_thermal_t_a4", "datasets/thermal/manifests/T-A4_label_semantics_ambiguity", False),
        ("T-A5", "scripts.validate_thermal_t_a5", "datasets/thermal/manifests/T-A5_grouping_immutable_split", False),
    ]
    for phase, module_name, relative, verify_payload in calls:
        try:
            module = __import__(module_name, fromlist=["validate_evidence"])
            validator = module.validate_evidence
            evidence_dir = repo_root / relative
            if phase == "T-A0":
                result = validator(evidence_dir, repo_root)
            else:
                result = validator(repo_root=repo_root, evidence_dir=evidence_dir, check_checksums=True, verify_real_payload=verify_payload)
            results[phase] = result
            if result.get("evidence_validation") != "PASS":
                _error(errors, f"{phase}_VALIDATION_FAILED", relative, str(result.get("overall_outcome", result.get("evidence_validation"))))
        except Exception as exc:
            results[phase] = {"evidence_validation": "FAIL", "overall_outcome": "NOT_VERIFIABLE", "error": str(exc)}
            _error(errors, f"{phase}_VALIDATOR_FAILED", relative, str(exc))
    return results


def _validate_contracts(docs: Mapping[str, Any], root: Path, mode: str, errors: list[dict[str, str]], warnings: list[dict[str, str]]) -> None:
    contract = docs.get("canonical_dataset_contract.json", {})
    required_contract = {
        "source_partition": "test", "source_domain": "REAL", "safenest_role": "REAL_EVAL_DEVELOPMENT",
        "geometry_profile_id": "G1_FIXED_ASPECT_CROP_BILINEAR", "canonical_shape": [62, 80],
        "canonical_dtype": "float32_little_endian", "canonical_unit": "CELSIUS",
        "normalization": "NONE_T_A6_PHYSICAL_ARTIFACT", "model_dependency": "NONE",
    }
    for key, expected in required_contract.items():
        if contract.get(key) != expected:
            _error(errors, "CANONICAL_CONTRACT_MISMATCH", f"canonical_dataset_contract.json:{key}", f"expected {expected!r}, got {contract.get(key)!r}")
    source = docs.get("real_source_partition_inventory.json", {})
    if source.get("expected_frame_count") != 8000 or source.get("safenest_role") != "REAL_EVAL_DEVELOPMENT":
        _error(errors, "REAL_SOURCE_CONTRACT_MISMATCH", "real_source_partition_inventory.json", "real test must remain an 8,000-frame development partition")
    if source.get("synthetic_payload_access") != "PROHIBITED_STAGE_1":
        _error(errors, "SYNTHETIC_ACCESS_POLICY_MISSING", "real_source_partition_inventory.json", "Mac Stage 1 synthetic access must be prohibited")
    schema = docs.get("sample_provenance_schema.json", {})
    if schema.get("no_object_array") is not True or schema.get("ordering") != "canonical_sample_index ascending":
        _error(errors, "PROVENANCE_SCHEMA_MISMATCH", "sample_provenance_schema.json", "bounded deterministic provenance schema is not locked")
    try:
        from datasets.thermal.canonical_converter import near_duplicate_profile

        if docs.get("near_duplicate_profile.json") != near_duplicate_profile():
            _error(errors, "NEAR_DUPLICATE_PROFILE_TAMPERED", "near_duplicate_profile.json", "near-duplicate profile differs from the frozen implementation")
    except Exception as exc:
        _error(errors, "NEAR_DUPLICATE_PROFILE_UNREADABLE", "near_duplicate_profile.json", str(exc))
    if docs.get("colab_execution_contract.json", {}).get("auto_start") is not False:
        _error(errors, "COLAB_AUTO_START_FORBIDDEN", "colab_execution_contract.json:auto_start", "Colab execution must be owner-started")
    leakage = docs.get("real_leakage_audit.json", {})
    for key in ("train_to_validation_exact", "train_to_real_exact", "validation_to_real_exact", "cross_domain_near_duplicates"):
        if leakage.get(key) != "PENDING_COLAB_STAGE2":
            _error(errors, "CROSS_ROLE_PENDING_STATE_INVALID", f"real_leakage_audit.json:{key}", "Stage 1 may not report cross-partition zero or complete")
    if leakage.get("subject_session_event_generalization") != "NOT_VERIFIABLE_SOURCE_PROVENANCE_ABSENT":
        _error(errors, "GROUPING_LIMITATION_MISSING", "real_leakage_audit.json:subject_session_event_generalization", "source grouping limitations must remain explicit")
    stage = docs.get("stage1_status.json", {})
    if stage.get("full_t_a6_gate") != "NOT_YET_COMPLETE" or stage.get("t_b_authorized") is not False:
        _error(errors, "DOWNSTREAM_GATE_ESCALATION", "stage1_status.json", "Stage 1 must not authorize full T-A6 or T-B")
    if mode == "FULL_DATASET":
        _error(errors, "FULL_DATASET_NOT_IMPLEMENTED_STAGE1", "mode", "FULL_DATASET requires Colab synthetic and cross-role artifacts")
        return
    runner = root / "scripts/run_thermal_t_a6_colab.py"
    if not runner.is_file():
        _error(errors, "COLAB_RUNNER_MISSING", "scripts/run_thermal_t_a6_colab.py", "Stage-2 runner is required")
    elif "/Users/" in runner.read_text(encoding="utf-8") or "MyDrive/" in runner.read_text(encoding="utf-8"):
        _error(errors, "COLAB_PERSONAL_PATH_HARDCODED", "scripts/run_thermal_t_a6_colab.py", "runner contains a personal path")
    limitations = docs.get("limitations.json", {})
    if limitations.get("phase_scope") != "STAGE1_ONLY":
        _error(errors, "PHASE_SCOPE_MISMATCH", "limitations.json:phase_scope", "T-A6 Stage 1 scope is required")
    if limitations.get("legacy_npz") != "LEGACY_NON_AUTHORITATIVE_NOT_USED":
        _error(errors, "LEGACY_NPZ_AUTHORITY_ESCALATION", "limitations.json:legacy_npz", "legacy NPZ cannot be canonical evidence")
    if not (root / "datasets/thermal/processed_thermal_80x62.npz").exists():
        # The file is intentionally ignored and may be absent in a clean
        # checkout; absence is not a conversion failure.
        _warning(warnings, "LEGACY_NPZ_NOT_PRESENT_IN_WORKTREE", "datasets/thermal/processed_thermal_80x62.npz", "legacy NPZ was not used")


def _validate_real_artifact(docs: Mapping[str, Any], root: Path, errors: list[dict[str, str]], warnings: list[dict[str, str]]) -> bool:
    summary = docs.get("real_conversion_status_summary.json", {})
    registry = docs.get("real_canonical_artifact_registry.json", {})
    stage = docs.get("stage1_status.json", {})
    complete = stage.get("stage1_gate") == "T_A6_STAGE1_COMPLETE"
    if not complete:
        _warning(warnings, "REAL_STAGE1_BLOCKED", "stage1_status.json", str(stage.get("reason", "real artifact is not finalized")))
        return False
    counts = summary.get("status_counts", {})
    if summary.get("source_frames_measured") != 8000 or summary.get("canonical_rows") != 8000:
        _error(errors, "REAL_SOURCE_ACCOUNTING_INVALID", "real_conversion_status_summary.json", "complete Stage 1 requires 8,000 measured and canonical rows")
    if sum(int(counts.get(key, 0)) for key in ("SUCCESS", "SUCCESS_WITH_WARNING", "EXCLUDED", "FAILED")) != 8000:
        _error(errors, "REAL_STATUS_RECONCILIATION_INVALID", "real_conversion_status_summary.json:status_counts", "status counts must sum to 8,000")
    if int(counts.get("SUCCESS", 0)) + int(counts.get("SUCCESS_WITH_WARNING", 0)) != int(summary.get("canonical_rows", -1)):
        _error(errors, "CANONICAL_ROW_RECONCILIATION_INVALID", "real_conversion_status_summary.json", "canonical rows must equal successful rows")
    if registry.get("finalized_status") != "FINALIZED" or registry.get("checksum_status") != "LOCKED":
        _error(errors, "REAL_ARTIFACT_NOT_FINALIZED", "real_canonical_artifact_registry.json", "finalization/checksum lock is missing")
        return False
    artifact = root / ARTIFACT_REL / "real_eval_development_canonical.npy"
    provenance = root / ARTIFACT_REL / "real_eval_development_provenance.jsonl"
    ledger = root / ARTIFACT_REL / "real_eval_development_conversion_ledger.json"
    for path in (artifact, provenance, ledger):
        if not path.is_file():
            _error(errors, "REAL_BULK_ARTIFACT_MISSING", path.relative_to(root).as_posix(), "finalized bulk artifact is missing")
    if errors:
        return False
    try:
        ledger_value = json.loads(ledger.read_text(encoding="utf-8"))
        if ledger_value.get("finalized_status") != "FINALIZED":
            _error(errors, "REAL_LEDGER_NOT_FINALIZED", ledger.relative_to(root).as_posix(), "conversion ledger is not finalized")
    except (OSError, json.JSONDecodeError) as exc:
        _error(errors, "REAL_LEDGER_INVALID", ledger.relative_to(root).as_posix(), str(exc))
        return False
    if registry.get("artifact_sha256") != sha256_file(artifact):
        _error(errors, "REAL_ARTIFACT_CHECKSUM_MISMATCH", "real_canonical_artifact_registry.json:artifact_sha256", "measured canonical tensor hash differs")
    if registry.get("provenance_sha256") != sha256_file(provenance):
        _error(errors, "REAL_PROVENANCE_CHECKSUM_MISMATCH", "real_canonical_artifact_registry.json:provenance_sha256", "measured provenance hash differs")
    output_checksums = docs.get("real_output_checksum_registry.json", {})
    if output_checksums.get("ledger_sha256") != sha256_file(ledger):
        _error(errors, "REAL_LEDGER_CHECKSUM_MISMATCH", "real_output_checksum_registry.json:ledger_sha256", "measured ledger hash differs")
    if output_checksums.get("artifact_sha256") != sha256_file(artifact) or output_checksums.get("provenance_sha256") != sha256_file(provenance):
        _error(errors, "REAL_OUTPUT_CHECKSUM_REGISTRY_MISMATCH", "real_output_checksum_registry.json", "output checksum registry does not match measured files")
    try:
        import numpy as np
        frames = np.load(artifact, mmap_mode="r")
        if tuple(frames.shape) != (8000, 62, 80) or frames.dtype != np.dtype("<f4"):
            _error(errors, "REAL_ARTIFACT_SHAPE_DTYPE_INVALID", artifact.relative_to(root).as_posix(), f"got shape={frames.shape}, dtype={frames.dtype}")
        if not np.all(np.isfinite(frames)):
            _error(errors, "REAL_ARTIFACT_NONFINITE", artifact.relative_to(root).as_posix(), "canonical output contains non-finite values")
    except Exception as exc:
        _error(errors, "REAL_ARTIFACT_READ_FAILED", artifact.relative_to(root).as_posix(), str(exc))
        return False
    try:
        from datasets.thermal.canonical_converter import verify_provenance_alignment

        alignment = verify_provenance_alignment(artifact, provenance, 8000)
        if not all(alignment.get(key) is True for key in ("tensor_provenance_1_to_1", "tensor_label_1_to_1", "tensor_assignment_1_to_1")):
            _error(errors, "ONE_TO_ONE_ALIGNMENT_FAILED", provenance.relative_to(root).as_posix(), str(alignment))
    except Exception as exc:
        _error(errors, "ONE_TO_ONE_ALIGNMENT_FAILED", provenance.relative_to(root).as_posix(), str(exc))
    exact = docs.get("real_exact_duplicate_audit.json", {})
    if not exact.get("audit_scope") == "WITHIN_REAL_EVAL_DEVELOPMENT":
        _error(errors, "EXACT_AUDIT_SCOPE_INVALID", "real_exact_duplicate_audit.json", "exact audit must cover the real role")
    for key in ("source_member_byte_hashes", "decoded_frame_hashes", "canonical_frame_hashes"):
        if key not in exact:
            _error(errors, "EXACT_AUDIT_INCOMPLETE", f"real_exact_duplicate_audit.json:{key}", "all three exact hash layers are required")
    if isinstance(exact.get("audit_sha256"), str):
        exact_without_hash = dict(exact); exact_without_hash.pop("audit_sha256", None)
        if exact["audit_sha256"] != hashlib.sha256(canonical_json(exact_without_hash).encode("utf-8")).hexdigest():
            _error(errors, "EXACT_AUDIT_CHECKSUM_MISMATCH", "real_exact_duplicate_audit.json:audit_sha256", "exact audit checksum is stale")
    quality = docs.get("real_quality_audit.json", {})
    if quality.get("status") != "PASS" or quality.get("quality", {}).get("silent_skips") != 0:
        _error(errors, "QUALITY_AUDIT_INCOMPLETE", "real_quality_audit.json", "quality audit must be complete and must not hide silent skips")
    near = docs.get("real_near_duplicate_audit.json", {})
    if near.get("audit_scope") != "WITHIN_REAL_EVAL_DEVELOPMENT" or near.get("exhaustiveness_claim") != "DETERMINISTIC_SCREENING_NOT_EXHAUSTIVE":
        _error(errors, "NEAR_AUDIT_SCOPE_INVALID", "real_near_duplicate_audit.json", "near audit scope/profile is not locked")
    if int(near.get("confirmed_pairs_total", near.get("confirmed_pair_count", 0))) < len(near.get("confirmed_pairs", [])):
        _error(errors, "NEAR_AUDIT_WITNESS_COUNT_INVALID", "real_near_duplicate_audit.json", "bounded pair witnesses exceed the reported total")
    if isinstance(near.get("audit_sha256"), str):
        near_without_hash = dict(near); near_without_hash.pop("audit_sha256", None)
        if near["audit_sha256"] != hashlib.sha256(canonical_json(near_without_hash).encode("utf-8")).hexdigest():
            _error(errors, "NEAR_AUDIT_CHECKSUM_MISMATCH", "real_near_duplicate_audit.json:audit_sha256", "near audit checksum is stale")
    deterministic = docs.get("real_determinism_audit.json", {})
    if deterministic.get("status") != "PASS" or deterministic.get("repeated_checksum_match") is not True or deterministic.get("full_second_conversion") is not True:
        _error(errors, "DETERMINISM_AUDIT_INCOMPLETE", "real_determinism_audit.json", "full second conversion and matching checksums are required")
    return not errors


def _validate_full_dataset(
    *,
    repo_root: Path,
    evidence_dir: Path,
    errors: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate the live Stage-2 compact bundle and inherited predecessors.

    The compact validator is intentionally called on the current evidence
    directory rather than trusting its persisted ``validation_result.json``.
    This keeps the full validator fail-closed if any Stage-2 evidence file is
    changed after a prior validation run.
    """

    predecessors = _run_predecessors(repo_root, errors)
    try:
        from datasets.thermal.t_a6_stage2 import validate_stage2_bundle

        stage2 = validate_stage2_bundle(evidence_dir, require_validation_result=True)
    except Exception as exc:
        stage2 = {
            "schema_version": "1.0",
            "phase": FULL_DATASET_PHASE,
            "evidence_validation": "FAIL",
            "overall_outcome": "NOT_VERIFIABLE",
            "full_t_a6_gate": "NOT_YET_COMPLETE",
            "t_b_authorized": False,
            "error_count": 1,
            "warning_count": 0,
            "errors": [{"code": "STAGE2_VALIDATOR_EXCEPTION", "location": "evidence_dir", "message": str(exc)}],
            "warnings": [],
        }

    for item in stage2.get("errors", []):
        if not isinstance(item, Mapping):
            _error(errors, "STAGE2_VALIDATION_FAILED", "T-A6_execution_result", str(item))
            continue
        _error(
            errors,
            f"STAGE2_{item.get('code', 'VALIDATION_FAILED')}",
            f"T-A6_execution_result/{item.get('location', '')}",
            str(item.get("message", "Stage-2 compact evidence validation failed.")),
        )
    return stage2, predecessors


def validate_evidence(*, repo_root: Path = ROOT, evidence_dir: Path | None = None, mode: str = "REAL_STAGE1", check_checksums: bool = True) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    evidence_dir = (evidence_dir or repo_root / EVIDENCE_REL).resolve()
    if mode == "FULL_DATASET":
        errors: list[dict[str, str]] = []
        stage2, predecessors = _validate_full_dataset(repo_root=repo_root, evidence_dir=evidence_dir, errors=errors)
        predecessor_pass = all(result.get("evidence_validation") == "PASS" for result in predecessors.values()) and len(predecessors) == 6
        stage2_pass = stage2.get("evidence_validation") == "PASS"
        sorted_errors = sorted(errors, key=lambda item: (item["code"], item["location"], item["message"]))
        sorted_warnings = sorted(stage2.get("warnings", []), key=lambda item: (str(item.get("code", "")), str(item.get("location", "")), str(item.get("message", ""))))
        gate = not sorted_errors and predecessor_pass and stage2_pass
        predecessor_summary = {
            phase: {
                "evidence_validation": result.get("evidence_validation", "FAIL"),
                "overall_outcome": result.get("overall_outcome", "NOT_VERIFIABLE"),
            }
            for phase, result in sorted(predecessors.items())
        }
        return {
            "schema_version": "1.0",
            "phase": FULL_DATASET_PHASE,
            "mode": mode,
            "evidence_validation": "PASS" if gate else "FAIL",
            "overall_outcome": "PASS_WITH_LIMITATIONS" if gate else "NOT_VERIFIABLE",
            "stage1_gate": "T_A6_STAGE1_COMPLETE" if stage2_pass and predecessor_pass else "NOT_YET_COMPLETE",
            "stage1_gate_source": "REAL_EVAL_DEVELOPMENT role independently validated inside the Stage-2 bundle",
            "full_t_a6_gate": "T_A6_FULL_COMPLETE_WITH_LIMITATIONS" if gate else "NOT_YET_COMPLETE",
            "t_b_authorized": False,
            "predecessors": predecessor_summary,
            "stage2_validation": stage2,
            "error_count": len(sorted_errors),
            "warning_count": len(sorted_warnings),
            "errors": sorted_errors,
            "warnings": sorted_warnings,
        }
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if mode not in {"REAL_STAGE1", "FULL_DATASET"}:
        _error(errors, "MODE_INVALID", "mode", mode)
    predecessors = _run_predecessors(repo_root, errors)
    docs = _load_documents(evidence_dir, errors)
    if len(docs) >= len(CORE_JSON):
        _validate_contracts(docs, repo_root, mode, errors, warnings)
    real_complete = False
    if mode == "REAL_STAGE1" and len(docs) >= len(CORE_JSON):
        real_complete = _validate_real_artifact(docs, repo_root, errors, warnings)
    elif mode == "FULL_DATASET":
        warnings.append({"code": "FULL_DATASET_RESERVED", "location": "mode", "message": "synthetic TRAIN/VALIDATION and cross-role artifacts are not available in Stage 1"})
    if check_checksums and len(docs) >= len(CORE_JSON):
        _validate_checksums(evidence_dir, errors)
    sorted_errors = sorted(errors, key=lambda item: (item["code"], item["location"], item["message"]))
    sorted_warnings = sorted(warnings, key=lambda item: (item["code"], item["location"], item["message"]))
    predecessor_pass = all(result.get("evidence_validation") == "PASS" for result in predecessors.values()) and len(predecessors) == 6
    if mode == "FULL_DATASET":
        outcome = "NOT_VERIFIABLE" if not sorted_errors else "NOT_VERIFIABLE"
        gate = "NOT_YET_COMPLETE"
    elif sorted_errors:
        outcome = "NOT_VERIFIABLE"
        gate = "BLOCKED" if not real_complete else "NOT_VERIFIABLE"
    elif not predecessor_pass:
        outcome = "NOT_VERIFIABLE"
        gate = "BLOCKED"
    elif real_complete:
        outcome = "PASS_WITH_LIMITATIONS"
        gate = "T_A6_STAGE1_COMPLETE"
    else:
        outcome = "BLOCKED"
        gate = "BLOCKED"
    return {
        "schema_version": "1.0", "phase": "T-A6_STAGE1", "mode": mode,
        "evidence_validation": "PASS" if not sorted_errors and predecessor_pass else "FAIL",
        "overall_outcome": outcome, "stage1_gate": gate,
        "full_t_a6_gate": "NOT_YET_COMPLETE", "t_b_authorized": False,
        "predecessors": {phase: {"evidence_validation": result.get("evidence_validation", "FAIL"), "overall_outcome": result.get("overall_outcome", "NOT_VERIFIABLE")} for phase, result in sorted(predecessors.items())},
        "error_count": len(sorted_errors), "warning_count": len(sorted_warnings),
        "errors": sorted_errors, "warnings": sorted_warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--evidence-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=("REAL_STAGE1", "FULL_DATASET"), default="REAL_STAGE1")
    parser.add_argument("--skip-checksums", action="store_true")
    args = parser.parse_args()
    result = validate_evidence(repo_root=args.repo_root, evidence_dir=args.evidence_dir, mode=args.mode, check_checksums=not args.skip_checksums)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    raise SystemExit(0 if result["evidence_validation"] == "PASS" else 1)


if __name__ == "__main__":
    main()
