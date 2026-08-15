#!/usr/bin/env python3
"""Guarded, non-interactive Colab execution package for Thermal T-A6.

This module performs startup and multipart safety checks before any synthetic
payload is opened.  It is intentionally usable as a dry-run on a Mac or CI;
synthetic execution is accepted only in ``COLAB_STAGE2`` mode and only after
all required Drive files pass the completeness gates.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.thermal.canonical_converter import sha256_file
from datasets.thermal.t_a6_stage2 import ROLE_ORDER, Stage2AuditError

SYNTHETIC_PARTS = ("train.zip.001", "train.zip.002", "train.zip.003", "train.zip.004")
REQUIRED_SYNTHETIC = SYNTHETIC_PARTS + ("validation.zip",)
EXPECTED_SOURCE_SIZES = {
    "train.zip.001": 4_194_304_000,
    "train.zip.002": 4_194_304_000,
    "train.zip.003": 4_194_304_000,
    "train.zip.004": 1_408_015_891,
    "validation.zip": 3_492_475_558,
    "test.zip": 1_740_348_425,
}
INCOMPLETE_CODE = "SOURCE_PAYLOAD_INCOMPLETE"
STORAGE_CODE = "COLAB_STORAGE_INSUFFICIENT"
FORMAT_CODE = "MULTIPART_FORMAT_UNVERIFIED"


class ColabExecutionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.detail = message
        super().__init__(f"{code}: {message}")


class SyntheticMacAccessError(ColabExecutionError):
    def __init__(self, message: str) -> None:
        super().__init__("MAC_SYNTHETIC_PAYLOAD_ACCESS_PROHIBITED", message)


@dataclass(frozen=True)
class FileObservation:
    logical_path: str
    path_exists: bool
    size_bytes: int | None
    mode: int | None
    first_signature_hex: str | None
    last_signature_hex: str | None
    readable: bool
    stable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_path": self.logical_path,
            "path_exists": self.path_exists,
            "size_bytes": self.size_bytes,
            "mode": self.mode,
            "first_signature_hex": self.first_signature_hex,
            "last_signature_hex": self.last_signature_hex,
            "readable": self.readable,
            "stable": self.stable,
        }


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(_json(value), encoding="utf-8")
    os.replace(partial, path)


def _logical(path: Path, root: Path | None = None) -> str:
    if root is not None:
        try:
            return path.absolute().relative_to(root.absolute()).as_posix()
        except ValueError:
            pass
    return path.name


def _read_signature(path: Path, *, offset: int = 0, length: int = 16) -> bytes:
    with path.open("rb") as handle:
        if offset:
            handle.seek(offset)
        return handle.read(length)


def stable_file_observation(path: Path, *, logical_root: Path | None = None) -> FileObservation:
    """Observe bounded signatures and two stat snapshots without full reads."""
    try:
        first = path.stat()
    except (FileNotFoundError, OSError):
        return FileObservation(_logical(path, logical_root), False, None, None, None, None, False, False)
    if not path.is_file() or first.st_size <= 0:
        return FileObservation(_logical(path, logical_root), True, int(first.st_size), int(first.st_mode), None, None, False, False)
    try:
        first_bytes = _read_signature(path)
        last_offset = max(0, int(first.st_size) - 22)
        last_bytes = _read_signature(path, offset=last_offset, length=22)
        second = path.stat()
    except (OSError, ValueError):
        return FileObservation(_logical(path, logical_root), True, int(first.st_size), int(first.st_mode), None, None, False, False)
    return FileObservation(
        logical_path=_logical(path, logical_root),
        path_exists=True,
        size_bytes=int(first.st_size),
        mode=int(first.st_mode),
        first_signature_hex=first_bytes.hex(),
        last_signature_hex=last_bytes.hex(),
        readable=True,
        stable=(first.st_size == second.st_size and first.st_mtime_ns == second.st_mtime_ns),
    )


def check_required_files(paths: Sequence[Path], *, logical_root: Path | None = None) -> list[FileObservation]:
    observations = [stable_file_observation(path, logical_root=logical_root) for path in paths]
    failures = [item for item in observations if not item.path_exists or not item.readable or not item.stable or not item.size_bytes]
    if failures:
        details = ", ".join(item.logical_path for item in failures)
        raise ColabExecutionError(INCOMPLETE_CODE, f"required source files are absent, unreadable, zero-byte, or changing: {details}")
    return observations


def check_storage(work_root: Path, required_bytes: int, *, safety_factor: float = 1.20) -> dict[str, Any]:
    usage = shutil.disk_usage(work_root)
    required_with_margin = int(required_bytes * safety_factor)
    result = {
        "work_root": _logical(work_root),
        "free_bytes": int(usage.free),
        "required_bytes": int(required_bytes),
        "required_with_margin_bytes": required_with_margin,
        "safety_factor": safety_factor,
        "pass": int(usage.free) >= required_with_margin,
    }
    if not result["pass"]:
        raise ColabExecutionError(STORAGE_CODE, _json(result).strip())
    return result


def check_output_writable(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    probe = output_root / ".thermal_t_a6_write_probe.partial"
    try:
        probe.write_bytes(b"T-A6")
        probe.unlink()
    except OSError as exc:
        raise ColabExecutionError("OUTPUT_NOT_WRITABLE", str(exc)) from exc
    return {"path": _logical(output_root), "writable": True}


def inspect_multipart_format(parts: Sequence[Path]) -> dict[str, Any]:
    """Classify multipart bytes before any reconstruction.

    Only bounded signatures are read.  The result is ``INDEPENDENT_ZIPS`` when
    every piece is a self-contained ZIP, ``RAW_BYTE_SPLIT_CANDIDATE`` when only
    the first piece starts with ZIP and the final piece ends with EOCD, and
    ``UNKNOWN`` otherwise.  A candidate still needs archive validation before
    reconstruction is allowed.
    """
    if not parts:
        raise ColabExecutionError(FORMAT_CODE, "no multipart parts supplied")
    observations = check_required_files(parts)
    starts = [bytes.fromhex(item.first_signature_hex or "") for item in observations]
    ends = [bytes.fromhex(item.last_signature_hex or "") for item in observations]
    zip_start = b"PK\x03\x04"
    zip_eocd = b"PK\x05\x06"
    if all(start.startswith(zip_start) for start in starts):
        fmt = "INDEPENDENT_ZIPS"
        reconstruction = "FORBIDDEN_UNLESS_ARCHIVE_MANIFEST_PROVES_JOIN"
    elif starts[0].startswith(zip_start) and all(not start.startswith(zip_start) for start in starts[1:]) and ends[-1].find(zip_eocd) >= 0:
        fmt = "RAW_BYTE_SPLIT_CANDIDATE"
        reconstruction = "ALLOWED_AFTER_LOGICAL_ARCHIVE_VALIDATION"
    else:
        fmt = "UNKNOWN"
        reconstruction = "REJECT"
    return {
        "format": fmt,
        "reconstruction_policy": reconstruction,
        "part_count": len(parts),
        "observations": [item.to_dict() for item in observations],
        "signature_only": True,
    }


def reconstruct_raw_byte_split(parts: Sequence[Path], output_path: Path, *, format_report: Mapping[str, Any], chunk_size: int = 8 * 1024 * 1024) -> dict[str, Any]:
    if format_report.get("format") != "RAW_BYTE_SPLIT_CANDIDATE":
        raise ColabExecutionError(FORMAT_CODE, "raw reconstruction requested without a verified raw-byte-split candidate")
    check_required_files(parts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    if partial.exists():
        partial.unlink()
    total = 0
    import hashlib

    digest = hashlib.sha256()
    with partial.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                while True:
                    chunk = source.read(chunk_size)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
        output.flush()
        os.fsync(output.fileno())
    try:
        with zipfile.ZipFile(partial, "r") as archive:
            bad = archive.testzip()
            if bad is not None:
                raise ColabExecutionError("RECONSTRUCTED_ARCHIVE_CORRUPT", bad)
            member_count = len(archive.infolist())
    except (zipfile.BadZipFile, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise ColabExecutionError("RECONSTRUCTED_ARCHIVE_INVALID", str(exc)) from exc
    os.replace(partial, output_path)
    return {"path": _logical(output_path), "size_bytes": total, "sha256": digest.hexdigest(), "member_count": member_count, "finalized": True}


def load_resume_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": "1.0", "phase": "T-A6_COLAB_STAGE2", "partitions": {}, "finalized": False}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ColabExecutionError("RESUME_LEDGER_INVALID", str(exc)) from exc
    if value.get("finalized") is True and not value.get("execution_result_bundle"):
        raise ColabExecutionError("RESUME_LEDGER_INVALID", "finalized ledger lacks execution-result bundle")
    return value


def save_resume_ledger(path: Path, ledger: Mapping[str, Any]) -> None:
    _atomic_json(path, dict(ledger))


def git_identity(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"), "user_name": run("config", "user.name"), "user_email": run("config", "user.email")}


def startup_checks(*, drive_raw_root: Path, work_root: Path, drive_output_root: Path, repo_root: Path = ROOT, required_bytes: int | None = None, include_real_test: bool = False) -> dict[str, Any]:
    drive_raw_root = drive_raw_root.expanduser()
    work_root = work_root.expanduser()
    drive_output_root = drive_output_root.expanduser()
    work_root.mkdir(parents=True, exist_ok=True)
    drive_output_root.mkdir(parents=True, exist_ok=True)
    names = list(REQUIRED_SYNTHETIC)
    if include_real_test:
        names.append("test.zip")
    observations = check_required_files([drive_raw_root / name for name in names], logical_root=drive_raw_root)
    size_mismatches = [
        f"{item.logical_path}: expected={EXPECTED_SOURCE_SIZES[item.logical_path]}, measured={item.size_bytes}"
        for item in observations
        if item.logical_path in EXPECTED_SOURCE_SIZES and item.size_bytes != EXPECTED_SOURCE_SIZES[item.logical_path]
    ]
    if size_mismatches:
        raise ColabExecutionError("SOURCE_PAYLOAD_SIZE_MISMATCH", "; ".join(size_mismatches))
    if required_bytes is None:
        required_bytes = sum(int(item.size_bytes or 0) for item in observations) + 4 * 1024 * 1024 * 1024
    storage = check_storage(work_root, required_bytes)
    output = check_output_writable(drive_output_root)
    return {
        "drive_mounted": drive_raw_root.exists(),
        "drive_raw_root": _logical(drive_raw_root),
        "work_root": _logical(work_root),
        "drive_output_root": _logical(drive_output_root),
        "required_files": [item.to_dict() for item in observations],
        "storage": storage,
        "output": output,
        "git": git_identity(repo_root),
        "python": sys.version,
        "numpy": _package_version("numpy"),
        "pillow": _package_version("PIL"),
        "gpu_required": False,
    }


def _package_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def build_execution_result_bundle(output_root: Path, *, startup: Mapping[str, Any], status: str, message: str, ledger: Mapping[str, Any]) -> Path:
    bundle = output_root / "T-A6_execution_result"
    bundle.mkdir(parents=True, exist_ok=True)
    _atomic_json(bundle / "execution_summary.json", {"phase": "T-A6_COLAB_STAGE2", "status": status, "message": message, "startup": startup, "ledger": dict(ledger)})
    _atomic_json(bundle / "execution_environment.json", {"python": sys.version, "git": startup.get("git", {}), "numpy": startup.get("numpy"), "pillow": startup.get("pillow"), "gpu_required": False})
    _atomic_json(bundle / "source_identity.json", {"source_paths_are_logical": True, "required_files": startup.get("required_files", [])})
    return bundle


def _copy_to_work(source: Path, target: Path) -> dict[str, Any]:
    """Copy one stable Drive file to ``/content`` with atomic finalization."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        partial.unlink()
    total = 0
    import hashlib

    digest = hashlib.sha256()
    with source.open("rb") as input_file, partial.open("wb") as output_file:
        while True:
            chunk = input_file.read(8 * 1024 * 1024)
            if not chunk:
                break
            output_file.write(chunk)
            digest.update(chunk)
            total += len(chunk)
        output_file.flush()
        os.fsync(output_file.fileno())
    if total != int(source.stat().st_size):
        partial.unlink(missing_ok=True)
        raise ColabExecutionError("STAGED_SOURCE_SIZE_MISMATCH", f"{source.name}: {total} != {source.stat().st_size}")
    os.replace(partial, target)
    return {"path": _logical(target), "size_bytes": total, "sha256": digest.hexdigest()}


def _ensure_train_reconstruction(
    *,
    parts: Sequence[Path],
    staged_path: Path,
    format_report: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Reuse a finalized staged archive after a disconnect when its identity matches."""
    previous = dict(ledger.get("partitions", {}).get("train", {}).get("reconstructed", {}))
    if staged_path.is_file() and previous.get("size_bytes") == staged_path.stat().st_size:
        from datasets.thermal.canonical_converter import sha256_file

        digest = sha256_file(staged_path)
        if previous.get("sha256") == digest:
            return {**previous, "path": _logical(staged_path), "resumed": True}
    return reconstruct_raw_byte_split(parts, staged_path, format_report=format_report)


def _stage_real_archive(repo_root: Path, source_archive: Path) -> tuple[Path, bool]:
    """Expose Drive ``test.zip`` at the locked repository-relative reader path."""
    locked = repo_root / "datasets/raw_archives/thermal_split_zips/test.zip"
    locked.parent.mkdir(parents=True, exist_ok=True)
    if locked.exists() or locked.is_symlink():
        if locked.is_symlink() and locked.resolve() == source_archive.resolve():
            return locked, False
        raise ColabExecutionError("REAL_SOURCE_PATH_OCCUPIED", str(locked))
    locked.symlink_to(source_archive)
    return locked, True


def _stage2_bundle_json(bundle: Path, name: str, value: Mapping[str, Any]) -> None:
    _atomic_json(bundle / name, dict(value))


def _reuse_or_convert(
    *,
    artifact_root: Path,
    stem: str,
    converter: Any,
) -> dict[str, Any]:
    """Resume a finalized partition without re-reading its source archive."""
    artifact = artifact_root / f"{stem}_canonical.npy"
    provenance = artifact_root / f"{stem}_provenance.jsonl"
    ledger = artifact_root / f"{stem}_conversion_ledger.json"
    existing = [artifact, provenance, ledger]
    if any(path.exists() for path in existing):
        if not all(path.is_file() for path in existing):
            raise ColabExecutionError("PARTIAL_ARTIFACT_PRESENT", stem)
        try:
            summary = json.loads(ledger.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ColabExecutionError("PARTITION_LEDGER_INVALID", f"{stem}: {exc}") from exc
        if summary.get("finalized_status") != "FINALIZED":
            raise ColabExecutionError("PARTITION_NOT_FINALIZED", stem)
        if summary.get("artifact_sha256") != sha256_file(artifact) or summary.get("provenance_sha256") != sha256_file(provenance):
            raise ColabExecutionError("PARTITION_CHECKSUM_MISMATCH", stem)
        return {**summary, "resumed": True}
    return converter()


def run(*, mode: str, drive_raw_root: Path, work_root: Path, drive_output_root: Path, repo_root: Path = ROOT, dry_run: bool = True, include_real_test: bool = False) -> dict[str, Any]:
    from datasets.thermal.canonical_converter import (
        COLAB_STAGE2_MODE,
        STAGE1_MODE,
        ConversionConfig,
        convert_partition,
        sha256_file,
    )
    from datasets.thermal.t_a6_stage2 import (
        Stage2AuditError,
        audit_cross_role_leakage,
        audit_exact_duplicates,
        audit_near_duplicates_cross_role,
        build_checksums,
        build_conversion_status,
        build_output_checksums,
        build_quality_audit,
        build_role_registry,
        validate_role_artifact,
        validate_stage2_bundle,
        verify_synthetic_source_contract,
    )

    if mode == "MAC_STAGE1" and not include_real_test:
        raise SyntheticMacAccessError("MAC_STAGE1 cannot inspect synthetic train/validation files")
    if mode not in {"COLAB_STAGE2", "MAC_STAGE1"}:
        raise ColabExecutionError("MODE_INVALID", mode)
    startup = startup_checks(
        drive_raw_root=drive_raw_root,
        work_root=work_root,
        drive_output_root=drive_output_root,
        repo_root=repo_root,
        include_real_test=include_real_test,
    )
    ledger_path = drive_output_root / "T-A6_execution_result" / "resume_ledger.json"
    ledger = load_resume_ledger(ledger_path)
    if dry_run:
        bundle = build_execution_result_bundle(drive_output_root, startup=startup, status="DRY_RUN_READY", message="startup and source/storage guards passed; no conversion started", ledger=ledger)
        return {"status": "DRY_RUN_READY", "bundle": _logical(bundle), "startup": startup}
    if ledger.get("finalized") is True:
        bundle = drive_output_root / "T-A6_execution_result"
        final_result = validate_stage2_bundle(bundle, require_validation_result=True)
        if final_result.get("evidence_validation") == "PASS":
            return {"status": "ALREADY_FINALIZED", "bundle": _logical(bundle), "startup": startup, "validation_result": final_result}
        raise ColabExecutionError("RESUME_LEDGER_INVALID", json.dumps(final_result, ensure_ascii=False, sort_keys=True))
    if mode != "COLAB_STAGE2":
        raise SyntheticMacAccessError("synthetic execution is available only in COLAB_STAGE2 mode")
    if not include_real_test:
        raise ColabExecutionError("REAL_REFERENCE_REQUIRED", "full Stage 2 requires test.zip as REAL_EVAL_DEVELOPMENT reference; pass --include-real-test")
    train_parts = [drive_raw_root / name for name in SYNTHETIC_PARTS]
    format_report = inspect_multipart_format(train_parts)
    if format_report.get("format") != "RAW_BYTE_SPLIT_CANDIDATE":
        raise ColabExecutionError(FORMAT_CODE, "train multipart format is not a verified raw-byte split; no blind concatenation or conversion was attempted")
    ledger = dict(ledger)
    ledger.setdefault("partitions", {})
    ledger["partitions"]["train"] = {**ledger["partitions"].get("train", {}), "format_report": format_report, "status": "FORMAT_IDENTIFIED"}
    save_resume_ledger(ledger_path, ledger)
    staged_train = work_root / "train.zip"
    reconstructed = _ensure_train_reconstruction(parts=train_parts, staged_path=staged_train, format_report=format_report, ledger=ledger)
    try:
        train_contract = verify_synthetic_source_contract(staged_train, source_split="train", expected_count=32000)
    except Stage2AuditError as exc:
        raise ColabExecutionError(exc.code, exc.detail) from exc
    staged_validation = work_root / "validation.zip"
    if not staged_validation.is_file() or staged_validation.stat().st_size != (drive_raw_root / "validation.zip").stat().st_size:
        staged_validation_record = _copy_to_work(drive_raw_root / "validation.zip", staged_validation)
    else:
        staged_validation_record = {"path": _logical(staged_validation), "size_bytes": staged_validation.stat().st_size, "sha256": sha256_file(staged_validation), "resumed": True}
    try:
        validation_contract = verify_synthetic_source_contract(staged_validation, source_split="validation", expected_count=8000)
    except Stage2AuditError as exc:
        raise ColabExecutionError(exc.code, exc.detail) from exc

    output_artifact_root = drive_output_root / "T-A6_real_and_synthetic_canonical"
    train_config = ConversionConfig(mode=COLAB_STAGE2_MODE, source_split="train", source_domain="SYNTHETIC", safenest_role="TRAIN")
    validation_archive = staged_validation
    validation_config = ConversionConfig(mode=COLAB_STAGE2_MODE, source_split="validation", source_domain="SYNTHETIC", safenest_role="VALIDATION")
    train_summary = _reuse_or_convert(
        artifact_root=output_artifact_root,
        stem="train",
        converter=lambda: convert_partition(config=train_config, repo_root=repo_root, source_archive=staged_train, artifact_dir=output_artifact_root, overwrite=False),
    )
    validation_summary = _reuse_or_convert(
        artifact_root=output_artifact_root,
        stem="validation",
        converter=lambda: convert_partition(config=validation_config, repo_root=repo_root, source_archive=validation_archive, artifact_dir=output_artifact_root, overwrite=False),
    )

    real_archive = drive_raw_root / "test.zip"
    locked_real, created_link = _stage_real_archive(repo_root, real_archive)
    real_config = ConversionConfig(mode=STAGE1_MODE, source_split="test", source_domain="REAL", safenest_role="REAL_EVAL_DEVELOPMENT")
    try:
        real_summary = _reuse_or_convert(
            artifact_root=output_artifact_root,
            stem="real_eval_development",
            converter=lambda: convert_partition(config=real_config, repo_root=repo_root, source_archive=locked_real, artifact_dir=output_artifact_root, overwrite=False),
        )
    finally:
        if created_link:
            locked_real.unlink(missing_ok=True)

    def role_data(summary: Mapping[str, Any], role: str, stem: str) -> dict[str, Any]:
        artifact = output_artifact_root / f"{stem}_canonical.npy"
        provenance = output_artifact_root / f"{stem}_provenance.jsonl"
        ledger_file = output_artifact_root / f"{stem}_conversion_ledger.json"
        validated = validate_role_artifact(role=role, artifact_path=artifact, provenance_path=provenance, expected_count=32000 if role == "TRAIN" else 8000)
        validated.update({"summary": dict(summary), "ledger_sha256": sha256_file(ledger_file)})
        return validated

    role_records = {
        "TRAIN": role_data(train_summary, "TRAIN", "train"),
        "VALIDATION": role_data(validation_summary, "VALIDATION", "validation"),
        "REAL_EVAL_DEVELOPMENT": role_data(real_summary, "REAL_EVAL_DEVELOPMENT", "real_eval_development"),
    }
    near_audit = audit_near_duplicates_cross_role(role_records)
    exact_audit = audit_exact_duplicates(role_records)
    leakage_audit = audit_cross_role_leakage(role_records, near_audit)

    replay_root = work_root / "determinism_replay"
    replay_root.mkdir(parents=True, exist_ok=True)
    replay_train = convert_partition(config=train_config, repo_root=repo_root, source_archive=staged_train, artifact_dir=replay_root, overwrite=False)
    replay_validation = convert_partition(config=validation_config, repo_root=repo_root, source_archive=staged_validation, artifact_dir=replay_root, overwrite=False)
    locked_real, created_link = _stage_real_archive(repo_root, real_archive)
    try:
        replay_real = convert_partition(config=real_config, repo_root=repo_root, source_archive=locked_real, artifact_dir=replay_root, overwrite=False)
    finally:
        if created_link:
            locked_real.unlink(missing_ok=True)
    determinism_rows = {
        "TRAIN": (train_summary, replay_train),
        "VALIDATION": (validation_summary, replay_validation),
        "REAL_EVAL_DEVELOPMENT": (real_summary, replay_real),
    }
    determinism_roles = {
        role: {
            "artifact_checksum_match": first.get("artifact_sha256") == second.get("artifact_sha256"),
            "provenance_checksum_match": first.get("provenance_sha256") == second.get("provenance_sha256"),
            "first_artifact_sha256": first.get("artifact_sha256"),
            "replay_artifact_sha256": second.get("artifact_sha256"),
            "first_provenance_sha256": first.get("provenance_sha256"),
            "replay_provenance_sha256": second.get("provenance_sha256"),
        }
        for role, (first, second) in determinism_rows.items()
    }
    determinism = {
        "schema_version": "1.0",
        "phase": "T-A6_COLAB_STAGE2",
        "status": "PASS" if all(item["artifact_checksum_match"] and item["provenance_checksum_match"] for item in determinism_roles.values()) else "FAIL",
        "full_second_conversion": True,
        "artifact_checksum_match": all(item["artifact_checksum_match"] for item in determinism_roles.values()),
        "provenance_checksum_match": all(item["provenance_checksum_match"] for item in determinism_roles.values()),
        "roles": determinism_roles,
        "ordering": "source frame index ascending; deterministic archive member order",
    }

    bundle = drive_output_root / "T-A6_execution_result"
    bundle.mkdir(parents=True, exist_ok=True)
    registry = build_role_registry(role_records, output_artifact_root)
    status_summary = build_conversion_status(role_records)
    quality_summary = build_quality_audit(role_records)
    output_checksums = build_output_checksums(role_records)
    source_identity = {
        "schema_version": "1.0",
        "phase": "T-A6_COLAB_STAGE2",
        "source_paths_are_logical": True,
        "train_multipart": format_report,
        "train_reconstructed": reconstructed,
        "validation_staged": staged_validation_record,
        "synthetic_physical_contract": {"train": train_contract, "validation": validation_contract},
        "real_test": {"source_path": "test.zip", "safenest_role": "REAL_EVAL_DEVELOPMENT", "locked_test_available": False},
    }
    execution = {
        "schema_version": "1.0",
        "phase": "T-A6_COLAB_STAGE2",
        "status": "FULL_AUDIT_COMPLETE_WITH_LIMITATIONS",
        "message": "synthetic TRAIN/VALIDATION and real development reference converted, audited, and determinism-replayed",
        "full_t_a6_gate": "T_A6_FULL_COMPLETE_WITH_LIMITATIONS",
        "t_b_authorized": False,
        "locked_test_available": False,
        "roles": list(ROLE_ORDER),
        "grouping_limitation": "SUBJECT_SESSION_SEQUENCE_EVENT_NOT_VERIFIABLE_SOURCE_PROVENANCE_ABSENT",
    }
    environment = {
        "schema_version": "1.0",
        "phase": "T-A6_COLAB_STAGE2",
        "python": sys.version,
        "git": startup.get("git", {}),
        "numpy": startup.get("numpy"),
        "pillow": startup.get("pillow"),
        "gpu_required": False,
        "runner": "scripts/run_thermal_t_a6_colab.py",
    }
    for name, value in {
        "execution_summary.json": execution,
        "source_identity.json": source_identity,
        "canonical_artifact_registry.json": registry,
        "conversion_status_summary.json": status_summary,
        "output_checksums.json": output_checksums,
        "quality_audit_summary.json": quality_summary,
        "exact_duplicate_audit.json": exact_audit,
        "near_duplicate_audit.json": near_audit,
        "cross_role_leakage_audit.json": leakage_audit,
        "determinism_summary.json": determinism,
        "execution_environment.json": environment,
    }.items():
        _stage2_bundle_json(bundle, name, value)
    ledger["partitions"] = {
        role: {"status": "FINALIZED", "summary": role_records[role]["summary"]}
        for role in ROLE_ORDER
    }
    ledger["finalized"] = True
    ledger["execution_result_bundle"] = "T-A6_execution_result"
    ledger["audits"] = {"exact": exact_audit.get("audit_sha256"), "near": near_audit.get("audit_sha256"), "leakage": leakage_audit.get("audit_sha256"), "determinism": determinism.get("status")}
    _atomic_json(bundle / "resume_ledger.json", ledger)
    build_checksums(bundle)
    preliminary = validate_stage2_bundle(bundle, require_validation_result=False)
    _atomic_json(bundle / "validation_result.json", preliminary)
    build_checksums(bundle)
    final_validation = validate_stage2_bundle(bundle, require_validation_result=True)
    _atomic_json(bundle / "validation_result.json", final_validation)
    build_checksums(bundle)
    if final_validation.get("evidence_validation") != "PASS":
        raise ColabExecutionError("STAGE2_BUNDLE_VALIDATION_FAILED", json.dumps(final_validation, ensure_ascii=False, sort_keys=True))
    return {
        "status": "FULL_AUDIT_COMPLETE_WITH_LIMITATIONS",
        "bundle": _logical(bundle),
        "startup": startup,
        "multipart": format_report,
        "train": train_summary,
        "validation": validation_summary,
        "real": real_summary,
        "validation_result": final_validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("COLAB_STAGE2", "MAC_STAGE1"), default="COLAB_STAGE2")
    parser.add_argument("--drive-raw-root", type=Path, default=None)
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--drive-output-root", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--include-real-test", action="store_true")
    args = parser.parse_args()
    raw = args.drive_raw_root or Path(os.environ.get("THERMAL_A6_DRIVE_RAW_ROOT", ""))
    work = args.work_root or Path(os.environ.get("THERMAL_A6_WORK_ROOT", "/content/thermal_t_a6_work"))
    output = args.drive_output_root or Path(os.environ.get("THERMAL_A6_DRIVE_OUTPUT_ROOT", ""))
    if not str(raw) or not str(output):
        raise SystemExit("Drive roots are required via arguments or THERMAL_A6_* environment variables")
    try:
        result = run(mode=args.mode, drive_raw_root=raw, work_root=work, drive_output_root=output, repo_root=args.repo_root, dry_run=not args.execute, include_real_test=args.include_real_test)
    except (ColabExecutionError, Stage2AuditError) as exc:
        code = getattr(exc, "code", "STAGE2_AUDIT_FAILED")
        detail = getattr(exc, "detail", str(exc))
        print(json.dumps({"status": "BLOCKED", "code": code, "message": detail}, ensure_ascii=False, sort_keys=True, indent=2))
        raise SystemExit(2)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
