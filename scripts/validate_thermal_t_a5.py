#!/usr/bin/env python3
"""Standalone compact validator for Thermal T-A5.

The validator validates governance evidence rather than loading image arrays.
It independently evaluates candidate policies and every materialized real-test
assignment, and fails closed on missing provenance, tampering, or path leakage.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.thermal.split_policy import (  # noqa: E402
    ASSIGNMENT_RULE_ID,
    DATASET_ID,
    GROUPING_POLICY_ID,
    REAL_TEST_FRAME_COUNT,
    SEMANTIC_POLICY_ID,
    SOURCE_ARCHIVE_PATH,
    SOURCE_ARCHIVE_SHA256,
    SPLIT_POLICY_ID,
    SPLIT_SELECTION_POLICY_ID,
    access_history_definition,
    assignment_rule_contract,
    candidate_policy_definitions,
    evaluate_candidates,
    grouping_evidence_definition,
    selected_candidate,
    selected_split_policy_profile,
    selection_policy_definition,
    source_partition_definitions,
    validate_assignment_inventory,
)

EVIDENCE_REL = "datasets/thermal/manifests/T-A5_grouping_immutable_split"
T_A0_REL = "datasets/thermal/manifests/T-A0_source_identity"
T_A1_REL = "datasets/thermal/manifests/T-A1_safe_reader_raw_unit_contract"
T_A2_REL = "datasets/thermal/manifests/T-A2_geometry_calibration_canonical_frame"
T_A3_REL = "datasets/thermal/manifests/T-A3_sequence_window_event_policy"
T_A4_REL = "datasets/thermal/manifests/T-A4_label_semantics_ambiguity"
CORE_JSON = [
    "assignment_rule_contract.json", "augmentation_inheritance_policy.json", "data_access_history.json", "grouping_evidence_registry.json", "leakage_policy.json", "limitations.json", "locked_test_eligibility.json", "real_test_assignment_inventory.json", "selected_split_policy.json", "source_partition_contract.json", "split_distribution_summary.json", "split_policy_candidates.json", "split_selection_policy.json",
]
REQUIRED_JSON = CORE_JSON + ["validation_result.json"]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _sha256(path: Path) -> str:
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
    if value.startswith(("/", "~/", "file://")) or "\\" in value or "/Users/" in value or "/private/" in value or "iCloud" in value:
        return False
    pure = PurePosixPath(value)
    return not pure.is_absolute() and ".." not in pure.parts


def _load_documents(evidence_dir: Path, errors: list[dict[str, str]]) -> tuple[dict[str, Any], list[Path]]:
    documents: dict[str, Any] = {}
    paths: list[Path] = []
    for name in CORE_JSON:
        path = evidence_dir / name
        if not path.is_file():
            _error(errors, "REQUIRED_ARTIFACT_MISSING", name, "Required T-A5 artifact is missing.")
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _error(errors, "JSON_READ_FAILED", name, str(exc))
            continue
        documents[name] = value
        paths.append(path)
        if path.read_text(encoding="utf-8") != canonical_json(value):
            _error(errors, "NONDETERMINISTIC_JSON", name, "JSON must use canonical sorted formatting.")
        for location, item in _walk(value):
            if isinstance(item, str) and not _portable(item) and (item.startswith(("/", "~/", "file://")) or "/Users/" in item or "/private/" in item or "iCloud" in item):
                _error(errors, "NONPORTABLE_PATH", f"{name}:{location}", item)
            if isinstance(item, str) and "Thermal-44" in item and any(token in item.upper() for token in ("VERIFIED", "CONFIRMED", "VALIDATED")):
                _error(errors, "UNSUPPORTED_THERMAL44_ASSERTION", f"{name}:{location}", item)
            if isinstance(item, dict):
                for key in item:
                    if str(key).lower() in {"accuracy", "precision", "recall", "f1", "macro_f1", "confusion_matrix", "prediction_distribution", "loss", "auc"}:
                        _error(errors, "MODEL_METRIC_CONTAMINATION", f"{name}:{location}.{key}", "T-A5 must not select a split using model metrics.")
    return documents, paths


def _validate_predecessors(repo_root: Path, errors: list[dict[str, str]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    try:
        from scripts.validate_thermal_t_a0 import validate_evidence as a0
        results["T-A0"] = a0(repo_root / T_A0_REL, repo_root)
    except Exception as exc:
        results["T-A0"] = {"evidence_validation": "FAIL"}
        _error(errors, "T_A0_VALIDATOR_ERROR", T_A0_REL, str(exc))
    try:
        from scripts.validate_thermal_t_a1 import validate_evidence as a1
        results["T-A1"] = a1(repo_root=repo_root, evidence_dir=repo_root / T_A1_REL, check_checksums=True, verify_real_payload=False)
    except Exception as exc:
        results["T-A1"] = {"evidence_validation": "FAIL"}
        _error(errors, "T_A1_VALIDATOR_ERROR", T_A1_REL, str(exc))
    try:
        from scripts.validate_thermal_t_a2 import validate_evidence as a2
        results["T-A2"] = a2(
            repo_root=repo_root,
            evidence_dir=repo_root / T_A2_REL,
            check_checksums=True,
            verify_real_payload=False,
        )
    except Exception as exc:
        results["T-A2"] = {"evidence_validation": "FAIL"}
        _error(errors, "T_A2_VALIDATOR_ERROR", T_A2_REL, str(exc))
    try:
        from scripts.validate_thermal_t_a3 import validate_evidence as a3
        results["T-A3"] = a3(repo_root=repo_root, evidence_dir=repo_root / T_A3_REL, check_checksums=True, verify_real_payload=False)
    except Exception as exc:
        results["T-A3"] = {"evidence_validation": "FAIL"}
        _error(errors, "T_A3_VALIDATOR_ERROR", T_A3_REL, str(exc))
    try:
        from scripts.validate_thermal_t_a4 import validate_evidence as a4
        results["T-A4"] = a4(repo_root=repo_root, evidence_dir=repo_root / T_A4_REL, check_checksums=True, verify_real_payload=False)
    except Exception as exc:
        results["T-A4"] = {"evidence_validation": "FAIL"}
        _error(errors, "T_A4_VALIDATOR_ERROR", T_A4_REL, str(exc))
    for phase, result in results.items():
        if result.get("evidence_validation") != "PASS":
            _error(errors, f"{phase.replace('-', '_')}_VALIDATION_FAILED", phase, f"{phase} predecessor did not pass: {result.get('overall_outcome')}")
    return results


def _validate_source_partitions(document: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    expected = source_partition_definitions()
    actual = document.get("partitions")
    if actual != expected:
        _error(errors, "SOURCE_PARTITION_CONTRACT_MISMATCH", "source_partition_contract.json:partitions", "Official source partitions or roles do not match the policy module.")
    if document.get("source_partition_preservation") is not True or document.get("source_partitions_must_remain_separate") is not True:
        _error(errors, "SOURCE_PARTITIONS_NOT_PRESERVED", "source_partition_contract.json", "Official source boundaries must remain separate.")
    if document.get("partition_domains") != {"train": "SYNTHETIC", "validation": "SYNTHETIC", "test": "REAL"}:
        _error(errors, "SOURCE_DOMAIN_CONFLATION", "source_partition_contract.json:partition_domains", "Synthetic and real source domains must remain explicit.")
    for row in actual or []:
        if row.get("source_split") == "test" and row.get("planned_safenest_role") == "LOCKED_TEST":
            _error(errors, "CONTAMINATED_LOCKED_TEST", "source_partition_contract.json", "Accessed real test cannot be locked test.")
        if row.get("source_split") in {"train", "validation"} and row.get("sample_inventory_status") != "SAMPLE_LEVEL_INVENTORY_PENDING_MATERIALIZATION":
            _error(errors, "UNMATERIALIZED_PARTITION_OVERCLAIM", f"source_partition_contract.json:{row.get('source_split')}", "Placeholder bytes cannot be claimed fully audited.")


def _validate_grouping(document: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    expected = grouping_evidence_definition()
    if document != expected:
        _error(errors, "GROUPING_EVIDENCE_MISMATCH", "grouping_evidence_registry.json", "Grouping evidence differs from independent source-policy recomputation.")
    for dimension, row in (document.get("dimensions") or {}).items():
        if row.get("usable_for_split") and dimension != "official_source_partition":
            _error(errors, "UNVERIFIED_GROUP_USED", f"grouping_evidence_registry.json:dimensions.{dimension}", "Only authoritative independent groups are usable.")
    if document.get("frame_index_as_group") is not False or document.get("label_as_group") is not False:
        _error(errors, "INVALID_GROUP_KEY", "grouping_evidence_registry.json", "Frame index and labels cannot define groups.")
    if document.get("generalization_performance") != "NOT_VERIFIABLE":
        _error(errors, "UNVERIFIED_GENERALIZATION_CLAIM", "grouping_evidence_registry.json", "No subject/session/event group generalization is verifiable.")


def _validate_access(document: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    expected = access_history_definition()
    actual = {key: document.get(key) for key in ("phase", "schema_version", "dataset_id", "history_policy_id", "source_partitions", "entries", "pristine_locked_test_available", "pristine_locked_test_reason", "model_metrics_used")}
    if actual != expected:
        _error(errors, "ACCESS_HISTORY_MISMATCH", "data_access_history.json", "T-A0-T-A4 access history was changed or omitted.")
    test = [row for row in document.get("entries", []) if row.get("source_split") == "test"]
    access_types = {row.get("access_type") for row in test}
    for required in ("GEOMETRY_SELECTION", "TEMPORAL_CAPABILITY_ANALYSIS", "SEMANTIC_POLICY_SELECTION"):
        if required not in access_types:
            _error(errors, "ACCESS_HISTORY_INCOMPLETE", "data_access_history.json", f"Missing test access type {required}.")
    if document.get("pristine_locked_test_available") != "NO":
        _error(errors, "PRISTINE_LOCKED_TEST_OVERCLAIM", "data_access_history.json", "No untouched independent holdout is available.")


def _validate_selection(documents: Mapping[str, Any], errors: list[dict[str, str]]) -> str | None:
    policy = selection_policy_definition()
    if documents["split_selection_policy.json"] != policy:
        _error(errors, "SELECTION_POLICY_MISMATCH", "split_selection_policy.json", "Selection policy differs from independent policy.")
    expected_evaluated = evaluate_candidates(candidate_policy_definitions(), policy)
    recorded = documents["split_policy_candidates.json"].get("candidates")
    if recorded != expected_evaluated:
        _error(errors, "CANDIDATE_EVALUATION_MISMATCH", "split_policy_candidates.json:candidates", "Candidate gates/ranking were not independently reproducible.")
    try:
        winner = selected_candidate(expected_evaluated)
    except Exception as exc:
        _error(errors, "NO_SELECTED_POLICY", "split_policy_candidates.json", str(exc))
        return None
    selected = documents["selected_split_policy.json"]
    expected_profile = selected_split_policy_profile()
    for key, value in expected_profile.items():
        if selected.get(key) != value:
            _error(errors, "SELECTED_POLICY_MISMATCH", f"selected_split_policy.json:{key}", "Selected policy is not independently derived.")
    if selected.get("selected_candidate_id") != winner["candidate_id"]:
        _error(errors, "WINNER_TAMPERED", "selected_split_policy.json:selected_candidate_id", "Candidate winner does not match independent recomputation.")
    return winner["candidate_id"]


def _validate_locked(document: Mapping[str, Any], history: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    expected = {
        "eligible": False,
        "status": "DISQUALIFIED_BY_PRIOR_ACCESS",
        "reason": "USED_FOR_PREPROCESSING_GEOMETRY_SELECTION",
    }
    if document.get("pristine_locked_test_eligible") is not False or document.get("status") != expected["status"] or document.get("disqualification_reason") != expected["reason"]:
        _error(errors, "LOCKED_TEST_ELIGIBILITY_MISMATCH", "locked_test_eligibility.json", "Accessed test cannot be relabeled as pristine locked test.")
    if document.get("current_pristine_locked_test_available") != "NO":
        _error(errors, "LOCKED_TEST_AVAILABILITY_OVERCLAIM", "locked_test_eligibility.json", "No pristine locked test is present.")
    if not document.get("geometry_selection_access") or not document.get("semantic_policy_selection_access"):
        _error(errors, "LOCKED_TEST_ACCESS_HISTORY_MISSING", "locked_test_eligibility.json", "Geometry/semantic access must disqualify lock.")


def _validate_inventory(document: Mapping[str, Any], errors: list[dict[str, str]]) -> dict[str, Any] | None:
    records = document.get("records")
    if not isinstance(records, list):
        _error(errors, "ASSIGNMENT_RECORDS_MISSING", "real_test_assignment_inventory.json:records", "Assignment records must be a list.")
        return None
    try:
        summary = validate_assignment_inventory(records)
    except Exception as exc:
        _error(errors, getattr(exc, "code", "ASSIGNMENT_INVENTORY_INVALID"), "real_test_assignment_inventory.json", str(exc))
        return None
    if document.get("record_count") != REAL_TEST_FRAME_COUNT or len(records) != REAL_TEST_FRAME_COUNT:
        _error(errors, "ASSIGNMENT_COUNT_MISMATCH", "real_test_assignment_inventory.json", "All 8000 real test members require exactly one assignment.")
    if document.get("assignment_rule_id") != ASSIGNMENT_RULE_ID or document.get("split_policy_id") != SPLIT_POLICY_ID:
        _error(errors, "ASSIGNMENT_POLICY_MISMATCH", "real_test_assignment_inventory.json", "Inventory policy IDs do not match.")
    if any(row.get("safenest_assignment_role") in {"TRAIN", "VALIDATION", "LOCKED_TEST"} for row in records):
        _error(errors, "REAL_TEST_ROLE_CONFLATION", "real_test_assignment_inventory.json", "Real test cannot be assigned to train/validation/locked test.")
    if len({row.get("source_member") for row in records}) != REAL_TEST_FRAME_COUNT:
        _error(errors, "DUPLICATE_SOURCE_MEMBER", "real_test_assignment_inventory.json", "Source members must be unique.")
    return summary


def _validate_distribution(document: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    roles = document.get("roles", {})
    expected = {"TRAIN": ("train", "SYNTHETIC", 32000), "VALIDATION": ("validation", "SYNTHETIC", 8000), "REAL_EVAL_DEVELOPMENT": ("test", "REAL", 8000), "LOCKED_TEST": ("NONE", "NONE", 0)}
    for role, (split, domain, count) in expected.items():
        row = roles.get(role, {})
        if (row.get("source_split"), row.get("source_domain"), row.get("count")) != (split, domain, count):
            _error(errors, "DISTRIBUTION_MISMATCH", f"split_distribution_summary.json:roles.{role}", "Role distribution is inconsistent with official partitions.")
    if roles.get("TRAIN", {}).get("sample_inventory") != "SAMPLE_LEVEL_INVENTORY_PENDING_MATERIALIZATION" or roles.get("VALIDATION", {}).get("sample_inventory") != "SAMPLE_LEVEL_INVENTORY_PENDING_MATERIALIZATION":
        _error(errors, "PLACEHOLDER_AUDIT_OVERCLAIM", "split_distribution_summary.json", "Train/validation placeholders are not fully audited.")
    for key in ("subject_count", "session_count", "sequence_count", "event_count", "scene_count", "camera_count"):
        if document.get(key) != "NOT_VERIFIABLE":
            _error(errors, "UNKNOWN_DIMENSION_NOT_EXPLICIT", f"split_distribution_summary.json:{key}", "Unavailable provenance must remain explicit.")


def _validate_inheritance(documents: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    inheritance = documents["augmentation_inheritance_policy.json"]
    if inheritance.get("augmentation_rule") != "AUGMENTATION_TRAIN_ONLY" or not inheritance.get("derived_artifact_rule") or inheritance.get("role_change_requires_new_policy_version") is not True:
        _error(errors, "INHERITANCE_POLICY_INVALID", "augmentation_inheritance_policy.json", "Derived artifacts must inherit immutable assignment.")
    leakage = documents["leakage_policy.json"]
    taxonomy = leakage.get("taxonomy", {})
    if leakage.get("cross_role_member_overlap_count") != 0 or leakage.get("cross_role_frame_overlap_count") != 0:
        _error(errors, "CROSS_ROLE_OVERLAP", "leakage_policy.json", "Exact source member/frame overlap must be zero.")
    for dimension in ("subject_overlap", "session_overlap", "event_overlap", "sequence_overlap", "scene_overlap", "camera_overlap"):
        if taxonomy.get(dimension) != "NOT_VERIFIABLE_NO_SUBJECT_ID" and dimension != "subject_overlap":
            # Other dimensions use a more specific reason, but must remain NOT_VERIFIABLE.
            if not str(taxonomy.get(dimension, "")).startswith("NOT_VERIFIABLE"):
                _error(errors, "GROUP_LEAKAGE_OVERCLAIM", f"leakage_policy.json:taxonomy.{dimension}", "Missing group provenance is not zero overlap evidence.")
    if taxonomy.get("near_duplicate_content") != "DEFERRED_T_A6" or taxonomy.get("exact_duplicate_content") != "DEFERRED_T_A6":
        _error(errors, "DUPLICATE_AUDIT_OVERCLAIM", "leakage_policy.json", "Duplicate audit belongs to T-A6.")


def _validate_legacy(repo_root: Path, documents: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    limitation_text = json.dumps(documents["limitations.json"], sort_keys=True)
    if "processed_thermal_80x62.npz" not in limitation_text or "not split authority" not in limitation_text.lower():
        _error(errors, "LEGACY_ARTIFACT_BOUNDARY_MISSING", "limitations.json", "Legacy NPZ must not define split authority.")
    source = documents["source_partition_contract.json"]
    if "processed_thermal_80x62.npz" in json.dumps(source, sort_keys=True):
        _error(errors, "LEGACY_ARTIFACT_USED_AS_AUTHORITY", "source_partition_contract.json", "Legacy NPZ cannot be source partition authority.")


def _validate_static_implementation(repo_root: Path, errors: list[dict[str, str]]) -> None:
    path = repo_root / "datasets/thermal/split_policy.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _error(errors, "SPLIT_POLICY_SOURCE_INVALID", str(path), str(exc))
        return
    source = path.read_text(encoding="utf-8")
    forbidden = ("train_test_split", "np.random.permutation", "random.shuffle", "hash(frame", "hash(member")
    for token in forbidden:
        if token in source:
            _error(errors, "FORBIDDEN_SPLIT_IMPLEMENTATION", str(path), token)
    imports = {node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import) and node.names}
    if {"tensorflow", "torch", "tflite_runtime"} & imports or "thermal_interpreter" in source:
        _error(errors, "MODEL_COUPLING_IN_T_A5", str(path), "T-A5 split policy cannot couple to model/runtime implementation.")


def _validate_checksums(repo_root: Path, evidence_dir: Path, errors: list[dict[str, str]]) -> None:
    path = evidence_dir / "checksums.sha256"
    if not path.is_file():
        _error(errors, "CHECKSUM_REGISTRY_MISSING", "checksums.sha256", "T-A5 checksum registry missing.")
        return
    entries: dict[str, str] = {}
    previous = ""
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", line)
        if not match:
            _error(errors, "CHECKSUM_LINE_INVALID", f"checksums.sha256:{number}", line)
            continue
        digest, relative = match.groups()
        if relative <= previous:
            _error(errors, "CHECKSUM_ORDER_NONDETERMINISTIC", f"checksums.sha256:{number}", relative)
        previous = relative
        if not _portable(relative):
            _error(errors, "CHECKSUM_PATH_NOT_PORTABLE", f"checksums.sha256:{number}", relative)
        entries[relative] = digest
    for name in REQUIRED_JSON:
        target = evidence_dir / name
        if not target.is_file():
            continue
        relative = f"{EVIDENCE_REL}/{name}"
        if entries.get(relative) != _sha256(target):
            _error(errors, "CHECKSUM_MISMATCH", relative, "Measured checksum differs or coverage is missing.")
    for relative in entries:
        if not relative.startswith(EVIDENCE_REL + "/"):
            _error(errors, "CHECKSUM_SCOPE_INVALID", relative, "Checksum registry may cover only T-A5 evidence.")


def validate_evidence(*, repo_root: Path = ROOT, evidence_dir: Path | None = None, check_checksums: bool = True, verify_real_payload: bool = False) -> dict[str, Any]:
    del verify_real_payload  # T-A5 is compact and deliberately non-hydrating.
    evidence_dir = evidence_dir or repo_root / EVIDENCE_REL
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    documents, paths = _load_documents(evidence_dir, errors)
    if len(documents) == len(CORE_JSON):
        predecessors = _validate_predecessors(repo_root, errors)
        _validate_source_partitions(documents["source_partition_contract.json"], errors)
        _validate_grouping(documents["grouping_evidence_registry.json"], errors)
        _validate_access(documents["data_access_history.json"], errors)
        winner = _validate_selection(documents, errors)
        _validate_locked(documents["locked_test_eligibility.json"], documents["data_access_history.json"], errors)
        _validate_inventory(documents["real_test_assignment_inventory.json"], errors)
        _validate_distribution(documents["split_distribution_summary.json"], errors)
        _validate_inheritance(documents, errors)
        _validate_legacy(repo_root, documents, errors)
        _validate_static_implementation(repo_root, errors)
        if winner is not None and winner != "S0_OFFICIAL_SOURCE_PARTITION_PRESERVATION":
            _error(errors, "UNEXPECTED_POLICY_WINNER", "selected_split_policy.json", winner)
    if check_checksums:
        _validate_checksums(repo_root, evidence_dir, errors)
    _warning(warnings, "NO_PRISTINE_LOCKED_TEST", "locked_test_eligibility.json", "An independent untouched Thermal holdout is not currently available.")
    _warning(warnings, "GROUP_GENERALIZATION_NOT_VERIFIABLE", "grouping_evidence_registry.json", "Subject/session/event/sequence/scene/camera generalization cannot be verified from source provenance.")
    _warning(warnings, "PLACEHOLDER_PARTITIONS_NOT_MATERIALIZED", "source_partition_contract.json", "Train and validation sample-level audits require explicit hydration authorization.")
    outcome = "PASS_WITH_LIMITATIONS" if not errors else "NOT_VERIFIABLE"
    return {
        "phase": "T-A5",
        "schema_version": "1.0",
        "evidence_validation": "PASS" if not errors else "FAIL",
        "overall_outcome": outcome,
        "t_a6_authorized": not errors,
        "t_a6_full_completion_requires_placeholder_hydration": True,
        "t_a5_intermediate_release_readiness": "YES_WITH_LIMITATIONS" if not errors else "NO",
        "predecessor_validation": {phase: result.get("evidence_validation", "FAIL") for phase, result in (predecessors if len(documents) == len(CORE_JSON) else {}).items()},
        "error_count": len(errors),
        "errors": errors,
        "warning_count": len(warnings),
        "warnings": warnings,
        "model_metrics_used": False,
        "checked_artifact_count": len(paths),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--evidence-dir", type=Path, default=None)
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--skip-real-payload", action="store_true")
    args = parser.parse_args()
    result = validate_evidence(repo_root=args.repo_root, evidence_dir=args.evidence_dir, check_checksums=not args.skip_checksums, verify_real_payload=False)
    print(canonical_json(result), end="")
    return 0 if result["evidence_validation"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
