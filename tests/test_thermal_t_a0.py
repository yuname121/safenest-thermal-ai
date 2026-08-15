"""Focused synthetic-fixture tests for the Thermal T-A0 validator."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_thermal_t_a0 import (
    REQUIRED_CANDIDATE_FIELDS,
    REQUIRED_JSON,
    canonical_json,
    validate_evidence,
)


def valid_candidate(candidate_id: str = "local_selected") -> dict:
    data = {field: "UNKNOWN" for field in REQUIRED_CANDIDATE_FIELDS}
    data.update(
        {
            "candidate_id": candidate_id,
            "official_dataset_name": "Synthetic fixture thermal source",
            "stable_identifier": "doi:10.0000/fixture",
            "official_distribution_location": "https://example.org/fixture",
            "original_publication": "https://example.org/paper",
            "license_terms": "CC-BY-4.0",
            "research_use_permission": "PERMITTED",
            "model_training_permission": "PERMITTED",
            "redistribution_restrictions": "ATTRIBUTION_REQUIRED",
            "access_registration_requirements": "NONE",
            "genuine_thermal_status": "REAL_THERMAL_NUMERIC_MEASUREMENTS",
            "rgb_colorized_only": False,
            "representation_classification": "RADIOMETRIC_TEMPERATURE",
            "sensor_model": "FixtureSensor",
            "source_frame_shape": [24, 32],
            "dtype": "float32",
            "bit_depth": 32,
            "channels": 1,
            "file_format": "NPY",
            "timestamp_availability": "FRAME_LEVEL",
            "subject_identifiers": "SUBJECT_ID",
            "session_identifiers": "SESSION_ID",
            "scene_identifiers": "SCENE_ID",
            "sequence_identifiers": "SEQUENCE_ID",
            "event_identifiers": "EVENT_ID",
            "camera_identifiers": "CAMERA_ID",
            "fall_labels": "STAGED FALL WITH ONSET/END",
            "fall_event_boundary_quality": "FRAMEWISE",
            "normal_activity_coverage": "DOCUMENTED",
            "hard_negative_coverage": "DOCUMENTED",
            "staged_vs_natural_fall_semantics": "STAGED",
            "subject_count": 4,
            "session_count": 8,
            "sequence_count": 16,
            "event_count": 8,
            "subject_wise_split_feasibility": "YES",
            "fallback_grouping_feasibility": "YES",
            "duplicate_near_duplicate_risk": "AUDIT_REQUIRED_LATER",
            "event_level_evaluation_compatibility": "YES",
            "approximate_download_storage_impact": "SMALL_FIXTURE",
            "checksum_availability": "SHA256",
            "thermal44_relevance": "THERMAL44_COMPARISON_NOT_VERIFIABLE",
            "known_limitations": ["Synthetic metadata fixture only."],
            "materialization_state": "LOCALLY_MATERIALIZED",
            "overall_status": "SELECTED",
            "explicit_justification": "Fixture satisfies every selection metadata gate.",
            "source_identity_status": "VERIFIED",
            "license_status": "VERIFIED_ACCEPTABLE",
            "inventory_status": "DETERMINISTIC_INVENTORY",
            "label_semantics_status": "USABLE",
            "grouping_status": "USABLE",
            "safe_reader_documentation_status": "DOCUMENTED",
            "official_source_or_limitation": "Official fixture source.",
            "evidence_category": "LOCALLY_MEASURED",
        }
    )
    return data


def valid_asset(asset_id: str = "owner_local") -> dict:
    return {
        "asset_id": asset_id,
        "path": "datasets/thermal/fixture",
        "observation_source": ["OWNER_CONFIRMED_LOCAL_STATE", "LOCALLY_MEASURED"],
        "existence": "PATH_EXISTS",
        "git_visibility": False,
        "git_ignore_state": "GIT_IGNORED_PAYLOAD",
        "materialization_state": "LOCALLY_MATERIALIZED",
        "logical_size_bytes": 10,
        "locally_readable_status": "READABLE_OFFLINE",
        "inventory_summary": {"files": 1},
        "representation_status": "RADIOMETRIC_TEMPERATURE",
        "source_identity_status": "VERIFIED",
        "license_status": "VERIFIED_ACCEPTABLE",
        "label_status": "USABLE",
        "grouping_status": "USABLE",
        "checksum_status": "SHA256",
        "warnings": [],
    }


def valid_documents() -> dict[str, dict]:
    candidate = valid_candidate()
    return {
        "candidate_registry.json": {
            "candidates": [candidate],
            "phase": "T-A0",
            "schema_version": "1.0",
        },
        "limitations.json": {
            "limitations": [{"id": "T-A0-L001", "issue": "Fixture limitation", "status": "OPEN"}],
            "overall_outcome": "PASS_WITH_LIMITATIONS",
            "phase": "T-A0",
            "schema_version": "1.0",
        },
        "local_asset_registry.json": {
            "assets": [valid_asset()],
            "phase": "T-A0",
            "schema_version": "1.0",
        },
        "model_artifact_audit.json": {
            "artifact_path": "models/thermal/fixture.tflite",
            "phase": "T-A0",
            "schema_version": "1.0",
            "validation_claim": "ARTIFACT_ONLY",
        },
        "processed_lineage.json": {
            "artifact_path": "datasets/thermal/fixture.npz",
            "phase": "T-A0",
            "schema_version": "1.0",
        },
        "selected_source_identity.json": {
            "phase": "T-A0",
            "schema_version": "1.0",
            "selected_candidate_id": "local_selected",
            "selection_status": "PASS_WITH_LIMITATIONS",
            "t_a1_authorized": True,
        },
        "source_license_evidence.json": {
            "phase": "T-A0",
            "schema_version": "1.0",
            "sources": [{"url": "https://example.org/fixture"}],
        },
    }


class Fixture:
    def __init__(self, documents: dict[str, dict]):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.evidence = self.root / "evidence"
        self.evidence.mkdir()
        for name in REQUIRED_JSON:
            (self.evidence / name).write_text(canonical_json(documents[name]), encoding="utf-8")
        lines = []
        for name in REQUIRED_JSON:
            path = self.evidence / name
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.relative_to(self.root).as_posix()}")
        (self.evidence / "checksums.sha256").write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")

    def close(self) -> None:
        self._tmp.cleanup()


class TestThermalTA0Validator(unittest.TestCase):
    def validate(self, mutate=None):
        docs = valid_documents()
        if mutate:
            mutate(docs)
        fixture = Fixture(docs)
        self.addCleanup(fixture.close)
        return validate_evidence(fixture.evidence, fixture.root)

    @staticmethod
    def codes(result: dict) -> set[str]:
        return {entry["code"] for entry in result["errors"]}

    def test_valid_selected_local_source(self):
        result = self.validate()
        self.assertEqual(result["evidence_validation"], "PASS")
        self.assertTrue(result["t_a1_authorized"])

    def test_selected_post_fall_posture_proxy_with_official_split_limitation(self):
        def mutate(docs):
            item = docs["candidate_registry.json"]["candidates"][0]
            item.update(
                {
                    "license_status": "VERIFIED_ACCEPTABLE_WITH_NONCOMMERCIAL_RESEARCH_RESTRICTION",
                    "inventory_status": "DETERMINISTIC_INVENTORY_WITH_OFFICIAL_CHECKSUMS",
                    "label_semantics_status": "USABLE_DERIVED_POST_FALL_POSTURE_PROXY",
                    "grouping_status": "ACCEPTED_OFFICIAL_SPLIT_LIMITATION",
                    "fallback_grouping_feasibility": "Preserve official train/validation/test; never perform a frame-random resplit",
                    "safenest_sensor_role": "Post-fall posture evidence; no single thermal frame confirms a fall event",
                    "safenest_label_mapping": {"0": {"source_label": "lying", "target_label": "HUMAN_FALL", "mapping_type": "DERIVED_POST_FALL_POSTURE_PROXY"}},
                }
            )

        result = self.validate(mutate)
        self.assertEqual(result["evidence_validation"], "PASS")
        self.assertEqual(result["overall_outcome"], "PASS_WITH_LIMITATIONS")
        self.assertTrue(result["t_a1_authorized"])

    def test_post_fall_proxy_without_single_frame_guard_fails(self):
        def mutate(docs):
            item = docs["candidate_registry.json"]["candidates"][0]
            item.update(
                {
                    "label_semantics_status": "USABLE_DERIVED_POST_FALL_POSTURE_PROXY",
                    "safenest_sensor_role": "Fall detector",
                    "safenest_label_mapping": {"0": {"source_label": "lying", "target_label": "HUMAN_FALL", "mapping_type": "DIRECT_FALL_EVENT"}},
                }
            )

        result = self.validate(mutate)
        self.assertIn("POST_FALL_PROXY_GUARD_MISSING", self.codes(result))

    def test_accepted_official_split_without_resplit_guard_fails(self):
        def mutate(docs):
            item = docs["candidate_registry.json"]["candidates"][0]
            item.update(
                {
                    "grouping_status": "ACCEPTED_OFFICIAL_SPLIT_LIMITATION",
                    "fallback_grouping_feasibility": "Use the official split when convenient",
                }
            )

        result = self.validate(mutate)
        self.assertIn("OFFICIAL_SPLIT_GUARD_MISSING", self.codes(result))

    def test_local_git_ignored_source_is_valid(self):
        result = self.validate()
        self.assertEqual(result["evidence_validation"], "PASS")
        self.assertFalse(valid_documents()["local_asset_registry.json"]["assets"][0]["git_visibility"])

    def test_owner_confirmed_source_not_visible_remotely_is_not_absent(self):
        result = self.validate()
        self.assertNotIn("OWNER_CONFIRMED_SOURCE_MISLABELED_ABSENT", self.codes(result))

    def test_cloud_placeholder_archive_is_explicit(self):
        def mutate(docs):
            asset = valid_asset("placeholder_archive")
            asset.update({"path": "datasets/raw_archives/fixture.zip", "materialization_state": "LOCAL_CLOUD_PLACEHOLDER", "locally_readable_status": "LOCAL_CLOUD_PLACEHOLDER; readable_offline=False"})
            docs["local_asset_registry.json"]["assets"].append(asset)

        result = self.validate(mutate)
        self.assertEqual(result["evidence_validation"], "PASS")

    def test_missing_license_fails(self):
        def mutate(docs):
            del docs["candidate_registry.json"]["candidates"][0]["license_status"]

        result = self.validate(mutate)
        self.assertIn("LICENSE_ACCESS_STATUS_MISSING", self.codes(result))

    def test_unknown_representation_cannot_be_selected(self):
        def mutate(docs):
            docs["candidate_registry.json"]["candidates"][0]["representation_classification"] = "UNKNOWN"

        result = self.validate(mutate)
        self.assertIn("SELECTED_CANDIDATE_REQUIREMENT_FAILED", self.codes(result))

    def test_rgb_rendering_cannot_be_radiometric(self):
        def mutate(docs):
            item = docs["candidate_registry.json"]["candidates"][0]
            item["channels"] = 3
            item["rgb_colorized_only"] = True

        result = self.validate(mutate)
        self.assertIn("RGB_FALSELY_RADIOMETRIC", self.codes(result))

    def test_missing_grouping_provenance_fails(self):
        def mutate(docs):
            del docs["candidate_registry.json"]["candidates"][0]["grouping_status"]

        result = self.validate(mutate)
        self.assertIn("GROUPING_STATUS_MISSING", self.codes(result))

    def test_missing_label_semantics_fails(self):
        def mutate(docs):
            del docs["candidate_registry.json"]["candidates"][0]["label_semantics_status"]

        result = self.validate(mutate)
        self.assertIn("LABEL_SEMANTICS_MISSING", self.codes(result))

    def test_rejected_candidate_requires_reason(self):
        def mutate(docs):
            item = docs["candidate_registry.json"]["candidates"][0]
            item["overall_status"] = "REJECTED_PROVENANCE"
            item["explicit_justification"] = ""
            docs["selected_source_identity.json"].update({"selected_candidate_id": None, "selection_status": "BLOCKED", "t_a1_authorized": False})

        result = self.validate(mutate)
        self.assertIn("CANDIDATE_REASON_MISSING", self.codes(result))

    def test_unsupported_thermal44_hardware_assertion_fails(self):
        def mutate(docs):
            docs["candidate_registry.json"]["candidates"][0]["thermal44_relevance"] = "THERMAL44_VALIDATED"

        result = self.validate(mutate)
        self.assertIn("UNSUPPORTED_THERMAL44_ASSERTION", self.codes(result))

    def test_absolute_path_leakage_fails(self):
        def mutate(docs):
            docs["local_asset_registry.json"]["assets"][0]["path"] = "/Users/example/private/dataset"

        result = self.validate(mutate)
        self.assertIn("ABSOLUTE_PATH_LEAKAGE", self.codes(result))

    def test_archive_cannot_be_active_source(self):
        def mutate(docs):
            docs["local_asset_registry.json"]["assets"][0]["path"] = "archive/version_snapshots/thermal"

        result = self.validate(mutate)
        self.assertIn("ARCHIVE_TREATED_AS_ACTIVE", self.codes(result))

    def test_nondeterministic_candidate_order_fails(self):
        def mutate(docs):
            second = copy.deepcopy(docs["candidate_registry.json"]["candidates"][0])
            second["candidate_id"] = "a_first"
            second["overall_status"] = "ACCEPTABLE_BACKUP"
            docs["candidate_registry.json"]["candidates"].append(second)

        result = self.validate(mutate)
        self.assertIn("CANDIDATE_ORDER_NONDETERMINISTIC", self.codes(result))

    def test_selected_source_failing_mandatory_requirement_fails(self):
        def mutate(docs):
            docs["candidate_registry.json"]["candidates"][0]["license_status"] = "LICENSE_UNVERIFIED"

        result = self.validate(mutate)
        self.assertIn("SELECTED_CANDIDATE_REQUIREMENT_FAILED", self.codes(result))

    def test_no_selected_source_with_blocked_outcome_is_valid(self):
        def mutate(docs):
            item = docs["candidate_registry.json"]["candidates"][0]
            item["overall_status"] = "REJECTED_GROUPING"
            item["explicit_justification"] = "No grouping provenance."
            docs["selected_source_identity.json"].update({"selected_candidate_id": None, "selection_status": "BLOCKED", "t_a1_authorized": False})

        result = self.validate(mutate)
        self.assertEqual(result["evidence_validation"], "PASS")
        self.assertEqual(result["overall_outcome"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
