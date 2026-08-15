"""Focused T-B5 compact-evidence contract tests.

These tests copy only the small JSON evidence bundle; they never include or
open canonical arrays, ZIP archives, checkpoints, or TFLite binaries.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from datasets.thermal.t_b5_runner import EVIDENCE_REL, _profile, _sha256_text
from datasets.thermal.t_b1_preprocessing import canonical_json, sha256_file
from scripts.validate_thermal_t_b5 import FULL_JSON, validate_evidence


ROOT = Path(__file__).resolve().parents[1]


class ThermalTB5EvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="thermal_tb5_fixture_")
        self.evidence = Path(self.temp.name) / "evidence"
        shutil.copytree(ROOT / EVIDENCE_REL, self.evidence)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _load(self, name: str) -> dict:
        return json.loads((self.evidence / name).read_text(encoding="utf-8"))

    def _save(self, name: str, value: dict) -> None:
        (self.evidence / name).write_text(canonical_json(value), encoding="utf-8")
        rows = []
        for path in sorted(self.evidence.iterdir(), key=lambda item: item.name):
            if path.is_file() and path.name not in {"checksums.sha256", "validation_result.json"}:
                rows.append(f"{sha256_file(path)}  {path.name}")
        (self.evidence / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")

    def _result(self):
        return validate_evidence(repo_root=ROOT, evidence_dir=self.evidence, mode="FULL_EXPERIMENT", check_checksums=True, verify_predecessors=False)

    def test_valid_selected_offline_int8_fixture_passes(self) -> None:
        self.assertEqual(self._result()["evidence_validation"], "PASS")

    def test_local_fixture_has_deterministic_profile_order(self) -> None:
        profile = _profile()
        self.assertEqual(_sha256_text(canonical_json(profile)), _sha256_text(canonical_json(_profile())))

    def test_dynamic_range_candidate_is_rejected(self) -> None:
        doc = self._load("artifact_registry.json")
        next(item for item in doc["artifacts"] if item["candidate_id"] == "TFLITE_DYNAMIC_RANGE")["eligible"] = True
        self._save("artifact_registry.json", doc)
        result = self._result()
        self.assertTrue(any(item["code"] in {"DYNAMIC_RANGE_REGISTRY_INVALID", "DYNAMIC_RANGE_ELIGIBLE"} for item in result["errors"]))

    def test_float32_io_alone_does_not_pass_fp32_policy(self) -> None:
        doc = self._load("candidate_set.json")
        next(item for item in doc["candidates"] if item["candidate_id"] == "TFLITE_FP32")["conversion_policy"]["optimizations"] = ["DEFAULT"]
        self._save("candidate_set.json", doc)
        result = self._result()
        self.assertTrue(any(item["code"] == "FP32_CONVERSION_POLICY_INVALID" for item in result["errors"]))

    def test_wrong_artifact_sha_is_rejected(self) -> None:
        doc = self._load("artifact_registry.json")
        next(item for item in doc["artifacts"] if item["candidate_id"] == "FULL_INT8")["sha256"] = "0" * 64
        self._save("artifact_registry.json", doc)
        self.assertTrue(any(item["code"] == "ARTIFACT_IDENTITY_INVALID" for item in self._result()["errors"]))

    def test_profile_mutation_is_rejected(self) -> None:
        doc = self._load("robustness_profile.json")
        doc["families"][0]["levels"] = [-99.0]
        self._save("robustness_profile.json", doc)
        self.assertTrue(any(item["code"] == "PROFILE_CHECKSUM_INVALID" for item in self._result()["errors"]))

    def test_missing_frame_must_remain_fail_closed(self) -> None:
        doc = self._load("robustness_results.json")
        case = next(item for item in doc["cases"] if item["family_id"] == "MISSING_FRAME")
        case["model_inference_performed"] = True
        self._save("robustness_results.json", doc)
        self.assertTrue(any(item["code"] == "MISSING_FRAME_NOT_FAIL_CLOSED" for item in self._result()["errors"]))

    def test_real_development_cannot_select_candidate(self) -> None:
        doc = self._load("candidate_lock.json")
        doc["real_used_for_selection"] = True
        self._save("candidate_lock.json", doc)
        self.assertTrue(any(item["code"] == "CANDIDATE_LOCK_INVALID" for item in self._result()["errors"]))

    def test_thermal44_positive_claim_is_rejected(self) -> None:
        doc = self._load("evidence_handoff.json")
        doc["thermal44_status"] = "VALIDATED"
        self._save("evidence_handoff.json", doc)
        self.assertTrue(any(item["code"] == "HANDOFF_OVERCLAIM" for item in self._result()["errors"]))

    def test_mac_latency_cannot_be_labeled_pi(self) -> None:
        doc = self._load("latency_results.json")
        doc["environment"]["pi_measured"] = True
        self._save("latency_results.json", doc)
        self.assertTrue(any(item["code"] == "LATENCY_RESULT_SCOPE_INVALID" for item in self._result()["errors"]))

    def test_absolute_path_leak_is_rejected(self) -> None:
        doc = self._load("execution_summary.json")
        doc["output_root"] = "/Users/example/thermal"
        self._save("execution_summary.json", doc)
        self.assertTrue(any(item["code"] == "ABSOLUTE_PATH_LEAK" for item in self._result()["errors"]))

    def test_checksum_coverage_is_required(self) -> None:
        checksum = self.evidence / "checksums.sha256"
        lines = [line for line in checksum.read_text().splitlines() if not line.endswith("  latency_results.json")]
        checksum.write_text("\n".join(lines) + "\n")
        self.assertTrue(any(item["code"] == "CHECKSUM_COVERAGE_MISSING" for item in self._result()["errors"]))

    def test_selected_candidate_cannot_be_deployment_validated(self) -> None:
        doc = self._load("candidate_lock.json")
        doc["thermal44_deployment_validated"] = True
        self._save("candidate_lock.json", doc)
        self.assertTrue(any(item["code"] == "CANDIDATE_LOCK_INVALID" for item in self._result()["errors"]))


if __name__ == "__main__":
    unittest.main()
