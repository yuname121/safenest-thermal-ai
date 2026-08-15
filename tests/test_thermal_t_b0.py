"""Focused metadata-only tests for Thermal T-B0 protocol integrity."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.validate_thermal_t_b0 import EVIDENCE_REL, validate_evidence


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / EVIDENCE_REL


class TestThermalTB0(unittest.TestCase):
    def _mutated_result(self, filename: str, mutate) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "T-B0"
            shutil.copytree(MANIFEST, target)
            path = target / filename
            data = json.loads(path.read_text(encoding="utf-8"))
            mutate(data)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return validate_evidence(repo_root=ROOT, evidence_dir=target)

    def test_clean_validator_passes_with_limitations(self) -> None:
        result = validate_evidence(repo_root=ROOT, evidence_dir=MANIFEST)
        self.assertEqual(result["evidence_validation"], "PASS")
        self.assertEqual(result["overall_outcome"], "PASS_WITH_LIMITATIONS")
        self.assertEqual(result["t_b1_authorized"], "YES_WITH_LIMITATIONS")

    def test_legacy_npz_rejected_as_authoritative(self) -> None:
        result = self._mutated_result("dataset_authority.json", lambda d: d["legacy_npz"].update(authority="ALLOWED"))
        self.assertEqual(result["evidence_validation"], "FAIL")

    def test_real_cannot_become_locked_test(self) -> None:
        result = self._mutated_result("dataset_authority.json", lambda d: d["locked_test"].update(available=True))
        self.assertEqual(result["evidence_validation"], "FAIL")

    def test_real_cannot_be_used_for_winner_selection(self) -> None:
        result = self._mutated_result("evaluation_role_policy.json", lambda d: d["roles"]["REAL_EVAL_DEVELOPMENT"].update(winner_selection=True))
        self.assertEqual(result["evidence_validation"], "FAIL")

    def test_validation_cannot_fit_preprocessing(self) -> None:
        result = self._mutated_result("evaluation_role_policy.json", lambda d: d["roles"]["VALIDATION"].update(preprocessing_fit=True))
        self.assertEqual(result["evidence_validation"], "FAIL")

    def test_train_only_fitted_preprocessing_is_enforced(self) -> None:
        def mutate(data):
            data["candidate_profiles"][1]["fit_partition"] = "VALIDATION"

        result = self._mutated_result("preprocessing_candidate_registry.json", mutate)
        self.assertEqual(result["evidence_validation"], "FAIL")

    def test_source_labels_cannot_be_overwritten_by_proxy(self) -> None:
        result = self._mutated_result("target_contract.json", lambda d: d["label_layers"].update(proxy_overwrites_source=True))
        self.assertEqual(result["evidence_validation"], "FAIL")

    def test_human_fall_remains_derived_posture_proxy(self) -> None:
        result = self._mutated_result("target_contract.json", lambda d: d["compatibility_proxy"]["LYING"].update(mapping_type="DIRECT_SOURCE_EQUIVALENT"))
        self.assertEqual(result["evidence_validation"], "FAIL")

    def test_random_and_hash_resplits_are_rejected(self) -> None:
        result = self._mutated_result("dataset_authority.json", lambda d: d.update(random_resplit="ALLOWED"))
        self.assertEqual(result["evidence_validation"], "FAIL")

    def test_near_duplicate_limitation_cannot_be_silently_removed(self) -> None:
        result = self._mutated_result("near_duplicate_sensitivity_policy.json", lambda d: d["measured_counts_in_t_a6"].update(train_validation_confirmed_pairs=0))
        self.assertEqual(result["evidence_validation"], "FAIL")

    def test_metric_policy_is_deterministic_and_frozen(self) -> None:
        first = validate_evidence(repo_root=ROOT, evidence_dir=MANIFEST)
        second = validate_evidence(repo_root=ROOT, evidence_dir=MANIFEST)
        self.assertEqual(first, second)
        self.assertEqual(first["error_count"], 0)

    def test_winner_selection_policy_is_frozen(self) -> None:
        result = self._mutated_result("winner_selection_policy.json", lambda d: d.update(primary_metric="accuracy"))
        self.assertEqual(result["evidence_validation"], "FAIL")

    def test_seed_policy_is_complete(self) -> None:
        result = self._mutated_result("randomness_policy.json", lambda d: d["seed_bindings"].pop("tensorflow"))
        self.assertEqual(result["evidence_validation"], "FAIL")

    def test_legacy_model_cannot_be_marked_canonical_trained(self) -> None:
        result = self._mutated_result("existing_model_inventory.json", lambda d: d.update(canonical_t_a6_trained_claim_allowed=True))
        self.assertEqual(result["evidence_validation"], "FAIL")

    def test_preprocessing_profile_identity_tamper_is_rejected(self) -> None:
        result = self._mutated_result("preprocessing_candidate_registry.json", lambda d: d["candidate_profiles"][0].update(profile_id="P9_TAMPERED"))
        self.assertEqual(result["evidence_validation"], "FAIL")

    def test_dataset_checksum_identity_tamper_is_rejected(self) -> None:
        result = self._mutated_result("dataset_authority.json", lambda d: d["roles"]["TRAIN"].update(artifact_sha256="0" * 64))
        self.assertEqual(result["evidence_validation"], "FAIL")

    def test_t_b1_results_are_rejected(self) -> None:
        result = self._mutated_result("t_b0_protocol.json", lambda d: d.update(performance_result={"accuracy": 1.0}))
        self.assertEqual(result["evidence_validation"], "FAIL")

    def test_new_trained_model_artifact_is_rejected(self) -> None:
        def mutate(data):
            data["candidates"][1]["artifact_path"] = "models/thermal/new_candidate.tflite"

        result = self._mutated_result("model_candidate_registry.json", mutate)
        self.assertEqual(result["evidence_validation"], "FAIL")

    def test_t_b0_does_not_create_new_model_files(self) -> None:
        tracked = list(MANIFEST.glob("*.tflite")) + list(MANIFEST.glob("*.h5")) + list(MANIFEST.glob("*.keras")) + list(MANIFEST.glob("*.npy")) + list(MANIFEST.glob("*.npz"))
        self.assertEqual(tracked, [])
        protocol = json.loads((MANIFEST / "t_b0_protocol.json").read_text(encoding="utf-8"))
        self.assertFalse(protocol["scope"]["full_training_performed"])
        self.assertFalse(protocol["scope"]["tflite_candidate_generated"])


if __name__ == "__main__":
    unittest.main()
