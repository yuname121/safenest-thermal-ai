"""Payload-free focused tests for the Thermal T-B4 evidence contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from datasets.thermal.t_b1_preprocessing import canonical_json
from scripts import validate_thermal_t_b4 as validator


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / validator.EVIDENCE_REL


def _json(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def _errors(function, document, *args):
    errors: list[dict[str, str]] = []
    function(document, *args, errors)
    return errors


def test_full_evidence_passes_with_live_predecessors(monkeypatch):
    passing = {phase: {"evidence_validation": "PASS", "overall_outcome": "PASS_WITH_LIMITATIONS"} for phase in ("T-A6", "T-B0", "T-B1", "T-B2", "T-B3")}
    monkeypatch.setattr(validator, "_validate_predecessors", lambda repo_root, errors: passing)
    result = validator.validate_evidence(repo_root=ROOT, evidence_dir=EVIDENCE, mode=validator.FULL_MODE, check_checksums=True)
    assert result["evidence_validation"] == "PASS"


def test_train_only_representative_manifest_passes():
    policy = _json("representative_calibration_policy.json")
    manifest = _json("representative_sample_manifest.json")
    errors: list[dict[str, str]] = []
    validator._validate_calibration(policy, manifest, errors)
    assert errors == []


def test_validation_sample_in_representative_manifest_is_rejected():
    policy = _json("representative_calibration_policy.json")
    manifest = copy.deepcopy(_json("representative_sample_manifest.json"))
    manifest["rows"][0]["role"] = "VALIDATION"
    errors: list[dict[str, str]] = []
    validator._validate_calibration(policy, manifest, errors)
    assert any(item["code"] == "REPRESENTATIVE_ROW_INVALID" for item in errors)


def test_real_sample_in_calibration_policy_is_rejected():
    policy = copy.deepcopy(_json("representative_calibration_policy.json"))
    manifest = _json("representative_sample_manifest.json")
    policy["real_samples_used"] = 1
    errors: list[dict[str, str]] = []
    validator._validate_calibration(policy, manifest, errors)
    assert any(item["code"] == "REPRESENTATIVE_POLICY_INVALID" for item in errors)


def test_calibration_policy_checksum_change_is_rejected():
    policy = copy.deepcopy(_json("representative_calibration_policy.json"))
    manifest = _json("representative_sample_manifest.json")
    policy["selection_algorithm"] = "POST_HOC_VALIDATION_TUNING"
    errors: list[dict[str, str]] = []
    validator._validate_calibration(policy, manifest, errors)
    assert any(item["code"] == "REPRESENTATIVE_POLICY_CHECKSUM_INVALID" for item in errors)


def test_temperature_bins_changed_after_results_are_rejected():
    document = copy.deepcopy(_json("temperature_range_policy.json"))
    document["boundaries_celsius"][3] += 0.1
    audit = _json("temperature_range_error.json")
    errors: list[dict[str, str]] = []
    validator._validate_temperature_error(audit, document, errors)
    assert any(item["code"] == "TEMPERATURE_POLICY_CHECKSUM_MISMATCH" for item in errors)


def test_full_int8_dtype_or_fallback_is_rejected():
    documents = {name: _json(name) for name in validator.FULL_JSON}
    documents["tflite_int8_artifact.json"]["output"]["dtype"] = "float32"
    errors: list[dict[str, str]] = []
    validator._validate_artifacts(documents, errors)
    assert any(item["code"] == "TENSOR_CONTRACT_INVALID" for item in errors)

    documents = {name: _json(name) for name in validator.FULL_JSON}
    documents["op_inventory.json"]["full_int8"]["ops"].append("FLOAT32")
    errors = []
    validator._validate_artifacts(documents, errors)
    assert any(item["code"] == "FLOAT_FALLBACK_PRESENT" for item in errors)


def test_float_io_alone_does_not_classify_dynamic_range_as_fp32():
    documents = {name: _json(name) for name in validator.FULL_JSON}
    fp32 = documents["tflite_fp32_artifact.json"]
    # Keep the valid float32 input/output contract but inject the historical
    # Optimize.DEFAULT/internal-int8 evidence.  The validator must reject it.
    fp32["conversion"].update({"optimizations": ["DEFAULT"], "representative_dataset_attached": False, "float16_enabled": False, "dynamic_range_quantization": True, "quantization_mode": "DYNAMIC_RANGE"})
    fp32["internal_dtype_counts"] = {"float32": 15, "int8": 2, "int32": 1}
    fp32["quantized_tensor_count"] = 2
    fp32["quantized_parameter_tensor_count"] = 2
    fp32["nonzero_quantization_tensor_count"] = 2
    errors: list[dict[str, str]] = []
    validator._validate_artifacts(documents, errors)
    codes = {item["code"] for item in errors}
    assert "FP32_QUANTIZATION_POLICY_INVALID" in codes
    assert "FP32_INTERNAL_QUANTIZATION_INVALID" in codes


def test_absolute_and_archive_paths_are_rejected():
    import tempfile

    errors: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad.json"
        path.write_text(canonical_json({"path": "/Users/junwoo/raw.tflite", "history": "archive/old.json"}), encoding="utf-8")
        validator._read_documents(Path(directory), ["bad.json"], errors)
    assert {item["code"] for item in errors} >= {"NONPORTABLE_PATH", "ARCHIVE_TREATED_AS_ACTIVE"}


def test_protocol_rejects_retraining_real_calibration_and_later_phase():
    document = copy.deepcopy(_json("t_b4_protocol.json"))
    document["retraining"] = True
    document["real_used_for_calibration"] = True
    document["t_b5_started"] = True
    errors = _errors(validator._validate_protocol, document)
    assert {item["code"] for item in errors} >= {"PROTOCOL_INVALID"}


def test_missing_absolute_threshold_remains_explicit_limitation():
    document = _json("t_b4_protocol.json")
    assert document["equivalence_contract"] == "EQUIVALENCE_MEASURED_NO_PREEXISTING_ABSOLUTE_ACCEPTANCE_THRESHOLD"
