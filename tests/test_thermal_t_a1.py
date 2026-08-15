"""Focused tests for Thermal T-A1 safe reading and raw-unit evidence."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.thermal.raw_reader import (  # noqa: E402
    BBoxInvalidError,
    FrameChannelMismatchError,
    FrameDtypeMismatchError,
    FrameInvalidRangeError,
    FrameLabelLinkageError,
    FrameShapeMismatchError,
    LabelCountMismatchError,
    LabelFileMissingError,
    LabelParseError,
    LabelValueInvalidError,
    PathPolicyError,
    PNGDecodeError,
    PNGTruncatedError,
    SDTThermalRawReader,
    SourceArchiveIdentityMismatchError,
    SourceArchiveCorruptError,
    SourceArchiveNotFoundError,
    SourceFrameIndexInvalidError,
    SourceMemberDuplicateError,
    SourceMemberMissingError,
    SourceMemberUnexpectedError,
    encoded_to_celsius,
    encoded_to_kelvin,
)


def _png(array: np.ndarray, mode: str | None = None) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG")
    return buffer.getvalue()


def _thermal_png(value: int = 30_000) -> bytes:
    frame = np.full((480, 640), value, dtype=np.uint16)
    frame[0, 0] = value - 1
    return _png(frame)


def _make_archive(
    tmp_path: Path,
    *,
    count: int = 4,
    labels: list[str] | None = None,
    thermal_payloads: dict[int, bytes] | None = None,
    omit: set[str] | None = None,
    extras: list[tuple[str, bytes]] | None = None,
) -> Path:
    path = tmp_path / "fixture.zip"
    omit = omit or set()
    labels = labels or [
        "0,1,2,10,20",
        "1,1.5,2.5,10.5,20.5",
        "2,3,4,30,40",
        "3,-1,-1,-1,-1",
    ][:count]
    thermal_payloads = thermal_payloads or {}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(count):
            t_name = f"test/image_t_{index}.png"
            d_name = f"test/image_d_{index}.png"
            if t_name not in omit:
                archive.writestr(t_name, thermal_payloads.get(index, _thermal_png(29_900 + index)))
            if d_name not in omit:
                archive.writestr(d_name, b"depth-not-read-by-thermal-reader")
        if "test/labels.txt" not in omit:
            archive.writestr("test/labels.txt", "\n".join(labels) + "\n")
        for name, payload in extras or []:
            archive.writestr(name, payload)
    return path


def _reader(path: Path, count: int = 4, **overrides: object) -> SDTThermalRawReader:
    payload = path.read_bytes()
    kwargs: dict[str, object] = {
        "repo_root": path.parent,
        "archive_path": path.name,
        "expected_archive_size": len(payload),
        "expected_archive_md5": hashlib.md5(payload).hexdigest(),
        "expected_archive_sha256": hashlib.sha256(payload).hexdigest(),
        "expected_frame_count": count,
    }
    kwargs.update(overrides)
    return SDTThermalRawReader(**kwargs)


def test_valid_uint16_source_values_and_provenance_are_preserved(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path)
    reader = _reader(archive)
    inventory = reader.inspect_archive()
    frame = reader.read_frame(1)
    assert inventory["thermal_depth_label_linkage"] == "ONE_TO_ONE_BY_ZERO_BASED_INDEX"
    assert inventory["class_counts"] == {"0": 1, "1": 1, "2": 1, "3": 1}
    assert frame.raw_encoded_frame.shape == (480, 640)
    assert frame.raw_encoded_frame.dtype == np.uint16
    assert int(frame.raw_encoded_frame[0, 1]) == 29_901
    assert frame.source_pose_name == "SITTING"
    assert frame.source_bbox == (1.5, 2.5, 10.5, 20.5)
    assert frame.raw_encoded_frame.flags.writeable is False
    assert frame.source_subject_status == "ABSENT"


def test_official_temperature_conversion_witness() -> None:
    encoded = np.array([30_000], dtype=np.uint16)
    assert encoded_to_kelvin(encoded).tolist() == [300.0]
    assert encoded_to_celsius(encoded).tolist() == [26.85]


@pytest.mark.parametrize("value", [np.nan, np.inf, -1, 65_536, 1.5])
def test_conversion_rejects_nonfinite_out_of_range_or_fractional(value: float) -> None:
    with pytest.raises(Exception):
        encoded_to_celsius(np.array([value]))


def test_wrong_shape_fails_closed(tmp_path: Path) -> None:
    bad = _png(np.ones((120, 160), dtype=np.uint16))
    reader = _reader(_make_archive(tmp_path, thermal_payloads={0: bad}))
    with pytest.raises(FrameShapeMismatchError):
        reader.read_frame(0)


def test_wrong_channel_count_fails_closed(tmp_path: Path) -> None:
    bad = _png(np.ones((480, 640, 3), dtype=np.uint8))
    reader = _reader(_make_archive(tmp_path, thermal_payloads={0: bad}))
    with pytest.raises((FrameChannelMismatchError, FrameDtypeMismatchError)):
        reader.read_frame(0)


def test_wrong_bit_depth_fails_closed(tmp_path: Path) -> None:
    bad = _png(np.ones((480, 640), dtype=np.uint8))
    reader = _reader(_make_archive(tmp_path, thermal_payloads={0: bad}))
    with pytest.raises(FrameDtypeMismatchError):
        reader.read_frame(0)


@pytest.mark.parametrize("payload,error_type", [(b"not-png", PNGTruncatedError), (b"x" * 40, PNGDecodeError)])
def test_invalid_png_fails_closed(tmp_path: Path, payload: bytes, error_type: type[Exception]) -> None:
    reader = _reader(_make_archive(tmp_path, thermal_payloads={0: payload}))
    with pytest.raises(error_type):
        reader.read_frame(0)


def test_missing_thermal_member_fails_closed(tmp_path: Path) -> None:
    reader = _reader(_make_archive(tmp_path, omit={"test/image_t_2.png"}))
    with pytest.raises(SourceMemberMissingError):
        reader.inspect_archive()


def test_missing_labels_fails_closed(tmp_path: Path) -> None:
    reader = _reader(_make_archive(tmp_path, omit={"test/labels.txt"}))
    with pytest.raises(LabelFileMissingError):
        reader.inspect_archive()


@pytest.mark.parametrize(
    "labels,error_type",
    [
        (["0,1,2,3"], LabelCountMismatchError),
        (["x,1,2,3,4"] * 4, LabelParseError),
        (["9,1,2,3,4"] * 4, LabelValueInvalidError),
        (["3,0,0,0,0"] * 4, BBoxInvalidError),
        (["0,-1,0,10,20"] * 4, BBoxInvalidError),
    ],
)
def test_invalid_labels_fail_closed(tmp_path: Path, labels: list[str], error_type: type[Exception]) -> None:
    reader = _reader(_make_archive(tmp_path, labels=labels))
    with pytest.raises(error_type):
        reader.inspect_archive()


def test_duplicate_member_name_fails_closed(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        archive = _make_archive(tmp_path, extras=[("test/image_t_0.png", _thermal_png())])
    reader = _reader(archive)
    with pytest.raises(SourceMemberDuplicateError):
        reader.inspect_archive()


def test_duplicate_frame_index_with_distinct_name_fails_closed(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path, extras=[("test/image_t_00.png", _thermal_png())])
    reader = _reader(archive)
    with pytest.raises(Exception) as caught:
        reader.inspect_archive()
    assert caught.value.code == "SOURCE_FRAME_DUPLICATE"


def test_thermal_depth_label_index_mismatch_fails_closed(tmp_path: Path) -> None:
    archive = _make_archive(
        tmp_path,
        omit={"test/image_d_3.png"},
        extras=[("test/image_d_4.png", b"depth")],
    )
    reader = _reader(archive)
    with pytest.raises(FrameLabelLinkageError):
        reader.inspect_archive()


@pytest.mark.parametrize(
    "name,error_type",
    [
        ("../escape.png", SourceMemberUnexpectedError),
        ("/absolute.png", SourceMemberUnexpectedError),
        ("test/extra.txt", SourceMemberUnexpectedError),
        ("test/image_t_x.png", SourceFrameIndexInvalidError),
    ],
)
def test_unsafe_or_unexpected_member_fails_closed(
    tmp_path: Path, name: str, error_type: type[Exception]
) -> None:
    reader = _reader(_make_archive(tmp_path, extras=[(name, b"x")]))
    with pytest.raises(error_type):
        reader.inspect_archive()


def test_archive_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path)
    reader = _reader(archive, expected_archive_sha256="0" * 64)
    with pytest.raises(SourceArchiveIdentityMismatchError):
        reader.inspect_archive()


def test_missing_and_corrupt_archives_fail_closed(tmp_path: Path) -> None:
    missing = SDTThermalRawReader(
        repo_root=tmp_path,
        archive_path="missing.zip",
        expected_archive_size=None,
        expected_archive_md5=None,
        expected_archive_sha256=None,
        expected_frame_count=1,
    )
    with pytest.raises(SourceArchiveNotFoundError):
        missing.inspect_archive()

    corrupt_path = tmp_path / "corrupt.zip"
    corrupt_path.write_bytes(b"PK\x03\x04not-a-complete-zip")
    corrupt = _reader(corrupt_path, count=1)
    with pytest.raises(SourceArchiveCorruptError):
        corrupt.inspect_archive()


def test_index_validation_and_repeat_decode_are_deterministic(tmp_path: Path) -> None:
    reader = _reader(_make_archive(tmp_path))
    first = reader.read_frame(2)
    second = reader.read_frame(2)
    assert first.provenance_dict() == second.provenance_dict()
    assert np.array_equal(first.raw_encoded_frame, second.raw_encoded_frame)
    for invalid in (-1, 4, True, 1.0):
        with pytest.raises(SourceFrameIndexInvalidError):
            reader.read_frame(invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, 65_535])
def test_fully_constant_container_extreme_fails_closed(tmp_path: Path, value: int) -> None:
    payload = _png(np.full((480, 640), value, dtype=np.uint16))
    reader = _reader(_make_archive(tmp_path, thermal_payloads={0: payload}))
    with pytest.raises(FrameInvalidRangeError):
        reader.read_frame(0)


def test_absolute_archive_path_is_forbidden(tmp_path: Path) -> None:
    with pytest.raises(PathPolicyError):
        SDTThermalRawReader(repo_root=tmp_path, archive_path="/tmp/source.zip")


def test_reader_has_no_preprocessing_model_or_legacy_npz_path() -> None:
    source = (ROOT / "datasets/thermal/raw_reader.py").read_text(encoding="utf-8")
    forbidden = ["processed_thermal_80x62.npz", "ThermalInterpreter", "tflite", "extractall(", ".resize("]
    assert not any(term in source for term in forbidden)
    assert "min-max" not in source.lower()


def test_tracked_t_a1_json_is_canonical_and_portable() -> None:
    evidence = ROOT / "datasets/thermal/manifests/T-A1_safe_reader_raw_unit_contract"
    if not evidence.exists():
        pytest.skip("T-A1 evidence is generated after reader unit tests")
    for path in sorted(evidence.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert path.read_text(encoding="utf-8") == json.dumps(
            data, ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"
        assert "/Users/" not in path.read_text(encoding="utf-8")


def test_standalone_validator_independently_rejects_bad_unit_formula(tmp_path: Path) -> None:
    source = ROOT / "datasets/thermal/manifests/T-A1_safe_reader_raw_unit_contract"
    if not source.exists():
        pytest.skip("T-A1 evidence is generated after reader unit tests")
    copied = tmp_path / "evidence"
    shutil.copytree(source, copied)
    contract_path = copied / "raw_unit_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["celsius_formula"] = "encoded_uint16 / 100"
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    from scripts.validate_thermal_t_a1 import validate_evidence

    result = validate_evidence(
        repo_root=ROOT,
        evidence_dir=copied,
        check_checksums=False,
        verify_real_payload=False,
    )
    assert result["evidence_validation"] == "FAIL"
    assert "UNIT_FORMULA_INVALID" in {item["code"] for item in result["errors"]}


def test_deterministic_pilot_manifest_generation_from_real_archive() -> None:
    archive = ROOT / "datasets/raw_archives/thermal_split_zips/test.zip"
    if not archive.is_file():
        pytest.skip("owner-local Git-ignored SDT test.zip is unavailable in this checkout")
    from scripts.generate_thermal_t_a1 import build_artifacts, canonical_json

    reader = SDTThermalRawReader(repo_root=ROOT)
    first = canonical_json(build_artifacts(reader))
    second = canonical_json(build_artifacts(reader))
    assert first == second


def test_real_sdt_archive_pilot_when_materialized() -> None:
    archive = ROOT / "datasets/raw_archives/thermal_split_zips/test.zip"
    if not archive.is_file():
        pytest.skip("owner-local Git-ignored SDT test.zip is unavailable in this checkout")
    reader = SDTThermalRawReader(repo_root=ROOT)
    inventory = reader.inspect_archive()
    assert inventory["thermal_member_count"] == 8_000
    assert inventory["class_counts"] == {"0": 2_000, "1": 2_000, "2": 2_000, "3": 2_000}
    for index, pose in [(0, 0), (2_000, 1), (4_000, 2), (6_000, 3), (7_999, 3)]:
        frame = reader.read_frame(index)
        assert frame.source_pose_label == pose
        assert frame.raw_encoded_frame.shape == (480, 640)
        assert frame.raw_encoded_frame.dtype == np.uint16
