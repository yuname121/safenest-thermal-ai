#!/usr/bin/env python3
"""Deterministic fail-closed reader for the T-A0-selected SDT Thermal source.

The reader preserves the distributed uint16 ``image_t`` values and original
pose labels.  It does not resize, normalize, relabel, quantize, extract the
archive, or invoke a model.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
import stat
import struct
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError


SDT_DATASET_ID = "local_sdt_zenodo_4124309"
SDT_DOI = "10.5281/zenodo.4124309"
SDT_SOURCE_SPLIT = "test"
DEFAULT_ARCHIVE_PATH = "datasets/raw_archives/thermal_split_zips/test.zip"
DEFAULT_ARCHIVE_SIZE = 1_740_348_425
DEFAULT_ARCHIVE_MD5 = "d59a739f3b5ecf373c94046fb94cd94f"
DEFAULT_ARCHIVE_SHA256 = "3a838bd70835e579ecfaa820a6c0b4cbc6ba7b76729417c73845f0c959281449"
DEFAULT_FRAME_COUNT = 8_000
LABELS_MEMBER = "test/labels.txt"
THERMAL_MEMBER_RE = re.compile(r"test/image_t_(\d+)\.png")
DEPTH_MEMBER_RE = re.compile(r"test/image_d_(\d+)\.png")
POSE_NAMES = {0: "LYING", 1: "SITTING", 2: "STANDING", 3: "EMPTY_ROOM"}
DISTRIBUTED_FRAME_SHAPE = (480, 640)
NATIVE_SENSOR_SHAPE = (120, 160)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_IMAGE_MEMBER_BYTES = 2 * 1024 * 1024
MAX_LABEL_MEMBER_BYTES = 2 * 1024 * 1024


class SDTThermalReaderError(Exception):
    """Base class with a stable machine-readable failure code."""

    code = "SDT_THERMAL_READER_ERROR"

    def __init__(self, message: str) -> None:
        self.detail = message
        super().__init__(f"{self.code}: {message}")


class SourceArchiveNotFoundError(SDTThermalReaderError):
    code = "SOURCE_ARCHIVE_NOT_FOUND"


class SourceArchiveNotMaterializedError(SDTThermalReaderError):
    code = "SOURCE_ARCHIVE_NOT_MATERIALIZED"


class SourceArchiveIdentityMismatchError(SDTThermalReaderError):
    code = "SOURCE_ARCHIVE_IDENTITY_MISMATCH"


class SourceArchiveCorruptError(SDTThermalReaderError):
    code = "SOURCE_ARCHIVE_CORRUPT"


class SourceMemberMissingError(SDTThermalReaderError):
    code = "SOURCE_MEMBER_MISSING"


class SourceMemberUnexpectedError(SDTThermalReaderError):
    code = "SOURCE_MEMBER_UNEXPECTED"


class SourceMemberDuplicateError(SDTThermalReaderError):
    code = "SOURCE_MEMBER_DUPLICATE"


class SourceFrameIndexInvalidError(SDTThermalReaderError):
    code = "SOURCE_FRAME_INDEX_INVALID"


class SourceFrameDuplicateError(SDTThermalReaderError):
    code = "SOURCE_FRAME_DUPLICATE"


class PNGDecodeError(SDTThermalReaderError):
    code = "PNG_DECODE_FAILED"


class PNGTruncatedError(SDTThermalReaderError):
    code = "PNG_TRUNCATED"


class FrameShapeMismatchError(SDTThermalReaderError):
    code = "FRAME_SHAPE_MISMATCH"


class FrameDtypeMismatchError(SDTThermalReaderError):
    code = "FRAME_DTYPE_MISMATCH"


class FrameChannelMismatchError(SDTThermalReaderError):
    code = "FRAME_CHANNEL_MISMATCH"


class FrameNonfiniteError(SDTThermalReaderError):
    code = "FRAME_NONFINITE"


class FrameInvalidRangeError(SDTThermalReaderError):
    code = "FRAME_INVALID_RANGE"


class LabelFileMissingError(SDTThermalReaderError):
    code = "LABEL_FILE_MISSING"


class LabelParseError(SDTThermalReaderError):
    code = "LABEL_PARSE_FAILED"


class LabelCountMismatchError(SDTThermalReaderError):
    code = "LABEL_COUNT_MISMATCH"


class LabelValueInvalidError(SDTThermalReaderError):
    code = "LABEL_VALUE_INVALID"


class BBoxInvalidError(SDTThermalReaderError):
    code = "BBOX_INVALID"


class FrameLabelLinkageError(SDTThermalReaderError):
    code = "FRAME_LABEL_LINKAGE_FAILED"


class UnsupportedRepresentationError(SDTThermalReaderError):
    code = "UNSUPPORTED_REPRESENTATION"


class PhysicalConversionError(SDTThermalReaderError):
    code = "PHYSICAL_CONVERSION_FAILED"


class PathPolicyError(SDTThermalReaderError):
    code = "PATH_POLICY_VIOLATION"


@dataclass(frozen=True)
class SDTSourceLabel:
    source_frame_index: int
    source_pose_label: int
    source_pose_name: str
    source_bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class SDTThermalSourceFrame:
    source_dataset_id: str
    source_doi: str
    source_split: str
    source_archive_path: str
    source_archive_size_bytes: int
    source_archive_md5: str
    source_archive_sha256: str
    source_member_name: str
    source_member_index: int
    source_member_crc32: str
    source_member_sha256: str
    source_frame_index: int
    source_pose_label: int
    source_pose_name: str
    source_bbox: tuple[float, float, float, float]
    distributed_frame_shape: tuple[int, int]
    native_sensor_shape: tuple[int, int]
    source_dtype: str
    decoded_byte_order: str
    source_representation: str
    source_temperature_encoding: str
    source_timestamp_status: str
    source_subject_status: str
    source_session_status: str
    source_sequence_status: str
    source_event_status: str
    quality_flags: tuple[str, ...]
    raw_encoded_frame_sha256: str
    raw_encoded_frame: np.ndarray

    def kelvin(self) -> np.ndarray:
        return encoded_to_kelvin(self.raw_encoded_frame)

    def celsius(self) -> np.ndarray:
        return encoded_to_celsius(self.raw_encoded_frame)

    def provenance_dict(self) -> dict[str, Any]:
        """Return compact provenance without serializing the frame payload."""
        return {
            "source_dataset_id": self.source_dataset_id,
            "source_doi": self.source_doi,
            "source_split": self.source_split,
            "source_archive_path": self.source_archive_path,
            "source_archive_size_bytes": self.source_archive_size_bytes,
            "source_archive_md5": self.source_archive_md5,
            "source_archive_sha256": self.source_archive_sha256,
            "source_member_name": self.source_member_name,
            "source_member_index": self.source_member_index,
            "source_member_crc32": self.source_member_crc32,
            "source_member_sha256": self.source_member_sha256,
            "source_frame_index": self.source_frame_index,
            "source_pose_label": self.source_pose_label,
            "source_pose_name": self.source_pose_name,
            "source_bbox": list(self.source_bbox),
            "distributed_frame_shape": list(self.distributed_frame_shape),
            "native_sensor_shape": list(self.native_sensor_shape),
            "source_dtype": self.source_dtype,
            "decoded_byte_order": self.decoded_byte_order,
            "source_representation": self.source_representation,
            "source_temperature_encoding": self.source_temperature_encoding,
            "source_timestamp_status": self.source_timestamp_status,
            "source_subject_status": self.source_subject_status,
            "source_session_status": self.source_session_status,
            "source_sequence_status": self.source_sequence_status,
            "source_event_status": self.source_event_status,
            "quality_flags": list(self.quality_flags),
            "raw_encoded_frame_sha256": self.raw_encoded_frame_sha256,
        }


def _canonical_encoded_bytes(array: np.ndarray) -> bytes:
    return np.asarray(array, dtype="<u2", order="C").tobytes(order="C")


def encoded_frame_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(_canonical_encoded_bytes(array)).hexdigest()


def _validated_encoded_numeric(encoded: Any) -> np.ndarray:
    try:
        numeric = np.asarray(encoded)
    except Exception as exc:
        raise PhysicalConversionError(f"cannot interpret encoded input: {exc}") from exc
    if not np.issubdtype(numeric.dtype, np.number):
        raise PhysicalConversionError(f"encoded input is not numeric: {numeric.dtype}")
    finite = np.asarray(numeric, dtype=np.float64)
    if not np.all(np.isfinite(finite)):
        raise FrameNonfiniteError("encoded or derived numeric path contains NaN or infinity")
    if np.any(finite < 0) or np.any(finite > np.iinfo(np.uint16).max):
        raise FrameInvalidRangeError("encoded values are outside the uint16 container range")
    if not np.all(finite == np.floor(finite)):
        raise PhysicalConversionError("encoded values must be integer-valued")
    return finite


def encoded_to_kelvin(encoded: Any) -> np.ndarray:
    """Apply the official SDT Kelvin conversion without changing source values."""
    result = _validated_encoded_numeric(encoded) / 100.0
    if not np.all(np.isfinite(result)):
        raise FrameNonfiniteError("Kelvin conversion produced NaN or infinity")
    result.setflags(write=False)
    return result


def encoded_to_celsius(encoded: Any) -> np.ndarray:
    """Apply the official SDT Celsius conversion ``(raw - 27315) / 100``."""
    result = (_validated_encoded_numeric(encoded) - 27_315.0) / 100.0
    if not np.all(np.isfinite(result)):
        raise FrameNonfiniteError("Celsius conversion produced NaN or infinity")
    result.setflags(write=False)
    return result


def compute_file_hashes(path: Path, chunk_size: int = 1024 * 1024) -> tuple[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def _validate_repository_relative_path(value: str) -> str:
    if not value or value.startswith(("/", "~/", "file://")) or "\\" in value:
        raise PathPolicyError(f"path must be repository-relative POSIX: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or any(part in {"", "."} for part in pure.parts):
        raise PathPolicyError(f"unsafe repository-relative path: {value!r}")
    if re.match(r"^[A-Za-z]:", value):
        raise PathPolicyError(f"drive-specific path is forbidden: {value!r}")
    return pure.as_posix()


def _validate_member_path(name: str) -> None:
    if not name or name.startswith("/") or "\\" in name or re.match(r"^[A-Za-z]:", name):
        raise SourceMemberUnexpectedError(f"unsafe archive member path: {name!r}")
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts:
        raise SourceMemberUnexpectedError(f"path traversal archive member: {name!r}")


class SDTThermalRawReader:
    """Read verified SDT thermal members directly from a ZIP archive."""

    def __init__(
        self,
        *,
        repo_root: str | Path | None = None,
        archive_path: str = DEFAULT_ARCHIVE_PATH,
        expected_archive_size: int | None = DEFAULT_ARCHIVE_SIZE,
        expected_archive_md5: str | None = DEFAULT_ARCHIVE_MD5,
        expected_archive_sha256: str | None = DEFAULT_ARCHIVE_SHA256,
        expected_frame_count: int = DEFAULT_FRAME_COUNT,
    ) -> None:
        self.repo_root = (
            Path(repo_root).resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self.archive_rel_path = _validate_repository_relative_path(archive_path)
        self.archive_path = self.repo_root / self.archive_rel_path
        self.expected_archive_size = expected_archive_size
        self.expected_archive_md5 = expected_archive_md5
        self.expected_archive_sha256 = expected_archive_sha256
        self.expected_frame_count = expected_frame_count
        if expected_frame_count <= 0:
            raise ValueError("expected_frame_count must be positive")
        self._archive_identity: dict[str, Any] | None = None
        self._inventory: dict[str, Any] | None = None
        self._labels: tuple[SDTSourceLabel, ...] | None = None
        self._thermal_info: dict[int, tuple[int, zipfile.ZipInfo]] = {}

    def _require_materialized_archive(self) -> None:
        if not self.archive_path.exists():
            raise SourceArchiveNotFoundError(self.archive_rel_path)
        metadata = self.archive_path.stat()
        dataless_flag = getattr(stat, "SF_DATALESS", 0x40000000)
        if metadata.st_size > 0 and (
            getattr(metadata, "st_blocks", 1) == 0
            or bool(getattr(metadata, "st_flags", 0) & dataless_flag)
        ):
            raise SourceArchiveNotMaterializedError(self.archive_rel_path)
        if not self.archive_path.is_file():
            raise SourceArchiveNotFoundError(f"not a regular file: {self.archive_rel_path}")

    def verify_archive_identity(self) -> dict[str, Any]:
        if self._archive_identity is not None:
            return dict(self._archive_identity)
        self._require_materialized_archive()
        size = self.archive_path.stat().st_size
        if self.expected_archive_size is not None and size != self.expected_archive_size:
            raise SourceArchiveIdentityMismatchError(
                f"size expected={self.expected_archive_size}, measured={size}"
            )
        md5, sha256 = compute_file_hashes(self.archive_path)
        if self.expected_archive_md5 is not None and md5 != self.expected_archive_md5:
            raise SourceArchiveIdentityMismatchError(
                f"MD5 expected={self.expected_archive_md5}, measured={md5}"
            )
        if self.expected_archive_sha256 is not None and sha256 != self.expected_archive_sha256:
            raise SourceArchiveIdentityMismatchError(
                f"SHA-256 expected={self.expected_archive_sha256}, measured={sha256}"
            )
        self._archive_identity = {
            "path": self.archive_rel_path,
            "size_bytes": size,
            "md5": md5,
            "sha256": sha256,
            "materialization_state": "LOCALLY_MATERIALIZED",
        }
        return dict(self._archive_identity)

    @staticmethod
    def _read_bounded_member(
        archive: zipfile.ZipFile,
        info: zipfile.ZipInfo,
        *,
        limit: int,
    ) -> bytes:
        if info.file_size > limit:
            raise SourceMemberUnexpectedError(
                f"member exceeds bounded read limit: {info.filename} ({info.file_size}>{limit})"
            )
        try:
            with archive.open(info, "r") as member:
                payload = member.read(limit + 1)
        except (zipfile.BadZipFile, RuntimeError, EOFError, OSError) as exc:
            raise SourceArchiveCorruptError(
                f"member read failed for {info.filename}: {exc}"
            ) from exc
        if len(payload) > limit:
            raise SourceMemberUnexpectedError(f"member exceeded read bound: {info.filename}")
        if len(payload) != info.file_size:
            raise PNGTruncatedError(
                f"member length mismatch for {info.filename}: {len(payload)} != {info.file_size}"
            )
        return payload

    @staticmethod
    def _parse_labels(payload: bytes, expected_count: int) -> tuple[SDTSourceLabel, ...]:
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise LabelParseError(f"labels.txt is not UTF-8: {exc}") from exc
        lines = text.splitlines()
        if len(lines) != expected_count:
            raise LabelCountMismatchError(
                f"expected {expected_count} label rows, found {len(lines)}"
            )
        labels: list[SDTSourceLabel] = []
        for index, line in enumerate(lines):
            parts = line.split(",")
            if len(parts) != 5:
                raise LabelParseError(f"row {index} has {len(parts)} fields, expected 5")
            try:
                pose = int(parts[0])
                bbox = tuple(float(value) for value in parts[1:])
            except ValueError as exc:
                raise LabelParseError(f"row {index} numeric parse failed: {line!r}") from exc
            if pose not in POSE_NAMES:
                raise LabelValueInvalidError(f"row {index} has unsupported pose {pose}")
            if not all(math.isfinite(value) for value in bbox):
                raise BBoxInvalidError(f"row {index} bbox contains NaN or infinity")
            if pose == 3:
                if bbox != (-1.0, -1.0, -1.0, -1.0):
                    raise BBoxInvalidError(
                        f"row {index} empty-room bbox is not the documented sentinel: {bbox}"
                    )
            else:
                x_min, y_min, x_max, y_max = bbox
                if not (
                    0.0 <= x_min <= x_max <= DISTRIBUTED_FRAME_SHAPE[1] + 0.5
                    and 0.0 <= y_min <= y_max <= DISTRIBUTED_FRAME_SHAPE[0] + 0.5
                ):
                    raise BBoxInvalidError(f"row {index} person bbox is invalid: {bbox}")
            labels.append(
                SDTSourceLabel(
                    source_frame_index=index,
                    source_pose_label=pose,
                    source_pose_name=POSE_NAMES[pose],
                    source_bbox=bbox,  # type: ignore[arg-type]
                )
            )
        return tuple(labels)

    def inspect_archive(self) -> dict[str, Any]:
        if self._inventory is not None:
            return dict(self._inventory)
        identity = self.verify_archive_identity()
        try:
            with zipfile.ZipFile(self.archive_path, "r") as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                for name in names:
                    _validate_member_path(name)
                duplicate_names = sorted(
                    name for name, count in Counter(names).items() if count > 1
                )
                if duplicate_names:
                    raise SourceMemberDuplicateError(
                        f"duplicate member names: {duplicate_names[:5]}"
                    )

                thermal: dict[int, tuple[int, zipfile.ZipInfo]] = {}
                depth: dict[int, tuple[int, zipfile.ZipInfo]] = {}
                labels_info: zipfile.ZipInfo | None = None
                unexpected: list[str] = []
                for member_index, info in enumerate(infos):
                    name = info.filename
                    if info.is_dir() and name == "test/":
                        continue
                    if name == LABELS_MEMBER:
                        labels_info = info
                        continue
                    thermal_match = THERMAL_MEMBER_RE.fullmatch(name)
                    depth_match = DEPTH_MEMBER_RE.fullmatch(name)
                    if thermal_match:
                        index = int(thermal_match.group(1))
                        if index in thermal:
                            raise SourceFrameDuplicateError(f"duplicate thermal index {index}")
                        thermal[index] = (member_index, info)
                    elif depth_match:
                        index = int(depth_match.group(1))
                        if index in depth:
                            raise SourceFrameDuplicateError(f"duplicate depth index {index}")
                        depth[index] = (member_index, info)
                    elif name.startswith(("test/image_t_", "test/image_d_")):
                        raise SourceFrameIndexInvalidError(f"invalid frame member name: {name}")
                    else:
                        unexpected.append(name)
                if unexpected:
                    raise SourceMemberUnexpectedError(
                        f"unexpected members: {sorted(unexpected)[:5]}"
                    )
                if labels_info is None:
                    raise LabelFileMissingError(LABELS_MEMBER)

                expected_indices = set(range(self.expected_frame_count))
                thermal_indices = set(thermal)
                depth_indices = set(depth)
                if len(thermal) != self.expected_frame_count:
                    raise SourceMemberMissingError(
                        f"expected {self.expected_frame_count} thermal members, found {len(thermal)}"
                    )
                if len(depth) != self.expected_frame_count:
                    raise SourceMemberMissingError(
                        f"expected {self.expected_frame_count} depth members, found {len(depth)}"
                    )
                if thermal_indices != expected_indices or depth_indices != expected_indices:
                    raise FrameLabelLinkageError(
                        "thermal/depth indices are not continuous and identical to label row indices"
                    )

                label_payload = self._read_bounded_member(
                    archive, labels_info, limit=MAX_LABEL_MEMBER_BYTES
                )
                labels = self._parse_labels(label_payload, self.expected_frame_count)
        except SDTThermalReaderError:
            raise
        except (zipfile.BadZipFile, EOFError, OSError) as exc:
            raise SourceArchiveCorruptError(f"ZIP central-directory read failed: {exc}") from exc

        class_counts = Counter(label.source_pose_label for label in labels)
        index_text = "".join(f"{index}\n" for index in sorted(expected_indices)).encode()
        self._labels = labels
        self._thermal_info = thermal
        self._inventory = {
            "archive_identity": identity,
            "member_count": len(infos),
            "file_count": sum(not info.is_dir() for info in infos),
            "directory_count": sum(info.is_dir() for info in infos),
            "thermal_member_count": len(thermal),
            "depth_member_count": len(depth),
            "label_member_count": 1,
            "label_row_count": len(labels),
            "thermal_member_pattern": r"test/image_t_(\d+)\.png",
            "depth_member_pattern": r"test/image_d_(\d+)\.png",
            "labels_member_path": LABELS_MEMBER,
            "index_base": 0,
            "index_last": self.expected_frame_count - 1,
            "index_continuous": True,
            "index_set_sha256": hashlib.sha256(index_text).hexdigest(),
            "class_counts": {str(key): class_counts[key] for key in sorted(class_counts)},
            "unexpected_members": [],
            "missing_thermal_indices": [],
            "missing_depth_indices": [],
            "duplicate_member_names": [],
            "duplicate_thermal_indices": [],
            "duplicate_depth_indices": [],
            "thermal_depth_label_linkage": "ONE_TO_ONE_BY_ZERO_BASED_INDEX",
        }
        return dict(self._inventory)

    @staticmethod
    def _decode_thermal_png(payload: bytes, member_name: str) -> tuple[np.ndarray, tuple[str, ...]]:
        if len(payload) < 33:
            raise PNGTruncatedError(f"PNG is shorter than signature and IHDR: {member_name}")
        if payload[:8] != PNG_SIGNATURE:
            raise PNGDecodeError(f"invalid PNG signature: {member_name}")
        ihdr_length = struct.unpack(">I", payload[8:12])[0]
        if ihdr_length != 13 or payload[12:16] != b"IHDR":
            raise PNGDecodeError(f"first PNG chunk is not a valid IHDR: {member_name}")
        width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
            ">IIBBBBB", payload[16:29]
        )
        if (height, width) != DISTRIBUTED_FRAME_SHAPE:
            raise FrameShapeMismatchError(
                f"{member_name} shape {(height, width)} != {DISTRIBUTED_FRAME_SHAPE}"
            )
        if bit_depth != 16:
            raise FrameDtypeMismatchError(
                f"{member_name} PNG bit depth {bit_depth} != 16"
            )
        if color_type != 0:
            raise FrameChannelMismatchError(
                f"{member_name} PNG color type {color_type} is not grayscale"
            )
        if compression != 0 or filtering != 0 or interlace not in (0, 1):
            raise UnsupportedRepresentationError(
                f"unsupported PNG coding fields for {member_name}"
            )
        try:
            with Image.open(io.BytesIO(payload)) as image:
                if image.format != "PNG":
                    raise PNGDecodeError(f"member is not decoded as PNG: {member_name}")
                image.load()
                array = np.asarray(image)
        except UnidentifiedImageError as exc:
            raise PNGDecodeError(f"Pillow cannot identify {member_name}: {exc}") from exc
        except (OSError, SyntaxError, ValueError) as exc:
            message = str(exc).lower()
            if "truncat" in message or "broken data stream" in message:
                raise PNGTruncatedError(f"truncated PNG {member_name}: {exc}") from exc
            raise PNGDecodeError(f"PNG decode failed for {member_name}: {exc}") from exc

        if array.ndim != 2:
            raise FrameChannelMismatchError(
                f"decoded frame must be single-channel, got shape {array.shape}"
            )
        if array.shape != DISTRIBUTED_FRAME_SHAPE:
            raise FrameShapeMismatchError(
                f"decoded shape {array.shape} != {DISTRIBUTED_FRAME_SHAPE}"
            )
        if array.dtype.kind != "u" or array.dtype.itemsize != 2:
            raise FrameDtypeMismatchError(f"decoded dtype {array.dtype} is not uint16")

        encoded = np.asarray(array, dtype=np.uint16, order="C").copy()
        min_value = int(encoded.min())
        max_value = int(encoded.max())
        flags: list[str] = []
        if min_value == max_value:
            flags.append("CONSTANT_FRAME")
        if np.any(encoded == 0):
            flags.append("CONTAINER_MIN_PRESENT")
        if np.any(encoded == np.iinfo(np.uint16).max):
            flags.append("CONTAINER_MAX_PRESENT")
        if min_value == max_value and min_value in (0, np.iinfo(np.uint16).max):
            raise FrameInvalidRangeError(
                f"fully constant container-extreme frame is malformed: value={min_value}"
            )
        encoded.setflags(write=False)
        return encoded, tuple(sorted(flags))

    def read_frame(self, source_frame_index: int) -> SDTThermalSourceFrame:
        self.inspect_archive()
        if isinstance(source_frame_index, bool) or not isinstance(source_frame_index, int):
            raise SourceFrameIndexInvalidError(f"index is not an integer: {source_frame_index!r}")
        if source_frame_index < 0 or source_frame_index >= self.expected_frame_count:
            raise SourceFrameIndexInvalidError(
                f"index {source_frame_index} is outside 0..{self.expected_frame_count - 1}"
            )
        if self._labels is None or source_frame_index not in self._thermal_info:
            raise FrameLabelLinkageError(f"frame {source_frame_index} lacks label or thermal member")
        label = self._labels[source_frame_index]
        member_index, cached_info = self._thermal_info[source_frame_index]
        try:
            with zipfile.ZipFile(self.archive_path, "r") as archive:
                info = archive.getinfo(cached_info.filename)
                payload = self._read_bounded_member(
                    archive, info, limit=MAX_IMAGE_MEMBER_BYTES
                )
        except KeyError as exc:
            raise SourceMemberMissingError(cached_info.filename) from exc
        except SDTThermalReaderError:
            raise
        except (zipfile.BadZipFile, EOFError, OSError) as exc:
            raise SourceArchiveCorruptError(f"ZIP read failed: {exc}") from exc

        encoded, quality_flags = self._decode_thermal_png(payload, info.filename)
        identity = self.verify_archive_identity()
        return SDTThermalSourceFrame(
            source_dataset_id=SDT_DATASET_ID,
            source_doi=SDT_DOI,
            source_split=SDT_SOURCE_SPLIT,
            source_archive_path=self.archive_rel_path,
            source_archive_size_bytes=identity["size_bytes"],
            source_archive_md5=identity["md5"],
            source_archive_sha256=identity["sha256"],
            source_member_name=info.filename,
            source_member_index=member_index,
            source_member_crc32=f"{info.CRC:08x}",
            source_member_sha256=hashlib.sha256(payload).hexdigest(),
            source_frame_index=source_frame_index,
            source_pose_label=label.source_pose_label,
            source_pose_name=label.source_pose_name,
            source_bbox=label.source_bbox,
            distributed_frame_shape=DISTRIBUTED_FRAME_SHAPE,
            native_sensor_shape=NATIVE_SENSOR_SHAPE,
            source_dtype="uint16",
            decoded_byte_order=np.dtype(np.uint16).byteorder,
            source_representation="RADIOMETRIC_TEMPERATURE_ENCODED_UINT16",
            source_temperature_encoding="kelvin_centiunits; kelvin=raw/100; celsius=(raw-27315)/100",
            source_timestamp_status="ABSENT",
            source_subject_status="ABSENT",
            source_session_status="ABSENT",
            source_sequence_status="ABSENT",
            source_event_status="ABSENT",
            quality_flags=quality_flags,
            raw_encoded_frame_sha256=encoded_frame_sha256(encoded),
            raw_encoded_frame=encoded,
        )

    def iter_frames(self, indices: Iterable[int]) -> Iterable[SDTThermalSourceFrame]:
        for index in indices:
            yield self.read_frame(index)
