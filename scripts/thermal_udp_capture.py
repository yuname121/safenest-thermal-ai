#!/usr/bin/env python3
"""Capture Thermal_Test UDP frames into the SafeNest real-capture contract v1.

The current preferred transport is framed UDP V2: every MTU-safe chunk carries
an explicit frame ID, chunk index/count, byte offset/length, logical frame
length, and whole-frame CRC32. ``--reassemble-udp-chunks`` enables that mode.
The exact 10,080-byte Thermal_Test frame remains the logical payload and is
stored unchanged after fail-closed reassembly. The historic blind byte-stream
mode remains available only as ``--legacy-stream-reassembly`` for preserving
old diagnostic evidence; it is not suitable for new model-input capture.

The script needs only Python's standard library and is suitable for Raspberry
Pi OS. It must run before the ESP32 sender begins emitting packets.
"""

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
import re
import socket
import struct
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


COLLECTION_SCHEMA = "safenest.thermal.real_capture.collection.v1"
SESSION_SCHEMA = "safenest.thermal.real_capture.session.v1"
FRAME_SCHEMA = "safenest.thermal.real_capture.frame.v1"
ANNOTATION_SCHEMA = "safenest.thermal.real_capture.annotation.v1"
CONTRACT_ID = "safenest.thermal.real_capture.v1"
COLLECTOR_VERSION = "thermal_udp_capture_v2"

HEADER_WORDS = 80
WIDTH = 80
HEIGHT = 62
PIXEL_WORDS = WIDTH * HEIGHT
FRAME_WORDS = HEADER_WORDS + PIXEL_WORDS
FRAME_BYTES = FRAME_WORDS * 2
UINT16_LE = struct.Struct("<{}H".format(FRAME_WORDS))
PIXEL_UINT16_LE = struct.Struct("<{}H".format(PIXEL_WORDS))
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# SafeNest Thermal raw UDP V2. Header fields use network byte order while the
# payload remains the unmodified 10,080-byte little-endian Thermal_Test frame.
CHUNK_MAGIC = b"SNTR"
CHUNK_PROTOCOL_VERSION = 2
CHUNK_MESSAGE_TYPE_RAW_U16_LE = 1
CHUNK_HEADER = struct.Struct("!4sBBHIHHIIHHI")
CHUNK_HEADER_BYTES = CHUNK_HEADER.size
CHUNK_DATAGRAM_BYTES = 1200
CHUNK_PAYLOAD_BYTES = CHUNK_DATAGRAM_BYTES - CHUNK_HEADER_BYTES
CHUNK_COUNT = math.ceil(FRAME_BYTES / CHUNK_PAYLOAD_BYTES)
CHUNK_MAX_COUNT = 16

SOURCE_LABELS = ("NOT_ANNOTATED", "EMPTY", "STANDING", "SITTING", "LYING", "UNKNOWN")


class ChunkProtocolError(ValueError):
    """A framed UDP V2 datagram violated the transport contract."""


@dataclass(frozen=True)
class FramedChunk:
    frame_id: int
    chunk_index: int
    chunk_count: int
    frame_size: int
    chunk_offset: int
    frame_crc32: int
    payload: bytes


@dataclass
class ChunkMetrics:
    received_datagrams: int = 0
    invalid_datagrams: int = 0
    completed_frames: int = 0
    incomplete_frames: int = 0
    duplicate_chunks: int = 0
    conflicting_duplicates: int = 0
    out_of_order_chunks: int = 0
    reconstruction_timeouts: int = 0
    pending_limit_evictions: int = 0
    checksum_failures: int = 0


@dataclass
class _PendingFrame:
    frame_id: int
    chunk_count: int
    frame_size: int
    frame_crc32: int
    started_at: float
    updated_at: float
    chunks: Dict[int, bytes] = field(default_factory=dict)


def decode_framed_chunk(datagram: bytes) -> FramedChunk:
    """Decode one SNTR V2 datagram without accepting ambiguous offsets."""

    if len(datagram) < CHUNK_HEADER_BYTES:
        raise ChunkProtocolError("datagram is shorter than the framed UDP V2 header")
    (
        magic,
        version,
        message_type,
        header_size,
        frame_id,
        chunk_index,
        chunk_count,
        frame_size,
        chunk_offset,
        chunk_length,
        reserved,
        frame_crc32,
    ) = CHUNK_HEADER.unpack_from(datagram)
    if magic != CHUNK_MAGIC:
        raise ChunkProtocolError("invalid framed UDP V2 magic")
    if version != CHUNK_PROTOCOL_VERSION:
        raise ChunkProtocolError("unsupported framed UDP protocol version")
    if message_type != CHUNK_MESSAGE_TYPE_RAW_U16_LE:
        raise ChunkProtocolError("unsupported framed UDP message type")
    if header_size != CHUNK_HEADER_BYTES or reserved != 0:
        raise ChunkProtocolError("invalid framed UDP header fields")
    if frame_size != FRAME_BYTES:
        raise ChunkProtocolError("logical Thermal frame size is not 10,080 bytes")
    expected_count = math.ceil(frame_size / CHUNK_PAYLOAD_BYTES)
    if (
        chunk_count != expected_count
        or chunk_count < 1
        or chunk_count > CHUNK_MAX_COUNT
        or chunk_index >= chunk_count
    ):
        raise ChunkProtocolError("invalid chunk count or index")
    expected_offset = chunk_index * CHUNK_PAYLOAD_BYTES
    expected_length = min(CHUNK_PAYLOAD_BYTES, frame_size - expected_offset)
    payload = datagram[header_size:]
    if chunk_offset != expected_offset or chunk_length != expected_length or len(payload) != chunk_length:
        raise ChunkProtocolError("invalid chunk offset or length")
    return FramedChunk(
        frame_id=frame_id,
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        frame_size=frame_size,
        chunk_offset=chunk_offset,
        frame_crc32=frame_crc32,
        payload=payload,
    )


def encode_framed_frame(payload: bytes, frame_id: int) -> List[bytes]:
    """Reference encoder used by loopback and loss/reordering tests."""

    if len(payload) != FRAME_BYTES:
        raise ValueError("Thermal raw frame must be exactly 10,080 bytes")
    if not 0 <= int(frame_id) <= 0xFFFFFFFF:
        raise ValueError("transport frame ID must fit uint32")
    frame_crc32 = zlib.crc32(payload) & 0xFFFFFFFF
    datagrams: List[bytes] = []
    for chunk_index in range(CHUNK_COUNT):
        offset = chunk_index * CHUNK_PAYLOAD_BYTES
        chunk = payload[offset : offset + CHUNK_PAYLOAD_BYTES]
        header = CHUNK_HEADER.pack(
            CHUNK_MAGIC,
            CHUNK_PROTOCOL_VERSION,
            CHUNK_MESSAGE_TYPE_RAW_U16_LE,
            CHUNK_HEADER_BYTES,
            int(frame_id),
            chunk_index,
            CHUNK_COUNT,
            len(payload),
            offset,
            len(chunk),
            0,
            frame_crc32,
        )
        datagrams.append(header + chunk)
    return datagrams


class FramedChunkReassembler:
    """Bounded, timeout-aware reassembler that never joins different frames."""

    def __init__(self, frame_timeout_seconds: float = 1.0, max_pending_frames: int = 8) -> None:
        if not math.isfinite(float(frame_timeout_seconds)) or frame_timeout_seconds <= 0:
            raise ValueError("frame timeout must be positive and finite")
        if isinstance(max_pending_frames, bool) or max_pending_frames < 1:
            raise ValueError("max pending frames must be positive")
        self.frame_timeout_seconds = float(frame_timeout_seconds)
        self.max_pending_frames = int(max_pending_frames)
        self.pending: Dict[Tuple[str, int, int], _PendingFrame] = {}
        self.completed: Dict[Tuple[str, int, int], float] = {}
        self.metrics = ChunkMetrics()

    def _evict_expired(self, now: float) -> int:
        expired = [key for key, item in self.pending.items() if now - item.updated_at >= self.frame_timeout_seconds]
        for key in expired:
            del self.pending[key]
        self.metrics.incomplete_frames += len(expired)
        self.metrics.reconstruction_timeouts += len(expired)
        stale_completed = [key for key, completed_at in self.completed.items() if now - completed_at >= self.frame_timeout_seconds * 4]
        for key in stale_completed:
            del self.completed[key]
        return len(expired)

    def evict_expired(self, now: Optional[float] = None) -> int:
        return self._evict_expired(time.monotonic() if now is None else float(now))

    def accept(
        self,
        datagram: bytes,
        peer: Tuple[str, int],
        received_monotonic: Optional[float] = None,
    ) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        now = time.monotonic() if received_monotonic is None else float(received_monotonic)
        self.metrics.received_datagrams += 1
        self._evict_expired(now)
        try:
            chunk = decode_framed_chunk(datagram)
        except ChunkProtocolError:
            self.metrics.invalid_datagrams += 1
            return None

        key = (peer[0], peer[1], chunk.frame_id)
        if key in self.completed:
            self.metrics.duplicate_chunks += 1
            return None
        pending = self.pending.get(key)
        if pending is None:
            if len(self.pending) >= self.max_pending_frames:
                oldest_key = min(self.pending, key=lambda item: self.pending[item].updated_at)
                del self.pending[oldest_key]
                self.metrics.incomplete_frames += 1
                self.metrics.pending_limit_evictions += 1
            pending = _PendingFrame(
                frame_id=chunk.frame_id,
                chunk_count=chunk.chunk_count,
                frame_size=chunk.frame_size,
                frame_crc32=chunk.frame_crc32,
                started_at=now,
                updated_at=now,
            )
            self.pending[key] = pending
        elif (
            pending.chunk_count != chunk.chunk_count
            or pending.frame_size != chunk.frame_size
            or pending.frame_crc32 != chunk.frame_crc32
        ):
            del self.pending[key]
            self.metrics.invalid_datagrams += 1
            self.metrics.incomplete_frames += 1
            return None

        existing = pending.chunks.get(chunk.chunk_index)
        if existing is not None:
            if existing == chunk.payload:
                self.metrics.duplicate_chunks += 1
            else:
                del self.pending[key]
                self.metrics.conflicting_duplicates += 1
                self.metrics.invalid_datagrams += 1
                self.metrics.incomplete_frames += 1
            return None
        if chunk.chunk_index != len(pending.chunks):
            self.metrics.out_of_order_chunks += 1
        pending.chunks[chunk.chunk_index] = chunk.payload
        pending.updated_at = now
        if len(pending.chunks) != pending.chunk_count:
            return None

        del self.pending[key]
        payload = b"".join(pending.chunks[index] for index in range(pending.chunk_count))
        if len(payload) != pending.frame_size:
            self.metrics.invalid_datagrams += 1
            self.metrics.incomplete_frames += 1
            return None
        if zlib.crc32(payload) & 0xFFFFFFFF != pending.frame_crc32:
            self.metrics.checksum_failures += 1
            self.metrics.incomplete_frames += 1
            return None
        self.metrics.completed_frames += 1
        self.completed[key] = now
        return payload, {
            "protocol": "SAFENEST_THERMAL_RAW_UDP_V2",
            "transport_frame_id": pending.frame_id,
            "chunk_count": pending.chunk_count,
            "frame_crc32": "{:08x}".format(pending.frame_crc32),
            "reassembly_seconds": max(0.0, now - pending.started_at),
        }

    def finalize(self) -> int:
        incomplete = len(self.pending)
        self.pending.clear()
        self.metrics.incomplete_frames += incomplete
        return incomplete

    def snapshot(self) -> Dict[str, Any]:
        result = asdict(self.metrics)
        result.update(
            {
                "pending_frames": len(self.pending),
                "frame_timeout_seconds": self.frame_timeout_seconds,
                "max_pending_frames": self.max_pending_frames,
                "chunk_payload_bytes": CHUNK_PAYLOAD_BYTES,
                "datagram_bytes_max": CHUNK_DATAGRAM_BYTES,
                "expected_chunks_per_frame": CHUNK_COUNT,
            }
        )
        return result


def utc_offset_now() -> str:
    offset = datetime.now().astimezone().strftime("%z")
    return "{}:{}".format(offset[:3], offset[3:]) if offset else "UNKNOWN"


def wall_time_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_identifier(label: str, value: str) -> str:
    if not ID_PATTERN.fullmatch(value):
        raise ValueError("{} must match {}: {!r}".format(label, ID_PATTERN.pattern, value))
    return value


def append_jsonl(handle: Any, value: Dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def derived_annotation(source_label: str) -> Dict[str, str]:
    mapping = {
        "EMPTY": ("NO_ANNOTATED_HUMAN_IN_REPRESENTED_FRAME", "DERIVED_POSTURE_PROXY", "DERIVED_ONLY"),
        "STANDING": ("HUMAN_STANDING_POSTURE", "DERIVED_POSTURE_PROXY", "DERIVED_ONLY"),
        "SITTING": ("HUMAN_SITTING_POSTURE", "DERIVED_POSTURE_PROXY", "DERIVED_ONLY"),
        "LYING": ("HUMAN_LYING_POSTURE", "DERIVED_POSTURE_PROXY", "DERIVED_ONLY"),
        "UNKNOWN": ("UNKNOWN", "NOT_ASSIGNED", "UNKNOWN"),
        "NOT_ANNOTATED": ("NOT_ASSIGNED", "NOT_ASSIGNED", "NOT_ASSIGNED"),
    }
    label, mapping_type, status = mapping[source_label]
    return {"label": label, "mapping_type": mapping_type, "status": status}


class CaptureWriter:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.collection_dir = Path(args.output).expanduser().resolve() / args.collection_id
        self.subject_dir = self.collection_dir / "subjects" / args.subject_id
        self.session_dir = self.subject_dir / "sessions" / args.session_id
        self.raw_dir = self.session_dir / "raw"
        self.raw_chunks_dir = self.session_dir / "raw_chunks"
        self.decoded_dir = self.session_dir / "decoded_native"
        self.frames_path = self.session_dir / "frames.jsonl"
        self.annotations_path = self.session_dir / "annotations.jsonl"
        self.session_path = self.session_dir / "session.json"
        self.collection_path = self.collection_dir / "collection.json"
        self.checksums_path = self.session_dir / "checksums.sha256"

        self.started_at = wall_time_now()
        self.frame_index = 0
        self.manifest_frame_count = 0
        self.valid_frame_count = 0
        self.invalid_frame_count = 0
        self.duplicate_counter_count = 0
        self.packet_loss_count = 0
        self.decode_failure_count = 0
        self.last_counter: Optional[int] = None
        self.transport_metrics: Dict[str, Any] = {}
        self.frames_handle: Any = None
        self.annotations_handle: Any = None

    def _chunk_capture_enabled(self) -> bool:
        return bool(self.args.reassemble_udp_chunks or self.args.legacy_stream_reassembly)

    def prepare(self) -> None:
        if self.session_dir.exists():
            raise FileExistsError("Refusing to overwrite existing session directory: {}".format(self.session_dir))

        if self.collection_dir.exists() and not self.collection_path.is_file():
            raise FileExistsError(
                "Collection directory exists without collection.json; choose a new collection ID: {}".format(self.collection_dir)
            )

        self.raw_dir.mkdir(parents=True, exist_ok=False)
        if self._chunk_capture_enabled():
            self.raw_chunks_dir.mkdir(parents=True, exist_ok=False)
        self.decoded_dir.mkdir(parents=True, exist_ok=False)
        self.frames_handle = self.frames_path.open("x", encoding="utf-8")
        self.annotations_handle = self.annotations_path.open("x", encoding="utf-8")
        self._upsert_collection()

    def _upsert_collection(self) -> None:
        if self.collection_path.is_file():
            collection = json.loads(self.collection_path.read_text(encoding="utf-8"))
            if collection.get("collection_id") != self.args.collection_id:
                raise ValueError("Existing collection.json has a different collection_id")
            if collection.get("collection_role") != self.args.collection_role:
                raise ValueError("Existing collection.json has a different collection_role")
        else:
            collection = {
                "schema_version": COLLECTION_SCHEMA,
                "contract_id": CONTRACT_ID,
                "collection_id": self.args.collection_id,
                "collection_role": self.args.collection_role,
                "created_at": self.started_at,
                "subject_ids": [],
                "session_ids": [],
                "full_frame_collection_status": "FULL_FRAME_AVAILABLE",
                "split_policy": {
                    "assignment_unit": "NOT_ASSIGNED",
                    "assignment_method": "NO_MODEL_SPLIT_DEVICE_CONTRACT_PILOT",
                    "frame_random_split_allowed": False,
                    "locked_test_access": "NOT_APPLICABLE",
                    "frozen_before_model_inspection": None,
                },
                "source": {
                    "source_status": "TEAM_DEVICE_CAPTURE",
                    "source_description": "Thermal-90 raw UDP capture from XIAO ESP32-C6 to Raspberry Pi.",
                    "team_repository_base_commit": self.args.team_repository_commit,
                    "collector_git_commit": self.args.collector_commit,
                },
                "privacy": {
                    "pseudonymous_ids_only": True,
                    "names_in_filenames": False,
                    "unnecessary_personal_metadata_present": False,
                },
                "notes": self.args.collection_notes,
            }

        if self.args.subject_id not in collection["subject_ids"]:
            collection["subject_ids"].append(self.args.subject_id)
        if self.args.session_id not in collection["session_ids"]:
            collection["session_ids"].append(self.args.session_id)
        collection["subject_ids"].sort()
        collection["session_ids"].sort()
        write_json(self.collection_path, collection)

    def _frame_id(self) -> str:
        return "frame_{}_{}".format(self.args.session_id, str(self.frame_index).zfill(6))

    def _next_index(self) -> Tuple[str, int]:
        frame_id = self._frame_id()
        index = self.frame_index
        self.frame_index += 1
        self.manifest_frame_count += 1
        return frame_id, index

    def _write_annotation(self, frame_id: str) -> None:
        source_label = self.args.source_label
        confidence: Optional[float] = 1.0 if source_label in {"EMPTY", "STANDING", "SITTING", "LYING"} else None
        method = "OPERATOR_DECLARED_STATIC_SESSION" if confidence is not None else "NOT_ANNOTATED_AT_CAPTURE"
        annotation = {
            "schema_version": ANNOTATION_SCHEMA,
            "annotation_id": "annotation_{}".format(frame_id[len("frame_"):]),
            "collection_id": self.args.collection_id,
            "subject_id": self.args.subject_id,
            "session_id": self.args.session_id,
            "annotation_scope": "FRAME",
            "event_id": None,
            "frame_id": frame_id,
            "source_annotation": {
                "label": source_label,
                "ground_truth_method": method,
                "confidence": confidence,
                "notes": "Capture-time session label only; review before any model use.",
            },
            "derived_safenest_annotation": derived_annotation(source_label),
            "provenance": {
                "annotator_code": self.args.operator_code,
                "annotation_time": wall_time_now(),
                "revision": 1,
                "notes": "Automatically recorded at capture; not a training authorization.",
            },
            "revision": 1,
            "notes": "No fall-event claim is made by this capture annotation.",
        }
        append_jsonl(self.annotations_handle, annotation)

    def _write_missing_marker(self, missing_counter: int) -> None:
        frame_id, index = self._next_index()
        frame = {
            "schema_version": FRAME_SCHEMA,
            "frame_id": frame_id,
            "collection_id": self.args.collection_id,
            "subject_id": self.args.subject_id,
            "session_id": self.args.session_id,
            "recording_id": self.args.recording_id,
            "sequence_id": self.args.sequence_id,
            "event_id": None,
            "sequence_index": index,
            "sequence_index_status": "RECEIVED_ORDER_ONLY",
            "sensor_frame_counter": missing_counter,
            "sensor_frame_counter_status": "UNKNOWN",
            "sensor_timestamp": None,
            "sensor_timestamp_unit": None,
            "sensor_clock_domain": None,
            "sensor_timestamp_status": "UNKNOWN",
            "device_monotonic_timestamp_ns": None,
            "host_receive_monotonic_timestamp_ns": None,
            "host_wall_time": None,
            "host_wall_timezone": utc_offset_now(),
            "raw_file": None,
            "decoded_native_file": None,
            "raw_representation": "UNKNOWN",
            "byte_count": None,
            "native_shape": None,
            "native_dtype": None,
            "raw_encoding": "UNKNOWN",
            "raw_unit_claim": "UNKNOWN_NOT_VERIFIED",
            "unit_status": "UNKNOWN",
            "crc_or_packet_status": "NO_DATAGRAM_RECEIVED",
            "packet_loss_status": "HEADER_COUNTER_GAP",
            "validity_status": "MISSING",
            "exclude_reason": "No UDP datagram arrived for this header-counter value.",
            "annotation_status": "NOT_APPLICABLE",
            "raw_sha256": None,
            "decoded_native_sha256": None,
            "scalar_thermal_max_c": None,
            "capture_error_code": "HEADER_COUNTER_GAP",
            "notes": "Missing-frame marker generated from the next received header counter.",
        }
        append_jsonl(self.frames_handle, frame)
        self.invalid_frame_count += 1

    def _packet_loss_status(self, frame_counter: int) -> str:
        if self.last_counter is None:
            self.last_counter = frame_counter
            return "FIRST_OBSERVED_HEADER_COUNTER"

        delta = (frame_counter - self.last_counter) & 0xFFFF
        if delta == 0:
            self.duplicate_counter_count += 1
            return "DUPLICATE_HEADER_COUNTER"
        if delta == 1:
            self.last_counter = frame_counter
            return "NO_OBSERVED_HEADER_COUNTER_GAP"

        # A small forward delta represents recoverable packet loss. Preserve a
        # manifest marker for every absent counter without inventing raw data.
        if 1 < delta <= self.args.max_gap_markers + 1:
            for offset in range(1, delta):
                self._write_missing_marker((self.last_counter + offset) & 0xFFFF)
            self.packet_loss_count += delta - 1
            self.last_counter = frame_counter
            return "HEADER_COUNTER_GAP_{}".format(delta - 1)

        # A very large delta may be counter reset, corruption, or a long outage.
        # Do not turn an ambiguous reset into a fabricated loss count.
        self.last_counter = frame_counter
        return "HEADER_COUNTER_LARGE_OR_RESET_DELTA_{}".format(delta)

    def record_invalid_datagram(self, data: bytes, reason: str) -> None:
        frame_id, index = self._next_index()
        raw_name = "raw/{}_unexpected_{}B.bin".format(frame_id, len(data))
        raw_path = self.session_dir / raw_name
        raw_path.write_bytes(data)
        frame = {
            "schema_version": FRAME_SCHEMA,
            "frame_id": frame_id,
            "collection_id": self.args.collection_id,
            "subject_id": self.args.subject_id,
            "session_id": self.args.session_id,
            "recording_id": self.args.recording_id,
            "sequence_id": self.args.sequence_id,
            "event_id": None,
            "sequence_index": index,
            "sequence_index_status": "RECEIVED_ORDER_ONLY",
            "sensor_frame_counter": None,
            "sensor_frame_counter_status": "UNKNOWN",
            "sensor_timestamp": None,
            "sensor_timestamp_unit": None,
            "sensor_clock_domain": None,
            "sensor_timestamp_status": "UNKNOWN",
            "device_monotonic_timestamp_ns": None,
            "host_receive_monotonic_timestamp_ns": time.monotonic_ns(),
            "host_wall_time": wall_time_now(),
            "host_wall_timezone": utc_offset_now(),
            "raw_file": raw_name,
            "decoded_native_file": None,
            "raw_representation": "RAW_PACKET_ONLY",
            "byte_count": len(data),
            "native_shape": None,
            "native_dtype": None,
            "raw_encoding": "UNKNOWN_UNEXPECTED_UDP_DATAGRAM",
            "raw_unit_claim": "UNKNOWN_NOT_VERIFIED",
            "unit_status": "UNKNOWN",
            "crc_or_packet_status": "UDP_DATAGRAM_LENGTH_INVALID",
            "packet_loss_status": "UNKNOWN",
            "validity_status": "PARTIAL" if len(data) < FRAME_BYTES else "CORRUPT",
            "exclude_reason": reason,
            "annotation_status": "NOT_APPLICABLE",
            "raw_sha256": sha256_file(raw_path),
            "decoded_native_sha256": None,
            "scalar_thermal_max_c": None,
            "capture_error_code": "UNEXPECTED_UDP_DATAGRAM_SIZE",
            "notes": "Unmodified unexpected UDP datagram retained for transport diagnosis.",
        }
        append_jsonl(self.frames_handle, frame)
        self.invalid_frame_count += 1

    def record_udp_chunk(self, data: bytes, chunk_index: int) -> None:
        """Preserve each UDP datagram used by chunked reassembly."""
        chunk_name = "chunk_{:08d}_{}B.bin".format(chunk_index, len(data))
        (self.raw_chunks_dir / chunk_name).write_bytes(data)

    def record_valid_datagram(
        self,
        data: bytes,
        transport_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        received_monotonic_ns = time.monotonic_ns()
        received_wall_time = wall_time_now()
        try:
            words = UINT16_LE.unpack(data)
        except struct.error as error:
            self.decode_failure_count += 1
            self.record_invalid_datagram(data, "uint16 decode failed: {}".format(error))
            return

        frame_counter = words[0]
        packet_loss_status = self._packet_loss_status(frame_counter)
        frame_id, index = self._next_index()
        raw_name = "raw/{}.udp.bin".format(frame_id)
        decoded_name = "decoded_native/{}_pixels_u16le.bin".format(frame_id)
        raw_path = self.session_dir / raw_name
        decoded_path = self.session_dir / decoded_name
        raw_path.write_bytes(data)
        decoded_path.write_bytes(PIXEL_UINT16_LE.pack(*words[HEADER_WORDS:]))

        duplicate = packet_loss_status == "DUPLICATE_HEADER_COUNTER"
        frame = {
            "schema_version": FRAME_SCHEMA,
            "frame_id": frame_id,
            "collection_id": self.args.collection_id,
            "subject_id": self.args.subject_id,
            "session_id": self.args.session_id,
            "recording_id": self.args.recording_id,
            "sequence_id": self.args.sequence_id,
            "event_id": None,
            "sequence_index": index,
            "sequence_index_status": "RECEIVED_ORDER_ONLY",
            "sensor_frame_counter": frame_counter,
            "sensor_frame_counter_status": "RECEIVED_ONLY",
            "sensor_timestamp": None,
            "sensor_timestamp_unit": None,
            "sensor_clock_domain": None,
            "sensor_timestamp_status": "UNKNOWN",
            "device_monotonic_timestamp_ns": None,
            "host_receive_monotonic_timestamp_ns": received_monotonic_ns,
            "host_wall_time": received_wall_time,
            "host_wall_timezone": utc_offset_now(),
            "raw_file": raw_name,
            "decoded_native_file": decoded_name,
            "raw_representation": "RAW_PACKET_AND_NATIVE",
            "byte_count": len(data),
            "native_shape": [HEIGHT, WIDTH],
            "native_dtype": "uint16",
            "raw_encoding": (
                "LITTLE_ENDIAN_UINT16_WORDS_5040_REASSEMBLED_FROM_FRAMED_UDP_V2"
                if self.args.reassemble_udp_chunks
                else "LITTLE_ENDIAN_UINT16_WORDS_5040_REASSEMBLED_FROM_UNSAFE_LEGACY_STREAM"
                if self.args.legacy_stream_reassembly
                else "LITTLE_ENDIAN_UINT16_WORDS_5040"
            ),
            "raw_unit_claim": "UNKNOWN_NOT_VERIFIED",
            "unit_status": "NOT_VERIFIED",
            "crc_or_packet_status": (
                "FRAME_CRC32_VERIFIED"
                if transport_metadata is not None
                else "UDP_DATAGRAM_LENGTH_OK_NO_CRC_IN_PROTOCOL"
            ),
            "packet_loss_status": packet_loss_status,
            "validity_status": "DUPLICATE" if duplicate else "VALID",
            "exclude_reason": "Duplicate header counter retained as evidence." if duplicate else None,
            "annotation_status": "ANNOTATED",
            "raw_sha256": sha256_file(raw_path),
            "decoded_native_sha256": sha256_file(decoded_path),
            "scalar_thermal_max_c": None,
            "capture_error_code": None,
            "transport_protocol": transport_metadata.get("protocol") if transport_metadata else "THERMAL_TEST_UDP_RAW_V1",
            "transport_frame_id": transport_metadata.get("transport_frame_id") if transport_metadata else None,
            "transport_chunk_count": transport_metadata.get("chunk_count") if transport_metadata else None,
            "transport_frame_crc32": transport_metadata.get("frame_crc32") if transport_metadata else None,
            "transport_reassembly_seconds": transport_metadata.get("reassembly_seconds") if transport_metadata else None,
            "notes": "Header words are preserved only in raw UDP datagram; decoded file contains untransformed pixel words 80..5039.",
        }
        append_jsonl(self.frames_handle, frame)
        self._write_annotation(frame_id)
        if duplicate:
            self.invalid_frame_count += 1
        else:
            self.valid_frame_count += 1

    def _session_manifest(self) -> Dict[str, Any]:
        if self.args.reassemble_udp_chunks:
            protocol = "SAFENEST_THERMAL_RAW_UDP_V2"
            protocol_version = "2"
            datagram_mode = "FRAME_ID_CHUNKED_REASSEMBLY"
            crc_status = "WHOLE_FRAME_CRC32_REQUIRED"
            reassembly_safety = "FRAME_BOUNDARY_EXPLICIT_FAIL_CLOSED"
        elif self.args.legacy_stream_reassembly:
            protocol = "THERMAL_TEST_UDP_RAW_V1"
            protocol_version = "1"
            datagram_mode = "UNSAFE_LEGACY_STREAM_REASSEMBLY"
            crc_status = "NO_CRC_IN_PROTOCOL"
            reassembly_safety = "NOT_SAFE_FOR_NEW_MODEL_INPUT_CAPTURE"
        else:
            protocol = "THERMAL_TEST_UDP_RAW_V1"
            protocol_version = "1"
            datagram_mode = "SINGLE_FRAME_DATAGRAM"
            crc_status = "NO_CRC_IN_PROTOCOL"
            reassembly_safety = "EXACT_DATAGRAM_LENGTH_ONLY"
        return {
            "schema_version": SESSION_SCHEMA,
            "collection_id": self.args.collection_id,
            "subject_id": self.args.subject_id,
            "session_id": self.args.session_id,
            "recording_id": self.args.recording_id,
            "sequence_id": self.args.sequence_id,
            "capture_start_time": self.started_at,
            "capture_end_time": wall_time_now(),
            "timezone": utc_offset_now(),
            "operator_code": self.args.operator_code,
            "sensor": {
                "sensor_vendor": self.args.sensor_vendor,
                "sensor_model": self.args.sensor_model,
                "sensor_hardware_revision": self.args.sensor_hardware_revision,
                "sensor_serial_or_pseudonymous_device_id": self.args.sensor_device_id,
                "firmware_version": self.args.firmware_version,
                "collector_software_version": COLLECTOR_VERSION,
                "collector_git_commit": self.args.collector_commit,
                "native_width": WIDTH,
                "native_height": HEIGHT,
                "native_dtype": "uint16",
                "raw_encoding": "LITTLE_ENDIAN_UINT16_WORDS_5040",
                "raw_unit_claim": "UNKNOWN_NOT_VERIFIED",
                "unit_verification_status": "NOT_VERIFIED",
                "configured_fps": self.args.configured_fps,
                "verified_fps_status": "CONFIGURED_ONLY",
                "orientation": "UNKNOWN_NOT_VERIFIED",
                "mount_height_m": self.args.mount_height_m,
                "mount_angle_deg": self.args.mount_angle_deg,
                "sensor_to_subject_distance_m": self.args.sensor_distance_m,
            },
            "transport": {
                "transport_path": "XIAO_ESP32C6_TO_RASPBERRY_PI_UDP",
                "protocol": protocol,
                "protocol_version": protocol_version,
                "udp_datagram_mode": datagram_mode,
                "full_frame_status": "PRESERVED",
                "scalar_thermal_max_status": "PRESENT",
                "raw_packet_bytes_preserved": True,
                "raw_udp_chunks_preserved": self._chunk_capture_enabled(),
                "chunk_header_bytes": CHUNK_HEADER_BYTES if self.args.reassemble_udp_chunks else None,
                "chunk_payload_bytes": CHUNK_PAYLOAD_BYTES if self.args.reassemble_udp_chunks else None,
                "expected_chunks_per_frame": CHUNK_COUNT if self.args.reassemble_udp_chunks else None,
                "frame_crc_status": crc_status,
                "reassembly_safety": reassembly_safety,
                "transport_latency_status": "NOT_MEASURED",
            },
            "timing": {
                "continuous_session": True,
                "sensor_timestamp_status": "UNKNOWN_NO_WIRE_FIELD",
                "sensor_sequence_counter_status": "PRESENT_HEADER_WORD_0_UNVERIFIED_PHYSICAL_SEMANTICS",
                "device_monotonic_status": "UNKNOWN_NO_WIRE_FIELD",
                "host_receive_monotonic_status": "PRESENT_PYTHON_TIME_MONOTONIC_NS",
                "host_wall_clock_status": "PRESENT_LOCAL_TIMEZONE",
                "timestamp_source_policy": "Use UDP header word 0 and Raspberry Pi receive monotonic clock; never infer time from filenames.",
            },
            "environment": {
                "room_code": self.args.room_code,
                "ambient_temperature_c": None,
                "ambient_temperature_status": "UNKNOWN",
                "lighting_or_heat_sources": self.args.heat_source_notes,
                "background_variation": self.args.background_variation,
                "clothing_or_ppe": self.args.clothing_or_ppe,
                "occlusion_condition": self.args.occlusion_condition,
                "notes": self.args.session_notes,
            },
            "storage": {
                "frames_file": "frames.jsonl",
                "annotations_file": "annotations.jsonl",
                "checksums_file": "checksums.sha256",
                "raw_root": "raw",
                "raw_chunks_root": "raw_chunks" if self._chunk_capture_enabled() else None,
                "decoded_native_root": "decoded_native",
                "model_input_root": None,
            },
            "quality": {
                "expected_frame_count": None,
                "received_frame_count": self.manifest_frame_count,
                "sequence_gap_count": self.packet_loss_count,
                "duplicate_counter_count": self.duplicate_counter_count,
                "decode_failure_count": self.decode_failure_count,
                "packet_loss_count": self.packet_loss_count,
                "transport_metrics": self.transport_metrics,
            },
            "role_governance": {
                "role": self.args.collection_role,
                "model_access_status": "DEVELOPMENT_ALLOWED",
                "locked_test_status": "NOT_LOCKED_TEST",
                "split_frozen_at": None,
            },
            "temporal_evidence_claim": {
                "claimed_status": "TEMPORAL_ORDER_ONLY",
                "source": "HEADER_COUNTER_AND_HOST_RECEIVE_MONOTONIC_ONLY",
                "filename_order_used_as_time": False,
            },
            "safety": {
                "fall_like_capture_authorized": False,
                "safety_control_status": "NO_FREE_FALL_CAPTURE",
                "uncontrolled_free_fall_experiment": False,
            },
            "notes": "DEVICE_CONTRACT_PILOT only. Validator success does not authorize T-C, T-D, training, or locked-test use.",
        }

    def _write_checksums(self) -> None:
        paths: List[Path] = [self.session_path, self.frames_path, self.annotations_path]
        paths.extend(sorted(path for path in self.raw_dir.rglob("*") if path.is_file()))
        if self._chunk_capture_enabled():
            paths.extend(sorted(path for path in self.raw_chunks_dir.rglob("*") if path.is_file()))
        paths.extend(sorted(path for path in self.decoded_dir.rglob("*") if path.is_file()))
        paths.sort(key=lambda path: path.relative_to(self.session_dir).as_posix())
        with self.checksums_path.open("w", encoding="utf-8", newline="\n") as handle:
            for path in paths:
                relative = path.relative_to(self.session_dir).as_posix()
                handle.write("{}  {}\n".format(sha256_file(path), relative))

    def close(self) -> None:
        if self.frames_handle is not None:
            self.frames_handle.close()
            self.frames_handle = None
        if self.annotations_handle is not None:
            self.annotations_handle.close()
            self.annotations_handle = None
        write_json(self.session_path, self._session_manifest())
        self._write_checksums()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture Thermal_Test UDP frames into the SafeNest real-capture contract."
    )
    parser.add_argument("--output", required=True, help="Parent directory for collection folders; keep outside Git.")
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--subject-id", required=True, help="Pseudonymous ID only, e.g. S001.")
    parser.add_argument("--session-id", required=True, help="New ID for every restart or scene/install change.")
    parser.add_argument("--recording-id", default=None)
    parser.add_argument("--sequence-id", default=None)
    parser.add_argument("--operator-code", required=True, help="Pseudonymous operator code; no personal names.")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    reassembly = parser.add_mutually_exclusive_group()
    reassembly.add_argument(
        "--reassemble-udp-chunks",
        action="store_true",
        help="Reassemble framed SNTR UDP V2 chunks by frame ID/index and verify whole-frame CRC32.",
    )
    reassembly.add_argument(
        "--legacy-stream-reassembly",
        action="store_true",
        help="Diagnostic compatibility only: blindly concatenate historic UDP chunks; unsafe for new model-input capture.",
    )
    parser.add_argument("--chunk-timeout-seconds", type=float, default=1.0)
    parser.add_argument("--max-pending-frames", type=int, default=8)
    parser.add_argument("--duration-seconds", type=float, default=120.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-gap-markers", type=int, default=300)
    parser.add_argument("--collection-role", choices=("DEVICE_CONTRACT_PILOT",), default="DEVICE_CONTRACT_PILOT")
    parser.add_argument("--source-label", choices=SOURCE_LABELS, default="NOT_ANNOTATED")
    parser.add_argument("--configured-fps", type=float, default=7.0)
    parser.add_argument("--sensor-vendor", default="UNKNOWN")
    parser.add_argument("--sensor-model", default="Thermal-90")
    parser.add_argument("--sensor-hardware-revision", default="UNKNOWN")
    parser.add_argument("--sensor-device-id", default="DEVICE_UNKNOWN")
    parser.add_argument("--firmware-version", default="UNKNOWN")
    parser.add_argument("--collector-commit", default=None)
    parser.add_argument("--team-repository-commit", default=None)
    parser.add_argument("--room-code", default=None)
    parser.add_argument("--mount-height-m", type=float, default=None)
    parser.add_argument("--mount-angle-deg", type=float, default=None)
    parser.add_argument("--sensor-distance-m", type=float, default=None)
    parser.add_argument("--occlusion-condition", default="UNKNOWN")
    parser.add_argument("--background-variation", default="UNKNOWN")
    parser.add_argument("--heat-source-notes", default=None)
    parser.add_argument("--clothing-or-ppe", default=None)
    parser.add_argument("--collection-notes", default="")
    parser.add_argument("--session-notes", default="")
    args = parser.parse_args(argv)

    for label, value in (
        ("collection_id", args.collection_id),
        ("subject_id", args.subject_id),
        ("session_id", args.session_id),
    ):
        validate_identifier(label, value)
    if args.recording_id is None:
        args.recording_id = "recording_{}".format(args.session_id)
    if args.sequence_id is None:
        args.sequence_id = "sequence_{}".format(args.session_id)
    validate_identifier("recording_id", args.recording_id)
    validate_identifier("sequence_id", args.sequence_id)
    if args.port < 1 or args.port > 65535:
        parser.error("--port must be 1..65535")
    if args.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.max_gap_markers < 0:
        parser.error("--max-gap-markers must be zero or greater")
    if args.configured_fps <= 0:
        parser.error("--configured-fps must be positive")
    if not math.isfinite(args.chunk_timeout_seconds) or args.chunk_timeout_seconds <= 0:
        parser.error("--chunk-timeout-seconds must be positive and finite")
    if args.max_pending_frames <= 0:
        parser.error("--max-pending-frames must be positive")
    return args


def run_capture(args: argparse.Namespace) -> int:
    writer = CaptureWriter(args)
    writer.prepare()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.bind((args.bind, args.port))
    sock.settimeout(0.5)

    if args.reassemble_udp_chunks:
        mode = "framed SNTR UDP V2 reassembly"
    elif args.legacy_stream_reassembly:
        mode = "UNSAFE legacy sequential UDP stream reassembly"
    else:
        mode = "exact UDP datagrams"
    print("[capture] listening on {}:{} for {} (frame bytes={})".format(args.bind, args.port, mode, FRAME_BYTES))
    print("[capture] output: {}".format(writer.collection_dir))
    deadline = time.monotonic() + args.duration_seconds
    packet_count = 0
    chunk_buffer = bytearray()
    chunk_index = 0
    reassembler = (
        FramedChunkReassembler(
            frame_timeout_seconds=args.chunk_timeout_seconds,
            max_pending_frames=args.max_pending_frames,
        )
        if args.reassemble_udp_chunks
        else None
    )
    try:
        while time.monotonic() < deadline:
            if args.max_frames is not None and writer.valid_frame_count >= args.max_frames:
                break
            try:
                # Preserve an entire unexpected datagram for diagnosis instead
                # of truncating it to the expected frame length.
                data, address = sock.recvfrom(65535)
            except socket.timeout:
                if reassembler is not None:
                    reassembler.evict_expired()
                continue
            packet_count += 1
            if args.reassemble_udp_chunks:
                writer.record_udp_chunk(data, chunk_index)
                chunk_index += 1
                assert reassembler is not None
                completed = reassembler.accept(data, address)
                if completed is not None:
                    frame_data, transport_metadata = completed
                    writer.record_valid_datagram(frame_data, transport_metadata=transport_metadata)
            elif args.legacy_stream_reassembly:
                writer.record_udp_chunk(data, chunk_index)
                chunk_index += 1
                chunk_buffer.extend(data)
                while len(chunk_buffer) >= FRAME_BYTES:
                    frame_data = bytes(chunk_buffer[:FRAME_BYTES])
                    del chunk_buffer[:FRAME_BYTES]
                    writer.record_valid_datagram(frame_data)
                    if args.max_frames is not None and writer.valid_frame_count >= args.max_frames:
                        break
            else:
                if len(data) == FRAME_BYTES:
                    writer.record_valid_datagram(data)
                else:
                    writer.record_invalid_datagram(
                        data,
                        "Expected {} bytes from THERMAL_TEST_UDP_RAW_V1, received {} bytes.".format(FRAME_BYTES, len(data)),
                    )
            if packet_count % 30 == 0:
                print("[capture] datagrams={} valid={} invalid={} packet_loss={}".format(
                    packet_count, writer.valid_frame_count, writer.invalid_frame_count, writer.packet_loss_count
                ))
    except KeyboardInterrupt:
        print("\n[capture] interrupted by operator; finalizing captured evidence.")
    finally:
        if reassembler is not None:
            reassembler.finalize()
            writer.transport_metrics = reassembler.snapshot()
        elif args.legacy_stream_reassembly and chunk_buffer:
            writer.record_invalid_datagram(
                bytes(chunk_buffer),
                "Capture ended with an incomplete reassembled frame of {} bytes.".format(len(chunk_buffer)),
            )
            writer.transport_metrics = {
                "legacy_stream_partial_bytes": len(chunk_buffer),
                "integrity_status": "UNSAFE_STREAM_BOUNDARY_NOT_VERIFIABLE",
            }
        sock.close()
        writer.close()

    print("[capture] complete: manifest_records={} valid={} invalid={} output={}".format(
        writer.manifest_frame_count, writer.valid_frame_count, writer.invalid_frame_count, writer.collection_dir
    ))
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    try:
        return run_capture(parse_args(argv))
    except (ValueError, FileExistsError, OSError) as error:
        print("[capture] ERROR: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
