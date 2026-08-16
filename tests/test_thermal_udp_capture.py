"""Regression tests for fail-closed Thermal raw UDP V2 reassembly."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.thermal_udp_capture import (  # noqa: E402
    CHUNK_COUNT,
    CHUNK_DATAGRAM_BYTES,
    CHUNK_HEADER_BYTES,
    CHUNK_PAYLOAD_BYTES,
    FRAME_BYTES,
    ChunkProtocolError,
    CaptureWriter,
    FramedChunkReassembler,
    decode_framed_chunk,
    encode_framed_frame,
    parse_args,
)
from scripts.validate_thermal_real_capture import validate_capture  # noqa: E402


PEER = ("192.0.2.10", 40000)


def _frame(seed: int) -> bytes:
    return bytes(((index * 17) + seed) & 0xFF for index in range(FRAME_BYTES))


def test_v2_contract_stays_mtu_safe_and_preserves_full_raw_frame() -> None:
    payload = _frame(3)
    datagrams = encode_framed_frame(payload, frame_id=42)

    assert CHUNK_HEADER_BYTES == 32
    assert CHUNK_PAYLOAD_BYTES == 1168
    assert CHUNK_COUNT == 9
    assert len(datagrams) == CHUNK_COUNT
    assert all(len(item) <= CHUNK_DATAGRAM_BYTES for item in datagrams)

    reassembler = FramedChunkReassembler()
    completed = None
    for index, datagram in enumerate(datagrams):
        completed = reassembler.accept(datagram, PEER, received_monotonic=index * 0.01)
    assert completed is not None
    rebuilt, metadata = completed
    assert rebuilt == payload
    assert metadata["transport_frame_id"] == 42
    assert metadata["chunk_count"] == 9
    assert reassembler.metrics.completed_frames == 1
    assert reassembler.metrics.incomplete_frames == 0


def test_out_of_order_chunks_are_reassembled_by_explicit_index() -> None:
    payload = _frame(7)
    datagrams = encode_framed_frame(payload, frame_id=100)
    order = [3, 0, 8, 2, 1, 7, 6, 5, 4]
    reassembler = FramedChunkReassembler()

    completed = None
    for tick, index in enumerate(order):
        result = reassembler.accept(datagrams[index], PEER, received_monotonic=tick * 0.01)
        if result is not None:
            completed = result

    assert completed is not None
    assert completed[0] == payload
    assert reassembler.metrics.out_of_order_chunks > 0
    assert reassembler.metrics.incomplete_frames == 0


def test_lost_chunk_cannot_desynchronize_the_following_frame() -> None:
    first = _frame(11)
    second = _frame(19)
    first_datagrams = encode_framed_frame(first, frame_id=200)
    second_datagrams = encode_framed_frame(second, frame_id=201)
    reassembler = FramedChunkReassembler(frame_timeout_seconds=0.5)

    for index, datagram in enumerate(first_datagrams):
        if index != 4:
            assert reassembler.accept(datagram, PEER, received_monotonic=index * 0.01) is None

    completed = None
    for index, datagram in enumerate(second_datagrams):
        result = reassembler.accept(datagram, PEER, received_monotonic=0.20 + index * 0.01)
        if result is not None:
            completed = result

    assert completed is not None
    assert completed[0] == second
    assert completed[1]["transport_frame_id"] == 201
    assert reassembler.evict_expired(now=1.0) == 1
    assert reassembler.metrics.incomplete_frames == 1
    assert reassembler.metrics.completed_frames == 1


def test_duplicate_chunk_is_ignored_without_duplicate_frame() -> None:
    payload = _frame(23)
    datagrams = encode_framed_frame(payload, frame_id=300)
    reassembler = FramedChunkReassembler()

    assert reassembler.accept(datagrams[0], PEER, received_monotonic=0.0) is None
    assert reassembler.accept(datagrams[0], PEER, received_monotonic=0.01) is None
    completed = None
    for index, datagram in enumerate(datagrams[1:], 1):
        completed = reassembler.accept(datagram, PEER, received_monotonic=0.01 + index * 0.01)

    assert completed is not None
    assert completed[0] == payload
    assert reassembler.metrics.duplicate_chunks == 1
    assert reassembler.metrics.completed_frames == 1


def test_conflicting_duplicate_discards_the_pending_frame() -> None:
    datagrams = encode_framed_frame(_frame(29), frame_id=400)
    reassembler = FramedChunkReassembler()
    assert reassembler.accept(datagrams[0], PEER, received_monotonic=0.0) is None

    conflicting = bytearray(datagrams[0])
    conflicting[-1] ^= 0xFF
    assert reassembler.accept(bytes(conflicting), PEER, received_monotonic=0.01) is None
    assert reassembler.metrics.conflicting_duplicates == 1
    assert reassembler.metrics.incomplete_frames == 1
    assert not reassembler.pending


def test_whole_frame_crc_rejects_payload_corruption() -> None:
    datagrams = encode_framed_frame(_frame(31), frame_id=500)
    corrupted = bytearray(datagrams[5])
    corrupted[-1] ^= 0x01
    datagrams[5] = bytes(corrupted)
    reassembler = FramedChunkReassembler()

    assert all(
        reassembler.accept(datagram, PEER, received_monotonic=index * 0.01) is None
        for index, datagram in enumerate(datagrams)
    )
    assert reassembler.metrics.checksum_failures == 1
    assert reassembler.metrics.completed_frames == 0


def test_decoder_rejects_bad_magic_and_offset() -> None:
    datagram = bytearray(encode_framed_frame(_frame(37), frame_id=600)[0])
    datagram[0:4] = b"BAD!"
    with pytest.raises(ChunkProtocolError):
        decode_framed_chunk(bytes(datagram))

    datagram = bytearray(encode_framed_frame(_frame(37), frame_id=600)[1])
    datagram[20:24] = (0).to_bytes(4, "big")
    with pytest.raises(ChunkProtocolError):
        decode_framed_chunk(bytes(datagram))


def test_legacy_and_v2_reassembly_modes_are_mutually_exclusive() -> None:
    required = [
        "--output", "out",
        "--collection-id", "C001",
        "--subject-id", "S001",
        "--session-id", "session_001",
        "--operator-code", "OP001",
    ]
    with pytest.raises(SystemExit):
        parse_args(required + ["--reassemble-udp-chunks", "--legacy-stream-reassembly"])


def _capture_args(tmp_path: Path, session_id: str):
    return parse_args(
        [
            "--output", str(tmp_path),
            "--collection-id", "collection_v2",
            "--subject-id", "S001",
            "--session-id", session_id,
            "--operator-code", "OP001",
            "--source-label", "EMPTY",
            "--reassemble-udp-chunks",
        ]
    )


def test_v2_artifacts_pass_real_capture_validator(tmp_path: Path) -> None:
    args = _capture_args(tmp_path, "session_v2_valid")
    writer = CaptureWriter(args)
    writer.prepare()
    reassembler = FramedChunkReassembler()
    completed = None
    for index, datagram in enumerate(encode_framed_frame(_frame(41), frame_id=700)):
        writer.record_udp_chunk(datagram, index)
        result = reassembler.accept(datagram, PEER, received_monotonic=index * 0.01)
        if result is not None:
            completed = result
    assert completed is not None
    writer.record_valid_datagram(completed[0], transport_metadata=completed[1])
    writer.transport_metrics = reassembler.snapshot()
    writer.close()

    result = validate_capture(writer.collection_dir)
    assert result["capture_status"] == "CAPTURE_STRUCTURE_VALID_WITH_LIMITATIONS"
    assert result["errors"] == []
    session = result["sessions"][0]
    assert session["valid_frame_count"] == 1


def test_v2_incomplete_frame_is_retained_and_validator_fails_closed(tmp_path: Path) -> None:
    args = _capture_args(tmp_path, "session_v2_incomplete")
    writer = CaptureWriter(args)
    writer.prepare()
    reassembler = FramedChunkReassembler()
    datagrams = encode_framed_frame(_frame(43), frame_id=701)
    for index, datagram in enumerate(datagrams[:-1]):
        writer.record_udp_chunk(datagram, index)
        assert reassembler.accept(datagram, PEER, received_monotonic=index * 0.01) is None
    reassembler.finalize()
    writer.transport_metrics = reassembler.snapshot()
    writer.close()

    result = validate_capture(writer.collection_dir)
    codes = {item["code"] for item in result["errors"]}
    assert result["capture_status"] == "CAPTURE_INVALID"
    assert "FRAMED_UDP_V2_INTEGRITY_FAILED" in codes
