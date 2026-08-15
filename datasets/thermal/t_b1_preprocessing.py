"""Reusable Thermal T-B1 preprocessing, target, metric, and ranking contracts.

This module contains only deterministic, model-independent operations.  It never
mutates the canonical arrays and never fits statistics outside TRAIN.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PROFILE_IDS = (
    "P0_CANONICAL_CELSIUS_DIRECT",
    "P1_TRAIN_FITTED_GLOBAL_ZSCORE",
    "P2_LEGACY_PER_FRAME_MINMAX",
)
CLASS_ORDER = ("NOT_HUMAN", "HUMAN_NORMAL", "HUMAN_FALL")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_ORDER)}
SOURCE_TO_TARGET = {
    "EMPTY_ROOM": "NOT_HUMAN",
    "SITTING": "HUMAN_NORMAL",
    "STANDING": "HUMAN_NORMAL",
    "LYING": "HUMAN_FALL",
}
CANONICAL_SHAPE = (62, 80)
CANONICAL_DTYPE = np.dtype("<f4")
P1_EPSILON = 1e-6


class PreprocessingContractError(ValueError):
    """Raised when a profile or canonical input violates the frozen contract."""


@dataclass(frozen=True)
class P1Statistics:
    mean: float
    std: float
    fit_sample_count: int
    fit_pixel_count: int
    fit_role: str
    train_artifact_sha256: str | None = None
    epsilon: float = P1_EPSILON

    def __post_init__(self) -> None:
        if self.fit_role != "TRAIN":
            raise PreprocessingContractError("P1 statistics may be fitted from TRAIN only")
        if self.fit_sample_count <= 0 or self.fit_pixel_count <= 0:
            raise PreprocessingContractError("P1 fit counts must be positive")
        if not np.isfinite(self.mean) or not np.isfinite(self.std):
            raise PreprocessingContractError("P1 statistics must be finite")
        if self.std < 0 or self.epsilon <= 0:
            raise PreprocessingContractError("P1 std/epsilon contract is invalid")

    @property
    def effective_std(self) -> float:
        return max(float(self.std), float(self.epsilon))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": float(self.mean),
            "std": float(self.std),
            "effective_std": float(self.effective_std),
            "epsilon": float(self.epsilon),
            "fit_sample_count": int(self.fit_sample_count),
            "fit_pixel_count": int(self.fit_pixel_count),
            "fit_role": self.fit_role,
            "train_artifact_sha256": self.train_artifact_sha256,
            "profile_id": "P1_TRAIN_FITTED_GLOBAL_ZSCORE",
        }

    def checksum(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _as_batch(frames: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(frames)
    if array.ndim == 2 and tuple(array.shape) == CANONICAL_SHAPE:
        array = array[None, ...]
    elif array.ndim == 3 and tuple(array.shape[-2:]) == CANONICAL_SHAPE:
        # Both (N, 62, 80) and (62, 80, 1) are handled explicitly below.
        if tuple(array.shape) == (62, 80, 1):
            array = array[None, :, :, 0]
    elif array.ndim == 4 and tuple(array.shape[1:]) == (*CANONICAL_SHAPE, 1):
        array = array[..., 0]
    if array.ndim != 3 or tuple(array.shape[1:]) != CANONICAL_SHAPE:
        raise PreprocessingContractError(
            f"canonical input must be (N,62,80), (62,80), (N,62,80,1), or (62,80,1); got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise PreprocessingContractError(f"canonical input dtype must be floating point; got {array.dtype}")
    array = np.asarray(array, dtype=CANONICAL_DTYPE)
    if not np.all(np.isfinite(array)):
        raise PreprocessingContractError("canonical input contains NaN or infinity")
    return array


def canonical_batch(frames: np.ndarray | Sequence[float]) -> np.ndarray:
    """Validate canonical Celsius frames without modifying the caller's array."""

    return _as_batch(frames)


def _with_channel(frames: np.ndarray) -> np.ndarray:
    return np.asarray(frames, dtype=CANONICAL_DTYPE)[..., None]


def apply_p0(frames: np.ndarray | Sequence[float]) -> np.ndarray:
    """P0: direct canonical Celsius with only the NHWC channel dimension added."""

    return _with_channel(_as_batch(frames))


def fit_p1_statistics(
    train_frames: np.ndarray | Sequence[float],
    *,
    fit_role: str = "TRAIN",
    train_artifact_sha256: str | None = None,
    chunk_rows: int = 256,
) -> P1Statistics:
    """Fit scalar global mean/std over TRAIN pixels only.

    Chunked accumulation keeps the later SSD/memmap path bounded in RAM and
    avoids converting the entire 635 MB TRAIN artifact to float64 at once.
    """

    if fit_role != "TRAIN":
        raise PreprocessingContractError("P1 statistics may be fitted from TRAIN only")
    array = _as_batch(train_frames)
    if chunk_rows <= 0:
        raise PreprocessingContractError("chunk_rows must be positive")
    total = 0
    sum_value = 0.0
    sum_square = 0.0
    for start in range(0, array.shape[0], chunk_rows):
        chunk = np.asarray(array[start : start + chunk_rows], dtype=np.float64)
        total += int(chunk.size)
        sum_value += float(chunk.sum(dtype=np.float64))
        sum_square += float(np.square(chunk, dtype=np.float64).sum(dtype=np.float64))
    if total <= 0:
        raise PreprocessingContractError("cannot fit P1 statistics on an empty TRAIN array")
    mean = sum_value / total
    variance = max(0.0, (sum_square / total) - (mean * mean))
    return P1Statistics(
        mean=mean,
        std=float(np.sqrt(variance)),
        fit_sample_count=int(array.shape[0]),
        fit_pixel_count=total,
        fit_role=fit_role,
        train_artifact_sha256=train_artifact_sha256,
    )


def apply_p1(frames: np.ndarray | Sequence[float], statistics: P1Statistics) -> np.ndarray:
    """Apply frozen TRAIN-fitted global z-score statistics."""

    if not isinstance(statistics, P1Statistics):
        raise PreprocessingContractError("P1 requires P1Statistics fitted from TRAIN")
    array = _as_batch(frames).astype(np.float32, copy=True)
    array = (array - np.float32(statistics.mean)) / np.float32(statistics.effective_std)
    if not np.all(np.isfinite(array)):
        raise PreprocessingContractError("P1 produced non-finite values")
    return _with_channel(array)


def apply_p2(frames: np.ndarray | Sequence[float]) -> np.ndarray:
    """P2: exact legacy ThermalInterpreter per-frame normalization behavior."""

    array = _as_batch(frames).astype(np.float32, copy=True)
    mins = array.min(axis=(1, 2), keepdims=True)
    maxes = array.max(axis=(1, 2), keepdims=True)
    needs_normalization = (mins < 0.0) | (maxes > 1.0)
    ranges = maxes - mins
    variable = needs_normalization & (ranges > 0.0)
    constant = needs_normalization & ~variable
    if np.any(variable):
        array = np.where(variable, (array - mins) / ranges, array)
    if np.any(constant):
        array = np.where(constant, np.clip(array, 0.0, 1.0), array)
    if not np.all(np.isfinite(array)):
        raise PreprocessingContractError("P2 produced non-finite values")
    return _with_channel(np.clip(array, 0.0, 1.0))


def apply_profile(
    profile_id: str,
    frames: np.ndarray | Sequence[float],
    *,
    p1_statistics: P1Statistics | None = None,
) -> np.ndarray:
    if profile_id == "P0_CANONICAL_CELSIUS_DIRECT":
        return apply_p0(frames)
    if profile_id == "P1_TRAIN_FITTED_GLOBAL_ZSCORE":
        return apply_p1(frames, p1_statistics)  # type: ignore[arg-type]
    if profile_id == "P2_LEGACY_PER_FRAME_MINMAX":
        return apply_p2(frames)
    raise PreprocessingContractError(f"unknown preprocessing profile: {profile_id}")


def map_source_label(source_label: str) -> tuple[str, int]:
    try:
        target = SOURCE_TO_TARGET[str(source_label)]
    except KeyError as exc:
        raise PreprocessingContractError(f"unsupported source label: {source_label!r}") from exc
    return target, CLASS_TO_INDEX[target]


def labels_from_provenance(path: str | Path, expected_rows: int) -> tuple[np.ndarray, list[str]]:
    """Read immutable source labels and derived targets from T-A6 JSONL."""

    source_labels: list[str] = []
    targets: list[int] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= expected_rows:
                raise PreprocessingContractError("provenance has more rows than the canonical role")
            row = json.loads(line)
            source = str(row.get("original_label_name", ""))
            target_name, target_index = map_source_label(source)
            if row.get("compatibility_target") != target_name:
                raise PreprocessingContractError("provenance compatibility target contradicts frozen mapping")
            if row.get("source_label_modified") is True:
                raise PreprocessingContractError("source labels must remain immutable")
            source_labels.append(source)
            targets.append(target_index)
    if len(targets) != expected_rows:
        raise PreprocessingContractError(f"provenance row count {len(targets)} != {expected_rows}")
    return np.asarray(targets, dtype=np.int32), source_labels


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compute_metrics(y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if true.shape != pred.shape or true.size == 0:
        raise PreprocessingContractError("metric inputs must have equal non-empty lengths")
    if np.any((true < 0) | (true >= len(CLASS_ORDER))) or np.any((pred < 0) | (pred >= len(CLASS_ORDER))):
        raise PreprocessingContractError("metric labels are outside the frozen class order")
    confusion = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=np.int64)
    for actual, predicted in zip(true, pred):
        confusion[int(actual), int(predicted)] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    recalls: list[float] = []
    for index, name in enumerate(CLASS_ORDER):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(confusion[index].sum())}
        f1_values.append(f1)
        recalls.append(recall)
    return {
        "class_order": list(CLASS_ORDER),
        "sample_count": int(true.size),
        "accuracy": float(np.mean(true == pred)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "h_fall_posture_proxy_recall": float(per_class["HUMAN_FALL"]["recall"]),
        "confusion_matrix": confusion.tolist(),
        "prediction_distribution": {name: int(np.sum(pred == index)) for index, name in enumerate(CLASS_ORDER)},
    }


def compare_float_desc(left: float, right: float, tolerance: float = 1e-5) -> int:
    if abs(float(left) - float(right)) < tolerance:
        return 0
    return -1 if left > right else 1


def compare_validation_rows(left: Mapping[str, Any], right: Mapping[str, Any], tolerance: float = 1e-5) -> int:
    """Return -1 when left ranks ahead of right under frozen T-B0 policy."""

    left_metrics = left.get("validation_metrics", left.get("metrics", {}))
    right_metrics = right.get("validation_metrics", right.get("metrics", {}))
    for key in ("macro_f1", "balanced_accuracy", "h_fall_posture_proxy_recall"):
        comparison = compare_float_desc(float(left_metrics.get(key, 0.0)), float(right_metrics.get(key, 0.0)), tolerance)
        if comparison:
            return comparison
    left_params = int(left.get("parameter_count", 2**63 - 1))
    right_params = int(right.get("parameter_count", 2**63 - 1))
    if left_params != right_params:
        return -1 if left_params < right_params else 1
    left_size = int(left.get("tflite_artifact_size_bytes", 2**63 - 1))
    right_size = int(right.get("tflite_artifact_size_bytes", 2**63 - 1))
    if left_size != right_size:
        return -1 if left_size < right_size else 1
    left_id = str(left.get("profile_id", left.get("candidate_id", "")))
    right_id = str(right.get("profile_id", right.get("candidate_id", "")))
    return -1 if left_id < right_id else (1 if left_id > right_id else 0)


def select_validation_winner(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise PreprocessingContractError("winner selection requires at least one VALIDATION result")
    if any(row.get("real_metrics") is not None for row in rows):
        raise PreprocessingContractError("REAL metrics cannot enter winner selection")
    ordered = sorted(rows, key=cmp_to_key(compare_validation_rows))
    winner = dict(ordered[0])
    winner["selection_role"] = "VALIDATION"
    winner["rule_id"] = "THERMAL_T_B0_WINNER_RULE_001"
    winner["tie_tolerance"] = 1e-5
    return winner


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
