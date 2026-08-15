"""T-B1 SMALL_CNN_BASELINE_V1 construction and controlled training helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import platform
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from datasets.thermal.t_b1_preprocessing import compute_metrics

try:  # TensorFlow is optional for metadata-only validation and compact tests.
    import tensorflow as tf
except Exception:  # pragma: no cover - exercised on dependency-free hosts
    tf = None  # type: ignore[assignment]


BASELINE_ID = "SMALL_CNN_BASELINE_V1"
PRIMARY_SEED = 20260813
EXPECTED_PARAMETER_COUNT = 312131


class ModelContractError(RuntimeError):
    """Raised when TensorFlow/model execution cannot satisfy the T-B1 contract."""


def require_tensorflow() -> Any:
    if tf is None:
        raise ModelContractError("TensorFlow is unavailable; Stage 1 metadata validation can still run")
    return tf


def seed_everything(seed: int = PRIMARY_SEED, *, enable_determinism: bool = True) -> None:
    if not isinstance(seed, int) or seed < 0:
        raise ModelContractError("seed must be a non-negative integer")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.keras.utils.set_random_seed(seed)
        if enable_determinism:
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:
                pass


def create_small_cnn_baseline() -> Any:
    """Build exactly the preregistered T-B0 frame CNN.

    Dropout is intentionally absent: the frozen T-B0 architecture description
    contains only the two convolutions, two pools, Flatten, Dense(32), and
    Dense(3).  The historical script's Dropout did not change parameter count
    but is not silently imported into this controlled comparison.
    """

    tensorflow = require_tensorflow()
    return tensorflow.keras.Sequential(
        [
            tensorflow.keras.layers.Input(shape=(62, 80, 1), name="input_frame"),
            tensorflow.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same", name="conv1"),
            tensorflow.keras.layers.MaxPooling2D((2, 2), name="pool1"),
            tensorflow.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv2"),
            tensorflow.keras.layers.MaxPooling2D((2, 2), name="pool2"),
            tensorflow.keras.layers.Flatten(name="flatten"),
            tensorflow.keras.layers.Dense(32, activation="relu", name="dense32"),
            tensorflow.keras.layers.Dense(3, activation="softmax", name="class_output"),
        ],
        name="small_cnn_baseline_v1",
    )


def architecture_fingerprint(model: Any) -> str:
    payload = {
        "candidate_id": BASELINE_ID,
        "input_shape": [1, 62, 80, 1],
        "output_shape": [1, 3],
        "layers": [
            {"name": "conv1", "type": "Conv2D", "filters": 16, "kernel": [3, 3], "activation": "relu", "padding": "same"},
            {"name": "pool1", "type": "MaxPooling2D", "pool": [2, 2]},
            {"name": "conv2", "type": "Conv2D", "filters": 32, "kernel": [3, 3], "activation": "relu", "padding": "same"},
            {"name": "pool2", "type": "MaxPooling2D", "pool": [2, 2]},
            {"name": "flatten", "type": "Flatten"},
            {"name": "dense32", "type": "Dense", "units": 32, "activation": "relu"},
            {"name": "class_output", "type": "Dense", "units": 3, "activation": "softmax"},
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def initial_weights(seed: int = PRIMARY_SEED) -> tuple[list[np.ndarray], str, str]:
    seed_everything(seed)
    model = create_small_cnn_baseline()
    weights = [np.asarray(weight.numpy(), dtype=np.dtype(weight.numpy().dtype)).copy() for weight in model.weights]
    fingerprint = fingerprint_weight_arrays(weights, model.weights)
    if model.count_params() != EXPECTED_PARAMETER_COUNT:
        raise ModelContractError(f"baseline parameter count {model.count_params()} != {EXPECTED_PARAMETER_COUNT}")
    return weights, fingerprint, architecture_fingerprint(model)


def fingerprint_weight_arrays(weights: Sequence[np.ndarray], tensors: Sequence[Any] | None = None) -> str:
    digest = hashlib.sha256()
    for index, array in enumerate(weights):
        name = str(tensors[index].name) if tensors is not None else f"weight_{index}"
        value = np.ascontiguousarray(array)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def backend_info() -> dict[str, Any]:
    devices: list[dict[str, str]] = []
    gpu_visible = False
    if tf is not None:
        for device in tf.config.list_physical_devices():
            devices.append({"name": str(device.name), "device_type": str(device.device_type)})
        gpu_visible = bool(tf.config.list_physical_devices("GPU"))
    is_apple_silicon = platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "tensorflow": getattr(tf, "__version__", None),
        "numpy": np.__version__,
        "physical_devices": devices,
        "gpu_visible": gpu_visible,
        "apple_silicon": is_apple_silicon,
        "backend_selected": "APPLE_METAL" if gpu_visible and is_apple_silicon else "CPU",
        "gpu_optional": True,
    }


def model_contract() -> dict[str, Any]:
    model = None
    measured_parameters: int | None = None
    architecture_sha256 = architecture_fingerprint(None)
    if tf is not None:
        model = create_small_cnn_baseline()
        measured_parameters = int(model.count_params())
    return {
        "candidate_id": BASELINE_ID,
        "architecture_fingerprint": architecture_sha256,
        "parameter_count_target": EXPECTED_PARAMETER_COUNT,
        "parameter_count_measured": measured_parameters,
        "input_shape": [1, 62, 80, 1],
        "input_dtype": "float32",
        "output_shape": [1, 3],
        "output_dtype": "float32 probabilities",
        "class_order": ["NOT_HUMAN", "HUMAN_NORMAL", "HUMAN_FALL"],
        "tflite_conversion_in_stage1": False,
        "legacy_model_replacement": False,
    }


_CallbackBase = tf.keras.callbacks.Callback if tf is not None else object


class ValidationMacroF1Callback(_CallbackBase):
    """Keras callback that makes validation Macro F1 the monitored quantity."""

    def __init__(self, validation_x: np.ndarray, validation_y: np.ndarray, batch_size: int = 64) -> None:
        if tf is None:
            raise ModelContractError("TensorFlow is unavailable")
        super().__init__()
        self.validation_x = validation_x
        self.validation_y = validation_y
        self.batch_size = batch_size
        self.epoch_metrics: list[dict[str, Any]] = []

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:  # type: ignore[override]
        logs = logs if logs is not None else {}
        probabilities = self.model.predict(self.validation_x, batch_size=self.batch_size, verbose=0)  # type: ignore[attr-defined]
        metrics = compute_metrics(self.validation_y, np.argmax(probabilities, axis=1))
        logs["val_macro_f1"] = metrics["macro_f1"]
        logs["val_balanced_accuracy"] = metrics["balanced_accuracy"]
        logs["val_human_fall_posture_proxy_recall"] = metrics["h_fall_posture_proxy_recall"]
        self.epoch_metrics.append({"epoch": int(epoch + 1), **metrics, "loss": float(logs.get("loss", 0.0)), "val_loss": float(logs.get("val_loss", 0.0)), "val_accuracy": float(logs.get("val_accuracy", 0.0))})


def train_profile(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    profile_id: str,
    seed: int,
    frozen_initial_weights: Sequence[np.ndarray],
    budget: Mapping[str, Any],
    checkpoint_path: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Train one registered profile; this is called only by FULL_EXPERIMENT."""

    seed_everything(seed)
    model = create_small_cnn_baseline()
    model.set_weights([np.asarray(value).copy() for value in frozen_initial_weights])
    before = fingerprint_weight_arrays([np.asarray(weight.numpy()) for weight in model.weights], model.weights)
    if before != fingerprint_weight_arrays(frozen_initial_weights, model.weights):
        raise ModelContractError("profile did not receive the frozen initial weights")
    baseline = budget["baseline_budget"]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(baseline["initial_learning_rate"])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    metric_callback = ValidationMacroF1Callback(validation_x, validation_y, int(baseline["batch_size"]))
    schedule = baseline["learning_rate_schedule"]
    callbacks = [
        metric_callback,
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_macro_f1",
            factor=float(schedule["factor"]),
            patience=int(schedule["patience"]),
            min_lr=float(schedule["minimum_learning_rate"]),
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            patience=int(baseline["early_stopping"]["patience"]),
            restore_best_weights=bool(baseline["early_stopping"]["restore_best_weights"]),
            mode="max",
        ),
    ]
    history = model.fit(
        train_x,
        train_y,
        validation_data=(validation_x, validation_y),
        epochs=int(baseline["maximum_epochs"]),
        batch_size=int(baseline["batch_size"]),
        shuffle=True,
        callbacks=callbacks,
        verbose=0,
    )
    probabilities = model.predict(validation_x, batch_size=int(baseline["batch_size"]), verbose=0)
    metrics = compute_metrics(validation_y, np.argmax(probabilities, axis=1))
    best_epoch = max(metric_callback.epoch_metrics, key=lambda item: (float(item["macro_f1"]), -int(item["epoch"]))) if metric_callback.epoch_metrics else {"epoch": 0}
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(checkpoint))
    result = {
        "status": "VALIDATION_COMPLETE",
        "profile_id": profile_id,
        "candidate_id": BASELINE_ID,
        "seed": int(seed),
        "initial_weight_fingerprint": before,
        "architecture_fingerprint": architecture_fingerprint(model),
        "parameter_count": int(model.count_params()),
        "best_epoch": int(best_epoch.get("epoch", 0)),
        "validation_metrics": metrics,
        "epoch_metrics": metric_callback.epoch_metrics,
        "history_keys": sorted(history.history),
    }
    return model, result
