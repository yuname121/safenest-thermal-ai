"""T-B2 architecture candidates and deterministic training helpers.

T-B2 deliberately reuses the T-B1 preprocessing/metric implementation.  The
only experimental factor exposed here is the executable architecture.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from datasets.thermal.t_b1_model import (
    PRIMARY_SEED,
    ValidationMacroF1Callback,
    backend_info,
    fingerprint_weight_arrays,
    require_tensorflow,
    seed_everything,
)
from datasets.thermal.t_b1_preprocessing import compute_metrics


SMALL_CNN_ID = "SMALL_CNN_BASELINE_V1"
DEPTHWISE_ID = "DEPTHWISE_SEPARABLE_CNN_V1"
ARCHITECTURE_IDS = (SMALL_CNN_ID, DEPTHWISE_ID)
DEPTHWISE_PARAMETER_BOUND = 30_000


def _architecture_payload(candidate_id: str) -> dict[str, Any]:
    if candidate_id == SMALL_CNN_ID:
        return {
            "candidate_id": SMALL_CNN_ID,
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
    if candidate_id == DEPTHWISE_ID:
        return {
            "candidate_id": DEPTHWISE_ID,
            "input_shape": [1, 62, 80, 1],
            "output_shape": [1, 3],
            "layers": [
                {"name": "conv8", "type": "Conv2D", "filters": 8, "kernel": [3, 3], "activation": "relu", "padding": "same"},
                {"name": "pool1", "type": "MaxPooling2D", "pool": [2, 2]},
                {"name": "depthwise_sep16", "type": "SeparableConv2D", "filters": 16, "kernel": [3, 3], "activation": "relu", "padding": "same", "depth_multiplier": 1},
                {"name": "pool2", "type": "MaxPooling2D", "pool": [2, 2]},
                {"name": "global_average_pool", "type": "GlobalAveragePooling2D"},
                {"name": "class_output", "type": "Dense", "units": 3, "activation": "softmax"},
            ],
        }
    raise ValueError(f"unknown T-B2 architecture: {candidate_id}")


def architecture_fingerprint(candidate_id: str) -> str:
    payload = _architecture_payload(candidate_id)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def create_model(candidate_id: str) -> Any:
    tensorflow = require_tensorflow()
    if candidate_id == SMALL_CNN_ID:
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
    if candidate_id == DEPTHWISE_ID:
        return tensorflow.keras.Sequential(
            [
                tensorflow.keras.layers.Input(shape=(62, 80, 1), name="input_frame"),
                tensorflow.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same", name="conv8"),
                tensorflow.keras.layers.MaxPooling2D((2, 2), name="pool1"),
                tensorflow.keras.layers.SeparableConv2D(16, (3, 3), activation="relu", padding="same", depth_multiplier=1, name="depthwise_sep16"),
                tensorflow.keras.layers.MaxPooling2D((2, 2), name="pool2"),
                tensorflow.keras.layers.GlobalAveragePooling2D(name="global_average_pool"),
                tensorflow.keras.layers.Dense(3, activation="softmax", name="class_output"),
            ],
            name="depthwise_separable_cnn_v1",
        )
    raise ValueError(f"unknown T-B2 architecture: {candidate_id}")


def architecture_contract(candidate_id: str) -> dict[str, Any]:
    model = create_model(candidate_id)
    return {
        "candidate_id": candidate_id,
        "architecture_fingerprint": architecture_fingerprint(candidate_id),
        "parameter_count": int(model.count_params()),
        "parameter_bound": DEPTHWISE_PARAMETER_BOUND if candidate_id == DEPTHWISE_ID else 312131,
        "parameter_bound_inclusive": True,
        "input_shape": [1, 62, 80, 1],
        "output_shape": [1, 3],
        "class_order": ["NOT_HUMAN", "HUMAN_NORMAL", "HUMAN_FALL"],
        "layers": _architecture_payload(candidate_id)["layers"],
    }


def initial_weights(candidate_id: str, seed: int = PRIMARY_SEED) -> tuple[list[np.ndarray], str, str]:
    seed_everything(seed)
    model = create_model(candidate_id)
    weights = [np.asarray(weight.numpy()).copy() for weight in model.weights]
    fingerprint = fingerprint_weight_arrays(weights, model.weights)
    return weights, fingerprint, architecture_fingerprint(candidate_id)


def train_architecture(
    candidate_id: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    seed: int,
    budget: Mapping[str, Any],
    checkpoint_path: str | Path,
) -> tuple[Any, dict[str, Any]]:
    """Train one preregistered architecture exactly once under the frozen B1 budget."""

    tensorflow = require_tensorflow()
    seed_everything(seed)
    model = create_model(candidate_id)
    initial_fingerprint = fingerprint_weight_arrays([np.asarray(weight.numpy()) for weight in model.weights], model.weights)
    baseline = budget["baseline_budget"]
    model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=float(baseline["initial_learning_rate"])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    metric_callback = ValidationMacroF1Callback(validation_x, validation_y, int(baseline["batch_size"]))
    schedule = baseline["learning_rate_schedule"]
    callbacks = [
        metric_callback,
        tensorflow.keras.callbacks.ReduceLROnPlateau(
            monitor="val_macro_f1",
            factor=float(schedule["factor"]),
            patience=int(schedule["patience"]),
            min_lr=float(schedule["minimum_learning_rate"]),
            mode="max",
        ),
        tensorflow.keras.callbacks.EarlyStopping(
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
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(checkpoint))
    return model, {
        "status": "VALIDATION_COMPLETE",
        "candidate_id": candidate_id,
        "seed": int(seed),
        "initial_weight_fingerprint": initial_fingerprint,
        "architecture_fingerprint": architecture_fingerprint(candidate_id),
        "parameter_count": int(model.count_params()),
        "best_epoch": int(best_epoch.get("epoch", 0)),
        "validation_metrics": metrics,
        "epoch_metrics": metric_callback.epoch_metrics,
        "history_keys": sorted(history.history),
    }


def evaluate_model(model: Any, frames: np.ndarray, labels: np.ndarray, batch_size: int = 64) -> dict[str, Any]:
    probabilities = model.predict(frames, batch_size=batch_size, verbose=0)
    return compute_metrics(labels, np.argmax(probabilities, axis=1))


def frozen_training_contract(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "datasets/thermal/manifests/T-B0_offline_model_protocol/training_budget_policy.json"
    return json.loads(path.read_text(encoding="utf-8"))


def backend_contract() -> dict[str, Any]:
    return backend_info()
