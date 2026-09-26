# Numerical model from episodes/07.7/scene.py; no renderer dependency.
"""D2L 7.7 — a dense block: each layer concatenates on channels, x ← [x, f(x)].

Every plotted number comes from the arrays below.  f is a tiny untrained
1×1 map plus ReLU.  New maps dock onto the stack; old cells keep their
values.  That is the contrast with 7.6, where F(x) is added.
"""

from __future__ import annotations
import numpy as np

INPUT_X = np.array([[[1.0, 0.5], [0.2, 0.8]], [[0.4, 0.1], [0.7, 0.3]]], dtype=float)
WEIGHTS_1 = np.array([[0.5, 0.4]], dtype=float)
BIAS_1 = np.array([0.1], dtype=float)
WEIGHTS_2 = np.array([[0.2, 0.3, 0.5]], dtype=float)
BIAS_2 = np.array([-0.05], dtype=float)


def relu(values: np.ndarray) -> np.ndarray:
    """Component-wise max(z, 0) for the same values drawn on screen."""
    return np.maximum(values, 0.0)


def conv1x1(maps: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Untrained 1×1 mix: each spatial cell is a dense map on its channel vector."""
    _channels, height, width = maps.shape
    out_channels = weight.shape[0]
    output = np.zeros((out_channels, height, width), dtype=float)
    for out_index in range(out_channels):
        for row in range(height):
            for col in range(width):
                output[out_index, row, col] = float(
                    np.dot(weight[out_index], maps[:, row, col]) + bias[out_index]
                )
    return relu(output)


def dense_concat(current: np.ndarray, layer_out: np.ndarray) -> np.ndarray:
    """Channel concat: x ← [x, f(x)], never an add."""
    return np.concatenate((current, layer_out), axis=0)


FEATURE_1 = conv1x1(INPUT_X, WEIGHTS_1, BIAS_1)
STACK_1 = dense_concat(INPUT_X, FEATURE_1)
FEATURE_2 = conv1x1(STACK_1, WEIGHTS_2, BIAS_2)
STACK_2 = dense_concat(STACK_1, FEATURE_2)


def format_cell(value: float) -> str:
    """Two decimals with a true minus glyph, never a hyphen-minus."""
    rounded = round(float(value), 2)
    if rounded < 0:
        return f"−{abs(rounded):.2f}"
    return f"{rounded:.2f}"
