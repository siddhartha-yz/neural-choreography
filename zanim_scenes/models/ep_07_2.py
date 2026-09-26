# Numerical model from episodes/07.2/scene.py; no renderer dependency.
"""D2L 7.2 — the same VGG block twice: conv keeps H,W, pool halves, channels double.

Every on-screen map is a real untrained numpy forward pass: 3×3 conv with
pad 1 and ReLU, then 2×2 max-pool stride 2. Not VGG-11, not a 13-layer tower.
"""

from __future__ import annotations
import numpy as np


def conv2d_pad1_relu(inputs: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Same-size 3×3 cross-correlation (pad 1) plus ReLU. ``inputs`` is H×W×Cin."""
    height, width, _in_channels = inputs.shape
    out_channels = weight.shape[0]
    padded = np.pad(inputs, ((1, 1), (1, 1), (0, 0)))
    output = np.zeros((height, width, out_channels), dtype=float)
    for out_index in range(out_channels):
        for row in range(height):
            for col in range(width):
                window = padded[row : row + 3, col : col + 3, :]
                output[row, col, out_index] = float(np.sum(window * weight[out_index]))
    return np.maximum(output, 0.0)


def max_pool2x2(inputs: np.ndarray) -> np.ndarray:
    """2×2 max-pool, stride 2, matching D2L's VGG block."""
    height, width, channels = inputs.shape
    output = np.zeros((height // 2, width // 2, channels), dtype=float)
    for row in range(height // 2):
        for col in range(width // 2):
            output[row, col, :] = np.max(
                inputs[2 * row : 2 * row + 2, 2 * col : 2 * col + 2, :], axis=(0, 1)
            )
    return output


RNG = np.random.default_rng(72)
INPUT_X = np.zeros((8, 8, 1), dtype=float)
INPUT_X[1:4, 1:4, 0] = 1.0
INPUT_X[5:7, 5:7, 0] = 0.6
WEIGHT_1 = RNG.normal(0.2, 0.28, (2, 3, 3, 1))
WEIGHT_2 = RNG.normal(0.12, 0.22, (4, 3, 3, 2))
CONV_1 = conv2d_pad1_relu(INPUT_X, WEIGHT_1)
POOL_1 = max_pool2x2(CONV_1)
CONV_2 = conv2d_pad1_relu(POOL_1, WEIGHT_2)
POOL_2 = max_pool2x2(CONV_2)


def format_number(value: float, decimals: int) -> str:
    rounded = round(float(value), decimals)
    if rounded < 0:
        return f"−{abs(rounded):.{decimals}f}"
    return f"{rounded:.{decimals}f}"
