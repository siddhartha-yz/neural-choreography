# Numerical model from episodes/06.6/scene.py; no renderer dependency.
"""D2L 6.6 — one haloed sample walking a tiny untrained LeNet-style net.

The arrays below are the only numerical source of truth: an 8×8 input, two
valid cross-correlations, two 2×2 average pools, flatten, then a dense map
to logits o. Weights are drawn once from a fixed RNG. Nothing is trained,
and o is not a class prediction.
"""

from __future__ import annotations
import numpy as np


def correlate2d(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Valid 2-D cross-correlation, matching D2L's conv layer."""
    kernel_h, kernel_w = kernel.shape
    out_h = array.shape[0] - kernel_h + 1
    out_w = array.shape[1] - kernel_w + 1
    output = np.empty((out_h, out_w), dtype=float)
    for row in range(out_h):
        for col in range(out_w):
            window = array[row : row + kernel_h, col : col + kernel_w]
            output[row, col] = float(np.sum(window * kernel))
    return output


def average_pool_2x2(array: np.ndarray) -> np.ndarray:
    """2×2 average pooling with stride 2."""
    rows, cols = array.shape
    return array.reshape(rows // 2, 2, cols // 2, 2).mean(axis=(1, 3))


RNG = np.random.default_rng(66)
INPUT_X = np.zeros((8, 8), dtype=float)
INPUT_X[1:4, 1:4] = 1.0
INPUT_X[5:7, 5:7] = 0.7
KERNEL_1 = RNG.normal(0.0, 0.55, (3, 3))
KERNEL_2 = RNG.normal(0.0, 0.55, (2, 2))
WEIGHTS = RNG.normal(0.0, 0.85, (3, 1))
BIAS = RNG.normal(0.0, 0.2, (3, 1))
CONV_1 = correlate2d(INPUT_X, KERNEL_1)
POOL_1 = average_pool_2x2(CONV_1)
CONV_2 = correlate2d(POOL_1, KERNEL_2)
POOL_2 = average_pool_2x2(CONV_2)
FLAT = POOL_2.reshape(-1, 1)
OUTPUT_O = WEIGHTS @ FLAT + BIAS


def format_signed(value: float, decimals: int) -> str:
    """ASCII digits with a true minus glyph, matching episodes 3.1 / 3.4."""
    if value < 0:
        return f"−{abs(value):.{decimals}f}"
    return f"+{value:.{decimals}f}" if decimals >= 3 else f"{value:.{decimals}f}"
