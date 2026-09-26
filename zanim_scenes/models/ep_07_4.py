# Numerical model from episodes/07.4/scene.py; no renderer dependency.
"""D2L 7.4 — Inception: four parallel windows, concat on the channel axis.

Every on-screen number comes from the arrays and helpers below. The four
branches share one input map. Kernels are not trained. Spatial size is kept
(pad so H,W match); the traveler is one cell whose channels grow by concat.
"""

from __future__ import annotations
import numpy as np

INPUT_X = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], dtype=float)
KERNEL_1 = np.array([[1.0]], dtype=float)
KERNEL_3 = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], dtype=float)
KERNEL_5 = np.array(
    [
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 2.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
    ],
    dtype=float,
)
POOL = 3
TRAVELER = (1, 1)
PAD_WINDOW = (-1, 3)


def corr2d(input_x: np.ndarray, kernel: np.ndarray, padding: int) -> np.ndarray:
    """Cross-correlation, D2L (no flip). Pad with zeros."""
    padded = np.pad(input_x, padding)
    kernel_h, kernel_w = kernel.shape
    out_h = padded.shape[0] - kernel_h + 1
    out_w = padded.shape[1] - kernel_w + 1
    output = np.zeros((out_h, out_w), dtype=float)
    for row in range(out_h):
        for col in range(out_w):
            window = padded[row : row + kernel_h, col : col + kernel_w]
            output[row, col] = float(np.sum(window * kernel))
    return output


def max_pool(
    input_x: np.ndarray, size: int = 3, padding: int = 1, stride: int = 1
) -> np.ndarray:
    """3 × 3 max-pool, stride 1, pad 1 — Inception keeps H, W."""
    padded = np.pad(input_x, padding)
    out_h = (padded.shape[0] - size) // stride + 1
    out_w = (padded.shape[1] - size) // stride + 1
    output = np.zeros((out_h, out_w), dtype=float)
    for row in range(out_h):
        for col in range(out_w):
            patch = padded[
                row * stride : row * stride + size, col * stride : col * stride + size
            ]
            output[row, col] = float(np.max(patch))
    return output


Y1 = corr2d(INPUT_X, KERNEL_1, padding=0)
Y3 = corr2d(INPUT_X, KERNEL_3, padding=1)
Y5 = corr2d(INPUT_X, KERNEL_5, padding=2)
YP = max_pool(INPUT_X, size=POOL, padding=1, stride=1)
CONCAT = np.stack([Y1, Y3, Y5, YP], axis=-1)
assert Y1.shape == Y3.shape == Y5.shape == YP.shape == INPUT_X.shape
assert CONCAT.shape == (3, 3, 4)
TRAVELER_CHANNELS = CONCAT[TRAVELER]


def fmt_int(value: float) -> str:
    rounded = int(round(value))
    if rounded < 0:
        return f"−{abs(rounded)}"
    return str(rounded)
