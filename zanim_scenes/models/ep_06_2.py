# Numerical model from episodes/06.2/scene.py; no renderer dependency.
"""D2L 6.2 — cross-correlation: a 3×3 kernel writes one output cell, then steps.

Every on-screen number comes from the arrays and ``corr2d`` below. This is
forward computation only: the kernel is not trained and no bias is added.
"""

from __future__ import annotations
import numpy as np

INPUT_X = np.array(
    [
        [0.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
        [8.0, 9.0, 0.0, 1.0],
        [2.0, 3.0, 4.0, 5.0],
    ],
    dtype=float,
)
KERNEL = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=float)


def corr2d(input_x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """D2L ``corr2d``: elementwise product of each window with K, then sum."""
    kernel_h, kernel_w = kernel.shape
    out_h = input_x.shape[0] - kernel_h + 1
    out_w = input_x.shape[1] - kernel_w + 1
    output = np.zeros((out_h, out_w), dtype=float)
    for row in range(out_h):
        for col in range(out_w):
            window = input_x[row : row + kernel_h, col : col + kernel_w]
            output[row, col] = float(np.sum(window * kernel))
    return output


OUTPUT = corr2d(INPUT_X, KERNEL)
KERNEL_H, KERNEL_W = KERNEL.shape


def fmt_int(value: float) -> str:
    """Integer glyph for a cell; use a true minus if a later kernel needs it."""
    rounded = int(round(value))
    if rounded < 0:
        return f"−{abs(rounded)}"
    return str(rounded)
