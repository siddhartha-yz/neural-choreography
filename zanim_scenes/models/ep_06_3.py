# Numerical model from episodes/06.3/scene.py; no renderer dependency.
"""D2L 6.3 — same kernel, padding and stride change the output lattice.

Every output height, width, and cell value comes from the arrays below.
The kernel is never trained. Output shape is always
``floor((n - k + 2p) / s) + 1``, checked against the numpy lattice.
"""

from __future__ import annotations
import numpy as np

INPUT_X = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], dtype=float)
KERNEL = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float)
N = INPUT_X.shape[0]
K_SIZE = KERNEL.shape[0]
LAYOUT_PAD = 1
TRAVELER = (-1, 0)


def output_size(n: int, k: int, padding: int, stride: int) -> int:
    """D2L output length on one axis: floor((n − k + 2p) / s) + 1."""
    return int(np.floor((n - k + 2 * padding) / stride) + 1)


def corr2d(x: np.ndarray, kernel: np.ndarray, padding: int, stride: int) -> np.ndarray:
    """Cross-correlation that matches D2L (no kernel flip)."""
    padded = np.pad(x, padding)
    kernel_h, kernel_w = kernel.shape
    out_h = (padded.shape[0] - kernel_h) // stride + 1
    out_w = (padded.shape[1] - kernel_w) // stride + 1
    result = np.zeros((out_h, out_w), dtype=float)
    for row in range(out_h):
        for col in range(out_w):
            window = padded[
                row * stride : row * stride + kernel_h,
                col * stride : col * stride + kernel_w,
            ]
            result[row, col] = float(np.sum(window * kernel))
    return result


def lattice(padding: int, stride: int) -> dict[tuple[int, int], float]:
    """Map kernel top-left (r, c) in original coordinates to the output value."""
    values = corr2d(x=INPUT_X, kernel=KERNEL, padding=padding, stride=stride)
    cells: dict[tuple[int, int], float] = {}
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            origin = (-padding + row * stride, -padding + col * stride)
            cells[origin] = float(values[row, col])
    return cells


def fmt(value: float) -> str:
    rounded = int(round(value))
    if abs(value - rounded) < 1e-09:
        return str(rounded)
    return f"{value:.1f}"


Y_P0_S1 = corr2d(INPUT_X, KERNEL, padding=0, stride=1)
Y_P1_S1 = corr2d(INPUT_X, KERNEL, padding=1, stride=1)
Y_P1_S2 = corr2d(INPUT_X, KERNEL, padding=1, stride=2)
LATTICE_P0_S1 = lattice(0, 1)
LATTICE_P1_S1 = lattice(1, 1)
LATTICE_P1_S2 = lattice(1, 2)
