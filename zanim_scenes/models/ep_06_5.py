# Numerical model from episodes/06.5/scene.py; no renderer dependency.
"""D2L 6.5 — 2×2 max pooling then average pooling on a tiny real map.

Every number on screen comes from the arrays below. There are no learned
weights: pooling is a deterministic window operator. Max keeps one cell;
average uses every cell in the same window. The 4×4 → 2×2 drop is stride 2.
"""

from __future__ import annotations
import numpy as np

INPUT = np.array(
    [
        [1.0, 9.0, 2.0, 0.0],
        [1.0, 1.0, 8.0, 6.0],
        [0.0, 2.0, 5.0, 7.0],
        [8.0, 2.0, 1.0, 3.0],
    ],
    dtype=float,
)
POOL_SIZE = (2, 2)
STRIDE = 2


def format_number(value: float) -> str:
    """Integers stay integers; non-integers keep one decimal."""
    if abs(value - round(value)) < 1e-09:
        return str(int(round(value)))
    return f"{value:.1f}"


def pooling_records(
    map_values: np.ndarray,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Slide a 2×2 window with stride 2. Numpy is the only arithmetic."""
    height, width = map_values.shape
    pool_h, pool_w = POOL_SIZE
    out_h = (height - pool_h) // STRIDE + 1
    out_w = (width - pool_w) // STRIDE + 1
    max_out = np.zeros((out_h, out_w), dtype=float)
    avg_out = np.zeros((out_h, out_w), dtype=float)
    records: list[dict] = []
    for out_row in range(out_h):
        for out_col in range(out_w):
            row = out_row * STRIDE
            col = out_col * STRIDE
            patch = map_values[row : row + pool_h, col : col + pool_w]
            max_out[out_row, out_col] = float(np.max(patch))
            avg_out[out_row, out_col] = float(np.mean(patch))
            local_row, local_col = np.unravel_index(int(np.argmax(patch)), patch.shape)
            records.append(
                {
                    "out_row": out_row,
                    "out_col": out_col,
                    "origin": (row, col),
                    "patch": np.array(patch, copy=True),
                    "max": float(max_out[out_row, out_col]),
                    "avg": float(avg_out[out_row, out_col]),
                    "winner": (row + int(local_row), col + int(local_col)),
                }
            )
    return (records, max_out, avg_out)


WINDOWS, MAX_OUT, AVG_OUT = pooling_records(INPUT)
