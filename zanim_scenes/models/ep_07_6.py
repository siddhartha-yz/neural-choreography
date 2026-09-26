# Renderer-independent numerical model from 07.6
"""D2L 7.6 — a residual block: F(x) on the main path, x on a bypass.

Every plotted number comes from the arrays below.  F is two tiny affine
stages with a ReLU in between, untrained.  The shortcut copies x and never
enters F.  y = relu(F(x) + x).  When F(x) is small, y stays next to x.
"""

from __future__ import annotations
import numpy as np

INPUT_X = np.array([[1.0], [0.5]], dtype=float)
WEIGHTS_1 = np.array([[0.14, 0.03], [-0.11, -0.09]], dtype=float)
BIAS_1 = np.array([[-0.01], [-0.025]], dtype=float)
WEIGHTS_2 = np.array([[0.2, 0.07], [-0.16, 0.14]], dtype=float)
BIAS_2 = np.array([[0.04], [-0.045]], dtype=float)


def relu(values: np.ndarray) -> np.ndarray:
    """Component-wise max(z, 0) for the same values drawn on screen."""
    return np.maximum(values, 0.0)


def residual_block(x: np.ndarray) -> dict[str, np.ndarray]:
    """Untrained residual block: F = W₂ relu(W₁x + b₁) + b₂, y = relu(F + x)."""
    hidden_pre = WEIGHTS_1 @ x + BIAS_1
    hidden = relu(hidden_pre)
    residual = WEIGHTS_2 @ hidden + BIAS_2
    summed = residual + x
    output = relu(summed)
    return {"z": hidden_pre, "h": hidden, "F": residual, "s": summed, "y": output}


BLOCK = residual_block(INPUT_X)
RESIDUAL_F = np.asarray(BLOCK["F"], dtype=float)
SUMMED = np.asarray(BLOCK["s"], dtype=float)
OUTPUT_Y = np.asarray(BLOCK["y"], dtype=float)


def signed_glyphs(value: float, digits: int = 3) -> str:
    """Match 3.1: a true minus glyph, never a hyphen-minus."""
    sign = "−" if value < 0 else "+"
    return f"{sign}{abs(value):.{digits}f}"
