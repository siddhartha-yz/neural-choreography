# Renderer-independent numerical model from 05.1
"""D2L 5.1 — layers nested in a Sequential block, forward only.

Every moving value comes from the column-vector computation below.
The traveler is rewritten as it crosses each inner box:

    h = relu(W1 x + b1)
    o = W2 h + b2

Weights are never trained. Geometry is nested stroke rectangles, not a
class diagram and not a weight spreadsheet.
"""

from __future__ import annotations
import numpy as np

INPUT_X = np.array([[1.0], [0.5]], dtype=float)
WEIGHTS_1 = np.array([[1.2, 0.4], [-0.8, 0.2], [0.4, 0.8]], dtype=float)
BIAS_1 = np.array([[0.1], [-0.3], [0.05]], dtype=float)
PREACTIVATIONS = WEIGHTS_1 @ INPUT_X + BIAS_1


def relu(values: np.ndarray) -> np.ndarray:
    """Component-wise max(z, 0) for the same values drawn on screen."""
    return np.maximum(values, 0.0)


HIDDEN = relu(PREACTIVATIONS)
WEIGHTS_2 = np.array([[0.6, 0.2, 0.4], [-0.8, 0.4, 0.2]], dtype=float)
BIAS_2 = np.array([[0.1], [0.25]], dtype=float)
OUTPUTS = WEIGHTS_2 @ HIDDEN + BIAS_2
