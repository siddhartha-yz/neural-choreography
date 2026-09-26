# Renderer-independent numerical model from 04.1
"""D2L 4.1 — one hidden layer, forward only, with a visible ReLU clip.

The arrays below are the only numerical source of truth for the traveler,
affine pre-activations, ReLU hidden units, and output. This is forward
computation only: no parameters are trained or updated.
"""

from __future__ import annotations
import numpy as np

INPUT_X = np.array([[0.8], [0.6]], dtype=float)
WEIGHTS_1 = np.array([[1.0, 0.5], [-0.5, -0.5]], dtype=float)
BIAS_1 = np.array([[0.1], [-0.1]], dtype=float)
PREACTIVATIONS = WEIGHTS_1 @ INPUT_X + BIAS_1


def relu(values: np.ndarray) -> np.ndarray:
    """Component-wise max(z, 0) for the same values drawn on screen."""
    return np.maximum(values, 0.0)


HIDDEN = relu(PREACTIVATIONS)
WEIGHTS_2 = np.array([[1.0, 0.5], [-0.75, 0.8]], dtype=float)
BIAS_2 = np.array([[0.1], [0.2]], dtype=float)
OUTPUTS = WEIGHTS_2 @ HIDDEN + BIAS_2
