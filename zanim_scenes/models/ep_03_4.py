# Renderer-independent numerical model from 03.4
"""D2L 3.4 — one 2-D sample moving from logits to softmax probabilities.

The arrays below are the only numerical source of truth for the traveler,
three-class affine map, logits, and softmax probabilities. This is forward
computation only: no parameters are trained or updated.
"""

from __future__ import annotations
import numpy as np

INPUT_X = np.array([[1.0], [0.5]], dtype=float)
WEIGHTS = np.array([[1.1, 0.4], [-0.8, 0.65], [0.25, 1.05]], dtype=float)
BIAS = np.array([[0.36], [-0.295], [-0.075]], dtype=float)
LOGITS = WEIGHTS @ INPUT_X + BIAS


def softmax(values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for the same values drawn on screen."""
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


PROBABILITIES = softmax(LOGITS)
MAX_CLASS = int(np.argmax(PROBABILITIES))
