# Numerical model from 10.6; renderer colors omitted.
"""D2L 10.6 — one query attending over the same four tokens that are also keys and values.

The 4 × 2 array ``TOKENS`` is the only numerical source: Q = K = V = X,
scores are X Xᵀ / √d, rows of α sum to 1, and ŷ is α V. Nothing is trained.
"""

from __future__ import annotations
import numpy as np

TOKENS = np.array([[0.2, 1.2], [1.0, 0.5], [1.45, 0.85], [-1.0, -0.9]], dtype=float)
TRAVELER = 1
FEATURE_DIM = TOKENS.shape[1]
SCALE = 1.0 / np.sqrt(FEATURE_DIM)


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax, the same α drawn on screen."""
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


SCORES = TOKENS @ TOKENS.T * SCALE
WEIGHTS = softmax_rows(SCORES)
OUTPUTS = WEIGHTS @ TOKENS
TRAVELER_WEIGHTS = WEIGHTS[TRAVELER]
TRAVELER_YHAT = OUTPUTS[TRAVELER]
