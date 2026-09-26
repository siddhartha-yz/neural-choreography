# Numerical model from 10.1; renderer colors omitted.
"""D2L 10.1 — a query against keys; the weights light up the matching values.

Every heatmap cell, key glow, and ŷ comes from the same numpy arrays below.
Nothing is trained. The on-screen formula is only the weighted sum of values.
"""

from __future__ import annotations
import numpy as np

QUERIES = np.array([[0.2, 1.2], [1.0, 0.5], [-1.0, -0.9]], dtype=float)
KEYS = np.array([[0.2, 2.0], [1.6, 0.2], [-1.5, -1.2]], dtype=float)
VALUES = np.array([[0.5], [2.0], [1.2]], dtype=float)
TRAVELER = 1


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax, the same α drawn on the heatmap."""
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


SCORES = QUERIES @ KEYS.T
WEIGHTS = softmax_rows(SCORES)
OUTPUTS = WEIGHTS @ VALUES
TRAVELER_WEIGHTS = WEIGHTS[TRAVELER]
TRAVELER_YHAT = float(OUTPUTS[TRAVELER, 0])
WINNER = int(np.argmax(TRAVELER_WEIGHTS))
