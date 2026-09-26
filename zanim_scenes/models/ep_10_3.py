# Numerical model from 10.3; renderer colors omitted.
"""D2L 10.3 — two scoring functions a(q, k) on the same q, k, v.

Additive and scaled-dot scores, softmax weights, and ŷ all come from the
numpy arrays below. Nothing is trained. Values equal keys so the weighted
sum sits in the same plane as the query.
"""

from __future__ import annotations
import numpy as np

QUERY = np.array([1.0, 0.5], dtype=float)
KEYS = np.array([[0.3, 1.95], [-1.5, 1.05], [0.5, -1.55]], dtype=float)
VALUES = KEYS.copy()
W_Q = np.array([[-0.09, 0.92], [0.32, -0.91]], dtype=float)
W_K = np.array([[-0.06, 0.93], [-0.83, -0.02]], dtype=float)
W_V = np.array([0.05, 1.0], dtype=float)
FEATURE_DIM = QUERY.shape[0]
SCALE = 1.0 / np.sqrt(FEATURE_DIM)


def softmax(values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for the same scores drawn on screen."""
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def additive_score(key: np.ndarray) -> float:
    """D2L (10.3.3): a(q, k) = w_vᵀ tanh(W_q q + W_k k)."""
    return float(W_V @ np.tanh(W_Q @ QUERY + W_K @ key))


ADDITIVE_SCORES = np.array([additive_score(key) for key in KEYS], dtype=float)
DOT_SCORES = KEYS @ QUERY * SCALE
ADDITIVE_WEIGHTS = softmax(ADDITIVE_SCORES)
DOT_WEIGHTS = softmax(DOT_SCORES)
ADDITIVE_YHAT = ADDITIVE_WEIGHTS @ VALUES
DOT_YHAT = DOT_WEIGHTS @ VALUES
ADDITIVE_WINNER = int(np.argmax(ADDITIVE_WEIGHTS))
DOT_WINNER = int(np.argmax(DOT_WEIGHTS))
