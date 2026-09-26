# Numerical model from 15.5; renderer colors omitted.
"""D2L 15.5 — one premise token queries the hypothesis keys, then a 3-way score.

Premise rows ``A`` and hypothesis rows ``B`` are the only numerical source:
scores are A Bᵀ / √d, each α row sums to 1, β = α B, and the tiny untrained
inference is softmax of a 3-vector built from a·β and ||a−β||. Nothing is trained.
"""

from __future__ import annotations
import numpy as np

PREMISE = np.array([[0.2, 1.2], [1.0, 0.5], [-0.5, 0.9]], dtype=float)
HYPOTHESIS = np.array([[1.8, 0.2], [1.65, 1.15], [-1.1, -0.95]], dtype=float)
TRAVELER = 1
FEATURE_DIM = PREMISE.shape[1]
SCALE = 1.0 / np.sqrt(FEATURE_DIM)


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax, the same α drawn on screen."""
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def softmax(values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for the 3-way score."""
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


SCORES = PREMISE @ HYPOTHESIS.T * SCALE
WEIGHTS = softmax_rows(SCORES)
ALIGNED = WEIGHTS @ HYPOTHESIS
TRAVELER_WEIGHTS = WEIGHTS[TRAVELER]
TRAVELER_BETA = ALIGNED[TRAVELER]
TRAVELER_A = PREMISE[TRAVELER]
ALIGN_DOT = float(TRAVELER_A @ TRAVELER_BETA)
ALIGN_GAP = float(np.linalg.norm(TRAVELER_A - TRAVELER_BETA))
LOGITS = np.array([0.7 * ALIGN_DOT, -0.35 * ALIGN_DOT, 0.8 * ALIGN_GAP], dtype=float)
PROBABILITIES = softmax(LOGITS)
MAX_CLASS = int(np.argmax(PROBABILITIES))
CLASS_NAMES = ("e", "c", "n")
