# Numerical model from 10.4; renderer colors omitted.
"""D2L 10.4 — Bahdanau: decoder hidden queries encoder steps.

Every α, every context needle, and the box that lights up come from the
additive-attention numpy below. Encoder hiddens are keys and values;
the halo traveler is the decoder query s. Untrained. Tiny 3×2 sequence.
"""

from __future__ import annotations
import numpy as np

ENCODER_H = np.array([[1.2, 0.2], [0.35, 0.95], [-0.85, 0.8]], dtype=float)
QUERY_S0 = np.array([1.0, 0.5], dtype=float)
QUERY_S1 = np.array([-0.55, 1.05], dtype=float)
W_Q = np.array([[-0.48712729, 0.68103308], [-0.86384803, 3.08807261]], dtype=float)
W_K = np.array([[2.42289974, -0.6491824], [-2.02660596, 0.34492875]], dtype=float)
V_ATT = np.array([-1.17912302, -2.25548609], dtype=float)


def softmax(scores: np.ndarray) -> np.ndarray:
    """Softmax over encoder steps, the same α drawn on screen."""
    shifted = scores - np.max(scores)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def bahdanau(query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Additive scores, attention weights, and context for one decoder query."""
    energy = np.tanh((W_Q @ query)[None, :] + ENCODER_H @ W_K.T)
    scores = energy @ V_ATT
    alpha = softmax(scores)
    context = alpha @ ENCODER_H
    return (scores, alpha, context)


SCORES_0, ALPHA_0, CONTEXT_0 = bahdanau(QUERY_S0)
SCORES_1, ALPHA_1, CONTEXT_1 = bahdanau(QUERY_S1)


def signed_text(value: float, digits: int = 2) -> str:
    """True minus glyph for negative values, matching 3.1 / 9.2."""
    if value < 0:
        return f"−{abs(value):.{digits}f}"
    return f"{value:.{digits}f}"
