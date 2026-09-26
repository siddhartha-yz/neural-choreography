# Numerical model from 10.2; renderer colors omitted.
"""D2L 10.2 — Nadaraya–Watson: closer key, larger weight on its value.

Every needle, the yellow ŷ, and the curve come from the same numpy
softmax kernel below. This *is* the estimator: nothing is trained.
"""

from __future__ import annotations
import numpy as np

KEYS = np.array([0.4, 0.9, 1.4, 1.9, 2.4, 2.9, 3.4, 3.9, 4.4], dtype=float)
NOISE = np.array([0.18, -0.22, 0.12, -0.18, 0.2, -0.15, 0.1, -0.2, 0.16], dtype=float)


def latent_function(x_values: np.ndarray | float) -> np.ndarray | float:
    """D2L 10.2 generator: 2 sin(x) + x^0.8, without the noise term."""
    x_array = np.asarray(x_values, dtype=float)
    return 2.0 * np.sin(x_array) + np.power(np.maximum(x_array, 0.0), 0.8)


VALUES = np.asarray(latent_function(KEYS), dtype=float) + NOISE
QUERY_A = 1.0
QUERY_B = 3.6


def attention_weights(query: float) -> np.ndarray:
    """Gaussian-kernel softmax over the same keys drawn on screen."""
    scores = -0.5 * (query - KEYS) ** 2
    shifted = scores - np.max(scores)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def nw_predict(query: float) -> float:
    """Nadaraya–Watson ŷ = Σ αᵢ vᵢ for this query."""
    return float(np.dot(attention_weights(query), VALUES))


WEIGHTS_A = attention_weights(QUERY_A)
WEIGHTS_B = attention_weights(QUERY_B)
YHAT_A = nw_predict(QUERY_A)
YHAT_B = nw_predict(QUERY_B)


def signed_glyphs(value: float, digits: int) -> str:
    """Match 3.1: a true minus glyph, never a hyphen-minus."""
    sign = "−" if value < 0 else "+"
    return f"{sign}{abs(value):.{digits}f}"
