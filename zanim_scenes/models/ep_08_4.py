# Numerical model from 08.4; renderer colors omitted.
"""D2L 8.4 — a hidden state carrying history across three RNN steps.

Every moving value comes from the tanh recurrence below.  Weights are fixed
and untrained.  The haloed traveler is the current token x_t; the yellow blob
is the same h, updated in place, with a faint trail of previous h left behind.
"""

from __future__ import annotations
import numpy as np

TOKENS = (
    np.array([[1.0], [0.5]], dtype=float),
    np.array([[-0.8], [1.1]], dtype=float),
    np.array([[0.6], [-0.9]], dtype=float),
)
WEIGHT_XH = np.array([[0.9, 0.2], [-0.25, 0.85]], dtype=float)
WEIGHT_HH = np.array([[0.6, -0.35], [0.4, 0.55]], dtype=float)
BIAS = np.array([[0.05], [-0.1]], dtype=float)
HIDDEN_START = np.zeros((2, 1), dtype=float)


def rnn_step(token: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    """One real RNN update; ϕ = tanh."""
    pre_activation = WEIGHT_XH @ token + WEIGHT_HH @ hidden + BIAS
    return np.tanh(pre_activation)


def hidden_trajectory() -> list[np.ndarray]:
    hidden = HIDDEN_START.copy()
    states = [hidden.copy()]
    for token in TOKENS:
        hidden = rnn_step(token, hidden)
        states.append(hidden.copy())
    return states


HIDDEN_STATES = hidden_trajectory()
