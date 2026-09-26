# Numerical model from 09.4; renderer colors omitted.
"""D2L 9.4 — forward and backward hidden states concatenated at one t.

Every moving value comes from the tanh bidirectional recurrence below.
Weights are fixed and untrained. The haloed traveler is the concatenated
vector H_t = [→h_t, ←h_t] at one time step, not a token.
"""

from __future__ import annotations
import numpy as np

TOKENS = (
    np.array([[1.0]], dtype=float),
    np.array([[0.5]], dtype=float),
    np.array([[-0.8]], dtype=float),
    np.array([[0.6]], dtype=float),
)
WEIGHT_XH_FORWARD = np.array([[1.1]], dtype=float)
WEIGHT_HH_FORWARD = np.array([[0.4]], dtype=float)
BIAS_FORWARD = np.array([[-0.05]], dtype=float)
WEIGHT_XH_BACKWARD = np.array([[-0.9]], dtype=float)
WEIGHT_HH_BACKWARD = np.array([[0.55]], dtype=float)
BIAS_BACKWARD = np.array([[0.2]], dtype=float)
HIDDEN_START = np.zeros((1, 1), dtype=float)
CONCAT_INDEX = 2


def format_token_label(value: float) -> str:
    """Match the 3.1 minus glyph for negative features."""
    if value < 0:
        return f"−{abs(value):.2f}"
    return f"{value:.2f}"


def forward_hidden_states() -> np.ndarray:
    """Left-to-right pass; each column is one time step."""
    hidden = HIDDEN_START.copy()
    states = []
    for token in TOKENS:
        hidden = np.tanh(
            WEIGHT_XH_FORWARD @ token + WEIGHT_HH_FORWARD @ hidden + BIAS_FORWARD
        )
        states.append(hidden.copy())
    return np.hstack(states)


def backward_hidden_states() -> np.ndarray:
    """Right-to-left pass; each column is one time step."""
    hidden = HIDDEN_START.copy()
    states = [np.zeros((1, 1), dtype=float) for _ in TOKENS]
    for index in range(len(TOKENS) - 1, -1, -1):
        hidden = np.tanh(
            WEIGHT_XH_BACKWARD @ TOKENS[index]
            + WEIGHT_HH_BACKWARD @ hidden
            + BIAS_BACKWARD
        )
        states[index] = hidden.copy()
    return np.hstack(states)


FORWARD_H = forward_hidden_states()
BACKWARD_H = backward_hidden_states()
CONCAT_H = np.vstack((FORWARD_H[:, [CONCAT_INDEX]], BACKWARD_H[:, [CONCAT_INDEX]]))
