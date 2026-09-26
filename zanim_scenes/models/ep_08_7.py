# Numerical model from 08.7; renderer colors omitted.
"""D2L 8.7 — unroll one scalar RNN and walk a real gradient back in time.

The input sequence, recurrent state, terminal loss gradient, and truncated
window are all computed from the NumPy arrays below.  This is a fixed
forward/backward computation, not a training animation.
"""

from __future__ import annotations
import numpy as np

INPUTS = np.array([[0.35], [-0.6], [0.75], [0.1], [-0.45], [0.65]], dtype=float)
W_XH = np.array([[0.74]], dtype=float)
W_HH = np.array([[0.7]], dtype=float)
BIAS = np.array([[-0.05]], dtype=float)
TARGET = np.array([[0.2]], dtype=float)
TRUNCATION_STEPS = 3


def forward_states(inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return z_t and h_t for h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)."""
    preactivations = np.zeros_like(inputs)
    hidden = np.zeros((len(inputs) + 1, 1), dtype=float)
    for index, x_t in enumerate(inputs, start=1):
        preactivations[index - 1] = W_XH @ x_t + W_HH @ hidden[index - 1] + BIAS
        hidden[index] = np.tanh(preactivations[index - 1])
    return (preactivations, hidden)


def bptt_gradients(hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return dL/dh_t and dL/dz_t for the same unrolled scalar chain."""
    gradient_h = np.zeros_like(hidden)
    gradient_z = np.zeros_like(hidden)
    gradient_h[-1] = hidden[-1] - TARGET
    for time_index in range(len(hidden) - 1, 0, -1):
        gradient_z[time_index] = gradient_h[time_index] * (
            1.0 - hidden[time_index] ** 2
        )
        gradient_h[time_index - 1] += W_HH @ gradient_z[time_index]
    return (gradient_h[1:], gradient_z[1:])


PREACTIVATIONS, HIDDEN = forward_states(INPUTS)
GRADIENT_H, GRADIENT_Z = bptt_gradients(HIDDEN)
