# Numerical model from 09.3; renderer colors omitted.
"""D2L 9.3 — a hidden state handed both to the next time and to the next layer.

Every moving value comes from the two-layer tanh recurrence below.  Weights
are fixed and untrained.  Each layer has one hidden unit.  The haloed traveler
is that unit in the upper layer, unrolled across three time steps.
"""

from __future__ import annotations
import numpy as np

INPUTS = np.array([[1.0], [-0.8], [0.6]], dtype=float)
W_XH_1 = np.array([[0.85]], dtype=float)
W_HH_1 = np.array([[0.5]], dtype=float)
B_1 = np.array([[0.1]], dtype=float)
W_XH_2 = np.array([[0.9]], dtype=float)
W_HH_2 = np.array([[0.4]], dtype=float)
B_2 = np.array([[-0.08]], dtype=float)


def deep_rnn_layers() -> tuple[np.ndarray, np.ndarray]:
    """Return hidden trajectories including t=0 zeros. Shape (4, 1)."""
    hidden_1 = np.zeros((1, 1), dtype=float)
    hidden_2 = np.zeros((1, 1), dtype=float)
    layer_1 = [hidden_1.copy()]
    layer_2 = [hidden_2.copy()]
    for token in INPUTS:
        hidden_1 = np.tanh(W_XH_1 @ token + W_HH_1 @ hidden_1 + B_1)
        hidden_2 = np.tanh(W_XH_2 @ hidden_1 + W_HH_2 @ hidden_2 + B_2)
        layer_1.append(hidden_1.copy())
        layer_2.append(hidden_2.copy())
    return (np.stack(layer_1), np.stack(layer_2))


LAYER_1, LAYER_2 = deep_rnn_layers()
