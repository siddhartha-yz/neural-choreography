# Numerical model from 09.6; renderer colors omitted.
"""D2L 9.6 — variable-length input compressed to a fixed-shape state.

Every moving value comes from the encoder–decoder arrays below.  Three
source tokens are folded into one 2-vector ``c``; the decoder then emits
two tokens, one at a time, from that same state.  Untrained.  The halo
traveler is ``c``, not a source token.
"""

from __future__ import annotations
import numpy as np

SOURCE = (
    np.array([[1.0], [0.5]], dtype=float),
    np.array([[-0.8], [1.1]], dtype=float),
    np.array([[0.6], [-0.9]], dtype=float),
)
WEIGHT_E = np.array([[0.85, 0.15], [-0.2, 0.9]], dtype=float)
RECUR_E = np.array([[0.55, -0.25], [0.3, 0.6]], dtype=float)
BIAS_E = np.array([[0.05], [-0.08]], dtype=float)
WEIGHT_D = np.array([[0.9, -0.4], [0.2, 0.85]], dtype=float)
RECUR_D = np.array([[0.6, 0.25], [-0.45, 0.5]], dtype=float)
BIAS_D = np.array([[-0.05], [0.12]], dtype=float)
WEIGHT_OUT = np.array([[1.25, -1.1], [0.85, 0.95]], dtype=float)
BIAS_OUT = np.array([[0.15], [-0.25]], dtype=float)
BOS = np.zeros((2, 1), dtype=float)
N_OUT = 2


def encode(tokens: tuple[np.ndarray, ...]) -> list[np.ndarray]:
    """Return [h_0, h_1, …, h_T]; h_0 is zeros, h_T is the fixed-shape state."""
    hidden = np.zeros((2, 1), dtype=float)
    states = [hidden.copy()]
    for token in tokens:
        hidden = np.tanh(WEIGHT_E @ token + RECUR_E @ hidden + BIAS_E)
        states.append(hidden.copy())
    return states


def decode(
    state: np.ndarray, n_out: int = N_OUT
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Autoregressive decoder starting at s_0 = c, y_0 = 0."""
    hidden = state.copy()
    previous = BOS.copy()
    outputs: list[np.ndarray] = []
    decoder_states = [hidden.copy()]
    for _ in range(n_out):
        hidden = np.tanh(WEIGHT_D @ previous + RECUR_D @ hidden + BIAS_D)
        token = WEIGHT_OUT @ hidden + BIAS_OUT
        outputs.append(token.copy())
        decoder_states.append(hidden.copy())
        previous = token
    return (outputs, decoder_states)


ENCODER_STATES = encode(SOURCE)
CONTEXT = ENCODER_STATES[-1]
OUTPUTS, DECODER_STATES = decode(CONTEXT)


def signed_glyphs(value: float, digits: int = 2) -> str:
    """True minus glyph, matching the 3.1 remake."""
    sign = "−" if value < 0 else "+"
    return f"{sign}{abs(value):.{digits}f}"
