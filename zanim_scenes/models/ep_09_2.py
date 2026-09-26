# Numerical model from 09.2; renderer colors omitted.
"""D2L 9.2 — LSTM cell state as a persistent reservoir.

Every moving value comes from the two-step numpy LSTM below.  Forget
scales the standing cell; input adds a gated candidate; the cell stays
in the box while the output gate releases H.  Untrained.  Not a gate
dashboard of weight matrices.
"""

from __future__ import annotations
import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Element-wise logistic sigmoid for the same gates drawn on screen."""
    return 1.0 / (1.0 + np.exp(-values))


def lstm_step(
    x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray
) -> dict[str, np.ndarray]:
    """One untrained LSTM step: F, I, O, candidate, C, H."""
    forget = sigmoid(W_XF @ x_t + W_HF @ h_prev + B_F)
    input_gate = sigmoid(W_XI @ x_t + W_HI @ h_prev + B_I)
    output_gate = sigmoid(W_XO @ x_t + W_HO @ h_prev + B_O)
    candidate = np.tanh(W_XC @ x_t + W_HC @ h_prev + B_C)
    cell = forget * c_prev + input_gate * candidate
    hidden = output_gate * np.tanh(cell)
    return {
        "F": forget,
        "I": input_gate,
        "O": output_gate,
        "C_tilde": candidate,
        "C": cell,
        "H": hidden,
    }


INPUT_X1 = np.array([[1.0], [0.5]], dtype=float)
INPUT_X2 = np.array([[0.4], [1.0]], dtype=float)
HIDDEN_0 = np.array([[0.25], [0.1]], dtype=float)
CELL_0 = np.array([[1.5], [1.1]], dtype=float)
W_XF = np.array([[-3.10613331, 3.4396779], [3.6942479, -2.99404665]], dtype=float)
W_HF = np.array([[0.15, -0.05], [0.08, 0.1]], dtype=float)
B_F = np.array([[-0.0325], [-0.03]], dtype=float)
W_XI = np.array([[2.41950063, -2.06641254], [-2.59930193, 2.42601513]], dtype=float)
W_HI = np.array([[0.1, 0.05], [-0.06, 0.08]], dtype=float)
B_I = np.array([[-0.03], [0.007]], dtype=float)
W_XO = np.array([[0.8437042, 0.50981618], [0.19268835, 1.30921902]], dtype=float)
W_HO = np.array([[0.08, 0.04], [0.05, 0.09]], dtype=float)
B_O = np.array([[-0.024], [-0.0215]], dtype=float)
W_XC = np.array([[1.71356652, -0.48269405], [-1.10436152, 1.69789742]], dtype=float)
W_HC = np.array([[0.12, 0.04], [-0.05, 0.07]], dtype=float)
B_C = np.array([[-0.034], [0.0055]], dtype=float)
STEP1 = lstm_step(INPUT_X1, HIDDEN_0, CELL_0)
STEP2 = lstm_step(INPUT_X2, STEP1["H"], STEP1["C"])


def signed_text(value: float, digits: int = 2) -> str:
    """True minus glyph for negative values, matching 3.1 / 5.1."""
    if value < 0:
        return f"−{abs(value):.{digits}f}"
    return f"{value:.{digits}f}"
