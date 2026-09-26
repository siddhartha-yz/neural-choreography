# Renderer-independent numerical model from 04.7
"""D2L 4.7 — forward then reverse on the same tiny-net path.

The arrays and ``run_graph`` below are the only numerical source of truth.
Forward activations and reverse local gradients are the same numpy chain
rule; a finite-difference check on ∂ℓ/∂x is printed at scene start.
No parameters are trained.
"""

from __future__ import annotations
import numpy as np

INPUT_X = np.array([[1.0], [0.5]], dtype=float)
TARGET_Y = 0.69
WEIGHTS_1 = np.array([[0.8, 0.4], [-0.6, 0.3]], dtype=float)
BIAS_1 = np.array([[0.1], [-0.4]], dtype=float)
WEIGHTS_2 = np.array([[0.9, -0.5]], dtype=float)
BIAS_2 = np.array([[0.2]], dtype=float)


def run_graph(x: np.ndarray, y: float) -> dict[str, np.ndarray | float]:
    """Forward activations, then reverse local gradients, one numpy path."""
    hidden_pre = WEIGHTS_1 @ x + BIAS_1
    relu_mask = (hidden_pre > 0.0).astype(float)
    hidden = hidden_pre * relu_mask
    prediction = float((WEIGHTS_2 @ hidden + BIAS_2).item())
    residual = prediction - y
    loss = residual**2
    d_loss_d_prediction = 2.0 * residual
    d_loss_d_hidden = WEIGHTS_2.T * d_loss_d_prediction
    d_loss_d_pre = d_loss_d_hidden * relu_mask
    d_loss_d_x = WEIGHTS_1.T @ d_loss_d_pre
    return {
        "z": hidden_pre,
        "h": hidden,
        "yhat": prediction,
        "y": y,
        "residual": residual,
        "loss": loss,
        "dL_dyhat": d_loss_d_prediction,
        "dL_dh": d_loss_d_hidden,
        "dL_dz": d_loss_d_pre,
        "dL_dx": d_loss_d_x,
    }


GRAPH = run_graph(INPUT_X, TARGET_Y)


def finite_difference_dL_dx(epsilon: float = 1e-06) -> np.ndarray:
    """Central differences on ℓ(x); must match GRAPH['dL_dx']."""
    gradient = np.zeros_like(INPUT_X)
    for index in range(INPUT_X.shape[0]):
        shifted_plus = INPUT_X.copy()
        shifted_minus = INPUT_X.copy()
        shifted_plus[index, 0] += epsilon
        shifted_minus[index, 0] -= epsilon
        loss_plus = float(run_graph(shifted_plus, TARGET_Y)["loss"])
        loss_minus = float(run_graph(shifted_minus, TARGET_Y)["loss"])
        gradient[index, 0] = (loss_plus - loss_minus) / (2.0 * epsilon)
    return gradient


def signed_text(value: float, digits: int = 3) -> str:
    """ASCII + / Unicode minus, matching the 3.1 pocket glyphs."""
    return f"{value:+.{digits}f}".replace("-", "−")
