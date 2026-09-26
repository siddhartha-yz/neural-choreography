# Renderer-independent numerical model from 04.5
"""D2L 4.5 — L2 weight decay pulling a high-capacity polynomial smoother.

Every moving value comes from the ridge / penalized-polynomial fit below.
The haloed traveler is held out: it never enters the design matrix.
"""

from __future__ import annotations
import numpy as np

FEATURES = np.array(
    [-2.25, -1.8, -1.35, -0.9, -0.45, -0.05, 1.35, 1.75, 2.15, 2.45], dtype=float
)
TRUE_COEFF_1 = 0.55
TRUE_COEFF_3 = -0.085
NOISE = np.array(
    [0.22, -0.18, 0.2, -0.16, 0.18, -0.19, 0.17, -0.21, 0.15, 0.18], dtype=float
)
TARGETS = TRUE_COEFF_1 * FEATURES + TRUE_COEFF_3 * FEATURES**3 + NOISE
X_HOLD = 0.55
Y_HOLD = 0.29
POLYNOMIAL_DEGREE = 9
LAMBDA_FINAL = 0.08
DESIGN = np.vander(FEATURES, N=POLYNOMIAL_DEGREE + 1, increasing=True)


def ridge_weights(penalty: float) -> np.ndarray:
    """min ||Xw − y||² + λ||w_{1:}||², intercept unpenalized."""
    if penalty <= 1e-12:
        weights, *_ = np.linalg.lstsq(DESIGN, TARGETS, rcond=None)
        return weights
    regularizer = np.eye(POLYNOMIAL_DEGREE + 1)
    regularizer[0, 0] = 0.0
    gram = DESIGN.T @ DESIGN + penalty * regularizer
    return np.linalg.solve(gram, DESIGN.T @ TARGETS)


def polynomial_values(x_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    vandermonde = np.vander(
        np.asarray(x_values, dtype=float).ravel(), N=len(weights), increasing=True
    )
    return vandermonde @ weights


def polynomial_value(x_value: float, weights: np.ndarray) -> float:
    return float(polynomial_values(np.array([x_value], dtype=float), weights)[0])


def weight_norm(weights: np.ndarray) -> float:
    """L2 norm of the penalized coefficients, matching the ridge term."""
    return float(np.linalg.norm(weights[1:]))


def held_out_residual(weights: np.ndarray) -> float:
    return polynomial_value(X_HOLD, weights) - Y_HOLD


def displayed_lambda(unit: float) -> float:
    """Map a 0–1 tracker onto λ so the wiggle dies across the move, not in a jump."""
    if unit <= 1e-08:
        return 0.0
    return LAMBDA_FINAL * (10.0 ** (1.7 * unit) - 1.0) / (10.0**1.7 - 1.0)


WEIGHTS_UNPENALIZED = ridge_weights(0.0)
WEIGHTS_DECAYED = ridge_weights(LAMBDA_FINAL)
