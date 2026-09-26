# Numerical model extracted from episodes/11.4/scene.py; no renderer dependency.
"""D2L 11.4 — full gradient vs one-sample SGD on the same 2-D bowl.

The bowl is the 11.3 family: the empirical mean of the per-sample losses is
f(w) = w₁² + 2 w₂² + const, so ∇f = (2 w₁, 4 w₂). Each SGD step uses one
sample’s gradient ∇fᵢ, not the mean. Numpy is the only arithmetic.
"""

from __future__ import annotations
import numpy as np

W_START = np.array([-5.0, -2.0], dtype=float)
ETA = 0.1
STEPS_FULL = 5
STEPS_SGD = 8
CONTOUR_LEVELS = (2.0, 6.0, 12.0, 22.0, 33.0)
SAMPLES = np.array(
    [
        [-1.331203, 1.583793],
        [0.205971, -0.940458],
        [-1.303238, 1.177641],
        [0.224042, -0.179648],
        [2.204428, -1.641329],
    ],
    dtype=float,
)
SGD_INDEX = np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=int)


def bowl(weights: np.ndarray) -> float:
    """The shared 11.3 bowl, ignoring the sample-variance constant."""
    return float(weights[0] ** 2 + 2.0 * weights[1] ** 2)


def full_gradient(weights: np.ndarray) -> np.ndarray:
    """∇f(w) = (2 w₁, 4 w₂), equal to the mean of the five ∇fᵢ."""
    return np.array([2.0 * weights[0], 4.0 * weights[1]], dtype=float)


def sample_gradient(weights: np.ndarray, index: int) -> np.ndarray:
    """∇fᵢ(w) for one sample: 2(w − offset) with the 2× scale on w₂."""
    offset = SAMPLES[index]
    return np.array(
        [2.0 * (weights[0] - offset[0]), 4.0 * (weights[1] - offset[1])], dtype=float
    )


def run_full(steps: int) -> tuple[np.ndarray, np.ndarray]:
    weights = np.array(W_START, dtype=float, copy=True)
    path = np.zeros((steps + 1, 2), dtype=float)
    values = np.zeros(steps + 1, dtype=float)
    path[0] = weights
    values[0] = bowl(weights)
    for index in range(steps):
        weights = weights - ETA * full_gradient(weights)
        path[index + 1] = weights
        values[index + 1] = bowl(weights)
    return (path, values)


def run_sgd(steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.array(W_START, dtype=float, copy=True)
    path = np.zeros((steps + 1, 2), dtype=float)
    values = np.zeros(steps + 1, dtype=float)
    grads = np.zeros((steps, 2), dtype=float)
    path[0] = weights
    values[0] = bowl(weights)
    for index in range(steps):
        sample = int(SGD_INDEX[index])
        grad = sample_gradient(weights, sample)
        grads[index] = grad
        weights = weights - ETA * grad
        path[index + 1] = weights
        values[index + 1] = bowl(weights)
    return (path, values, grads)


PATH_FULL, F_FULL = run_full(STEPS_FULL)
PATH_SGD, F_SGD, GRADS_SGD = run_sgd(STEPS_SGD)
