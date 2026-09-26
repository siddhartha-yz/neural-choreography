# Numerical model extracted from episodes/11.10/scene.py; no renderer dependency.
"""D2L 11.10 — Adam versus SGD on the 11.6 narrow valley.

The valley is f(w) = 0.1 w₁² + 2 w₂² + const, the empirical mean of
per-sample losses fᵢ(w) = 0.1(w₁−Aᵢ)² + 2(w₂−Bᵢ)². Offsets are
mean-centered, so E[∇fᵢ] = ∇f = (0.2 w₁, 4 w₂). SGD takes w ← w − η∇fᵢ.
Adam keeps a first moment v and a second moment s, bias-corrects them,
then steps. Numpy is the only arithmetic.
"""

from __future__ import annotations
import numpy as np

W_START = np.array([-5.0, -2.0], dtype=float)
ETA = 0.4
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-06
STEPS_SGD = 5
STEPS_ADAM = 6
CONTOUR_LEVELS = (0.4, 1.0, 2.5, 5.0, 10.5)
SAMPLES = np.array(
    [[-0.4, 0.1], [0.25, -0.08], [0.15, 0.12], [-0.2, -0.06], [0.2, -0.08]], dtype=float
)
SGD_INDEX = np.array([0, 1, 2, 3, 4, 0], dtype=int)


def valley(weights: np.ndarray) -> float:
    """The shared 11.6 valley, ignoring the sample-variance constant."""
    return float(0.1 * weights[0] ** 2 + 2.0 * weights[1] ** 2)


def sample_gradient(weights: np.ndarray, index: int) -> np.ndarray:
    """∇fᵢ(w) = (0.2(w₁−Aᵢ), 4(w₂−Bᵢ))."""
    offset = SAMPLES[index]
    return np.array(
        [0.2 * (weights[0] - offset[0]), 4.0 * (weights[1] - offset[1])], dtype=float
    )


def run_sgd(steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.array(W_START, dtype=float, copy=True)
    path = np.zeros((steps + 1, 2), dtype=float)
    values = np.zeros(steps + 1, dtype=float)
    grads = np.zeros((steps, 2), dtype=float)
    path[0] = weights
    values[0] = valley(weights)
    for index in range(steps):
        grad = sample_gradient(weights, int(SGD_INDEX[index]))
        grads[index] = grad
        weights = weights - ETA * grad
        path[index + 1] = weights
        values[index + 1] = valley(weights)
    return (path, values, grads)


def run_adam(steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.array(W_START, dtype=float, copy=True)
    first_moment = np.zeros(2, dtype=float)
    second_moment = np.zeros(2, dtype=float)
    path = np.zeros((steps + 1, 2), dtype=float)
    values = np.zeros(steps + 1, dtype=float)
    grads = np.zeros((steps, 2), dtype=float)
    path[0] = weights
    values[0] = valley(weights)
    for index in range(steps):
        time_step = index + 1
        grad = sample_gradient(weights, int(SGD_INDEX[index]))
        grads[index] = grad
        first_moment = BETA1 * first_moment + (1.0 - BETA1) * grad
        second_moment = BETA2 * second_moment + (1.0 - BETA2) * (grad * grad)
        first_hat = first_moment / (1.0 - BETA1**time_step)
        second_hat = second_moment / (1.0 - BETA2**time_step)
        weights = weights - ETA * first_hat / (np.sqrt(second_hat) + EPS)
        path[index + 1] = weights
        values[index + 1] = valley(weights)
    return (path, values, grads)


PATH_SGD, F_SGD, GRADS_SGD = run_sgd(STEPS_SGD)
PATH_ADAM, F_ADAM, GRADS_ADAM = run_adam(STEPS_ADAM)
