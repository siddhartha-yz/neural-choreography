# Numerical model extracted from episodes/11.7/scene.py; no renderer dependency.
"""D2L 11.7 — AdaGrad versus vanilla GD on a narrow quadratic valley.

Every moving value comes from the numpy paths below. The traveler is the
parameter point w on f(w) = 0.1 w₁² + 2 w₂². Contours are stroke-only
level sets. Beat 1 is w ← w − η∇f. Beat 2 is s ← s + (∇f)²,
w ← w − η ∇f / √s. Same η; the steep axis is the one 1/√s damps.
"""

from __future__ import annotations
import numpy as np

W_START = np.array([-5.0, -2.0], dtype=float)
ETA = 0.4
EPS = 1e-06
STEPS_GD = 10
STEPS_ADA = 12
CONTOUR_LEVELS = (0.4, 1.0, 2.5, 5.0, 10.5)


def objective(weights: np.ndarray) -> float:
    """f(w) = 0.1 w₁² + 2 w₂²."""
    return float(0.1 * weights[0] ** 2 + 2.0 * weights[1] ** 2)


def gradient(weights: np.ndarray) -> np.ndarray:
    """∇f(w) = (0.2 w₁, 4 w₂)."""
    return np.array([0.2 * weights[0], 4.0 * weights[1]], dtype=float)


def gradient_descent(
    start: np.ndarray, eta: float, steps: int
) -> tuple[np.ndarray, np.ndarray]:
    path = np.zeros((steps + 1, 2), dtype=float)
    values = np.zeros(steps + 1, dtype=float)
    weights = np.array(start, dtype=float, copy=True)
    path[0] = weights
    values[0] = objective(weights)
    for index in range(steps):
        weights = weights - eta * gradient(weights)
        path[index + 1] = weights
        values[index + 1] = objective(weights)
    return (path, values)


def adagrad_descent(
    start: np.ndarray, eta: float, steps: int, eps: float = EPS
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """AdaGrad: s ← s + g⊙g, w ← w − η g / √(s+ε). Also return s and 1/√s."""
    path = np.zeros((steps + 1, 2), dtype=float)
    values = np.zeros(steps + 1, dtype=float)
    accum = np.zeros((steps + 1, 2), dtype=float)
    inv_sqrt = np.zeros((steps + 1, 2), dtype=float)
    weights = np.array(start, dtype=float, copy=True)
    state = np.zeros(2, dtype=float)
    path[0] = weights
    values[0] = objective(weights)
    for index in range(steps):
        grad = gradient(weights)
        state = state + grad * grad
        scale = 1.0 / np.sqrt(state + eps)
        weights = weights - eta * scale * grad
        path[index + 1] = weights
        values[index + 1] = objective(weights)
        accum[index + 1] = state
        inv_sqrt[index + 1] = scale
    return (path, values, accum, inv_sqrt)


PATH_GD, F_GD = gradient_descent(W_START, ETA, STEPS_GD)
PATH_ADA, F_ADA, S_ADA, INV_SQRT_S = adagrad_descent(W_START, ETA, STEPS_ADA)
