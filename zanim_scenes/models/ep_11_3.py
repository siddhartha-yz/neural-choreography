# Numerical model extracted from episodes/11.3/scene.py; no renderer dependency.
"""D2L 11.3 — gradient descent on a 2-D elliptical bowl.

Every moving value comes from the numpy path below. The traveler is the
parameter point w. Contours are stroke-only level sets of
f(w) = w₁² + 2 w₂². Updates are exactly w ← w − η ∇f.
"""

from __future__ import annotations
import numpy as np

W_START = np.array([-5.0, -2.0], dtype=float)
ETA_GOOD = 0.1
ETA_LARGE = 0.4
STEPS_GOOD = 8
STEPS_LARGE = 5
CONTOUR_LEVELS = (2.0, 6.0, 12.0, 22.0, 33.0)


def objective(weights: np.ndarray) -> float:
    """f(w) = w₁² + 2 w₂²."""
    return float(weights[0] ** 2 + 2.0 * weights[1] ** 2)


def gradient(weights: np.ndarray) -> np.ndarray:
    """∇f(w) = (2 w₁, 4 w₂)."""
    return np.array([2.0 * weights[0], 4.0 * weights[1]], dtype=float)


def gradient_descent(
    start: np.ndarray, eta: float, steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps+1) parameter points and the matching f values."""
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


PATH_GOOD, F_GOOD = gradient_descent(W_START, ETA_GOOD, STEPS_GOOD)
PATH_LARGE, F_LARGE = gradient_descent(W_START, ETA_LARGE, STEPS_LARGE)
