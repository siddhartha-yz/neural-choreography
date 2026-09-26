# Numerical model extracted from episodes/11.11/scene.py; no renderer dependency.
"""D2L 11.11 — one descent path whose step size follows a schedule.

Every moving value comes from the numpy arrays below. The traveler is the
parameter point w on the same bowl as 11.3: f(w) = w₁² + 2 w₂². Updates are
exactly w ← w − η(t) ∇f. η(t) is cosine decay after a linear warmup — one
schedule, not an inventory. Contours are stroke-only. Kernels are not trained.
"""

from __future__ import annotations
import numpy as np

W_START = np.array([[-5.0], [-2.0]], dtype=float)
MAX_UPDATE = 10
WARMUP_STEPS = 3
BASE_LR = 0.18
FINAL_LR = 0.02
WARMUP_BEGIN_LR = 0.03
STEPS = 10
CONTOUR_LEVELS = (2.0, 6.0, 12.0, 22.0, 33.0)


def objective(weights: np.ndarray) -> float:
    """f(w) = w₁² + 2 w₂². ``weights`` is a 2 × 1 column."""
    return float(weights[0, 0] ** 2 + 2.0 * weights[1, 0] ** 2)


def gradient(weights: np.ndarray) -> np.ndarray:
    """∇f(w) = (2 w₁, 4 w₂) as a 2 × 1 column."""
    return np.array([[2.0 * weights[0, 0]], [4.0 * weights[1, 0]]], dtype=float)


def eta_of(step: float) -> float:
    """D2L CosineScheduler: linear warmup, then cosine to η_T."""
    time = float(step)
    if time < WARMUP_STEPS:
        return WARMUP_BEGIN_LR + (BASE_LR - WARMUP_BEGIN_LR) * time / WARMUP_STEPS
    if time <= MAX_UPDATE:
        cosine = np.cos(np.pi * (time - WARMUP_STEPS) / (MAX_UPDATE - WARMUP_STEPS))
        return FINAL_LR + (BASE_LR - FINAL_LR) * (1.0 + float(cosine)) / 2.0
    return FINAL_LR


def scheduled_descent(
    start: np.ndarray, steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """w ← w − η(t) ∇f. Returns path (steps+1, 2), f values, and η at each index."""
    path = np.zeros((steps + 1, 2), dtype=float)
    values = np.zeros(steps + 1, dtype=float)
    rates = np.zeros(steps + 1, dtype=float)
    weights = np.array(start, dtype=float, copy=True)
    path[0] = weights[:, 0]
    values[0] = objective(weights)
    rates[0] = eta_of(0.0)
    for index in range(steps):
        weights = weights - eta_of(index) * gradient(weights)
        path[index + 1] = weights[:, 0]
        values[index + 1] = objective(weights)
        rates[index + 1] = eta_of(index + 1)
    return (path, values, rates)


PATH, F_PATH, ETAS = scheduled_descent(W_START, STEPS)
