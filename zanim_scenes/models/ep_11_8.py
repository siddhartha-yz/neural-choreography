# Numerical model extracted from episodes/11.8/scene.py; no renderer dependency.
"""D2L 11.8 — RMSProp keeps walking where AdaGrad's sum stalls.

Both trajectories use the same fixed narrow quadratic valley and start point.
Every displayed metric, segment, and traveler position is derived from the
NumPy update paths below; neither optimizer is trained in the animation.
"""

from __future__ import annotations
import numpy as np

W_START = np.array([-5.0, -2.0], dtype=float)
ETA = 0.3
GAMMA = 0.9
EPSILON = 1e-08
ADAGRAD_STEPS = 12
RMSPROP_STEPS = 12
CONTOUR_LEVELS = (0.4, 1.0, 2.5, 5.0, 10.5)


def objective(weights: np.ndarray) -> float:
    """f(w) = 0.1 w_1^2 + 2 w_2^2."""
    return float(0.1 * weights[0] ** 2 + 2.0 * weights[1] ** 2)


def gradient(weights: np.ndarray) -> np.ndarray:
    """∇f(w) = (0.2 w_1, 4 w_2)."""
    return np.array([0.2 * weights[0], 4.0 * weights[1]], dtype=float)


def adagrad_descent(
    start: np.ndarray, eta: float, steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return positions, f values, cumulative squares, and x-axis step scales."""
    path = np.zeros((steps + 1, 2), dtype=float)
    values = np.zeros(steps + 1, dtype=float)
    accumulator_history = np.zeros((steps + 1, 2), dtype=float)
    x_step_scales = np.zeros(steps + 1, dtype=float)
    weights = np.array(start, dtype=float, copy=True)
    accumulator = np.zeros(2, dtype=float)
    path[0], values[0] = (weights, objective(weights))
    for index in range(steps):
        current_gradient = gradient(weights)
        accumulator += current_gradient**2
        weights = weights - eta * current_gradient / (np.sqrt(accumulator) + EPSILON)
        path[index + 1], values[index + 1] = (weights, objective(weights))
        accumulator_history[index + 1] = accumulator
        x_step_scales[index + 1] = eta / np.sqrt(accumulator[0])
    return (path, values, accumulator_history, x_step_scales)


def rmsprop_descent(
    start: np.ndarray, eta: float, gamma: float, steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return positions, f values, EMA squares, and x-axis step scales."""
    path = np.zeros((steps + 1, 2), dtype=float)
    values = np.zeros(steps + 1, dtype=float)
    square_history = np.zeros((steps + 1, 2), dtype=float)
    x_step_scales = np.zeros(steps + 1, dtype=float)
    weights = np.array(start, dtype=float, copy=True)
    square_average = np.zeros(2, dtype=float)
    path[0], values[0] = (weights, objective(weights))
    for index in range(steps):
        current_gradient = gradient(weights)
        square_average = gamma * square_average + (1.0 - gamma) * current_gradient**2
        weights = weights - eta * current_gradient / (np.sqrt(square_average) + EPSILON)
        path[index + 1], values[index + 1] = (weights, objective(weights))
        square_history[index + 1] = square_average
        x_step_scales[index + 1] = eta / np.sqrt(square_average[0])
    return (path, values, square_history, x_step_scales)


PATH_ADA, F_ADA, S_ADA, ALPHA_ADA = adagrad_descent(W_START, ETA, ADAGRAD_STEPS)
PATH_RMS, F_RMS, S_RMS, ALPHA_RMS = rmsprop_descent(W_START, ETA, GAMMA, RMSPROP_STEPS)
