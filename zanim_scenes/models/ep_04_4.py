# Renderer-independent numerical model from 04.4
"""D2L 4.4 — capacity vs fit, from two real polynomial least-squares curves.

Every plotted number comes from the arrays below.  A quadratic generates the
noisy train cloud; the traveler is a held-out x with its true y.  The two
curves are ``numpy.polyfit`` of degree 1 and degree 6 — no training loop,
no closed-form overlay besides those two fits.
"""

from __future__ import annotations
import numpy as np

TRAIN_X = np.array([-2.25, -1.5, -0.8, -0.1, 0.55, 1.9, 2.35], dtype=float)
TRUE_INTERCEPT = 1.4
TRUE_QUAD = -0.45
NOISE = np.array([0.18, -0.22, 0.3, -0.2, 0.4, -0.4, 0.2], dtype=float)


def true_function(x_values: np.ndarray | float) -> np.ndarray | float:
    """Noiseless quadratic that generated the cloud and the traveler."""
    return TRUE_INTERCEPT + TRUE_QUAD * np.asarray(x_values, dtype=float) ** 2


TRAIN_Y = np.asarray(true_function(TRAIN_X), dtype=float) + NOISE
TRAVELER_X = 1.2
TRAVELER_Y = float(true_function(TRAVELER_X))
LOW_DEGREE = 1
HIGH_DEGREE = len(TRAIN_X) - 1
LOW_COEFFS = np.polyfit(TRAIN_X, TRAIN_Y, LOW_DEGREE)
HIGH_COEFFS = np.polyfit(TRAIN_X, TRAIN_Y, HIGH_DEGREE)


def mean_squared_error(
    coeffs: np.ndarray, x_values: np.ndarray, y_values: np.ndarray
) -> float:
    """MSE of a polynomial against the same points drawn on screen."""
    residuals = np.polyval(coeffs, x_values) - y_values
    return float(np.mean(residuals**2))


LOW_TRAIN_LOSS = mean_squared_error(LOW_COEFFS, TRAIN_X, TRAIN_Y)
HIGH_TRAIN_LOSS = mean_squared_error(HIGH_COEFFS, TRAIN_X, TRAIN_Y)
LOW_VAL_LOSS = mean_squared_error(
    LOW_COEFFS, np.array([TRAVELER_X]), np.array([TRAVELER_Y])
)
HIGH_VAL_LOSS = mean_squared_error(
    HIGH_COEFFS, np.array([TRAVELER_X]), np.array([TRAVELER_Y])
)
LOW_PREDICTION = float(np.polyval(LOW_COEFFS, TRAVELER_X))
HIGH_PREDICTION = float(np.polyval(HIGH_COEFFS, TRAVELER_X))
HIGH_RESIDUAL = HIGH_PREDICTION - TRAVELER_Y


def signed_glyphs(value: float, digits: int) -> str:
    """Match 3.1: a true minus glyph, never a hyphen-minus."""
    sign = "−" if value < 0 else "+"
    return f"{sign}{abs(value):.{digits}f}"
