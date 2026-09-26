# Renderer-independent numerical model from 11.2
"""D2L 11.2 — convex chord above the surface vs a non-convex extra pit.

Every moving value comes from the numpy samples below. Left bowl is
f(x) = 0.5 x²; right bowl is g(x) = cos(π x). The traveler sits on the
chord, never on the surface: z = λx + (1−λ)x′.
"""

from __future__ import annotations
import numpy as np


def convex_f(x: np.ndarray | float) -> np.ndarray | float:
    """f(x) = 0.5 x²."""
    return 0.5 * np.asarray(x, dtype=float) ** 2


def nonconvex_g(x: np.ndarray | float) -> np.ndarray | float:
    """g(x) = cos(π x)."""
    return np.cos(np.pi * np.asarray(x, dtype=float))


XA = -1.5
XB = 1.0
LAM = 0.4
Z = float(LAM * XA + (1.0 - LAM) * XB)
X_GRID = np.linspace(-2.0, 2.0, 241)
F_GRID = np.asarray(convex_f(X_GRID), dtype=float)
G_GRID = np.asarray(nonconvex_g(X_GRID), dtype=float)
F_A = float(convex_f(XA))
F_B = float(convex_f(XB))
F_Z = float(convex_f(Z))
F_CHORD = float(LAM * F_A + (1.0 - LAM) * F_B)
G_A = float(nonconvex_g(XA))
G_B = float(nonconvex_g(XB))
G_Z = float(nonconvex_g(Z))
G_CHORD = float(LAM * G_A + (1.0 - LAM) * G_B)
PIT_X = -1.0
G_PIT = float(nonconvex_g(PIT_X))
SAMPLE_X = np.array([XA, PIT_X, Z, XB], dtype=float)


def chord_x(lam: float) -> float:
    return float(lam * XA + (1.0 - lam) * XB)


def chord_f(lam: float) -> float:
    return float(lam * F_A + (1.0 - lam) * F_B)
