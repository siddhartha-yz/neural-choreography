# Renderer-independent numerical model from 14.7
"""D2L 14.7 — word analogy as a parallelogram in 2-D.

Every on-screen coordinate comes from the toy vectors below.
``composed = king − man + woman`` is numpy; embeddings are not trained.
The haloed traveler is that composed point, landing near queen.
"""

from __future__ import annotations
import numpy as np

MAN = np.array([0.3, 0.35], dtype=float)
WOMAN = np.array([1.8, 0.48], dtype=float)
KING = np.array([0.52, 1.62], dtype=float)
QUEEN = np.array([1.84, 1.52], dtype=float)
COMPOSED = KING - MAN + WOMAN
RESIDUAL = COMPOSED - QUEEN
TOKENS = (MAN, WOMAN, KING, QUEEN)
TOKEN_NAMES = ("man", "woman", "king", "queen")


def fmt_pair(value: float) -> str:
    return f"{value:.2f}"
