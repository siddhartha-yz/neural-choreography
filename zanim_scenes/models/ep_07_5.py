# Renderer-independent numerical model from 07.5
"""D2L 7.5 — batch norm on a tiny batch of five scalars.

Every moving value comes from the arrays below. Mean and std are the batch
statistics of the same five points drawn on screen. γ and β are fixed (not
trained): identity first, then one visible affine shift.
"""

from __future__ import annotations
import numpy as np

BATCH_X = np.array([0.4, 1.2, 2.0, 2.8, 3.6], dtype=float)
TRAVELER_INDEX = 2
EPS = 0.0
BATCH_MU = float(BATCH_X.mean())
BATCH_VAR = float(((BATCH_X - BATCH_MU) ** 2).mean() + EPS)
BATCH_SIGMA = float(np.sqrt(BATCH_VAR))
X_HAT = (BATCH_X - BATCH_MU) / BATCH_SIGMA
GAMMA_ID = 1.0
BETA_ID = 0.0
GAMMA = 1.5
BETA = 0.4
Y_OUT = GAMMA * X_HAT + BETA
