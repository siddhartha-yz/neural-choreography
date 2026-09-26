# Renderer-independent numerical model from 04.6
"""D2L 4.6 — dropout on the traveler’s hidden units.

Every moving value comes from the arrays below. Hidden activations are a real
ReLU MLP forward pass; masks are real Bernoulli draws. Weights are never
trained. Training uses classic dropout (mask zeros some hᵢ); inference keeps
every unit and scales by the keep probability 1−p.
"""

from __future__ import annotations
import numpy as np

INPUT_X = np.array([[1.0], [0.5]], dtype=float)
WEIGHTS = np.array([[1.2, 0.8], [0.4, 0.2], [0.8, 0.4], [0.5, 0.6]], dtype=float)
BIAS = np.array([[0.2], [0.1], [0.2], [0.1]], dtype=float)
PRE_ACTIVATION = WEIGHTS @ INPUT_X + BIAS
HIDDEN = np.maximum(PRE_ACTIVATION, 0.0)
DROP_P = 0.5
KEEP_P = 1.0 - DROP_P
MASK_SEED = 47
N_MASKS = 4


def sample_masks() -> np.ndarray:
    """Four distinct Bernoulli masks from one seeded generator."""
    rng = np.random.default_rng(MASK_SEED)
    return np.stack(
        [(rng.random(HIDDEN.size) > DROP_P).astype(float) for _ in range(N_MASKS)]
    )


MASKS = sample_masks()
H_TRAIN = MASKS * HIDDEN.ravel()
H_INFER = KEEP_P * HIDDEN.ravel()
