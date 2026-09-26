# Numerical model from 09.1; renderer colors omitted.
"""D2L 9.1 — the GRU update gate as a componentwise convex combination.

Only the NumPy tensors below supply the values seen in the scene.  The
animation is a fixed, untrained forward update: no recurrent parameters are
learned and no state is optimized.
"""

from __future__ import annotations
import numpy as np

H_PREV = np.array([[0.6], [-0.4], [0.2]], dtype=float)
Z = np.array([[0.82], [0.25], [0.6]], dtype=float)
H_CANDIDATE = np.array([[-0.2], [0.8], [-0.6]], dtype=float)
H_NEXT = Z * H_PREV + (1.0 - Z) * H_CANDIDATE


def signed(value: float, decimals: int = 2) -> str:
    """Formatting shared by all state values in the scene."""
    return f"{('+' if value >= 0 else '−')}{abs(value):.{decimals}f}"
