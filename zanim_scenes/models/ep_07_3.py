# Numerical model from episodes/07.3/scene.py; no renderer dependency.
"""D2L 7.3 — 1×1 convolution as a per-pixel MLP, then GAP to a class vector.

Every on-screen number comes from the arrays, ``conv1x1``, and
``global_avg_pool`` below. This is forward computation only: nothing is trained.
"""

from __future__ import annotations
import numpy as np

INPUT_X = np.array([[[1.0, 3.0], [2.0, 4.0]], [[0.0, 1.0], [2.0, 3.0]]], dtype=float)
WEIGHT_1X1 = np.array([[1.0, 1.0], [2.0, 0.0], [0.0, 2.0]], dtype=float)
TRAVELER = (0, 0)


def conv1x1(inputs: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Channel mix at every pixel. Spatial size is unchanged."""
    in_channels, height, width = inputs.shape
    mixed = weight @ inputs.reshape(in_channels, height * width)
    return mixed.reshape(weight.shape[0], height, width)


def global_avg_pool(feature_maps: np.ndarray) -> np.ndarray:
    """Mean over H and W for each channel. Numpy is the only arithmetic."""
    return feature_maps.mean(axis=(1, 2))


FEATURES = conv1x1(INPUT_X, WEIGHT_1X1)
LOGITS = global_avg_pool(FEATURES)
assert FEATURES.shape[1:] == INPUT_X.shape[1:]
assert FEATURES.shape[0] == WEIGHT_1X1.shape[0]
assert LOGITS.shape == (WEIGHT_1X1.shape[0],)
assert np.allclose(FEATURES, (WEIGHT_1X1 @ INPUT_X.reshape(2, 4)).reshape(3, 2, 2))
assert np.allclose(LOGITS, np.array([4.0, 5.0, 3.0]))


def fmt_int(value: float) -> str:
    """Integer glyph for a cell; use a true minus if a later weight needs it."""
    rounded = int(round(value))
    if rounded < 0:
        return f"−{abs(rounded)}"
    return str(rounded)
