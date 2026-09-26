# Renderer-independent numerical model from 13.3
"""D2L 13.3 — a box around an object; corners convert to center+size and back.

The arrays and conversion helpers below are the only numerical source of
truth. Image coordinates: origin at the top-left, +x right, +y down.
"""

from __future__ import annotations
import numpy as np

CORNER = np.array([[0.5, 0.3, 2.1, 1.5]], dtype=float)
OBJECT_CENTER = np.array([1.3, 0.9], dtype=float)
OBJECT_RX = 0.66
OBJECT_RY = 0.42
OBJECT_DEG = 28.0


def box_corner_to_center(boxes: np.ndarray) -> np.ndarray:
    """D2L: (x1, y1, x2, y2) → (cx, cy, w, h)."""
    x1, y1, x2, y2 = (boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
    return np.stack(((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1), axis=-1)


def box_center_to_corner(boxes: np.ndarray) -> np.ndarray:
    """D2L: (cx, cy, w, h) → (x1, y1, x2, y2)."""
    cx, cy, w, h = (boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
    return np.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), axis=-1)


CENTER = box_corner_to_center(CORNER)
ROUNDTRIP = box_center_to_corner(CENTER)
X1, Y1, X2, Y2 = (float(CORNER[0, i]) for i in range(4))
CX, CY, WIDTH, HEIGHT = (float(CENTER[0, i]) for i in range(4))


def object_boundary(samples: int = 256) -> np.ndarray:
    """Tilted ellipse in the same image coordinates as the box."""
    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    local = np.stack((OBJECT_RX * np.cos(angles), OBJECT_RY * np.sin(angles)), axis=1)
    radians = np.deg2rad(OBJECT_DEG)
    cosine, sine = (np.cos(radians), np.sin(radians))
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + OBJECT_CENTER
