# Renderer-independent numerical model from 13.4
"""D2L 13.4 — one grid cell lays two shaped anchors; IoU vs a ground-truth box.

Every on-screen box and IoU comes from ``multibox_prior_cell`` and ``box_iou``
below, matching D2L. Nothing is trained. The traveler is the winning anchor.
"""

from __future__ import annotations
import numpy as np

IMAGE_N = 6
CELL = (2, 2)
SIZES = (0.4,)
RATIOS = (1.0, 4.0)
GROUND_TRUTH = np.array([0.15, 0.15, 0.55, 0.75], dtype=float)


def multibox_prior_cell(
    n: int, row: int, col: int, sizes: tuple[float, ...], ratios: tuple[float, ...]
) -> np.ndarray:
    """D2L ``multibox_prior`` for one pixel. Boxes are (n_anchor, 4) in [0, 1]."""
    center_x = (col + 0.5) / n
    center_y = (row + 0.5) / n
    size_array = np.array(sizes, dtype=float)
    ratio_array = np.array(ratios, dtype=float)
    widths = np.concatenate(
        (
            size_array * np.sqrt(ratio_array[:1]),
            size_array[:1] * np.sqrt(ratio_array[1:]),
        )
    )
    heights = np.concatenate(
        (
            size_array / np.sqrt(ratio_array[:1]),
            size_array[:1] / np.sqrt(ratio_array[1:]),
        )
    )
    return np.stack(
        [
            center_x - widths / 2.0,
            center_y - heights / 2.0,
            center_x + widths / 2.0,
            center_y + heights / 2.0,
        ],
        axis=1,
    )


def box_area(boxes: np.ndarray) -> np.ndarray:
    """D2L area: (x2 − x1) * (y2 − y1)."""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Pairwise IoU = |A ∩ B| / |A ∪ B|. Numpy is the only arithmetic."""
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    inter_upper_left = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lower_right = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inter_wh = np.clip(inter_lower_right - inter_upper_left, 0.0, None)
    inter_areas = inter_wh[:, :, 0] * inter_wh[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


def intersection_box(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    """Corner box of A ∩ B, for the dashed overlap on screen."""
    upper_left = np.maximum(box_a[:2], box_b[:2])
    lower_right = np.minimum(box_a[2:], box_b[2:])
    return np.concatenate([upper_left, lower_right])


ANCHORS = multibox_prior_cell(IMAGE_N, CELL[0], CELL[1], SIZES, RATIOS)
IOU = box_iou(ANCHORS, GROUND_TRUTH.reshape(1, 4)).reshape(-1)
INTER_BOXES = np.stack([intersection_box(anchor, GROUND_TRUTH) for anchor in ANCHORS])
WINNER = int(np.argmax(IOU))
assert ANCHORS.shape == (2, 4)
assert np.allclose(IOU, np.array([0.5, 0.25]))
assert WINNER == 0


def fmt_iou(value: float) -> str:
    """Two-decimal IoU glyph from the same numpy value drawn on screen."""
    return f"{float(value):.2f}"
