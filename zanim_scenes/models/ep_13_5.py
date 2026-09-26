# Renderer-independent numerical model from 13.5
"""D2L 13.5 — a fine map watches a small object; a coarse map's large anchor misses.

Every box and IoU comes from the two-scale ``multibox_prior`` below.  Nothing
is trained.  The haloed traveler is the small object.  Both grids stay up.
"""

from __future__ import annotations
import numpy as np

FINE_H, FINE_W = (4, 4)
COARSE_H, COARSE_W = (2, 2)
S_FINE = 0.15
S_COARSE = 0.4
HIT_FINE = (1, 2)
OBJECT_HALF = 0.06
NODE_RADIUS = 0.078


def feature_centers(fmap_h: int, fmap_w: int) -> np.ndarray:
    """D2L: center of cell (i, j) is ((j+0.5)/w, (i+0.5)/h), y down."""
    rows, cols = np.meshgrid(np.arange(fmap_h), np.arange(fmap_w), indexing="ij")
    center_x = (cols + 0.5) / fmap_w
    center_y = (rows + 0.5) / fmap_h
    return np.stack((center_x, center_y), axis=-1)


def square_anchors(fmap_h: int, fmap_w: int, size: float) -> np.ndarray:
    """One square anchor per cell, ratio 1. Shape (h, w, 4) in [0, 1]."""
    centers = feature_centers(fmap_h, fmap_w)
    half = size / 2.0
    return np.stack(
        (
            centers[..., 0] - half,
            centers[..., 1] - half,
            centers[..., 0] + half,
            centers[..., 1] + half,
        ),
        axis=-1,
    )


def box_area(boxes: np.ndarray) -> np.ndarray:
    flat = boxes.reshape(-1, 4)
    return ((flat[:, 2] - flat[:, 0]) * (flat[:, 3] - flat[:, 1])).reshape(
        boxes.shape[:-1]
    )


def box_iou(boxes: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """IoU of each box against one ground-truth box. Numpy only."""
    flat = boxes.reshape(-1, 4)
    truth = np.asarray(truth, dtype=float).reshape(1, 4)
    inter_ul = np.maximum(flat[:, :2], truth[:, :2])
    inter_lr = np.minimum(flat[:, 2:], truth[:, 2:])
    inter_wh = np.clip(inter_lr - inter_ul, 0.0, None)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    union = box_area(flat) + box_area(truth).reshape(-1) - inter
    return (inter / union).reshape(boxes.shape[:-1])


FINE_ANCHORS = square_anchors(FINE_H, FINE_W, S_FINE)
COARSE_ANCHORS = square_anchors(COARSE_H, COARSE_W, S_COARSE)
FINE_CENTERS = feature_centers(FINE_H, FINE_W)
HIT_CENTER = FINE_CENTERS[HIT_FINE]
GROUND_TRUTH = np.array(
    [
        float(HIT_CENTER[0] - OBJECT_HALF),
        float(HIT_CENTER[1] - OBJECT_HALF),
        float(HIT_CENTER[0] + OBJECT_HALF),
        float(HIT_CENTER[1] + OBJECT_HALF),
    ],
    dtype=float,
)
FINE_IOU = box_iou(FINE_ANCHORS, GROUND_TRUTH)
COARSE_IOU = box_iou(COARSE_ANCHORS, GROUND_TRUTH)
HIT_COARSE = (HIT_FINE[0] // 2, HIT_FINE[1] // 2)
assert FINE_ANCHORS.shape == (4, 4, 4)
assert COARSE_ANCHORS.shape == (2, 2, 4)
assert np.allclose(FINE_IOU[HIT_FINE], 0.64)
assert np.allclose(COARSE_IOU[HIT_COARSE], 0.09)
assert int(np.count_nonzero(FINE_IOU > 1e-12)) == 1
assert int(np.count_nonzero(COARSE_IOU > 1e-12)) == 1
