"""D2L 13.5 — a fine map watches a small object; a coarse map's large anchor misses.

Every box and IoU comes from the two-scale ``multibox_prior`` below.  Nothing
is trained.  The haloed traveler is the small object.  Both grids stay up.
"""

from __future__ import annotations

import numpy as np
from scripts.episode_intro import show_episode_intro
from manim import (
    BLACK,
    BLUE,
    BLUE_A,
    BLUE_E,
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Flash,
    LaggedStart,
    Line,
    Scene,
    Text,
    VGroup,
    ValueTracker,
    WHITE,
    YELLOW,
    YELLOW_A,
    config,
    linear,
    smooth,
)


config.background_color = BLACK
config.renderer = "cairo"


# ---------------------------------------------------------------------------
# One numerical source of truth: D2L multibox_prior on two feature maps.
# Fine   4×4, s=0.15, a=1 (square).  Coarse 2×2, s=0.40, a=1.
# Image is the unit square.  The small ground-truth box sits on one fine cell.
# ---------------------------------------------------------------------------
FINE_H, FINE_W = 4, 4
COARSE_H, COARSE_W = 2, 2
S_FINE = 0.15
S_COARSE = 0.40
# Fine cell (row=1, col=2).  Coarse cell is (row//2, col//2) = (0, 1).
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
    return ((flat[:, 2] - flat[:, 0]) * (flat[:, 3] - flat[:, 1])).reshape(boxes.shape[:-1])


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


class Episode135(Scene):
    """One continuous silent D2L 13.5 visualization, about 32 seconds."""

    grid_left = -2.70
    grid_top = 2.48
    extent = 5.20
    node_radius = NODE_RADIUS

    def image_to_scene(self, x_value: float, y_value: float) -> np.ndarray:
        """Map image corners in [0, 1], y down, onto the stroked square."""
        scene_x = self.grid_left + x_value * self.extent
        scene_y = self.grid_top - y_value * self.extent
        return np.array([scene_x, scene_y, 0.0])

    def stroke_rect(
        self,
        box: np.ndarray,
        color,
        width: float = 2.4,
        opacity: float = 0.94,
    ) -> VGroup:
        """Four stroked sides. Never fill."""
        x1, y1, x2, y2 = (float(value) for value in box)
        top_left = self.image_to_scene(x1, y1)
        top_right = self.image_to_scene(x2, y1)
        bottom_right = self.image_to_scene(x2, y2)
        bottom_left = self.image_to_scene(x1, y2)
        group = VGroup(
            Line(top_left, top_right),
            Line(top_right, bottom_right),
            Line(bottom_right, bottom_left),
            Line(bottom_left, top_left),
        )
        group.set_stroke(color, width=width, opacity=opacity)
        group.set_fill(opacity=0.0)
        return group

    def make_value_dot(self, center: np.ndarray, color, opacity: float = 1.0) -> Dot:
        """Fixed-radius disk. Never scaled."""
        return (
            Dot(center, radius=self.node_radius, color=color)
            .set_fill(color, opacity=opacity)
            .set_stroke(color, width=0.0, opacity=0.0)
        )

    def make_halo(self, center: np.ndarray) -> tuple[Circle, Circle]:
        outer = Circle(radius=0.22).move_to(center).set_stroke(BLUE_A, width=2.1, opacity=0.50)
        inner = Circle(radius=0.14).move_to(center).set_stroke(YELLOW, width=2.8, opacity=0.98)
        return outer, inner

    def construct(self) -> None:
        print(
            "D2L 13.5 fine fmap={}x{} s={} a=1 n_anchor={}".format(
                FINE_H, FINE_W, S_FINE, FINE_H * FINE_W
            )
        )
        print(
            "D2L 13.5 coarse fmap={}x{} s={} a=1 n_anchor={}".format(
                COARSE_H, COARSE_W, S_COARSE, COARSE_H * COARSE_W
            )
        )
        print(
            "D2L 13.5 y={}".format(np.array2string(GROUND_TRUTH, precision=3, separator=", "))
        )
        print(
            "D2L 13.5 fine_iou[4x4]={}".format(
                np.array2string(FINE_IOU, precision=2, separator=", ")
            )
        )
        print(
            "D2L 13.5 coarse_iou[2x2]={}".format(
                np.array2string(COARSE_IOU, precision=2, separator=", ")
            )
        )
        print(
            "D2L 13.5 hit_cell={} iou={:.2f}  miss_cell={} iou={:.2f}".format(
                HIT_FINE,
                float(FINE_IOU[HIT_FINE]),
                HIT_COARSE,
                float(COARSE_IOU[HIT_COARSE]),
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "13.5")

        frame = self.stroke_rect(np.array([0.0, 0.0, 1.0, 1.0]), BLUE, width=2.4, opacity=0.92)

        fine_cells = []
        for row in range(FINE_H):
            for col in range(FINE_W):
                cell_box = np.array(
                    [col / FINE_W, row / FINE_H, (col + 1) / FINE_W, (row + 1) / FINE_H],
                    dtype=float,
                )
                fine_cells.append(self.stroke_rect(cell_box, BLUE_E, width=1.2, opacity=0.55))

        coarse_cells = []
        for row in range(COARSE_H):
            for col in range(COARSE_W):
                cell_box = np.array(
                    [
                        col / COARSE_W,
                        row / COARSE_H,
                        (col + 1) / COARSE_W,
                        (row + 1) / COARSE_H,
                    ],
                    dtype=float,
                )
                coarse_cells.append(self.stroke_rect(cell_box, YELLOW_A, width=2.8, opacity=0.88))

        fine_name = Text("4×4", font_size=26, color=BLUE_A).move_to(
            self.image_to_scene(0.0, 0.0) + np.array([-0.55, 0.22, 0.0])
        )
        coarse_name = Text("2×2", font_size=26, color=YELLOW_A).move_to(
            self.image_to_scene(0.0, 1.0) + np.array([-0.55, -0.22, 0.0])
        )

        formula = Text("4×4, s=0.15      2×2, s=0.40", font_size=32, color=WHITE).move_to(
            np.array([0.00, 3.32, 0.0])
        )

        object_center = self.image_to_scene(float(HIT_CENTER[0]), float(HIT_CENTER[1]))
        traveler = self.make_value_dot(object_center, YELLOW)
        halo_outer, halo_inner = self.make_halo(object_center)
        y_rect = self.stroke_rect(GROUND_TRUTH, WHITE, width=2.6, opacity=0.98)
        y_name = Text("y", font_size=24, color=WHITE).move_to(
            self.image_to_scene(float(GROUND_TRUTH[2]), float(GROUND_TRUTH[1]))
            + np.array([0.34, 0.08, 0.0])
        )

        fine_boxes = [
            self.stroke_rect(FINE_ANCHORS[row, col], BLUE, width=1.7, opacity=0.50)
            for row in range(FINE_H)
            for col in range(FINE_W)
        ]
        coarse_boxes = [
            self.stroke_rect(COARSE_ANCHORS[row, col], BLUE_A, width=2.0, opacity=0.55)
            for row in range(COARSE_H)
            for col in range(COARSE_W)
        ]

        hit_box = FINE_ANCHORS[HIT_FINE]
        miss_box = COARSE_ANCHORS[HIT_COARSE]
        hit_iou = Text("0.64", font_size=26, color=YELLOW).move_to(
            self.image_to_scene(float(hit_box[0]), float(hit_box[1]))
            + np.array([-0.40, 0.22, 0.0])
        )
        miss_iou = Text("0.09", font_size=26, color=BLUE_A).move_to(
            self.image_to_scene(float(miss_box[2]), float(miss_box[3]))
            + np.array([0.36, -0.22, 0.0])
        )

        # 0.40–2.50 s: one image, the small object, halo traveler.
        self.play(
            FadeIn(frame),
            FadeIn(y_rect),
            FadeIn(traveler),
            FadeIn(halo_outer),
            FadeIn(halo_inner),
            FadeIn(y_name),
            run_time=0.55,
            rate_func=linear,
        )
        self.wait(1.55)

        # 2.50–4.70 s: both scales on that image, still empty of anchors.
        self.play(
            FadeIn(fine_name),
            FadeIn(coarse_name),
            LaggedStart(*[Create(cell) for cell in fine_cells], lag_ratio=0.03),
            LaggedStart(*[Create(cell) for cell in coarse_cells], lag_ratio=0.10),
            run_time=1.10,
            rate_func=linear,
        )
        self.wait(1.10)

        # 4.70–6.80 s: shape once, in the empty band above the image.
        self.play(FadeIn(formula), run_time=0.70, rate_func=smooth)
        self.wait(1.40)

        # 6.80–9.00 s: every fine cell lays a small square anchor.
        self.play(FadeIn(VGroup(*fine_boxes)), run_time=0.90, rate_func=smooth)
        self.wait(1.30)

        # 9.00–11.20 s: the small object is a hit on the fine map.
        self.play(
            fine_boxes[HIT_FINE[0] * FINE_W + HIT_FINE[1]].animate.set_stroke(
                YELLOW, width=3.6, opacity=1.0
            ),
            FadeIn(hit_iou),
            Flash(object_center, color=YELLOW, flash_radius=0.32, line_length=0.08),
            run_time=1.20,
            rate_func=smooth,
        )
        self.wait(1.00)

        # 11.20–13.40 s: every coarse cell lays a larger anchor.
        self.play(FadeIn(VGroup(*coarse_boxes)), run_time=0.90, rate_func=smooth)
        self.wait(1.30)

        # 13.40–15.60 s: the same small object is a miss on the coarse map.
        self.play(
            coarse_boxes[HIT_COARSE[0] * COARSE_W + HIT_COARSE[1]].animate.set_stroke(
                BLUE_A, width=3.6, opacity=1.0
            ),
            FadeIn(miss_iou),
            run_time=1.20,
            rate_func=smooth,
        )
        self.wait(1.00)

        # Keep both scales, both anchor sets, y, and the traveler through the last frame.
        self.add(y_rect, traveler, halo_outer, halo_inner, hit_iou, miss_iou)
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=16.40, rate_func=linear)
