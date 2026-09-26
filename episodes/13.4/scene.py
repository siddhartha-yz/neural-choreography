"""D2L 13.4 — one grid cell lays two shaped anchors; IoU vs a ground-truth box.

Every on-screen box and IoU comes from ``multibox_prior_cell`` and ``box_iou``
below, matching D2L. Nothing is trained. The traveler is the winning anchor.
"""

from __future__ import annotations

import numpy as np
from scripts.episode_intro import show_episode_intro
from manim import (
    BLACK,
    BLUE,
    BLUE_A,
    BLUE_D,
    BLUE_E,
    Circle,
    Create,
    DashedVMobject,
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


# Tiny 6 × 6 image. One cell drops two D2L anchors: (s=0.4, r=1) and (s=0.4, r=4).
IMAGE_N = 6
CELL = (2, 2)
SIZES = (0.4,)
RATIOS = (1.0, 4.0)
# Ground-truth box in [0, 1] corner format (x1, y1, x2, y2), y down.
GROUND_TRUTH = np.array([0.15, 0.15, 0.55, 0.75], dtype=float)


def multibox_prior_cell(
    n: int,
    row: int,
    col: int,
    sizes: tuple[float, ...],
    ratios: tuple[float, ...],
) -> np.ndarray:
    """D2L ``multibox_prior`` for one pixel. Boxes are (n_anchor, 4) in [0, 1]."""
    center_x = (col + 0.5) / n
    center_y = (row + 0.5) / n
    size_array = np.array(sizes, dtype=float)
    ratio_array = np.array(ratios, dtype=float)
    # Square image: in_height / in_width = 1. Combinations (s1, r1), (s1, r2), …
    widths = np.concatenate(
        (size_array * np.sqrt(ratio_array[:1]), size_array[:1] * np.sqrt(ratio_array[1:]))
    )
    heights = np.concatenate(
        (size_array / np.sqrt(ratio_array[:1]), size_array[:1] / np.sqrt(ratio_array[1:]))
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
assert np.allclose(IOU, np.array([0.50, 0.25]))
assert WINNER == 0


def fmt_iou(value: float) -> str:
    """Two-decimal IoU glyph from the same numpy value drawn on screen."""
    return f"{float(value):.2f}"


class Episode134(Scene):
    """A ~28-second continuous, silent D2L 13.4 visualization."""

    n = IMAGE_N
    cell = 0.72
    grid_left = -2.55
    grid_top = 2.48

    def image_to_scene(self, x_value: float, y_value: float) -> np.ndarray:
        """Map image corners in [0, 1], y down, onto the stroked lattice."""
        scene_x = self.grid_left + x_value * self.n * self.cell
        scene_y = self.grid_top - y_value * self.n * self.cell
        return np.array([scene_x, scene_y, 0.0])

    def cell_center(self, row: int, col: int) -> np.ndarray:
        return self.image_to_scene((col + 0.5) / self.n, (row + 0.5) / self.n)

    def stroke_rect(
        self,
        box: np.ndarray,
        color,
        width: float = 2.4,
        opacity: float = 0.94,
        dashed: bool = False,
    ) -> VGroup:
        """Four stroked sides. Never fill — filled boxes read as white slabs."""
        x1, y1, x2, y2 = (float(value) for value in box)
        top_left = self.image_to_scene(x1, y1)
        top_right = self.image_to_scene(x2, y1)
        bottom_right = self.image_to_scene(x2, y2)
        bottom_left = self.image_to_scene(x1, y2)
        sides = [
            Line(top_left, top_right),
            Line(top_right, bottom_right),
            Line(bottom_right, bottom_left),
            Line(bottom_left, top_left),
        ]
        if dashed:
            sides = [DashedVMobject(side, num_dashes=18) for side in sides]
        group = VGroup(*sides)
        group.set_stroke(color, width=width, opacity=opacity)
        group.set_fill(opacity=0.0)
        return group

    def make_halo(self, center: np.ndarray) -> tuple[Circle, Circle]:
        outer = Circle(radius=0.28).move_to(center).set_stroke(BLUE_A, width=2.1, opacity=0.50)
        inner = Circle(radius=0.18).move_to(center).set_stroke(YELLOW, width=2.8, opacity=0.98)
        return outer, inner

    def construct(self) -> None:
        print(
            "D2L 13.4 A={}\ny={}\nIoU={}\ninter={}\nwinner=A{}".format(
                np.array2string(ANCHORS, precision=4, separator=", "),
                np.array2string(GROUND_TRUTH, precision=2, separator=", "),
                np.array2string(IOU, precision=2, separator=", "),
                np.array2string(INTER_BOXES, precision=4, separator=", "),
                WINNER + 1,
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "13.4")

        grid_cells = []
        for row in range(self.n):
            for col in range(self.n):
                cell_box = np.array(
                    [col / self.n, row / self.n, (col + 1) / self.n, (row + 1) / self.n],
                    dtype=float,
                )
                grid_cells.append(self.stroke_rect(cell_box, BLUE_E, width=1.2, opacity=0.55))
        grid = VGroup(*grid_cells)

        cell_box = np.array(
            [CELL[1] / self.n, CELL[0] / self.n, (CELL[1] + 1) / self.n, (CELL[0] + 1) / self.n],
            dtype=float,
        )
        cell_window = self.stroke_rect(cell_box, YELLOW, width=3.6, opacity=0.98)

        a1_rect = self.stroke_rect(ANCHORS[0], BLUE, width=3.2, opacity=0.96)
        a2_rect = self.stroke_rect(ANCHORS[1], BLUE_D, width=3.4, opacity=0.96)
        y_rect = self.stroke_rect(GROUND_TRUTH, WHITE, width=3.2, opacity=0.98)
        inter1 = self.stroke_rect(INTER_BOXES[0], YELLOW_A, width=2.6, opacity=0.98, dashed=True)
        inter2 = self.stroke_rect(INTER_BOXES[1], YELLOW_A, width=2.6, opacity=0.98, dashed=True)

        a1_corner = self.image_to_scene(float(ANCHORS[0, 0]), float(ANCHORS[0, 1]))
        a2_left = self.image_to_scene(float(ANCHORS[1, 0]), float((ANCHORS[1, 1] + ANCHORS[1, 3]) / 2.0))
        a2_right = self.image_to_scene(float(ANCHORS[1, 2]), float((ANCHORS[1, 1] + ANCHORS[1, 3]) / 2.0))
        y_top_right = self.image_to_scene(float(GROUND_TRUTH[2]), float(GROUND_TRUTH[1]))
        a1_name = Text("A₁", font_size=26, color=BLUE).move_to(a1_corner + np.array([-0.54, 0.34, 0.0]))
        a2_name = Text("A₂", font_size=26, color=BLUE_D).move_to(a2_left + np.array([-0.48, 0.0, 0.0]))
        y_name = Text("y", font_size=26, color=WHITE).move_to(y_top_right + np.array([0.32, 0.28, 0.0]))

        # IoU glyphs sit next to the boxes, not in a vector HUD.
        iou1_label = Text(fmt_iou(IOU[0]), font_size=28, color=WHITE).move_to(
            a1_corner + np.array([-0.54, -0.22, 0.0])
        )
        iou2_label = Text(fmt_iou(IOU[1]), font_size=28, color=WHITE).move_to(
            a2_right + np.array([0.48, 0.0, 0.0])
        )

        formula = Text("IoU = |A ∩ y| / |A ∪ y|", font_size=32, color=WHITE).move_to(
            np.array([0.0, 3.42, 0.0])
        )

        # Halo on the winning box, away from the generating cell and the 0.50 glyph.
        winner_center = self.image_to_scene(
            float(ANCHORS[0, 2]) - 0.07,
            float(ANCHORS[0, 3]) - 0.02,
        )
        halo_outer, halo_inner = self.make_halo(winner_center)

        # 0.40–4.20 s: empty lattice first, held before any anchor.
        self.play(
            LaggedStart(*[Create(cell) for cell in grid_cells], lag_ratio=0.012),
            run_time=0.90,
            rate_func=linear,
        )
        self.wait(2.90)

        # 4.20–6.40 s: one cell is the site that will drop the anchors.
        self.play(Create(cell_window), run_time=0.70, rate_func=smooth)
        self.wait(1.50)

        # 6.40–8.00 s: two shapes from that cell. Both are complete by t = 8.
        self.play(
            Create(a1_rect),
            FadeIn(a1_name),
            run_time=0.70,
            rate_func=smooth,
        )
        self.play(
            Create(a2_rect),
            FadeIn(a2_name),
            run_time=0.70,
            rate_func=smooth,
        )
        self.wait(2.00)

        # Ground-truth box, then the IoU formula once.
        self.play(Create(y_rect), FadeIn(y_name), run_time=1.00, rate_func=smooth)
        self.wait(1.80)
        self.play(FadeIn(formula), run_time=0.70, rate_func=smooth)
        self.wait(0.40)

        # A₁ ∩ y, then the IoU number on that box.
        self.play(Create(inter1), FadeIn(iou1_label), run_time=1.10, rate_func=smooth)
        self.wait(0.90)

        # A₂ ∩ y. Same formula, second number.
        self.play(Create(inter2), FadeIn(iou2_label), run_time=1.10, rate_func=smooth)
        self.wait(0.80)

        # Better IoU lights up. Halo traveler is the winning anchor, not y.
        self.play(FadeOut(inter1), FadeOut(inter2), run_time=0.35, rate_func=linear)
        self.play(
            a1_rect.animate.set_stroke(YELLOW, width=4.2, opacity=1.0),
            a1_name.animate.set_color(YELLOW),
            iou1_label.animate.set_color(YELLOW),
            Create(halo_outer),
            Create(halo_inner),
            Flash(winner_center, color=YELLOW, flash_radius=0.32, line_length=0.08),
            run_time=1.20,
            rate_func=smooth,
        )

        # Hold grid, both anchors, y, both IoUs, and the traveler through the last frame.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=8.40, rate_func=linear)
