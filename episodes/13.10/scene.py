"""D2L 13.10 — transposed convolution: one input cell times K, laid out larger.

Every on-screen number comes from the arrays and ``trans_conv`` below. This is
forward computation only: the kernel is not trained and no bias is added.
The film writes one traveler cell, not the full summed output.
"""

from __future__ import annotations

import numpy as np
from scripts.episode_intro import show_episode_intro
from manim import (
    BLACK,
    BLUE_A,
    BLUE_D,
    Circle,
    Create,
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


# D2L 13.10 book tensors: a real 2 × 2 input and the same 2 × 2 kernel.
INPUT_X = np.array(
    [
        [0.0, 1.0],
        [2.0, 3.0],
    ],
    dtype=float,
)
KERNEL = np.array(
    [
        [0.0, 1.0],
        [2.0, 3.0],
    ],
    dtype=float,
)

# Halo traveler: one input cell. Its scaled kernel is stamped onto Y.
TRAVELER = (1, 1)


def trans_conv(input_x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """D2L ``trans_conv``: each x lays ``x * K`` onto a (n + k − 1) output."""
    kernel_h, kernel_w = kernel.shape
    output = np.zeros(
        (input_x.shape[0] + kernel_h - 1, input_x.shape[1] + kernel_w - 1),
        dtype=float,
    )
    for row in range(input_x.shape[0]):
        for col in range(input_x.shape[1]):
            output[row : row + kernel_h, col : col + kernel_w] += input_x[row, col] * kernel
    return output


OUTPUT = trans_conv(INPUT_X, KERNEL)
KERNEL_H, KERNEL_W = KERNEL.shape
TRAVELER_VALUE = float(INPUT_X[TRAVELER])
PATCH = TRAVELER_VALUE * KERNEL
FILM_Y = np.zeros_like(OUTPUT)
FILM_Y[
    TRAVELER[0] : TRAVELER[0] + KERNEL_H,
    TRAVELER[1] : TRAVELER[1] + KERNEL_W,
] = PATCH


def fmt_int(value: float) -> str:
    """Integer glyph for a cell; use a true minus if a later kernel needs it."""
    rounded = int(round(value))
    if rounded < 0:
        return f"−{abs(rounded)}"
    return str(rounded)


class Episode1310(Scene):
    """A ~31-second continuous, silent D2L 13.10 visualization."""

    # Same cell size on X, K, and Y so the 3 × 3 output is physically larger.
    cell = 1.08
    box_half = 0.46
    x_origin = np.array([-5.05, 0.08, 0.0])
    k_origin = np.array([0.02, 0.08, 0.0])
    y_origin = np.array([5.12, 0.08, 0.0])

    def cell_center(self, origin: np.ndarray, shape: tuple[int, int], row: int, col: int) -> np.ndarray:
        rows, cols = shape
        x_value = origin[0] + (col - (cols - 1) / 2.0) * self.cell
        y_value = origin[1] + ((rows - 1) / 2.0 - row) * self.cell
        return np.array([x_value, y_value, 0.0])

    def stroke_box(self, center: np.ndarray, color, width: float = 2.0, opacity: float = 0.94) -> VGroup:
        """Four stroked sides. Never fill — filled grids read as white slabs."""
        x_value, y_value = center[0], center[1]
        half = self.box_half
        box = VGroup(
            Line(np.array([x_value - half, y_value + half, 0.0]), np.array([x_value + half, y_value + half, 0.0])),
            Line(np.array([x_value + half, y_value + half, 0.0]), np.array([x_value + half, y_value - half, 0.0])),
            Line(np.array([x_value + half, y_value - half, 0.0]), np.array([x_value - half, y_value - half, 0.0])),
            Line(np.array([x_value - half, y_value - half, 0.0]), np.array([x_value - half, y_value + half, 0.0])),
        )
        box.set_stroke(color, width=width, opacity=opacity)
        box.set_fill(opacity=0.0)
        return box

    def make_cells(
        self,
        values: np.ndarray,
        origin: np.ndarray,
        stroke_color,
        font_size: int,
        number_color,
    ) -> tuple[list[list[VGroup]], list[list[Text]]]:
        rows, cols = values.shape
        boxes: list[list[VGroup]] = []
        labels: list[list[Text]] = []
        for row in range(rows):
            box_row: list[VGroup] = []
            label_row: list[Text] = []
            for col in range(cols):
                position = self.cell_center(origin, values.shape, row, col)
                box = self.stroke_box(position, stroke_color)
                label = Text(fmt_int(values[row, col]), font_size=font_size, color=number_color)
                label.move_to(position)
                box_row.append(box)
                label_row.append(label)
            boxes.append(box_row)
            labels.append(label_row)
        return boxes, labels

    def make_empty_grid(self, origin: np.ndarray, shape: tuple[int, int], stroke_color) -> list[list[VGroup]]:
        rows, cols = shape
        boxes: list[list[VGroup]] = []
        for row in range(rows):
            box_row: list[VGroup] = []
            for col in range(cols):
                position = self.cell_center(origin, shape, row, col)
                box_row.append(self.stroke_box(position, stroke_color))
            boxes.append(box_row)
        return boxes

    def make_window(self, centers: list[np.ndarray], pad: float = 0.10) -> VGroup:
        xs = [center[0] for center in centers]
        ys = [center[1] for center in centers]
        half = self.box_half
        x0, x1 = min(xs) - half - pad, max(xs) + half + pad
        y0, y1 = min(ys) - half - pad, max(ys) + half + pad
        box = VGroup(
            Line(np.array([x0, y1, 0.0]), np.array([x1, y1, 0.0])),
            Line(np.array([x1, y1, 0.0]), np.array([x1, y0, 0.0])),
            Line(np.array([x1, y0, 0.0]), np.array([x0, y0, 0.0])),
            Line(np.array([x0, y0, 0.0]), np.array([x0, y1, 0.0])),
        )
        box.set_stroke(YELLOW, width=3.4, opacity=0.98)
        box.set_fill(opacity=0.0)
        return box

    def make_halo(self, center: np.ndarray) -> tuple[Circle, Circle]:
        # Radius stays inside the traveler cell: neighbor box edge is
        # cell - box_half = 0.62 away, so 0.54 does not clip the 2 or the 1.
        outer = Circle(radius=0.54).move_to(center).set_stroke(BLUE_A, width=2.1, opacity=0.50)
        inner = Circle(radius=0.47).move_to(center).set_stroke(YELLOW, width=2.6, opacity=0.98)
        outer.set_fill(opacity=0.0)
        inner.set_fill(opacity=0.0)
        return outer, inner

    def construct(self) -> None:
        assert OUTPUT.shape == (3, 3)
        assert INPUT_X.shape == (2, 2)
        assert KERNEL.shape == (2, 2)
        print(
            "D2L 13.10 X={}\nK={}\nY_full={}\nY.shape={}\ntraveler={} x={:.0f}\n"
            "patch=x*K={}\nY_film={}".format(
                np.array2string(INPUT_X, precision=0, separator=", "),
                np.array2string(KERNEL, precision=0, separator=", "),
                np.array2string(OUTPUT, precision=0, separator=", "),
                OUTPUT.shape,
                TRAVELER,
                TRAVELER_VALUE,
                np.array2string(PATCH, precision=0, separator=", "),
                np.array2string(FILM_Y, precision=0, separator=", "),
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "13.10")

        x_boxes, x_labels = self.make_cells(INPUT_X, self.x_origin, BLUE_D, 32, WHITE)
        k_boxes, k_labels = self.make_cells(KERNEL, self.k_origin, YELLOW_A, 32, YELLOW_A)
        y_shape = OUTPUT.shape
        y_boxes = self.make_empty_grid(self.y_origin, y_shape, BLUE_A)
        y_centers = [
            [self.cell_center(self.y_origin, y_shape, row, col) for col in range(y_shape[1])]
            for row in range(y_shape[0])
        ]

        x_group = VGroup(*[box for row in x_boxes for box in row], *[label for row in x_labels for label in row])
        k_group = VGroup(*[box for row in k_boxes for box in row], *[label for row in k_labels for label in row])
        y_group = VGroup(*[box for row in y_boxes for box in row])
        x_name = Text("X", font_size=32, color=BLUE_A).next_to(x_group, np.array([-1.0, 0.0, 0.0]), buff=0.22)
        k_name = Text("K", font_size=32, color=YELLOW_A).next_to(k_group, np.array([0.0, 1.0, 0.0]), buff=0.22)
        y_name = Text("Y", font_size=32, color=BLUE_A).next_to(y_group, np.array([0.0, 1.0, 0.0]), buff=0.22)
        formula = Text("Y += x * K", font_size=32, color=WHITE).move_to(np.array([0.0, 3.18, 0.0]))
        shape = Text("2 × 2 → 3 × 3", font_size=26, color=WHITE).next_to(
            y_group, np.array([0.0, -1.0, 0.0]), buff=0.28
        )

        traveler_center = self.cell_center(self.x_origin, INPUT_X.shape, *TRAVELER)
        halo_outer, halo_inner = self.make_halo(traveler_center)
        landing_centers = [
            y_centers[TRAVELER[0] + row][TRAVELER[1] + col]
            for row in range(KERNEL_H)
            for col in range(KERNEL_W)
        ]
        window = self.make_window(landing_centers)

        products = [
            (row, col, float(KERNEL[row, col]), float(PATCH[row, col]))
            for row in range(KERNEL_H)
            for col in range(KERNEL_W)
        ]

        # 0.40–2.50 s: the 2 × 2 input, numbers on the cells.
        # Whole map together so t=1 is a complete lattice, not a half-faded 3.
        self.play(
            FadeIn(x_group),
            FadeIn(x_name),
            run_time=1.10,
            rate_func=linear,
        )
        self.wait(1.00)

        # 2.50–4.60 s: halo is that one input cell, not an output dashboard.
        traveler_box = x_boxes[TRAVELER[0]][TRAVELER[1]]
        traveler_label = x_labels[TRAVELER[0]][TRAVELER[1]]
        self.play(
            Create(halo_outer),
            Create(halo_inner),
            traveler_box.animate.set_stroke(YELLOW, width=4.6, opacity=1.0),
            traveler_label.animate.set_color(YELLOW),
            Flash(traveler_center, color=YELLOW, flash_radius=0.46, line_length=0.12),
            run_time=0.85,
            rate_func=smooth,
        )
        self.wait(1.25)

        # 4.60–6.70 s: the untrained kernel, same cell size as X.
        self.play(
            LaggedStart(
                *[FadeIn(mobject) for row in range(2) for col in range(2) for mobject in (k_boxes[row][col], k_labels[row][col])],
                lag_ratio=0.06,
            ),
            FadeIn(k_name),
            run_time=0.90,
            rate_func=smooth,
        )
        self.wait(1.20)

        # 6.70–8.90 s: empty 3 × 3 output (larger), formula and shape once.
        self.play(
            LaggedStart(*[FadeIn(box) for row in y_boxes for box in row], lag_ratio=0.05),
            FadeIn(y_name),
            FadeIn(formula),
            FadeIn(shape),
            run_time=1.00,
            rate_func=smooth,
        )
        self.wait(1.20)

        # 8.90–10.90 s: the stamp lands on Y[i:i+h, j:j+w] for this one x.
        self.play(Create(window), run_time=0.80, rate_func=smooth)
        self.wait(1.20)

        # 10.90–19.10 s: four products, each beat ~2 s, numbers from numpy.
        y_written: list[Text] = []
        for row, col, _k_val, product in products:
            k_box = k_boxes[row][col]
            k_label = k_labels[row][col]
            dest_row = TRAVELER[0] + row
            dest_col = TRAVELER[1] + col
            dest_center = y_centers[dest_row][dest_col]
            dest_box = y_boxes[dest_row][dest_col]
            written = Text(fmt_int(product), font_size=32, color=YELLOW).move_to(dest_center)
            self.play(
                k_box.animate.set_stroke(YELLOW, width=5.0, opacity=1.0),
                k_label.animate.set_color(YELLOW),
                dest_box.animate.set_stroke(YELLOW, width=4.4, opacity=1.0),
                FadeIn(written),
                Flash(dest_center, color=YELLOW, flash_radius=0.42, line_length=0.11),
                run_time=0.85,
                rate_func=smooth,
            )
            self.play(
                k_box.animate.set_stroke(YELLOW_A, width=2.0, opacity=0.94),
                k_label.animate.set_color(YELLOW_A),
                run_time=0.20,
                rate_func=smooth,
            )
            self.wait(1.00)
            y_written.append(written)

        # Hold the larger 3 × 3, the four stamped cells, empty pockets, and halo.
        _ = y_written
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=12.00, rate_func=linear)
