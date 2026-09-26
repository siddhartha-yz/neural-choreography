"""D2L 7.4 — Inception: four parallel windows, concat on the channel axis.

Every on-screen number comes from the arrays and helpers below. The four
branches share one input map. Kernels are not trained. Spatial size is kept
(pad so H,W match); the traveler is one cell whose channels grow by concat.
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
    FadeIn,
    FadeOut,
    Flash,
    LaggedStart,
    Line,
    Scene,
    Square,
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


# Same 3 × 3 lattice as 6.2 / 6.3 / 6.4. One input channel.
INPUT_X = np.array(
    [
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
    ],
    dtype=float,
)
# 1 × 1: this cell only.
KERNEL_1 = np.array([[1.0]], dtype=float)
# 3 × 3 plus, pad 1.
KERNEL_3 = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)
# 5 × 5 diamond, pad 2. Wider than the plus; not all-ones (that map is flat).
KERNEL_5 = np.array(
    [
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 2.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
    ],
    dtype=float,
)
POOL = 3
TRAVELER = (1, 1)
# 5 × 5 on the traveler covers original coords [−1, 3]; one ring of pad zeros.
PAD_WINDOW = (-1, 3)


def corr2d(input_x: np.ndarray, kernel: np.ndarray, padding: int) -> np.ndarray:
    """Cross-correlation, D2L (no flip). Pad with zeros."""
    padded = np.pad(input_x, padding)
    kernel_h, kernel_w = kernel.shape
    out_h = padded.shape[0] - kernel_h + 1
    out_w = padded.shape[1] - kernel_w + 1
    output = np.zeros((out_h, out_w), dtype=float)
    for row in range(out_h):
        for col in range(out_w):
            window = padded[row : row + kernel_h, col : col + kernel_w]
            output[row, col] = float(np.sum(window * kernel))
    return output


def max_pool(input_x: np.ndarray, size: int = 3, padding: int = 1, stride: int = 1) -> np.ndarray:
    """3 × 3 max-pool, stride 1, pad 1 — Inception keeps H, W."""
    padded = np.pad(input_x, padding)
    out_h = (padded.shape[0] - size) // stride + 1
    out_w = (padded.shape[1] - size) // stride + 1
    output = np.zeros((out_h, out_w), dtype=float)
    for row in range(out_h):
        for col in range(out_w):
            patch = padded[
                row * stride : row * stride + size,
                col * stride : col * stride + size,
            ]
            output[row, col] = float(np.max(patch))
    return output


Y1 = corr2d(INPUT_X, KERNEL_1, padding=0)
Y3 = corr2d(INPUT_X, KERNEL_3, padding=1)
Y5 = corr2d(INPUT_X, KERNEL_5, padding=2)
YP = max_pool(INPUT_X, size=POOL, padding=1, stride=1)
CONCAT = np.stack([Y1, Y3, Y5, YP], axis=-1)
assert Y1.shape == Y3.shape == Y5.shape == YP.shape == INPUT_X.shape
assert CONCAT.shape == (3, 3, 4)
TRAVELER_CHANNELS = CONCAT[TRAVELER]


def fmt_int(value: float) -> str:
    rounded = int(round(value))
    if rounded < 0:
        return f"−{abs(rounded)}"
    return str(rounded)


class Episode074(Scene):
    """A ~33-second silent Inception block: four branches, then concat."""

    x_cell = 0.70
    y_cell = 0.50
    c_cell = 0.62
    square_scale = 0.90
    title_y = 3.26
    x_origin = np.array([-4.88, 0.08, 0.0])
    y_origins = (
        np.array([-1.50, 1.40, 0.0]),
        np.array([1.22, 1.40, 0.0]),
        np.array([-1.50, -1.74, 0.0]),
        np.array([1.22, -1.74, 0.0]),
    )
    concat_origin = np.array([4.58, 0.08, 0.0])
    branch_colors = (BLUE, YELLOW, BLUE_A, BLUE_D)
    branch_names = ("1×1", "3×3", "5×5", "3×3 max")
    branch_maps = (Y1, Y3, Y5, YP)
    branch_tags = ("Y₁", "Y₃", "Y₅", "Yₚ")

    def cell_center(self, origin: np.ndarray, cell: float, shape: tuple[int, int], row: int, col: int) -> np.ndarray:
        rows, cols = shape
        x_value = origin[0] + (col - (cols - 1) / 2.0) * cell
        y_value = origin[1] + ((rows - 1) / 2.0 - row) * cell
        return np.array([x_value, y_value, 0.0])

    def stroke_square(self, side: float, color, width: float = 2.0, opacity: float = 0.92) -> Square:
        """Stroke-only cell. Never fill; never set_opacity on a filled mesh."""
        square = Square(side_length=side)
        square.set_fill(BLACK, opacity=0.0)
        square.set_stroke(color, width=width, opacity=opacity)
        return square

    def stroke_box(self, x0: float, y0: float, x1: float, y1: float, color, width: float) -> VGroup:
        box = VGroup(
            Line(np.array([x0, y1, 0.0]), np.array([x1, y1, 0.0])),
            Line(np.array([x1, y1, 0.0]), np.array([x1, y0, 0.0])),
            Line(np.array([x1, y0, 0.0]), np.array([x0, y0, 0.0])),
            Line(np.array([x0, y0, 0.0]), np.array([x0, y1, 0.0])),
        )
        box.set_stroke(color, width=width, opacity=0.96)
        box.set_fill(opacity=0.0)
        return box

    def make_cells(
        self,
        values: np.ndarray,
        origin: np.ndarray,
        cell: float,
        stroke_color,
        font_size: int,
        number_color,
    ) -> tuple[list[list[Square]], list[VGroup], VGroup, VGroup]:
        rows, cols = values.shape
        squares: list[list[Square]] = []
        labeled: list[VGroup] = []
        side = cell * self.square_scale
        for row in range(rows):
            square_row: list[Square] = []
            for col in range(cols):
                position = self.cell_center(origin, cell, values.shape, row, col)
                square = self.stroke_square(side, stroke_color)
                square.move_to(position)
                label = Text(fmt_int(values[row, col]), font_size=font_size, color=number_color)
                label.move_to(position)
                square_row.append(square)
                labeled.append(VGroup(square, label))
            squares.append(square_row)
        square_group = VGroup(*[square for row in squares for square in row])
        label_group = VGroup(*[cell[1] for cell in labeled])
        return squares, labeled, square_group, label_group

    def window_on_input(self, row0: int, col0: int, row1: int, col1: int) -> VGroup:
        """Yellow window in input-cell coordinates; may extend into pad."""
        top_left = self.cell_center(self.x_origin, self.x_cell, INPUT_X.shape, row0, col0)
        bottom_right = self.cell_center(self.x_origin, self.x_cell, INPUT_X.shape, row1, col1)
        half = 0.5 * self.x_cell
        pad = 0.06
        return self.stroke_box(
            top_left[0] - half - pad,
            bottom_right[1] - half - pad,
            bottom_right[0] + half + pad,
            top_left[1] + half + pad,
            YELLOW,
            3.6,
        )

    def make_pad_ring(self) -> VGroup:
        """Zeros the 5 × 5 traveler window actually covers; not a HUD."""
        cells = []
        side = self.x_cell * self.square_scale
        lo, hi = PAD_WINDOW
        for row in range(lo, hi + 1):
            for col in range(lo, hi + 1):
                if 0 <= row < INPUT_X.shape[0] and 0 <= col < INPUT_X.shape[1]:
                    continue
                position = self.cell_center(self.x_origin, self.x_cell, INPUT_X.shape, row, col)
                square = self.stroke_square(side, BLUE_E, width=1.6, opacity=0.78)
                square.move_to(position)
                label = Text("0", font_size=20, color=BLUE_A)
                label.move_to(position)
                cells.append(VGroup(square, label))
        return VGroup(*cells)

    def construct(self) -> None:
        print("D2L 7.4 X=\n{}".format(np.array2string(INPUT_X, precision=0, separator=", ")))
        print("D2L 7.4 Y1 1x1=\n{}".format(np.array2string(Y1, precision=0, separator=", ")))
        print("D2L 7.4 Y3 3x3=\n{}".format(np.array2string(Y3, precision=0, separator=", ")))
        print("D2L 7.4 Y5 5x5=\n{}".format(np.array2string(Y5, precision=0, separator=", ")))
        print("D2L 7.4 Yp max=\n{}".format(np.array2string(YP, precision=0, separator=", ")))
        print(
            "D2L 7.4 concat shape={} traveler {} channels={}".format(
                CONCAT.shape,
                TRAVELER,
                np.array2string(TRAVELER_CHANNELS, precision=0, separator=", "),
            )
        )
        print(
            "D2L 7.4 K1={} K3=\n{} K5=\n{}".format(
                np.array2string(KERNEL_1, precision=0, separator=", "),
                np.array2string(KERNEL_3, precision=0, separator=", "),
                np.array2string(KERNEL_5, precision=0, separator=", "),
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "07.4")

        _x_squares, x_labeled, _x_group, _x_label_group = self.make_cells(
            INPUT_X, self.x_origin, self.x_cell, BLUE, 26, WHITE
        )
        x_name = Text("X", font_size=30, color=BLUE_A).move_to(
            np.array([self.x_origin[0], self.title_y, 0.0])
        )
        traveler_in = self.cell_center(self.x_origin, self.x_cell, INPUT_X.shape, *TRAVELER)
        halo_outer = Circle(radius=0.48).move_to(traveler_in).set_stroke(BLUE_A, width=2.1, opacity=0.55)
        halo_inner = Circle(radius=0.34).move_to(traveler_in).set_stroke(YELLOW, width=2.8, opacity=0.98)
        halo_outer.set_fill(opacity=0.0)
        halo_inner.set_fill(opacity=0.0)

        # One empty pocket on the title row, same baseline as X / Y — not a HUD.
        formula = Text("Y = concat(Y₁, Y₃, Y₅, Yₚ)", font_size=28, color=WHITE).move_to(
            np.array([0.12, self.title_y, 0.0])
        )

        branch_labeled_groups = []
        branch_title_mobs = []
        for origin, values, color, name in zip(
            self.y_origins, self.branch_maps, self.branch_colors, self.branch_names
        ):
            _squares, labeled, _square_group, _label_group = self.make_cells(
                values, origin, self.y_cell, color, 18, WHITE
            )
            title = Text(name, font_size=22, color=color).move_to(
                np.array([origin[0], origin[1] + 0.88, 0.0])
            )
            traveler_square = _squares[TRAVELER[0]][TRAVELER[1]]
            traveler_square.set_stroke(YELLOW, width=3.6, opacity=1.0)
            branch_labeled_groups.append(labeled)
            branch_title_mobs.append(title)

        concat_squares = []
        concat_labels = []
        for index, value in enumerate(TRAVELER_CHANNELS):
            position = self.concat_origin + np.array([0.0, (1.5 - index) * self.c_cell, 0.0])
            square = self.stroke_square(self.c_cell * self.square_scale, self.branch_colors[index], width=2.6)
            square.move_to(position)
            label = Text(fmt_int(value), font_size=26, color=WHITE).move_to(position)
            channel_tag = Text(self.branch_tags[index], font_size=20, color=self.branch_colors[index]).next_to(
                square, np.array([1.0, 0.0, 0.0]), buff=0.22
            )
            concat_squares.append(square)
            concat_labels.append(VGroup(label, channel_tag))

        concat_name = Text("Y", font_size=30, color=YELLOW_A).move_to(
            np.array([self.concat_origin[0], self.title_y, 0.0])
        )
        channel_brace_x = self.concat_origin[0] - 0.58
        y_top = self.concat_origin[1] + 1.5 * self.c_cell + 0.28
        y_bottom = self.concat_origin[1] - 1.5 * self.c_cell - 0.28
        channel_brace = VGroup(
            Line(np.array([channel_brace_x, y_top, 0.0]), np.array([channel_brace_x + 0.12, y_top, 0.0])),
            Line(np.array([channel_brace_x, y_top, 0.0]), np.array([channel_brace_x, y_bottom, 0.0])),
            Line(np.array([channel_brace_x, y_bottom, 0.0]), np.array([channel_brace_x + 0.12, y_bottom, 0.0])),
        ).set_stroke(YELLOW_A, width=1.8, opacity=0.90)
        channel_brace.set_fill(opacity=0.0)
        channel_label = Text("c = 4", font_size=22, color=YELLOW_A).move_to(
            np.array([channel_brace_x - 0.50, self.concat_origin[1], 0.0])
        )

        concat_halo = self.stroke_box(
            self.concat_origin[0] - 0.36,
            self.concat_origin[1] - 1.5 * self.c_cell - 0.32,
            self.concat_origin[0] + 0.36,
            self.concat_origin[1] + 1.5 * self.c_cell + 0.32,
            YELLOW,
            2.8,
        )

        windows = (
            self.window_on_input(1, 1, 1, 1),
            self.window_on_input(0, 0, 2, 2),
            self.window_on_input(-1, -1, 3, 3),
            self.window_on_input(0, 0, 2, 2),
        )
        pad_group = self.make_pad_ring()

        # 0.40–2.60 s: the shared input, halo on the traveler cell.
        self.play(
            LaggedStart(*[FadeIn(cell) for cell in x_labeled], lag_ratio=0.05),
            FadeIn(x_name),
            run_time=1.00,
            rate_func=linear,
        )
        self.play(
            Create(halo_outer),
            Create(halo_inner),
            Flash(traveler_in, color=YELLOW, flash_radius=0.50, line_length=0.12),
            run_time=0.70,
            rate_func=smooth,
        )
        self.wait(0.50)

        # 2.60–4.60 s: concat formula once, on the title-row pocket.
        self.play(FadeIn(formula), run_time=0.80, rate_func=smooth)
        self.wait(1.20)

        # Four parallel branches, ~2.2 s each. Same X; windows of different size.
        # 4.60–6.80, 6.80–9.00, 9.00–11.20, 11.20–13.40
        for index in range(4):
            intro = [FadeIn(windows[index])]
            if index == 2:
                intro.append(LaggedStart(*[FadeIn(cell) for cell in pad_group], lag_ratio=0.04))
            self.play(*intro, run_time=0.45, rate_func=smooth)
            self.play(
                LaggedStart(*[FadeIn(cell) for cell in branch_labeled_groups[index]], lag_ratio=0.05),
                FadeIn(branch_title_mobs[index]),
                run_time=1.15,
                rate_func=smooth,
            )
            self.wait(0.20)
            self.play(FadeOut(windows[index]), run_time=0.40, rate_func=linear)

        # 13.40–15.60 s: four maps held — pad ring and all branches still on screen.
        self.wait(2.20)

        # 15.60–18.20 s: the traveler cell’s four values stack on the channel axis.
        self.play(FadeIn(concat_name), FadeIn(concat_halo), run_time=0.50, rate_func=smooth)
        self.play(
            LaggedStart(
                *[FadeIn(square) for square in concat_squares],
                *[FadeIn(label) for label in concat_labels],
                lag_ratio=0.12,
            ),
            FadeIn(channel_brace),
            FadeIn(channel_label),
            Flash(self.concat_origin, color=YELLOW, flash_radius=0.55, line_length=0.12),
            run_time=1.60,
            rate_func=smooth,
        )
        self.wait(0.50)

        # 18.20–33.20 s: hold X, four branches, pad ring, and the c = 4 stack.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=15.00, rate_func=linear)
