"""D2L 6.4 — two input slices, two kernels, one summed output cell.

Every on-screen number comes from the arrays and ``corr2d_multi_in`` below.
This is forward computation only: the kernels are not trained.
"""

from __future__ import annotations

import numpy as np
from scripts.episode_intro import show_episode_intro
from manim import (
    BLACK,
    BLUE,
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
    Square,
    SurroundingRectangle,
    Text,
    Transform,
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


# D2L figure 6.4.1 tensors. X is 2 × 3 × 3; each output channel has a
# 2 × 2 × 2 kernel. O[c] = Σ_i (X[i] * K[c, i]).
INPUT_X = np.array(
    [
        [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
        ],
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
    ],
    dtype=float,
)
KERNEL_1 = np.array(
    [
        [
            [0.0, 1.0],
            [2.0, 3.0],
        ],
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
    ],
    dtype=float,
)
# Second output channel: D2L stacks K and K+1. Two maps is enough.
KERNEL_OUT = np.stack([KERNEL_1, KERNEL_1 + 1.0], axis=0)
KERNEL_H, KERNEL_W = 2, 2
WINDOW = (0, 0)


def corr2d(input_x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """D2L corr2d: elementwise product of each window with K, then sum."""
    kernel_h, kernel_w = kernel.shape
    out_h = input_x.shape[0] - kernel_h + 1
    out_w = input_x.shape[1] - kernel_w + 1
    output = np.zeros((out_h, out_w), dtype=float)
    for row in range(out_h):
        for col in range(out_w):
            window = input_x[row : row + kernel_h, col : col + kernel_w]
            output[row, col] = float(np.sum(window * kernel))
    return output


def corr2d_multi_in(input_x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Sum per-channel cross-correlations. Numpy is the only arithmetic."""
    return sum(corr2d(channel, kernel_c) for channel, kernel_c in zip(input_x, kernel))


PARTIALS = np.stack([corr2d(INPUT_X[index], KERNEL_1[index]) for index in range(2)])
OUTPUTS = np.stack([corr2d_multi_in(INPUT_X, KERNEL_OUT[index]) for index in range(2)])
assert np.allclose(OUTPUTS[0], PARTIALS[0] + PARTIALS[1])


def fmt_int(value: float) -> str:
    """Integer glyph for a cell; use a true minus if a later kernel needs it."""
    rounded = int(round(value))
    if rounded < 0:
        return f"−{abs(rounded)}"
    return str(rounded)


class Episode064(Scene):
    """A ~32-second continuous, silent multi-channel convolution visualization."""

    x_cell = 0.78
    k_cell = 0.58
    o_cell = 0.98
    p_cell = 0.90
    square_scale = 0.90
    x_centers = (np.array([-4.95, 1.68, 0.0]), np.array([-4.95, -1.48, 0.0]))
    k_centers = (np.array([-2.18, 1.68, 0.0]), np.array([-2.18, -1.48, 0.0]))
    p_centers = (np.array([0.58, 1.68, 0.0]), np.array([0.58, -1.48, 0.0]))
    o_centers = (np.array([4.58, 1.68, 0.0]), np.array([4.58, -1.48, 0.0]))
    channel_colors = (BLUE, BLUE_A)

    def cell_center(self, origin: np.ndarray, cell: float, shape: tuple[int, int], row: int, col: int) -> np.ndarray:
        rows, cols = shape
        x_value = origin[0] + (col - (cols - 1) / 2.0) * cell
        y_value = origin[1] + ((rows - 1) / 2.0 - row) * cell
        return np.array([x_value, y_value, 0.0])

    def stroke_square(self, side: float, color, width: float = 2.0, opacity: float = 0.90) -> Square:
        """Stroke-only cell. Never fill; never call set_opacity on the mesh."""
        square = Square(side_length=side)
        square.set_fill(BLACK, opacity=0.0)
        square.set_stroke(color, width=width, opacity=opacity)
        return square

    def make_cells(
        self,
        values: np.ndarray,
        origin: np.ndarray,
        cell: float,
        stroke_color,
        font_size: int,
        number_color,
    ) -> tuple[list[list[Square]], list[list[Text]]]:
        rows, cols = values.shape
        squares: list[list[Square]] = []
        labels: list[list[Text]] = []
        side = cell * self.square_scale
        for row in range(rows):
            square_row: list[Square] = []
            label_row: list[Text] = []
            for col in range(cols):
                position = self.cell_center(origin, cell, values.shape, row, col)
                square = self.stroke_square(side, stroke_color)
                square.move_to(position)
                label = Text(fmt_int(values[row, col]), font_size=font_size, color=number_color)
                label.move_to(position)
                square_row.append(square)
                label_row.append(label)
            squares.append(square_row)
            labels.append(label_row)
        return squares, labels

    def patch_squares(self, squares: list[list[Square]], start_row: int, start_col: int) -> VGroup:
        return VGroup(
            *[
                squares[start_row + row][start_col + col]
                for row in range(KERNEL_H)
                for col in range(KERNEL_W)
            ]
        )

    def make_window(self, patch: VGroup) -> SurroundingRectangle:
        window = SurroundingRectangle(patch, color=YELLOW, buff=0.07, corner_radius=0.04)
        window.set_fill(BLACK, opacity=0.0)
        window.set_stroke(YELLOW, width=3.4, opacity=0.98)
        return window

    def make_halo(self, center: np.ndarray) -> tuple[Circle, Circle]:
        outer = Circle(radius=0.50).move_to(center).set_stroke(BLUE_A, width=2.1, opacity=0.50)
        inner = Circle(radius=0.38).move_to(center).set_stroke(YELLOW, width=2.8, opacity=0.98)
        return outer, inner

    def construct(self) -> None:
        traveler_partials = (float(PARTIALS[0, 0, 0]), float(PARTIALS[1, 0, 0]))
        traveler_value = float(OUTPUTS[0, 0, 0])
        print(
            "D2L 6.4 X={}\nK={}\nP={}\nO={}\ntraveler P1={:.0f} P2={:.0f} O00={:.0f}".format(
                np.array2string(INPUT_X, precision=0, separator=", "),
                np.array2string(KERNEL_OUT, precision=0, separator=", "),
                np.array2string(PARTIALS, precision=0, separator=", "),
                np.array2string(OUTPUTS, precision=0, separator=", "),
                traveler_partials[0],
                traveler_partials[1],
                traveler_value,
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "06.4")

        x_squares: list[list[list[Square]]] = []
        x_labels: list[list[list[Text]]] = []
        k_squares: list[list[list[Square]]] = []
        k_labels: list[list[list[Text]]] = []
        x_groups = []
        x_label_groups = []
        k_groups = []
        k_label_groups = []
        x_names = []
        k_names = []
        for index in range(2):
            squares, labels = self.make_cells(
                INPUT_X[index],
                self.x_centers[index],
                self.x_cell,
                self.channel_colors[index],
                26,
                WHITE,
            )
            x_squares.append(squares)
            x_labels.append(labels)
            x_group = VGroup(*[square for row in squares for square in row])
            x_label_group = VGroup(*[label for row in labels for label in row])
            x_groups.append(x_group)
            x_label_groups.append(x_label_group)
            x_names.append(
                Text(f"X{'₁₂'[index]}", font_size=28, color=self.channel_colors[index]).next_to(
                    x_group, np.array([-1.0, 0.0, 0.0]), buff=0.18
                )
            )
            k_sq, k_lb = self.make_cells(
                KERNEL_1[index],
                self.k_centers[index],
                self.k_cell,
                YELLOW_A,
                22,
                YELLOW_A,
            )
            k_squares.append(k_sq)
            k_labels.append(k_lb)
            k_group = VGroup(*[square for row in k_sq for square in row])
            k_label_group = VGroup(*[label for row in k_lb for label in row])
            k_groups.append(k_group)
            k_label_groups.append(k_label_group)
            k_names.append(
                Text(f"K{'₁₂'[index]}", font_size=26, color=YELLOW_A).next_to(
                    k_group, np.array([0.0, 1.0, 0.0]), buff=0.14
                )
            )

        formula = Text("o = (X₁ * K₁) + (X₂ * K₂)", font_size=32, color=WHITE).move_to(
            np.array([1.55, 3.28, 0.0])
        )

        o_squares = [
            [
                [
                    self.stroke_square(self.o_cell * self.square_scale, BLUE_A).move_to(
                        self.cell_center(self.o_centers[map_index], self.o_cell, (2, 2), row, col)
                    )
                    for col in range(2)
                ]
                for row in range(2)
            ]
            for map_index in range(2)
        ]
        o1_group = VGroup(*[square for row in o_squares[0] for square in row])
        o2_group = VGroup(*[square for row in o_squares[1] for square in row])
        o1_name = Text("O₁", font_size=28, color=BLUE_A).next_to(o1_group, np.array([0.0, 1.0, 0.0]), buff=0.16)
        o2_name = Text("O₂", font_size=28, color=BLUE_A).next_to(o2_group, np.array([0.0, 1.0, 0.0]), buff=0.16)

        traveler_center = o_squares[0][0][0].get_center()
        halo_outer, halo_inner = self.make_halo(traveler_center)
        first_patch = self.patch_squares(x_squares[0], *WINDOW)
        second_patch = self.patch_squares(x_squares[1], *WINDOW)
        window = self.make_window(first_patch)

        partial_squares = []
        partial_labels = []
        partial_names = []
        for index in range(2):
            square = self.stroke_square(self.p_cell * self.square_scale, self.channel_colors[index], width=2.4)
            square.move_to(self.p_centers[index])
            label = Text(fmt_int(traveler_partials[index]), font_size=32, color=WHITE).move_to(
                self.p_centers[index]
            )
            name = Text(f"X{'₁₂'[index]} * K{'₁₂'[index]}", font_size=18, color=self.channel_colors[index]).next_to(
                square, np.array([0.0, 1.0, 0.0]), buff=0.10
            )
            partial_squares.append(square)
            partial_labels.append(label)
            partial_names.append(name)

        plus_sign = Text("+", font_size=40, color=WHITE).move_to(np.array([0.58, 0.10, 0.0]))
        traveler_readout = Text(fmt_int(traveler_value), font_size=34, color=YELLOW).move_to(traveler_center)
        rest_readouts_1 = [
            Text(fmt_int(OUTPUTS[0, row, col]), font_size=28, color=WHITE).move_to(
                o_squares[0][row][col].get_center()
            )
            for row in range(2)
            for col in range(2)
            if not (row == 0 and col == 0)
        ]
        rest_readouts_2 = [
            Text(fmt_int(OUTPUTS[1, row, col]), font_size=28, color=WHITE).move_to(
                o_squares[1][row][col].get_center()
            )
            for row in range(2)
            for col in range(2)
        ]
        connector = Line(
            plus_sign.get_right() + np.array([0.18, 0.0, 0.0]),
            traveler_center + np.array([-0.55, 0.0, 0.0]),
        ).set_stroke(BLUE_D, width=2.2, opacity=0.62)

        def highlight_channel(index: int):
            x_row = x_squares[index]
            x_lab = x_labels[index]
            k_row = k_squares[index]
            k_lab = k_labels[index]
            animations = []
            for row in range(KERNEL_H):
                for col in range(KERNEL_W):
                    animations.extend(
                        [
                            x_row[WINDOW[0] + row][WINDOW[1] + col]
                            .animate.set_stroke(YELLOW, width=5.0, opacity=1.0),
                            x_lab[WINDOW[0] + row][WINDOW[1] + col].animate.set_color(YELLOW),
                            k_row[row][col].animate.set_stroke(YELLOW, width=5.0, opacity=1.0),
                            k_lab[row][col].animate.set_color(YELLOW),
                        ]
                    )
            return animations

        def restore_channel(index: int):
            x_row = x_squares[index]
            x_lab = x_labels[index]
            k_row = k_squares[index]
            k_lab = k_labels[index]
            animations = []
            for row in range(KERNEL_H):
                for col in range(KERNEL_W):
                    animations.extend(
                        [
                            x_row[WINDOW[0] + row][WINDOW[1] + col]
                            .animate.set_stroke(self.channel_colors[index], width=2.0, opacity=0.90),
                            x_lab[WINDOW[0] + row][WINDOW[1] + col].animate.set_color(WHITE),
                            k_row[row][col].animate.set_stroke(YELLOW_A, width=2.0, opacity=0.90),
                            k_lab[row][col].animate.set_color(YELLOW_A),
                        ]
                    )
            return animations

        # 0.40–4.20 s: two input slices first, held before any kernel or window.
        self.play(
            LaggedStart(
                *[FadeIn(mobject) for pair in zip(x_groups[0], x_label_groups[0]) for mobject in pair],
                lag_ratio=0.04,
            ),
            FadeIn(x_names[0]),
            run_time=0.90,
            rate_func=linear,
        )
        self.play(
            LaggedStart(
                *[FadeIn(mobject) for pair in zip(x_groups[1], x_label_groups[1]) for mobject in pair],
                lag_ratio=0.04,
            ),
            FadeIn(x_names[1]),
            run_time=0.90,
            rate_func=linear,
        )
        self.wait(2.00)

        # 4.20–6.40 s: one kernel per input channel, then an empty output map.
        self.play(
            LaggedStart(
                *[FadeIn(mobject) for pair in zip(k_groups[0], k_label_groups[0]) for mobject in pair],
                lag_ratio=0.05,
            ),
            FadeIn(k_names[0]),
            LaggedStart(
                *[FadeIn(mobject) for pair in zip(k_groups[1], k_label_groups[1]) for mobject in pair],
                lag_ratio=0.05,
            ),
            FadeIn(k_names[1]),
            run_time=1.10,
            rate_func=smooth,
        )
        self.play(
            LaggedStart(*[FadeIn(square) for square in o1_group], lag_ratio=0.08),
            FadeIn(o1_name),
            FadeIn(formula),
            run_time=1.10,
            rate_func=smooth,
        )
        self.wait(1.80)

        # 8.20–11.00 s: channel-1 window writes the first partial.
        self.play(Create(window), run_time=0.70, rate_func=smooth)
        self.play(*highlight_channel(0), run_time=0.55, rate_func=smooth)
        self.play(
            FadeIn(partial_squares[0]),
            FadeIn(partial_labels[0]),
            FadeIn(partial_names[0]),
            run_time=1.35,
            rate_func=smooth,
        )
        self.play(*restore_channel(0), run_time=0.35, rate_func=smooth)
        self.wait(0.40)

        # 11.70–14.80 s: the same window on channel 2 writes the second partial.
        stepped_window = self.make_window(second_patch)
        self.play(Transform(window, stepped_window), run_time=1.10, rate_func=smooth)
        self.play(*highlight_channel(1), run_time=0.55, rate_func=smooth)
        self.play(
            FadeIn(partial_squares[1]),
            FadeIn(partial_labels[1]),
            FadeIn(partial_names[1]),
            run_time=1.35,
            rate_func=smooth,
        )
        self.play(*restore_channel(1), run_time=0.35, rate_func=smooth)
        self.wait(0.45)

        # 15.25–17.70 s: the two partials add into the haloed output cell.
        self.play(FadeIn(plus_sign), Create(connector), run_time=0.70, rate_func=smooth)
        self.play(
            Create(halo_outer),
            Create(halo_inner),
            FadeIn(traveler_readout),
            Flash(traveler_center, color=YELLOW, flash_radius=0.50, line_length=0.12),
            run_time=1.20,
            rate_func=smooth,
        )
        self.wait(1.90)

        # Other cells of O₁ appear; they are the same sum, not a second lesson.
        self.play(
            LaggedStart(*[FadeIn(label) for label in rest_readouts_1], lag_ratio=0.12),
            run_time=1.20,
            rate_func=smooth,
        )
        self.wait(0.80)

        # Short second beat: a second output channel is another stacked map.
        # The same sum formula still holds; do not swap in a braced 4-D HUD.
        self.play(
            LaggedStart(*[FadeIn(square) for square in o2_group], lag_ratio=0.08),
            FadeIn(o2_name),
            run_time=1.15,
            rate_func=smooth,
        )
        self.play(
            LaggedStart(*[FadeIn(label) for label in rest_readouts_2], lag_ratio=0.10),
            run_time=1.20,
            rate_func=smooth,
        )

        # Hold both maps, both partials, and the traveler through the last frame.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=8.40, rate_func=linear)
