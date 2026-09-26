"""D2L 6.2 — cross-correlation: a 3×3 kernel writes one output cell, then steps.

Every on-screen number comes from the arrays and ``corr2d`` below. This is
forward computation only: the kernel is not trained and no bias is added.
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
    DecimalNumber,
    FadeIn,
    FadeOut,
    Flash,
    LaggedStart,
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


# Tiny real input and an untrained 3×3 kernel. Output is 2×2; the film writes
# the top row only (one cell, then one step to its neighbor).
INPUT_X = np.array(
    [
        [0.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
        [8.0, 9.0, 0.0, 1.0],
        [2.0, 3.0, 4.0, 5.0],
    ],
    dtype=float,
)
KERNEL = np.array(
    [
        [0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
    ],
    dtype=float,
)


def corr2d(input_x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """D2L ``corr2d``: elementwise product of each window with K, then sum."""
    kernel_h, kernel_w = kernel.shape
    out_h = input_x.shape[0] - kernel_h + 1
    out_w = input_x.shape[1] - kernel_w + 1
    output = np.zeros((out_h, out_w), dtype=float)
    for row in range(out_h):
        for col in range(out_w):
            window = input_x[row : row + kernel_h, col : col + kernel_w]
            output[row, col] = float(np.sum(window * kernel))
    return output


OUTPUT = corr2d(INPUT_X, KERNEL)
KERNEL_H, KERNEL_W = KERNEL.shape


def fmt_int(value: float) -> str:
    """Integer glyph for a cell; use a true minus if a later kernel needs it."""
    rounded = int(round(value))
    if rounded < 0:
        return f"−{abs(rounded)}"
    return str(rounded)


class Episode062(Scene):
    """A ~32-second continuous, silent D2L 6.2 visualization."""

    # X left, K center, O right — one row, tops roughly aligned, no long
    # connector across a live cell.
    x_origin = np.array([-4.55, -0.30, 0.0])
    x_cell = 0.92
    k_origin = np.array([0.12, 0.08, 0.0])
    k_cell = 0.72
    o_origin = np.array([4.72, 0.08, 0.0])
    o_cell = 1.12
    square_scale = 0.92

    def cell_center(self, origin: np.ndarray, cell: float, shape: tuple[int, int], row: int, col: int) -> np.ndarray:
        rows, cols = shape
        x_value = origin[0] + (col - (cols - 1) / 2.0) * cell
        y_value = origin[1] + ((rows - 1) / 2.0 - row) * cell
        return np.array([x_value, y_value, 0.0])

    def stroke_square(self, side: float, color, width: float = 2.0, opacity: float = 0.90) -> Square:
        """Stroke-only cell. Never fill; never call ``set_opacity`` on the mesh."""
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

    def patch_squares(self, x_squares: list[list[Square]], start_row: int, start_col: int) -> VGroup:
        return VGroup(
            *[
                x_squares[start_row + row][start_col + col]
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
        outer = Circle(radius=0.54).move_to(center).set_stroke(BLUE_A, width=2.1, opacity=0.50)
        inner = Circle(radius=0.47).move_to(center).set_stroke(YELLOW, width=2.6, opacity=0.98)
        return outer, inner

    def construct(self) -> None:
        first_terms = [
            (row, col, float(INPUT_X[row, col]), float(KERNEL[row, col]), float(INPUT_X[row, col] * KERNEL[row, col]))
            for row in range(KERNEL_H)
            for col in range(KERNEL_W)
        ]
        running = np.cumsum([term[4] for term in first_terms])
        print(
            "D2L 6.2 X={}\nK={}\nO={}\nO[0,0]={:.0f} terms={} running={}\nO[0,1]={:.0f}".format(
                np.array2string(INPUT_X, precision=0, separator=", "),
                np.array2string(KERNEL, precision=0, separator=", "),
                np.array2string(OUTPUT, precision=0, separator=", "),
                OUTPUT[0, 0],
                [(fmt_int(x_val), fmt_int(k_val), fmt_int(product)) for _, _, x_val, k_val, product in first_terms],
                np.array2string(running, precision=0, separator=", "),
                OUTPUT[0, 1],
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "06.2")

        x_squares, x_labels = self.make_cells(INPUT_X, self.x_origin, self.x_cell, BLUE_D, 28, WHITE)
        k_squares, k_labels = self.make_cells(KERNEL, self.k_origin, self.k_cell, YELLOW_A, 24, YELLOW_A)
        o_shape = OUTPUT.shape
        o_squares = [
            [
                self.stroke_square(self.o_cell * self.square_scale, BLUE_A).move_to(
                    self.cell_center(self.o_origin, self.o_cell, o_shape, row, col)
                )
                for col in range(o_shape[1])
            ]
            for row in range(o_shape[0])
        ]
        x_group = VGroup(*[square for row in x_squares for square in row])
        x_label_group = VGroup(*[label for row in x_labels for label in row])
        k_group = VGroup(*[square for row in k_squares for square in row])
        k_label_group = VGroup(*[label for row in k_labels for label in row])
        o_group = VGroup(*[square for row in o_squares for square in row])
        x_name = Text("X", font_size=32, color=BLUE_A).next_to(x_group, np.array([-1.0, 0.0, 0.0]), buff=0.22)
        k_name = Text("K", font_size=32, color=YELLOW_A).next_to(k_group, np.array([0.0, 1.0, 0.0]), buff=0.20)
        o_name = Text("O", font_size=32, color=BLUE_A).next_to(o_group, np.array([0.0, 1.0, 0.0]), buff=0.48)
        formula = Text("o = (X * K)", font_size=34, color=WHITE).move_to(np.array([0.0, 3.22, 0.0]))

        first_center = o_squares[0][0].get_center()
        neighbor_center = o_squares[0][1].get_center()
        halo_outer, halo_inner = self.make_halo(first_center)
        first_patch = self.patch_squares(x_squares, 0, 0)
        neighbor_patch = self.patch_squares(x_squares, 0, 1)
        window = self.make_window(first_patch)
        term_anchor = window.get_top() + np.array([0.0, 0.42, 0.0])

        sum_tracker = ValueTracker(0.0)
        first_readout = DecimalNumber(
            0,
            num_decimal_places=0,
            mob_class=Text,
            include_sign=False,
            color=YELLOW,
            font_size=36,
        )
        first_readout.move_to(first_center)
        first_readout.add_updater(
            lambda mob: mob.set_value(sum_tracker.get_value()).move_to(first_center)
        )
        neighbor_readout = Text(fmt_int(OUTPUT[0, 1]), font_size=36, color=YELLOW).move_to(neighbor_center)

        # 0.40–3.20 s: input grid first, so the later window has somewhere to sit.
        self.play(
            LaggedStart(
                *[FadeIn(mobject) for pair in zip(x_group, x_label_group) for mobject in pair],
                lag_ratio=0.03,
            ),
            FadeIn(x_name),
            run_time=1.15,
            rate_func=linear,
        )
        self.wait(1.20)

        # 3.20–4.55 s: the untrained kernel, numbers visible, between X and O.
        self.play(
            LaggedStart(
                *[FadeIn(mobject) for pair in zip(k_group, k_label_group) for mobject in pair],
                lag_ratio=0.04,
            ),
            FadeIn(k_name),
            run_time=0.85,
            rate_func=smooth,
        )
        self.wait(0.50)

        # 4.55–6.10 s: empty 2×2 output (not a feature-map dashboard) and one formula.
        self.play(
            LaggedStart(*[FadeIn(square) for square in o_group], lag_ratio=0.08),
            FadeIn(o_name),
            FadeIn(formula),
            run_time=0.80,
            rate_func=smooth,
        )
        self.wait(0.55)

        # 6.10–7.70 s: window on the top-left patch; halo is the cell being written.
        self.play(Create(window), run_time=0.55, rate_func=smooth)
        self.play(
            Create(halo_outer),
            Create(halo_inner),
            FadeIn(first_readout),
            Flash(first_center, color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=0.70,
            rate_func=smooth,
        )
        self.wait(0.35)

        # 7.70–15.35 s: nine multiply-adds for O[0,0], matching numpy / D2L corr2d.
        term_mob: Text | None = None
        for index, (row, col, x_val, k_val, _product) in enumerate(first_terms):
            x_square = x_squares[row][col]
            x_label = x_labels[row][col]
            k_square = k_squares[row][col]
            k_label = k_labels[row][col]
            next_term = Text(
                f"{fmt_int(x_val)} × {fmt_int(k_val)} = {fmt_int(_product)}",
                font_size=24,
                color=YELLOW_A,
            )
            next_term.move_to(term_anchor)
            highlight = [
                x_square.animate.set_stroke(YELLOW, width=5.2, opacity=1.0),
                x_label.animate.set_color(YELLOW),
                k_square.animate.set_stroke(YELLOW, width=5.2, opacity=1.0),
                k_label.animate.set_color(YELLOW),
            ]
            if term_mob is None:
                self.play(
                    *highlight,
                    FadeIn(next_term),
                    sum_tracker.animate.set_value(float(running[index])),
                    run_time=0.62,
                    rate_func=smooth,
                )
            else:
                # Instant swap: morphing Text turns `0 × 0 = 0` into an unreadable
                # overlay, and a same-slot crossfade collides the two equations.
                self.remove(term_mob)
                self.add(next_term)
                self.play(
                    *highlight,
                    sum_tracker.animate.set_value(float(running[index])),
                    run_time=0.62,
                    rate_func=smooth,
                )
            term_mob = next_term
            self.play(
                x_square.animate.set_stroke(BLUE_D, width=2.0, opacity=0.90),
                x_label.animate.set_color(WHITE),
                k_square.animate.set_stroke(YELLOW_A, width=2.0, opacity=0.90),
                k_label.animate.set_color(YELLOW_A),
                run_time=0.16,
                rate_func=smooth,
            )
        first_readout.clear_updaters()
        first_readout.set_value(float(OUTPUT[0, 0]))
        first_readout.move_to(first_center)
        self.play(FadeOut(term_mob), run_time=0.22, rate_func=linear)
        self.play(
            Flash(first_center, color=YELLOW, flash_radius=0.46, line_length=0.12),
            run_time=0.50,
            rate_func=smooth,
        )
        self.wait(0.65)

        # 16.00–18.20 s: kernel window steps one cell; halo reappears on the neighbor
        # instead of sliding across the already-written 26.
        stepped_window = self.make_window(neighbor_patch)
        self.play(Transform(window, stepped_window), run_time=1.20, rate_func=smooth)
        self.play(FadeOut(halo_outer), FadeOut(halo_inner), run_time=0.22, rate_func=linear)
        halo_outer, halo_inner = self.make_halo(neighbor_center)
        self.play(
            Create(halo_outer),
            Create(halo_inner),
            run_time=0.40,
            rate_func=smooth,
        )
        self.wait(0.50)

        # 18.00–20.40 s: the neighbor output cell lights; no second 9-term lesson.
        self.play(
            LaggedStart(
                *[
                    x_squares[row][col].animate.set_stroke(YELLOW, width=4.6, opacity=1.0)
                    for row in range(KERNEL_H)
                    for col in range(1, 1 + KERNEL_W)
                ],
                lag_ratio=0.04,
            ),
            run_time=0.55,
            rate_func=smooth,
        )
        self.play(
            FadeIn(neighbor_readout),
            Flash(neighbor_center, color=YELLOW, flash_radius=0.46, line_length=0.12),
            run_time=0.60,
            rate_func=smooth,
        )
        self.play(
            LaggedStart(
                *[
                    x_squares[row][col].animate.set_stroke(BLUE_D, width=2.0, opacity=0.90)
                    for row in range(KERNEL_H)
                    for col in range(1, 1 + KERNEL_W)
                ],
                lag_ratio=0.03,
            ),
            run_time=0.40,
            rate_func=smooth,
        )

        # Hold both written cells, empty bottom row, and the traveler halo.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=11.20, rate_func=linear)
