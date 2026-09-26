"""D2L 6.3 — same kernel, padding and stride change the output lattice.

Every output height, width, and cell value comes from the arrays below.
The kernel is never trained. Output shape is always
``floor((n - k + 2p) / s) + 1``, checked against the numpy lattice.
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
    always_redraw,
)


config.background_color = BLACK


# D2L 6.2 / figure 6.3.1 tensors: a real 3 × 3 input and a 2 × 2 kernel.
INPUT_X = np.array(
    [
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
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

N = INPUT_X.shape[0]
K_SIZE = KERNEL.shape[0]
# Layout reserves a one-cell halo so padding can appear without moving X.
LAYOUT_PAD = 1

# The traveler is one output cell: kernel top-left at (−1, 0).
# Missing with no padding, present with p=1 s=1, skipped by s=2.
TRAVELER = (-1, 0)


def output_size(n: int, k: int, padding: int, stride: int) -> int:
    """D2L output length on one axis: floor((n − k + 2p) / s) + 1."""
    return int(np.floor((n - k + 2 * padding) / stride) + 1)


def corr2d(x: np.ndarray, kernel: np.ndarray, padding: int, stride: int) -> np.ndarray:
    """Cross-correlation that matches D2L (no kernel flip)."""
    padded = np.pad(x, padding)
    kernel_h, kernel_w = kernel.shape
    out_h = (padded.shape[0] - kernel_h) // stride + 1
    out_w = (padded.shape[1] - kernel_w) // stride + 1
    result = np.zeros((out_h, out_w), dtype=float)
    for row in range(out_h):
        for col in range(out_w):
            window = padded[
                row * stride : row * stride + kernel_h,
                col * stride : col * stride + kernel_w,
            ]
            result[row, col] = float(np.sum(window * kernel))
    return result


def lattice(padding: int, stride: int) -> dict[tuple[int, int], float]:
    """Map kernel top-left (r, c) in original coordinates to the output value."""
    values = corr2d(x=INPUT_X, kernel=KERNEL, padding=padding, stride=stride)
    cells: dict[tuple[int, int], float] = {}
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            origin = (-padding + row * stride, -padding + col * stride)
            cells[origin] = float(values[row, col])
    return cells


def fmt(value: float) -> str:
    rounded = int(round(value))
    if abs(value - rounded) < 1e-9:
        return str(rounded)
    return f"{value:.1f}"


Y_P0_S1 = corr2d(INPUT_X, KERNEL, padding=0, stride=1)
Y_P1_S1 = corr2d(INPUT_X, KERNEL, padding=1, stride=1)
Y_P1_S2 = corr2d(INPUT_X, KERNEL, padding=1, stride=2)
LATTICE_P0_S1 = lattice(0, 1)
LATTICE_P1_S1 = lattice(1, 1)
LATTICE_P1_S2 = lattice(1, 2)


class Episode063(Scene):
    """A ~33-second continuous, silent padding-and-stride visualization."""

    cell = 0.78
    input_left = -6.48
    input_top = 2.02
    output_left = 1.08
    kernel_left = -1.92
    kernel_top = 3.66
    kernel_cell = 0.50

    def edge_x(self, col: float) -> float:
        return self.input_left + (col + LAYOUT_PAD) * self.cell

    def edge_y(self, row: float) -> float:
        return self.input_top - (row + LAYOUT_PAD) * self.cell

    def input_center(self, row: float, col: float) -> np.ndarray:
        return np.array(
            [self.edge_x(col) + 0.5 * self.cell, self.edge_y(row) - 0.5 * self.cell, 0.0]
        )

    def output_center(self, row: float, col: float) -> np.ndarray:
        return np.array(
            [
                self.output_left + (col + LAYOUT_PAD + 0.5) * self.cell,
                self.edge_y(row) - 0.5 * self.cell,
                0.0,
            ]
        )

    def stroke_box(self, center: np.ndarray, half: float, color, width: float, opacity: float) -> VGroup:
        """Four stroked sides. Never fill — filled grids read as white slabs."""
        x_value, y_value = center[0], center[1]
        box = VGroup(
            Line(np.array([x_value - half, y_value + half, 0.0]), np.array([x_value + half, y_value + half, 0.0])),
            Line(np.array([x_value + half, y_value + half, 0.0]), np.array([x_value + half, y_value - half, 0.0])),
            Line(np.array([x_value + half, y_value - half, 0.0]), np.array([x_value - half, y_value - half, 0.0])),
            Line(np.array([x_value - half, y_value - half, 0.0]), np.array([x_value - half, y_value + half, 0.0])),
        )
        box.set_stroke(color, width=width, opacity=opacity)
        box.set_fill(opacity=0.0)
        return box

    def labeled_cell(
        self,
        center: np.ndarray,
        value: float,
        box_color,
        text_color,
        font_size: int,
        half: float,
        stroke_width: float = 2.0,
        stroke_opacity: float = 0.94,
    ) -> VGroup:
        box = self.stroke_box(center, half, box_color, stroke_width, stroke_opacity)
        label = Text(fmt(value), font_size=font_size, color=text_color).move_to(center)
        return VGroup(box, label)

    def h_brace(
        self,
        x_left: float,
        x_right: float,
        y_value: float,
        label: str,
        color,
        ticks_toward: float,
        label_dy: float,
    ) -> VGroup:
        """Horizontal size brace. ``ticks_toward`` is +1 (up at the object) or −1 (down)."""
        tick = 0.11 * ticks_toward
        bar = VGroup(
            Line(np.array([x_left, y_value + tick, 0.0]), np.array([x_left, y_value, 0.0])),
            Line(np.array([x_left, y_value, 0.0]), np.array([x_right, y_value, 0.0])),
            Line(np.array([x_right, y_value, 0.0]), np.array([x_right, y_value + tick, 0.0])),
        ).set_stroke(color, width=1.8, opacity=0.90)
        bar.set_fill(opacity=0.0)
        text = Text(label, font_size=22, color=color).move_to(
            np.array([0.5 * (x_left + x_right), y_value + label_dy, 0.0])
        )
        return VGroup(bar, text)

    def construct(self) -> None:
        cases = ((0, 1, Y_P0_S1), (1, 1, Y_P1_S1), (1, 2, Y_P1_S2))
        for padding, stride, values in cases:
            formula_n = output_size(N, K_SIZE, padding, stride)
            assert values.shape == (formula_n, formula_n)
            print(
                "D2L 6.3 (p,s)=({}, {}) formula={} Y.shape={} Y={}".format(
                    padding,
                    stride,
                    formula_n,
                    values.shape,
                    np.array2string(values, precision=0, separator=", "),
                )
            )
        print(
            "D2L 6.3 X={}, K={}, traveler origin={}, traveler value={}".format(
                np.array2string(INPUT_X, precision=0, separator=", "),
                np.array2string(KERNEL, precision=0, separator=", "),
                TRAVELER,
                fmt(LATTICE_P1_S1[TRAVELER]),
            )
        )
        assert TRAVELER not in LATTICE_P0_S1
        assert TRAVELER in LATTICE_P1_S1
        assert TRAVELER not in LATTICE_P1_S2

        inner_half = 0.46 * self.cell
        output_half = 0.44 * self.cell
        kernel_half = 0.44 * self.kernel_cell

        x_cells = VGroup(
            *[
                self.labeled_cell(self.input_center(row, col), INPUT_X[row, col], BLUE, WHITE, 24, inner_half)
                for row in range(N)
                for col in range(N)
            ]
        )
        pad_cells = [
            self.labeled_cell(
                self.input_center(row, col),
                0.0,
                BLUE_E,
                BLUE_A,
                20,
                inner_half,
                stroke_width=1.6,
                stroke_opacity=0.78,
            )
            for row in range(-LAYOUT_PAD, N + LAYOUT_PAD)
            for col in range(-LAYOUT_PAD, N + LAYOUT_PAD)
            if not (0 <= row < N and 0 <= col < N)
        ]
        pad_group = VGroup(*pad_cells)

        k_cells = VGroup(
            *[
                self.labeled_cell(
                    np.array(
                        [
                            self.kernel_left + (col + 0.5) * self.kernel_cell,
                            self.kernel_top - (row + 0.5) * self.kernel_cell,
                            0.0,
                        ]
                    ),
                    KERNEL[row, col],
                    YELLOW,
                    YELLOW_A,
                    22,
                    kernel_half,
                    stroke_width=2.2,
                )
                for row in range(K_SIZE)
                for col in range(K_SIZE)
            ]
        )
        k_name = Text("K", font_size=28, color=YELLOW).next_to(k_cells, np.array([-1.0, 0.0, 0.0]), buff=0.16)
        k_brace = self.h_brace(
            self.kernel_left,
            self.kernel_left + K_SIZE * self.kernel_cell,
            self.kernel_top - K_SIZE * self.kernel_cell - 0.06,
            "k = 2",
            YELLOW_A,
            ticks_toward=1.0,
            label_dy=-0.28,
        )

        x_name = Text("X", font_size=30, color=BLUE_A).move_to(np.array([-4.55, 3.22, 0.0]))
        y_name = Text("Y", font_size=30, color=BLUE_A).move_to(np.array([1.58, 3.22, 0.0]))
        # ASCII floor: the default font maps U+230A/B to a bar, which reads as |·|.
        formula = Text("floor((n − k + 2p) / s) + 1", font_size=24, color=WHITE).move_to(
            np.array([4.95, 3.22, 0.0])
        )
        s_one = Text("s = 1", font_size=22, color=YELLOW_A).move_to(
            np.array(
                [
                    self.kernel_left + K_SIZE * self.kernel_cell + 0.62,
                    self.kernel_top - K_SIZE * self.kernel_cell - 0.34,
                    0.0,
                ]
            )
        )
        s_two = Text("s = 2", font_size=22, color=YELLOW_A).move_to(s_one.get_center())

        # n measures the inner 3 above the pad ring; p measures the zeros below it.
        n_brace = self.h_brace(
            self.edge_x(0),
            self.edge_x(N),
            self.edge_y(-LAYOUT_PAD) + 0.20,
            "n = 3",
            BLUE_A,
            ticks_toward=-1.0,
            label_dy=0.28,
        )
        p_zero = Text("p = 0", font_size=22, color=BLUE_A).move_to(
            np.array([0.5 * (self.edge_x(0) + self.edge_x(N)), self.edge_y(N) - 0.42, 0.0])
        )
        p_brace = self.h_brace(
            self.edge_x(-LAYOUT_PAD),
            self.edge_x(N + LAYOUT_PAD),
            self.edge_y(N + LAYOUT_PAD) - 0.16,
            "p = 1",
            BLUE_A,
            ticks_toward=1.0,
            label_dy=-0.28,
        )

        size_anchor = np.array(
            [
                0.5 * (self.output_center(-1, -1)[0] + self.output_center(2, 2)[0]),
                self.output_center(2, 1)[1] - 0.78,
                0.0,
            ]
        )
        size_two = Text("2 × 2", font_size=28, color=WHITE).move_to(size_anchor)
        size_four = Text("4 × 4", font_size=28, color=WHITE).move_to(size_anchor)
        size_two_again = Text("2 × 2", font_size=28, color=WHITE).move_to(size_anchor)

        y_mobs: dict[tuple[int, int], VGroup] = {}
        for origin, value in LATTICE_P1_S1.items():
            is_traveler = origin == TRAVELER
            y_mobs[origin] = self.labeled_cell(
                self.output_center(*origin),
                value,
                YELLOW if is_traveler else BLUE,
                YELLOW_A if is_traveler else WHITE,
                20,
                output_half,
                stroke_width=2.6 if is_traveler else 2.0,
            )

        traveler_center = self.output_center(*TRAVELER)
        traveler_ring = Circle(radius=0.44).move_to(traveler_center)
        traveler_ring.set_stroke(BLUE_A, width=2.0, opacity=0.72)
        traveler_ring.set_fill(opacity=0.0)
        halo_outer = DashedVMobject(traveler_ring, num_dashes=18)
        halo_inner = Circle(radius=0.31).move_to(traveler_center).set_stroke(YELLOW, width=2.8, opacity=0.98)
        halo_inner.set_fill(opacity=0.0)

        row_tracker = ValueTracker(0.0)
        col_tracker = ValueTracker(0.0)

        def window_box() -> VGroup:
            row = row_tracker.get_value()
            col = col_tracker.get_value()
            x0 = self.edge_x(col) - 0.04
            x1 = self.edge_x(col + K_SIZE) + 0.04
            y1 = self.edge_y(row) + 0.04
            y0 = self.edge_y(row + K_SIZE) - 0.04
            box = VGroup(
                Line(np.array([x0, y1, 0.0]), np.array([x1, y1, 0.0])),
                Line(np.array([x1, y1, 0.0]), np.array([x1, y0, 0.0])),
                Line(np.array([x1, y0, 0.0]), np.array([x0, y0, 0.0])),
                Line(np.array([x0, y0, 0.0]), np.array([x0, y1, 0.0])),
            )
            box.set_stroke(YELLOW, width=4.0, opacity=0.96)
            box.set_fill(opacity=0.0)
            return box

        window = always_redraw(window_box)

        map_arrow = VGroup(
            Line(
                np.array([self.edge_x(N) + 0.28, self.input_center(1, 1)[1], 0.0]),
                np.array([self.output_left - 0.22, self.input_center(1, 1)[1], 0.0]),
            ),
            Line(
                np.array([self.output_left - 0.22, self.input_center(1, 1)[1], 0.0]),
                np.array([self.output_left - 0.38, self.input_center(1, 1)[1] + 0.12, 0.0]),
            ),
            Line(
                np.array([self.output_left - 0.22, self.input_center(1, 1)[1], 0.0]),
                np.array([self.output_left - 0.38, self.input_center(1, 1)[1] - 0.12, 0.0]),
            ),
        )
        map_arrow.set_stroke(BLUE_D, width=2.2, opacity=0.72)
        map_arrow.set_fill(opacity=0.0)

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "06.3")

        # 0.40–1.40 s: the real 3 × 3 input, with n on that lattice.
        self.play(
            LaggedStart(*[FadeIn(cell) for cell in x_cells], lag_ratio=0.06),
            FadeIn(x_name),
            FadeIn(n_brace),
            run_time=1.00,
            rate_func=linear,
        )

        # 1.40–2.20 s: the same kernel reused on every later beat.
        self.play(FadeIn(k_cells), FadeIn(k_name), FadeIn(k_brace), run_time=0.80, rate_func=smooth)

        # 2.20–3.20 s: output-size formula once; p starts at 0.
        self.play(
            FadeIn(formula),
            FadeIn(p_zero),
            FadeIn(map_arrow),
            FadeIn(y_name),
            run_time=1.00,
            rate_func=smooth,
        )

        # 3.20–4.10 s: stride-1 window on the unpadded lattice.
        self.play(FadeIn(window), FadeIn(s_one), run_time=0.90, rate_func=smooth)

        # 4.10–6.90 s: four adjacent writes; output lattice is 2 × 2.
        for origin in ((0, 0), (0, 1), (1, 0), (1, 1)):
            self.play(
                row_tracker.animate.set_value(origin[0]),
                col_tracker.animate.set_value(origin[1]),
                FadeIn(y_mobs[origin]),
                run_time=0.70,
                rate_func=smooth,
            )

        # 6.90–9.30 s: hold the small lattice; dashed halo marks the missing edge cell.
        self.play(FadeIn(size_two), Create(halo_outer), run_time=0.80, rate_func=smooth)
        self.wait(1.60)

        # 9.30–10.80 s: zeros around X. The output lattice can now grow.
        self.play(
            LaggedStart(*[FadeIn(cell) for cell in pad_group], lag_ratio=0.04),
            FadeOut(p_zero),
            FadeIn(p_brace),
            run_time=1.50,
            rate_func=smooth,
        )

        # 10.80–11.80 s: the window sits on the traveler’s padded origin.
        self.play(
            row_tracker.animate.set_value(TRAVELER[0]),
            col_tracker.animate.set_value(TRAVELER[1]),
            run_time=1.00,
            rate_func=smooth,
        )

        # 11.80–14.20 s: edge cells appear; the haloed traveler is one of them.
        new_cells = [
            y_mobs[origin] for origin in LATTICE_P1_S1 if origin not in LATTICE_P0_S1 and origin != TRAVELER
        ]
        self.play(
            LaggedStart(*[FadeIn(cell) for cell in new_cells], lag_ratio=0.05),
            run_time=1.35,
            rate_func=linear,
        )
        self.play(
            FadeIn(y_mobs[TRAVELER]),
            Create(halo_inner),
            Flash(traveler_center, color=YELLOW, flash_radius=0.50, line_length=0.12),
            FadeOut(size_two),
            FadeIn(size_four),
            run_time=1.05,
            rate_func=smooth,
        )

        # 14.20–19.50 s: hold the grown 4 × 4 so the new edge cells can be read.
        self.wait(5.30)

        # 19.50–20.70 s: stride becomes 2; window returns to a kept corner.
        self.play(
            FadeOut(s_one),
            FadeIn(s_two),
            row_tracker.animate.set_value(-1.0),
            col_tracker.animate.set_value(-1.0),
            run_time=1.20,
            rate_func=smooth,
        )

        # 20.70–25.20 s: the kernel jumps; cells off the stride-2 lattice leave.
        extras = [y_mobs[origin] for origin in LATTICE_P1_S1 if origin not in LATTICE_P1_S2]
        self.play(
            col_tracker.animate.set_value(1.0),
            FadeOut(VGroup(*extras)),
            FadeOut(halo_inner),
            FadeOut(size_four),
            FadeIn(size_two_again),
            run_time=1.70,
            rate_func=smooth,
        )
        self.wait(0.80)
        self.play(
            row_tracker.animate.set_value(1.0),
            col_tracker.animate.set_value(-1.0),
            run_time=1.00,
            rate_func=smooth,
        )
        self.play(col_tracker.animate.set_value(1.0), run_time=1.00, rate_func=smooth)

        # 25.20–33.20 s: sparse 2 × 2 remains; traveler hole stays marked.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=8.00, rate_func=linear)
