"""D2L 7.3 — 1×1 convolution as a per-pixel MLP, then GAP to a class vector.

Every on-screen number comes from the arrays, ``conv1x1``, and
``global_avg_pool`` below. This is forward computation only: nothing is trained.
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


# Tiny real tensors. X is Cin×H×W = 2×2×2. W is a 1×1 kernel as Cout×Cin.
# No bias, so every cell stays an integer. Untrained.
INPUT_X = np.array(
    [
        [
            [1.0, 3.0],
            [2.0, 4.0],
        ],
        [
            [0.0, 1.0],
            [2.0, 3.0],
        ],
    ],
    dtype=float,
)
WEIGHT_1X1 = np.array(
    [
        [1.0, 1.0],
        [2.0, 0.0],
        [0.0, 2.0],
    ],
    dtype=float,
)
TRAVELER = (0, 0)


def conv1x1(inputs: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Channel mix at every pixel. Spatial size is unchanged."""
    in_channels, height, width = inputs.shape
    mixed = weight @ inputs.reshape(in_channels, height * width)
    return mixed.reshape(weight.shape[0], height, width)


def global_avg_pool(feature_maps: np.ndarray) -> np.ndarray:
    """Mean over H and W for each channel. Numpy is the only arithmetic."""
    return feature_maps.mean(axis=(1, 2))


FEATURES = conv1x1(INPUT_X, WEIGHT_1X1)
LOGITS = global_avg_pool(FEATURES)
assert FEATURES.shape[1:] == INPUT_X.shape[1:]
assert FEATURES.shape[0] == WEIGHT_1X1.shape[0]
assert LOGITS.shape == (WEIGHT_1X1.shape[0],)
assert np.allclose(FEATURES, (WEIGHT_1X1 @ INPUT_X.reshape(2, 4)).reshape(3, 2, 2))
assert np.allclose(LOGITS, np.array([4.0, 5.0, 3.0]))


def fmt_int(value: float) -> str:
    """Integer glyph for a cell; use a true minus if a later weight needs it."""
    rounded = int(round(value))
    if rounded < 0:
        return f"−{abs(rounded)}"
    return str(rounded)


class Episode073(Scene):
    """A ~32-second continuous, silent NiN 1×1-then-GAP visualization."""

    x_cell = 0.86
    w_cell = 0.58
    h_cell = 0.86
    o_cell = 0.92
    square_scale = 0.90
    x_centers = (np.array([-5.05, 1.92, 0.0]), np.array([-5.05, -0.88, 0.0]))
    w_center = np.array([-2.05, 0.52, 0.0])
    h_centers = (
        np.array([1.28, 2.18, 0.0]),
        np.array([1.28, 0.08, 0.0]),
        np.array([1.28, -2.02, 0.0]),
    )
    o_centers = (
        np.array([4.72, 2.18, 0.0]),
        np.array([4.72, 0.08, 0.0]),
        np.array([4.72, -2.02, 0.0]),
    )
    x_colors = (BLUE, BLUE_A)
    h_colors = (BLUE, BLUE_A, BLUE_D)

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

    def make_window(self, target: Square | VGroup, buff: float = 0.07) -> SurroundingRectangle:
        window = SurroundingRectangle(target, color=YELLOW, buff=buff, corner_radius=0.04)
        window.set_fill(BLACK, opacity=0.0)
        window.set_stroke(YELLOW, width=3.4, opacity=0.98)
        return window

    def make_halo(self, center: np.ndarray, outer_radius: float = 0.44, inner_radius: float = 0.32) -> tuple[Circle, Circle]:
        outer = Circle(radius=outer_radius).move_to(center).set_stroke(BLUE_A, width=2.1, opacity=0.50)
        inner = Circle(radius=inner_radius).move_to(center).set_stroke(YELLOW, width=2.8, opacity=0.98)
        return outer, inner

    def construct(self) -> None:
        traveler_x = INPUT_X[:, TRAVELER[0], TRAVELER[1]]
        traveler_h = FEATURES[:, TRAVELER[0], TRAVELER[1]]
        print(
            "D2L 7.3 X={}\nW={}\nH={}\no={}\ntraveler x={} h={} o1={:.0f}".format(
                np.array2string(INPUT_X, precision=0, separator=", "),
                np.array2string(WEIGHT_1X1, precision=0, separator=", "),
                np.array2string(FEATURES, precision=0, separator=", "),
                np.array2string(LOGITS, precision=0, separator=", "),
                np.array2string(traveler_x, precision=0, separator=", "),
                np.array2string(traveler_h, precision=0, separator=", "),
                float(LOGITS[0]),
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "07.3")

        x_squares: list[list[list[Square]]] = []
        x_labels: list[list[list[Text]]] = []
        x_groups = []
        x_label_groups = []
        x_names = []
        for index in range(2):
            squares, labels = self.make_cells(
                INPUT_X[index],
                self.x_centers[index],
                self.x_cell,
                self.x_colors[index],
                28,
                WHITE,
            )
            x_squares.append(squares)
            x_labels.append(labels)
            x_group = VGroup(*[square for row in squares for square in row])
            x_label_group = VGroup(*[label for row in labels for label in row])
            x_groups.append(x_group)
            x_label_groups.append(x_label_group)
            x_names.append(
                Text(f"X{'₁₂'[index]}", font_size=28, color=self.x_colors[index]).next_to(
                    x_group, np.array([-1.0, 0.0, 0.0]), buff=0.18
                )
            )

        w_squares, w_labels = self.make_cells(
            WEIGHT_1X1,
            self.w_center,
            self.w_cell,
            YELLOW_A,
            22,
            YELLOW_A,
        )
        w_group = VGroup(*[square for row in w_squares for square in row])
        w_label_group = VGroup(*[label for row in w_labels for label in row])
        w_name = Text("1×1", font_size=28, color=YELLOW_A).next_to(
            w_group, np.array([0.0, 1.0, 0.0]), buff=0.16
        )

        mix_formula = Text("h = Wx", font_size=32, color=WHITE).move_to(np.array([1.55, 3.42, 0.0]))
        gap_formula = Text("o = mean(H)", font_size=32, color=WHITE).move_to(np.array([1.55, 3.42, 0.0]))

        h_squares: list[list[list[Square]]] = []
        h_groups = []
        h_names = []
        for index in range(3):
            squares = [
                [
                    self.stroke_square(self.h_cell * self.square_scale, self.h_colors[index]).move_to(
                        self.cell_center(self.h_centers[index], self.h_cell, (2, 2), row, col)
                    )
                    for col in range(2)
                ]
                for row in range(2)
            ]
            h_squares.append(squares)
            h_group = VGroup(*[square for row in squares for square in row])
            h_groups.append(h_group)
            h_names.append(
                Text(f"H{'₁₂₃'[index]}", font_size=26, color=self.h_colors[index]).next_to(
                    h_group, np.array([-1.0, 0.0, 0.0]), buff=0.16
                )
            )

        traveler_h_labels = [
            Text(fmt_int(FEATURES[index, TRAVELER[0], TRAVELER[1]]), font_size=28, color=WHITE).move_to(
                h_squares[index][TRAVELER[0]][TRAVELER[1]].get_center()
            )
            for index in range(3)
        ]
        rest_h_labels = [
            Text(fmt_int(FEATURES[index, row, col]), font_size=28, color=WHITE).move_to(
                h_squares[index][row][col].get_center()
            )
            for index in range(3)
            for row in range(2)
            for col in range(2)
            if (row, col) != TRAVELER
        ]

        o_squares = [
            self.stroke_square(self.o_cell * self.square_scale, self.h_colors[index], width=2.4).move_to(
                self.o_centers[index]
            )
            for index in range(3)
        ]
        o_names = [
            Text(f"o{'₁₂₃'[index]}", font_size=26, color=self.h_colors[index]).next_to(
                o_squares[index], np.array([0.0, 1.0, 0.0]), buff=0.12
            )
            for index in range(3)
        ]
        o_labels = [
            Text(
                fmt_int(LOGITS[index]),
                font_size=34,
                color=YELLOW if index == 0 else WHITE,
            ).move_to(self.o_centers[index])
            for index in range(3)
        ]

        traveler_x_center = x_squares[0][TRAVELER[0]][TRAVELER[1]].get_center()
        traveler_h_center = h_squares[0][TRAVELER[0]][TRAVELER[1]].get_center()
        traveler_o_center = o_squares[0].get_center()
        halo_outer, halo_inner = self.make_halo(traveler_x_center)
        halo_at_h_outer, halo_at_h_inner = self.make_halo(traveler_h_center)
        halo_at_o_outer, halo_at_o_inner = self.make_halo(
            traveler_o_center, outer_radius=0.50, inner_radius=0.38
        )
        window_x1 = self.make_window(x_squares[0][TRAVELER[0]][TRAVELER[1]])
        window_x2 = self.make_window(x_squares[1][TRAVELER[0]][TRAVELER[1]])
        gap_windows = [self.make_window(h_groups[index], buff=0.10) for index in range(3)]

        # Fixed empty pocket: two lines, never a vector HUD.
        pocket_x1 = Text("x₁ = 1", font_size=22, color=BLUE_A).move_to(np.array([-5.05, -2.72, 0.0]))
        pocket_x2 = Text("x₂ = 0", font_size=22, color=BLUE_A).move_to(np.array([-5.05, -3.14, 0.0]))

        def highlight_pixel():
            animations = []
            for index in range(2):
                animations.extend(
                    [
                        x_squares[index][TRAVELER[0]][TRAVELER[1]]
                        .animate.set_stroke(YELLOW, width=5.0, opacity=1.0),
                        x_labels[index][TRAVELER[0]][TRAVELER[1]].animate.set_color(YELLOW),
                    ]
                )
            for row in range(3):
                for col in range(2):
                    animations.extend(
                        [
                            w_squares[row][col].animate.set_stroke(YELLOW, width=5.0, opacity=1.0),
                            w_labels[row][col].animate.set_color(YELLOW),
                        ]
                    )
            return animations

        def restore_pixel():
            animations = []
            for index in range(2):
                animations.extend(
                    [
                        x_squares[index][TRAVELER[0]][TRAVELER[1]]
                        .animate.set_stroke(self.x_colors[index], width=2.0, opacity=0.90),
                        x_labels[index][TRAVELER[0]][TRAVELER[1]].animate.set_color(WHITE),
                    ]
                )
            for row in range(3):
                for col in range(2):
                    animations.extend(
                        [
                            w_squares[row][col].animate.set_stroke(YELLOW_A, width=2.0, opacity=0.90),
                            w_labels[row][col].animate.set_color(YELLOW_A),
                        ]
                    )
            return animations

        # 0.40–4.20 s: two input slices first, held before the 1×1.
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

        # 4.20–8.30 s: 1×1 weights, empty H maps (same H×W), mix formula.
        self.play(
            LaggedStart(
                *[FadeIn(mobject) for pair in zip(w_group, w_label_group) for mobject in pair],
                lag_ratio=0.05,
            ),
            FadeIn(w_name),
            run_time=1.10,
            rate_func=smooth,
        )
        self.play(
            LaggedStart(*[FadeIn(square) for group in h_groups for square in group], lag_ratio=0.05),
            *[FadeIn(name) for name in h_names],
            FadeIn(mix_formula),
            run_time=1.20,
            rate_func=smooth,
        )
        self.wait(1.80)

        # 8.30–11.80 s: 1×1 window on one spatial cell of both slices, then h = Wx.
        self.play(Create(window_x1), Create(window_x2), run_time=0.70, rate_func=smooth)
        self.play(
            FadeIn(pocket_x1),
            FadeIn(pocket_x2),
            Create(halo_outer),
            Create(halo_inner),
            Flash(traveler_x_center, color=YELLOW, flash_radius=0.42, line_length=0.10),
            run_time=0.90,
            rate_func=smooth,
        )
        self.play(*highlight_pixel(), run_time=0.55, rate_func=smooth)
        self.play(
            LaggedStart(*[FadeIn(label) for label in traveler_h_labels], lag_ratio=0.18),
            run_time=1.35,
            rate_func=smooth,
        )
        self.play(*restore_pixel(), run_time=0.35, rate_func=smooth)

        # Halo stays on that spatial cell after the mix: now H₁[0,0].
        self.play(
            Transform(halo_outer, halo_at_h_outer),
            Transform(halo_inner, halo_at_h_inner),
            traveler_h_labels[0].animate.set_color(YELLOW),
            Flash(traveler_h_center, color=YELLOW, flash_radius=0.42, line_length=0.10),
            run_time=0.90,
            rate_func=smooth,
        )
        self.wait(0.40)

        # Other spatial cells of H appear; they are the same 1×1, not a second lesson.
        self.play(
            LaggedStart(*[FadeIn(label) for label in rest_h_labels], lag_ratio=0.08),
            run_time=1.50,
            rate_func=smooth,
        )
        self.wait(2.00)

        # GAP once: o = mean(H), empty class cells, then each map collapses to 1×1.
        self.play(Transform(mix_formula, gap_formula), run_time=0.80, rate_func=smooth)
        self.play(
            LaggedStart(*[FadeIn(square) for square in o_squares], lag_ratio=0.12),
            *[FadeIn(name) for name in o_names],
            run_time=1.00,
            rate_func=smooth,
        )
        for index in range(3):
            self.play(Create(gap_windows[index]), run_time=0.45, rate_func=smooth)
            self.play(FadeIn(o_labels[index]), run_time=0.35, rate_func=smooth)
            self.play(FadeOut(gap_windows[index]), run_time=0.25, rate_func=linear)

        self.play(
            Transform(halo_outer, halo_at_o_outer),
            Transform(halo_inner, halo_at_o_inner),
            traveler_h_labels[0].animate.set_color(WHITE),
            Flash(traveler_o_center, color=YELLOW, flash_radius=0.48, line_length=0.12),
            run_time=1.10,
            rate_func=smooth,
        )

        # Hold X, W, H maps, class vector, and traveler through the last frame.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=8.40, rate_func=linear)
