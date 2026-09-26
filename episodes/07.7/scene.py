"""D2L 7.7 — a dense block: each layer concatenates on channels, x ← [x, f(x)].

Every plotted number comes from the arrays below.  f is a tiny untrained
1×1 map plus ReLU.  New maps dock onto the stack; old cells keep their
values.  That is the contrast with 7.6, where F(x) is added.
"""

from __future__ import annotations

import numpy as np
from scripts.episode_intro import show_episode_intro
from manim import (
    BLACK,
    BLUE,
    BLUE_A,
    BLUE_D,
    Create,
    DecimalNumber,
    FadeIn,
    FadeOut,
    Flash,
    LaggedStart,
    Rectangle,
    RoundedRectangle,
    Scene,
    Square,
    Text,
    Transform,
    VGroup,
    VMobject,
    ValueTracker,
    WHITE,
    YELLOW,
    YELLOW_A,
    config,
    linear,
    smooth,
)


config.background_color = BLACK


# ---------------------------------------------------------------------------
# One numerical source of truth.  X is two 2×2 maps (the traveler).  Each
# dense layer is a 1×1 channel mix + ReLU, growth rate 1.  Concatenate on
# axis 0 (channels).  Nothing is trained.
# ---------------------------------------------------------------------------
INPUT_X = np.array(
    [
        [
            [1.00, 0.50],
            [0.20, 0.80],
        ],
        [
            [0.40, 0.10],
            [0.70, 0.30],
        ],
    ],
    dtype=float,
)
WEIGHTS_1 = np.array([[0.50, 0.40]], dtype=float)
BIAS_1 = np.array([0.10], dtype=float)
WEIGHTS_2 = np.array([[0.20, 0.30, 0.50]], dtype=float)
BIAS_2 = np.array([-0.05], dtype=float)


def relu(values: np.ndarray) -> np.ndarray:
    """Component-wise max(z, 0) for the same values drawn on screen."""
    return np.maximum(values, 0.0)


def conv1x1(maps: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Untrained 1×1 mix: each spatial cell is a dense map on its channel vector."""
    _channels, height, width = maps.shape
    out_channels = weight.shape[0]
    output = np.zeros((out_channels, height, width), dtype=float)
    for out_index in range(out_channels):
        for row in range(height):
            for col in range(width):
                output[out_index, row, col] = float(
                    np.dot(weight[out_index], maps[:, row, col]) + bias[out_index]
                )
    return relu(output)


def dense_concat(current: np.ndarray, layer_out: np.ndarray) -> np.ndarray:
    """Channel concat: x ← [x, f(x)], never an add."""
    return np.concatenate((current, layer_out), axis=0)


FEATURE_1 = conv1x1(INPUT_X, WEIGHTS_1, BIAS_1)
STACK_1 = dense_concat(INPUT_X, FEATURE_1)
FEATURE_2 = conv1x1(STACK_1, WEIGHTS_2, BIAS_2)
STACK_2 = dense_concat(STACK_1, FEATURE_2)


def format_cell(value: float) -> str:
    """Two decimals with a true minus glyph, never a hyphen-minus."""
    rounded = round(float(value), 2)
    if rounded < 0:
        return f"−{abs(rounded):.2f}"
    return f"{rounded:.2f}"


class Episode077(Scene):
    """A ~32-second continuous, silent dense-block visualization."""

    cell = 0.90
    map_y = 0.12
    stack_x = (-4.82, -2.64, -0.46, 1.72)
    spawn = np.array([5.38, -2.22, 0.0])
    f_center = np.array([5.38, 2.62, 0.0])
    pocket = np.array([-6.28, -2.62, 0.0])

    def map_origin(self, slot: int) -> np.ndarray:
        return np.array([self.stack_x[slot], self.map_y, 0.0])

    def spawn_origin(self) -> np.ndarray:
        return self.spawn.copy()

    def stroke_square(self, color, width: float = 2.0, opacity: float = 0.92) -> Square:
        """Stroke-only cell. Never fill; never call set_opacity on the mesh."""
        square = Square(side_length=self.cell * 0.90)
        square.set_fill(BLACK, opacity=0.0)
        square.set_stroke(color, width=width, opacity=opacity)
        return square

    def make_map(
        self,
        values: np.ndarray,
        origin: np.ndarray,
        stroke_color,
        title: str,
        title_color,
    ) -> VGroup:
        """One 2×2 feature map. Numbers sit on the cells they name."""
        squares = VGroup()
        numbers = VGroup()
        rows, cols = values.shape
        for row in range(rows):
            for col in range(cols):
                position = origin + np.array(
                    [
                        (col - (cols - 1) / 2.0) * self.cell,
                        ((rows - 1) / 2.0 - row) * self.cell,
                        0.0,
                    ]
                )
                square = self.stroke_square(stroke_color)
                square.move_to(position)
                number = Text(format_cell(values[row, col]), font_size=22, color=WHITE)
                number.move_to(position)
                squares.add(square)
                numbers.add(number)
        title_mob = Text(title, font_size=26, color=title_color).move_to(
            origin + np.array([0.0, self.cell + 0.40, 0.0])
        )
        return VGroup(squares, numbers, title_mob)

    def make_halo(self, channel_count: int) -> tuple[RoundedRectangle, RoundedRectangle]:
        """Stroke-only rounded rects around the current concatenated stack."""
        half = self.cell
        left = self.stack_x[0] - half
        right = self.stack_x[channel_count - 1] + half
        center = np.array([(left + right) / 2.0, self.map_y + 0.18, 0.0])
        width = right - left
        height = 2.0 * self.cell
        outer = RoundedRectangle(
            width=width + 0.70, height=height + 1.18, corner_radius=0.14
        )
        inner = RoundedRectangle(
            width=width + 0.38, height=height + 0.86, corner_radius=0.10
        )
        outer.move_to(center).set_fill(BLACK, opacity=0.0).set_stroke(
            BLUE_A, width=2.1, opacity=0.50
        )
        inner.move_to(center).set_fill(BLACK, opacity=0.0).set_stroke(
            YELLOW, width=2.8, opacity=0.98
        )
        return outer, inner

    def make_link(self, channel_count: int) -> VMobject:
        """Orthogonal stroke from the halo's top-right to f. Never through titles."""
        half = self.cell
        halo_right = self.stack_x[channel_count - 1] + half + 0.42
        halo_top = self.map_y + half + 0.82
        f_left = self.f_center + np.array([-0.85, 0.0, 0.0])
        link = VMobject(fill_opacity=0.0, stroke_opacity=0.78)
        link.set_points_as_corners(
            [
                np.array([halo_right, halo_top, 0.0]),
                np.array([halo_right, self.f_center[1], 0.0]),
                f_left,
            ]
        )
        link.set_fill(BLACK, opacity=0.0)
        link.set_stroke(BLUE_D, width=2.2, opacity=0.78)
        return link

    def construct(self) -> None:
        print("D2L 7.7 X=\n{}".format(np.array2string(INPUT_X, precision=2, separator=", ")))
        print("D2L 7.7 W1={}, b1={}".format(
            np.array2string(WEIGHTS_1.ravel(), precision=2, separator=", "),
            np.array2string(BIAS_1, precision=2, separator=", "),
        ))
        print("D2L 7.7 f1=\n{}".format(np.array2string(FEATURE_1, precision=3, separator=", ")))
        print("D2L 7.7 [X,f1] shape={} =\n{}".format(
            STACK_1.shape,
            np.array2string(STACK_1, precision=3, separator=", "),
        ))
        print("D2L 7.7 W2={}, b2={}".format(
            np.array2string(WEIGHTS_2.ravel(), precision=2, separator=", "),
            np.array2string(BIAS_2, precision=2, separator=", "),
        ))
        print("D2L 7.7 f2=\n{}".format(np.array2string(FEATURE_2, precision=3, separator=", ")))
        print("D2L 7.7 [X,f1,f2] shape={} =\n{}".format(
            STACK_2.shape,
            np.array2string(STACK_2, precision=3, separator=", "),
        ))
        print(
            "D2L 7.7 channels {} → {} → {}".format(
                INPUT_X.shape[0], STACK_1.shape[0], STACK_2.shape[0]
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "07.7")

        map_x0 = self.make_map(INPUT_X[0], self.map_origin(0), BLUE, "x₀", BLUE_A)
        map_x1 = self.make_map(INPUT_X[1], self.map_origin(1), BLUE_A, "x₁", BLUE_A)
        map_f1 = self.make_map(FEATURE_1[0], self.spawn_origin(), YELLOW, "f₁", YELLOW)
        map_f2 = self.make_map(FEATURE_2[0], self.spawn_origin(), YELLOW_A, "f₂", YELLOW_A)

        halo_outer, halo_inner = self.make_halo(2)
        halo_outer_3, halo_inner_3 = self.make_halo(3)
        halo_outer_4, halo_inner_4 = self.make_halo(4)

        c_prefix = Text("C = ", font_size=26, color=BLUE_A)
        c_number = DecimalNumber(
            2,
            num_decimal_places=0,
            mob_class=Text,
            include_sign=False,
            color=WHITE,
            font_size=26,
        )
        c_row = VGroup(c_prefix, c_number).arrange(np.array([1.0, 0.0, 0.0]), buff=0.08)
        c_row.move_to(self.pocket)

        formula = Text("x ← [x, f(x)]", font_size=34, color=WHITE).move_to(
            np.array([0.15, 3.18, 0.0])
        )

        f_box = Rectangle(width=1.70, height=1.18).move_to(self.f_center)
        f_box.set_fill(BLACK, opacity=0.0)
        f_box.set_stroke(BLUE_D, width=2.2, opacity=0.92)
        f_name = Text("f", font_size=32, color=BLUE_A).move_to(self.f_center)

        link = self.make_link(2)
        link_3 = self.make_link(3)
        link_4 = self.make_link(4)

        # 0.40–2.20 s: the two input maps. Short fade so t=1s is already readable.
        self.play(
            LaggedStart(FadeIn(map_x0), FadeIn(map_x1), lag_ratio=0.16),
            run_time=1.00,
            rate_func=smooth,
        )
        self.wait(0.80)

        # 2.20–4.20 s: halo traveler is the concatenated stack; pocket C = 2.
        self.play(
            Create(halo_outer),
            Create(halo_inner),
            FadeIn(c_row),
            Flash(self.map_origin(0), color=YELLOW, flash_radius=0.55, line_length=0.12),
            run_time=1.10,
            rate_func=smooth,
        )
        self.wait(0.90)

        # 4.20–6.20 s: formula once.
        self.play(FadeIn(formula), run_time=1.20, rate_func=smooth)
        self.wait(0.80)

        # 6.20–8.20 s: f box. Dense layer reads the current stack, not a skip-add.
        self.play(
            Create(f_box),
            FadeIn(f_name),
            Create(link),
            run_time=1.20,
            rate_func=smooth,
        )
        self.wait(0.80)

        # 8.20–10.20 s: f(x) is a new map under f, same 2×2, one new channel.
        self.play(FadeIn(map_f1), run_time=1.20, rate_func=smooth)
        self.wait(0.80)

        # 10.20–13.00 s: concat — f₁ docks onto x. C grows. x₀ cells do not change.
        unchanged = map_x0[1][0]
        self.play(
            map_f1.animate.shift(self.map_origin(2) - self.spawn_origin()),
            Transform(halo_outer, halo_outer_3),
            Transform(halo_inner, halo_inner_3),
            Transform(link, link_3),
            c_number.animate.set_value(3),
            run_time=1.80,
            rate_func=smooth,
        )
        self.play(
            Flash(unchanged.get_center(), color=YELLOW, flash_radius=0.32, line_length=0.09),
            run_time=0.50,
            rate_func=smooth,
        )
        self.wait(0.50)

        # 13.00–15.80 s: rest on the three-map stack (t=15 is this beat).
        self.wait(2.80)

        # 15.80–17.80 s: second layer reads [x, f₁], writes one more map.
        self.play(FadeIn(map_f2), run_time=1.20, rate_func=smooth)
        self.wait(0.80)

        # 17.80–20.60 s: second concat. Stack is now four maps; last frame keeps them.
        self.play(
            map_f2.animate.shift(self.map_origin(3) - self.spawn_origin()),
            Transform(halo_outer, halo_outer_4),
            Transform(halo_inner, halo_inner_4),
            Transform(link, link_4),
            c_number.animate.set_value(4),
            run_time=1.80,
            rate_func=smooth,
        )
        self.wait(1.00)

        # 20.60–32.00 s: hold the grown stack, formula, and f.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=11.40, rate_func=linear)
