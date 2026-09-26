"""D2L 10.5 — two attention heads, then concat and a linear map back.

Every weight, head scalar, and ŷ comes from the numpy multi-head pass
below. Projections are fixed and untrained. The haloed traveler is the
query. Head 1 and head 2 look at the same three tokens with different α.
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
    DecimalNumber,
    Dot,
    FadeIn,
    FadeOut,
    Flash,
    LaggedStart,
    Line,
    Scene,
    Square,
    SurroundingRectangle,
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


# ---------------------------------------------------------------------------
# One numerical source of truth. Tiny 2-D tokens, two 1-D heads, then
#   ŷ = W_o [h₁; h₂]
# with scaled-dot-product attention after the projections. Untrained.
# D2L 10.5: h_i = f(W_i^q q, W_i^k k, W_i^v v), f = softmax(q k / √p) v.
# ---------------------------------------------------------------------------
TOKENS = np.array(
    [
        [0.20, 1.20],
        [1.00, 0.50],
        [-1.00, -0.90],
    ],
    dtype=float,
)
QUERY_INDEX = 1
# Each W is p × d with p_q = p_k = p_v = 1, d = 2.
WEIGHT_Q = (
    np.array([[0.80, 0.40]], dtype=float),
    np.array([[0.70, 0.40]], dtype=float),
)
WEIGHT_K = (
    np.array([[0.40, 2.40]], dtype=float),
    np.array([[-1.40, -1.20]], dtype=float),
)
WEIGHT_V = (
    np.array([[0.30, 1.10]], dtype=float),
    np.array([[-1.10, -0.40]], dtype=float),
)
WEIGHT_O = np.array([[0.90, -0.35], [-0.15, -0.30]], dtype=float)
HEAD_DIM = 1
SCALE = 1.0 / np.sqrt(HEAD_DIM)


def softmax(values: np.ndarray) -> np.ndarray:
    """The same softmax drawn as cell opacity and ray weight."""
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def one_head(
    weight_q: np.ndarray, weight_k: np.ndarray, weight_v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """One real head: project q,k,v, then scaled-dot-product pooling."""
    query = TOKENS[QUERY_INDEX].reshape(2, 1)
    projected_query = float((weight_q @ query).item())
    projected_keys = (weight_k @ TOKENS.T).ravel()
    projected_values = (weight_v @ TOKENS.T).ravel()
    scores = projected_query * projected_keys * SCALE
    weights = softmax(scores)
    head_value = float(np.dot(weights, projected_values))
    return projected_keys, projected_values, scores, weights, head_value


HEAD_KEYS = []
HEAD_VALUES = []
HEAD_SCORES = []
HEAD_WEIGHTS = []
HEAD_OUTPUTS = []
for weight_q, weight_k, weight_v in zip(WEIGHT_Q, WEIGHT_K, WEIGHT_V):
    keys, values, scores, weights, head_value = one_head(weight_q, weight_k, weight_v)
    HEAD_KEYS.append(keys)
    HEAD_VALUES.append(values)
    HEAD_SCORES.append(scores)
    HEAD_WEIGHTS.append(weights)
    HEAD_OUTPUTS.append(head_value)

CONCAT = np.array(HEAD_OUTPUTS, dtype=float).reshape(2, 1)
YHAT = (WEIGHT_O @ CONCAT).ravel()
HEAD_COLORS = (YELLOW, BLUE)
TOKEN_COLORS = (BLUE, YELLOW, BLUE_D)
TOKEN_NAMES = ("1", "2", "3")


class Episode105(Scene):
    """One continuous silent D2L 10.5 visualization, about 32 seconds."""

    x_min, x_max = -1.70, 1.70
    y_min, y_max = -1.45, 1.55
    plot_left, plot_right = -5.35, -0.15
    plot_bottom, plot_top = -3.15, 2.58
    cell_size = 0.70
    cell_step = 0.80
    heat_left = 1.88
    heat_top = 1.38

    def plot_point(self, x_value: float, y_value: float) -> np.ndarray:
        scene_x = self.plot_left + (x_value - self.x_min) / (self.x_max - self.x_min) * (
            self.plot_right - self.plot_left
        )
        scene_y = self.plot_bottom + (y_value - self.y_min) / (self.y_max - self.y_min) * (
            self.plot_top - self.plot_bottom
        )
        return np.array([scene_x, scene_y, 0.0])

    def cell_center(self, row: int, column: int) -> np.ndarray:
        return np.array(
            [
                self.heat_left + column * self.cell_step,
                self.heat_top - row * self.cell_step,
                0.0,
            ]
        )

    def make_grid_and_axes(self) -> tuple[VGroup, VGroup]:
        """Stroke-only Line grid; never a filled VMobject mesh."""
        grid = VGroup()
        for x_value in np.arange(-1.5, 1.51, 0.5):
            if abs(x_value) > 1e-8:
                grid.add(
                    Line(self.plot_point(x_value, self.y_min), self.plot_point(x_value, self.y_max)).set_stroke(
                        BLUE_E, width=1.0, opacity=0.30
                    )
                )
        for y_value in np.arange(-1.0, 1.51, 0.5):
            if abs(y_value) > 1e-8:
                grid.add(
                    Line(self.plot_point(self.x_min, y_value), self.plot_point(self.x_max, y_value)).set_stroke(
                        BLUE_E, width=1.0, opacity=0.30
                    )
                )
        x_axis = Line(self.plot_point(self.x_min, 0.0), self.plot_point(self.x_max, 0.0)).set_stroke(
            BLUE_D, width=2.0, opacity=0.88
        )
        y_axis = Line(self.plot_point(0.0, self.y_min), self.plot_point(0.0, self.y_max)).set_stroke(
            BLUE_D, width=2.0, opacity=0.88
        )
        x_symbol = Text("x₁", font_size=26, color=BLUE_A).next_to(
            x_axis.get_end(), np.array([-1.0, -1.0, 0.0]), buff=0.12
        )
        y_symbol = Text("x₂", font_size=26, color=BLUE_A).next_to(
            y_axis.get_end(), np.array([-1.0, 0.15, 0.0]), buff=0.10
        )
        return grid, VGroup(x_axis, y_axis, x_symbol, y_symbol)

    def construct(self) -> None:
        print("D2L 10.5 X={}".format(np.array2string(TOKENS, precision=3, separator=", ")))
        for head_index in range(2):
            print(
                "D2L 10.5 head{} Wq={} Wk={} Wv={}".format(
                    head_index + 1,
                    np.array2string(WEIGHT_Q[head_index], precision=3, separator=", "),
                    np.array2string(WEIGHT_K[head_index], precision=3, separator=", "),
                    np.array2string(WEIGHT_V[head_index], precision=3, separator=", "),
                )
            )
            print(
                "D2L 10.5 head{} scores={} alpha={} h={:.6f} sum={:.6f}".format(
                    head_index + 1,
                    np.array2string(HEAD_SCORES[head_index], precision=4, separator=", "),
                    np.array2string(HEAD_WEIGHTS[head_index], precision=6, separator=", "),
                    HEAD_OUTPUTS[head_index],
                    float(np.sum(HEAD_WEIGHTS[head_index])),
                )
            )
        print(
            "D2L 10.5 concat={} Wo={} yhat={}".format(
                np.array2string(CONCAT.ravel(), precision=6, separator=", "),
                np.array2string(WEIGHT_O, precision=3, separator=", "),
                np.array2string(YHAT, precision=6, separator=", "),
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "10.5")

        grid, axes = self.make_grid_and_axes()
        token_dots = [
            Dot(self.plot_point(token[0], token[1]), radius=0.070, color=color)
            .set_fill(color, opacity=0.94)
            .set_stroke(color, width=0.0, opacity=0.0)
            for token, color in zip(TOKENS, TOKEN_COLORS)
        ]
        point_cloud = VGroup(*token_dots)
        tag_offsets = (
            np.array([0.40, 0.30, 0.0]),
            np.array([0.56, 0.36, 0.0]),
            np.array([-0.42, -0.34, 0.0]),
        )
        token_tags = VGroup(
            *[
                Text(name, font_size=22, color=WHITE).move_to(self.plot_point(token[0], token[1]) + offset)
                for name, token, offset in zip(TOKEN_NAMES, TOKENS, tag_offsets)
            ]
        )

        traveler_center = self.plot_point(TOKENS[QUERY_INDEX, 0], TOKENS[QUERY_INDEX, 1])
        halo_outer = Circle(radius=0.278).move_to(traveler_center).set_stroke(BLUE_A, width=2.1, opacity=0.50)
        halo_inner = Circle(radius=0.178).move_to(traveler_center).set_stroke(YELLOW, width=2.8, opacity=0.98)
        traveler_x1_label = Text(
            "x₁ = {:.2f}".format(TOKENS[QUERY_INDEX, 0]), font_size=21, color=BLUE_A
        ).move_to(np.array([-6.32, -1.72, 0.0]))
        traveler_x2_label = Text(
            "x₂ = {:.2f}".format(TOKENS[QUERY_INDEX, 1]), font_size=21, color=BLUE_A
        ).move_to(np.array([-6.32, -2.14, 0.0]))

        formula = Text("ŷ = Wₒ [h₁; h₂]", font_size=32, color=WHITE).move_to(
            np.array([3.55, 3.22, 0.0])
        )

        ray_reveal = [ValueTracker(0.0), ValueTracker(0.0)]
        yhat_reveal = ValueTracker(0.0)

        def ray_end(index: int) -> np.ndarray:
            destination = self.plot_point(TOKENS[index, 0], TOKENS[index, 1])
            direction = destination - traveler_center
            length = np.linalg.norm(direction)
            if length < 1e-6:
                return traveler_center
            return traveler_center + direction * ((length - 0.16) / length)

        def offset_pair(start: np.ndarray, end: np.ndarray, amount: float) -> tuple[np.ndarray, np.ndarray]:
            direction = end - start
            length = np.linalg.norm(direction)
            if length < 1e-6:
                return start, end
            normal = np.array([-direction[1], direction[0], 0.0]) / length
            shift = amount * normal
            return start + shift, end + shift

        def make_ray(head_index: int, token_index: int):
            def draw() -> Line:
                alpha = float(HEAD_WEIGHTS[head_index][token_index]) * ray_reveal[head_index].get_value()
                start, end = traveler_center, ray_end(token_index)
                if head_index == 1:
                    start, end = offset_pair(start, end, 0.10)
                opacity = 0.0 if alpha < 0.010 else min(0.20 + 1.8 * alpha, 0.96)
                width = 2.0 + 8.0 * alpha
                return Line(start, end).set_stroke(HEAD_COLORS[head_index], width=width, opacity=opacity)

            return always_redraw(draw)

        head_rays = [
            VGroup(
                *[make_ray(head_index, token_index) for token_index in range(3) if token_index != QUERY_INDEX]
            )
            for head_index in range(2)
        ]
        self_rings = [
            always_redraw(
                lambda head_index=head_index: Circle(radius=0.318 + 0.055 * head_index)
                .move_to(traveler_center)
                .set_stroke(
                    HEAD_COLORS[head_index],
                    width=2.6,
                    opacity=min(
                        0.16 + 1.7 * float(HEAD_WEIGHTS[head_index][QUERY_INDEX]) * ray_reveal[head_index].get_value(),
                        0.88,
                    ),
                )
                .set_fill(BLACK, opacity=0.0)
            )
            for head_index in range(2)
        ]

        yhat_center = self.plot_point(float(YHAT[0]), float(YHAT[1]))
        yhat_dot = always_redraw(
            lambda: Dot(yhat_center, radius=0.078, color=YELLOW)
            .set_fill(YELLOW, opacity=0.98 * yhat_reveal.get_value())
            .set_stroke(YELLOW, width=0.0, opacity=0.0)
        )
        yhat_label = Text("ŷ", font_size=24, color=YELLOW).move_to(
            yhat_center + np.array([0.46, -0.32, 0.0])
        )

        # Two rows of three stroke cells: fill opacity is α. Not a number sheet.
        heat_cells = []
        for row in range(2):
            for column in range(3):
                cell = Square(side_length=self.cell_size).move_to(self.cell_center(row, column))
                cell.set_stroke(BLUE_E, width=1.6, opacity=0.90)
                cell.set_fill(HEAD_COLORS[row], opacity=0.0)
                heat_cells.append(cell)
        heat_group = VGroup(*heat_cells)
        row_tags = VGroup(
            *[
                Text(f"h{['₁', '₂'][row]}", font_size=22, color=HEAD_COLORS[row]).move_to(
                    self.cell_center(row, 0) + np.array([-0.78, 0.0, 0.0])
                )
                for row in range(2)
            ]
        )
        col_tags = VGroup(
            *[
                Text(f"k{['₁', '₂', '₃'][column]}", font_size=20, color=WHITE).move_to(
                    self.cell_center(1, column) + np.array([0.0, -0.54, 0.0])
                )
                for column in range(3)
            ]
        )

        winner_index = [int(np.argmax(HEAD_WEIGHTS[0])), int(np.argmax(HEAD_WEIGHTS[1]))]
        winner_boxes = []
        winner_alphas = VGroup()
        winner_offsets = (np.array([0.0, 0.56, 0.0]), np.array([0.0, -1.02, 0.0]))
        for row, offset in zip(range(2), winner_offsets):
            cell = heat_cells[row * 3 + winner_index[row]]
            box = SurroundingRectangle(cell, color=HEAD_COLORS[row], buff=0.06, corner_radius=0.05)
            box.set_fill(opacity=0.0)
            box.set_stroke(width=2.6, opacity=1.0)
            winner_boxes.append(box)
            number = DecimalNumber(
                float(HEAD_WEIGHTS[row][winner_index[row]]),
                num_decimal_places=3,
                mob_class=Text,
                include_sign=False,
                color=HEAD_COLORS[row],
                font_size=20,
            )
            number.move_to(self.cell_center(row, winner_index[row]) + offset)
            winner_alphas.add(number)

        concat_h1 = Text("h₁ = {:+.2f}".format(HEAD_OUTPUTS[0]), font_size=22, color=YELLOW)
        concat_h2 = Text("h₂ = {:+.2f}".format(HEAD_OUTPUTS[1]), font_size=22, color=BLUE)
        concat_stack = VGroup(concat_h1, concat_h2).arrange(np.array([0.0, -1.0, 0.0]), buff=0.16)
        concat_stack.move_to(np.array([5.48, 1.00, 0.0]))
        concat_box = SurroundingRectangle(concat_stack, color=WHITE, buff=0.16, corner_radius=0.08)
        concat_box.set_fill(opacity=0.0)
        concat_box.set_stroke(width=2.0, opacity=0.90)

        # 0.40–2.80 s: three tokens before any head.
        self.play(
            FadeIn(grid),
            FadeIn(axes),
            LaggedStart(*[FadeIn(point) for point in point_cloud], lag_ratio=0.10),
            FadeIn(token_tags),
            run_time=0.90,
            rate_func=linear,
        )
        self.wait(1.50)

        # 2.80–5.00 s: halo the query; pocket holds its two coordinates.
        self.play(
            Create(halo_outer),
            Create(halo_inner),
            FadeIn(traveler_x1_label),
            FadeIn(traveler_x2_label),
            Flash(traveler_center, color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=0.70,
            rate_func=smooth,
        )
        self.wait(1.50)

        # 5.00–7.00 s: formula once.
        self.play(FadeIn(formula), run_time=0.80, rate_func=smooth)
        self.wait(1.20)

        # 7.00–9.00 s: empty two-row slots so later fills have a place to sit.
        self.play(
            FadeIn(heat_group),
            FadeIn(row_tags),
            FadeIn(col_tags),
            run_time=0.90,
            rate_func=smooth,
        )
        self.wait(1.10)

        # 9.00–13.00 s: head 1. Same tokens; yellow weights pile on token 1.
        self.add(head_rays[0], self_rings[0])
        head1_fills = [
            heat_cells[column].animate.set_fill(
                YELLOW, opacity=min(0.16 + 1.25 * float(HEAD_WEIGHTS[0][column]), 0.92)
            )
            for column in range(3)
        ]
        self.play(
            ray_reveal[0].animate.set_value(1.0),
            *head1_fills,
            run_time=2.40,
            rate_func=smooth,
        )
        self.play(
            Create(winner_boxes[0]),
            FadeIn(winner_alphas[0]),
            run_time=0.55,
            rate_func=smooth,
        )
        self.wait(1.05)

        # 13.00–17.00 s: head 2. Blue weights pile on token 3. Head 1 stays.
        self.add(head_rays[1], self_rings[1])
        head2_fills = [
            heat_cells[3 + column].animate.set_fill(
                BLUE, opacity=min(0.16 + 1.25 * float(HEAD_WEIGHTS[1][column]), 0.92)
            )
            for column in range(3)
        ]
        self.play(
            ray_reveal[1].animate.set_value(1.0),
            *head2_fills,
            run_time=2.40,
            rate_func=smooth,
        )
        self.play(
            Create(winner_boxes[1]),
            FadeIn(winner_alphas[1]),
            run_time=0.55,
            rate_func=smooth,
        )
        self.wait(1.05)

        # 17.00–20.50 s: concat the two head scalars.
        self.play(FadeIn(concat_stack), Create(concat_box), run_time=1.20, rate_func=smooth)
        self.wait(2.30)

        # 20.50–24.00 s: W_o maps the stacked heads back into the token plane.
        self.add(yhat_dot)
        self.play(yhat_reveal.animate.set_value(1.0), FadeIn(yhat_label), run_time=1.20, rate_func=smooth)
        self.wait(2.30)

        # Keep both heads, the concat, and ŷ through the last encoded frame.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=8.00, rate_func=linear)
