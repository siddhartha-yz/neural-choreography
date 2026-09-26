"""D2L 3.1 — a line fitting noisy points through real mini-batch SGD.

Every moving value comes from the column-vector computation below.  No
closed-form fit is used; ``training_states`` is produced only by mini-batch
gradient updates on mean squared error.
"""

from __future__ import annotations

import numpy as np
from scripts.episode_intro import show_episode_intro
from manim import (
    AnimationGroup,
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
# One numerical source of truth: scalar inputs are represented as 1 × 1
# column vectors, including w and b.  The point cloud, line, residual,
# readouts, and updates all use these arrays.
# ---------------------------------------------------------------------------
FEATURES = np.array(
    [
        -2.70,
        -2.30,
        -1.90,
        -1.50,
        -1.10,
        -0.70,
        -0.30,
        0.10,
        0.50,
        0.90,
        1.25,
        1.65,
        2.05,
        2.45,
    ],
    dtype=float,
).reshape(-1, 1)

TRUE_W = np.array([[1.15]], dtype=float)
TRUE_B = np.array([[0.55]], dtype=float)
NOISE = np.array(
    [-0.20, 0.12, -0.08, 0.10, -0.12, 0.05, -0.17, 0.16, -0.06, 0.08, -0.15, 0.13, -0.09, 0.19],
    dtype=float,
).reshape(-1, 1)
TARGETS = FEATURES @ TRUE_W + TRUE_B + NOISE

HIGHLIGHT_INDEX = 3
MINIBATCH_SIZE = 4
LEARNING_RATE = 0.05
INITIAL_W = np.array([[-0.45]], dtype=float)
INITIAL_B = np.array([[1.75]], dtype=float)
SGD_STEPS = 9


def mean_squared_loss(weight: np.ndarray, bias: np.ndarray) -> float:
    """MSE of the same noisy point cloud drawn in the scene."""
    residuals = FEATURES @ weight + bias - TARGETS
    return float(np.mean(residuals**2))


def make_minibatches() -> list[np.ndarray]:
    """Fixed mini-batches, with the traveler present in each visible update."""
    generator = np.random.default_rng(31)
    other_indices = np.delete(np.arange(len(FEATURES)), HIGHLIGHT_INDEX)
    batches: list[np.ndarray] = []
    for _ in range(SGD_STEPS - 1):
        batches.append(
            np.concatenate(
                (
                    [HIGHLIGHT_INDEX],
                    generator.choice(other_indices, size=MINIBATCH_SIZE - 1, replace=False),
                )
            )
        )
    # The final batch is another real batch from the point cloud. It leaves
    # a small visible noise residual rather than fitting the haloed sample exactly.
    batches.append(np.array([HIGHLIGHT_INDEX, 0, 4, 7]))
    return batches


MINIBATCHES = make_minibatches()


def mini_batch_sgd() -> list[tuple[np.ndarray, np.ndarray]]:
    """Return model states created by mini-batch SGD, from start to finish."""
    weight = INITIAL_W.copy()
    bias = INITIAL_B.copy()
    states = [(weight.copy(), bias.copy())]
    for batch_indices in MINIBATCHES:
        batch_x = FEATURES[batch_indices]
        batch_y = TARGETS[batch_indices]
        batch_residuals = batch_x @ weight + bias - batch_y
        gradient_w = (2.0 / MINIBATCH_SIZE) * (batch_x.T @ batch_residuals)
        gradient_b = (2.0 / MINIBATCH_SIZE) * np.sum(batch_residuals, axis=0, keepdims=True)
        weight = weight - LEARNING_RATE * gradient_w
        bias = bias - LEARNING_RATE * gradient_b
        states.append((weight.copy(), bias.copy()))
    return states


TRAINING_STATES = mini_batch_sgd()


class Episode031(Scene):
    """One continuous 25-second silent D2L 3.1 visualization."""

    x_min, x_max = -3.0, 3.0
    y_min, y_max = -3.2, 4.0
    # Leave a dedicated label pocket to the left of the point cloud.
    plot_left, plot_right = -5.35, 2.05
    plot_bottom, plot_top = -3.35, 2.75

    def plot_point(self, x_value: float, y_value: float) -> np.ndarray:
        """Map data coordinates to the left plot without relying on filled axes."""
        scene_x = self.plot_left + (x_value - self.x_min) / (self.x_max - self.x_min) * (
            self.plot_right - self.plot_left
        )
        scene_y = self.plot_bottom + (y_value - self.y_min) / (self.y_max - self.y_min) * (
            self.plot_top - self.plot_bottom
        )
        return np.array([scene_x, scene_y, 0.0])

    def clipped_line_points(self, weight: float, bias: float) -> tuple[np.ndarray, np.ndarray]:
        """Find the two intersections of y = wx+b with the visible plot rectangle."""
        candidates: list[tuple[float, float]] = []
        for x_value in (self.x_min, self.x_max):
            y_value = weight * x_value + bias
            if self.y_min <= y_value <= self.y_max:
                candidates.append((x_value, y_value))
        if abs(weight) > 1e-8:
            for y_value in (self.y_min, self.y_max):
                x_value = (y_value - bias) / weight
                if self.x_min <= x_value <= self.x_max:
                    candidates.append((x_value, y_value))
        candidates.sort()
        first, last = candidates[0], candidates[-1]
        return self.plot_point(*first), self.plot_point(*last)

    def make_grid_and_axes(self) -> tuple[VGroup, VGroup]:
        """Create only stroke Line objects: never a filled VMobject mesh."""
        grid_lines = VGroup()
        for x_value in np.arange(-3.0, 3.01, 0.5):
            if abs(x_value) > 1e-8:
                grid_lines.add(
                    Line(self.plot_point(x_value, self.y_min), self.plot_point(x_value, self.y_max)).set_stroke(
                        BLUE_E, width=1.0, opacity=0.30
                    )
                )
        for y_value in np.arange(-3.0, 4.01, 0.5):
            if abs(y_value) > 1e-8:
                grid_lines.add(
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
        x_symbol = Text("x", font_size=27, color=BLUE_A).next_to(x_axis.get_end(), np.array([-1.0, -1.0, 0.0]), buff=0.12)
        y_symbol = Text("y", font_size=27, color=BLUE_A).next_to(y_axis.get_end(), np.array([1.0, -1.0, 0.0]), buff=0.12)
        return grid_lines, VGroup(x_axis, y_axis, x_symbol, y_symbol)

    def construct(self) -> None:
        initial_weight, initial_bias = TRAINING_STATES[0]
        final_weight, final_bias = TRAINING_STATES[-1]
        highlighted_x = FEATURES[HIGHLIGHT_INDEX]
        highlighted_y = TARGETS[HIGHLIGHT_INDEX]
        final_prediction = highlighted_x @ final_weight + final_bias
        final_residual = final_prediction - highlighted_y
        print(
            "D2L 3.1 final w={:.6f}, final b={:.6f}, x={:.6f}, y={:.6f}, "
            "final yhat={:.6f}, final residual={:+.6f}".format(
                final_weight.item(),
                final_bias.item(),
                highlighted_x.item(),
                highlighted_y.item(),
                final_prediction.item(),
                final_residual.item(),
            )
        )

        # 0.00–0.40 s: a short card, with no overlap into the scatter reveal.
        show_episode_intro(self, "03.1")

        grid, axes = self.make_grid_and_axes()
        palette = (YELLOW, BLUE, YELLOW, BLUE, YELLOW, BLUE, YELLOW, BLUE, YELLOW, BLUE, YELLOW, BLUE, YELLOW, BLUE)
        data_dots = [
            Dot(self.plot_point(feature.item(), target.item()), radius=0.068, color=color)
            .set_fill(color, opacity=0.94)
            .set_stroke(color, width=0.0, opacity=0.0)
            for feature, target, color in zip(FEATURES, TARGETS, palette)
        ]
        point_cloud = VGroup(*data_dots)

        weight_tracker = ValueTracker(initial_weight.item())
        bias_tracker = ValueTracker(initial_bias.item())

        def current_weight() -> np.ndarray:
            return np.array([[weight_tracker.get_value()]], dtype=float)

        def current_bias() -> np.ndarray:
            return np.array([[bias_tracker.get_value()]], dtype=float)

        def current_prediction() -> float:
            return float((highlighted_x @ current_weight() + current_bias()).item())

        prediction_line = always_redraw(
            lambda: Line(*self.clipped_line_points(weight_tracker.get_value(), bias_tracker.get_value())).set_stroke(
                YELLOW, width=5.0, opacity=0.96
            )
        )
        prediction_dot = always_redraw(
            lambda: Dot(
                self.plot_point(highlighted_x.item(), current_prediction()),
                radius=0.078,
                color=YELLOW,
            ).set_fill(YELLOW, opacity=1.0)
        )
        residual_segment = always_redraw(
            lambda: Line(
                self.plot_point(highlighted_x.item(), highlighted_y.item()),
                self.plot_point(highlighted_x.item(), current_prediction()),
            ).set_stroke(YELLOW_A, width=3.6, opacity=0.98)
        )

        halo_center = self.plot_point(highlighted_x.item(), highlighted_y.item())
        halo_outer = Circle(radius=0.265).move_to(halo_center).set_stroke(BLUE_A, width=2.1, opacity=0.48)
        halo_inner = Circle(radius=0.172).move_to(halo_center).set_stroke(YELLOW, width=2.8, opacity=0.98)
        # Fixed left-pocket labels cannot collide with the traveler, residual,
        # yellow line, or a neighboring data point. Use a true minus glyph so
        # the negative feature value remains unmistakable.
        sample_x_label = Text(f"x = −{abs(highlighted_x.item()):.2f}", font_size=24, color=BLUE_A).move_to(
            np.array([-6.18, -1.18, 0.0])
        )
        sample_y_label = Text(f"y = −{abs(highlighted_y.item()):.3f}", font_size=24, color=BLUE_A).move_to(
            np.array([-6.18, -1.63, 0.0])
        )

        formula = Text("ŷ = wx + b", font_size=35, color=WHITE).move_to(np.array([4.65, 3.08, 0.0]))

        def make_number_row(label: str, y_position: float, color, signed: bool) -> tuple[VGroup, DecimalNumber]:
            prefix = Text(label, font_size=21, color=WHITE)
            number = DecimalNumber(
                0.0,
                num_decimal_places=3,
                mob_class=Text,
                include_sign=signed,
                color=color,
                font_size=22,
            )
            row = VGroup(prefix, number).arrange(np.array([1.0, 0.0, 0.0]), buff=0.10).move_to(
                np.array([4.78, y_position, 0.0])
            )
            return row, number

        weight_row, weight_number = make_number_row("w =", 2.43, BLUE_A, True)
        bias_row, bias_number = make_number_row("b =", 2.04, YELLOW_A, True)
        loss_row, loss_number = make_number_row("L =", 1.65, WHITE, False)
        weight_number.set_value(weight_tracker.get_value())
        bias_number.set_value(bias_tracker.get_value())
        loss_number.set_value(mean_squared_loss(current_weight(), current_bias()))
        weight_number.add_updater(lambda number: number.set_value(weight_tracker.get_value()))
        bias_number.add_updater(lambda number: number.set_value(bias_tracker.get_value()))
        loss_number.add_updater(lambda number: number.set_value(mean_squared_loss(current_weight(), current_bias())))
        secondary_readout = VGroup(weight_row, bias_row, loss_row)

        prediction_prefix = Text("ŷ = ", font_size=24, color=YELLOW)
        prediction_number = DecimalNumber(
            current_prediction(),
            num_decimal_places=3,
            mob_class=Text,
            include_sign=True,
            color=YELLOW_A,
            font_size=24,
        )
        prediction_label = VGroup(prediction_prefix, prediction_number).arrange(
            np.array([1.0, 0.0, 0.0]), buff=0.08
        )
        prediction_number.add_updater(lambda number: number.set_value(current_prediction()))

        def prediction_label_position() -> np.ndarray:
            """Place ŷ along the line's normal, never over the yellow stroke."""
            horizontal_scale = (self.plot_right - self.plot_left) / (self.x_max - self.x_min)
            vertical_scale = (self.plot_top - self.plot_bottom) / (self.y_max - self.y_min)
            scene_slope = weight_tracker.get_value() * vertical_scale / horizontal_scale
            normal = np.array([-scene_slope, 1.0, 0.0])
            normal /= np.linalg.norm(normal)
            return self.plot_point(highlighted_x.item(), current_prediction()) + normal * 0.78

        prediction_label.add_updater(lambda label: label.move_to(prediction_label_position()))
        residual_label = Text("r", font_size=24, color=YELLOW_A)

        def residual_label_position() -> np.ndarray:
            prediction_position = self.plot_point(highlighted_x.item(), current_prediction())
            target_position = self.plot_point(highlighted_x.item(), highlighted_y.item())
            if np.linalg.norm(prediction_position - target_position) >= 0.58:
                return (prediction_position + target_position) / 2 + np.array([0.27, 0.0, 0.0])
            # Near convergence, protect the tiny residual from label clutter:
            # ŷ sits on the upper line normal while r stays lower-right.
            return prediction_position + np.array([0.45, -0.38, 0.0])

        residual_label.add_updater(lambda label: label.move_to(residual_label_position()))

        # 0.40–3.45 s: plot and label the traveler before any incorrect line appears.
        self.play(
            FadeIn(grid),
            FadeIn(axes),
            LaggedStart(*[FadeIn(point) for point in point_cloud], lag_ratio=0.025),
            run_time=0.75,
            rate_func=linear,
        )
        self.play(
            Create(halo_outer),
            Create(halo_inner),
            FadeIn(sample_x_label),
            FadeIn(sample_y_label),
            Flash(halo_center, color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=0.55,
            rate_func=smooth,
        )
        self.wait(1.75)

        # 3.45–4.75 s: the bad line and its long, visually unambiguous residual.
        self.play(FadeIn(prediction_line), FadeIn(formula), run_time=0.75, rate_func=smooth)
        self.play(
            FadeIn(residual_segment),
            FadeIn(prediction_dot),
            FadeIn(prediction_label),
            FadeIn(residual_label),
            FadeIn(secondary_readout),
            run_time=0.55,
            rate_func=smooth,
        )

        # 4.75–21.85 s: each of nine true mini-batch updates is deliberately slow.
        # The active four points brighten in their own source colors; their Dot
        # geometry is never scaled, so the full cloud keeps its original size.
        for update_index, (next_weight, next_bias) in enumerate(TRAINING_STATES[1:]):
            active_indices = MINIBATCHES[update_index]
            brightening = AnimationGroup(
                *[
                    data_dots[index]
                    .animate.set_fill(palette[index], opacity=1.0)
                    .set_stroke(palette[index], width=3.2, opacity=1.0)
                    for index in active_indices
                ],
                lag_ratio=0.06,
            )
            dimming = AnimationGroup(
                *[
                    data_dots[index]
                    .animate.set_fill(palette[index], opacity=0.94)
                    .set_stroke(palette[index], width=0.0, opacity=0.0)
                    for index in active_indices
                ],
                lag_ratio=0.06,
            )
            self.play(brightening, run_time=0.28, rate_func=smooth)
            self.play(
                weight_tracker.animate.set_value(next_weight.item()),
                bias_tracker.animate.set_value(next_bias.item()),
                run_time=1.38,
                rate_func=smooth,
            )
            self.play(dimming, run_time=0.24, rate_func=smooth)

        # 21.85–25.10 s: a timed no-op keeps every live updater rendered through
        # the final encoded frame (a terminal ``wait`` can freeze them unevenly).
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=3.25, rate_func=linear)
