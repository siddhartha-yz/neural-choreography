"""D2L 4.7 — forward then reverse on the same tiny-net path.

The arrays and ``run_graph`` below are the only numerical source of truth.
Forward activations and reverse local gradients are the same numpy chain
rule; a finite-difference check on ∂ℓ/∂x is printed at scene start.
No parameters are trained.
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
# One numerical source of truth: 2-D x → 2 hidden (ReLU) → scalar ŷ,
# loss ℓ = (ŷ − y)² on this one sample. Reverse uses the same graph.
# ---------------------------------------------------------------------------
INPUT_X = np.array([[1.00], [0.50]], dtype=float)
TARGET_Y = 0.69
WEIGHTS_1 = np.array(
    [
        [0.80, 0.40],
        [-0.60, 0.30],
    ],
    dtype=float,
)
BIAS_1 = np.array([[0.10], [-0.40]], dtype=float)
WEIGHTS_2 = np.array([[0.90, -0.50]], dtype=float)
BIAS_2 = np.array([[0.20]], dtype=float)


def run_graph(x: np.ndarray, y: float) -> dict[str, np.ndarray | float]:
    """Forward activations, then reverse local gradients, one numpy path."""
    hidden_pre = WEIGHTS_1 @ x + BIAS_1
    relu_mask = (hidden_pre > 0.0).astype(float)
    hidden = hidden_pre * relu_mask
    prediction = float((WEIGHTS_2 @ hidden + BIAS_2).item())
    residual = prediction - y
    loss = residual**2
    # Reverse on the same path: ∂ℓ/∂ŷ, then ∂ℓ/∂h = (∂ℓ/∂ŷ)(∂ŷ/∂h), then ∂ℓ/∂x.
    d_loss_d_prediction = 2.0 * residual
    d_loss_d_hidden = WEIGHTS_2.T * d_loss_d_prediction
    d_loss_d_pre = d_loss_d_hidden * relu_mask
    d_loss_d_x = WEIGHTS_1.T @ d_loss_d_pre
    return {
        "z": hidden_pre,
        "h": hidden,
        "yhat": prediction,
        "y": y,
        "residual": residual,
        "loss": loss,
        "dL_dyhat": d_loss_d_prediction,
        "dL_dh": d_loss_d_hidden,
        "dL_dz": d_loss_d_pre,
        "dL_dx": d_loss_d_x,
    }


GRAPH = run_graph(INPUT_X, TARGET_Y)


def finite_difference_dL_dx(epsilon: float = 1e-6) -> np.ndarray:
    """Central differences on ℓ(x); must match GRAPH['dL_dx']."""
    gradient = np.zeros_like(INPUT_X)
    for index in range(INPUT_X.shape[0]):
        shifted_plus = INPUT_X.copy()
        shifted_minus = INPUT_X.copy()
        shifted_plus[index, 0] += epsilon
        shifted_minus[index, 0] -= epsilon
        loss_plus = float(run_graph(shifted_plus, TARGET_Y)["loss"])
        loss_minus = float(run_graph(shifted_minus, TARGET_Y)["loss"])
        gradient[index, 0] = (loss_plus - loss_minus) / (2.0 * epsilon)
    return gradient


def signed_text(value: float, digits: int = 3) -> str:
    """ASCII + / Unicode minus, matching the 3.1 pocket glyphs."""
    return f"{value:+.{digits}f}".replace("-", "−")


class Episode047(Scene):
    """A ~32-second continuous, silent backprop visualization."""

    x_min, x_max = -2.8, 2.8
    y_min, y_max = -2.5, 2.5
    # Left-side label pocket stays outside the plotted data rectangle.
    plot_left, plot_right = -4.85, 0.30
    plot_bottom, plot_top = -3.20, 2.70
    node_radius = 0.29
    needle_scale = 0.82

    def plot_point(self, x_value: float, y_value: float) -> np.ndarray:
        scene_x = self.plot_left + (x_value - self.x_min) / (self.x_max - self.x_min) * (
            self.plot_right - self.plot_left
        )
        scene_y = self.plot_bottom + (y_value - self.y_min) / (self.y_max - self.y_min) * (
            self.plot_top - self.plot_bottom
        )
        return np.array([scene_x, scene_y, 0.0])

    def make_grid_and_axes(self) -> tuple[VGroup, VGroup]:
        """Stroke-only Line mobjects; never a filled grid mesh."""
        grid = VGroup()
        for x_value in np.arange(-2.5, 2.51, 0.5):
            if abs(x_value) > 1e-8:
                grid.add(
                    Line(self.plot_point(x_value, self.y_min), self.plot_point(x_value, self.y_max)).set_stroke(
                        BLUE_E, width=1.0, opacity=0.30
                    )
                )
        for y_value in np.arange(-2.0, 2.01, 0.5):
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
            y_axis.get_end(), np.array([1.0, -1.0, 0.0]), buff=0.12
        )
        return grid, VGroup(x_axis, y_axis, x_symbol, y_symbol)

    def node_center(self, column_x: float, row: int) -> np.ndarray:
        """row 0 is the upper hidden unit; row 1 is the lower unit."""
        row_y = 1.42 if row == 0 else -0.68
        return np.array([column_x, row_y, 0.0])

    def make_node_circle(self, center: np.ndarray) -> Circle:
        circle = Circle(radius=self.node_radius).move_to(center)
        circle.set_stroke(BLUE_E, width=1.8, opacity=0.42)
        circle.set_fill(BLACK, opacity=0.0)
        return circle

    def activation_color(self, value: float):
        if value < -1e-8:
            return YELLOW
        if value > 1e-8:
            return BLUE
        return BLUE_D

    def attach_activation_style(self, circle: Circle, reveal: ValueTracker, value: float) -> None:
        color = self.activation_color(value)

        def update(mobject: Circle) -> None:
            t = reveal.get_value()
            idle = BLUE_E
            mobject.set_stroke(
                idle if t < 0.04 else color,
                width=1.8 + 2.2 * t,
                opacity=0.42 + 0.56 * t,
            )
            fill_opacity = 0.0 if abs(value) < 1e-8 else 0.20 * t
            mobject.set_fill(color, opacity=fill_opacity)

        circle.add_updater(update)
        update(circle)

    def make_back_arrow(self, center: np.ndarray, value: float, reveal: ValueTracker) -> VGroup:
        """Left-pointing local-gradient arrow on a path node. Sign lives in the label."""

        def draw() -> VGroup:
            t = reveal.get_value()
            length = max(abs(value) * self.needle_scale * t, 0.001)
            start = center + np.array([-self.node_radius * 0.18, 0.0, 0.0])
            end = start + np.array([-length, 0.0, 0.0])
            shaft = Line(start, end).set_stroke(YELLOW, width=4.2, opacity=0.25 + 0.73 * t)
            direction = np.array([-1.0, 0.0, 0.0])
            head_size = 0.11 if length > 0.16 else max(length * 0.55, 0.04)
            perp = np.array([0.0, 1.0, 0.0])
            left = end - direction * head_size + perp * head_size * 0.48
            right = end - direction * head_size - perp * head_size * 0.48
            head = VGroup(
                Line(end, left).set_stroke(YELLOW, width=4.0, opacity=0.25 + 0.73 * t),
                Line(end, right).set_stroke(YELLOW, width=4.0, opacity=0.25 + 0.73 * t),
            )
            return VGroup(shaft, head)

        return always_redraw(draw)

    def construct(self) -> None:
        numeric_dx = np.asarray(GRAPH["dL_dx"], dtype=float)
        finite_dx = finite_difference_dL_dx()
        print(
            "D2L 4.7 y={:.6f}, yhat={:.6f}, loss={:.6f}, "
            "dL/dyhat={:.6f}, dL/dh={}, dL/dx={}, fd_dL/dx={}, fd_max_abs_err={:.3e}".format(
                float(GRAPH["y"]),
                float(GRAPH["yhat"]),
                float(GRAPH["loss"]),
                float(GRAPH["dL_dyhat"]),
                np.array2string(np.asarray(GRAPH["dL_dh"]).ravel(), precision=6, separator=", "),
                np.array2string(numeric_dx.ravel(), precision=6, separator=", "),
                np.array2string(finite_dx.ravel(), precision=6, separator=", "),
                float(np.max(np.abs(numeric_dx - finite_dx))),
            )
        )

        z_values = np.asarray(GRAPH["z"]).ravel()
        h_values = np.asarray(GRAPH["h"]).ravel()
        d_h = np.asarray(GRAPH["dL_dh"]).ravel()
        d_x = numeric_dx.ravel()
        yhat = float(GRAPH["yhat"])
        residual = float(GRAPH["residual"])
        loss = float(GRAPH["loss"])
        d_yhat = float(GRAPH["dL_dyhat"])

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "04.7")

        grid, axes = self.make_grid_and_axes()
        class_points = (
            (BLUE, [(1.00, 0.50), (0.72, 0.82), (1.35, 0.84), (1.52, 0.24), (0.62, 0.20)]),
            (YELLOW, [(-1.72, 1.04), (-1.38, 0.62), (-1.08, 1.34), (-1.95, 0.42), (-0.88, 0.82)]),
            (BLUE_D, [(-0.30, -1.52), (0.10, -1.78), (-0.70, -1.25), (0.42, -1.22), (-0.10, -2.02)]),
        )
        point_cloud = VGroup(
            *[
                Dot(self.plot_point(x_value, y_value), radius=0.070, color=color)
                .set_fill(color, opacity=0.94)
                .set_stroke(color, width=0.0, opacity=0.0)
                for color, points in class_points
                for x_value, y_value in points
            ]
        )

        traveler_center = self.plot_point(INPUT_X[0, 0], INPUT_X[1, 0])
        halo_outer = Circle(radius=0.278).move_to(traveler_center).set_stroke(BLUE_A, width=2.1, opacity=0.50)
        halo_inner = Circle(radius=0.178).move_to(traveler_center).set_stroke(YELLOW, width=2.8, opacity=0.98)
        traveler_x1_label = Text("x₁ = 1.00", font_size=21, color=BLUE_A).move_to(np.array([-6.02, -1.46, 0.0]))
        traveler_x2_label = Text("x₂ = 0.50", font_size=21, color=BLUE_A).move_to(np.array([-6.02, -1.88, 0.0]))
        traveler_y_label = Text("y = 0.690", font_size=21, color=BLUE_A).move_to(np.array([-6.02, -2.30, 0.0]))

        # Residual / loss stick sits on the traveler, straight up from the halo
        # (3.1 analog). Labels take the empty sky to the right of the stick.
        residual_scale = 1.28
        stick_bottom = traveler_center + np.array([0.0, 0.82, 0.0])
        stick_top = stick_bottom + np.array([0.0, residual * residual_scale, 0.0])
        residual_leader = Line(
            traveler_center + np.array([0.0, 0.278, 0.0]),
            stick_bottom,
        ).set_stroke(YELLOW_A, width=2.4, opacity=0.78)
        residual_segment = Line(stick_bottom, stick_top).set_stroke(YELLOW_A, width=3.6, opacity=0.98)
        prediction_tick = (
            Dot(stick_top, radius=0.070, color=YELLOW).set_fill(YELLOW, opacity=1.0).set_stroke(YELLOW, width=0.0)
        )
        target_tick = (
            Dot(stick_bottom, radius=0.062, color=BLUE_A).set_fill(BLUE_A, opacity=1.0).set_stroke(BLUE_A, width=0.0)
        )
        residual_yhat_label = Text(f"ŷ = {signed_text(yhat)}", font_size=20, color=YELLOW).move_to(
            stick_top + np.array([0.76, 0.12, 0.0])
        )
        residual_loss_label = Text(f"ℓ = {loss:.3f}", font_size=20, color=WHITE).move_to(
            stick_top + np.array([0.76, -0.30, 0.0])
        )
        residual_group = VGroup(
            residual_leader,
            residual_segment,
            prediction_tick,
            target_tick,
            residual_yhat_label,
            residual_loss_label,
        )

        column_x = {"x": 1.22, "z": 3.08, "h": 4.90}
        yhat_center = np.array([6.48, 0.37, 0.0])
        x_centers = [self.node_center(column_x["x"], row) for row in range(2)]
        z_centers = [self.node_center(column_x["z"], row) for row in range(2)]
        h_centers = [self.node_center(column_x["h"], row) for row in range(2)]

        x_nodes = [self.make_node_circle(center) for center in x_centers]
        z_nodes = [self.make_node_circle(center) for center in z_centers]
        h_nodes = [self.make_node_circle(center) for center in h_centers]
        yhat_node = self.make_node_circle(yhat_center)

        x_reveal = [ValueTracker(0.0) for _ in range(2)]
        z_reveal = [ValueTracker(0.0) for _ in range(2)]
        h_reveal = [ValueTracker(0.0) for _ in range(2)]
        yhat_reveal = ValueTracker(0.0)
        for node, tracker, value in zip(x_nodes, x_reveal, INPUT_X.ravel()):
            self.attach_activation_style(node, tracker, float(value))
        for node, tracker, value in zip(z_nodes, z_reveal, z_values):
            self.attach_activation_style(node, tracker, float(value))
        for node, tracker, value in zip(h_nodes, h_reveal, h_values):
            self.attach_activation_style(node, tracker, float(value))
        self.attach_activation_style(yhat_node, yhat_reveal, yhat)

        def edge(start: np.ndarray, end: np.ndarray) -> Line:
            direction = end - start
            direction /= np.linalg.norm(direction)
            return Line(start + direction * self.node_radius, end - direction * self.node_radius).set_stroke(
                BLUE_D, width=1.8, opacity=0.48
            )

        path_edges = VGroup(
            edge(x_centers[0], z_centers[0]),
            edge(x_centers[0], z_centers[1]),
            edge(x_centers[1], z_centers[0]),
            edge(x_centers[1], z_centers[1]),
            edge(z_centers[0], h_centers[0]),
            edge(z_centers[1], h_centers[1]),
            edge(h_centers[0], yhat_center),
            edge(h_centers[1], yhat_center),
        )
        relu_dead_edge = path_edges[5]

        connector = Line(
            traveler_center + np.array([0.32, 0.0, 0.0]),
            np.array([column_x["x"] - self.node_radius - 0.08, 0.37, 0.0]),
        ).set_stroke(BLUE_D, width=2.2, opacity=0.62)

        x_name_labels = VGroup(
            *[
                Text(name, font_size=22, color=BLUE_A).move_to(center + np.array([0.0, 0.50, 0.0]))
                for name, center in zip(("x₁", "x₂"), x_centers)
            ]
        )
        z_value_labels = VGroup(
            *[
                Text(f"{name} = {signed_text(value)}", font_size=20, color=WHITE).move_to(
                    center + np.array([0.0, 0.50, 0.0])
                )
                for name, value, center in zip(("z₁", "z₂"), z_values, z_centers)
            ]
        )
        h_value_labels = VGroup(
            Text(f"h₁ = {signed_text(h_values[0])}", font_size=20, color=WHITE).move_to(
                h_centers[0] + np.array([0.0, 0.50, 0.0])
            ),
            Text("h₂ = 0.000", font_size=20, color=BLUE_D).move_to(h_centers[1] + np.array([0.0, 0.50, 0.0])),
        )
        yhat_name = Text("ŷ", font_size=22, color=YELLOW)
        yhat_number = Text(signed_text(yhat), font_size=18, color=YELLOW)
        yhat_value_label = VGroup(yhat_name, yhat_number).arrange(np.array([0.0, -1.0, 0.0]), buff=0.04)
        yhat_value_label.move_to(yhat_center + np.array([0.16, 0.72, 0.0]))

        path_title = Text("x → z → h → ŷ", font_size=34, color=WHITE).move_to(np.array([3.85, 3.12, 0.0]))
        chain_formula = Text("∂ℓ/∂h = (∂ℓ/∂ŷ)(∂ŷ/∂h)", font_size=28, color=WHITE).move_to(
            np.array([3.85, 3.12, 0.0])
        )

        grad_yhat_reveal = ValueTracker(0.0)
        grad_h_reveal = [ValueTracker(0.0) for _ in range(2)]
        grad_x_reveal = [ValueTracker(0.0) for _ in range(2)]
        grad_yhat_arrow = self.make_back_arrow(yhat_center, d_yhat, grad_yhat_reveal)
        grad_h_arrows = VGroup(
            *[self.make_back_arrow(h_centers[index], d_h[index], grad_h_reveal[index]) for index in range(2)]
        )
        grad_x_arrows = VGroup(
            *[self.make_back_arrow(x_centers[index], d_x[index], grad_x_reveal[index]) for index in range(2)]
        )

        def gradient_caption(name: str, value: float, position: np.ndarray) -> VGroup:
            """Two-line caption so ∂ℓ/∂· never becomes a wide HUD colliding with neighbors."""
            title = Text(name, font_size=17, color=YELLOW_A)
            number = Text(signed_text(value), font_size=18, color=YELLOW)
            caption = VGroup(title, number).arrange(np.array([0.0, -1.0, 0.0]), buff=0.05)
            caption.move_to(position)
            return caption

        grad_yhat_label = gradient_caption("∂ℓ/∂ŷ", d_yhat, yhat_center + np.array([0.02, -0.72, 0.0]))
        grad_h_labels = VGroup(
            gradient_caption("∂ℓ/∂h₁", d_h[0], h_centers[0] + np.array([-0.62, -0.54, 0.0])),
            gradient_caption("∂ℓ/∂h₂", d_h[1], h_centers[1] + np.array([-0.08, -0.54, 0.0])),
        )
        grad_x_labels = VGroup(
            gradient_caption("∂ℓ/∂x₁", d_x[0], x_centers[0] + np.array([-0.70, -0.54, 0.0])),
            gradient_caption("∂ℓ/∂x₂", d_x[1], x_centers[1] + np.array([-0.08, -0.54, 0.0])),
        )

        # 0.40–3.50 s: establish the 2-D cloud and traveler before any path.
        self.play(
            FadeIn(grid),
            FadeIn(axes),
            LaggedStart(*[FadeIn(point) for point in point_cloud], lag_ratio=0.025),
            run_time=0.80,
            rate_func=linear,
        )
        self.play(
            Create(halo_outer),
            Create(halo_inner),
            FadeIn(traveler_x1_label),
            FadeIn(traveler_x2_label),
            FadeIn(traveler_y_label),
            Flash(traveler_center, color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=0.60,
            rate_func=smooth,
        )
        self.wait(1.70)

        # 3.50–5.10 s: the same path that reverse will later walk.
        self.play(
            Create(connector),
            FadeIn(path_edges),
            FadeIn(VGroup(*x_nodes, *z_nodes, *h_nodes, yhat_node)),
            FadeIn(x_name_labels),
            FadeIn(path_title),
            run_time=1.20,
            rate_func=smooth,
        )
        self.wait(0.40)

        # 5.10–7.00 s: x lights along the path.
        self.play(
            x_reveal[0].animate.set_value(1.0),
            x_reveal[1].animate.set_value(1.0),
            path_edges[0].animate.set_stroke(YELLOW, width=2.6, opacity=0.90),
            path_edges[1].animate.set_stroke(YELLOW, width=2.6, opacity=0.90),
            path_edges[2].animate.set_stroke(YELLOW, width=2.6, opacity=0.90),
            path_edges[3].animate.set_stroke(YELLOW, width=2.6, opacity=0.90),
            run_time=1.20,
            rate_func=smooth,
        )
        self.play(
            path_edges[0].animate.set_stroke(BLUE_D, width=1.8, opacity=0.62),
            path_edges[1].animate.set_stroke(BLUE_D, width=1.8, opacity=0.62),
            path_edges[2].animate.set_stroke(BLUE_D, width=1.8, opacity=0.62),
            path_edges[3].animate.set_stroke(BLUE_D, width=1.8, opacity=0.62),
            run_time=0.30,
            rate_func=linear,
        )
        self.wait(0.40)

        # 7.00–10.15 s: z needles of the path; the lower pre-activation is negative.
        self.play(
            z_reveal[0].animate.set_value(1.0),
            FadeIn(z_value_labels[0]),
            run_time=1.05,
            rate_func=smooth,
        )
        self.play(
            z_reveal[1].animate.set_value(1.0),
            FadeIn(z_value_labels[1]),
            run_time=1.05,
            rate_func=smooth,
        )
        self.wait(0.65)

        # 10.15–13.20 s: ReLU on the same path. Lower hidden unit is the zero.
        self.play(
            h_reveal[0].animate.set_value(1.0),
            FadeIn(h_value_labels[0]),
            path_edges[4].animate.set_stroke(YELLOW, width=2.6, opacity=0.90),
            run_time=1.15,
            rate_func=smooth,
        )
        self.play(path_edges[4].animate.set_stroke(BLUE_D, width=1.8, opacity=0.62), run_time=0.22, rate_func=linear)
        self.play(
            h_reveal[1].animate.set_value(1.0),
            FadeIn(h_value_labels[1]),
            relu_dead_edge.animate.set_stroke(BLUE_E, width=1.4, opacity=0.28),
            run_time=1.05,
            rate_func=smooth,
        )
        self.wait(0.63)

        # 13.20–16.00 s: ŷ, then residual / loss stick on the traveler.
        self.play(
            yhat_reveal.animate.set_value(1.0),
            FadeIn(yhat_value_label),
            path_edges[6].animate.set_stroke(YELLOW, width=2.6, opacity=0.90),
            path_edges[7].animate.set_stroke(YELLOW, width=2.6, opacity=0.90),
            run_time=1.20,
            rate_func=smooth,
        )
        self.play(
            path_edges[6].animate.set_stroke(BLUE_D, width=1.8, opacity=0.62),
            path_edges[7].animate.set_stroke(BLUE_D, width=1.8, opacity=0.62),
            FadeIn(residual_group),
            Flash(stick_top, color=YELLOW, flash_radius=0.28, line_length=0.08),
            run_time=0.80,
            rate_func=smooth,
        )
        self.wait(0.80)

        # 16.00–17.40 s: swap the path title for the one chain-rule line. No overlap.
        self.play(FadeOut(path_title), run_time=0.28, rate_func=linear)
        self.wait(0.12)
        self.play(FadeIn(chain_formula), run_time=0.50, rate_func=smooth)
        self.wait(0.50)

        # 17.40–20.10 s: reverse starts at ŷ.
        self.add(grad_yhat_arrow)
        self.play(
            grad_yhat_reveal.animate.set_value(1.0),
            FadeIn(grad_yhat_label),
            path_edges[6].animate.set_stroke(YELLOW, width=2.8, opacity=0.92),
            path_edges[7].animate.set_stroke(YELLOW, width=2.8, opacity=0.92),
            run_time=1.35,
            rate_func=smooth,
        )
        self.wait(0.85)
        self.play(
            path_edges[6].animate.set_stroke(BLUE_D, width=1.8, opacity=0.62),
            path_edges[7].animate.set_stroke(BLUE_D, width=1.8, opacity=0.62),
            run_time=0.22,
            rate_func=linear,
        )

        # 20.10–23.70 s: local gradients on the same hidden nodes.
        self.add(grad_h_arrows)
        self.play(
            grad_h_reveal[0].animate.set_value(1.0),
            FadeIn(grad_h_labels[0]),
            path_edges[4].animate.set_stroke(YELLOW, width=2.8, opacity=0.92),
            run_time=1.20,
            rate_func=smooth,
        )
        self.play(path_edges[4].animate.set_stroke(BLUE_D, width=1.8, opacity=0.62), run_time=0.18, rate_func=linear)
        self.play(
            grad_h_reveal[1].animate.set_value(1.0),
            FadeIn(grad_h_labels[1]),
            run_time=1.10,
            rate_func=smooth,
        )
        self.wait(0.72)

        # 23.70–27.40 s: reverse arrives at x, and the traveler is the same sample.
        self.add(grad_x_arrows)
        self.play(
            grad_x_reveal[0].animate.set_value(1.0),
            grad_x_reveal[1].animate.set_value(1.0),
            FadeIn(grad_x_labels),
            path_edges[0].animate.set_stroke(YELLOW, width=2.8, opacity=0.92),
            path_edges[2].animate.set_stroke(YELLOW, width=2.8, opacity=0.92),
            Flash(traveler_center, color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=1.45,
            rate_func=smooth,
        )
        self.play(
            path_edges[0].animate.set_stroke(BLUE_D, width=1.8, opacity=0.62),
            path_edges[2].animate.set_stroke(BLUE_D, width=1.8, opacity=0.62),
            run_time=0.22,
            rate_func=linear,
        )
        self.wait(1.03)

        # Hold live updaters through the final encoded frame.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=5.20, rate_func=linear)
