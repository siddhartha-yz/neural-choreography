"""D2L 10.4 — Bahdanau: decoder hidden queries encoder steps.

Every α, every context needle, and the box that lights up come from the
additive-attention numpy below. Encoder hiddens are keys and values;
the halo traveler is the decoder query s. Untrained. Tiny 3×2 sequence.
"""

from __future__ import annotations

import numpy as np
from scripts.episode_intro import show_episode_intro
from manim import (
    BLACK,
    BLUE,
    BLUE_A,
    BLUE_E,
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Flash,
    LaggedStart,
    Line,
    Rectangle,
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
from manim.utils.color import interpolate_color


config.background_color = BLACK


# ---------------------------------------------------------------------------
# One numerical source of truth. Additive attention:
#   a(s, h) = vᵀ tanh(W_q s + W_k h),  α = softmax(a),  c = Σ α h
# Query s is the decoder hidden. Keys = values = encoder hiddens.
# ---------------------------------------------------------------------------
ENCODER_H = np.array(
    [
        [1.20, 0.20],
        [0.35, 0.95],
        [-0.85, 0.80],
    ],
    dtype=float,
)
QUERY_S0 = np.array([1.00, 0.50], dtype=float)
QUERY_S1 = np.array([-0.55, 1.05], dtype=float)
W_Q = np.array([[-0.48712729, 0.68103308], [-0.86384803, 3.08807261]], dtype=float)
W_K = np.array([[2.42289974, -0.64918240], [-2.02660596, 0.34492875]], dtype=float)
V_ATT = np.array([-1.17912302, -2.25548609], dtype=float)


def softmax(scores: np.ndarray) -> np.ndarray:
    """Softmax over encoder steps, the same α drawn on screen."""
    shifted = scores - np.max(scores)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def bahdanau(query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Additive scores, attention weights, and context for one decoder query."""
    energy = np.tanh((W_Q @ query)[None, :] + ENCODER_H @ W_K.T)
    scores = energy @ V_ATT
    alpha = softmax(scores)
    context = alpha @ ENCODER_H
    return scores, alpha, context


SCORES_0, ALPHA_0, CONTEXT_0 = bahdanau(QUERY_S0)
SCORES_1, ALPHA_1, CONTEXT_1 = bahdanau(QUERY_S1)


def signed_text(value: float, digits: int = 2) -> str:
    """True minus glyph for negative values, matching 3.1 / 9.2."""
    if value < 0:
        return f"−{abs(value):.{digits}f}"
    return f"{value:.{digits}f}"


class Episode104(Scene):
    """A ~32-second continuous, silent Bahdanau alignment visualization."""

    needle_scale = 0.80
    traveler_radius = 0.078
    halo_inner_r = 0.178
    halo_outer_r = 0.278
    box_w, box_h = 2.46, 2.48
    enc_xs = (-4.52, -1.52, 1.48)
    enc_y = 1.28
    ctx_x = 4.88
    path_y = -2.28
    needle_baseline_offset = -0.18
    alpha_y = 2.96
    ctx_label_y = -0.48
    step_label_y = -3.22
    formula_c = np.array([-1.55, 3.62, 0.0])
    colors = (BLUE, YELLOW)

    def stroke_rect(self, width: float, height: float, center: np.ndarray, color, stroke_width: float) -> Rectangle:
        rect = Rectangle(width=width, height=height)
        rect.move_to(center)
        rect.set_fill(BLACK, opacity=0.0)
        rect.set_stroke(color, width=stroke_width, opacity=0.94)
        return rect

    def box_center(self, index: int) -> np.ndarray:
        return np.array([self.enc_xs[index], self.enc_y, 0.0])

    def box_bottom(self, index: int) -> np.ndarray:
        return np.array([self.enc_xs[index], self.enc_y - self.box_h / 2, 0.0])

    def needle_xs(self, center_x: float) -> tuple[float, float]:
        return (center_x - 0.38, center_x + 0.10)

    def needle_baseline(self) -> float:
        return self.enc_y + self.needle_baseline_offset

    def make_needles(self, xs: tuple[float, float], value_fn, colors: tuple) -> VGroup:
        baseline = self.needle_baseline()

        def draw_needle(index: int):
            def redraw() -> Line:
                value = float(value_fn()[index])
                start = np.array([xs[index], baseline, 0.0])
                end = np.array([xs[index], baseline + value * self.needle_scale, 0.0])
                return Line(start, end).set_stroke(colors[index], width=8.0, opacity=0.98)

            return always_redraw(redraw)

        def draw_foot(index: int):
            def redraw() -> Dot:
                value = abs(float(value_fn()[index]))
                opacity = 0.94 if value > 1e-6 else 0.50
                return (
                    Dot(np.array([xs[index], baseline, 0.0]), radius=0.055, color=colors[index])
                    .set_fill(colors[index], opacity=opacity)
                    .set_stroke(colors[index], width=0.0, opacity=0.0)
                )

            return always_redraw(redraw)

        axis = Line(
            np.array([xs[0] - 0.22, baseline, 0.0]),
            np.array([xs[1] + 0.22, baseline, 0.0]),
        ).set_stroke(BLUE_E, width=1.6, opacity=0.84)
        return VGroup(axis, *[draw_foot(index) for index in range(2)], *[draw_needle(index) for index in range(2)])

    def construct(self) -> None:
        print(
            "D2L 10.4 H={}".format(np.array2string(ENCODER_H, precision=3, separator=", "))
        )
        print(
            "D2L 10.4 s0={} s1={}".format(
                np.array2string(QUERY_S0, precision=3, separator=", "),
                np.array2string(QUERY_S1, precision=3, separator=", "),
            )
        )
        print(
            "D2L 10.4 scores0={} alpha0={} sum0={:.6f} c0={}".format(
                np.array2string(SCORES_0, precision=4, separator=", "),
                np.array2string(ALPHA_0, precision=6, separator=", "),
                float(np.sum(ALPHA_0)),
                np.array2string(CONTEXT_0, precision=4, separator=", "),
            )
        )
        print(
            "D2L 10.4 scores1={} alpha1={} sum1={:.6f} c1={}".format(
                np.array2string(SCORES_1, precision=4, separator=", "),
                np.array2string(ALPHA_1, precision=6, separator=", "),
                float(np.sum(ALPHA_1)),
                np.array2string(CONTEXT_1, precision=4, separator=", "),
            )
        )
        print(
            "D2L 10.4 Wq={} Wk={} v={}".format(
                np.array2string(W_Q, precision=4, separator=", "),
                np.array2string(W_K, precision=4, separator=", "),
                np.array2string(V_ATT, precision=4, separator=", "),
            )
        )

        blend = ValueTracker(0.0)
        attn_reveal = ValueTracker(0.0)
        mix_reveal = ValueTracker(0.0)

        def current_query() -> np.ndarray:
            amount = float(blend.get_value())
            return (1.0 - amount) * QUERY_S0 + amount * QUERY_S1

        def current_alpha() -> np.ndarray:
            return bahdanau(current_query())[1]

        def current_context() -> np.ndarray:
            return bahdanau(current_query())[2]

        def traveler_point() -> np.ndarray:
            amount = float(blend.get_value())
            x_value = (1.0 - amount) * self.enc_xs[0] + amount * self.enc_xs[2]
            return np.array([x_value, self.path_y, 0.0])

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "10.4")

        enc_boxes = VGroup(
            *[
                self.stroke_rect(self.box_w, self.box_h, self.box_center(index), BLUE_A, 2.8)
                for index in range(3)
            ]
        )
        enc_names = VGroup(
            *[
                Text(name, font_size=28, color=BLUE_A).move_to(
                    self.box_center(index) + np.array([0.0, self.box_h / 2 - 0.22, 0.0])
                )
                for index, name in enumerate(("h₁", "h₂", "h₃"))
            ]
        )
        chain = VGroup(
            Line(
                np.array([self.enc_xs[0] + self.box_w / 2, self.enc_y, 0.0]),
                np.array([self.enc_xs[1] - self.box_w / 2, self.enc_y, 0.0]),
            ).set_stroke(BLUE_E, width=2.2, opacity=0.70),
            Line(
                np.array([self.enc_xs[1] + self.box_w / 2, self.enc_y, 0.0]),
                np.array([self.enc_xs[2] - self.box_w / 2, self.enc_y, 0.0]),
            ).set_stroke(BLUE_E, width=2.2, opacity=0.70),
        )

        h_trackers = [[ValueTracker(0.0), ValueTracker(0.0)] for _ in range(3)]
        h_needles = VGroup(
            *[
                self.make_needles(
                    self.needle_xs(self.enc_xs[index]),
                    lambda step=index: (
                        h_trackers[step][0].get_value(),
                        h_trackers[step][1].get_value(),
                    ),
                    self.colors,
                )
                for index in range(3)
            ]
        )
        h_value_labels = VGroup()
        for index in range(3):
            column = VGroup(
                Text(signed_text(float(ENCODER_H[index, 0])), font_size=18, color=BLUE),
                Text(signed_text(float(ENCODER_H[index, 1])), font_size=18, color=YELLOW),
            ).arrange(np.array([0.0, -1.0, 0.0]), buff=0.08)
            column.move_to(self.box_center(index) + np.array([0.82, 0.22, 0.0]))
            h_value_labels.add(column)

        ctx_center = np.array([self.ctx_x, self.enc_y, 0.0])
        ctx_box = self.stroke_rect(self.box_w, self.box_h, ctx_center, YELLOW_A, 2.8)
        ctx_name = Text("c", font_size=28, color=YELLOW).move_to(
            ctx_center + np.array([0.0, self.box_h / 2 - 0.22, 0.0])
        )
        c_needles = self.make_needles(
            self.needle_xs(self.ctx_x),
            lambda: current_context() * float(mix_reveal.get_value()),
            self.colors,
        )

        def c_column() -> VGroup:
            context = current_context() * float(mix_reveal.get_value())
            rows = VGroup(
                Text(f"c₁ = {signed_text(float(context[0]))}", font_size=20, color=BLUE),
                Text(f"c₂ = {signed_text(float(context[1]))}", font_size=20, color=YELLOW),
            ).arrange(np.array([0.0, -1.0, 0.0]), buff=0.10)
            return rows.move_to(np.array([self.ctx_x, self.ctx_label_y, 0.0]))

        c_labels = always_redraw(c_column)

        def style_encoder_box(index: int) -> None:
            def update(mobject: Rectangle) -> None:
                alpha = float(current_alpha()[index]) * float(attn_reveal.get_value())
                glow = min(max((alpha - 0.05) / 0.55, 0.0), 1.0)
                color = interpolate_color(BLUE_A, YELLOW, glow)
                mobject.set_fill(BLACK, opacity=0.0)
                mobject.set_stroke(color, width=2.6 + 6.4 * glow, opacity=0.94)

            enc_boxes[index].add_updater(update)
            update(enc_boxes[index])

        formula = Text("c = Σ α(s, h) h", font_size=32, color=WHITE).move_to(self.formula_c)

        decoder_path = Line(
            np.array([self.enc_xs[0] - 0.15, self.path_y, 0.0]),
            np.array([self.enc_xs[2] + 0.15, self.path_y, 0.0]),
        ).set_stroke(BLUE_E, width=2.0, opacity=0.55)
        stations = VGroup(
            *[
                Circle(radius=0.34)
                .move_to(np.array([self.enc_xs[index], self.path_y, 0.0]))
                .set_stroke(BLUE_A, width=2.0, opacity=0.80)
                .set_fill(BLACK, opacity=0.0)
                for index in (0, 2)
            ]
        )
        step_tags = VGroup(
            Text("t′=1", font_size=22, color=WHITE).move_to(np.array([self.enc_xs[0], self.step_label_y, 0.0])),
            Text("t′=2", font_size=22, color=WHITE).move_to(np.array([self.enc_xs[2], self.step_label_y, 0.0])),
        )

        halo_outer = always_redraw(
            lambda: Circle(radius=self.halo_outer_r)
            .move_to(traveler_point())
            .set_stroke(BLUE_A, width=2.1, opacity=0.50)
            .set_fill(BLACK, opacity=0.0)
        )
        halo_inner = always_redraw(
            lambda: Circle(radius=self.halo_inner_r)
            .move_to(traveler_point())
            .set_stroke(YELLOW, width=2.8, opacity=0.98)
            .set_fill(BLACK, opacity=0.0)
        )
        traveler_dot = always_redraw(
            lambda: Dot(traveler_point(), radius=self.traveler_radius, color=YELLOW)
            .set_fill(YELLOW, opacity=1.0)
            .set_stroke(YELLOW, width=0.0, opacity=0.0)
        )

        def query_column() -> VGroup:
            query = current_query()
            rows = VGroup(
                Text(f"s₁ = {signed_text(float(query[0]))}", font_size=21, color=BLUE),
                Text(f"s₂ = {signed_text(float(query[1]))}", font_size=21, color=YELLOW),
            ).arrange(np.array([0.0, -1.0, 0.0]), buff=0.10)
            return rows.move_to(np.array([-6.18, -2.35, 0.0]))

        s_labels = always_redraw(query_column)
        s_tag = always_redraw(
            lambda: Text("s", font_size=22, color=YELLOW).move_to(traveler_point() + np.array([-0.55, 0.50, 0.0]))
        )

        def make_ray(index: int):
            def draw() -> Line:
                alpha = float(current_alpha()[index]) * float(attn_reveal.get_value())
                start = traveler_point() + np.array([0.0, 0.30, 0.0])
                end = self.box_bottom(index) + np.array([0.0, -0.04, 0.0])
                opacity = 0.0 if alpha < 0.025 else min(0.22 + 1.55 * alpha, 0.96)
                width = 2.0 + 9.0 * alpha
                return Line(start, end).set_stroke(YELLOW, width=width, opacity=opacity)

            return always_redraw(draw)

        rays = VGroup(*[make_ray(index) for index in range(3)])

        alpha_subs = ("α₁", "α₂", "α₃")

        def make_alpha_label(index: int):
            def draw() -> Text:
                weight = float(current_alpha()[index]) * float(attn_reveal.get_value())
                label = Text(
                    f"{alpha_subs[index]} = {weight:.3f}",
                    font_size=20,
                    color=YELLOW_A,
                )
                return label.move_to(np.array([self.enc_xs[index], self.alpha_y, 0.0]))

            return always_redraw(draw)

        alpha_labels = VGroup(*[make_alpha_label(index) for index in range(3)])
        sum_label = Text("Σα = 1.000", font_size=22, color=WHITE).move_to(
            np.array([3.20, 3.62, 0.0])
        )

        # 0.40–2.50 s: three encoder steps, keys and values.
        self.play(LaggedStart(*[Create(box) for box in enc_boxes], lag_ratio=0.12), Create(chain), run_time=1.20, rate_func=smooth)
        self.play(FadeIn(enc_names), run_time=0.90, rate_func=smooth)

        # 2.50–4.40 s: encoder hiddens stand in the boxes.
        self.add(h_needles)
        self.play(
            *[h_trackers[index][0].animate.set_value(float(ENCODER_H[index, 0])) for index in range(3)],
            *[h_trackers[index][1].animate.set_value(float(ENCODER_H[index, 1])) for index in range(3)],
            FadeIn(h_value_labels),
            run_time=1.90,
            rate_func=smooth,
        )

        # 4.40–5.20 s: empty context box, the mix has a place to land.
        self.play(Create(ctx_box), FadeIn(ctx_name), run_time=0.80, rate_func=smooth)

        # 5.20–7.20 s: halo traveler is the decoder query at t′=1.
        self.play(Create(decoder_path), Create(stations), FadeIn(step_tags), run_time=0.70, rate_func=smooth)
        self.add(halo_outer, halo_inner, traveler_dot, s_labels, s_tag)
        self.play(
            Flash(traveler_point(), color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=0.70,
            rate_func=smooth,
        )
        self.wait(0.60)

        # 7.20–8.70 s: formula once, in the empty pocket above the source.
        self.play(FadeIn(formula), run_time=0.70, rate_func=smooth)
        self.wait(0.80)

        # 8.70–11.10 s: query lights encoder step 1; alignment rays grow.
        for index in range(3):
            style_encoder_box(index)
        self.add(rays)
        self.play(attn_reveal.animate.set_value(1.0), run_time=2.40, rate_func=smooth)

        # 11.10–13.00 s: α sits on the encoder steps, and sums to 1.
        self.add(alpha_labels)
        self.play(FadeIn(sum_label), run_time=0.90, rate_func=smooth)
        self.wait(1.00)

        # 13.00–15.00 s: weighted mix — context needles grow to Σ α h.
        self.add(c_needles, c_labels)
        self.play(mix_reveal.animate.set_value(1.0), run_time=2.00, rate_func=smooth)

        # 15.00–17.20 s: first alignment holds.
        self.wait(2.20)

        # 17.20–20.00 s: decoder step 2 — query walks; alignment moves with it.
        self.play(blend.animate.set_value(1.0), run_time=2.80, rate_func=smooth)

        # 20.00–32.00 s: last frame still has the alignment.
        self.wait(2.20)
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=9.80, rate_func=linear)
