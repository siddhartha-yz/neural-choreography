"""D2L 7.5 — batch norm on a tiny batch of five scalars.

Every moving value comes from the arrays below. Mean and std are the batch
statistics of the same five points drawn on screen. γ and β are fixed (not
trained): identity first, then one visible affine shift.
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


# Five raw feature values. The traveler sits at the batch mean, so after
# centering and scaling it remains at standardized 0 until β moves it.
BATCH_X = np.array([0.40, 1.20, 2.00, 2.80, 3.60], dtype=float)
TRAVELER_INDEX = 2
EPS = 0.0
BATCH_MU = float(BATCH_X.mean())
BATCH_VAR = float(((BATCH_X - BATCH_MU) ** 2).mean() + EPS)
BATCH_SIGMA = float(np.sqrt(BATCH_VAR))
X_HAT = (BATCH_X - BATCH_MU) / BATCH_SIGMA
GAMMA_ID = 1.00
BETA_ID = 0.00
GAMMA = 1.50
BETA = 0.40
Y_OUT = GAMMA * X_HAT + BETA


class Episode075(Scene):
    """A ~32-second continuous, silent batch-norm visualization."""

    baseline_y = -0.85
    needle_scale = 0.92
    bar_x = (-3.10, -1.20, 0.70, 2.60, 4.50)
    unit_dot_radius = 0.078

    def value_to_y(self, value: float) -> float:
        return self.baseline_y + value * self.needle_scale

    def construct(self) -> None:
        print(
            "D2L 7.5 x={}, mu={:.6f}, sigma={:.6f}, xhat={}, gamma={:.2f}, beta={:.2f}, y={}".format(
                np.array2string(BATCH_X, precision=3, separator=", "),
                BATCH_MU,
                BATCH_SIGMA,
                np.array2string(X_HAT, precision=6, separator=", "),
                GAMMA,
                BETA,
                np.array2string(Y_OUT, precision=6, separator=", "),
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "07.5")

        axis = Line(np.array([-5.35, self.baseline_y, 0.0]), np.array([5.85, self.baseline_y, 0.0])).set_stroke(
            BLUE_D, width=2.0, opacity=0.88
        )
        zero_label = Text("0", font_size=22, color=BLUE_A).next_to(
            np.array([-5.35, self.baseline_y, 0.0]), np.array([-1.0, 0.0, 0.0]), buff=0.16
        )

        # Stroke-only ruler: ticks are Line segments, never a filled mesh.
        ruler = VGroup()
        ruler_x = -5.05
        for tick_value in (-2.0, -1.0, 1.0, 2.0, 3.0, 4.0):
            y_pos = self.value_to_y(tick_value)
            ruler.add(
                Line(np.array([ruler_x - 0.12, y_pos, 0.0]), np.array([ruler_x + 0.12, y_pos, 0.0])).set_stroke(
                    BLUE_E, width=1.4, opacity=0.55
                )
            )
            tick_name = f"{tick_value:.0f}" if tick_value > 0 else f"−{abs(tick_value):.0f}"
            ruler.add(Text(tick_name, font_size=18, color=BLUE_A).move_to(np.array([ruler_x - 0.42, y_pos, 0.0])))

        formula = Text("y = γ((x−μ)/σ)+β", font_size=34, color=WHITE).move_to(np.array([0.55, 3.12, 0.0]))
        gamma_prefix = Text("γ = ", font_size=26, color=YELLOW_A)
        gamma_number = DecimalNumber(
            GAMMA_ID, num_decimal_places=2, mob_class=Text, include_sign=False, color=WHITE, font_size=26
        )
        gamma_row = VGroup(gamma_prefix, gamma_number).arrange(np.array([1.0, 0.0, 0.0]), buff=0.06)
        beta_prefix = Text("β = ", font_size=26, color=YELLOW_A)
        beta_number = DecimalNumber(
            BETA_ID, num_decimal_places=2, mob_class=Text, include_sign=True, color=WHITE, font_size=26
        )
        beta_row = VGroup(beta_prefix, beta_number).arrange(np.array([1.0, 0.0, 0.0]), buff=0.06)
        affine_readout = VGroup(gamma_row, beta_row).arrange(np.array([0.0, -1.0, 0.0]), buff=0.18).move_to(
            np.array([5.55, 3.00, 0.0])
        )

        unit_colors = (BLUE, BLUE_D, YELLOW, BLUE, BLUE_D)
        display = [ValueTracker(0.0) for _ in BATCH_X]
        mu_tracker = ValueTracker(BATCH_MU)
        gamma_tracker = ValueTracker(GAMMA_ID)
        beta_tracker = ValueTracker(BETA_ID)
        gamma_number.add_updater(lambda mob: mob.set_value(gamma_tracker.get_value()))
        beta_number.add_updater(lambda mob: mob.set_value(beta_tracker.get_value()))

        def needle_points(index: int) -> tuple[np.ndarray, np.ndarray]:
            x_position = self.bar_x[index]
            start = np.array([x_position, self.baseline_y, 0.0])
            end = np.array([x_position, self.value_to_y(display[index].get_value()), 0.0])
            return start, end

        needles = VGroup(
            *[
                always_redraw(
                    lambda index=index: Line(*needle_points(index)).set_stroke(
                        unit_colors[index], width=11.0, opacity=0.98
                    )
                )
                for index in range(len(BATCH_X))
            ]
        )

        def make_dot(index: int):
            def draw_dot() -> Dot:
                _, tip = needle_points(index)
                return (
                    Dot(tip, radius=self.unit_dot_radius, color=unit_colors[index])
                    .set_fill(unit_colors[index], opacity=0.96)
                    .set_stroke(unit_colors[index], width=0.0, opacity=0.0)
                )

            return always_redraw(draw_dot)

        dots = VGroup(*[make_dot(index) for index in range(len(BATCH_X))])

        def traveler_tip() -> np.ndarray:
            return needle_points(TRAVELER_INDEX)[1]

        halo_outer = always_redraw(
            lambda: Circle(radius=0.278).move_to(traveler_tip()).set_stroke(BLUE_A, width=2.1, opacity=0.50)
        )
        halo_inner = always_redraw(
            lambda: Circle(radius=0.178).move_to(traveler_tip()).set_stroke(YELLOW, width=2.8, opacity=0.98)
        )

        def make_value_label(index: int) -> VGroup:
            number = DecimalNumber(
                0.0,
                num_decimal_places=3,
                mob_class=Text,
                include_sign=True,
                color=WHITE,
                font_size=22,
            )
            number.add_updater(lambda mob, index=index: mob.set_value(display[index].get_value()))

            def place(mob, index=index):
                value = display[index].get_value()
                _, tip = needle_points(index)
                if abs(value) < 0.14:
                    mob.move_to(tip + np.array([0.62, 0.02, 0.0]))
                else:
                    direction = 1.0 if value >= 0 else -1.0
                    mob.move_to(tip + np.array([0.0, direction * 0.36, 0.0]))

            number.add_updater(place)
            return number

        value_labels = VGroup(*[make_value_label(index) for index in range(len(BATCH_X))])

        traveler_prefix = Text("x = ", font_size=24, color=YELLOW)
        traveler_number = DecimalNumber(
            0.0, num_decimal_places=3, mob_class=Text, include_sign=True, color=YELLOW_A, font_size=24
        )
        traveler_pocket = VGroup(traveler_prefix, traveler_number).arrange(
            np.array([1.0, 0.0, 0.0]), buff=0.08
        ).move_to(np.array([-5.70, -3.42, 0.0]))
        traveler_number.add_updater(lambda mob: mob.set_value(display[TRAVELER_INDEX].get_value()))

        mu_line = always_redraw(
            lambda: Line(
                np.array([self.bar_x[0] - 0.55, self.value_to_y(mu_tracker.get_value()), 0.0]),
                np.array([self.bar_x[-1] + 0.55, self.value_to_y(mu_tracker.get_value()), 0.0]),
            ).set_stroke(YELLOW_A, width=2.4, opacity=0.90)
        )
        mu_prefix = Text("μ = ", font_size=24, color=YELLOW_A)
        mu_number = DecimalNumber(
            BATCH_MU, num_decimal_places=3, mob_class=Text, include_sign=False, color=YELLOW_A, font_size=24
        )
        mu_row = VGroup(mu_prefix, mu_number).arrange(np.array([1.0, 0.0, 0.0]), buff=0.08)
        mu_number.add_updater(lambda mob: mob.set_value(mu_tracker.get_value()))
        mu_row.add_updater(
            lambda mob: mob.move_to(
                np.array([self.bar_x[-1] + 1.42, self.value_to_y(mu_tracker.get_value()), 0.0])
            )
        )

        sigma_brace = Line(
            np.array([self.bar_x[0] - 0.72, self.baseline_y, 0.0]),
            np.array([self.bar_x[0] - 0.72, self.value_to_y(BATCH_SIGMA), 0.0]),
        ).set_stroke(BLUE_A, width=3.0, opacity=0.95)
        sigma_label = Text("σ = 1.131", font_size=22, color=BLUE_A).next_to(
            sigma_brace, np.array([0.0, 1.0, 0.0]), buff=0.14
        )

        # 0.40–4.50 s: the five raw values, traveler on the batch mean.
        self.play(FadeIn(axis), FadeIn(zero_label), FadeIn(ruler), run_time=0.70, rate_func=linear)
        self.add(needles, dots)
        self.play(
            *[display[index].animate.set_value(float(BATCH_X[index])) for index in range(len(BATCH_X))],
            FadeIn(value_labels),
            FadeIn(traveler_pocket),
            run_time=1.35,
            rate_func=smooth,
        )
        self.add(halo_outer, halo_inner)
        self.play(
            Flash(traveler_tip(), color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=0.55,
            rate_func=smooth,
        )
        self.wait(1.50)

        # 4.50–7.50 s: write the formula once; μ sits on the traveler.
        self.play(FadeIn(formula), run_time=0.50, rate_func=smooth)
        self.play(FadeIn(mu_line), FadeIn(mu_row), run_time=0.55, rate_func=smooth)
        self.wait(2.05)

        # 7.50–10.80 s: subtract the batch mean. Traveler goes to 0.
        self.play(
            *[display[index].animate.set_value(float(BATCH_X[index] - BATCH_MU)) for index in range(len(BATCH_X))],
            mu_tracker.animate.set_value(0.0),
            run_time=1.90,
            rate_func=smooth,
        )
        self.wait(1.40)
        mu_number.clear_updaters()
        mu_row.clear_updaters()
        self.play(FadeOut(mu_line), FadeOut(mu_row), run_time=0.35, rate_func=linear)

        # 11.15–15.40 s: σ on the centered batch, then divide so std is 1.
        self.play(FadeIn(sigma_brace), FadeIn(sigma_label), run_time=0.50, rate_func=smooth)
        self.wait(0.90)
        self.play(
            *[display[index].animate.set_value(float(X_HAT[index])) for index in range(len(BATCH_X))],
            FadeOut(sigma_brace),
            FadeOut(sigma_label),
            run_time=1.90,
            rate_func=smooth,
        )
        self.wait(1.10)

        # 15.20–17.40 s: identity affine — traveler stays at standardized 0.
        self.play(FadeIn(affine_readout), run_time=0.50, rate_func=smooth)
        self.wait(1.70)

        # 17.40–21.00 s: a short, untrained γ, β shift.
        self.play(
            *[display[index].animate.set_value(float(Y_OUT[index])) for index in range(len(BATCH_X))],
            gamma_tracker.animate.set_value(GAMMA),
            beta_tracker.animate.set_value(BETA),
            run_time=2.20,
            rate_func=smooth,
        )
        self.wait(1.40)

        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=11.00, rate_func=linear)
