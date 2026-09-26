"""D2L 9.1 — the GRU update gate as a componentwise convex combination.

Only the NumPy tensors below supply the values seen in the scene.  The
animation is a fixed, untrained forward update: no recurrent parameters are
learned and no state is optimized.
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
    Dot,
    FadeIn,
    FadeOut,
    Flash,
    LaggedStart,
    Line,
    Scene,
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
)


config.background_color = BLACK


# All hidden-state tensors are 3 × 1 column vectors.  The update gate is in
# (0, 1), so the result is componentwise between old state and candidate.
H_PREV = np.array([[0.60], [-0.40], [0.20]], dtype=float)
Z = np.array([[0.82], [0.25], [0.60]], dtype=float)
H_CANDIDATE = np.array([[-0.20], [0.80], [-0.60]], dtype=float)
H_NEXT = Z * H_PREV + (1.0 - Z) * H_CANDIDATE


def signed(value: float, decimals: int = 2) -> str:
    """Formatting shared by all state values in the scene."""
    return f"{'+' if value >= 0 else '−'}{abs(value):.{decimals}f}"


class Episode091(Scene):
    """A slow, continuous GRU update-gate mechanism visualization."""

    rows = (1.35, 0.0, -1.35)
    candidate_x = -4.60
    old_x = -1.50
    output_x = 3.80

    def construct(self) -> None:
        print(
            "D2L 9.1 H_prev={}, Z={}, H_candidate={}, H_next={}".format(
                np.array2string(H_PREV.ravel(), precision=3, separator=", "),
                np.array2string(Z.ravel(), precision=3, separator=", "),
                np.array2string(H_CANDIDATE.ravel(), precision=3, separator=", "),
                np.array2string(H_NEXT.ravel(), precision=3, separator=", "),
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "09.1")

        # Three clean component rails are strokes only; there is no filled mesh.
        rails = VGroup(
            *[
                Line(
                    np.array([self.candidate_x, row, 0.0]),
                    np.array([self.old_x, row, 0.0]),
                ).set_stroke(BLUE_E, width=2.0, opacity=0.56)
                for row in self.rows
            ]
        )
        old_dots = VGroup(
            *[
                Dot(np.array([self.old_x, row, 0.0]), radius=0.088, color=BLUE).set_fill(BLUE, opacity=0.98)
                for row in self.rows
            ]
        )
        candidate_dots = VGroup(
            *[
                Dot(np.array([self.candidate_x, row, 0.0]), radius=0.080, color=YELLOW).set_fill(YELLOW, opacity=0.98)
                for row in self.rows
            ]
        )
        mix_dots = VGroup(
            *[
                Dot(np.array([self.candidate_x, row, 0.0]), radius=0.088, color=WHITE).set_fill(WHITE, opacity=0.98)
                for row in self.rows
            ]
        )
        output_dots = VGroup(
            *[
                Dot(np.array([self.output_x, row, 0.0]), radius=0.088, color=BLUE_A).set_fill(BLUE_A, opacity=0.98)
                for row in self.rows
            ]
        )

        old_header = Text("Hₜ₋₁", font_size=30, color=BLUE_A).move_to(np.array([self.old_x, 2.32, 0.0]))
        candidate_header = Text("H̃ₜ", font_size=30, color=YELLOW_A).move_to(
            np.array([self.candidate_x, 2.32, 0.0])
        )
        gate_header = Text("Zₜ", font_size=30, color=BLUE_A).move_to(np.array([-3.10, 2.32, 0.0]))
        output_header = Text("Hₜ", font_size=30, color=BLUE_A).move_to(np.array([self.output_x, 2.32, 0.0]))

        old_values = VGroup(
            *[
                Text(signed(value.item()), font_size=23, color=WHITE).next_to(dot, np.array([1.0, 0.0, 0.0]), buff=0.16)
                for dot, value in zip(old_dots, H_PREV)
            ]
        )
        candidate_values = VGroup(
            *[
                Text(signed(value.item()), font_size=23, color=WHITE).next_to(
                    dot, np.array([-1.0, 0.0, 0.0]), buff=0.16
                )
                for dot, value in zip(candidate_dots, H_CANDIDATE)
            ]
        )
        output_values = VGroup(
            *[
                Text(signed(value.item(), 3), font_size=23, color=WHITE).next_to(
                    dot, np.array([1.0, 0.0, 0.0]), buff=0.16
                )
                for dot, value in zip(output_dots, H_NEXT)
            ]
        )

        # The double stroked frame carries the familiar halo identity, now
        # around a 3-component hidden state rather than a scalar sample.
        old_halo_outer = SurroundingRectangle(old_dots, color=BLUE_A, buff=0.28, corner_radius=0.12)
        old_halo_outer.set_fill(opacity=0.0)
        old_halo_outer.set_stroke(width=2.0, opacity=0.48)
        old_halo_inner = SurroundingRectangle(old_dots, color=YELLOW, buff=0.17, corner_radius=0.10)
        old_halo_inner.set_fill(opacity=0.0)
        old_halo_inner.set_stroke(width=2.7, opacity=0.98)

        formula = Text(
            "Hₜ = Zₜ ⊙ Hₜ₋₁ + (1 − Zₜ) ⊙ H̃ₜ",
            font_size=31,
            color=WHITE,
        ).move_to(np.array([0.15, 3.20, 0.0]))

        # Each gate tick is positioned directly at the true convex-combination
        # location between candidate and old-state endpoints.
        mix_positions = [
            np.array(
                [
                    self.candidate_x + gate.item() * (self.old_x - self.candidate_x),
                    row,
                    0.0,
                ]
            )
            for gate, row in zip(Z, self.rows)
        ]
        gate_ticks = VGroup(
            *[
                Line(position + np.array([0.0, -0.18, 0.0]), position + np.array([0.0, 0.18, 0.0])).set_stroke(
                    BLUE_A, width=3.0, opacity=0.96
                )
                for position in mix_positions
            ]
        )
        gate_values = VGroup(
            *[
                Text(f"z{index + 1} = {gate.item():.2f}", font_size=22, color=BLUE_A).move_to(
                    position + np.array([0.0, 0.40, 0.0])
                )
                for index, (gate, position) in enumerate(zip(Z, mix_positions))
            ]
        )

        # 0.40–3.50 s: establish the old hidden state and let it sit.
        self.play(
            FadeIn(old_dots),
            FadeIn(old_values),
            FadeIn(old_header),
            run_time=0.80,
            rate_func=smooth,
        )
        self.play(
            Create(old_halo_outer),
            Create(old_halo_inner),
            Flash(old_dots.get_center(), color=YELLOW, flash_radius=0.48, line_length=0.11),
            run_time=0.55,
            rate_func=smooth,
        )
        self.wait(1.75)

        # 3.50–6.35 s: reveal the candidate and the fixed gate values.
        self.play(
            FadeIn(rails),
            FadeIn(candidate_dots),
            FadeIn(candidate_values),
            FadeIn(candidate_header),
            FadeIn(formula),
            run_time=0.85,
            rate_func=smooth,
        )
        self.wait(1.10)
        self.play(
            FadeIn(gate_header),
            Create(gate_ticks),
            FadeIn(gate_values),
            run_time=0.75,
            rate_func=smooth,
        )
        self.wait(1.10)

        # 6.35–9.00 s: white mixture markers stop exactly at the gate-selected
        # convex locations. The endpoints remain visible as old and candidate.
        self.add(mix_dots)
        self.play(
            LaggedStart(
                *[dot.animate.move_to(position) for dot, position in zip(mix_dots, mix_positions)],
                lag_ratio=0.12,
            ),
            run_time=2.20,
            rate_func=smooth,
        )
        self.wait(1.20)

        # 10.20–14.20 s: the three mixed components settle into H_t.
        self.play(
            FadeIn(output_header),
            LaggedStart(
                *[FadeIn(dot) for dot in output_dots],
                lag_ratio=0.18,
            ),
            run_time=1.10,
            rate_func=smooth,
        )
        self.play(
            LaggedStart(*[FadeIn(value) for value in output_values], lag_ratio=0.18),
            run_time=1.10,
            rate_func=smooth,
        )
        self.wait(1.80)

        # The output frame confirms the traveler's next hidden state while
        # keeping both H_{t-1} and H_t visible through the final hold.
        output_frame = SurroundingRectangle(output_dots, color=BLUE_A, buff=0.26, corner_radius=0.12)
        output_frame.set_fill(opacity=0.0)
        output_frame.set_stroke(width=2.3, opacity=0.92)
        self.play(
            Create(output_frame),
            Flash(output_dots.get_center(), color=YELLOW, flash_radius=0.44, line_length=0.10),
            run_time=0.60,
            rate_func=smooth,
        )

        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=10.10, rate_func=linear)
