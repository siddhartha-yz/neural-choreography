"""D2L 9.2 — LSTM cell state as a persistent reservoir.

Every moving value comes from the two-step numpy LSTM below.  Forget
scales the standing cell; input adds a gated candidate; the cell stays
in the box while the output gate releases H.  Untrained.  Not a gate
dashboard of weight matrices.
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
    Rectangle,
    Scene,
    Text,
    VGroup,
    ValueTracker,
    WHITE,
    YELLOW,
    config,
    linear,
    smooth,
    always_redraw,
)


config.background_color = BLACK


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Element-wise logistic sigmoid for the same gates drawn on screen."""
    return 1.0 / (1.0 + np.exp(-values))


def lstm_step(
    x_t: np.ndarray,
    h_prev: np.ndarray,
    c_prev: np.ndarray,
) -> dict[str, np.ndarray]:
    """One untrained LSTM step: F, I, O, candidate, C, H."""
    forget = sigmoid(W_XF @ x_t + W_HF @ h_prev + B_F)
    input_gate = sigmoid(W_XI @ x_t + W_HI @ h_prev + B_I)
    output_gate = sigmoid(W_XO @ x_t + W_HO @ h_prev + B_O)
    candidate = np.tanh(W_XC @ x_t + W_HC @ h_prev + B_C)
    cell = forget * c_prev + input_gate * candidate
    hidden = output_gate * np.tanh(cell)
    return {
        "F": forget,
        "I": input_gate,
        "O": output_gate,
        "C_tilde": candidate,
        "C": cell,
        "H": hidden,
    }


# Tiny 2-D LSTM.  Step 1 forgets component 1 and writes it; step 2
# keeps component 1 and rewrites component 2.  The cell persists.
INPUT_X1 = np.array([[1.00], [0.50]], dtype=float)
INPUT_X2 = np.array([[0.40], [1.00]], dtype=float)
HIDDEN_0 = np.array([[0.25], [0.10]], dtype=float)
CELL_0 = np.array([[1.50], [1.10]], dtype=float)

W_XF = np.array([[-3.10613331, 3.43967790], [3.69424790, -2.99404665]], dtype=float)
W_HF = np.array([[0.15, -0.05], [0.08, 0.10]], dtype=float)
B_F = np.array([[-0.0325], [-0.0300]], dtype=float)
W_XI = np.array([[2.41950063, -2.06641254], [-2.59930193, 2.42601513]], dtype=float)
W_HI = np.array([[0.10, 0.05], [-0.06, 0.08]], dtype=float)
B_I = np.array([[-0.0300], [0.0070]], dtype=float)
W_XO = np.array([[0.84370420, 0.50981618], [0.19268835, 1.30921902]], dtype=float)
W_HO = np.array([[0.08, 0.04], [0.05, 0.09]], dtype=float)
B_O = np.array([[-0.0240], [-0.0215]], dtype=float)
W_XC = np.array([[1.71356652, -0.48269405], [-1.10436152, 1.69789742]], dtype=float)
W_HC = np.array([[0.12, 0.04], [-0.05, 0.07]], dtype=float)
B_C = np.array([[-0.0340], [0.0055]], dtype=float)

STEP1 = lstm_step(INPUT_X1, HIDDEN_0, CELL_0)
STEP2 = lstm_step(INPUT_X2, STEP1["H"], STEP1["C"])


def signed_text(value: float, digits: int = 2) -> str:
    """True minus glyph for negative values, matching 3.1 / 5.1."""
    if value < 0:
        return f"−{abs(value):.{digits}f}"
    return f"{value:.{digits}f}"


class Episode092(Scene):
    """A ~34-second continuous, silent LSTM cell-state visualization."""

    needle_scale = 2.15
    needle_baseline_y = -1.72
    traveler_radius = 0.078
    halo_inner_r = 0.178
    halo_outer_r = 0.278
    path_y = -2.42

    cell_c = np.array([0.10, 0.22, 0.0])
    cell_w, cell_h = 3.55, 5.05
    x_station = -5.55
    h_station = 4.72
    c_xs = (-0.42, 0.62)
    h_xs = (4.32, 5.12)
    pocket_y = (-3.18, -3.58)

    def stroke_rect(self, width: float, height: float, center: np.ndarray, color, stroke_width: float) -> Rectangle:
        rect = Rectangle(width=width, height=height)
        rect.move_to(center)
        rect.set_fill(BLACK, opacity=0.0)
        rect.set_stroke(color, width=stroke_width, opacity=0.94)
        return rect

    def station_axis(self, xs: tuple[float, float]) -> Line:
        return Line(
            np.array([xs[0] - 0.28, self.needle_baseline_y, 0.0]),
            np.array([xs[1] + 0.28, self.needle_baseline_y, 0.0]),
        ).set_stroke(BLUE_D, width=1.7, opacity=0.84)

    def make_needles(self, xs: tuple[float, float], trackers: list[ValueTracker], colors: tuple) -> VGroup:
        def draw_needle(index: int):
            def redraw() -> Line:
                start = np.array([xs[index], self.needle_baseline_y, 0.0])
                end = np.array(
                    [
                        xs[index],
                        self.needle_baseline_y + trackers[index].get_value() * self.needle_scale,
                        0.0,
                    ]
                )
                return Line(start, end).set_stroke(colors[index], width=8.5, opacity=0.98)

            return always_redraw(redraw)

        def draw_foot(index: int):
            def redraw() -> Dot:
                value = abs(trackers[index].get_value())
                opacity = 0.94 if value > 1e-6 else 0.50
                return (
                    Dot(
                        np.array([xs[index], self.needle_baseline_y, 0.0]),
                        radius=0.055,
                        color=colors[index],
                    )
                    .set_fill(colors[index], opacity=opacity)
                    .set_stroke(colors[index], width=0.0, opacity=0.0)
                )

            return always_redraw(redraw)

        return VGroup(
            self.station_axis(xs),
            *[draw_foot(index) for index in range(2)],
            *[draw_needle(index) for index in range(2)],
        )

    def make_pair_column(self, names: tuple[str, str], values: np.ndarray, colors: tuple, signed: tuple[bool, bool]) -> VGroup:
        rows = []
        for name, value, color, use_sign in zip(names, values, colors, signed):
            body = signed_text(float(value), 2) if use_sign else f"{float(value):.2f}"
            rows.append(Text(f"{name} = {body}", font_size=20, color=color))
        return VGroup(*rows).arrange(np.array([0.0, -1.0, 0.0]), buff=0.10)

    def construct(self) -> None:
        print(
            "D2L 9.2 step1 F={}, I={}, C_prev={}, C={}, H={}, O={}, Ctilde={}".format(
                np.array2string(STEP1["F"].ravel(), precision=3, separator=", "),
                np.array2string(STEP1["I"].ravel(), precision=3, separator=", "),
                np.array2string(CELL_0.ravel(), precision=3, separator=", "),
                np.array2string(STEP1["C"].ravel(), precision=3, separator=", "),
                np.array2string(STEP1["H"].ravel(), precision=3, separator=", "),
                np.array2string(STEP1["O"].ravel(), precision=3, separator=", "),
                np.array2string(STEP1["C_tilde"].ravel(), precision=3, separator=", "),
            )
        )
        print(
            "D2L 9.2 step2 F={}, I={}, C_prev={}, C={}, H={}, O={}, Ctilde={}".format(
                np.array2string(STEP2["F"].ravel(), precision=3, separator=", "),
                np.array2string(STEP2["I"].ravel(), precision=3, separator=", "),
                np.array2string(STEP1["C"].ravel(), precision=3, separator=", "),
                np.array2string(STEP2["C"].ravel(), precision=3, separator=", "),
                np.array2string(STEP2["H"].ravel(), precision=3, separator=", "),
                np.array2string(STEP2["O"].ravel(), precision=3, separator=", "),
                np.array2string(STEP2["C_tilde"].ravel(), precision=3, separator=", "),
            )
        )

        colors = (BLUE, YELLOW)
        kept1 = STEP1["F"] * CELL_0
        kept2 = STEP2["F"] * STEP1["C"]

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "09.2")

        cell = self.stroke_rect(self.cell_w, self.cell_h, self.cell_c, BLUE_A, 3.2)
        cell_name = Text("C", font_size=30, color=BLUE_A).move_to(
            np.array([self.cell_c[0], self.cell_c[1] + self.cell_h / 2 - 0.38, 0.0])
        )
        formula = Text("Cₜ = Fₜ ⊙ Cₜ₋₁ + Iₜ ⊙ C̃ₜ", font_size=28, color=WHITE).move_to(
            np.array([self.cell_c[0], 3.18, 0.0])
        )

        c_trackers = [ValueTracker(0.0), ValueTracker(0.0)]
        h_trackers = [ValueTracker(0.0), ValueTracker(0.0)]
        c_needles = self.make_needles(self.c_xs, c_trackers, colors)
        h_needles = self.make_needles(self.h_xs, h_trackers, colors)

        traveler_center = np.array([self.x_station, self.path_y, 0.0])
        traveler_dot = (
            Dot(traveler_center, radius=self.traveler_radius, color=YELLOW)
            .set_fill(YELLOW, opacity=1.0)
            .set_stroke(YELLOW, width=0.0, opacity=0.0)
        )
        halo_inner = Circle(radius=self.halo_inner_r).move_to(traveler_center).set_stroke(YELLOW, width=2.8, opacity=0.98)
        halo_outer = Circle(radius=self.halo_outer_r).move_to(traveler_center).set_stroke(BLUE_A, width=2.1, opacity=0.50)

        path_in = Line(
            traveler_center + np.array([0.32, 0.0, 0.0]),
            np.array([self.cell_c[0] - self.cell_w / 2, self.path_y, 0.0]),
        ).set_stroke(BLUE_E, width=2.0, opacity=0.55)
        path_out = Line(
            np.array([self.cell_c[0] + self.cell_w / 2, self.path_y, 0.0]),
            np.array([self.h_xs[0] - 0.35, self.path_y, 0.0]),
        ).set_stroke(BLUE_E, width=2.0, opacity=0.55)

        x1_column = self.make_pair_column(("x₁", "x₂"), INPUT_X1.ravel(), colors, (False, False)).move_to(
            np.array([self.x_station, (self.pocket_y[0] + self.pocket_y[1]) / 2, 0.0])
        )
        x2_column = self.make_pair_column(("x₁", "x₂"), INPUT_X2.ravel(), colors, (False, False)).move_to(
            np.array([self.x_station, (self.pocket_y[0] + self.pocket_y[1]) / 2, 0.0])
        )
        c_column_anchor = np.array([self.cell_c[0], (self.pocket_y[0] + self.pocket_y[1]) / 2, 0.0])
        c_prefix = VGroup(
            Text("C₁ = ", font_size=20, color=BLUE),
            Text("C₂ = ", font_size=20, color=YELLOW),
        )
        c_numbers = VGroup(
            DecimalNumber(0.0, num_decimal_places=2, mob_class=Text, include_sign=False, color=WHITE, font_size=20),
            DecimalNumber(0.0, num_decimal_places=2, mob_class=Text, include_sign=False, color=WHITE, font_size=20),
        )
        c_rows = VGroup(
            VGroup(c_prefix[0], c_numbers[0]).arrange(np.array([1.0, 0.0, 0.0]), buff=0.06),
            VGroup(c_prefix[1], c_numbers[1]).arrange(np.array([1.0, 0.0, 0.0]), buff=0.06),
        ).arrange(np.array([0.0, -1.0, 0.0]), buff=0.10).move_to(c_column_anchor)
        c_numbers[0].add_updater(lambda mob: mob.set_value(c_trackers[0].get_value()))
        c_numbers[1].add_updater(lambda mob: mob.set_value(c_trackers[1].get_value()))

        h_column_anchor = np.array([self.h_station, (self.pocket_y[0] + self.pocket_y[1]) / 2, 0.0])
        h_prefix = VGroup(
            Text("H₁ = ", font_size=20, color=BLUE),
            Text("H₂ = ", font_size=20, color=YELLOW),
        )
        h_numbers = VGroup(
            DecimalNumber(0.0, num_decimal_places=2, mob_class=Text, include_sign=False, color=WHITE, font_size=20),
            DecimalNumber(0.0, num_decimal_places=2, mob_class=Text, include_sign=False, color=WHITE, font_size=20),
        )
        h_rows = VGroup(
            VGroup(h_prefix[0], h_numbers[0]).arrange(np.array([1.0, 0.0, 0.0]), buff=0.06),
            VGroup(h_prefix[1], h_numbers[1]).arrange(np.array([1.0, 0.0, 0.0]), buff=0.06),
        ).arrange(np.array([0.0, -1.0, 0.0]), buff=0.10).move_to(h_column_anchor)
        h_numbers[0].add_updater(lambda mob: mob.set_value(h_trackers[0].get_value()))
        h_numbers[1].add_updater(lambda mob: mob.set_value(h_trackers[1].get_value()))

        path_x_mark = Text("xₜ", font_size=28, color=WHITE).move_to(np.array([self.x_station, -1.88, 0.0]))
        path_h_mark = Text("Hₜ", font_size=28, color=WHITE).move_to(np.array([self.h_station, -2.72, 0.0]))

        gate_left = np.array([-3.22, 1.15, 0.0])
        f1_col = self.make_pair_column(("F₁", "F₂"), STEP1["F"].ravel(), colors, (False, False)).move_to(gate_left)
        i1_col = self.make_pair_column(("I₁", "I₂"), STEP1["I"].ravel(), colors, (False, False)).move_to(
            np.array([-3.22, 1.55, 0.0])
        )
        ct1_col = self.make_pair_column(("C̃₁", "C̃₂"), STEP1["C_tilde"].ravel(), colors, (False, True)).move_to(
            np.array([-3.22, 0.55, 0.0])
        )
        o1_col = self.make_pair_column(("O₁", "O₂"), STEP1["O"].ravel(), colors, (False, False)).move_to(
            np.array([self.h_station, 1.55, 0.0])
        )
        f2_col = self.make_pair_column(("F₁", "F₂"), STEP2["F"].ravel(), colors, (False, False)).move_to(gate_left)
        i2_col = self.make_pair_column(("I₁", "I₂"), STEP2["I"].ravel(), colors, (False, False)).move_to(
            np.array([-3.22, 1.55, 0.0])
        )
        ct2_col = self.make_pair_column(("C̃₁", "C̃₂"), STEP2["C_tilde"].ravel(), colors, (False, False)).move_to(
            np.array([-3.22, 0.55, 0.0])
        )
        o2_col = self.make_pair_column(("O₁", "O₂"), STEP2["O"].ravel(), colors, (False, False)).move_to(
            np.array([self.h_station, 1.55, 0.0])
        )

        # 0.40–2.70 s: the reservoir already holds C_{t-1}.
        self.play(Create(cell), FadeIn(cell_name), run_time=1.05, rate_func=smooth)
        self.add(c_needles)
        self.play(
            c_trackers[0].animate.set_value(float(CELL_0[0, 0])),
            c_trackers[1].animate.set_value(float(CELL_0[1, 0])),
            FadeIn(c_rows),
            run_time=1.25,
            rate_func=smooth,
        )

        # 2.70–6.00 s: traveler x_t arrives; the cell is already full.
        self.play(Create(path_in), run_time=0.45, rate_func=smooth)
        self.play(
            FadeIn(traveler_dot),
            Create(halo_outer),
            Create(halo_inner),
            FadeIn(path_x_mark),
            FadeIn(x1_column),
            Flash(traveler_center, color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=0.90,
            rate_func=smooth,
        )
        self.wait(0.95)
        self.play(FadeIn(formula), run_time=0.55, rate_func=smooth)
        self.wait(0.40)

        # 6.00–10.20 s: forget gate scales the standing cell down.
        self.play(FadeIn(f1_col), run_time=0.45, rate_func=smooth)
        self.play(
            c_trackers[0].animate.set_value(float(kept1[0, 0])),
            c_trackers[1].animate.set_value(float(kept1[1, 0])),
            run_time=2.40,
            rate_func=smooth,
        )
        self.wait(0.55)

        # 10.20–15.10 s: input gate admits the candidate; C is rewritten.
        self.play(FadeOut(f1_col), run_time=0.28, rate_func=linear)
        self.play(FadeIn(i1_col), FadeIn(ct1_col), run_time=0.50, rate_func=smooth)
        self.play(
            c_trackers[0].animate.set_value(float(STEP1["C"][0, 0])),
            c_trackers[1].animate.set_value(float(STEP1["C"][1, 0])),
            run_time=2.20,
            rate_func=smooth,
        )
        self.wait(0.40)
        self.play(FadeOut(i1_col), FadeOut(ct1_col), run_time=0.28, rate_func=linear)

        # 15.10–19.20 s: output gate releases H; C stays in the reservoir.
        self.play(Create(path_out), run_time=0.40, rate_func=smooth)
        self.add(h_needles)
        self.play(
            FadeIn(o1_col),
            FadeIn(path_h_mark),
            FadeIn(h_rows),
            h_trackers[0].animate.set_value(float(STEP1["H"][0, 0])),
            h_trackers[1].animate.set_value(float(STEP1["H"][1, 0])),
            run_time=1.70,
            rate_func=smooth,
        )
        self.wait(0.70)
        self.play(FadeOut(o1_col), run_time=0.25, rate_func=linear)

        # 19.20–22.00 s: a new x_t; the same C is still standing.
        self.play(FadeOut(x1_column), run_time=0.22, rate_func=linear)
        self.play(
            FadeIn(x2_column),
            Flash(traveler_center, color=YELLOW, flash_radius=0.36, line_length=0.10),
            run_time=0.70,
            rate_func=smooth,
        )
        self.wait(0.55)

        # 22.00–26.20 s: second forget — now the yellow component drops.
        self.play(FadeIn(f2_col), run_time=0.40, rate_func=smooth)
        self.play(
            c_trackers[0].animate.set_value(float(kept2[0, 0])),
            c_trackers[1].animate.set_value(float(kept2[1, 0])),
            run_time=2.20,
            rate_func=smooth,
        )
        self.wait(0.40)
        self.play(FadeOut(f2_col), run_time=0.22, rate_func=linear)

        # 26.20–30.20 s: second write, then a new H.  C remains.
        self.play(FadeIn(i2_col), FadeIn(ct2_col), run_time=0.45, rate_func=smooth)
        self.play(
            c_trackers[0].animate.set_value(float(STEP2["C"][0, 0])),
            c_trackers[1].animate.set_value(float(STEP2["C"][1, 0])),
            run_time=1.90,
            rate_func=smooth,
        )
        self.play(FadeOut(i2_col), FadeOut(ct2_col), FadeIn(o2_col), run_time=0.40, rate_func=smooth)
        self.play(
            h_trackers[0].animate.set_value(float(STEP2["H"][0, 0])),
            h_trackers[1].animate.set_value(float(STEP2["H"][1, 0])),
            run_time=1.20,
            rate_func=smooth,
        )
        self.wait(0.35)
        self.play(FadeOut(o2_col), run_time=0.22, rate_func=linear)

        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=5.80, rate_func=linear)
