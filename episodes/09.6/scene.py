"""D2L 9.6 — variable-length input compressed to a fixed-shape state.

Every moving value comes from the encoder–decoder arrays below.  Three
source tokens are folded into one 2-vector ``c``; the decoder then emits
two tokens, one at a time, from that same state.  Untrained.  The halo
traveler is ``c``, not a source token.
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


config.background_color = BLACK
config.renderer = "cairo"


# ---------------------------------------------------------------------------
# One numerical source of truth.  Tokens are 2 × 1 column vectors.
# Encoder: h_t = tanh(W_e x_t + U_e h_{t-1} + b_e), c = h_T ∈ ℝ².
# Decoder: s_t = tanh(W_d y_{t-1} + U_d s_{t-1} + b_d), y_t = V s_t + d,
# with y_0 = 0 and s_0 = c.  T = 3, T′ = 2.  Nothing is trained.
# ---------------------------------------------------------------------------
SOURCE = (
    np.array([[1.00], [0.50]], dtype=float),
    np.array([[-0.80], [1.10]], dtype=float),
    np.array([[0.60], [-0.90]], dtype=float),
)
WEIGHT_E = np.array([[0.85, 0.15], [-0.20, 0.90]], dtype=float)
RECUR_E = np.array([[0.55, -0.25], [0.30, 0.60]], dtype=float)
BIAS_E = np.array([[0.05], [-0.08]], dtype=float)

WEIGHT_D = np.array([[0.90, -0.40], [0.20, 0.85]], dtype=float)
RECUR_D = np.array([[0.60, 0.25], [-0.45, 0.50]], dtype=float)
BIAS_D = np.array([[-0.05], [0.12]], dtype=float)
WEIGHT_OUT = np.array([[1.25, -1.10], [0.85, 0.95]], dtype=float)
BIAS_OUT = np.array([[0.15], [-0.25]], dtype=float)
BOS = np.zeros((2, 1), dtype=float)
N_OUT = 2
TOKEN_COLORS = (YELLOW, BLUE, BLUE_D)
COMPONENT_COLORS = (BLUE, YELLOW)


def encode(tokens: tuple[np.ndarray, ...]) -> list[np.ndarray]:
    """Return [h_0, h_1, …, h_T]; h_0 is zeros, h_T is the fixed-shape state."""
    hidden = np.zeros((2, 1), dtype=float)
    states = [hidden.copy()]
    for token in tokens:
        hidden = np.tanh(WEIGHT_E @ token + RECUR_E @ hidden + BIAS_E)
        states.append(hidden.copy())
    return states


def decode(state: np.ndarray, n_out: int = N_OUT) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Autoregressive decoder starting at s_0 = c, y_0 = 0."""
    hidden = state.copy()
    previous = BOS.copy()
    outputs: list[np.ndarray] = []
    decoder_states = [hidden.copy()]
    for _ in range(n_out):
        hidden = np.tanh(WEIGHT_D @ previous + RECUR_D @ hidden + BIAS_D)
        token = WEIGHT_OUT @ hidden + BIAS_OUT
        outputs.append(token.copy())
        decoder_states.append(hidden.copy())
        previous = token
    return outputs, decoder_states


ENCODER_STATES = encode(SOURCE)
CONTEXT = ENCODER_STATES[-1]
OUTPUTS, DECODER_STATES = decode(CONTEXT)


def signed_glyphs(value: float, digits: int = 2) -> str:
    """True minus glyph, matching the 3.1 remake."""
    sign = "−" if value < 0 else "+"
    return f"{sign}{abs(value):.{digits}f}"


class Episode096(Scene):
    """One continuous silent D2L 9.6 visualization, about 32 seconds."""

    needle_scale = 0.82
    baseline_y = 0.00
    source_x = (-6.05, -4.45, -2.85)
    output_x = (4.95, 6.40)
    enc_c = np.array([-1.55, 0.15, 0.0])
    enc_w, enc_h = 1.42, 2.85
    pocket_c = np.array([0.85, 0.15, 0.0])
    pocket_w, pocket_h = 2.70, 3.05
    dec_c = np.array([3.25, 0.15, 0.0])
    dec_w, dec_h = 1.42, 2.85
    # Halo sits on the state feet, not in empty air above the needles.
    halo_center = np.array([0.85, 0.46, 0.0])
    value_stack_y = -1.72

    def stroke_rect(self, width: float, height: float, center: np.ndarray, color, stroke_width: float) -> Rectangle:
        """Stroke-only rectangle; fill stays fully transparent."""
        rect = Rectangle(width=width, height=height)
        rect.move_to(center)
        rect.set_fill(BLACK, opacity=0.0)
        rect.set_stroke(color, width=stroke_width, opacity=0.94)
        return rect

    def needle_offsets(self, wide: bool = False) -> tuple[float, float]:
        return (-0.42, 0.42) if wide else (-0.20, 0.20)

    def station_axis(self, station_x: float, wide: bool = False) -> Line:
        offsets = self.needle_offsets(wide=wide)
        start = np.array([station_x + offsets[0] - 0.16, self.baseline_y, 0.0])
        end = np.array([station_x + offsets[-1] + 0.16, self.baseline_y, 0.0])
        return Line(start, end).set_stroke(BLUE_D, width=1.7, opacity=0.84)

    def make_feet(self, station_x: float, colors: tuple, wide: bool = False) -> list[Dot]:
        """Fixed-radius feet.  Never scaled; highlighting uses stroke only."""
        feet = []
        for offset, color in zip(self.needle_offsets(wide=wide), colors):
            foot = (
                Dot(
                    np.array([station_x + offset, self.baseline_y, 0.0]),
                    radius=0.055,
                    color=color,
                )
                .set_fill(color, opacity=0.94)
                .set_stroke(color, width=0.0, opacity=0.0)
            )
            feet.append(foot)
        return feet

    def make_needles(
        self,
        station_x: float,
        trackers: list[ValueTracker],
        colors: tuple,
        wide: bool = False,
    ) -> VGroup:
        offsets = self.needle_offsets(wide=wide)

        def draw_needle(index: int):
            def redraw() -> Line:
                x_position = station_x + offsets[index]
                start = np.array([x_position, self.baseline_y, 0.0])
                end = np.array(
                    [
                        x_position,
                        self.baseline_y + trackers[index].get_value() * self.needle_scale,
                        0.0,
                    ]
                )
                return Line(start, end).set_stroke(colors[index], width=8.0, opacity=0.98)

            return always_redraw(redraw)

        return VGroup(*[draw_needle(index) for index in range(len(trackers))])

    def make_value_stack(self, values: np.ndarray, colors: tuple, station_x: float) -> VGroup:
        """Compact signed numbers parked under one slot; never a shared HUD row."""
        rows = [
            Text(signed_glyphs(float(value)), font_size=18, color=color)
            for value, color in zip(values.ravel(), colors)
        ]
        column = VGroup(*rows).arrange(np.array([0.0, -1.0, 0.0]), buff=0.06)
        column.move_to(np.array([station_x, self.value_stack_y, 0.0]))
        return column

    def live_band_column(self, names: tuple[str, ...], trackers: list[ValueTracker], colors: tuple) -> VGroup:
        rows = []
        for name, tracker, color in zip(names, trackers, colors):
            prefix = Text(f"{name} = ", font_size=20, color=color)
            number = DecimalNumber(
                tracker.get_value(),
                num_decimal_places=2,
                mob_class=Text,
                include_sign=True,
                color=WHITE,
                font_size=20,
            )
            number.add_updater(lambda mob, tracker=tracker: mob.set_value(tracker.get_value()))
            rows.append(VGroup(prefix, number).arrange(np.array([1.0, 0.0, 0.0]), buff=0.06))
        return VGroup(*rows).arrange(np.array([0.0, -1.0, 0.0]), buff=0.08)

    def span(self, start: np.ndarray, end: np.ndarray, color, width: float = 2.6) -> Line:
        return Line(start, end).set_stroke(color, width=width, opacity=0.70)

    def elbow_connector(self, station_x: float, height: float, end_x: float, color, width: float = 2.2) -> VGroup:
        """Go up from a slot, then across; never cut through neighboring needles."""
        start = np.array([station_x, 0.52, 0.0])
        corner = np.array([station_x, height, 0.0])
        end = np.array([end_x, height, 0.0])
        return VGroup(
            Line(start, corner).set_stroke(color, width=width, opacity=0.62),
            Line(corner, end).set_stroke(color, width=width, opacity=0.62),
        )

    def brighten_feet(self, feet: list[Dot]):
        return [
            foot.animate.set_fill(foot.get_color(), opacity=1.0).set_stroke(foot.get_color(), width=3.2, opacity=1.0)
            for foot in feet
        ]

    def dim_feet(self, feet: list[Dot]):
        return [
            foot.animate.set_fill(foot.get_color(), opacity=0.94).set_stroke(foot.get_color(), width=0.0, opacity=0.0)
            for foot in feet
        ]

    def construct(self) -> None:
        print(
            "D2L 9.6 SOURCE x1={} x2={} x3={}".format(
                np.array2string(SOURCE[0].ravel(), precision=3, separator=", "),
                np.array2string(SOURCE[1].ravel(), precision=3, separator=", "),
                np.array2string(SOURCE[2].ravel(), precision=3, separator=", "),
            )
        )
        print(
            "D2L 9.6 W_e={} U_e={} b_e={}".format(
                np.array2string(WEIGHT_E, precision=3, separator=", "),
                np.array2string(RECUR_E, precision=3, separator=", "),
                np.array2string(BIAS_E.ravel(), precision=3, separator=", "),
            )
        )
        for index, hidden in enumerate(ENCODER_STATES):
            print(
                "D2L 9.6 h_{}={}".format(
                    index,
                    np.array2string(hidden.ravel(), precision=6, separator=", "),
                )
            )
        print(
            "D2L 9.6 c={} shape={} T={} T'={}".format(
                np.array2string(CONTEXT.ravel(), precision=6, separator=", "),
                CONTEXT.shape,
                len(SOURCE),
                N_OUT,
            )
        )
        print(
            "D2L 9.6 W_d={} U_d={} b_d={} V={} d={}".format(
                np.array2string(WEIGHT_D, precision=3, separator=", "),
                np.array2string(RECUR_D, precision=3, separator=", "),
                np.array2string(BIAS_D.ravel(), precision=3, separator=", "),
                np.array2string(WEIGHT_OUT, precision=3, separator=", "),
                np.array2string(BIAS_OUT.ravel(), precision=3, separator=", "),
            )
        )
        for index, (hidden, token) in enumerate(zip(DECODER_STATES[1:], OUTPUTS), start=1):
            print(
                "D2L 9.6 s_{}={} y_{}={}".format(
                    index,
                    np.array2string(hidden.ravel(), precision=6, separator=", "),
                    index,
                    np.array2string(token.ravel(), precision=6, separator=", "),
                )
            )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "09.6")

        source_rail = Line(
            np.array([self.source_x[0] - 0.55, -1.18, 0.0]),
            np.array([self.source_x[-1] + 0.55, -1.18, 0.0]),
        ).set_stroke(BLUE_E, width=2.0, opacity=0.55)
        output_rail = Line(
            np.array([self.output_x[0] - 0.55, -1.18, 0.0]),
            np.array([self.output_x[-1] + 0.55, -1.18, 0.0]),
        ).set_stroke(BLUE_E, width=2.0, opacity=0.55)

        source_trackers = [[ValueTracker(0.0), ValueTracker(0.0)] for _ in SOURCE]
        source_axes = VGroup(*[self.station_axis(x_position) for x_position in self.source_x])
        source_needles = VGroup(
            *[
                self.make_needles(x_position, trackers, COMPONENT_COLORS)
                for x_position, trackers in zip(self.source_x, source_trackers)
            ]
        )
        source_feet = [self.make_feet(x_position, COMPONENT_COLORS) for x_position in self.source_x]
        source_names = VGroup(
            Text("x₁", font_size=26, color=WHITE).move_to(np.array([self.source_x[0], 2.02, 0.0])),
            Text("x₂", font_size=26, color=WHITE).move_to(np.array([self.source_x[1], 2.02, 0.0])),
            Text("x₃", font_size=26, color=WHITE).move_to(np.array([self.source_x[2], 2.02, 0.0])),
        )
        source_columns = VGroup(
            *[
                self.make_value_stack(token, COMPONENT_COLORS, x_position)
                for token, x_position in zip(SOURCE, self.source_x)
            ]
        )

        enc_box = self.stroke_rect(self.enc_w, self.enc_h, self.enc_c, BLUE_D, 2.4)
        enc_label = Text("Enc", font_size=24, color=BLUE_A).move_to(
            self.enc_c + np.array([0.0, -self.enc_h / 2 + 0.28, 0.0])
        )
        pocket = self.stroke_rect(self.pocket_w, self.pocket_h, self.pocket_c, BLUE_A, 3.2)
        dec_box = self.stroke_rect(self.dec_w, self.dec_h, self.dec_c, BLUE_D, 2.4)
        dec_label = Text("Dec", font_size=24, color=BLUE_A).move_to(
            self.dec_c + np.array([0.0, -self.dec_h / 2 + 0.28, 0.0])
        )

        state_trackers = [ValueTracker(0.0), ValueTracker(0.0)]
        state_axis = self.station_axis(self.pocket_c[0], wide=True)
        state_needles = self.make_needles(self.pocket_c[0], state_trackers, COMPONENT_COLORS, wide=True)
        state_feet = self.make_feet(self.pocket_c[0], COMPONENT_COLORS, wide=True)
        traveler = (
            Dot(self.halo_center, radius=0.078, color=YELLOW)
            .set_fill(YELLOW, opacity=1.0)
            .set_stroke(YELLOW, width=0.0, opacity=0.0)
        )
        halo_inner = Circle(radius=0.178).move_to(self.halo_center).set_stroke(YELLOW, width=2.8, opacity=0.98)
        halo_outer = Circle(radius=0.278).move_to(self.halo_center).set_stroke(BLUE_A, width=2.1, opacity=0.50)
        state_name = Text("c", font_size=26, color=YELLOW_A).move_to(np.array([self.pocket_c[0], 2.02, 0.0]))
        state_column = self.live_band_column(("c₁", "c₂"), state_trackers, COMPONENT_COLORS).move_to(
            np.array([self.pocket_c[0], self.value_stack_y, 0.0])
        )

        output_trackers = [[ValueTracker(0.0), ValueTracker(0.0)] for _ in range(N_OUT)]
        output_axes = VGroup(*[self.station_axis(x_position) for x_position in self.output_x])
        output_needles = VGroup(
            *[
                self.make_needles(x_position, trackers, COMPONENT_COLORS)
                for x_position, trackers in zip(self.output_x, output_trackers)
            ]
        )
        output_feet = [self.make_feet(x_position, COMPONENT_COLORS) for x_position in self.output_x]
        output_names = VGroup(
            Text("ŷ₁", font_size=26, color=WHITE).move_to(np.array([self.output_x[0], 2.02, 0.0])),
            Text("ŷ₂", font_size=26, color=WHITE).move_to(np.array([self.output_x[1], 2.02, 0.0])),
        )
        output_columns = VGroup(
            *[
                self.make_value_stack(token, COMPONENT_COLORS, x_position)
                for token, x_position in zip(OUTPUTS, self.output_x)
            ]
        )

        formula = Text("c = Enc(x₁:T) ∈ ℝ²     Ŷ = Dec(c)", font_size=30, color=WHITE).move_to(
            np.array([0.35, 3.18, 0.0])
        )

        enc_left_x = self.enc_c[0] - self.enc_w / 2
        enc_right_x = self.enc_c[0] + self.enc_w / 2
        pocket_left_x = self.pocket_c[0] - self.pocket_w / 2
        pocket_right_x = self.pocket_c[0] + self.pocket_w / 2
        dec_left_x = self.dec_c[0] - self.dec_w / 2
        dec_right_x = self.dec_c[0] + self.dec_w / 2
        encode_heights = (1.50, 1.32, 1.14)
        encode_bridges = [
            self.elbow_connector(x_position, height, enc_left_x, TOKEN_COLORS[index])
            for index, (x_position, height) in enumerate(zip(self.source_x, encode_heights))
        ]
        enc_to_pocket = self.span(
            np.array([enc_right_x, self.baseline_y, 0.0]),
            np.array([pocket_left_x, self.baseline_y, 0.0]),
            YELLOW_A,
        )
        pocket_to_dec = self.span(
            np.array([pocket_right_x, self.baseline_y, 0.0]),
            np.array([dec_left_x, self.baseline_y, 0.0]),
            YELLOW_A,
        )
        decode_heights = (1.42, 1.22)
        decode_bridges = [
            self.elbow_connector(x_position, height, dec_right_x, YELLOW)
            for x_position, height in zip(self.output_x, decode_heights)
        ]

        # 0.40–2.50 s: names and the three source vectors in the same beat.
        self.add(source_needles, *[foot for pair in source_feet for foot in pair])
        self.play(
            FadeIn(source_rail),
            FadeIn(source_axes),
            LaggedStart(*[FadeIn(name) for name in source_names], lag_ratio=0.12),
            LaggedStart(
                *[trackers[0].animate.set_value(float(token[0, 0])) for token, trackers in zip(SOURCE, source_trackers)],
                lag_ratio=0.16,
            ),
            LaggedStart(
                *[trackers[1].animate.set_value(float(token[1, 0])) for token, trackers in zip(SOURCE, source_trackers)],
                lag_ratio=0.16,
            ),
            LaggedStart(*[FadeIn(column) for column in source_columns], lag_ratio=0.16),
            run_time=1.70,
            rate_func=smooth,
        )
        self.wait(0.40)

        # 2.50–4.40 s: encoder box and the still-empty fixed-shape pocket.
        self.play(
            Create(enc_box),
            FadeIn(enc_label),
            Create(pocket),
            Create(enc_to_pocket),
            run_time=1.50,
            rate_func=smooth,
        )
        self.wait(0.40)

        # 4.40–10.70 s: three real encoder steps fold into the same ℝ² pocket.
        self.add(state_axis, state_needles, *state_feet)
        for step_index, hidden in enumerate(ENCODER_STATES[1:]):
            self.play(
                *self.brighten_feet(source_feet[step_index]),
                Create(encode_bridges[step_index]),
                run_time=0.50,
                rate_func=smooth,
            )
            appear = []
            if step_index == 0:
                appear = [
                    FadeIn(traveler),
                    FadeIn(halo_inner),
                    FadeIn(halo_outer),
                    FadeIn(state_name),
                    FadeIn(state_column),
                ]
            extras = []
            if step_index == 2:
                extras.append(Flash(self.halo_center, color=YELLOW, flash_radius=0.42, line_length=0.11))
            self.play(
                state_trackers[0].animate.set_value(float(hidden[0, 0])),
                state_trackers[1].animate.set_value(float(hidden[1, 0])),
                *appear,
                *extras,
                run_time=1.20,
                rate_func=smooth,
            )
            self.play(*self.dim_feet(source_feet[step_index]), run_time=0.40, rate_func=smooth)

        # 10.70–12.80 s: formula and shape, once.
        self.play(FadeIn(formula), run_time=0.70, rate_func=smooth)
        self.wait(1.40)

        # 12.80–14.90 s: decoder box and two empty output slots (T′ ≠ T).
        self.play(
            Create(dec_box),
            FadeIn(dec_label),
            Create(pocket_to_dec),
            FadeIn(output_rail),
            FadeIn(output_axes),
            LaggedStart(*[FadeIn(name) for name in output_names], lag_ratio=0.18),
            run_time=1.70,
            rate_func=smooth,
        )
        self.wait(0.40)

        # 14.90–19.10 s: decoder emits one token at a time from the same c.
        self.add(output_needles, *[foot for pair in output_feet for foot in pair])
        for out_index, token in enumerate(OUTPUTS):
            self.play(Create(decode_bridges[out_index]), run_time=0.45, rate_func=smooth)
            self.play(
                output_trackers[out_index][0].animate.set_value(float(token[0, 0])),
                output_trackers[out_index][1].animate.set_value(float(token[1, 0])),
                FadeIn(output_columns[out_index]),
                Flash(
                    np.array([self.output_x[out_index], self.baseline_y, 0.0]),
                    color=YELLOW,
                    flash_radius=0.34,
                    line_length=0.09,
                ),
                run_time=1.25,
                rate_func=smooth,
            )
            self.wait(0.40)

        # Hold encoder state and both decoder outputs through the last frame.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=12.90, rate_func=linear)
