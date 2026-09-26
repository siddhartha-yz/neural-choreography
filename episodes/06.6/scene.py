"""D2L 6.6 — one haloed sample walking a tiny untrained LeNet-style net.

The arrays below are the only numerical source of truth: an 8×8 input, two
valid cross-correlations, two 2×2 average pools, flatten, then a dense map
to logits o. Weights are drawn once from a fixed RNG. Nothing is trained,
and o is not a class prediction.
"""

from __future__ import annotations

import numpy as np
from scripts.episode_intro import show_episode_intro
from manim import (
    BLACK,
    BLUE_A,
    BLUE_E,
    Create,
    FadeIn,
    FadeOut,
    Flash,
    LaggedStart,
    Line,
    RoundedRectangle,
    Scene,
    Square,
    Text,
    Transform,
    VGroup,
    ValueTracker,
    WHITE,
    YELLOW,
    config,
    linear,
    smooth,
)


config.background_color = BLACK


# ---------------------------------------------------------------------------
# Tiny real tensor walk. Not 28×28: every cell stays large enough to read.
# Shapes: 8×8 → 6×6 → 3×3 → 2×2 → 1×1 → f (1,) → o (3,).
# ---------------------------------------------------------------------------
def correlate2d(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Valid 2-D cross-correlation, matching D2L's conv layer."""
    kernel_h, kernel_w = kernel.shape
    out_h = array.shape[0] - kernel_h + 1
    out_w = array.shape[1] - kernel_w + 1
    output = np.empty((out_h, out_w), dtype=float)
    for row in range(out_h):
        for col in range(out_w):
            window = array[row : row + kernel_h, col : col + kernel_w]
            output[row, col] = float(np.sum(window * kernel))
    return output


def average_pool_2x2(array: np.ndarray) -> np.ndarray:
    """2×2 average pooling with stride 2."""
    rows, cols = array.shape
    return array.reshape(rows // 2, 2, cols // 2, 2).mean(axis=(1, 3))


RNG = np.random.default_rng(66)

INPUT_X = np.zeros((8, 8), dtype=float)
INPUT_X[1:4, 1:4] = 1.00
INPUT_X[5:7, 5:7] = 0.70

KERNEL_1 = RNG.normal(0.0, 0.55, (3, 3))
KERNEL_2 = RNG.normal(0.0, 0.55, (2, 2))
WEIGHTS = RNG.normal(0.0, 0.85, (3, 1))
BIAS = RNG.normal(0.0, 0.20, (3, 1))

CONV_1 = correlate2d(INPUT_X, KERNEL_1)
POOL_1 = average_pool_2x2(CONV_1)
CONV_2 = correlate2d(POOL_1, KERNEL_2)
POOL_2 = average_pool_2x2(CONV_2)
FLAT = POOL_2.reshape(-1, 1)
OUTPUT_O = WEIGHTS @ FLAT + BIAS


def format_signed(value: float, decimals: int) -> str:
    """ASCII digits with a true minus glyph, matching episodes 3.1 / 3.4."""
    if value < 0:
        return f"−{abs(value):.{decimals}f}"
    return f"+{value:.{decimals}f}" if decimals >= 3 else f"{value:.{decimals}f}"


class Episode066(Scene):
    """One continuous silent D2L 6.6 visualization, about 32 seconds."""

    sample_center = np.array([-0.70, -0.12, 0.0])
    kernel_center = np.array([3.85, -0.12, 0.0])

    def construct(self) -> None:
        print(
            "D2L 6.6 shapes: "
            f"X {INPUT_X.shape} → C1 {CONV_1.shape} → P1 {POOL_1.shape} → "
            f"C2 {CONV_2.shape} → P2 {POOL_2.shape} → f {tuple(FLAT.ravel().shape)} → o {tuple(OUTPUT_O.ravel().shape)}"
        )
        print("D2L 6.6 X=\n{}".format(np.array2string(INPUT_X, precision=2, separator=", ")))
        print("D2L 6.6 K1=\n{}".format(np.array2string(KERNEL_1, precision=3, separator=", ")))
        print("D2L 6.6 C1=\n{}".format(np.array2string(CONV_1, precision=3, separator=", ")))
        print("D2L 6.6 P1=\n{}".format(np.array2string(POOL_1, precision=3, separator=", ")))
        print("D2L 6.6 K2=\n{}".format(np.array2string(KERNEL_2, precision=3, separator=", ")))
        print("D2L 6.6 C2=\n{}".format(np.array2string(CONV_2, precision=3, separator=", ")))
        print("D2L 6.6 P2=\n{}".format(np.array2string(POOL_2, precision=3, separator=", ")))
        print("D2L 6.6 f={}".format(np.array2string(FLAT.ravel(), precision=3, separator=", ")))
        print("D2L 6.6 W={}".format(np.array2string(WEIGHTS.ravel(), precision=3, separator=", ")))
        print("D2L 6.6 b={}".format(np.array2string(BIAS.ravel(), precision=3, separator=", ")))
        print("D2L 6.6 o={}".format(np.array2string(OUTPUT_O.ravel(), precision=3, separator=", ")))

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "06.6")

        input_map, input_w, input_h = self.make_tensor(
            INPUT_X, cell_size=0.40, center=self.sample_center, show_values=False
        )
        shape_label = self.make_shape_label("8×8", input_w, input_h, self.sample_center)
        halo_outer, halo_inner = self.make_halo(input_w, input_h, self.sample_center)

        # Traveler is the whole 8×8 sample, not a pixel. Halo arrives with it.
        self.play(
            FadeIn(input_map),
            FadeIn(shape_label),
            FadeIn(halo_outer),
            FadeIn(halo_inner),
            Flash(self.sample_center, color=YELLOW, flash_radius=0.55, line_length=0.12),
            run_time=0.90,
            rate_func=smooth,
        )
        self.wait(1.25)

        kernel_1, kernel_1_w, kernel_1_h = self.make_tensor(
            KERNEL_1, cell_size=0.38, center=self.kernel_center, show_values=False
        )
        kernel_1_shape = self.make_shape_label("3×3", kernel_1_w, kernel_1_h, self.kernel_center)
        kernel_name = Text("K", font_size=26, color=YELLOW).next_to(
            kernel_1_shape, np.array([-1.0, 0.0, 0.0]), buff=0.16
        )
        star = Text("∗", font_size=40, color=WHITE).move_to(np.array([1.62, -0.12, 0.0]))

        # A 3×3 kernel stands next to the sample. Not an architecture poster.
        self.play(
            FadeIn(star),
            FadeIn(kernel_1),
            FadeIn(kernel_1_shape),
            FadeIn(kernel_name),
            run_time=0.70,
            rate_func=smooth,
        )
        self.wait(1.65)

        conv_1, conv_1_w, conv_1_h = self.make_tensor(
            CONV_1, cell_size=0.44, center=self.sample_center, show_values=False
        )
        conv_1_label = self.make_shape_label("6×6", conv_1_w, conv_1_h, self.sample_center)
        conv_1_halo_outer, conv_1_halo_inner = self.make_halo(conv_1_w, conv_1_h, self.sample_center)

        # Clear kernel and 8×8 before the 6×6 arrives (no stacked labels).
        self.play(
            FadeOut(star),
            FadeOut(kernel_1),
            FadeOut(kernel_1_shape),
            FadeOut(kernel_name),
            FadeOut(input_map),
            FadeOut(shape_label),
            run_time=0.42,
            rate_func=linear,
        )
        self.play(
            FadeIn(conv_1),
            FadeIn(conv_1_label),
            Transform(halo_outer, conv_1_halo_outer),
            Transform(halo_inner, conv_1_halo_inner),
            run_time=1.15,
            rate_func=smooth,
        )
        shape_label = conv_1_label
        self.wait(1.58)

        pool_windows = self.make_pool_windows(
            CONV_1.shape, cell_size=0.44, center=self.sample_center
        )
        pool_1, pool_1_w, pool_1_h = self.make_tensor(
            POOL_1, cell_size=0.86, center=self.sample_center, show_values=True, decimals=1, font_size=20
        )
        pool_1_label = self.make_shape_label("3×3", pool_1_w, pool_1_h, self.sample_center)
        pool_1_halo_outer, pool_1_halo_inner = self.make_halo(pool_1_w, pool_1_h, self.sample_center)

        self.play(
            LaggedStart(*[Create(window) for window in pool_windows], lag_ratio=0.06),
            run_time=0.80,
            rate_func=smooth,
        )
        self.wait(0.40)
        self.play(
            FadeOut(pool_windows),
            FadeOut(conv_1),
            FadeOut(shape_label),
            run_time=0.42,
            rate_func=linear,
        )
        self.play(
            FadeIn(pool_1),
            FadeIn(pool_1_label),
            Transform(halo_outer, pool_1_halo_outer),
            Transform(halo_inner, pool_1_halo_inner),
            run_time=1.05,
            rate_func=smooth,
        )
        shape_label = pool_1_label
        self.wait(0.70)

        kernel_2, kernel_2_w, kernel_2_h = self.make_tensor(
            KERNEL_2, cell_size=0.50, center=self.kernel_center, show_values=False
        )
        kernel_2_shape = self.make_shape_label("2×2", kernel_2_w, kernel_2_h, self.kernel_center)
        kernel_2_name = Text("K", font_size=26, color=YELLOW).next_to(
            kernel_2_shape, np.array([-1.0, 0.0, 0.0]), buff=0.16
        )
        star_2 = Text("∗", font_size=40, color=WHITE).move_to(np.array([1.62, -0.12, 0.0]))

        self.play(
            FadeIn(star_2),
            FadeIn(kernel_2),
            FadeIn(kernel_2_shape),
            FadeIn(kernel_2_name),
            run_time=0.65,
            rate_func=smooth,
        )
        self.wait(1.35)

        conv_2, conv_2_w, conv_2_h = self.make_tensor(
            CONV_2, cell_size=1.05, center=self.sample_center, show_values=True, decimals=2, font_size=22
        )
        conv_2_label = self.make_shape_label("2×2", conv_2_w, conv_2_h, self.sample_center)
        conv_2_halo_outer, conv_2_halo_inner = self.make_halo(conv_2_w, conv_2_h, self.sample_center)

        self.play(
            FadeOut(star_2),
            FadeOut(kernel_2),
            FadeOut(kernel_2_shape),
            FadeOut(kernel_2_name),
            FadeOut(pool_1),
            FadeOut(shape_label),
            run_time=0.42,
            rate_func=linear,
        )
        self.play(
            FadeIn(conv_2),
            FadeIn(conv_2_label),
            Transform(halo_outer, conv_2_halo_outer),
            Transform(halo_inner, conv_2_halo_inner),
            run_time=1.10,
            rate_func=smooth,
        )
        shape_label = conv_2_label
        self.wait(1.55)

        last_window = self.make_pool_windows(
            CONV_2.shape, cell_size=1.05, center=self.sample_center
        )
        pool_2, pool_2_w, pool_2_h = self.make_tensor(
            POOL_2, cell_size=1.22, center=self.sample_center, show_values=True, decimals=3, font_size=26
        )
        pool_2_label = self.make_shape_label("1×1", pool_2_w, pool_2_h, self.sample_center)
        pool_2_halo_outer, pool_2_halo_inner = self.make_halo(pool_2_w, pool_2_h, self.sample_center)

        self.play(Create(last_window), run_time=0.50, rate_func=smooth)
        self.wait(0.35)
        self.play(
            FadeOut(last_window),
            FadeOut(conv_2),
            FadeOut(shape_label),
            run_time=0.40,
            rate_func=linear,
        )
        self.play(
            FadeIn(pool_2),
            FadeIn(pool_2_label),
            Transform(halo_outer, pool_2_halo_outer),
            Transform(halo_inner, pool_2_halo_inner),
            run_time=1.00,
            rate_func=smooth,
        )
        shape_label = pool_2_label
        self.wait(0.85)

        flat_label = self.make_shape_label("f", pool_2_w, pool_2_h, self.sample_center)

        # Flatten: the 1×1 map is already one number. Swap the label only, no overlap.
        self.play(FadeOut(shape_label), run_time=0.28, rate_func=linear)
        self.play(FadeIn(flat_label), run_time=0.45, rate_func=smooth)
        shape_label = flat_label
        self.wait(1.05)

        output_map, output_w, output_h, output_labels = self.make_output_vector(
            OUTPUT_O.reshape(3, 1), cell_size=0.78, center=self.sample_center
        )
        output_shape = self.make_shape_label("o", output_w, output_h, self.sample_center)
        output_halo_outer, output_halo_inner = self.make_halo(output_w, output_h, self.sample_center)
        formula = Text("o = Wf + b", font_size=34, color=WHITE).move_to(np.array([0.15, 3.12, 0.0]))

        self.play(FadeOut(pool_2), FadeOut(shape_label), run_time=0.40, rate_func=linear)
        self.play(
            FadeIn(output_map),
            FadeIn(output_labels),
            FadeIn(formula),
            FadeIn(output_shape),
            Transform(halo_outer, output_halo_outer),
            Transform(halo_inner, output_halo_inner),
            run_time=1.35,
            rate_func=smooth,
        )
        self.wait(1.20)

        # Hold the logits. A timed no-op keeps the last encoded frame stable.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=7.40, rate_func=linear)

    def make_shape_label(self, text: str, width: float, height: float, center: np.ndarray) -> Text:
        """Park the shape on the tensor, clear of the halo stroke."""
        label = Text(text, font_size=30, color=WHITE)
        label.move_to(np.array([center[0], center[1] + height / 2 + 0.58, 0.0]))
        return label

    def make_halo(self, width: float, height: float, center: np.ndarray) -> tuple[RoundedRectangle, RoundedRectangle]:
        """Two stroke-only rounded rects. Fill stays off so the lattice never becomes a slab."""
        outer = RoundedRectangle(width=width + 0.62, height=height + 0.62, corner_radius=0.12)
        inner = RoundedRectangle(width=width + 0.32, height=height + 0.32, corner_radius=0.08)
        outer.move_to(center).set_fill(BLACK, opacity=0).set_stroke(BLUE_A, width=2.1, opacity=0.50)
        inner.move_to(center).set_fill(BLACK, opacity=0).set_stroke(YELLOW, width=2.8, opacity=0.98)
        return outer, inner

    def make_tensor(
        self,
        array: np.ndarray,
        cell_size: float,
        center: np.ndarray,
        show_values: bool,
        decimals: int = 2,
        font_size: int = 20,
    ) -> tuple[VGroup, float, float]:
        """Stroke lattice + stroke-only value squares. Never a filled mesh, never Dot.scale."""
        rows, cols = array.shape
        width = cols * cell_size
        height = rows * cell_size
        left = center[0] - width / 2
        top = center[1] + height / 2
        bottom = center[1] - height / 2

        lattice = VGroup()
        for row in range(rows + 1):
            y_value = top - row * cell_size
            lattice.add(
                Line(np.array([left, y_value, 0.0]), np.array([left + width, y_value, 0.0])).set_stroke(
                    BLUE_E, width=1.5, opacity=0.58
                )
            )
        for col in range(cols + 1):
            x_value = left + col * cell_size
            lattice.add(
                Line(np.array([x_value, bottom, 0.0]), np.array([x_value, top, 0.0])).set_stroke(
                    BLUE_E, width=1.5, opacity=0.58
                )
            )

        peak = float(np.max(np.abs(array)))
        peak = peak if peak > 1e-8 else 1.0
        values = VGroup()
        numbers = VGroup()
        for row in range(rows):
            for col in range(cols):
                value = float(array[row, col])
                cell_center = np.array(
                    [left + (col + 0.5) * cell_size, top - (row + 0.5) * cell_size, 0.0]
                )
                inset = Square(side_length=cell_size * 0.70)
                inset.move_to(cell_center)
                inset.set_fill(BLACK, opacity=0)
                magnitude = abs(value) / peak
                color = YELLOW if value >= 0 else BLUE_A
                inset.set_stroke(color, width=1.4 + 3.6 * magnitude, opacity=0.26 + 0.72 * magnitude)
                values.add(inset)
                if show_values:
                    number = Text(format_signed(value, decimals), font_size=font_size, color=WHITE)
                    number.move_to(cell_center)
                    numbers.add(number)

        group = VGroup(lattice, values, numbers) if show_values else VGroup(lattice, values)
        return group, width, height

    def make_pool_windows(self, shape: tuple[int, int], cell_size: float, center: np.ndarray) -> VGroup:
        """Stroke-only 2×2 windows sitting on the current map."""
        rows, cols = shape
        width = cols * cell_size
        height = rows * cell_size
        left = center[0] - width / 2
        top = center[1] + height / 2
        windows = VGroup()
        for row in range(rows // 2):
            for col in range(cols // 2):
                window_center = np.array(
                    [
                        left + (2 * col + 1) * cell_size,
                        top - (2 * row + 1) * cell_size,
                        0.0,
                    ]
                )
                window = Square(side_length=2 * cell_size * 0.92)
                window.move_to(window_center)
                window.set_fill(BLACK, opacity=0)
                window.set_stroke(YELLOW, width=2.6, opacity=0.95)
                windows.add(window)
        return windows

    def make_output_vector(
        self, array: np.ndarray, cell_size: float, center: np.ndarray
    ) -> tuple[VGroup, float, float, VGroup]:
        """A 3×1 logit column. Component labels sit on the right of each cell."""
        tensor, width, height = self.make_tensor(
            array, cell_size=cell_size, center=center, show_values=False
        )
        rows = array.shape[0]
        top = center[1] + height / 2
        labels = VGroup()
        names = ("o₁", "o₂", "o₃")
        for row, name in enumerate(names):
            value = float(array[row, 0])
            color = YELLOW if value >= 0 else BLUE_A
            cell_y = top - (row + 0.5) * cell_size
            label = Text(f"{name} = {format_signed(value, 3)}", font_size=24, color=color)
            label.move_to(np.array([center[0] + width / 2 + 1.28, cell_y, 0.0]))
            labels.add(label)
        return tensor, width, height, labels
