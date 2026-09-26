"""D2L 7.2 — the same VGG block twice: conv keeps H,W, pool halves, channels double.

Every on-screen map is a real untrained numpy forward pass: 3×3 conv with
pad 1 and ReLU, then 2×2 max-pool stride 2. Not VGG-11, not a 13-layer tower.
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
    YELLOW_A,
    config,
    linear,
    smooth,
)


config.background_color = BLACK


def conv2d_pad1_relu(inputs: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Same-size 3×3 cross-correlation (pad 1) plus ReLU. ``inputs`` is H×W×Cin."""
    height, width, _in_channels = inputs.shape
    out_channels = weight.shape[0]
    padded = np.pad(inputs, ((1, 1), (1, 1), (0, 0)))
    output = np.zeros((height, width, out_channels), dtype=float)
    for out_index in range(out_channels):
        for row in range(height):
            for col in range(width):
                window = padded[row : row + 3, col : col + 3, :]
                output[row, col, out_index] = float(np.sum(window * weight[out_index]))
    return np.maximum(output, 0.0)


def max_pool2x2(inputs: np.ndarray) -> np.ndarray:
    """2×2 max-pool, stride 2, matching D2L's VGG block."""
    height, width, channels = inputs.shape
    output = np.zeros((height // 2, width // 2, channels), dtype=float)
    for row in range(height // 2):
        for col in range(width // 2):
            output[row, col, :] = np.max(
                inputs[2 * row : 2 * row + 2, 2 * col : 2 * col + 2, :], axis=(0, 1)
            )
    return output


RNG = np.random.default_rng(72)
INPUT_X = np.zeros((8, 8, 1), dtype=float)
INPUT_X[1:4, 1:4, 0] = 1.00
INPUT_X[5:7, 5:7, 0] = 0.60
WEIGHT_1 = RNG.normal(0.20, 0.28, (2, 3, 3, 1))
WEIGHT_2 = RNG.normal(0.12, 0.22, (4, 3, 3, 2))
CONV_1 = conv2d_pad1_relu(INPUT_X, WEIGHT_1)
POOL_1 = max_pool2x2(CONV_1)
CONV_2 = conv2d_pad1_relu(POOL_1, WEIGHT_2)
POOL_2 = max_pool2x2(CONV_2)


def format_number(value: float, decimals: int) -> str:
    rounded = round(float(value), decimals)
    if rounded < 0:
        return f"−{abs(rounded):.{decimals}f}"
    return f"{rounded:.{decimals}f}"


class Episode072(Scene):
    """A ~32-second silent D2L 7.2 visualization: two identical conv→pool blocks."""

    sample_center = np.array([-0.55, -0.18, 0.0])
    cell_size = 0.46
    channel_offset = np.array([0.34, -0.28, 0.0])

    def construct(self) -> None:
        print("D2L 7.2 X {}".format(INPUT_X.shape))
        print("D2L 7.2 after conv1 {}".format(CONV_1.shape))
        print("D2L 7.2 after block1 {}".format(POOL_1.shape))
        print("D2L 7.2 after conv2 {}".format(CONV_2.shape))
        print("D2L 7.2 after block2 {}".format(POOL_2.shape))
        print("D2L 7.2 X=\n{}".format(np.array2string(INPUT_X[:, :, 0], precision=2, separator=", ")))
        print("D2L 7.2 P1 ch0=\n{}".format(np.array2string(POOL_1[:, :, 0], precision=3, separator=", ")))
        print("D2L 7.2 P2=\n{}".format(np.array2string(POOL_2, precision=3, separator=", ")))

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "07.2")

        input_stack, input_w, input_h = self.make_channel_stack(INPUT_X, self.sample_center)
        shape_label = self.make_shape_label("8×8", input_w, input_h, self.sample_center, channels=1)
        channel_label = self.make_channel_label("c = 1", input_w, input_h, self.sample_center, channels=1)
        halo_outer, halo_inner = self.make_halo(input_w, input_h, self.sample_center, channels=1)

        # Traveler is the whole sample, not a pixel.
        self.play(
            FadeIn(input_stack),
            FadeIn(shape_label),
            FadeIn(channel_label),
            FadeIn(halo_outer),
            FadeIn(halo_inner),
            Flash(self.sample_center, color=YELLOW, flash_radius=0.55, line_length=0.12),
            run_time=0.90,
            rate_func=smooth,
        )
        self.wait(1.50)

        # Block 1, conv: 3×3 window on the bright patch; H,W stay 8×8, c doubles.
        conv_window = self.make_conv_window(INPUT_X.shape[0], self.cell_size, self.sample_center, row=1, col=1)
        self.play(Create(conv_window), run_time=0.60, rate_func=smooth)
        self.wait(0.40)

        conv_1_stack, conv_1_w, conv_1_h = self.make_channel_stack(CONV_1, self.sample_center)
        conv_1_channel = self.make_channel_label("c = 2", conv_1_w, conv_1_h, self.sample_center, channels=2)
        conv_1_halo_outer, conv_1_halo_inner = self.make_halo(conv_1_w, conv_1_h, self.sample_center, channels=2)

        self.play(FadeOut(conv_window), FadeOut(input_stack), FadeOut(channel_label), run_time=0.40, rate_func=linear)
        self.play(
            FadeIn(conv_1_stack),
            FadeIn(conv_1_channel),
            Transform(halo_outer, conv_1_halo_outer),
            Transform(halo_inner, conv_1_halo_inner),
            run_time=1.20,
            rate_func=smooth,
        )
        self.wait(1.10)

        # Block 1, pool: 2×2 windows, then the map halves. c stays 2.
        pool_windows = self.make_pool_windows(CONV_1.shape[:2], self.cell_size, self.sample_center)
        self.play(
            LaggedStart(*[Create(window) for window in pool_windows], lag_ratio=0.05),
            run_time=0.85,
            rate_func=smooth,
        )
        self.wait(0.45)

        pool_1_stack, pool_1_w, pool_1_h = self.make_channel_stack(POOL_1, self.sample_center)
        pool_1_shape = self.make_shape_label("4×4", pool_1_w, pool_1_h, self.sample_center, channels=2)
        pool_1_channel = self.make_channel_label("c = 2", pool_1_w, pool_1_h, self.sample_center, channels=2)
        pool_1_halo_outer, pool_1_halo_inner = self.make_halo(pool_1_w, pool_1_h, self.sample_center, channels=2)

        self.play(
            FadeOut(pool_windows),
            FadeOut(conv_1_stack),
            FadeOut(shape_label),
            FadeOut(conv_1_channel),
            run_time=0.40,
            rate_func=linear,
        )
        self.play(
            FadeIn(pool_1_stack),
            FadeIn(pool_1_shape),
            FadeIn(pool_1_channel),
            Transform(halo_outer, pool_1_halo_outer),
            Transform(halo_inner, pool_1_halo_inner),
            run_time=1.15,
            rate_func=smooth,
        )
        shape_label = pool_1_shape
        self.wait(1.40)

        # Block 2 is the same motion: 3×3 keeps 4×4, channels 2 → 4.
        conv_window_2 = self.make_conv_window(POOL_1.shape[0], self.cell_size, self.sample_center, row=0, col=0)
        self.play(Create(conv_window_2), run_time=0.60, rate_func=smooth)
        self.wait(0.40)

        conv_2_stack, conv_2_w, conv_2_h = self.make_channel_stack(CONV_2, self.sample_center)
        conv_2_channel = self.make_channel_label("c = 4", conv_2_w, conv_2_h, self.sample_center, channels=4)
        conv_2_halo_outer, conv_2_halo_inner = self.make_halo(conv_2_w, conv_2_h, self.sample_center, channels=4)

        self.play(
            FadeOut(conv_window_2),
            FadeOut(pool_1_stack),
            FadeOut(pool_1_channel),
            run_time=0.40,
            rate_func=linear,
        )
        self.play(
            FadeIn(conv_2_stack),
            FadeIn(conv_2_channel),
            Transform(halo_outer, conv_2_halo_outer),
            Transform(halo_inner, conv_2_halo_inner),
            run_time=1.25,
            rate_func=smooth,
        )
        self.wait(1.00)

        pool_windows_2 = self.make_pool_windows(CONV_2.shape[:2], self.cell_size, self.sample_center)
        self.play(
            LaggedStart(*[Create(window) for window in pool_windows_2], lag_ratio=0.08),
            run_time=0.75,
            rate_func=smooth,
        )
        self.wait(0.40)

        pool_2_stack, pool_2_w, pool_2_h = self.make_channel_stack(
            POOL_2, self.sample_center, show_values=True, decimals=1, font_size=18
        )
        pool_2_shape = self.make_shape_label("2×2", pool_2_w, pool_2_h, self.sample_center, channels=4)
        pool_2_channel = self.make_channel_label("c = 4", pool_2_w, pool_2_h, self.sample_center, channels=4)
        pool_2_halo_outer, pool_2_halo_inner = self.make_halo(pool_2_w, pool_2_h, self.sample_center, channels=4)

        self.play(
            FadeOut(pool_windows_2),
            FadeOut(conv_2_stack),
            FadeOut(shape_label),
            FadeOut(conv_2_channel),
            run_time=0.40,
            rate_func=linear,
        )
        self.play(
            FadeIn(pool_2_stack),
            FadeIn(pool_2_shape),
            FadeIn(pool_2_channel),
            Transform(halo_outer, pool_2_halo_outer),
            Transform(halo_inner, pool_2_halo_inner),
            run_time=1.15,
            rate_func=smooth,
        )
        self.wait(0.80)

        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=12.20, rate_func=linear)

    def stack_shift(self, channels: int) -> np.ndarray:
        return self.channel_offset * max(channels - 1, 0)

    def make_shape_label(
        self, text: str, width: float, height: float, center: np.ndarray, channels: int
    ) -> Text:
        """H×W sits on top of the front map, clear of the halo."""
        shift = self.stack_shift(channels)
        label = Text(text, font_size=30, color=WHITE)
        label.move_to(
            np.array([center[0] + shift[0] / 2.0, center[1] + height / 2.0 + 0.58, 0.0])
        )
        return label

    def make_channel_label(
        self, text: str, width: float, height: float, center: np.ndarray, channels: int
    ) -> Text:
        """c = … is written on the tensor stack, to the right of the front map."""
        shift = self.stack_shift(channels)
        label = Text(text, font_size=28, color=YELLOW_A)
        label.move_to(
            np.array(
                [
                    center[0] + width / 2.0 + shift[0] + 0.92,
                    center[1] + shift[1] / 2.0,
                    0.0,
                ]
            )
        )
        return label

    def make_halo(
        self, width: float, height: float, center: np.ndarray, channels: int
    ) -> tuple[RoundedRectangle, RoundedRectangle]:
        """Stroke-only rounded rects around the sample. Fill stays off."""
        shift = self.stack_shift(channels)
        halo_center = center + shift / 2.0
        extra_w = abs(shift[0])
        extra_h = abs(shift[1])
        outer = RoundedRectangle(
            width=width + extra_w + 0.62, height=height + extra_h + 0.62, corner_radius=0.12
        )
        inner = RoundedRectangle(
            width=width + extra_w + 0.32, height=height + extra_h + 0.32, corner_radius=0.08
        )
        outer.move_to(halo_center).set_fill(BLACK, opacity=0).set_stroke(BLUE_A, width=2.1, opacity=0.50)
        inner.move_to(halo_center).set_fill(BLACK, opacity=0).set_stroke(YELLOW, width=2.8, opacity=0.98)
        return outer, inner

    def make_tensor(
        self,
        array: np.ndarray,
        cell_size: float,
        center: np.ndarray,
        show_values: bool,
        decimals: int = 2,
        font_size: int = 18,
        stroke_scale: float = 1.0,
    ) -> tuple[VGroup, float, float]:
        """Stroke lattice + stroke-only value squares. Never a filled mesh, never Dot.scale."""
        rows, cols = array.shape
        width = cols * cell_size
        height = rows * cell_size
        left = center[0] - width / 2
        top = center[1] + height / 2
        bottom = center[1] - height / 2
        lattice_opacity = 0.58 * stroke_scale
        lattice = VGroup()
        for row in range(rows + 1):
            y_value = top - row * cell_size
            lattice.add(
                Line(np.array([left, y_value, 0.0]), np.array([left + width, y_value, 0.0])).set_stroke(
                    BLUE_E, width=1.5, opacity=lattice_opacity
                )
            )
        for col in range(cols + 1):
            x_value = left + col * cell_size
            lattice.add(
                Line(np.array([x_value, bottom, 0.0]), np.array([x_value, top, 0.0])).set_stroke(
                    BLUE_E, width=1.5, opacity=lattice_opacity
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
                inset.set_stroke(
                    color,
                    width=(1.4 + 3.6 * magnitude) * stroke_scale,
                    opacity=(0.26 + 0.72 * magnitude) * stroke_scale,
                )
                values.add(inset)
                if show_values:
                    number = Text(format_number(value, decimals), font_size=font_size, color=WHITE)
                    number.move_to(cell_center)
                    numbers.add(number)

        group = VGroup(lattice, values, numbers) if show_values else VGroup(lattice, values)
        return group, width, height

    def make_channel_stack(
        self,
        array: np.ndarray,
        center: np.ndarray,
        show_values: bool = False,
        decimals: int = 1,
        font_size: int = 18,
    ) -> tuple[VGroup, float, float]:
        """Offset stroke planes, back to front. Only the front plane may carry digits."""
        _height, _width, channels = array.shape
        planes = VGroup()
        front_width = 0.0
        front_height = 0.0
        for channel_index in range(channels - 1, -1, -1):
            plane_center = center + self.channel_offset * channel_index
            is_front = channel_index == 0
            tensor, width, height = self.make_tensor(
                array[:, :, channel_index],
                self.cell_size,
                plane_center,
                show_values=show_values and is_front,
                decimals=decimals,
                font_size=font_size,
                stroke_scale=1.0 if is_front else 0.40,
            )
            planes.add(tensor)
            if is_front:
                front_width, front_height = width, height
        return planes, front_width, front_height

    def make_conv_window(
        self, map_size: int, cell_size: float, center: np.ndarray, row: int, col: int
    ) -> Square:
        """One 3×3 window on the map: conv keeps H,W (pad 1), unlike 6.6's valid shrink."""
        width = map_size * cell_size
        height = map_size * cell_size
        left = center[0] - width / 2
        top = center[1] + height / 2
        window_center = np.array(
            [
                left + (col + 1.5) * cell_size,
                top - (row + 1.5) * cell_size,
                0.0,
            ]
        )
        window = Square(side_length=3 * cell_size + 0.18)
        window.move_to(window_center)
        window.set_fill(BLACK, opacity=0)
        window.set_stroke(YELLOW, width=4.0, opacity=0.98)
        return window

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
