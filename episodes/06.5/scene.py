"""D2L 6.5 — 2×2 max pooling then average pooling on a tiny real map.

Every number on screen comes from the arrays below. There are no learned
weights: pooling is a deterministic window operator. Max keeps one cell;
average uses every cell in the same window. The 4×4 → 2×2 drop is stride 2.
"""

from __future__ import annotations

import numpy as np
from scripts.episode_intro import show_episode_intro
from manim import (
    AnimationGroup,
    BLACK,
    BLUE_A,
    BLUE_D,
    BLUE_E,
    Circle,
    Create,
    FadeIn,
    FadeOut,
    Flash,
    LaggedStart,
    Line,
    Scene,
    Square,
    Text,
    ValueTracker,
    WHITE,
    YELLOW,
    config,
    linear,
    smooth,
)


config.background_color = BLACK


# Tiny 4×4 map. Each 2×2 window has a unique max, and that max is not equal
# to the window mean, so the two operators cannot be confused on sight.
INPUT = np.array(
    [
        [1.0, 9.0, 2.0, 0.0],
        [1.0, 1.0, 8.0, 6.0],
        [0.0, 2.0, 5.0, 7.0],
        [8.0, 2.0, 1.0, 3.0],
    ],
    dtype=float,
)
POOL_SIZE = (2, 2)
STRIDE = 2


def format_number(value: float) -> str:
    """Integers stay integers; non-integers keep one decimal."""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"


def pooling_records(map_values: np.ndarray) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Slide a 2×2 window with stride 2. Numpy is the only arithmetic."""
    height, width = map_values.shape
    pool_h, pool_w = POOL_SIZE
    out_h = (height - pool_h) // STRIDE + 1
    out_w = (width - pool_w) // STRIDE + 1
    max_out = np.zeros((out_h, out_w), dtype=float)
    avg_out = np.zeros((out_h, out_w), dtype=float)
    records: list[dict] = []
    for out_row in range(out_h):
        for out_col in range(out_w):
            row = out_row * STRIDE
            col = out_col * STRIDE
            patch = map_values[row : row + pool_h, col : col + pool_w]
            max_out[out_row, out_col] = float(np.max(patch))
            avg_out[out_row, out_col] = float(np.mean(patch))
            local_row, local_col = np.unravel_index(int(np.argmax(patch)), patch.shape)
            records.append(
                {
                    "out_row": out_row,
                    "out_col": out_col,
                    "origin": (row, col),
                    "patch": np.array(patch, copy=True),
                    "max": float(max_out[out_row, out_col]),
                    "avg": float(avg_out[out_row, out_col]),
                    "winner": (row + int(local_row), col + int(local_col)),
                }
            )
    return records, max_out, avg_out


WINDOWS, MAX_OUT, AVG_OUT = pooling_records(INPUT)


class Episode065(Scene):
    """A ~30-second continuous, silent 2×2 pooling visualization."""

    cell = 1.00
    input_origin = np.array([-5.10, 1.62, 0.0])
    max_origin = np.array([2.92, 1.88, 0.0])
    avg_origin = np.array([2.92, -0.92, 0.0])

    def input_center(self, row: int, col: int) -> np.ndarray:
        return self.input_origin + np.array([col * self.cell, -row * self.cell, 0.0])

    def output_center(self, origin: np.ndarray, row: int, col: int) -> np.ndarray:
        return origin + np.array([col * self.cell, -row * self.cell, 0.0])

    def window_center(self, origin: tuple[int, int]) -> np.ndarray:
        row, col = origin
        top_left = self.input_center(row, col)
        bottom_right = self.input_center(row + 1, col + 1)
        return (top_left + bottom_right) / 2.0

    def make_stroke_square(self, center: np.ndarray, color=BLUE_D, width: float = 2.0, opacity: float = 0.90) -> Square:
        square = Square(side_length=self.cell)
        square.move_to(center)
        square.set_fill(opacity=0.0)
        square.set_stroke(color, width=width, opacity=opacity)
        return square

    def window_square(self, origin: tuple[int, int]) -> Square:
        square = Square(side_length=2.0 * self.cell + 0.14)
        square.move_to(self.window_center(origin))
        square.set_fill(opacity=0.0)
        square.set_stroke(YELLOW, width=3.6, opacity=0.96)
        return square

    def paint_input(self, origin: tuple[int, int], winner: tuple[int, int], mode: str) -> AnimationGroup:
        """Dim cells outside the window; max keeps only the winner bright."""
        row0, col0 = origin
        inside = {
            (row0, col0),
            (row0, col0 + 1),
            (row0 + 1, col0),
            (row0 + 1, col0 + 1),
        }
        animations = []
        for row in range(4):
            for col in range(4):
                square = self.input_squares[row][col]
                label = self.input_labels[row][col]
                if (row, col) == winner:
                    animations.append(square.animate.set_stroke(YELLOW, width=3.2, opacity=1.0))
                    animations.append(label.animate.set_color(YELLOW).set_opacity(1.0))
                elif (row, col) in inside:
                    if mode == "max":
                        animations.append(square.animate.set_stroke(BLUE_E, width=1.3, opacity=0.28))
                        animations.append(label.animate.set_color(WHITE).set_opacity(0.26))
                    else:
                        animations.append(square.animate.set_stroke(BLUE_A, width=2.3, opacity=0.95))
                        animations.append(label.animate.set_color(WHITE).set_opacity(1.0))
                else:
                    animations.append(square.animate.set_stroke(BLUE_E, width=1.2, opacity=0.38))
                    animations.append(label.animate.set_color(WHITE).set_opacity(0.38))
        return AnimationGroup(*animations)

    def construct(self) -> None:
        print("D2L 6.5 2×2 stride-2 pooling")
        print("X =")
        print(INPUT)
        for record in WINDOWS:
            row, col = record["origin"]
            print(
                "window @({}, {}) =\n{}  max={}  avg={}  winner={}".format(
                    row,
                    col,
                    record["patch"],
                    format_number(record["max"]),
                    format_number(record["avg"]),
                    record["winner"],
                )
            )
        print("max =")
        print(MAX_OUT)
        print("avg =")
        print(AVG_OUT)

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "06.5")

        self.input_squares = [
            [self.make_stroke_square(self.input_center(row, col)) for col in range(4)]
            for row in range(4)
        ]
        self.input_labels = [
            [
                Text(format_number(INPUT[row, col]), font_size=30, color=WHITE).move_to(
                    self.input_center(row, col)
                )
                for col in range(4)
            ]
            for row in range(4)
        ]
        x_label = Text("X", font_size=34, color=BLUE_A).move_to(np.array([-6.32, 0.12, 0.0]))

        max_squares = [
            [
                self.make_stroke_square(self.output_center(self.max_origin, row, col), color=BLUE_D)
                for col in range(2)
            ]
            for row in range(2)
        ]
        avg_squares = [
            [
                self.make_stroke_square(self.output_center(self.avg_origin, row, col), color=BLUE_D)
                for col in range(2)
            ]
            for row in range(2)
        ]
        max_values = [
            [
                Text(
                    format_number(MAX_OUT[row, col]),
                    font_size=30,
                    color=YELLOW,
                ).move_to(self.output_center(self.max_origin, row, col) + np.array([0.0, 0.12, 0.0]))
                for col in range(2)
            ]
            for row in range(2)
        ]
        avg_values = [
            [
                Text(
                    format_number(AVG_OUT[row, col]),
                    font_size=30,
                    color=WHITE,
                ).move_to(self.output_center(self.avg_origin, row, col) + np.array([0.0, 0.12, 0.0]))
                for col in range(2)
            ]
            for row in range(2)
        ]
        max_tags = [
            [
                Text("max", font_size=18, color=YELLOW).move_to(
                    self.output_center(self.max_origin, row, col) + np.array([0.0, -0.28, 0.0])
                )
                for col in range(2)
            ]
            for row in range(2)
        ]
        avg_tags = [
            [
                Text("avg", font_size=18, color=BLUE_A).move_to(
                    self.output_center(self.avg_origin, row, col) + np.array([0.0, -0.28, 0.0])
                )
                for col in range(2)
            ]
            for row in range(2)
        ]

        # 0.40–1.10 s: the 4×4 map, numbers sitting on the squares.
        self.play(
            LaggedStart(
                *[FadeIn(self.input_squares[row][col]) for row in range(4) for col in range(4)],
                lag_ratio=0.03,
            ),
            LaggedStart(
                *[FadeIn(self.input_labels[row][col]) for row in range(4) for col in range(4)],
                lag_ratio=0.03,
            ),
            FadeIn(x_label),
            run_time=0.70,
            rate_func=linear,
        )
        self.wait(0.45)

        # 1.55–2.20 s: empty 2×2 frames — same cell size, so the drop is visible.
        self.play(
            LaggedStart(
                *[FadeIn(max_squares[row][col]) for row in range(2) for col in range(2)],
                lag_ratio=0.08,
            ),
            LaggedStart(
                *[FadeIn(avg_squares[row][col]) for row in range(2) for col in range(2)],
                lag_ratio=0.08,
            ),
            run_time=0.65,
            rate_func=smooth,
        )

        first = WINDOWS[0]
        window = self.window_square(first["origin"])
        window_size = Text("2×2", font_size=22, color=YELLOW).next_to(window, np.array([0.0, 1.0, 0.0]), buff=0.10)
        # Same traveler rings as 3.1/3.4: around the winning digit, inside the cell.
        halo_outer = (
            Circle(radius=0.278).move_to(self.input_center(*first["winner"])).set_stroke(BLUE_A, width=2.1, opacity=0.50)
        )
        halo_inner = (
            Circle(radius=0.178).move_to(self.input_center(*first["winner"])).set_stroke(YELLOW, width=2.8, opacity=0.98)
        )

        # 2.20–2.90 s: the first window lands on the top-left 2×2.
        self.play(Create(window), FadeIn(window_size), run_time=0.55, rate_func=smooth)
        self.wait(0.15)

        def write_output(record: dict, mode: str) -> None:
            out_row, out_col = record["out_row"], record["out_col"]
            grid_right_x = self.input_origin[0] + 3.5 * self.cell + 0.18
            cell_left = np.array([-0.52, 0.0, 0.0])
            if mode == "max":
                target = self.output_center(self.max_origin, out_row, out_col) + cell_left
                value = max_values[out_row][out_col]
                tag = max_tags[out_row][out_col]
                start = np.array([grid_right_x, self.input_center(*record["winner"])[1], 0.0])
            else:
                target = self.output_center(self.avg_origin, out_row, out_col) + cell_left
                value = avg_values[out_row][out_col]
                tag = avg_tags[out_row][out_col]
                start = np.array([grid_right_x, self.window_center(record["origin"])[1], 0.0])
            connector = Line(start, target).set_stroke(YELLOW, width=2.2, opacity=0.80)
            self.play(
                Create(connector),
                FadeIn(value),
                FadeIn(tag),
                run_time=0.52,
                rate_func=smooth,
            )
            self.play(FadeOut(connector), run_time=0.18, rate_func=linear)

        # 2.90–4.20 s: winner stays, the other three dim, first max cell is written.
        self.play(
            self.paint_input(first["origin"], first["winner"], "max"),
            Create(halo_outer),
            Create(halo_inner),
            Flash(self.input_center(*first["winner"]), color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=0.75,
            rate_func=smooth,
        )
        self.play(FadeOut(window_size), run_time=0.16, rate_func=linear)
        write_output(first, "max")

        for record in WINDOWS[1:]:
            self.play(
                window.animate.move_to(self.window_center(record["origin"])),
                halo_outer.animate.move_to(self.input_center(*record["winner"])),
                halo_inner.animate.move_to(self.input_center(*record["winner"])),
                run_time=0.48,
                rate_func=smooth,
            )
            self.play(
                self.paint_input(record["origin"], record["winner"], "max"),
                Flash(self.input_center(*record["winner"]), color=YELLOW, flash_radius=0.34, line_length=0.09),
                run_time=0.55,
                rate_func=smooth,
            )
            write_output(record, "max")

        self.wait(0.55)

        # Second beat: same windows, every cell in the window counts, output is the mean.
        start = WINDOWS[0]
        self.play(
            window.animate.move_to(self.window_center(start["origin"])),
            halo_outer.animate.move_to(self.input_center(*start["winner"])),
            halo_inner.animate.move_to(self.input_center(*start["winner"])),
            run_time=0.55,
            rate_func=smooth,
        )
        for record in WINDOWS:
            if record is not start:
                self.play(
                    window.animate.move_to(self.window_center(record["origin"])),
                    halo_outer.animate.move_to(self.input_center(*record["winner"])),
                    halo_inner.animate.move_to(self.input_center(*record["winner"])),
                    run_time=0.42,
                    rate_func=smooth,
                )
            self.play(self.paint_input(record["origin"], record["winner"], "avg"), run_time=0.42, rate_func=smooth)
            write_output(record, "avg")

        # Hold the 4×4, both 2×2 outputs, window, and traveler through the last frame.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=11.20, rate_func=linear)
