"""D2L 13.3 — a box around an object; corners convert to center+size and back.

The arrays and conversion helpers below are the only numerical source of
truth. Image coordinates: origin at the top-left, +x right, +y down.
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
    Line,
    Scene,
    Text,
    VGroup,
    VMobject,
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
# One numerical source of truth: D2L box_corner_to_center / box_center_to_corner
# on a single axis-aligned box. The object is a tilted ellipse strictly inside.
# ---------------------------------------------------------------------------
CORNER = np.array([[0.50, 0.30, 2.10, 1.50]], dtype=float)
OBJECT_CENTER = np.array([1.30, 0.90], dtype=float)
OBJECT_RX = 0.66
OBJECT_RY = 0.42
OBJECT_DEG = 28.0


def box_corner_to_center(boxes: np.ndarray) -> np.ndarray:
    """D2L: (x1, y1, x2, y2) → (cx, cy, w, h)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack(((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1), axis=-1)


def box_center_to_corner(boxes: np.ndarray) -> np.ndarray:
    """D2L: (cx, cy, w, h) → (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), axis=-1)


CENTER = box_corner_to_center(CORNER)
ROUNDTRIP = box_center_to_corner(CENTER)
X1, Y1, X2, Y2 = (float(CORNER[0, i]) for i in range(4))
CX, CY, WIDTH, HEIGHT = (float(CENTER[0, i]) for i in range(4))


def object_boundary(samples: int = 256) -> np.ndarray:
    """Tilted ellipse in the same image coordinates as the box."""
    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    local = np.stack((OBJECT_RX * np.cos(angles), OBJECT_RY * np.sin(angles)), axis=1)
    radians = np.deg2rad(OBJECT_DEG)
    cosine, sine = np.cos(radians), np.sin(radians)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + OBJECT_CENTER


class Episode133(Scene):
    """A ~28-second continuous, silent D2L 13.3 visualization."""

    # Image rectangle in data units. +y is down, matching D2L 13.3.
    x_min, x_max = 0.0, 3.00
    y_min, y_max = 0.0, 2.10
    plot_left, plot_right = -5.15, 2.40
    plot_bottom, plot_top = -3.20, 2.70

    def plot_point(self, x_value: float, y_value: float) -> np.ndarray:
        """Map image coordinates (y down) onto the scene (y up)."""
        scene_x = self.plot_left + (x_value - self.x_min) / (self.x_max - self.x_min) * (
            self.plot_right - self.plot_left
        )
        scene_y = self.plot_top - (y_value - self.y_min) / (self.y_max - self.y_min) * (
            self.plot_top - self.plot_bottom
        )
        return np.array([scene_x, scene_y, 0.0])

    def make_grid_and_frame(self) -> tuple[VGroup, VGroup]:
        """Stroke-only Line mobjects; never a filled grid mesh."""
        grid = VGroup()
        for x_value in np.arange(0.5, self.x_max - 1e-8, 0.5):
            grid.add(
                Line(self.plot_point(x_value, self.y_min), self.plot_point(x_value, self.y_max)).set_stroke(
                    BLUE_E, width=1.0, opacity=0.30
                )
            )
        for y_value in np.arange(0.5, self.y_max - 1e-8, 0.5):
            grid.add(
                Line(self.plot_point(self.x_min, y_value), self.plot_point(self.x_max, y_value)).set_stroke(
                    BLUE_E, width=1.0, opacity=0.30
                )
            )

        top = Line(self.plot_point(self.x_min, self.y_min), self.plot_point(self.x_max, self.y_min)).set_stroke(
            BLUE_D, width=2.0, opacity=0.90
        )
        bottom = Line(self.plot_point(self.x_min, self.y_max), self.plot_point(self.x_max, self.y_max)).set_stroke(
            BLUE_D, width=2.0, opacity=0.90
        )
        left = Line(self.plot_point(self.x_min, self.y_min), self.plot_point(self.x_min, self.y_max)).set_stroke(
            BLUE_D, width=2.0, opacity=0.90
        )
        right = Line(self.plot_point(self.x_max, self.y_min), self.plot_point(self.x_max, self.y_max)).set_stroke(
            BLUE_D, width=2.0, opacity=0.90
        )
        x_symbol = Text("x", font_size=27, color=BLUE_A).next_to(
            top.get_end(), np.array([1.0, 1.0, 0.0]), buff=0.12
        )
        y_symbol = Text("y", font_size=27, color=BLUE_A).next_to(
            left.get_end(), np.array([-1.0, -1.0, 0.0]), buff=0.12
        )
        origin = Text("0", font_size=20, color=BLUE_A).next_to(
            self.plot_point(self.x_min, self.y_min), np.array([-1.0, 1.0, 0.0]), buff=0.10
        )
        return grid, VGroup(top, bottom, left, right, x_symbol, y_symbol, origin)

    def object_shape(self) -> VMobject:
        """Stroke object with a faint fill so it is not another rectangle."""
        boundary = object_boundary()
        points = [self.plot_point(float(x_value), float(y_value)) for x_value, y_value in boundary]
        shape = VMobject(fill_opacity=0.22, stroke_opacity=0.96)
        shape.set_points_as_corners(points + [points[0]])
        shape.set_fill(BLUE, opacity=0.22)
        shape.set_stroke(BLUE_A, width=3.4, opacity=0.96)
        return shape

    def box_outline(self) -> VGroup:
        top_left = self.plot_point(X1, Y1)
        top_right = self.plot_point(X2, Y1)
        bottom_right = self.plot_point(X2, Y2)
        bottom_left = self.plot_point(X1, Y2)
        return VGroup(
            Line(top_left, top_right),
            Line(top_right, bottom_right),
            Line(bottom_right, bottom_left),
            Line(bottom_left, top_left),
        ).set_stroke(YELLOW, width=4.2, opacity=0.98)

    def stacked_pair(self, left: str, right: str, color, font_size: int = 20) -> VGroup:
        first = Text(left, font_size=font_size, color=color)
        second = Text(right, font_size=font_size, color=color)
        return VGroup(first, second).arrange(np.array([0.0, -1.0, 0.0]), buff=0.08)

    def construct(self) -> None:
        roundtrip_error = float(np.max(np.abs(ROUNDTRIP - CORNER)))
        print(
            "D2L 13.3 corner x1,y1,x2,y2 = [{:.6f}, {:.6f}, {:.6f}, {:.6f}]".format(X1, Y1, X2, Y2)
        )
        print(
            "D2L 13.3 center cx,cy,w,h = [{:.6f}, {:.6f}, {:.6f}, {:.6f}]".format(
                CX, CY, WIDTH, HEIGHT
            )
        )
        print("D2L 13.3 roundtrip max abs err = {:.3e}".format(roundtrip_error))
        print("D2L 13.3 object center={} rx={:.3f} ry={:.3f} deg={:.1f}".format(
            np.array2string(OBJECT_CENTER, precision=3, separator=", "),
            OBJECT_RX,
            OBJECT_RY,
            OBJECT_DEG,
        ))

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "13.3")

        grid, frame = self.make_grid_and_frame()
        blob = self.object_shape()
        box = self.box_outline()

        top_left = self.plot_point(X1, Y1)
        bottom_right = self.plot_point(X2, Y2)
        center_point = self.plot_point(CX, CY)

        corner_dot_a = (
            Dot(top_left, radius=0.068, color=YELLOW)
            .set_fill(YELLOW, opacity=1.0)
            .set_stroke(YELLOW, width=0.0, opacity=0.0)
        )
        corner_dot_b = (
            Dot(bottom_right, radius=0.068, color=YELLOW)
            .set_fill(YELLOW, opacity=1.0)
            .set_stroke(YELLOW, width=0.0, opacity=0.0)
        )

        corner_a_label = self.stacked_pair("x₁ = 0.50", "y₁ = 0.30", YELLOW_A)
        # Left and slightly above the top-left corner, in the image margin.
        corner_a_label.move_to(top_left + np.array([-0.78, 0.48, 0.0]))
        corner_b_label = self.stacked_pair("x₂ = 2.10", "y₂ = 1.50", YELLOW_A)
        corner_b_label.move_to(bottom_right + np.array([0.86, -0.48, 0.0]))

        formula = VGroup(
            Text("cx = (x₁+x₂)/2", font_size=24, color=WHITE),
            Text("cy = (y₁+y₂)/2", font_size=24, color=WHITE),
            Text("w = x₂−x₁", font_size=24, color=WHITE),
            Text("h = y₂−y₁", font_size=24, color=WHITE),
        ).arrange(np.array([0.0, -1.0, 0.0]), buff=0.14, aligned_edge=np.array([-1.0, 0.0, 0.0])).move_to(
            np.array([5.08, 1.18, 0.0])
        )

        halo_outer = Circle(radius=0.265).move_to(center_point).set_stroke(BLUE_A, width=2.1, opacity=0.50)
        halo_inner = Circle(radius=0.172).move_to(center_point).set_stroke(YELLOW, width=2.8, opacity=0.98)
        halo_outer.set_fill(BLACK, opacity=0.0)
        halo_inner.set_fill(BLACK, opacity=0.0)

        pocket_cx = Text("cx = 1.30", font_size=22, color=BLUE_A).move_to(np.array([-6.18, -1.18, 0.0]))
        pocket_cy = Text("cy = 0.90", font_size=22, color=BLUE_A).move_to(np.array([-6.18, -1.63, 0.0]))

        extent_reveal = ValueTracker(0.0)

        def width_bar() -> Line:
            half = 0.5 * WIDTH * extent_reveal.get_value()
            start = self.plot_point(CX - half, CY)
            end = self.plot_point(CX + half, CY)
            return Line(start, end).set_stroke(YELLOW_A, width=3.4, opacity=0.96)

        def height_bar() -> Line:
            half = 0.5 * HEIGHT * extent_reveal.get_value()
            start = self.plot_point(CX, CY - half)
            end = self.plot_point(CX, CY + half)
            return Line(start, end).set_stroke(YELLOW_A, width=3.4, opacity=0.96)

        width_line = always_redraw(width_bar)
        height_line = always_redraw(height_bar)

        width_label = Text("w = 1.60", font_size=20, color=YELLOW)
        # Above the top edge, off the yellow stroke and off the object.
        width_label.move_to(self.plot_point(CX, Y1) + np.array([0.0, 0.48, 0.0]))
        height_label = Text("h = 1.20", font_size=20, color=YELLOW)
        # Right of the box, above the width bar, off the ellipse.
        height_label.move_to(self.plot_point(X2, CY) + np.array([0.70, 0.38, 0.0]))

        # 0.40–3.40 s: the image and the object, before any box.
        self.play(FadeIn(grid), FadeIn(frame), run_time=0.85, rate_func=linear)
        self.play(FadeIn(blob), run_time=0.70, rate_func=smooth)
        self.wait(1.45)

        # 3.40–6.90 s: one rectangle boxes the object.
        self.play(Create(box), run_time=1.05, rate_func=smooth)
        self.wait(1.70)

        # 6.90–9.60 s: corner form on the same rectangle.
        self.play(
            FadeIn(corner_dot_a),
            FadeIn(corner_dot_b),
            FadeIn(corner_a_label),
            FadeIn(corner_b_label),
            Flash(top_left, color=YELLOW, flash_radius=0.36, line_length=0.10),
            Flash(bottom_right, color=YELLOW, flash_radius=0.36, line_length=0.10),
            run_time=0.85,
            rate_func=smooth,
        )
        self.wait(1.85)

        # 9.60–12.00 s: the conversion, once.
        self.play(FadeIn(formula), run_time=0.55, rate_func=smooth)
        self.wait(1.85)

        # 12.00–16.20 s: center + width + height on that same rectangle.
        self.add(width_line, height_line)
        self.play(
            FadeIn(halo_outer),
            FadeIn(halo_inner),
            FadeIn(pocket_cx),
            FadeIn(pocket_cy),
            Flash(center_point, color=YELLOW, flash_radius=0.42, line_length=0.11),
            extent_reveal.animate.set_value(1.0),
            FadeIn(width_label),
            FadeIn(height_label),
            run_time=1.35,
            rate_func=smooth,
        )
        self.wait(3.20)

        # And back — corners still match the center form.
        self.play(
            Flash(top_left, color=YELLOW, flash_radius=0.40, line_length=0.11),
            Flash(bottom_right, color=YELLOW, flash_radius=0.40, line_length=0.11),
            run_time=0.70,
            rate_func=smooth,
        )
        self.wait(2.20)

        # Hold live updaters through the final encoded frame. The box stays.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=8.60, rate_func=linear)
