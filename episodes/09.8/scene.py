"""D2L 9.8 — beam search keeps k candidates per step; greedy is k=1.

Every probability on screen comes from the softmax tables below.  The same
scores feed both trees: left k=1, right k=2.  Pruned branches are not
expanded.  The yellow halo marks the surviving k=2 beam.
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
    Circle,
    Create,
    DOWN,
    FadeIn,
    FadeOut,
    Flash,
    LaggedStart,
    LEFT,
    Line,
    RIGHT,
    Scene,
    Text,
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
# One numerical source of truth.  Tiny vocab {A, B, C}.  Rows of STEP2_LOGITS
# are P(y₂ | y₁, c) after softmax.  Joints are P(y₁ | c) P(y₂ | y₁, c).
# ---------------------------------------------------------------------------
VOCAB = ("A", "B", "C")
TOKEN_COLORS = (YELLOW, BLUE, BLUE_D)
STEP1_LOGITS = np.array([1.00, 0.82, -1.60], dtype=float)
STEP2_LOGITS = np.array(
    [
        [-0.25, 0.45, -0.55],
        [2.40, -0.70, -0.30],
        [0.10, 0.00, -0.20],
    ],
    dtype=float,
)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Stable softmax for the same values drawn on screen."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


P_STEP1 = softmax(STEP1_LOGITS)
P_STEP2 = softmax(STEP2_LOGITS)
JOINT = P_STEP1[:, np.newaxis] * P_STEP2

K1_T1_KEEP = (int(np.argmax(P_STEP1)),)
K2_T1_KEEP = tuple(int(index) for index in np.argsort(-P_STEP1)[:2])
K1_T1_PRUNE = tuple(index for index in range(3) if index not in K1_T1_KEEP)
K2_T1_PRUNE = tuple(index for index in range(3) if index not in K2_T1_KEEP)


def topk_paths(prefixes: tuple[int, ...], width: int) -> list[tuple[int, int]]:
    """Rank prefix+token joints and keep ``width`` paths."""
    ranked = sorted(
        ((prefix, token, float(JOINT[prefix, token])) for prefix in prefixes for token in range(3)),
        key=lambda item: -item[2],
    )
    return [(prefix, token) for prefix, token, _ in ranked[:width]]


K1_T2_KEEP = topk_paths(K1_T1_KEEP, 1)
K2_T2_KEEP = topk_paths(K2_T1_KEEP, 2)
K1_T2_PRUNE = [(K1_T1_KEEP[0], token) for token in range(3) if (K1_T1_KEEP[0], token) not in K1_T2_KEEP]
K2_T2_PRUNE = [
    (prefix, token)
    for prefix in K2_T1_KEEP
    for token in range(3)
    if (prefix, token) not in K2_T2_KEEP
]


NODE_RADIUS = 0.22
PRUNE_OPACITY = 0.18


def score_text(value: float) -> str:
    return f"{value:.3f}"


class Episode098(Scene):
    """A ~32-second silent D2L 9.8 visualization: k=1 vs k=2 on one score table."""

    def stroke_ring(self, center: np.ndarray, color, width: float = 2.4, radius: float = NODE_RADIUS) -> Circle:
        return (
            Circle(radius=radius)
            .move_to(center)
            .set_fill(opacity=0.0)
            .set_stroke(color, width=width, opacity=1.0)
        )

    def edge_line(self, start: np.ndarray, end: np.ndarray, color, start_radius: float = NODE_RADIUS) -> Line:
        offset = end - start
        direction = offset / np.linalg.norm(offset)
        return Line(start + direction * start_radius, end - direction * NODE_RADIUS).set_stroke(
            color, width=2.4, opacity=0.95
        )

    def make_root(self, center: np.ndarray) -> Text:
        """Letter only: a ring around c reads as a copyright mark."""
        return Text("c", font_size=30, color=BLUE_A).move_to(center)

    def make_branch(
        self,
        parent: np.ndarray,
        child: np.ndarray,
        letter: str,
        color,
        value: float,
        score_side: str,
        start_radius: float = NODE_RADIUS,
    ) -> VGroup:
        """Stroke edge + node + on-geometry score. Never a filled disc, never a HUD."""
        edge = self.edge_line(parent, child, color, start_radius=start_radius)
        ring = self.stroke_ring(child, color)
        glyph = Text(letter, font_size=24, color=color).move_to(child)
        score = Text(score_text(value), font_size=18, color=WHITE)
        if score_side == "right":
            score.next_to(ring, RIGHT, buff=0.12)
        elif score_side == "left":
            score.next_to(ring, LEFT, buff=0.12)
        else:
            score.next_to(ring, DOWN, buff=0.22)
        return VGroup(edge, ring, glyph, score)

    def construct(self) -> None:
        print("D2L 9.8 vocab =", VOCAB)
        print("STEP1_LOGITS =", np.array2string(STEP1_LOGITS, precision=3, separator=", "))
        print("P(y1|c) =", np.array2string(P_STEP1, precision=6, separator=", "))
        print("STEP2_LOGITS =\n", np.array2string(STEP2_LOGITS, precision=3, separator=", "))
        print("P(y2|y1,c) =\n", np.array2string(P_STEP2, precision=6, separator=", "))
        print("joint P(y1,y2|c) =\n", np.array2string(JOINT, precision=6, separator=", "))
        print(
            "k=1 t1 keep {} prune {}; t2 keep {} score={:.6f}".format(
                "".join(VOCAB[i] for i in K1_T1_KEEP),
                "".join(VOCAB[i] for i in K1_T1_PRUNE),
                "".join(VOCAB[p] + VOCAB[t] for p, t in K1_T2_KEEP),
                float(JOINT[K1_T2_KEEP[0]]),
            )
        )
        print(
            "k=2 t1 keep {} prune {}; t2 keep {} scores={}".format(
                "".join(VOCAB[i] for i in K2_T1_KEEP),
                "".join(VOCAB[i] for i in K2_T1_PRUNE),
                ", ".join(VOCAB[p] + VOCAB[t] for p, t in K2_T2_KEEP),
                [float(JOINT[path]) for path in K2_T2_KEEP],
            )
        )
        print("surviving beam (k=2 top) =", VOCAB[K2_T2_KEEP[0][0]] + VOCAB[K2_T2_KEEP[0][1]])

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "09.8")

        # One empty pocket at the top: the keep-top-k formula, once.
        formula = Text("Y ← topk_k(Y × V)", font_size=34, color=WHITE).move_to(np.array([0.0, 3.42, 0.0]))
        k1_label = Text("k = 1", font_size=28, color=BLUE_A).move_to(np.array([-3.60, 2.82, 0.0]))
        k2_label = Text("k = 2", font_size=28, color=YELLOW).move_to(np.array([3.35, 2.82, 0.0]))

        k1_root_c = np.array([-3.60, 2.12, 0.0])
        k2_root_c = np.array([3.35, 2.12, 0.0])
        k1_t1_c = [
            np.array([-5.70, 0.40, 0.0]),
            np.array([-3.60, 0.40, 0.0]),
            np.array([-1.50, 0.40, 0.0]),
        ]
        k2_t1_c = [
            np.array([1.60, 0.40, 0.0]),
            np.array([4.55, 0.40, 0.0]),
            np.array([6.20, 0.40, 0.0]),
        ]
        k1_t2_c = [
            np.array([-6.75, -2.30, 0.0]),
            np.array([-5.50, -2.30, 0.0]),
            np.array([-4.25, -2.30, 0.0]),
        ]
        k2_t2_c = {
            (0, 0): np.array([0.45, -2.30, 0.0]),
            (0, 1): np.array([1.60, -2.30, 0.0]),
            (0, 2): np.array([2.75, -2.30, 0.0]),
            (1, 0): np.array([3.70, -2.30, 0.0]),
            (1, 1): np.array([4.85, -2.30, 0.0]),
            (1, 2): np.array([6.00, -2.30, 0.0]),
        }

        k1_root = self.make_root(k1_root_c)
        k2_root = self.make_root(k2_root_c)

        k1_t1 = VGroup(
            *[
                self.make_branch(
                    k1_root_c,
                    k1_t1_c[index],
                    VOCAB[index],
                    TOKEN_COLORS[index],
                    float(P_STEP1[index]),
                    score_side="right",
                    start_radius=0.18,
                )
                for index in range(3)
            ]
        )
        k2_t1 = VGroup(
            *[
                self.make_branch(
                    k2_root_c,
                    k2_t1_c[index],
                    VOCAB[index],
                    TOKEN_COLORS[index],
                    float(P_STEP1[index]),
                    score_side="down" if index == 2 else "right",
                    start_radius=0.18,
                )
                for index in range(3)
            ]
        )
        k1_t2 = VGroup(
            *[
                self.make_branch(
                    k1_t1_c[K1_T1_KEEP[0]],
                    k1_t2_c[token],
                    VOCAB[token],
                    TOKEN_COLORS[token],
                    float(JOINT[K1_T1_KEEP[0], token]),
                    score_side="down",
                )
                for token in range(3)
            ]
        )
        k2_t2_branches = {
            (prefix, token): self.make_branch(
                k2_t1_c[prefix],
                k2_t2_c[(prefix, token)],
                VOCAB[token],
                TOKEN_COLORS[token],
                float(JOINT[prefix, token]),
                score_side="down",
            )
            for prefix in K2_T1_KEEP
            for token in range(3)
        }

        # 0.40–2.20 s: two empty trees and the formula.
        self.play(
            FadeIn(formula),
            FadeIn(k1_label),
            FadeIn(k2_label),
            FadeIn(k1_root),
            FadeIn(k2_root),
            run_time=0.50,
            rate_func=smooth,
        )
        self.wait(1.30)

        # 2.20–4.40 s: the same t=1 scores grow on both beams.
        self.play(
            LaggedStart(
                *[
                    AnimationGroup(FadeIn(k1_t1[index]), FadeIn(k2_t1[index]))
                    for index in range(3)
                ],
                lag_ratio=0.34,
            ),
            run_time=2.20,
            rate_func=smooth,
        )

        # 4.40–6.60 s: keep k, prune the rest. Greedy drops B and C; k=2 drops C.
        k1_t1_pruned = VGroup(*[k1_t1[index] for index in K1_T1_PRUNE])
        k2_t1_pruned = VGroup(*[k2_t1[index] for index in K2_T1_PRUNE])
        self.play(
            k1_t1_pruned.animate.set_opacity(PRUNE_OPACITY),
            k2_t1_pruned.animate.set_opacity(PRUNE_OPACITY),
            run_time=2.20,
            rate_func=smooth,
        )

        # 6.60–8.80 s: expand only the kept prefixes.
        k2_expand_order = [(prefix, token) for prefix in K2_T1_KEEP for token in range(3)]
        self.play(
            LaggedStart(
                *[FadeIn(k1_t2[token]) for token in range(3)],
                *[FadeIn(k2_t2_branches[path]) for path in k2_expand_order],
                lag_ratio=0.12,
            ),
            run_time=2.20,
            rate_func=smooth,
        )

        # 8.80–11.00 s: prune to k again. Joints, not the one-step conditionals.
        k1_t2_pruned = VGroup(*[k1_t2[token] for _, token in K1_T2_PRUNE])
        k2_t2_pruned = VGroup(*[k2_t2_branches[path] for path in K2_T2_PRUNE])
        self.play(
            k1_t2_pruned.animate.set_opacity(PRUNE_OPACITY),
            k2_t2_pruned.animate.set_opacity(PRUNE_OPACITY),
            run_time=2.20,
            rate_func=smooth,
        )

        # 11.00–13.20 s: halo traveler lands on the surviving k=2 beam (BA).
        survivor_prefix, survivor_token = K2_T2_KEEP[0]
        survivor_center = k2_t2_c[(survivor_prefix, survivor_token)]
        survivor_branch = k2_t2_branches[(survivor_prefix, survivor_token)]
        halo_outer = (
            Circle(radius=0.36)
            .move_to(survivor_center)
            .set_fill(opacity=0.0)
            .set_stroke(BLUE_A, width=2.1, opacity=0.50)
        )
        halo_inner = (
            Circle(radius=0.27)
            .move_to(survivor_center)
            .set_fill(opacity=0.0)
            .set_stroke(YELLOW, width=2.8, opacity=0.98)
        )
        self.play(
            survivor_branch[0].animate.set_stroke(YELLOW, width=3.6, opacity=1.0),
            survivor_branch[1].animate.set_stroke(YELLOW, width=3.2, opacity=1.0),
            Create(halo_outer),
            Create(halo_inner),
            Flash(survivor_center, color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=2.20,
            rate_func=smooth,
        )
        self.wait(2.30)

        # Hold kept beams through the last encoded frame.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=16.50, rate_func=linear)
