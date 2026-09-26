# Numerical model from 09.8; renderer colors omitted.
"""D2L 9.8 — beam search keeps k candidates per step; greedy is k=1.

Every probability on screen comes from the softmax tables below.  The same
scores feed both trees: left k=1, right k=2.  Pruned branches are not
expanded.  The yellow halo marks the surviving k=2 beam.
"""

from __future__ import annotations
import numpy as np

VOCAB = ("A", "B", "C")
STEP1_LOGITS = np.array([1.0, 0.82, -1.6], dtype=float)
STEP2_LOGITS = np.array(
    [[-0.25, 0.45, -0.55], [2.4, -0.7, -0.3], [0.1, 0.0, -0.2]], dtype=float
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
K2_T1_KEEP = tuple((int(index) for index in np.argsort(-P_STEP1)[:2]))
K1_T1_PRUNE = tuple((index for index in range(3) if index not in K1_T1_KEEP))
K2_T1_PRUNE = tuple((index for index in range(3) if index not in K2_T1_KEEP))


def topk_paths(prefixes: tuple[int, ...], width: int) -> list[tuple[int, int]]:
    """Rank prefix+token joints and keep ``width`` paths."""
    ranked = sorted(
        (
            (prefix, token, float(JOINT[prefix, token]))
            for prefix in prefixes
            for token in range(3)
        ),
        key=lambda item: -item[2],
    )
    return [(prefix, token) for prefix, token, _ in ranked[:width]]


K1_T2_KEEP = topk_paths(K1_T1_KEEP, 1)
K2_T2_KEEP = topk_paths(K2_T1_KEEP, 2)
K1_T2_PRUNE = [
    (K1_T1_KEEP[0], token)
    for token in range(3)
    if (K1_T1_KEEP[0], token) not in K1_T2_KEEP
]
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
