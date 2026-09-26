# Renderer-independent numerical model from 14.1
"""D2L 14.1 — skip-gram: v_c pulls window context u_o closer.

Every plotted number comes from the arrays below.  Four tokens start as
axis-aligned 2-D embeddings (one-hot spirit: v_c ⊥ u_o).  One skip-gram
pair (c, o) is trained with real softmax SGD.  u_o approaches v_c; u_n
and u_k stay far.  The halo traveler is the center vector v_c.
"""

from __future__ import annotations
import numpy as np

INITIAL_U = np.array([[1.2, 0.0], [0.0, 1.2], [0.0, -1.2], [-1.2, 0.0]], dtype=float)
INITIAL_V = np.array([1.2, 0.0], dtype=float)
CENTER = 0
CONTEXT = 1
FAR = 2
OTHER = 3
LEARNING_RATE = 0.22
SGD_STEPS = 6
TOKEN_NAMES = ("v_c", "u_o", "u_n", "u_k")


def softmax(values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for the same scores drawn on screen."""
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def skipgram_states() -> list[dict[str, np.ndarray]]:
    """SGD on one skip-gram pair: L = -log softmax(U v_c)[o]."""
    context = INITIAL_U.copy()
    center = INITIAL_V.copy()
    states: list[dict[str, np.ndarray]] = []

    def snapshot() -> None:
        scores = context @ center
        states.append(
            {
                "U": context.copy(),
                "v": center.copy(),
                "scores": scores.copy(),
                "p": softmax(scores).copy(),
            }
        )

    snapshot()
    for _ in range(SGD_STEPS):
        scores = context @ center
        probabilities = softmax(scores)
        grad_scores = probabilities.copy()
        grad_scores[CONTEXT] -= 1.0
        grad_context = grad_scores[:, None] * center[None, :]
        grad_center = context.T @ grad_scores
        context = context - LEARNING_RATE * grad_context
        center = center - LEARNING_RATE * grad_center
        snapshot()
    return states


STATES = skipgram_states()


def mix_state(progress: float) -> dict[str, np.ndarray]:
    """Linear blend between stored SGD snapshots. Numpy is the only arithmetic."""
    last = len(STATES) - 1
    clamped = min(max(progress, 0.0), float(last))
    index = int(np.floor(clamped))
    if index >= last:
        current = STATES[-1]
        return {
            "U": current["U"].copy(),
            "v": current["v"].copy(),
            "scores": current["scores"].copy(),
            "p": current["p"].copy(),
        }
    fraction = clamped - index
    context = (1.0 - fraction) * STATES[index]["U"] + fraction * STATES[index + 1]["U"]
    center = (1.0 - fraction) * STATES[index]["v"] + fraction * STATES[index + 1]["v"]
    scores = context @ center
    return {"U": context, "v": center, "scores": scores, "p": softmax(scores)}
