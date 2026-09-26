# Renderer-independent numerical model from 03.1
"""D2L 3.1 — a line fitting noisy points through real mini-batch SGD.

Every moving value comes from the column-vector computation below.  No
closed-form fit is used; ``training_states`` is produced only by mini-batch
gradient updates on mean squared error.
"""

from __future__ import annotations
import numpy as np

FEATURES = np.array(
    [-2.7, -2.3, -1.9, -1.5, -1.1, -0.7, -0.3, 0.1, 0.5, 0.9, 1.25, 1.65, 2.05, 2.45],
    dtype=float,
).reshape(-1, 1)
TRUE_W = np.array([[1.15]], dtype=float)
TRUE_B = np.array([[0.55]], dtype=float)
NOISE = np.array(
    [
        -0.2,
        0.12,
        -0.08,
        0.1,
        -0.12,
        0.05,
        -0.17,
        0.16,
        -0.06,
        0.08,
        -0.15,
        0.13,
        -0.09,
        0.19,
    ],
    dtype=float,
).reshape(-1, 1)
TARGETS = FEATURES @ TRUE_W + TRUE_B + NOISE
HIGHLIGHT_INDEX = 3
MINIBATCH_SIZE = 4
LEARNING_RATE = 0.05
INITIAL_W = np.array([[-0.45]], dtype=float)
INITIAL_B = np.array([[1.75]], dtype=float)
SGD_STEPS = 9


def mean_squared_loss(weight: np.ndarray, bias: np.ndarray) -> float:
    """MSE of the same noisy point cloud drawn in the scene."""
    residuals = FEATURES @ weight + bias - TARGETS
    return float(np.mean(residuals**2))


def make_minibatches() -> list[np.ndarray]:
    """Fixed mini-batches, with the traveler present in each visible update."""
    generator = np.random.default_rng(31)
    other_indices = np.delete(np.arange(len(FEATURES)), HIGHLIGHT_INDEX)
    batches: list[np.ndarray] = []
    for _ in range(SGD_STEPS - 1):
        batches.append(
            np.concatenate(
                (
                    [HIGHLIGHT_INDEX],
                    generator.choice(
                        other_indices, size=MINIBATCH_SIZE - 1, replace=False
                    ),
                )
            )
        )
    batches.append(np.array([HIGHLIGHT_INDEX, 0, 4, 7]))
    return batches


MINIBATCHES = make_minibatches()


def mini_batch_sgd() -> list[tuple[np.ndarray, np.ndarray]]:
    """Return model states created by mini-batch SGD, from start to finish."""
    weight = INITIAL_W.copy()
    bias = INITIAL_B.copy()
    states = [(weight.copy(), bias.copy())]
    for batch_indices in MINIBATCHES:
        batch_x = FEATURES[batch_indices]
        batch_y = TARGETS[batch_indices]
        batch_residuals = batch_x @ weight + bias - batch_y
        gradient_w = 2.0 / MINIBATCH_SIZE * (batch_x.T @ batch_residuals)
        gradient_b = (
            2.0 / MINIBATCH_SIZE * np.sum(batch_residuals, axis=0, keepdims=True)
        )
        weight = weight - LEARNING_RATE * gradient_w
        bias = bias - LEARNING_RATE * gradient_b
        states.append((weight.copy(), bias.copy()))
    return states


TRAINING_STATES = mini_batch_sgd()
