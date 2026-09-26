# Renderer-independent numerical model from 04.8
"""D2L 4.8 — sigmoid saturation turning backward needles into stubs.

Every moving value comes from the shared affine+sigmoid chain below. The
network is never trained; the two inits are two fixed parameter sets, and
``network_at`` is the only numerical source for activations and |∂ℓ/∂h|.
"""

from __future__ import annotations
import numpy as np

INPUT_X = np.array([[1.0], [0.5]], dtype=float)
SAT_W1 = np.array([[2.4, 1.8]], dtype=float)
SAT_B1 = np.array([[1.1]], dtype=float)
SAT_W = np.array([3.2, 3.4, 2.8], dtype=float)
SAT_B = np.array([0.8, -0.4, 0.6], dtype=float)
GOOD_W1 = np.array([[0.8, 0.5]], dtype=float)
GOOD_B1 = np.array([[0.0]], dtype=float)
GOOD_W = np.array([1.0, 0.9, 1.0], dtype=float)
GOOD_B = np.array([0.0, 0.0, 0.0], dtype=float)


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Elementwise logistic, the same σ drawn on screen."""
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def sigmoid_prime(values: np.ndarray) -> np.ndarray:
    """σ'(z) = σ(z) (1 − σ(z)), the local factor in every backward needle."""
    activated = sigmoid(values)
    return activated * (1.0 - activated)


def network_at(mix: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward and backward pass at a convex combination of the two inits.

    Incoming sensitivity is ∂ℓ/∂h₄ = 1. Nothing is trained; mix only blends
    the two frozen parameter sets so the optional linear-region beat stays
    on the same numpy chain.
    """
    mix = float(mix)
    weight_1 = (1.0 - mix) * SAT_W1 + mix * GOOD_W1
    bias_1 = (1.0 - mix) * SAT_B1 + mix * GOOD_B1
    weights = (1.0 - mix) * SAT_W + mix * GOOD_W
    biases = (1.0 - mix) * SAT_B + mix * GOOD_B
    preactivations = []
    activations = []
    hidden = None
    for layer_index in range(4):
        if layer_index == 0:
            preactivation = weight_1 @ INPUT_X + bias_1
        else:
            preactivation = np.array(
                [
                    [
                        weights[layer_index - 1] * float(hidden.item())
                        + biases[layer_index - 1]
                    ]
                ],
                dtype=float,
            )
        hidden = sigmoid(preactivation)
        preactivations.append(preactivation)
        activations.append(hidden)
    grad_h = [np.array([[0.0]], dtype=float) for _ in range(4)]
    grad_h[3] = np.array([[1.0]], dtype=float)
    for layer_index in reversed(range(4)):
        local = sigmoid_prime(preactivations[layer_index])
        grad_z = grad_h[layer_index] * local
        if layer_index == 0:
            grad_x = weight_1.T @ grad_z
        else:
            grad_h[layer_index - 1] = grad_z * weights[layer_index - 1]
    zs = np.array([float(item.item()) for item in preactivations], dtype=float)
    hs = np.array([float(item.item()) for item in activations], dtype=float)
    grad_hs = np.array([abs(float(item.item())) for item in grad_h], dtype=float)
    return (zs, hs, grad_hs, np.abs(grad_x.reshape(-1)))


SAT_Z, SAT_H, SAT_GRAD_H, SAT_GRAD_X = network_at(0.0)
GOOD_Z, GOOD_H, GOOD_GRAD_H, GOOD_GRAD_X = network_at(1.0)
