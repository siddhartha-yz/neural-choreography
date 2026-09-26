# Numerical model from 10.5; renderer colors omitted.
"""D2L 10.5 — two attention heads, then concat and a linear map back.

Every weight, head scalar, and ŷ comes from the numpy multi-head pass
below. Projections are fixed and untrained. The haloed traveler is the
query. Head 1 and head 2 look at the same three tokens with different α.
"""

from __future__ import annotations
import numpy as np

TOKENS = np.array([[0.2, 1.2], [1.0, 0.5], [-1.0, -0.9]], dtype=float)
QUERY_INDEX = 1
WEIGHT_Q = (np.array([[0.8, 0.4]], dtype=float), np.array([[0.7, 0.4]], dtype=float))
WEIGHT_K = (np.array([[0.4, 2.4]], dtype=float), np.array([[-1.4, -1.2]], dtype=float))
WEIGHT_V = (np.array([[0.3, 1.1]], dtype=float), np.array([[-1.1, -0.4]], dtype=float))
WEIGHT_O = np.array([[0.9, -0.35], [-0.15, -0.3]], dtype=float)
HEAD_DIM = 1
SCALE = 1.0 / np.sqrt(HEAD_DIM)


def softmax(values: np.ndarray) -> np.ndarray:
    """The same softmax drawn as cell opacity and ray weight."""
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def one_head(
    weight_q: np.ndarray, weight_k: np.ndarray, weight_v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """One real head: project q,k,v, then scaled-dot-product pooling."""
    query = TOKENS[QUERY_INDEX].reshape(2, 1)
    projected_query = float((weight_q @ query).item())
    projected_keys = (weight_k @ TOKENS.T).ravel()
    projected_values = (weight_v @ TOKENS.T).ravel()
    scores = projected_query * projected_keys * SCALE
    weights = softmax(scores)
    head_value = float(np.dot(weights, projected_values))
    return (projected_keys, projected_values, scores, weights, head_value)


HEAD_KEYS = []
HEAD_VALUES = []
HEAD_SCORES = []
HEAD_WEIGHTS = []
HEAD_OUTPUTS = []
for weight_q, weight_k, weight_v in zip(WEIGHT_Q, WEIGHT_K, WEIGHT_V):
    keys, values, scores, weights, head_value = one_head(weight_q, weight_k, weight_v)
    HEAD_KEYS.append(keys)
    HEAD_VALUES.append(values)
    HEAD_SCORES.append(scores)
    HEAD_WEIGHTS.append(weights)
    HEAD_OUTPUTS.append(head_value)
CONCAT = np.array(HEAD_OUTPUTS, dtype=float).reshape(2, 1)
YHAT = (WEIGHT_O @ CONCAT).ravel()
TOKEN_NAMES = ("1", "2", "3")
