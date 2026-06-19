from __future__ import annotations

import numpy as np


def greedy_select(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


def sample_with_top_k_top_p(
    logits: np.ndarray,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    work = logits.astype(np.float64).copy()
    if temperature <= 0:
        return greedy_select(work)
    work /= temperature
    if 0 < top_k < len(work):
        idx = np.argpartition(work, -top_k)[-top_k:]
        mask = np.ones(len(work), dtype=bool)
        mask[idx] = False
        work[mask] = -np.inf
    if 0.0 < top_p < 1.0:
        order = np.argsort(work)[::-1]
        sorted_logits = work[order]
        probs = np.exp(sorted_logits - np.max(sorted_logits))
        probs /= probs.sum()
        cumsum = np.cumsum(probs)
        remove = cumsum > top_p
        if np.any(remove):
            remove[0] = False
            work[order[remove]] = -np.inf
    exp = np.exp(work - work.max())
    probs = exp / exp.sum()
    return int(np.random.choice(len(probs), p=probs))
