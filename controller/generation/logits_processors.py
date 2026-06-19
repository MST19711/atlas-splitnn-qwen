from __future__ import annotations

import numpy as np


def apply_presence_penalty(logits: np.ndarray, token_counts: dict[int, int], penalty: float) -> np.ndarray:
    if penalty <= 0:
        return logits
    updated = logits.copy()
    for token_id, count in token_counts.items():
        if count > 0:
            updated[token_id] -= penalty
    return updated


def apply_repetition_penalty(logits: np.ndarray, token_counts: dict[int, int], penalty: float) -> np.ndarray:
    if penalty in (0.0, 1.0):
        return logits
    updated = logits.copy()
    for token_id, count in token_counts.items():
        if count <= 0:
            continue
        if updated[token_id] > 0:
            updated[token_id] /= penalty
        else:
            updated[token_id] *= penalty
    return updated
