from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CacheSnapshot:
    s_states: list[np.ndarray] = field(default_factory=list)
    c_states: list[np.ndarray] = field(default_factory=list)
    k_states: list[np.ndarray] = field(default_factory=list)
    v_states: list[np.ndarray] = field(default_factory=list)

    def byte_size(self) -> int:
        total = 0
        for arr in self.s_states + self.c_states + self.k_states + self.v_states:
            total += arr.nbytes
        return total

    def is_empty(self) -> bool:
        return not any([self.s_states, self.c_states, self.k_states, self.v_states])

    def copy(self) -> "CacheSnapshot":
        return CacheSnapshot(
            s_states=[a.copy() for a in self.s_states],
            c_states=[a.copy() for a in self.c_states],
            k_states=[a.copy() for a in self.k_states],
            v_states=[a.copy() for a in self.v_states],
        )
