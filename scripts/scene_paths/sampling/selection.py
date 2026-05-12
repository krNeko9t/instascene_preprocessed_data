from __future__ import annotations

import random
import sys
from typing import Sequence

from ..paths.record import ScenePathRecord


def select_scene_subset(
    candidates: Sequence[ScenePathRecord],
    *,
    n_requested: int,
    seed: int | None,
    shuffle: bool,
) -> list[ScenePathRecord]:
    if n_requested == -1:
        selected = list(candidates)
        if shuffle:
            random.shuffle(selected)
        return selected
    k = min(n_requested, len(candidates))
    if k < n_requested:
        print(
            f"warning: requested N={n_requested} but only {len(candidates)} candidates; selecting {k}",
            file=sys.stderr,
        )
    return random.sample(list(candidates), k)
