from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np


@dataclass
class AdaptiveConfig:
    ants: int
    max_iter: int
    stagnation_limit: int
    restart_after: int
    candidate_k: int
    ls_candidate_k: int
    ls_top_ants: int
    ls_frequency: int
    alpha: float
    beta: float
    rho_start: float
    rho_end: float
    q0_start: float
    q0_end: float
    seed: int = 42
    use_local_search: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def adaptive_config(n: int, seed: int = 42, use_local_search: bool = True) -> AdaptiveConfig:
    ants = min(64, max(12, int(round(0.22 * n))))
    candidate_k = min(40, max(15, int(round(1.5 * np.sqrt(n)))))
    ls_candidate_k = min(30, max(10, int(round(np.sqrt(n)))))

    if n <= 80:
        max_iter = 220
        beta = 5.0
        rho_start = 0.30
        q0_start = 0.72
        ls_frequency = 1
    elif n <= 150:
        max_iter = 260
        beta = 4.7
        rho_start = 0.26
        q0_start = 0.68
        ls_frequency = 1
    elif n <= 250:
        max_iter = 320
        beta = 4.3
        rho_start = 0.22
        q0_start = 0.64
        ls_frequency = 2
    elif n <= 400:
        max_iter = 420
        beta = 3.9
        rho_start = 0.18
        q0_start = 0.60
        ls_frequency = 3
    else:
        max_iter = 520
        beta = 3.5
        rho_start = 0.15
        q0_start = 0.58
        ls_frequency = 5

    stagnation_limit = max(25, int(0.18 * max_iter))
    restart_after = max(12, int(0.45 * stagnation_limit))

    if n <= 100:
        ls_top_ants = max(2, int(0.20 * ants))
    elif n <= 250:
        ls_top_ants = max(2, int(0.12 * ants))
    else:
        ls_top_ants = max(1, int(0.06 * ants))

    return AdaptiveConfig(
        ants=ants,
        max_iter=max_iter,
        stagnation_limit=stagnation_limit,
        restart_after=restart_after,
        candidate_k=candidate_k,
        ls_candidate_k=ls_candidate_k,
        ls_top_ants=ls_top_ants,
        ls_frequency=ls_frequency,
        alpha=1.0,
        beta=beta,
        rho_start=rho_start,
        rho_end=0.08,
        q0_start=q0_start,
        q0_end=0.92,
        seed=seed,
        use_local_search=use_local_search,
    )


def linear_schedule(start: float, end: float, t: int, t_max: int) -> float:
    if t_max <= 1:
        return end
    return float(start + (end - start) * (t / (t_max - 1)))


def tour_cost(tour: np.ndarray | List[int], dist_matrix: np.ndarray) -> float:
    route = np.asarray(tour, dtype=np.int32)
    return float(np.sum(dist_matrix[route, np.roll(route, -1)]))


def normalize_tour(tour: Iterable[int]) -> List[int]:
    items = list(tour)
    if not items:
        return items
    idx = items.index(min(items))
    rotated = items[idx:] + items[:idx]
    reversed_rotated = [rotated[0]] + list(reversed(rotated[1:]))
    return min(rotated, reversed_rotated)


def ensure_dir(path: str | Path) -> Path:
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder
