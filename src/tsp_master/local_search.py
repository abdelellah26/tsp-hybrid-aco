from __future__ import annotations

import numpy as np

from .utils import tour_cost


def two_opt_first_improvement(
    tour: np.ndarray,
    dist_matrix: np.ndarray,
    candidate_lists: np.ndarray | None = None,
    max_passes: int = 10,
) -> tuple[np.ndarray, float]:
    """Restricted 2-opt using first improvement.

    If candidate_lists is provided, the second edge endpoint is searched only through
    candidate neighbors of the current city. This is much faster for large instances.
    """
    route = np.asarray(tour, dtype=np.int32).copy()
    n = route.shape[0]
    position = np.empty(n, dtype=np.int32)

    def update_positions() -> None:
        position[route] = np.arange(n, dtype=np.int32)

    update_positions()
    improved = True
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1

        for i in range(n - 1):
            a = route[i]
            b = route[(i + 1) % n]

            if candidate_lists is None:
                candidate_nodes = route
            else:
                candidate_nodes = candidate_lists[a]

            for node in candidate_nodes:
                j = int(position[node])
                if j <= i + 1 or j >= n - 1:
                    continue

                c = route[j]
                d = route[(j + 1) % n]
                delta = (
                    dist_matrix[a, c]
                    + dist_matrix[b, d]
                    - dist_matrix[a, b]
                    - dist_matrix[c, d]
                )
                if delta < -1e-12:
                    route[i + 1 : j + 1] = route[i + 1 : j + 1][::-1]
                    update_positions()
                    improved = True
                    break
            if improved:
                break

    return route, tour_cost(route, dist_matrix)
