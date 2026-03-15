from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .utils import tour_cost


def nearest_neighbor_from_start(dist_matrix: np.ndarray, start: int) -> np.ndarray:
    n = dist_matrix.shape[0]
    unvisited = np.ones(n, dtype=bool)
    unvisited[start] = False
    tour = [start]
    current = start

    for _ in range(n - 1):
        candidates = np.where(unvisited)[0]
        next_city = candidates[np.argmin(dist_matrix[current, candidates])]
        tour.append(int(next_city))
        unvisited[next_city] = False
        current = int(next_city)

    return np.asarray(tour, dtype=np.int32)


def multi_start_nearest_neighbor(dist_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    best_tour = None
    best_cost = float("inf")
    for start in range(dist_matrix.shape[0]):
        tour = nearest_neighbor_from_start(dist_matrix, start)
        cost = tour_cost(tour, dist_matrix)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour
    return best_tour, best_cost


def random_tour(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(n).astype(np.int32)


def select_initial_solution(dist_matrix: np.ndarray, seed: int = 42) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    greedy_tour, greedy_cost = multi_start_nearest_neighbor(dist_matrix)
    rand_tour = random_tour(dist_matrix.shape[0], rng)
    rand_cost = tour_cost(rand_tour, dist_matrix)

    if greedy_cost <= rand_cost:
        selected_name = "msnn"
        selected_tour = greedy_tour
        selected_cost = greedy_cost
    else:
        selected_name = "random"
        selected_tour = rand_tour
        selected_cost = rand_cost

    return {
        "greedy_tour": greedy_tour,
        "greedy_cost": greedy_cost,
        "random_tour": rand_tour,
        "random_cost": rand_cost,
        "selected_name": selected_name,
        "selected_tour": selected_tour,
        "selected_cost": selected_cost,
    }
