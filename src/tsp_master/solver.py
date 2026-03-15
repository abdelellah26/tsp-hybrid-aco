from __future__ import annotations

from dataclasses import replace
from typing import Dict

import numpy as np

from .aco import solve_mmas
from .distance import compute_distance_matrix
from .heuristics import select_initial_solution
from .parser import parse_tsplib
from .utils import adaptive_config


def solve_instance(file_path: str, seed: int = 42) -> Dict[str, object]:
    instance = parse_tsplib(file_path)
    coords = instance["coords"]
    dist_matrix = compute_distance_matrix(coords, instance["edge_weight_type"])

    init = select_initial_solution(dist_matrix, seed=seed)
    base_cfg = adaptive_config(dist_matrix.shape[0], seed=seed, use_local_search=False)
    hybrid_cfg = adaptive_config(dist_matrix.shape[0], seed=seed, use_local_search=True)

    pure = solve_mmas(dist_matrix, base_cfg, initial_tour=init["selected_tour"])
    hybrid = solve_mmas(dist_matrix, hybrid_cfg, initial_tour=init["selected_tour"])

    return {
        "instance": instance,
        "dist_matrix": dist_matrix,
        "initialization": init,
        "pure_aco": pure,
        "hybrid_aco": hybrid,
    }
