from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd

from .aco import solve_mmas
from .distance import compute_distance_matrix
from .heuristics import select_initial_solution
from .parser import parse_tsplib
from .utils import AdaptiveConfig, adaptive_config, ensure_dir


def default_param_grid() -> dict:
    return {
        "alpha": [1.0],
        "beta": [3.5, 4.0, 4.5, 5.0],
        "rho_start": [0.15, 0.20, 0.25, 0.30],
        "q0_start": [0.60, 0.75, 0.90],
        "candidate_k": [10, 15, 20, 30],
        "ls_top_ants_ratio": [0.05, 0.10, 0.20],
        "ls_frequency": [1, 2, 5],
        "ants_ratio": [0.15, 0.22, 0.30],
    }


BEST_KNOWN = {
    "eil51": 426,
    "berlin52": 7542,
    "st70": 675,
    "eil76": 538,
    "kroA100": 21282,
    "kroB100": 22141,
    "ch150": 6528,
    "rat195": 2323,
    "d493": 35002,
    "u574": 36905,
}


def generate_param_combinations(param_grid: dict) -> list[dict]:
    keys = list(param_grid.keys())
    values = [param_grid[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]



def build_tuned_config(n: int, base_seed: int, params: dict, use_local_search: bool = True) -> AdaptiveConfig:
    base = adaptive_config(n, seed=base_seed, use_local_search=use_local_search)
    ants = min(80, max(8, int(round(params["ants_ratio"] * n))))
    candidate_k = min(n - 1, int(params["candidate_k"]))
    ls_top_ants = max(1, int(round(params["ls_top_ants_ratio"] * ants)))

    if n >= 300:
        candidate_k = max(candidate_k, 20)
    if n >= 500:
        ants = min(ants, 64)

    return AdaptiveConfig(
        ants=ants,
        max_iter=base.max_iter,
        stagnation_limit=base.stagnation_limit,
        restart_after=base.restart_after,
        candidate_k=candidate_k,
        ls_candidate_k=min(candidate_k, max(10, int(np.sqrt(n)))),
        ls_top_ants=ls_top_ants,
        ls_frequency=max(1, int(params["ls_frequency"])),
        alpha=float(params["alpha"]),
        beta=float(params["beta"]),
        rho_start=float(params["rho_start"]),
        rho_end=base.rho_end,
        q0_start=float(params["q0_start"]),
        q0_end=base.q0_end,
        seed=base_seed,
        use_local_search=use_local_search,
    )



def run_single_tuning(
    dist_matrix: np.ndarray,
    config: AdaptiveConfig,
    initial_tour: np.ndarray,
) -> dict:
    start = time.perf_counter()
    result = solve_mmas(dist_matrix, config, initial_tour=initial_tour)
    elapsed = time.perf_counter() - start
    return {
        "cost": result["best_cost"],
        "tour": result["best_tour"],
        "convergence": result["convergence"],
        "time": elapsed,
    }



def evaluate_parameter_set(
    dist_matrix: np.ndarray,
    config: AdaptiveConfig,
    initial_tour: np.ndarray,
    n_runs: int,
    instance_name: str,
    optimum: float | None,
) -> dict:
    runs = []
    for run_idx in range(n_runs):
        run_cfg = AdaptiveConfig(**{**config.to_dict(), "seed": config.seed + run_idx})
        run_result = run_single_tuning(dist_matrix, run_cfg, initial_tour)
        runs.append(run_result)

    costs = [run["cost"] for run in runs]
    times = [run["time"] for run in runs]
    mean_cost = float(np.mean(costs))
    median_cost = float(np.median(costs))
    std_cost = float(np.std(costs))
    best_cost = float(np.min(costs))
    mean_time = float(np.mean(times))

    mean_gap = None if optimum is None else 100.0 * (mean_cost - optimum) / optimum
    best_gap = None if optimum is None else 100.0 * (best_cost - optimum) / optimum

    return {
        "instance": instance_name,
        "config": config.to_dict(),
        "mean_cost": mean_cost,
        "median_cost": median_cost,
        "std_cost": std_cost,
        "best_cost": best_cost,
        "mean_time": mean_time,
        "mean_gap": mean_gap,
        "best_gap": best_gap,
        "all_runs": runs,
    }



def select_best_configuration(results: list[dict]) -> dict:
    return sorted(results, key=lambda r: (r["mean_cost"], r["std_cost"], r["mean_time"]))[0]



def flatten_tuning_results(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {
            "instance": result["instance"],
            "mean_cost": result["mean_cost"],
            "median_cost": result["median_cost"],
            "std_cost": result["std_cost"],
            "best_cost": result["best_cost"],
            "mean_time": result["mean_time"],
            "mean_gap": result["mean_gap"],
            "best_gap": result["best_gap"],
        }
        row.update(result["config"])
        rows.append(row)
    return pd.DataFrame(rows)



def tune_instance(
    file_path: str,
    output_dir: str,
    param_grid: dict | None = None,
    n_runs: int = 5,
    max_combinations: int | None = None,
    seed: int = 42,
) -> dict:
    parsed = parse_tsplib(file_path)
    coords = parsed["coords"]
    name = parsed["name"]
    optimum = BEST_KNOWN.get(name)
    dist_matrix = compute_distance_matrix(coords, parsed["edge_weight_type"])
    init = select_initial_solution(dist_matrix, seed=seed)

    param_grid = default_param_grid() if param_grid is None else param_grid
    combinations = generate_param_combinations(param_grid)
    if max_combinations is not None:
        combinations = combinations[:max_combinations]

    all_results = []
    for idx, params in enumerate(combinations, start=1):
        config = build_tuned_config(dist_matrix.shape[0], seed, params, use_local_search=True)
        print(f"[{name}] tuning {idx}/{len(combinations)}: {config.to_dict()}")
        eval_result = evaluate_parameter_set(
            dist_matrix=dist_matrix,
            config=config,
            initial_tour=init["selected_tour"],
            n_runs=n_runs,
            instance_name=name,
            optimum=optimum,
        )
        all_results.append(eval_result)

    best = select_best_configuration(all_results)
    out_dir = ensure_dir(output_dir)
    df = flatten_tuning_results(all_results)
    df.to_csv(out_dir / f"{name}_tuning_results.csv", index=False)
    with (out_dir / f"{name}_best_config.json").open("w", encoding="utf-8") as handle:
        json.dump(best["config"], handle, indent=2)

    return {
        "instance_name": name,
        "coords": coords,
        "dist_matrix": dist_matrix,
        "initialization": init,
        "best": best,
        "all_results": all_results,
        "dataframe": df,
        "optimum": optimum,
    }
