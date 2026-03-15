from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

import numpy as np

from .distance import build_candidate_lists
from .local_search import two_opt_first_improvement
from .utils import AdaptiveConfig, linear_schedule, tour_cost


class MMASSolver:
    def __init__(self, dist_matrix: np.ndarray, config: AdaptiveConfig):
        self.dist_matrix = dist_matrix
        self.config = config
        self.n = dist_matrix.shape[0]
        self.rng = np.random.default_rng(config.seed)
        self.eta = 1.0 / np.maximum(dist_matrix, 1e-12)
        np.fill_diagonal(self.eta, 0.0)
        self.candidate_lists = build_candidate_lists(dist_matrix, config.candidate_k)
        self.ls_candidate_lists = build_candidate_lists(dist_matrix, config.ls_candidate_k)

    def _construct_solution(self, tau: np.ndarray, alpha: float, beta: float, q0: float) -> np.ndarray:
        start = int(self.rng.integers(0, self.n))
        tour = np.empty(self.n, dtype=np.int32)
        tour[0] = start
        unvisited = np.ones(self.n, dtype=bool)
        unvisited[start] = False
        current = start

        for idx in range(1, self.n):
            candidates = self.candidate_lists[current]
            allowed = candidates[unvisited[candidates]]
            if allowed.size == 0:
                allowed = np.where(unvisited)[0]

            pheromone = tau[current, allowed] ** alpha
            heuristic = self.eta[current, allowed] ** beta
            desirability = pheromone * heuristic

            if desirability.sum() <= 0:
                next_city = int(self.rng.choice(allowed))
            else:
                if self.rng.random() < q0:
                    next_city = int(allowed[np.argmax(desirability)])
                else:
                    prob = desirability / desirability.sum()
                    next_city = int(self.rng.choice(allowed, p=prob))

            tour[idx] = next_city
            unvisited[next_city] = False
            current = next_city

        return tour

    def _deposit(self, tau: np.ndarray, tour: np.ndarray, cost: float) -> None:
        delta = 1.0 / max(cost, 1e-12)
        rolled = np.roll(tour, -1)
        tau[tour, rolled] += delta
        tau[rolled, tour] += delta

    def solve(self, initial_tour: np.ndarray | None = None) -> Dict[str, object]:
        n = self.n
        cfg = self.config
        initial_cost = tour_cost(initial_tour, self.dist_matrix) if initial_tour is not None else None
        nn_cost = initial_cost if initial_cost is not None else float(np.mean(self.dist_matrix)) * n

        tau_max = 1.0 / max(cfg.rho_start * nn_cost, 1e-12)
        tau_min = tau_max / (2.0 * n)
        tau = np.full((n, n), tau_max, dtype=np.float64)
        np.fill_diagonal(tau, 0.0)

        best_tour = initial_tour.copy() if initial_tour is not None else None
        best_cost = initial_cost if initial_cost is not None else float("inf")
        convergence: List[float] = []
        no_improve = 0
        no_improve_restart = 0

        for iteration in range(cfg.max_iter):
            rho_t = linear_schedule(cfg.rho_start, cfg.rho_end, iteration, cfg.max_iter)
            q0_t = linear_schedule(cfg.q0_start, cfg.q0_end, iteration, cfg.max_iter)

            tours = []
            costs = []
            for _ in range(cfg.ants):
                tour = self._construct_solution(tau, cfg.alpha, cfg.beta, q0_t)
                cost = tour_cost(tour, self.dist_matrix)
                tours.append(tour)
                costs.append(cost)

            order = np.argsort(costs)

            if cfg.use_local_search and iteration % cfg.ls_frequency == 0:
                for idx in order[: cfg.ls_top_ants]:
                    improved_tour, improved_cost = two_opt_first_improvement(
                        tours[idx],
                        self.dist_matrix,
                        candidate_lists=self.ls_candidate_lists,
                    )
                    tours[idx] = improved_tour
                    costs[idx] = improved_cost
                order = np.argsort(costs)

            iter_best_tour = tours[int(order[0])]
            iter_best_cost = float(costs[int(order[0])])

            if iter_best_cost < best_cost:
                best_cost = iter_best_cost
                best_tour = iter_best_tour.copy()
                no_improve = 0
                no_improve_restart = 0
            else:
                no_improve += 1
                no_improve_restart += 1

            tau *= (1.0 - rho_t)
            self._deposit(tau, best_tour if best_tour is not None else iter_best_tour, best_cost)

            tau_max = 1.0 / max(rho_t * max(best_cost, 1.0), 1e-12)
            tau_min = tau_max / (2.0 * n)
            np.clip(tau, tau_min, tau_max, out=tau)
            np.fill_diagonal(tau, 0.0)

            if no_improve_restart >= cfg.restart_after:
                tau.fill(tau_max)
                np.fill_diagonal(tau, 0.0)
                self._deposit(tau, best_tour if best_tour is not None else iter_best_tour, best_cost)
                np.clip(tau, tau_min, tau_max, out=tau)
                np.fill_diagonal(tau, 0.0)
                no_improve_restart = 0

            convergence.append(best_cost)
            if no_improve >= cfg.stagnation_limit:
                break

        return {
            "best_tour": best_tour,
            "best_cost": float(best_cost),
            "convergence": convergence,
            "effective_iterations": len(convergence),
            "config": cfg.to_dict(),
        }


def solve_mmas(
    dist_matrix: np.ndarray,
    config: AdaptiveConfig,
    initial_tour: np.ndarray | None = None,
) -> Dict[str, object]:
    solver = MMASSolver(dist_matrix=dist_matrix, config=config)
    return solver.solve(initial_tour=initial_tour)
