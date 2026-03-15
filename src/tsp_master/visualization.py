from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(
    pure_curve: Sequence[float],
    hybrid_curve: Sequence[float],
    instance_name: str,
    output_path: str | Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(pure_curve, label="Pure MMAS")
    plt.plot(hybrid_curve, label="Hybrid MMAS + 2-opt")
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.title(f"Convergence - {instance_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()



def plot_tour(coords: np.ndarray, tour: Iterable[int], instance_name: str, output_path: str | Path) -> None:
    route = np.asarray(list(tour), dtype=np.int32)
    closed = np.append(route, route[0])
    xy = coords[closed]

    plt.figure(figsize=(7, 7))
    plt.plot(xy[:, 0], xy[:, 1], marker="o", markersize=3, linewidth=1)
    plt.title(f"Best Tour - {instance_name}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()



def plot_parameter_impact(df, parameter_name: str, metric: str, output_path: str | Path) -> None:
    grouped = df.groupby(parameter_name)[metric].mean().reset_index()
    plt.figure(figsize=(8, 5))
    plt.plot(grouped[parameter_name], grouped[metric], marker="o")
    plt.xlabel(parameter_name)
    plt.ylabel(metric)
    plt.title(f"Impact of {parameter_name} on {metric}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()



def plot_cost_time_tradeoff(df, output_path: str | Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(df["mean_time"], df["mean_cost"])
    for _, row in df.iterrows():
        plt.annotate(
            f"β={row['beta']}, ρ={row.get('rho_start', row.get('rho', ''))}",
            (row["mean_time"], row["mean_cost"]),
            fontsize=7,
        )
    plt.xlabel("Mean Time (s)")
    plt.ylabel("Mean Cost")
    plt.title("Cost-Time Tradeoff")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()



def plot_top_configs_boxplot(results: list[dict], output_path: str | Path, top_k: int = 5) -> None:
    sorted_results = sorted(results, key=lambda r: (r["mean_cost"], r["std_cost"], r["mean_time"]))
    selected = sorted_results[:top_k]
    data = []
    labels = []
    for idx, result in enumerate(selected, start=1):
        data.append([run["cost"] for run in result["all_runs"]])
        labels.append(f"Cfg {idx}")

    plt.figure(figsize=(9, 5))
    plt.boxplot(data, tick_labels=labels)
    plt.ylabel("Tour Cost")
    plt.title("Top Configurations Stability")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
