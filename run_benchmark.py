from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tsp_master.solver import solve_instance
from tsp_master.tuning import BEST_KNOWN
from tsp_master.utils import ensure_dir
from tsp_master.visualization import plot_convergence, plot_tour


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TSP benchmark on TSPLIB instances.")
    parser.add_argument("--data-dir", type=str, default="data/tsplib")
    parser.add_argument("--output-dir", type=str, default="outputs/benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pattern", type=str, default="*.tsp")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    output_dir = ensure_dir(ROOT / args.output_dir)
    plots_dir = ensure_dir(output_dir / "plots")

    rows = []
    for file_path in sorted(data_dir.glob(args.pattern)):
        start = time.perf_counter()
        result = solve_instance(str(file_path), seed=args.seed)
        elapsed = time.perf_counter() - start

        instance = result["instance"]
        name = instance["name"]
        size = instance["dimension"]
        init = result["initialization"]
        pure = result["pure_aco"]
        hybrid = result["hybrid_aco"]

        plot_convergence(
            pure_curve=pure["convergence"],
            hybrid_curve=hybrid["convergence"],
            instance_name=name,
            output_path=plots_dir / f"{name}_convergence.png",
        )
        plot_tour(
            coords=instance["coords"],
            tour=hybrid["best_tour"],
            instance_name=name,
            output_path=plots_dir / f"{name}_best_tour.png",
        )

        greedy_cost = float(init["greedy_cost"])
        pure_cost = float(pure["best_cost"])
        hybrid_cost = float(hybrid["best_cost"])
        pure_gain = 100.0 * (greedy_cost - pure_cost) / greedy_cost
        hybrid_gain = 100.0 * (greedy_cost - hybrid_cost) / greedy_cost
        optimum = BEST_KNOWN.get(name)
        pure_gap = None if optimum is None else 100.0 * (pure_cost - optimum) / optimum
        hybrid_gap = None if optimum is None else 100.0 * (hybrid_cost - optimum) / optimum

        rows.append(
            {
                "Instance Name": name,
                "Size": size,
                "Greedy Cost": greedy_cost,
                "Selected Init": init["selected_name"],
                "Pure ACO Cost": pure_cost,
                "Hybrid ACO Cost": hybrid_cost,
                "Execution Time (s)": round(elapsed, 3),
                "Pure Gain (%)": round(pure_gain, 3),
                "Hybrid Gain (%)": round(hybrid_gain, 3),
                "Best Known": optimum,
                "Pure Gap (%)": None if pure_gap is None else round(pure_gap, 3),
                "Hybrid Gap (%)": None if hybrid_gap is None else round(hybrid_gap, 3),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = output_dir / "benchmark_results.csv"
    md_path = output_dir / "benchmark_results.md"
    json_path = output_dir / "benchmark_results.json"
    df.to_csv(csv_path, index=False)
    df.to_markdown(md_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    print(df.to_string(index=False))
    print(f"\nSaved benchmark table to: {csv_path}")


if __name__ == "__main__":
    main()
