from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tsp_master.tuning import tune_instance
from tsp_master.utils import ensure_dir
from tsp_master.visualization import (
    plot_cost_time_tradeoff,
    plot_parameter_impact,
    plot_top_configs_boxplot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune MMAS parameters for one TSPLIB instance.")
    parser.add_argument("--instance", type=str, required=True, help="Path to a .tsp file")
    parser.add_argument("--output-dir", type=str, default="outputs/tuning")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-combinations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = ensure_dir(ROOT / args.output_dir)
    result = tune_instance(
        file_path=str(ROOT / args.instance) if not Path(args.instance).is_absolute() else args.instance,
        output_dir=output_dir,
        n_runs=args.runs,
        max_combinations=args.max_combinations,
        seed=args.seed,
    )

    name = result["instance_name"]
    df = result["dataframe"]
    best = result["best"]

    plot_parameter_impact(df, "beta", "mean_cost", output_dir / f"{name}_impact_beta_mean_cost.png")
    plot_parameter_impact(df, "rho_start", "mean_cost", output_dir / f"{name}_impact_rho_mean_cost.png")
    plot_cost_time_tradeoff(df, output_dir / f"{name}_cost_time_tradeoff.png")
    plot_top_configs_boxplot(result["all_results"], output_dir / f"{name}_top_configs_boxplot.png")

    summary = {
        "instance": name,
        "best_config": best["config"],
        "best_mean_cost": best["mean_cost"],
        "best_best_cost": best["best_cost"],
        "best_std_cost": best["std_cost"],
        "best_mean_time": best["mean_time"],
        "optimum": result["optimum"],
    }
    with (output_dir / f"{name}_tuning_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Best configuration found:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
