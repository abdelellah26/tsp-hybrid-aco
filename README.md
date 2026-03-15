# TSP Master Project — Adaptive MMAS + Hybrid 2-opt + Parameter Tuning

This project is a professional Python implementation for the Traveling Salesman Problem (TSP) using TSPLIB coordinate instances.

## Main features

- TSPLIB parser with `NODE_COORD_SECTION`
- NumPy distance matrix computation
- Multi-Start Nearest Neighbor (MSNN) from **every city**
- Random initialization baseline
- Automatic selection of the best initial solution
- Pure **MAX-MIN Ant System (MMAS)**
- Hybrid **MMAS + restricted 2-opt**
- Dynamic parameters based on instance size
- Candidate lists to reduce execution time
- Stagnation + restart strategy
- Parameter tuning framework with repeated runs
- Comparison tables, convergence curves, best-tour plots, tuning graphs

## Project tree

```text
.
├── data/tsplib/
├── outputs/
├── src/tsp_master/
├── run_benchmark.py
├── tune_parameters.py
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate     # Linux / macOS
# .venv\Scripts\activate      # Windows PowerShell
pip install -r requirements.txt
```

## Put your TSPLIB files here

```text
data/tsplib/
```

Recommended instances:

- `eil51.tsp`
- `berlin52.tsp`
- `st70.tsp`
- `eil76.tsp`
- `kroA100.tsp`
- `kroB100.tsp`
- `ch150.tsp`
- `rat195.tsp`
- `d493.tsp`
- `u574.tsp`

## Run the benchmark

```bash
python run_benchmark.py --data-dir data/tsplib --output-dir outputs/benchmark
```

Outputs:

- `benchmark_results.csv`
- `benchmark_results.md`
- `benchmark_results.json`
- convergence plots
- best tour plots

## Tune parameters for one instance

Example:

```bash
python tune_parameters.py --instance data/tsplib/berlin52.tsp --output-dir outputs/tuning --runs 5 --max-combinations 30
```

This performs repeated stochastic runs and selects parameters using:

1. lowest mean cost
2. then lowest standard deviation
3. then lowest mean execution time

Outputs:

- `*_tuning_results.csv`
- `*_best_config.json`
- `*_tuning_summary.json`
- parameter impact plots
- cost-time tradeoff plot
- top configurations boxplot

## Scientific use in your report

You can explain that:

- ACO is stochastic, so one run is not enough.
- Each parameter configuration is evaluated over multiple independent runs.
- The selected configuration is chosen based on **average performance and robustness**, not a lucky single run.
- Dynamic parameters are justified by instance size: a 51-city instance is not treated like a 574-city instance.

## Notes

- This implementation supports `EUC_2D` and `CEIL_2D` TSPLIB coordinate instances.
- Large instances still require time in pure Python; candidate lists and selective 2-opt are included to reduce runtime.
- The project includes a small sample instance for a quick sanity test.

## Quick test

```bash
python run_benchmark.py --data-dir data/tsplib --pattern test5.tsp --output-dir outputs/test
```
