# 🐜 TSP Master Project — Adaptive MMAS + Hybrid 2-opt + Parameter Tuning

This project is a **professional Python implementation** for solving the **Traveling Salesman Problem (TSP)** using **TSPLIB coordinate instances**.

The algorithm combines:

- **MAX-MIN Ant System (MMAS)**
- **Hybrid local search (restricted 2-opt)**
- **Adaptive parameter strategies**
- **Candidate lists for performance**

The project also includes a **benchmarking and parameter tuning framework** to evaluate performance across multiple TSPLIB instances.

---

# 📑 Table of Contents

- Overview
- Main Features
- Project Structure
- Installation
- Dataset Setup
- Running the Benchmark
- Parameter Tuning
- Benchmark Results
- Experimental Analysis
- Parameter Influence
- Scientific Interpretation
- Notes
- Quick Test

---

# 🚀 Overview

The **Traveling Salesman Problem (TSP)** is one of the most studied **NP-Hard combinatorial optimization problems**.

Given a set of cities and distances between them, the goal is to:

> Find the shortest possible tour that visits each city exactly once and returns to the starting city.

This project implements a **hybrid Ant Colony Optimization algorithm** with improvements designed to increase solution quality and reduce computation time.

---

# ✨ Main Features

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

---

# 📂 Project Structure

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

---

# ⚙️ Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate environment

Linux / macOS

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Put your TSPLIB files here

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

---

# ▶️ Run the Benchmark

```bash
python run_benchmark.py --data-dir data/tsplib --output-dir outputs/benchmark
```

Outputs:

- `benchmark_results.csv`
- `benchmark_results.md`
- `benchmark_results.json`
- convergence plots
- best tour plots

---

# ⚙️ Tune Parameters for One Instance

Example:

```bash
python tune_parameters.py --instance data/tsplib/berlin52.tsp --output-dir outputs/tuning --runs 5 --max-combinations 30
```

This performs repeated stochastic runs and selects parameters using:

1. lowest mean cost  
2. lowest standard deviation  
3. lowest mean execution time  

Outputs:

- `*_tuning_results.csv`
- `*_best_config.json`
- `*_tuning_summary.json`
- parameter impact plots
- cost-time tradeoff plot
- top configurations boxplot

---

# 📊 Benchmark Results

| Instance | Size | Greedy | Pure ACO | Hybrid ACO | Time (s) | Best Known | Hybrid Gap |
|------|------|------|------|------|------|------|------|
| berlin52 | 52 | 8181 | 7727 | **7542** | 3.6 | 7542 | **0.0%** |
| eil51 | 51 | 482 | 434 | **429** | 4.1 | 426 | 0.70% |
| st70 | 70 | 796 | 695 | **686** | 9.5 | 675 | 1.63% |
| kroA100 | 100 | 24698 | 21541 | **21388** | 24.4 | 21282 | 0.49% |
| kroB100 | 100 | 25884 | 22560 | **22358** | 22.9 | 22141 | 0.98% |
| rat195 | 195 | 2612 | 2363 | **2357** | 125 | 2323 | 1.46% |
| ch150 | 150 | 7113 | **6590** | 6617 | 50.5 | 6528 | 1.36% |
| d493 | 493 | 40186 | 37204 | **36558** | 1122 | 35002 | 4.44% |
| u574 | 574 | 45440 | 40840 | **38652** | 1159 | 36905 | 4.73% |

---

# 🔬 Experimental Analysis

The results show that the algorithm **consistently improves the greedy initialization**.

Average improvements:

| Metric | Average |
|------|------|
| Pure ACO improvement | **≈ 9.8%** |
| Hybrid ACO improvement | **≈ 11.2%** |

This demonstrates the effectiveness of combining **Ant Colony Optimization with local search**.

---

# 📈 Effect of Hybridization (2-opt)

In most instances, **Hybrid ACO outperforms Pure ACO**.

Examples:

| Instance | Pure ACO | Hybrid ACO |
|------|------|------|
| berlin52 | 7727 | **7542** |
| kroA100 | 21541 | **21388** |
| u574 | 40840 | **38652** |

The hybrid version improves tours by removing inefficient edges using **2-opt edge exchanges**.

---

# ⏱ Execution Time Behavior

Execution time increases with instance size.

| Size | Example | Runtime |
|------|------|------|
| Small (~50 cities) | berlin52 | ~3 s |
| Medium (~100 cities) | kroA100 | ~24 s |
| Large (~500 cities) | u574 | ~1150 s |

This reflects the exponential growth of the TSP search space.

---

# ⚙️ Parameter Influence

### α — pheromone importance

Higher α increases the influence of pheromone trails.

Effects:

- faster convergence
- higher risk of stagnation

### β — heuristic importance

Higher β prioritizes shorter edges.

Effects:

- faster greedy solutions
- less exploration

### Evaporation rate

Controls how quickly pheromone information disappears.

Low evaporation:

- strong memory
- risk of premature convergence

High evaporation:

- more exploration
- slower convergence

---

# 🧪 Scientific Interpretation

The experiments confirm that:

- ACO is a **stochastic metaheuristic**
- multiple runs are necessary for reliable evaluation
- performance should be evaluated using **average cost and optimality gap**

This ensures that the results are **statistically meaningful and reproducible**.

---

# 📝 Notes

- Supports `EUC_2D` and `CEIL_2D` TSPLIB coordinate instances.
- Large instances require significant runtime.
- Candidate lists and restricted 2-opt reduce computational cost.

---

# ⚡ Quick Test

```bash
python run_benchmark.py --data-dir data/tsplib --pattern test5.tsp --output-dir outputs/test
```

---

