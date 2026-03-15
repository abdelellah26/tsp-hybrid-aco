🐜 TSP Master Project — Adaptive MMAS + Hybrid 2-opt + Parameter Tuning

A professional Python implementation for solving the Traveling Salesman Problem (TSP) using Ant Colony Optimization (ACO) combined with local search (2-opt) and adaptive parameter strategies.

The project is designed for TSPLIB coordinate instances and includes a complete benchmarking and parameter tuning framework.

📑 Table of Contents

Overview

Main Features

Project Structure

Installation

Dataset Setup

Running the Benchmark

Parameter Tuning

Benchmark Results

Experimental Analysis

Parameter Influence

Scientific Interpretation

Notes

Quick Test

🚀 Overview

The Traveling Salesman Problem (TSP) is one of the most studied NP-Hard combinatorial optimization problems.

Given a set of cities and distances between them, the goal is to:

Find the shortest possible tour that visits each city exactly once and returns to the starting city.

This project implements an advanced Hybrid Ant Colony Optimization algorithm based on:

MAX-MIN Ant System (MMAS)

Restricted 2-opt local search

Adaptive parameters based on instance size

Candidate lists to reduce complexity

Stagnation detection and restart mechanisms

✨ Main Features

TSPLIB parser with NODE_COORD_SECTION

NumPy distance matrix computation

Multi-Start Nearest Neighbor (MSNN) from every city

Random initialization baseline

Automatic selection of the best initial solution

Pure MAX-MIN Ant System (MMAS)

Hybrid MMAS + restricted 2-opt

Dynamic parameters based on instance size

Candidate lists to reduce execution time

Stagnation + restart strategy

Parameter tuning framework with repeated runs

Comparison tables, convergence curves, best-tour plots, tuning graphs

📂 Project Structure
.
├── data/tsplib/
├── outputs/
├── src/tsp_master/
├── run_benchmark.py
├── tune_parameters.py
├── requirements.txt
└── README.md
⚙️ Installation

Create a virtual environment:

python -m venv .venv

Activate environment

Linux / macOS

source .venv/bin/activate

Windows

.venv\Scripts\activate

Install dependencies

pip install -r requirements.txt
📦 Put your TSPLIB files here
data/tsplib/

Recommended instances:

eil51.tsp

berlin52.tsp

st70.tsp

eil76.tsp

kroA100.tsp

kroB100.tsp

ch150.tsp

rat195.tsp

d493.tsp

u574.tsp

▶️ Run the Benchmark
python run_benchmark.py --data-dir data/tsplib --output-dir outputs/benchmark

Outputs generated:

benchmark_results.csv

benchmark_results.md

benchmark_results.json

convergence plots

best tour plots

⚙️ Tune Parameters for One Instance

Example:

python tune_parameters.py --instance data/tsplib/berlin52.tsp --output-dir outputs/tuning --runs 5 --max-combinations 30

The tuning process evaluates configurations using:

lowest mean cost

lowest standard deviation

lowest mean execution time

Outputs:

*_tuning_results.csv

*_best_config.json

*_tuning_summary.json

parameter impact plots

cost-time tradeoff plot

top configurations boxplot

📊 Benchmark Results
Instance	Size	Greedy	Pure ACO	Hybrid ACO	Time (s)	Best Known	Hybrid Gap
berlin52	52	8181	7727	7542	3.6	7542	0.0%
eil51	51	482	434	429	4.1	426	0.70%
st70	70	796	695	686	9.5	675	1.63%
kroA100	100	24698	21541	21388	24.4	21282	0.49%
kroB100	100	25884	22560	22358	22.9	22141	0.98%
rat195	195	2612	2363	2357	125	2323	1.46%
ch150	150	7113	6590	6617	50.5	6528	1.36%
d493	493	40186	37204	36558	1122	35002	4.44%
u574	574	45440	40840	38652	1159	36905	4.73%
🔬 Experimental Analysis

The results clearly demonstrate that the ACO algorithm significantly improves the greedy initialization.

Average improvements observed:

Metric	Average
Pure ACO improvement	≈ 9.5%
Hybrid ACO improvement	≈ 11.0%

This shows that combining ACO with local search produces better solutions.

📈 Effect of Hybridization (2-opt)

In most instances, the Hybrid ACO outperforms the Pure ACO.

Examples:

Instance	Pure ACO	Hybrid ACO
berlin52	7727	7542
kroA100	21541	21388
u574	40840	38652

The hybrid version improves tours by removing inefficient edges through 2-opt edge exchanges.

⏱ Execution Time Behavior

Execution time increases with the number of cities.

Size	Example	Runtime
Small (~50)	berlin52	~3 s
Medium (~100)	kroA100	~24 s
Large (~500)	u574	~1150 s

This confirms the exponential growth of the TSP search space.

⚙️ Parameter Influence

The performance of ACO depends strongly on its parameters.

α — pheromone importance

Higher α increases exploitation of pheromone trails.

Effect:

faster convergence

higher risk of stagnation

β — heuristic importance

Higher β favors short edges.

Effect:

faster greedy-like solutions

less exploration

Evaporation rate

Controls how fast pheromone disappears.

Low evaporation:

strong memory

risk of premature convergence

High evaporation:

more exploration

slower convergence

🧪 Scientific Interpretation

The experiments confirm that:

ACO is a stochastic metaheuristic

multiple runs are required for reliable evaluation

results must be interpreted using average performance and optimality gaps

This methodology ensures that the results are statistically meaningful.

📝 Notes

Supports EUC_2D and CEIL_2D TSPLIB instances.

Large instances require significant runtime.

Candidate lists and restricted 2-opt reduce computation cost.

⚡ Quick Test

To test the system quickly:

python run_benchmark.py --data-dir data/tsplib --pattern test5.tsp --output-dir outputs/test
👨‍💻 Author

Project developed for research and experimentation in metaheuristic optimization.
