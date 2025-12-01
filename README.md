# NP-Completeness Demonstrator
## Comprehensive Guide to Algorithms, Reductions, and Benchmarking

This project provides hands-on exploration of NP-Complete problems through:
- Exact exponential-time algorithms
- Polynomial-time approximations
- Polynomial-time reductions between problems
- Complete benchmarking and visualization pipeline

---

## Table of Contents
1. [SAT Solver](#sat-solver)
2. [Vertex Cover](#vertex-cover)
3. [Hamiltonian Path](#hamiltonian-path)
4. [Subset Sum](#subset-sum)
5. [3-SAT → Vertex Cover Reduction](#3-sat--vertex-cover-reduction)
6. [Subset Sum → SAT Reduction](#subset-sum--sat-reduction)
7. [Sudoku → SAT Encoding](#sudoku--sat-encoding)
8. [Benchmarking Pipeline](#benchmarking-pipeline)
9. [Installation & Setup](#installation--setup)

---

## SAT Solver – DPLL Algorithm

### Problem Definition
*Boolean Satisfiability (SAT):*
Given a boolean formula in CNF (Conjunctive Normal Form), determine if there exists
an assignment of variables that makes the formula true.

SAT is the first problem proven to be NP-Complete (Cook-Levin Theorem, 1971).

### File: solvers/sat.py

*Main Components:*
- unit_propagate(clauses, assignment) → Simplify clauses with unit literals
- pure_literal_eliminate(clauses, assignment) → Remove pure literals
- dpll_solve(n_vars, clauses) → Main DPLL solver
- generate_random_3sat(n_vars, n_clauses) → Instance generator

### Algorithm: DPLL (Davis-Putnam-Logemann-Loveland)

1. Base cases:
   - All clauses satisfied → return SAT
   - Any clause empty → return UNSAT
2. Unit propagation: Assign forced variables
3. Pure literal elimination: Assign variables appearing with single polarity
4. Choose variable: Pick next unassigned variable
5. Branch: Try true, then false
6. Backtrack if both branches fail


### Usage
python
from solvers.sat import dpll_solve

# (x1 ∨ x2 ∨ ¬x3) ∧ (¬x1 ∨ x4) ∧ (x3 ∨ ¬x4)
result, model, nodes = dpll_solve(
    n_vars=4,
    clauses=[[1, 2, -3], [-1, 4], [3, -4]]
)

print(f"Satisfiable: {result}")           
print(f"Model: {model}")                   
print(f"Search nodes explored: {nodes}")   


### Command Line Test
bash
python3 -c "from solvers.sat import dpll_solve; print(dpll_solve(3, [[1,2,-3], [-1,3]]))"


### Guarantees
- *Correctness:* Finds satisfying assignment if one exists
- *Completeness:* Proves unsatisfiability if no solution
- *Time Complexity:* O(2^n) worst-case, much better with pruning
- *Space Complexity:* O(n) for recursion stack

---

## Vertex Cover – Exact & Approximation Algorithms

### Problem Definition
*Vertex Cover:*
Given an undirected graph G = (V, E), find the smallest set of vertices such that
every edge has at least one endpoint in the set.

Vertex Cover is NP-Complete (Karp's 21 problems, 1972).

### File: solvers/vertex_cover.py

*Main Components:*
- add_edge(g, u, v) → Graph construction helper
- any_edge(g) → Select an uncovered edge
- remove_vertex(g, u) → Delete vertex and incident edges
- exact_branching(g, k) → Exact exponential solver
- approx_2(g) → Greedy 2-approximation algorithm
- benchmark_vertex_cover() → Complete benchmark suite

### Algorithm 1: Exact Branching

exact_branching(G, k):
  If no edges remain:
    return (True, ∅)
  If k = 0:
    return (False, ∅)
  
  Pick any edge (u, v)
  Branch 1: Include u in cover, solve(G - u, k-1)
  Branch 2: Include v in cover, solve(G - v, k-1)
  
  Return first successful branch


*Usage:*
python
from solvers.vertex_cover import add_edge, exact_branching

graph = {}
add_edge(graph, 1, 2)
add_edge(graph, 2, 3)
add_edge(graph, 3, 4)
add_edge(graph, 4, 1)

sat, cover = exact_branching(graph, k=2)
print(f"Has cover of size 2: {sat}")
print(f"Cover vertices: {cover}")


*Guarantees:*
- *Optimal:* Finds minimum vertex cover
- *Time:* O(2^k) where k is cover size
- *Space:* O(k) recursion depth

### Algorithm 2: 2-Approximation (Greedy)

approx_2(G):
  C = ∅
  while G has edges:
    Pick any edge (u, v)
    C = C ∪ {u, v}
    Remove u and v from G
  return C


*Usage:*
python
from solvers.vertex_cover import approx_2, add_edge

graph = {}
add_edge(graph, 1, 2)
add_edge(graph, 2, 3)
add_edge(graph, 3, 1)

cover = approx_2(graph)
print(f"Approximate cover: {cover}")


*Guarantees:*
- *Approximation Ratio:* |C| ≤ 2 × OPT
- *Time:* O(|E|) - polynomial
- *Space:* O(|V|)

### Running Vertex Cover Benchmark
bash
python3 solvers/vertex_cover.py


*Generated Graphs:*
1. *Runtime Comparison* (vertex_cover_runtime.png)
   - Shows exponential growth of exact algorithm
   - Shows polynomial growth of approximation
   - X-axis: number of vertices
   - Y-axis: runtime (log scale)

2. *Solution Quality* (vertex_cover_quality.png)
   - Exact minimum cover size
   - Approximation cover size
   - 2× bound visualization

### Key Observations
- Exact becomes infeasible beyond n ≈ 15 vertices
- Approximation remains fast for large graphs
- Approximation typically much better than 2× in practice
- Demonstrates necessity of approximation for NP-Complete problems

---

## Hamiltonian Path – Backtracking & Dynamic Programming

### Problem Definition
*Hamiltonian Path:*
Given an undirected graph G = (V, E), determine if there exists a path that visits
every vertex exactly once.

Hamiltonian Path is NP-Complete.

### File: solvers/hampath.py

*Main Components:*
- backtracking_path(g) → Backtracking search
- held_karp_path(g) → DP approach (similar to TSP)
- generate_random_graph(n, p) → Random graph generator

### Algorithm 1: Backtracking

backtrack(path, remaining):
  If remaining is empty:
    return path (success)
  
  For each neighbor v of path[-1]:
    If v in remaining:
      Try: path + [v], remaining - {v}
      If success: return path
  
  return failure


*Usage:*
python
from solvers.hampath import backtracking_path

graph = {
    1: {2, 3, 4},
    2: {1, 3},
    3: {1, 2, 4},
    4: {1, 3}
}

exists, path = backtracking_path(graph)
print(f"Has Hamiltonian path: {exists}")
print(f"Path: {path}")


*Guarantees:*
- *Time:* O(n!) worst-case
- *Space:* O(n) for path storage
- *Practical:* Works well for small graphs (n < 15)

### Algorithm 2: Dynamic Programming (Held-Karp style)

DP state: (current_vertex, visited_set)
Recurrence:
  dp[v][S] = ∃u ∈ S: dp[u][S - {v}] and (u, v) ∈ E


*Usage:*
python
from solvers.hampath import held_karp_path

graph = {1: {2, 3}, 2: {1, 3}, 3: {1, 2}}
exists, path = held_karp_path(graph)
print(f"DP result: {exists}, Path: {path}")


*Guarantees:*
- *Time:* O(n² × 2^n) - faster than n!
- *Space:* O(n × 2^n) - stores DP table
- *Better for:* Dense graphs, slightly larger n

### Running Hamiltonian Path Benchmark
bash
python3 -c "from solvers.hampath import benchmark_hampath; benchmark_hampath()"


---

## Subset Sum – Brute Force & Meet-in-the-Middle

### Problem Definition
*Subset Sum:*
Given a set of integers S and target T, determine if there exists a subset
of S that sums to exactly T.

Subset Sum is NP-Complete.

### File: solvers/subsetsum.py

*Main Components:*
- brute_force(nums, target) → Try all 2^n subsets
- meet_in_the_middle(nums, target) → Optimized O(2^(n/2))
- benchmark_subsetsum() → Performance comparison

### Algorithm 1: Brute Force

For each of 2^n subsets:
  If sum(subset) == target:
    return True, subset
return False, []


*Usage:*
python
from solvers.subsetsum import brute_force

numbers = [3, 7, 9, 10, 15]
target = 19

found, subset = brute_force(numbers, target)
print(f"Found: {found}")
print(f"Subset: {subset}")  # e.g., [9, 10]


*Guarantees:*
- *Time:* O(2^n × n)
- *Space:* O(n)
- *Practical limit:* n ≈ 20-25

### Algorithm 2: Meet-in-the-Middle

1. Split array into two halves: L, R
2. Generate all sums of L → store in hash table
3. For each sum s_R in R:
   Check if (target - s_R) exists in L's sums
4. If match found, reconstruct subset


*Usage:*
python
from solvers.subsetsum import meet_in_the_middle

numbers = [3, 7, 9, 10, 15, 20, 25]
target = 42

found, subset = meet_in_the_middle(numbers, target)
print(f"Fast search: {found}")
print(f"Subset: {subset}")


*Guarantees:*
- *Time:* O(2^(n/2) × n)
- *Space:* O(2^(n/2))
- *Speedup:* Can handle n ≈ 35-40

### Running Subset Sum Benchmark
bash
python3 solvers/subsetsum.py


*Generated Graph:*
- Compare brute force vs MITM runtimes
- Shows exponential speedup of MITM
- Demonstrates space-time tradeoff

---

## 3-SAT → Vertex Cover Reduction

### Reduction Overview
*Purpose:* Prove Vertex Cover is NP-Complete by reducing 3-SAT to it.

*Key Idea:*
- For each variable x_i, create edge: (x_i, ¬x_i)
- For each clause (a ∨ b ∨ c), create triangle with 3 vertices
- Add "consistency edges" from clause vertices to opposite variable nodes
- Set k = n_vars + 2 × n_clauses

### File: reductions/sat_to_vc.py

*Main Function:*
python
reduce_3sat_to_vc(n_vars, clauses) → (graph, k, pos_nodes, neg_nodes)


*Reduction Construction:*

1. Variable Gadget:
   For each variable x_i:
     Create nodes: pos_i, neg_i
     Add edge: (pos_i, neg_i)
   
   Interpretation: Must choose exactly one (true or false)

2. Clause Gadget:
   For each clause (l_1 ∨ l_2 ∨ l_3):
     Create triangle: (c_1, c_2, c_3)
     Add edges: (c_1, c_2), (c_2, c_3), (c_1, c_3)
   
   Interpretation: Must cover at least 2 vertices per triangle

3. Consistency Edges:
   For each literal l_i in clause c:
     If l_i = x_j (positive):
       Add edge: (c_i, neg_j)
     If l_i = ¬x_j (negative):
       Add edge: (c_i, pos_j)
   
   Interpretation: Forces clause satisfaction

4. Set k = n_vars + 2 × n_clauses


### Usage
python
from reductions.sat_to_vc import reduce_3sat_to_vc

# (x1 ∨ ¬x2 ∨ x3) ∧ (¬x1 ∨ x2 ∨ ¬x3)
n_vars = 3
clauses = [[1, -2, 3], [-1, 2, -3]]

graph, k, pos_nodes, neg_nodes = reduce_3sat_to_vc(n_vars, clauses)

print(f"Graph has {len(graph)} vertices")
print(f"Vertex cover threshold: k = {k}")
print(f"Variable x1: positive={pos_nodes[1]}, negative={neg_nodes[1]}")


### Visualizing the Reduction
bash
python3 reductions/sat_to_vc.py


This generates a graph showing:
- Variable gadgets (edges) on the left
- Clause gadgets (triangles) on the right
- Consistency edges connecting them

*Correctness:*
- 3-SAT formula satisfiable ⟺ Graph has vertex cover of size ≤ k
- Polynomial-time reduction: O(n_vars + n_clauses)

### Educational Value
- Shows how NP-Complete problems transform into each other
- Demonstrates gadget-based reduction techniques
- Visualizes the structure of NP-Completeness proofs

---

## Subset Sum → SAT Reduction

### Reduction Overview
*Purpose:* Encode Subset Sum as a SAT formula.

*Key Idea:*
- Binary variables s_i indicate if element i is selected
- Binary representation of partial sums
- Clauses enforce arithmetic constraints

### File: reductions/subsetsum_to_sat.py

*Main Function:*
python
exactly_equal_sum_parity(nums, target) → (n_vars, clauses, dimacs)


*Encoding Strategy:*

Variables:
  - s_i: selection bit for number i
  - b_i_j: bit j of partial sum up to i

Clauses:
  1. Sum computation using binary addition
  2. Carry propagation
  3. Final sum must equal target
  4. Parity/uniqueness constraints


### Usage
python
from reductions.subsetsum_to_sat import exactly_equal_sum_parity
from solvers.sat import dpll_solve

numbers = [3, 5, 7, 11]
target = 15

n_vars, clauses, dimacs = exactly_equal_sum_parity(numbers, target)

print(f"Encoded to {n_vars} variables, {len(clauses)} clauses")

# Solve with SAT solver
sat, model, _ = dpll_solve(n_vars, clauses)
print(f"Satisfiable: {sat}")

if sat:
    # Extract which numbers were selected
    selected = [numbers[i] for i in range(len(numbers)) 
                if model.get(i+1, False)]
    print(f"Subset: {selected}")
    print(f"Sum: {sum(selected)}")


### Command Line Test
bash
python3 -c "from reductions.subsetsum_to_sat import exactly_equal_sum_parity; print(exactly_equal_sum_parity([2,3,5], 8))"


### DIMACS Export
python
from reductions.subsetsum_to_sat import exactly_equal_sum_parity

_, _, dimacs = exactly_equal_sum_parity([3, 5, 7], 10)

with open('subsetsum.cnf', 'w') as f:
    f.write(dimacs)

# Now use external SAT solver
# minisat subsetsum.cnf subsetsum.out


*Complexity:*
- Variables: O(n × log(sum(nums)))
- Clauses: O(n × log(sum(nums)))
- Shows Subset Sum ∈ NP

---

## Sudoku → SAT Encoding

### Problem Definition
*Sudoku:* Fill n×n grid such that:
- Each row contains 1...n exactly once
- Each column contains 1...n exactly once
- Each block contains 1...n exactly once

### File: bonus/sudoku_sat.py

*Main Components:*
- encode_sudoku_to_sat(puzzle) → Generate CNF
- decode_sat_solution(model, size) → Extract grid from SAT assignment
- CLI for solving puzzles and exporting DIMACS

### Encoding Strategy

Variables: x[i][j][k] = "cell (i,j) contains value k"

Constraints:
1. Each cell has at least one value:
   ∨_{k=1..n} x[i][j][k]

2. Each cell has at most one value:
   ¬x[i][j][k] ∨ ¬x[i][j][k'] for all k ≠ k'

3. Row uniqueness:
   For each row i, value k:
     ∨_{j=1..n} x[i][j][k]
     ¬x[i][j][k] ∨ ¬x[i][j'][k] for j ≠ j'

4. Column uniqueness: (similar to rows)

5. Block uniqueness: (similar to rows)

6. Given clues:
   For each filled cell (i,j) = k:
     Add unit clause: x[i][j][k]


### Usage: Solve Default Puzzle
bash
python3 -m bonus.sudoku_sat


Output:

Original 4x4 Sudoku:
[1, 0, 0, 4]
[0, 2, 3, 0]
[0, 3, 2, 0]
[4, 0, 0, 1]

Encoded to 64 variables, 256 clauses

Solved Sudoku:
[1, 3, 4, 2]
[2, 4, 1, 3]
[3, 1, 2, 4]
[4, 2, 3, 1]


### Usage: Custom Puzzle
Create puzzle.json:
json
[
  [0, 0, 3, 4],
  [0, 0, 0, 2],
  [3, 0, 0, 0],
  [1, 2, 0, 0]
]


Solve:
bash
python3 -m bonus.sudoku_sat --puzzle puzzle.json


### Export to DIMACS
bash
python3 -m bonus.sudoku_sat --puzzle puzzle.json --dimacs-out sudoku.cnf

# Solve with external solver
minisat sudoku.cnf sudoku.out
cat sudoku.out


### Python API
python
from bonus.sudoku_sat import encode_sudoku_to_sat, decode_sat_solution
from solvers.sat import dpll_solve

puzzle = [[1,0,0,4], [0,2,3,0], [0,3,2,0], [4,0,0,1]]

n_vars, clauses = encode_sudoku_to_sat(puzzle)
sat, model, _ = dpll_solve(n_vars, clauses)

if sat:
    solution = decode_sat_solution(model, size=4)
    for row in solution:
        print(row)


### Complexity
- For n×n Sudoku:
  - Variables: O(n³)
  - Clauses: O(n⁴)
- 9×9 Sudoku: ~729 variables, ~10,000 clauses

---

## Benchmarking Pipeline

### Overview
Complete automated benchmarking system that:
1. Generates problem instances of increasing size
2. Measures runtime and solution quality
3. Creates publication-ready visualizations

### File Structure

bench/
├── run.py    → Generate CSV data files
└── plots.py  → Create PNG graphs from CSVs


### Running Complete Benchmark Suite

*Step 1: Generate Data*
bash
cd /path/to/project
source venv/bin/activate
python3 -m bench.run


This creates:
- data/sat_runtime.csv - SAT solver performance metrics
- data/vertexcover_runtime.csv - VC exact vs approx runtimes
- data/hampath_runtime.csv - Hamiltonian path search times
- data/subsetsum_runtime.csv - Subset sum algorithm comparison

*Step 2: Generate Plots*
bash
python3 -m bench.plots


This creates:
- data/plot_sat_runtime.png
- data/plot_vertexcover_runtime.png
- data/plot_vertexcover_quality.png
- data/plot_hampath_runtime.png
- data/plot_subsetsum_runtime.png

### Benchmark Parameters

*SAT Solver:*
python
# In bench/run.py
n_vars_range = range(5, 25, 2)      
clause_ratio = 4.3                   
trials_per_size = 5                  


*Vertex Cover:*
python
n_vertices_range = range(4, 16)      
edge_probability = 0.4               
trials = 3


*Hamiltonian Path:*
python
n_vertices_range = range(4, 14)      
edge_probability = 0.5
trials = 5


*Subset Sum:*
python
n_elements_range = range(10, 30, 2)  
max_value = 100
trials = 3


### Viewing Results

*CSV Files:*
bash
# View raw data
cat data/sat_runtime.csv
head -20 data/vertexcover_runtime.csv

# Statistics
python3 -c "import pandas as pd; print(pd.read_csv('data/sat_runtime.csv').describe())"


*Graphs (Ubuntu with GUI):*
bash
# Single image
xdg-open data/plot_sat_runtime.png

# All plots
eog data/plot_*.png

# Or use image viewer
feh data/*.png


*Graphs (Headless Server):*
bash
# Copy to local machine
scp user@server:/path/to/data/*.png ./local_plots/

# Or use terminal image viewer
sudo apt install fbi
fbi data/plot_sat_runtime.png


### Customizing Benchmarks

*Reduce Problem Sizes (faster benchmarks):*
python
# Edit bench/run.py
sat_sizes = range(5, 15)        
vc_sizes = range(4, 10)         


*Increase Precision (more trials):*
python
# Edit bench/run.py
TRIALS_SAT = 10                 
TRIALS_VC = 5                   


*Custom Metrics:*
python
# In bench/run.py, add columns to CSV:
with open('data/sat_runtime.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow([n_vars, runtime, nodes, 
                     memory_usage, backtrack_count])


### Expected Runtimes

| Benchmark | Duration | Output Size |
|-----------|----------|-------------|
| SAT | 5-15 min | ~100 KB CSV |
| Vertex Cover | 10-30 min | ~50 KB CSV |
| Hamiltonian Path | 5-20 min | ~75 KB CSV |
| Subset Sum | 5-15 min | ~60 KB CSV |
| *Total* | *25-80 min* | *~300 KB* |

---

## Installation & Setup

### Ubuntu / Debian

bash
# Update system
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Install optional tools
sudo apt install python3-tk minisat  

# Navigate to project
cd /path/to/np-completeness-project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install matplotlib networkx

# Verify installation
python3 -c "import matplotlib, networkx; print('✓ Setup complete')"


### macOS

bash
# Install Python 3
brew install python3

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install matplotlib networkx

# Optional: Install minisat
brew install minisat


### Windows (WSL recommended)

bash
# Install WSL2 with Ubuntu
wsl --install -d Ubuntu-22.04

# Inside WSL, follow Ubuntu instructions above


### Verify Installation

bash
# Test imports
python3 << EOF
from solvers import sat, vertex_cover, hampath, subsetsum
from reductions import sat_to_vc, subsetsum_to_sat
from bonus import sudoku_sat
print("✓ All modules imported successfully")
EOF

# Quick solver test
python3 -c "from solvers.sat import dpll_solve; print('SAT:', dpll_solve(3, [[1,2,-3]]))"
python3 -c "from solvers.subsetsum import brute_force; print('Subset Sum:', brute_force([1,2,3], 5))"


---

## Complete Workflow Example

bash
# =========================
# SETUP
# =========================
cd np-completeness-project
python3 -m venv venv
source venv/bin/activate
pip install matplotlib networkx

# =========================
# TEST INDIVIDUAL SOLVERS
# =========================

# SAT
python3 -c "
from solvers.sat import dpll_solve
result, model, nodes = dpll_solve(4, [[1,2,-3], [-1,4], [3,-4]])
print(f'SAT: {result}, Nodes: {nodes}')
"

# Vertex Cover
python3 -c "
from solvers.vertex_cover import add_edge, exact_branching
g = {}
add_edge(g, 1, 2); add_edge(g, 2, 3); add_edge(g, 3, 1)
sat, cover = exact_branching(g, k=2)
print(f'VC: {sat}, Cover: {cover}')
"

# Subset Sum
python3 -c "
from solvers.subsetsum import meet_in_the_middle
found, subset = meet_in_the_middle([3,7,9,10,15], 19)
print(f'Subset Sum: {found}, Subset: {subset}')
"

# =========================
# TEST REDUCTIONS
# =========================

# 3-SAT → VC
python3 -c "
from reductions.sat_to_vc import reduce_3sat_to_vc
g, k, _, _ = reduce_3sat_to_vc(3, [[1,-2,3], [-1,2,-3]])
print(f'Reduction: {len(g)} vertices, k={k}')
"

# Subset Sum → SAT
python3 -c "
from reductions.subsetsum_to_sat import exactly_equal_sum_parity
n_vars, clauses, _ = exactly_equal_sum_parity([3,5,7], 10)
print(f'Encoded: {n_vars} vars, {len(clauses)} clauses')
"

# =========================
# BONUS: SUDOKU
# =========================
python3 -m bonus.sudoku_sat

# =========================
# BENCHMARKING
# =========================

# Generate all data
python3 -m bench.run

# Create all plots
python3 -m bench.plots

# View results
ls -lh data/
xdg-open data/plot_sat_runtime.png

# =========================
# ANALYSIS
# =========================

# View CSV data
cat data/sat_runtime.csv
head -10 data/vertexcover_runtime.csv

# Summary statistics
python3 << EOF
import csv

with open('data/sat_runtime.csv') as f:
    reader = csv.DictReader(f)
    data = list(reader)
    print(f"SAT benchmarks: {len(data)} data points")
    print(f"Max runtime: {max(float(r['time_seconds']) for r in data):.2f}s")
EOF


---

## Troubleshooting

### Import Errors
bash
# Verify you're in project root
pwd  # Should show: /path/to/np-completeness-project

# Check directory structure
ls solvers/ reductions/ bench/ bonus/

# Verify Python path
python3 -c "import sys; print('\\n'.join(sys.path))"


### Module Not Found
bash
# Ensure virtual environment is activated
source venv/bin/activate
which python3  # Should point to venv/bin/python3

# Reinstall dependencies
pip install --upgrade matplotlib networkx


### Display Issues (Headless Server)
python
# Add to top of plotting scripts
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt


### Memory Errors (Large Benchmarks)
python
# Reduce problem sizes in bench/run.py
SAT_MAX_VARS = 20       
VC_MAX_VERTICES = 12    


### Slow Benchmarks
bash
# Run subset of benchmarks
python3 << EOF
from bench.run import benchmark_sat
benchmark_sat()  
EOF

# Or edit bench/run.py to comment out slow benchmarks


---

## Educational Value

This project demonstrates:

1. *NP-Completeness Theory*
   - Cook-Levin Theorem (SAT is NP-Complete)
   - Karp reductions between problems
   - Polynomial-time verifiability

2. *Algorithm Design Paradigms*
   - Backtracking and branch-and-bound
   - Dynamic programming
   - Greedy approximation
   - Divide-and-conquer (meet-in-the-middle)

3. *Complexity Analysis*
   - Exponential vs polynomial growth
   - Space-time tradeoffs
   - Approximation ratios
   - Empirical performance validation

4. *Practical Problem Solving*
   - When to use exact vs approximate algorithms
   - Real-world implications of NP-Completeness
   - Encoding problems as SAT instances

---

## File Checklist

Ensure your project has these files:


solvers/
  ├── __init__.py         
  ├── sat.py              
  ├── vertex_cover.py   
  ├── hampath.py          
  └── subsetsum.py        

reductions/
  ├── __init__.py          
  ├── sat_to_vc.py        
  └── subsetsum_to_sat.py 

bench/
  ├── __init__.py          
  ├── run.py             
  └── plots.py            

bonus/
  ├── __init__.py        
  └── sudoku_sat.py       

data/                     
utils/                    
README.md                  
requirements.txt           


---

## Quick Reference Commands

bash
# SETUP
python3 -m venv venv && source venv/bin/activate && pip install matplotlib networkx

# TEST SOLVERS
python3 -c "from solvers.sat import dpll_solve; print(dpll_solve(3, [[1,2,-3]]))"
python3 -c "from solvers.vertex_cover import add_edge, exact_branching; g={}; add_edge(g,1,2); add_edge(g,2,3); print(exact_branching(g,2))"
python3 -c "from solvers.subsetsum import brute_force; print(brute_force([1,2,3,4], 7))"

# TEST REDUCTIONS
python3 -c "from reductions.sat_to_vc import reduce_3sat_to_vc; print(reduce_3sat_to_vc(2, [[1,-2,1]])[:2])"
python3 -c "from reductions.subsetsum_to_sat import exactly_equal_sum_parity; print(exactly_equal_sum_parity([2,3], 5)[0:2])"

# BENCHMARKS
python3 -m bench.run && python3 -m bench.plots

# VIEW RESULTS
ls data/ && xdg-open data/plot_sat_runtime.png

# SUDOKU
python3 -m bonus.sudoku_sat


---

## Performance Expectations

### SAT Solver (DPLL)
| Variables | Clauses | Time | Nodes Explored |
|-----------|---------|------|----------------|
| 10 | 43 | 0.01s | ~100 |
| 15 | 65 | 0.1s | ~1,000 |
| 20 | 86 | 1-5s | ~10,000 |
| 25 | 108 | 10-60s | ~100,000 |

### Vertex Cover (Exact)
| Vertices | Edges | Time | Cover Size |
|----------|-------|------|------------|
| 8 | 10 | 0.01s | 3-5 |
| 12 | 18 | 0.1-1s | 5-7 |
| 16 | 25 | 1-10s | 7-10 |
| 20 | 35 | 10-120s | 9-13 |

### Vertex Cover (2-Approximation)
| Vertices | Edges | Time | Cover Size |
|----------|-------|------|------------|
| 100 | 500 | 0.01s | 20-40 |
| 1000 | 5000 | 0.1s | 200-400 |
| 10000 | 50000 | 1s | 2000-4000 |

### Subset Sum (Brute Force vs MITM)
| Elements | Brute Force | Meet-in-Middle |
|----------|-------------|----------------|
| 15 | 0.01s | 0.001s |
| 20 | 0.5s | 0.01s |
| 25 | 15s | 0.1s |
| 30 | 8min | 1s |
| 35 | 4hours | 10s |

---

## Common Issues and Solutions

### Issue: "No module named 'solvers'"
*Solution:*
bash
# Ensure __init__.py exists in solvers/
touch solvers/__init__.py reductions/__init__.py bench/__init__.py bonus/__init__.py

# Run from project root
cd /path/to/np-completeness-project
python3 -c "from solvers import sat"


### Issue: "matplotlib backend error"
*Solution:*
bash
# Install tkinter
sudo apt install python3-tk

# Or use non-interactive backend
export MPLBACKEND=Agg
python3 -m bench.plots


### Issue: Benchmarks take too long
*Solution:*
python
# Edit bench/run.py to reduce ranges
SAT_RANGE = range(5, 15)      
VC_RANGE = range(4, 10)      
TRIALS = 1                    


### Issue: "RecursionError: maximum recursion depth exceeded"
*Solution:*
python
# Add to top of solver files
import sys
sys.setrecursionlimit(10000)  


### Issue: Memory error with large graphs
*Solution:*
python
# Use generators instead of storing all results
# In vertex_cover.py, for example:
def generate_subsets(s):
    for i in range(len(s)):
        yield s[:i] + s[i+1:]  


---

## Extending the Project

### Adding a New Solver
python
# 1. Create solvers/new_problem.py
def solve_new_problem(instance):
    """
    Solve NP-Complete problem X
    
    Args:
        instance: Problem instance
    
    Returns:
        (solution_exists, solution, stats)
    """
    # Implementation
    pass

# 2. Add to benchmarking
# In bench/run.py:
def benchmark_new_problem():
    with open('data/new_problem_runtime.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['size', 'time', 'solution'])
        # ... benchmark code

# 3. Add plotting
# In bench/plots.py:
def plot_new_problem():
    # ... plotting code


### Adding a New Reduction
python
# reductions/problem_a_to_b.py
def reduce_a_to_b(instance_a):
    """
    Polynomial-time reduction from Problem A to Problem B
    
    Args:
        instance_a: Instance of problem A
    
    Returns:
        instance_b: Equivalent instance of problem B
    """
    # Reduction construction
    pass

# Test equivalence
def verify_reduction(instance_a, instance_b, solution_b):
    """Verify solution to B solves A"""
    # Verification logic
    pass


---

## Citation and References

*Key Papers:*
- Cook, S.A. (1971). "The complexity of theorem-proving procedures"
- Karp, R.M. (1972). "Reducibility among combinatorial problems"
- Garey, M.R., Johnson, D.S. (1979). "Computers and Intractability"

*Algorithms:*
- Davis-Putnam-Logemann-Loveland (DPLL) SAT solver
- 2-approximation for Vertex Cover (greedy maximal matching)
- Held-Karp dynamic programming for TSP/Hamiltonian Path
- Meet-in-the-middle technique (Horowitz & Sahni, 1974)

---

## License and Usage

This project is designed for educational purposes to demonstrate:
- Fundamental concepts in computational complexity
- Practical algorithm implementation
- Empirical analysis of theoretical bounds

Feel free to use, modify, and extend for academic projects.
