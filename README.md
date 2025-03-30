# 🧠 AI Search and Optimization Algorithms – Assignment-2

## 👥 Team Members
- **Dharmendra Chauhan** (Roll No: CS24M115)
- **Kodela Phanindra** (Roll No: CS24M121)

---

## 📁 Files Included

| Filename                                 | Description                                      |
|------------------------------------------|--------------------------------------------------|
| `Branch_and_Bound_Frozen_Lake.py`        | Branch and Bound on Frozen Lake environment     |
| `Iterative_Deepening_A*_Frozen Lake.py`  | IDA* on Frozen Lake environment                 |
| `Branch_and_Bound_Ant_Maze.py`           | Branch and Bound on custom Ant Maze             |
| `Iterative_Deepening_A*_Ant_Maze.py`     | IDA* on custom Ant Maze                         |
| `Hill_Climbing_TSP.py`                   | Hill Climbing on 10-city Traveling Salesman     |
| `Simulated_Annealing_TSP.py`            | Simulated Annealing on 10-city TSP              |
| `requirements.txt`                       | Python and library dependencies                 |

---

## 🔍 Environments Used

### 1. **Frozen Lake**
- Environment: `FrozenLake-v1`
- Config: `is_slippery=False`
- Size: `4x4`
- Goal: Reach state 15 from start state 0

### 2. **Ant Maze**
- Custom-built discrete environment
- Modeled as a graph with defined transitions
- Start and goal states specified
- Rendered using `gymnasium` compatible visuals

### 3. **Traveling Salesman Problem (TSP)**
- 10-city fully connected graph
- Random symmetric distance matrix
- Algorithms return best tour and total path cost

---

## 🧠 Algorithms Implemented

### ✅ Search Algorithms

#### 1. **Branch and Bound (B&B)**
- Explores least-cost paths using priority queues
- Avoids cycles and redundant expansions
- Terminates on goal detection or timeout (TAU)

#### 2. **Iterative Deepening A\***
- Combines A\* and iterative deepening
- Uses Manhattan Distance heuristic for grid-based search
- Expands threshold gradually to find optimal path
- More memory-efficient than A\*

---

### 🔧 Optimization Algorithms

#### 3. **Hill Climbing**
- Starts with a random TSP solution
- Repeatedly improves by swapping cities
- Stops when local maxima is reached
- Fast but prone to getting stuck

#### 4. **Simulated Annealing**
- Probabilistic optimization for TSP
- Accepts worse moves early to escape local minima
- Cooling schedule reduces randomness over time
- Returns near-optimal path within time constraints

---

## ⏱ Runtime & Evaluation

- Each search algorithm is run **5 times**
- Time limit (`TAU`) is **600 seconds per run**
- Logs include:
  - Whether the goal was reached
  - Time taken
  - Path cost and step count (for search tasks)

---

## 📦 Requirements

Create a virtual environment and install all dependencies:

```bash
pip install -r requirements.txt
