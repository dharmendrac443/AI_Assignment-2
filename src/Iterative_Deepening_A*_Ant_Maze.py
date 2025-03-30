import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import time


# Create results directory
os.makedirs("./results", exist_ok=True)

# Heuristic Function

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# IDA* Recursive Search

def ida_star_search(env, graph, run_id, TAU=600):
    start_time = time.time()
    start_state = env._state_to_index(env.start)
    goal_state = env._state_to_index(env.goal)

    def heuristic(state):
        pos = env._index_to_state(state)
        goal = env.goal
        return manhattan_distance(pos, goal)

    def dfs(path, g, bound):
        current = path[-1]
        f = g + heuristic(current)
        if f > bound:
            return f
        if current == goal_state:
            return "FOUND"
        min_bound = float('inf')
        for neighbor in graph[current]:
            if neighbor not in path:
                path.append(neighbor)
                t = dfs(path, g + 1, bound)
                if t == "FOUND":
                    return "FOUND"
                if t < min_bound:
                    min_bound = t
                path.pop()
        return min_bound

    bound = heuristic(start_state)
    path = [start_state]

    while True:
        if time.time() - start_time > TAU:
            print(f"[Run {run_id}] Timeout reached.")
            return 0, time.time() - start_time, None
        t = dfs(path, 0, bound)
        if t == "FOUND":
            elapsed = time.time() - start_time
            # Generate frames for GIF
            frames = []
            env.reset()
            for state in path:
                env.agent_pos = env._index_to_state(state)
                frames.append(env.render())
            gif_path = f"./results/ant_maze_ida_star_{run_id}.gif"
            imageio.mimsave(gif_path, frames, fps=4)
            print(f"[Run {run_id}] Goal found! Steps: {len(path)}, Time: {elapsed:.2f}s")
            return 1, elapsed, len(path)
        if t == float('inf'):
            print(f"[Run {run_id}] No path found.")
            return 0, time.time() - start_time, None
        bound = t

# Run IDA* on the Maze

env = DiscreteAntMaze(grid_size=5)
graph = build_graph(env)

results = [ida_star_search(env, graph, run_id=i) for i in range(5)]
rewards, times, steps = zip(*results)

# Averages
avg_time = sum(times) / 5
avg_reward = sum(rewards) / 5
successful_steps = [s for s in steps if s is not None]
avg_steps = sum(successful_steps) / len(successful_steps) if successful_steps else None

print(f"\nAnt Maze IDA* - Avg Time: {avg_time:.2f}s, Avg Reward: {avg_reward:.2f}")
if avg_steps:
    print(f"Avg Steps to Goal: {avg_steps:.2f}")
else:
    print("No successful runs — Avg Steps not available.")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), times, marker='o', linestyle='-', color='blue')
plt.title("IDA* - Time per Run (Ant Maze)")
plt.xlabel("Run")
plt.ylabel("Time (seconds)")
plt.grid(True)
plt.savefig("./results/ant_maze_ida_star_time_plot.png")
plt.close()
