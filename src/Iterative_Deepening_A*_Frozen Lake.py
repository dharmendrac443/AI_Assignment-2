import gymnasium as gym
import time
import os
import imageio
import matplotlib.pyplot as plt

# Setup
TAU = 600
goal_state = 15
os.makedirs("./results", exist_ok=True)

env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")

# Build transition graph
def build_graph(env):
    graph = {}
    for state in range(env.observation_space.n):
        graph[state] = []
        for action in range(env.action_space.n):
            for prob, next_state, reward, done in env.unwrapped.P[state][action]:
                if prob > 0:
                    graph[state].append(next_state)
    return graph

GRAPH = build_graph(env)

# Heuristic (Manhattan distance in 4x4 grid)
def heuristic(state):
    x1, y1 = state % 4, state // 4
    x2, y2 = goal_state % 4, goal_state // 4
    return abs(x1 - x2) + abs(y1 - y2)

# Recursive depth-limited search
def dfs(path, g, threshold, visited, start_time):
    node = path[-1]
    f = g + heuristic(node)

    if time.time() - start_time > TAU:
        return "timeout", float("inf")

    if f > threshold:
        return None, f

    if node == goal_state:
        return path, f

    min_threshold = float("inf")
    for neighbor in GRAPH[node]:
        if neighbor not in visited:
            visited.add(neighbor)
            path.append(neighbor)
            result, temp = dfs(path, g + 1, threshold, visited, start_time)
            if result == "timeout":
                return "timeout", float("inf")
            if result is not None:
                return result, temp
            path.pop()
            visited.remove(neighbor)
            min_threshold = min(min_threshold, temp)
    return None, min_threshold

# IDA* Wrapper
def ida_star(run_id):
    start_time = time.time()
    threshold = heuristic(0)
    path = [0]
    visited = set([0])

    while True:
        result, temp = dfs(path, 0, threshold, visited, start_time)
        if result == "timeout":
            print(f"[Run {run_id}] Timeout at threshold {threshold}")
            return 0, TAU, None
        if result is not None:
            elapsed = time.time() - start_time
            # Render to GIF
            frames = []
            env.reset()
            for state in result:
                env.unwrapped.s = state
                frames.append(env.render())
            gif_path = f"./results/frozen_lake_ida_star_{run_id}.gif"
            imageio.mimsave(gif_path, frames, fps=4)
            print(f"[Run {run_id}] Goal found! Cost: {len(result)-1}, Steps: {len(result)}, Time: {elapsed:.2f}s")
            return 1, elapsed, len(result)
        if temp == float("inf"):
            break  # No more thresholds
        threshold = temp

    elapsed = time.time() - start_time
    print(f"[Run {run_id}] No path found. Time: {elapsed:.2f}s")
    return 0, elapsed, None

# Run 5 times
results = [ida_star(i) for i in range(5)]
rewards, times, steps = zip(*results)

avg_time = sum(times) / 5
avg_reward = sum(rewards) / 5
successful_steps = [s for s in steps if s is not None]
avg_steps = sum(successful_steps) / len(successful_steps) if successful_steps else None

print(f"\nIDA* - Avg Time: {avg_time:.2f}s, Avg Reward: {avg_reward:.2f}")
if avg_steps:
    print(f"Avg Steps to Goal: {avg_steps:.2f}")
else:
    print("No successful runs — Avg Steps not available.")

# Time Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), times, marker='o', linestyle='-', color='purple')
plt.title("IDA* - Time per Run (Frozen Lake)")
plt.xlabel("Run")
plt.ylabel("Time (seconds)")
plt.grid(True)
plt.savefig("./results/frozen_lake_ida_star_time_plot.png")
plt.close()
