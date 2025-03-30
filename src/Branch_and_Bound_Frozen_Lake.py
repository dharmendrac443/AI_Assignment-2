import gymnasium as gym
import time
import os
import imageio
import matplotlib.pyplot as plt
import heapq

# Setup
TAU = 600
goal_state = 15
os.makedirs("./results", exist_ok=True)

# Initialize environment
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")

# Build transition graph from env.P
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

# Branch and Bound Search on built graph
def branch_and_bound(run_id):
    start_time = time.time()
    heap = [(0, [0])]  # (path cost, path)
    visited = set()

    while heap and (time.time() - start_time) < TAU:
        cost, path = heapq.heappop(heap)
        current = path[-1]

        if current == goal_state:
            elapsed = time.time() - start_time
            # Render the path to a GIF
            frames = []
            env.reset()
            for state in path:
                env.unwrapped.s = state
                frames.append(env.render())
            gif_path = f"./results/frozen_lake_bb_{run_id}.gif"
            imageio.mimsave(gif_path, frames, fps=4)
            print(f"[Run {run_id}] Goal found! Cost: {cost}, Steps: {len(path)}, Time: {elapsed:.2f}s")
            return 1, elapsed, len(path)

        if current in visited:
            continue
        visited.add(current)

        for neighbor in GRAPH[current]:
            if neighbor not in path:
                heapq.heappush(heap, (cost + 1, path + [neighbor]))

    elapsed = time.time() - start_time
    print(f"[Run {run_id}] No path found. Time: {elapsed:.2f}s")
    return 0, elapsed, None

# Run 5 times
results = [branch_and_bound(i) for i in range(5)]
rewards, times, steps = zip(*results)

# Summary
avg_time = sum(times) / 5
avg_reward = sum(rewards) / 5
successful_steps = [s for s in steps if s is not None]
avg_steps = sum(successful_steps) / len(successful_steps) if successful_steps else None

print(f"\nBranch & Bound - Avg Time: {avg_time:.2f}s, Avg Reward: {avg_reward:.2f}")
if avg_steps:
    print(f"Avg Steps to Goal: {avg_steps:.2f}")
else:
    print("No successful runs — Avg Steps not available.")

# Time Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), times, marker='o', linestyle='-', color='orange')
plt.title("Branch & Bound - Time per Run (Frozen Lake)")
plt.xlabel("Run")
plt.ylabel("Time (seconds)")
plt.grid(True)
plt.savefig("./results/frozen_lake_bb_time_plot.png")
plt.close()
