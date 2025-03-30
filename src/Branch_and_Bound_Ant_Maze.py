import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import heapq
import time

# Create results directory
os.makedirs("./results", exist_ok=True)

# Custom Discrete Ant Maze Class

class DiscreteAntMaze:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.start = (0, 0)
        self.goal = (4, 4)
        self.agent_pos = self.start
        self.render_mode = "rgb_array"

        # 0 = free, 1 = wall
        self.maze = np.zeros((grid_size, grid_size), dtype=int)
        self.maze[1, 1] = 1
        self.maze[1, 2] = 1
        self.maze[3, 3] = 1

    def reset(self):
        self.agent_pos = self.start
        return self._state_to_index(self.agent_pos)

    def step(self, action):
        x, y = self.agent_pos
        if action == 0: y -= 1  # up
        elif action == 1: x += 1  # right
        elif action == 2: y += 1  # down
        elif action == 3: x -= 1  # left

        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            if self.maze[y, x] == 0:
                self.agent_pos = (x, y)

        done = self.agent_pos == self.goal
        reward = 1 if done else 0
        return self._state_to_index(self.agent_pos), reward, done, {}

    def render(self):
        size = 500
        img = np.ones((size, size, 3), dtype=np.uint8) * 255
        cell_size = size // self.grid_size
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                color = (0, 0, 0) if self.maze[y, x] == 1 else (200, 200, 200)
                if (x, y) == self.agent_pos:
                    color = (0, 0, 255)
                elif (x, y) == self.goal:
                    color = (0, 255, 0)
                x1, y1 = x * cell_size, y * cell_size
                img[y1:y1+cell_size, x1:x1+cell_size] = color
        return img

    def _state_to_index(self, pos):
        return pos[1] * self.grid_size + pos[0]

    def _index_to_state(self, index):
        return (index % self.grid_size, index // self.grid_size)

    @property
    def observation_space(self):
        class Space:
            n = self.grid_size * self.grid_size
        return Space()

    @property
    def action_space(self):
        class Space:
            n = 4  # up, right, down, left
        return Space()

# Build Transition Graph
def build_graph(env):
    graph = {}
    for state in range(env.observation_space.n):
        graph[state] = []
        x, y = env._index_to_state(state)
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < env.grid_size and 0 <= ny < env.grid_size:
                if env.maze[ny, nx] == 0:
                    neighbor_index = env._state_to_index((nx, ny))
                    graph[state].append(neighbor_index)
    return graph

# Branch and Bound Algorithm

def branch_and_bound(env, graph, run_id, TAU=600):
    start_time = time.time()
    start_state = env._state_to_index(env.start)
    goal_state = env._state_to_index(env.goal)

    heap = [(0, [start_state])]
    visited = set()

    while heap and (time.time() - start_time) < TAU:
        cost, path = heapq.heappop(heap)
        current = path[-1]

        if current == goal_state:
            elapsed = time.time() - start_time
            frames = []
            env.reset()
            for state in path:
                env.agent_pos = env._index_to_state(state)
                frames.append(env.render())
            gif_path = f"./results/ant_maze_bb_{run_id}.gif"
            imageio.mimsave(gif_path, frames, fps=4)
            print(f"[Run {run_id}] Goal found! Cost: {cost}, Steps: {len(path)}, Time: {elapsed:.2f}s")
            return 1, elapsed, len(path)

        if current in visited:
            continue
        visited.add(current)

        for neighbor in graph[current]:
            if neighbor not in path:
                heapq.heappush(heap, (cost + 1, path + [neighbor]))

    elapsed = time.time() - start_time
    print(f"[Run {run_id}] No path found. Time: {elapsed:.2f}s")
    return 0, elapsed, None

# Run 5 Trials and Plot Results

env = DiscreteAntMaze(grid_size=5)
graph = build_graph(env)

results = [branch_and_bound(env, graph, run_id=i) for i in range(5)]
rewards, times, steps = zip(*results)

# Averages
avg_time = sum(times) / 5
avg_reward = sum(rewards) / 5
successful_steps = [s for s in steps if s is not None]
avg_steps = sum(successful_steps) / len(successful_steps) if successful_steps else None

print(f"\nAnt Maze BnB - Avg Time: {avg_time:.2f}s, Avg Reward: {avg_reward:.2f}")
if avg_steps:
    print(f"Avg Steps to Goal: {avg_steps:.2f}")
else:
    print("No successful runs — Avg Steps not available.")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), times, marker='o', linestyle='-', color='brown')
plt.title("Branch & Bound - Time per Run (Ant Maze)")
plt.xlabel("Run")
plt.ylabel("Time (seconds)")
plt.grid(True)
plt.savefig("./results/ant_maze_bb_time_plot.png")
plt.close()
