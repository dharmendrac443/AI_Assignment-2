import numpy as np
import random
import matplotlib.pyplot as plt
import imageio
import os
import time

# Create results directory
os.makedirs("./results", exist_ok=True)

# Define coordinates of cities
NUM_CITIES = 10  # Match my earlier suggestion for consistency
np.random.seed(42)  # For replicability
CITIES = np.random.rand(NUM_CITIES, 2) * 100  # Scale for visibility
TAU = 600  # 10 minutes in seconds

# Distance between cities
def total_distance(tour):
    dist = 0
    for i in range(len(tour)):
        dist += np.linalg.norm(CITIES[tour[i]] - CITIES[tour[(i + 1) % len(tour)]])
    return dist

# Generate neighbors by swapping two cities
def generate_neighbors(tour):
    neighbors = []
    for i in range(len(tour)):
        for j in range(i + 1, len(tour)):
            neighbor = tour.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)
    return neighbors

# Plot a tour
def plot_tour(tour, title):
    fig, ax = plt.subplots()
    cities = CITIES[tour + [tour[0]]]
    ax.plot(cities[:, 0], cities[:, 1], 'o-', color='blue')
    for i, (x, y) in enumerate(CITIES):
        ax.text(x, y, str(i), fontsize=12)
    ax.set_title(title)
    return fig

# Hill Climbing
def hill_climbing(run_id):
    start_time = time.time()
    current = list(range(NUM_CITIES))
    random.shuffle(current)
    current_cost = total_distance(current)
    frames = []
    iterations = 0

    while (time.time() - start_time) < TAU:
        neighbors = generate_neighbors(current)
        next_state = min(neighbors, key=total_distance)
        next_cost = total_distance(next_state)
        iterations += 1

        fig = plot_tour(current, f"Run {run_id} | Cost: {current_cost:.2f}")
        frame_path = f"./results/hc_temp_{run_id}.png"
        fig.savefig(frame_path)
        plt.close(fig)
        frames.append(imageio.v2.imread(frame_path))
        os.remove(frame_path)

        if next_cost >= current_cost:
            break  # Converged
        current, current_cost = next_state, next_cost

    elapsed = time.time() - start_time
    gif_path = f"./results/tsp_hc_{run_id}.gif"
    imageio.mimsave(gif_path, frames, fps=2)

    print(f"[Run {run_id}] Final Cost: {current_cost:.2f}, Time: {elapsed:.2f}s, Iterations: {iterations}")
    return current_cost, elapsed, iterations

# Run 5 times
results = [hill_climbing(i) for i in range(5)]
costs, times, iterations = zip(*results)

# Averages
avg_cost = sum(costs) / 5
avg_time = sum(times) / 5
avg_iterations = sum(iterations) / 5

print(f"\nHill Climbing - Avg Cost: {avg_cost:.2f}, Avg Time: {avg_time:.2f}s, Avg Iterations: {avg_iterations:.2f}")

# Plot run times
plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), times, marker='o', linestyle='-', color='green')
plt.title("Hill Climbing - Time per Run (TSP)")
plt.xlabel("Run")
plt.ylabel("Time (seconds)")
plt.grid(True)
plt.savefig("./results/tsp_hill_climbing_time_plot.png")
plt.close()