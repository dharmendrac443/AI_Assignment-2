import numpy as np
import random
import matplotlib.pyplot as plt
import imageio
import os
import time

# Create results directory
os.makedirs("./results", exist_ok=True)

# Define coordinates of cities
NUM_CITIES = 10  # Consistent with HC
np.random.seed(42)  # For replicability
CITIES = np.random.rand(NUM_CITIES, 2) * 100  # Scale for visibility
TAU = 600  # 10 minutes in seconds

# Distance between cities
def total_distance(tour):
    dist = 0
    for i in range(len(tour)):
        dist += np.linalg.norm(CITIES[tour[i]] - CITIES[tour[(i + 1) % len(tour)]])
    return dist

# Generate a neighbor by swapping two cities
def generate_neighbor(tour):
    i, j = random.sample(range(len(tour)), 2)  # Pick two distinct cities
    neighbor = tour.copy()
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor

# Plot a tour
def plot_tour(tour, title):
    fig, ax = plt.subplots()
    cities = CITIES[tour + [tour[0]]]
    ax.plot(cities[:, 0], cities[:, 1], 'o-', color='blue')
    for i, (x, y) in enumerate(CITIES):
        ax.text(x, y, str(i), fontsize=12)
    ax.set_title(title)
    return fig

# Simulated Annealing
def simulated_annealing(run_id):
    start_time = time.time()
    current = list(range(NUM_CITIES))
    random.shuffle(current)
    current_cost = total_distance(current)
    best = current.copy()
    best_cost = current_cost
    frames = []
    iterations = 0
    
    T = 1000.0  # Initial temperature
    alpha = 0.99  # Cooling rate
    min_T = 0.1  # Minimum temperature

    while T > min_T and (time.time() - start_time) < TAU:
        neighbor = generate_neighbor(current)
        neighbor_cost = total_distance(neighbor)
        delta = neighbor_cost - current_cost
        iterations += 1

        # Record frame every 100 iterations for manageable gif size
        if iterations % 100 == 0:
            fig = plot_tour(current, f"Run {run_id} | Cost: {current_cost:.2f} | T: {T:.2f}")
            frame_path = f"./results/sa_temp_{run_id}.png"
            fig.savefig(frame_path)
            plt.close(fig)
            frames.append(imageio.v2.imread(frame_path))
            os.remove(frame_path)

        # Accept neighbor if better or with probability e^(-delta/T)
        if delta < 0 or random.random() < np.exp(-delta / T):
            current, current_cost = neighbor, neighbor_cost
            if current_cost < best_cost:
                best, best_cost = current.copy(), current_cost

        T *= alpha  # Cool down

    elapsed = time.time() - start_time
    gif_path = f"./results/tsp_sa_{run_id}.gif"
    imageio.mimsave(gif_path, frames, fps=2)

    print(f"[Run {run_id}] Final Cost: {best_cost:.2f}, Time: {elapsed:.2f}s, Iterations: {iterations}")
    return best_cost, elapsed, iterations

# Run 5 times
results = [simulated_annealing(i) for i in range(5)]
costs, times, iterations = zip(*results)

# Averages
avg_cost = sum(costs) / 5
avg_time = sum(times) / 5
avg_iterations = sum(iterations) / 5

print(f"\nSimulated Annealing - Avg Cost: {avg_cost:.2f}, Avg Time: {avg_time:.2f}s, Avg Iterations: {avg_iterations:.2f}")

# Plot run times
plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), times, marker='o', linestyle='-', color='red')
plt.title("Simulated Annealing - Time per Run (TSP)")
plt.xlabel("Run")
plt.ylabel("Time (seconds)")
plt.grid(True)
plt.savefig("./results/tsp_simulated_annealing_time_plot.png")
plt.close()