import matplotlib.pyplot as plt
import numpy as np
import pickle

# Function to plot the improvement
def plot_fitness_progress(fitness_scores, generations):
    generations_range = np.arange(1, generations + 1)
    average_fitness = [np.mean([score for score, _ in fitness_scores[generation]]) for generation in range(generations)]
    best_scores = [max([score for score, _ in fitness_scores[generation]]) for generation in range(generations)]

    plt.figure(figsize=(10, 6))
    plt.plot(generations_range, average_fitness, label='Average Fitness')
    plt.plot(generations_range, best_scores, label='Best Score')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.title('Improvement Over Generations')
    plt.legend()
    plt.grid(True)
    plt.show()

# Load the saved fitness scores
try:
    with open('best_model.pkl', 'rb') as file:
        data = pickle.load(file)
        if isinstance(data, dict) and 'fitness_scores' in data:
            fitness_scores_over_generations = data['fitness_scores']
        else:
            fitness_scores_over_generations = []
except FileNotFoundError:
    print("No saved fitness scores found. Ensure that `best_model.pkl` exists.")
    fitness_scores_over_generations = []

# Number of generations
generations = len(fitness_scores_over_generations)

# Plot the fitness progress
if generations > 0:
    plot_fitness_progress(fitness_scores_over_generations, generations)
else:
    print("No fitness scores to plot.")
