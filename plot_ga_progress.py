import matplotlib.pyplot as plt
import numpy as np
import pickle
import gzip

# Function to plot the improvement
def plot_fitness_progress(average_fitness_scores, best_scores):
    generations = len(average_fitness_scores)
    generations_range = np.arange(1, generations + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(generations_range, average_fitness_scores, label='Average Fitness')
    plt.plot(generations_range, best_scores, label='Best Score')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.title('Improvement Over Generations')
    plt.legend()
    plt.grid(True)
    plt.show()

# Load the saved fitness scores from the gzip-compressed pickle file
try:
    with gzip.open('best_model.pkl.gz', 'rb') as file:
        data = pickle.load(file)
        if isinstance(data, dict) and 'average_fitness_scores' in data and 'best_scores' in data:
            average_fitness_scores = data['average_fitness_scores']
            best_scores = data['best_scores']
        else:
            average_fitness_scores = []
            best_scores = []
except FileNotFoundError:
    print("No saved fitness scores found. Ensure that `best_model.pkl.gz` exists.")
    average_fitness_scores = []
    best_scores = []

# Plot the fitness progress
if average_fitness_scores and best_scores:
    plot_fitness_progress(average_fitness_scores, best_scores)
else:
    print("No fitness scores to plot.")
