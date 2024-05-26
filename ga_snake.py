import random
from ga_models.ga_simple import SimpleModel
from ga_controller import GAController
from snake import SnakeGame

# Define the dimensions for the neural network
input_size = 11  # Number of features in the observation space
hidden_layer_size = 16  # Number of neurons in the hidden layer
output_size = 4  # Number of possible actions

# Parameters for the GA
population_size = 50
mutation_rate = 0.05
num_generations = 100

# Initialize the population
population = [SimpleModel(dims=(input_size, hidden_layer_size, output_size)) for _ in range(population_size)]


# Define the fitness function
def evaluate_fitness(model):
    game = SnakeGame(controller=None)  # Initialize the game without a controller first
    controller = GAController(game=game, model=model, display=False)
    game.controller = controller  # Assign the controller after initialization
    game.run()  # Run the game
    return game.snake.score


# Run the GA
for generation in range(num_generations):
    fitness_scores = []
    for idx, individual in enumerate(population):
        score = evaluate_fitness(individual)
        fitness_scores.append((score, individual))

    # Sort individuals based on fitness scores
    fitness_scores.sort(key=lambda x: x[0], reverse=False)

    # Select the best individuals
    selected_individuals = [individual for _, individual in fitness_scores[:population_size // 2]]

    # Create the next generation
    next_generation = []
    for _ in range(population_size):
        parent1, parent2 = random.sample(selected_individuals, 2)
        offspring = parent1 + parent2
        offspring.mutate(mutation_rate)
        next_generation.append(offspring)

    # Replace the old population with the new one
    population = next_generation

    # Print the average fitness of the generation
    average_fitness = sum(score for score, _ in fitness_scores) / population_size
    print(f'Generation {generation + 1}: Average Fitness = {average_fitness}')

# Run the GA
for generation in range(num_generations):
    fitness_scores = []
    for idx, individual in enumerate(population):
        score = evaluate_fitness(individual)
        fitness_scores.append(score)
        #print(f'Individual {idx + 1}: Fitness = {score}')

    # Print scores for debugging
    #print(f'Generation {generation + 1} scores: {fitness_scores}')

    # Select the best individuals
    selected_individuals = random.choices(
        population, weights=fitness_scores, k=population_size // 2)

    # Create the next generation
    next_generation = []
    for _ in range(population_size):
        parent1, parent2 = random.sample(selected_individuals, 2)
        offspring = parent1 + parent2
        offspring.mutate(mutation_rate)
        next_generation.append(offspring)

    # Replace the old population with the new one
    population = next_generation

    # Print the average fitness of the generation
    average_fitness = sum(fitness_scores) / population_size
    print(f'Generation {generation + 1}: Average Fitness = {average_fitness}')

# At the end of the run, the population contains the evolved models
best_individual = max(population, key=evaluate_fitness)
print(f'Best individual score: {evaluate_fitness(best_individual)}')
