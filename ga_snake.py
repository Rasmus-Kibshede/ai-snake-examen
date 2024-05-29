import random

import pygame

from ga_models.ga_simple import SimpleModel
from ga_controller import GAController
from snake import SnakeGame

# Define the dimensions for the neural network
input_size = 14  # Number of features in the observation space
hidden_layer_size = 64  # Number of neurons in the hidden layer
output_size = 4  # Number of possible actions

# Parameters for the GA
population_size = 5
mutation_rate = 0.9  # Adjust mutation rate
num_generations = 10

# Initialize the population
population = [SimpleModel(dims=(input_size, hidden_layer_size, output_size)) for _ in range(population_size)]


# Define the fitness function
def evaluate_fitness(model):
    game = SnakeGame(controller=None)  # Initialize the game without a controller first
    controller = GAController(game=game, model=model, display=False)
    game.controller = controller  # Assign the controller after initialization
    game.run()  # Run the game

    # Reward for eating food
    food_reward = 100 * game.snake.score

    # Penalty for hitting walls
    wall_collision_penalty = -10 if not game.snake.p.within(game.grid) else 0

    # Distance to food
    distance_to_food = 1 / (1 + game.snake.distance_to_food())

    # Reward for moving towards food
    food_direction_reward = 10 if game.snake.p + game.snake.v == game.food.p else 0

    # Reward for avoiding wall collision
    avoid_wall_reward = 5 if game.snake.p.within(game.grid) else 0

    # Reward for changing direction (not going in a straight line)
    change_direction_reward = 2 if game.snake.v != controller.current_direction else 0

    # Calculate fitness based on score, distance to food, and rewards/penalties
    fitness = (game.snake.score + food_reward +
               wall_collision_penalty + distance_to_food +
               food_direction_reward + avoid_wall_reward + change_direction_reward)

    # Reward for approaching food
    approach_food_reward = 10 if game.snake.distance_to_food() < 1 else 0  # Reward for getting closer to food

    # Reward for exploring new areas
    exploration_reward = 0.1 if game.snake.p not in game.snake.body else 0  # Reward for moving to unvisited areas

    # Update fitness with additional rewards
    fitness += approach_food_reward + exploration_reward

    # Add a baseline value to ensure positive fitness scores
    baseline = max(0, -wall_collision_penalty)
    fitness += baseline

    return fitness


# Run the GA
for generation in range(num_generations):
    pygame.init()
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

    # Print the average fitness and best score of the generation
    average_fitness = sum(score for score, _ in fitness_scores) / population_size
    best_score = max(score for score, _ in fitness_scores)
    print(f'Generation {generation + 1}: Average Fitness = {average_fitness}, Best Score = {best_score}')
