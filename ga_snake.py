import signal
import random
import numpy as np
import pickle
import gzip
import torch
from ga_models.ga_simple import SimpleModel
from ga_controller import GAController
from snake import SnakeGame, Food
from vector import Vector

# Define the dimensions for the neural network
input_size = 48 
hidden_layer_size = 64
output_size = 4

# Parameters for the GA
population_size = 100
mutation_rate = 0.1
num_generations = 1000
batch_size = 28  # Define batch size for batch processing

# Initialize the population
max_gen_score = []
population = [SimpleModel(dims=(input_size, hidden_layer_size, output_size))
              for _ in range(population_size)]

average_fitness_scores = []
best_scores = []
best_individual = None

import torch
import torch.nn.functional as F

def evaluate_fitness_batch(models, batch_size):
    fitness_scores = []
    batch_max_food_eaten = []

    for i in range(0, len(models), batch_size):
        batch_models = models[i:i + batch_size]

        # Create SnakeGame and GAController instances for each model
        games = [SnakeGame(controller=None, max_steps=40000) for _ in batch_models]
        controllers = [GAController(game=game, model=model, display=False) for game, model in zip(games, batch_models)]
        
        observations = []
        for controller in controllers:
            dn = controller.game.snake.p.y / controller.game.grid.y
            de = (controller.game.grid.x - controller.game.snake.p.x) / controller.game.grid.x
            ds = (controller.game.grid.y - controller.game.snake.p.y) / controller.game.grid.y
            dw = controller.game.snake.p.x / controller.game.grid.x
            dfx = (controller.game.food.p.x - controller.game.snake.p.x) / controller.game.grid.x
            dfy = (controller.game.food.p.y - controller.game.snake.p.y) / controller.game.grid.y

            left_obstacle = int(controller.game.snake.p.x == 0)
            right_obstacle = int(controller.game.snake.p.x == controller.game.grid.x - 1)
            up_obstacle = int(controller.game.snake.p.y == 0)
            down_obstacle = int(controller.game.snake.p.y == controller.game.grid.y - 1)

            vision = controller.game.snake.vision()

            obs = [dn, de, ds, dw, dfx, dfy, left_obstacle,
                   right_obstacle, up_obstacle, down_obstacle] + vision

            if len(obs) < 48:
                obs += [0] * (48 - len(obs))
            elif len(obs) > 48:
                obs = obs[:48]

            observations.append(obs)

        # Convert observations to tensor and move to device
        observations_tensor = torch.tensor(observations, dtype=torch.float32).to(batch_models[0].device)

        # Get actions from models
        actions = []
        for model in batch_models:
            with torch.no_grad():
                action_probabilities = model.model(observations_tensor)
                actions.append(torch.argmax(action_probabilities, dim=1))

        # Map action indices to Vectors
        action_space = [
            Vector(0, -1), Vector(0, 1), Vector(1, 0), Vector(-1, 0)
        ]
        actions = [[action_space[action.item()] for action in model_actions] for model_actions in actions]

        for j, (game, controller) in enumerate(zip(games, controllers)):
            for action in actions[j]:
                game.snake.v = action
                game.snake.move()
                fitness = calculate_fitness(game, controller)
                fitness_scores.append((fitness, batch_models[j]))
                batch_max_food_eaten.append(game.snake.score)

    return fitness_scores, batch_max_food_eaten


def calculate_fitness(game, controller):
    game.controller = controller
    food_eaten, total_steps = game.evaluate()

    initial_distance = game.snake.distance_to_food()
    previous_distance = initial_distance
    total_distance_reduction = 0
    penalties = 0
    deaths = 0

    # Calculate distance reduction
    current_distance = game.snake.distance_to_food()
    if current_distance < previous_distance:
        total_distance_reduction += (previous_distance - current_distance)

    # Calculate fitness score
    avg_steps = total_steps / food_eaten if food_eaten > 0 else total_steps
    fitness = (food_eaten * 5000 - deaths * 150 - avg_steps * 100 - penalties * 1000)
    return fitness

def save_best_model():
    if best_individual is not None:
        with gzip.open('best_model.pkl.gz', 'wb') as file:
            pickle.dump({
                'best_model': best_individual,
                'average_fitness_scores': average_fitness_scores,
                'best_scores': best_scores
            }, file)
        print('Best model saved.')

def signal_handler(sig, frame):
    print('Interrupt received, saving the best model...')
    save_best_model()
    exit(0)

# Register the signal handler for SIGINT
signal.signal(signal.SIGINT, signal_handler)

try:
    with gzip.open('best_model.pkl.gz', 'rb') as file:
        data = pickle.load(file)
        if isinstance(data, SimpleModel):
            best_model = data
            average_fitness_scores = []
            best_scores = []
        else:
            best_model = data['best_model']
            average_fitness_scores = data['average_fitness_scores']
            best_scores = data['best_scores']
        population = [best_model] + [SimpleModel(dims=(input_size, hidden_layer_size, output_size), init_weights=False) for _ in range(population_size - 1)]
        for model in population[1:]:  # Skip the first model (best_model)
            model.DNA = best_model.DNA  # Assign the weights of the best model to each new model
        print('Loaded existing model.')
except FileNotFoundError:
    print('No existing model found, starting fresh.')
except EOFError:
    print('The file is empty. Starting fresh.')

for generation in range(num_generations):
    fitness_scores = []
    max_gen_score = []

    batch_fitness_scores, batch_max_food_eaten = evaluate_fitness_batch(population, batch_size)
    fitness_scores.extend(batch_fitness_scores)
    max_gen_score.extend(batch_max_food_eaten)

    fitness_scores.sort(key=lambda x: x[0], reverse=True)
    selected_individuals = [individual for _, individual in fitness_scores[:population_size // 2]]
    next_generation = []
    for _ in range(population_size):
        parent1, parent2 = random.sample(selected_individuals, 2)
        offspring = parent1 + parent2
        offspring.mutate(mutation_rate)
        next_generation.append(offspring)

    population = next_generation
    average_fitness = sum(score for score, _ in fitness_scores) / population_size
    best_score = max(score for score, _ in fitness_scores)
    best_individual = max(fitness_scores, key=lambda x: x[0])[1]
    average_fitness_scores.append(round(average_fitness, 2))
    best_scores.append(round(best_score, 2))

    print(f'Generation {generation + 1}: Average Fitness = {round(average_fitness, 2)}, Best Score = {round(best_score, 2)}, '
          f'Highest food eaten = {max(max_gen_score)}')

# Save the best model and fitness scores after each generation
save_best_model()
