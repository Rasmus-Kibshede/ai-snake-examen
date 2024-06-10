import signal
import random
import numpy as np
import pickle
import gzip
from ga_models.ga_simple import SimpleModel
from ga_controller import GAController
from snake import SnakeGame, Food

# Define the dimensions for the neural network
input_size = 48 
hidden_layer_size = 64
output_size = 4

# Parameters for the GA
population_size = 500
mutation_rate = 0.05
num_generations = 1000

# Initialize the population
max_gen_score = []
population = [SimpleModel(dims=(input_size, hidden_layer_size, output_size))
              for _ in range(population_size)]

average_fitness_scores = []
best_scores = []
best_individual = None

def evaluate_fitness(model):
    game = SnakeGame(controller=None, max_steps=40000)
    controller = GAController(game=game, model=model, display=False)
    game.controller = controller

    initial_distance = game.snake.distance_to_food()
    previous_distance = initial_distance
    stagnant_steps = 0
    total_distance_reduction = 0
    unique_positions = set()
    food_eaten = 0
    total_steps = 0
    penalties = 0
    deaths = 0

    running = True
    while running:
        next_move = controller.update()
        if next_move:
            game.snake.v = next_move
            game.snake.move()
            game.current_step += 1
            total_steps += 1

            current_distance = game.snake.distance_to_food()
            if current_distance < previous_distance:
                total_distance_reduction += (previous_distance - current_distance)
            previous_distance = current_distance

            if game.snake.p in unique_positions:
                stagnant_steps += 1
            else:
                unique_positions.add(game.snake.p)
                stagnant_steps = 0

            if game.current_step >= game.max_steps:
                running = False
                # print("Terminated: Max steps reached")
            if stagnant_steps > 2500:
                running = False
                penalties += 1
                # print("Terminated: Stagnation")
            if not game.snake.p.within(game.grid):
                running = False
                deaths += 1
                # print("Terminated: Snake hit the wall")
            if game.snake.cross_own_tail:
                running = False
                deaths += 1
                # print("Terminated: Snake crossed its own tail")
            if game.snake.p == game.food.p:
                game.snake.add_score()
                food_eaten += 1
                game.food = Food(game=game)
    
    # Calculate average steps
    avg_steps = total_steps / food_eaten if food_eaten > 0 else total_steps

    # Calculate the fitness score
    fitness = (food_eaten * 5000 - deaths * 150 - avg_steps * 100 - penalties * 1000)

    max_gen_score.append(food_eaten)
    
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
    for idx, individual in enumerate(population):
        score = evaluate_fitness(individual)
        fitness_scores.append((score, individual))

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

    print(f'Generation {generation + 1}: Average Fitness = {round(average_fitness, 2)}, Best Score = {round(best_score, 2)}, 'f'Highest food eaten = {max(max_gen_score)}')

# Save the best model and fitness scores after each generation
save_best_model()
