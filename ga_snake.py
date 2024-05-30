import random
import pygame
from ga_models.ga_simple import SimpleModel
from ga_controller import GAController
from snake import SnakeGame, Food

# Define the dimensions for the neural network
input_size = 22  # Updated to match the new number of features
hidden_layer_size = 64  # Number of neurons in the hidden layer
output_size = 4  # Number of possible actions

# Parameters for the GA
population_size = 100
mutation_rate = 0.1
num_generations = 1000

# Initialize the population
population = [SimpleModel(dims=(input_size, hidden_layer_size, output_size))
              for _ in range(population_size)]

# Define the fitness function
def evaluate_fitness(model):
    game = SnakeGame(controller=None, max_steps=1000)
    controller = GAController(game=game, model=model, display=False)
    game.controller = controller

    initial_distance = game.snake.distance_to_food()
    previous_distance = initial_distance
    stagnant_steps = 0
    total_distance_reduction = 0
    unique_positions = set()
    food_eaten = 0

    running = True
    while running:
        next_move = controller.update()
        if next_move:
            game.snake.v = next_move
            game.snake.move()
            game.current_step += 1

            current_distance = game.snake.distance_to_food()
            if current_distance < previous_distance:
                total_distance_reduction += (previous_distance - current_distance)
            previous_distance = current_distance

            if game.snake.p in unique_positions:
                stagnant_steps += 1
            else:
                unique_positions.add(game.snake.p)
                stagnant_steps = 0

            if game.current_step >= game.max_steps or stagnant_steps > 100:
                running = False
            if not game.snake.p.within(game.grid) or game.snake.cross_own_tail:
                running = False
            if game.snake.p == game.food.p:
                game.snake.add_score()
                food_eaten += 1
                game.food = Food(game=game)

    final_distance = game.snake.distance_to_food()

    food_reward = food_eaten * 10000  # Increase reward for eating food
    collision_penalty = -10000 if not game.snake.p.within(game.grid) or game.snake.cross_own_tail else 0
    survival_reward = min(game.current_step * 10, 100)
    distance_penalty = final_distance * 200  # Increase penalty for distance to food
    distance_reward = total_distance_reduction * 1000  # Increase reward for reducing distance to food
    stagnation_penalty = stagnant_steps * 20  # Increase penalty for stagnation

    fitness = (food_reward + collision_penalty + survival_reward - distance_penalty + distance_reward - stagnation_penalty)

    return fitness

# Run the GA
try:
    best_model = SimpleModel.load('best_model.pkl')
    population = [best_model] + [SimpleModel(dims=(input_size, hidden_layer_size, output_size), init_weights=False) for _ in range(population_size - 1)]
    for model in population[1:]:  # Skip the first model (best_model)
        model.DNA = best_model.DNA  # Assign the weights of the best model to each new model
    print('Loaded existing model.')
except FileNotFoundError:
    print('No existing model found, starting fresh.')

for generation in range(num_generations):
    fitness_scores = []
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
    print(f'Generation {generation + 1}: Average Fitness = {average_fitness}, Best Score = {best_score}')

# Save the best model
best_individual = max(fitness_scores, key=lambda x: x[0])[1]
best_individual.save('best_model.pkl')
print('Best model saved.')
