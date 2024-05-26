import random
from typing import Protocol, Tuple, List, Sequence
import numpy as np
from ga_models.ga_protocol import GAModel
from ga_models.activation import sigmoid, tanh, softmax


class SimpleModel(GAModel):
    def __init__(self, *, dims: Tuple[int, ...]):
        assert len(dims) >= 2, 'Error: dims must be two or higher.'
        self.dims = dims
        self.DNA = []
        self.activation_functions = [sigmoid for _ in range(len(dims) - 2)] + [tanh]
        self.DNA = [np.random.rand(dim, dims[i+1]) for i, dim in enumerate(dims[:-1])]

    def update(self, obs: Sequence) -> Tuple[int, ...]:
        x = obs
        for layer, activation in zip(self.DNA, self.activation_functions):
            x = activation(np.dot(x, layer))
        return softmax(x)

    def action(self, obs: Sequence):
        # Prioritize actions that move towards food and away from obstacles
        if obs[4] < 0:  # Food is to the left
            return 3  # Move left
        elif obs[4] > 0:  # Food is to the right
            return 2  # Move right
        elif obs[5] < 0:  # Food is above
            return 0  # Move up
        elif obs[5] > 0:  # Food is below
            return 1  # Move down
        elif obs[8]:  # Tail is to the left
            return 3  # Move left
        elif obs[9]:  # Tail is to the right
            return 2  # Move right
        elif obs[10]:  # Tail is above
            return 0  # Move up
        elif obs[11]:  # Tail is below
            return 1  # Move down
        else:
            # If no food or tail is nearby, choose a random action
            return random.randint(0, 3)

    def mutate(self, mutation_rate) -> None:
        if random.random() < mutation_rate:
            random_layer = random.randint(0, len(self.DNA) - 1)
            row = random.randint(0, self.DNA[random_layer].shape[0] - 1)
            col = random.randint(0, self.DNA[random_layer].shape[1] - 1)
            self.DNA[random_layer][row][col] = random.uniform(-1, 1)

    def __add__(self, other):
        baby_DNA = []
        for mom, dad in zip(self.DNA, other.DNA):
            if random.random() > 0.5:
                baby_DNA.append(mom)
            else:
                baby_DNA.append(dad)
        baby = type(self)(dims=self.dims)
        baby.DNA = baby_DNA
        return baby

    def DNA(self):
        return self.DNA

