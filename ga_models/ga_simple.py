import random
from typing import Protocol, Tuple, List, Sequence
import numpy as np
from ga_models.ga_protocol import GAModel
from ga_models.activation import relu, sigmoid, tanh, softmax
import pickle


class SimpleModel(GAModel):
    def __init__(self, *, dims: Tuple[int, ...], init_weights: bool = True):
        assert len(dims) >= 2, 'Error: dims must be two or higher.'
        self.dims = dims
        self.DNA = []
        self.activation_functions = [
            tanh for _ in range(len(dims) - 2)] + [sigmoid]
        if init_weights:
            self.DNA = [2 * np.random.rand(dim, dims[i + 1]) - 1 for i, dim in enumerate(dims[:-1])]
        else:
            self.DNA = None

    def update(self, obs: Sequence) -> np.ndarray:
        x = np.array(obs)
        for layer, activation in zip(self.DNA, self.activation_functions):
            x = activation(np.dot(x, layer))
        return sigmoid(x)

    def action(self, obs: Sequence):
        probabilities = self.update(obs)
        action = np.argmax(probabilities)
        return action

    def mutate(self, mutation_rate) -> None:
        for layer_idx, layer in enumerate(self.DNA):
            if random.random() < mutation_rate:
                for row_idx in range(layer.shape[0]):
                    for col_idx in range(layer.shape[1]):
                        if random.random() < mutation_rate:
                            self.DNA[layer_idx][row_idx][col_idx] = random.uniform(
                                -1, 1)

    def __add__(self, other: "SimpleModel"):
        baby_DNA = []
        for mom_layer, dad_layer in zip(self.DNA, other.DNA):
            baby_layer = np.where(np.random.rand(
                *mom_layer.shape) > 0.5, mom_layer, dad_layer)
            baby_DNA.append(baby_layer)
        baby = type(self)(dims=self.dims)
        baby.DNA = baby_DNA
        return baby

    def DNA(self):
        return self.DNA

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
