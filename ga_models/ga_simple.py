import random
import numpy as np
import torch
import torch.nn as nn
import pickle
from typing import Tuple, Sequence
from ga_models.ga_protocol import GAModel

class SimpleModel(GAModel):
    def __init__(self, dims: Tuple[int, ...], init_weights: bool = True):
        assert len(dims) >= 2, 'Error: dims must be two or higher.'
        self.dims = dims
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = nn.Sequential()
        for i in range(len(dims) - 2):
            self.model.add_module(f"layer_{i}", nn.Linear(dims[i], dims[i+1]))
            self.model.add_module(f"activation_{i}", nn.Tanh())
        self.model.add_module("output_layer", nn.Linear(dims[-2], dims[-1]))
        self.model.add_module("output_activation", nn.Sigmoid())

        if init_weights:
            self.model.apply(self._init_weights)

        self.model.to(self.device)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.uniform_(layer.weight, -1.0, 1.0)

    def _get_DNA(self):
        return [param.data.cpu().numpy() for param in self.model.parameters()]

    def _set_DNA(self, DNA):
        with torch.no_grad():
            for param, dna in zip(self.model.parameters(), DNA):
                param.data = torch.tensor(dna).to(self.device)

    @property
    def DNA(self):
        return self._get_DNA()

    @DNA.setter
    def DNA(self, value):
        self._set_DNA(value)

    def update(self, obs: Sequence) -> np.ndarray:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(obs_tensor)
        return output.cpu().numpy()

    def action(self, obs: Sequence):
        probabilities = self.update(obs)
        action = np.argmax(probabilities)
        return action

    def mutate(self, mutation_rate) -> None:
        for param in self.model.parameters():
            if random.random() < mutation_rate:
                with torch.no_grad():
                    param.add_(torch.randn_like(param) * mutation_rate)

    def __add__(self, other: "SimpleModel"):
        baby_model = SimpleModel(dims=self.dims)
        baby_DNA = [(np.where(np.random.rand(*mom.shape) > 0.5, mom, dad)) for mom, dad in zip(self.DNA, other.DNA)]
        baby_model.DNA = baby_DNA
        return baby_model

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
