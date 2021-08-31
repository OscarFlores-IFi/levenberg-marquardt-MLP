from typing import Iterable, List
from functools import reduce
import numpy as np

np.random.seed(555)


class Dense:
    def __init__(self, neurons: int, inputs: int):
        self.neurons = neurons
        self.inputs = inputs
        self.weights = 0.1 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

    def _forward(self, inputs: Iterable) -> np.array:
        return np.dot(inputs, self.weights) + self.biases


class Sequential:
    def __init__(self, layers: List[Dense] = [], name: str = ""):
        self.name = name
        self.layers = layers

    def add(self, layer: Dense) -> None:
        self.layers.append(layer)

    def feed_forward(self, input: Iterable) -> np.array:
        layers = self.layers.copy()
        layers.insert(0, input)

        return reduce(lambda l1, l2: l2._forward(l1), layers)

    def summary(self) -> str:
        string = f"{self.name} Sequential Model"
        if len(self.layers) == 0:
            string += "\nEmpty model"
        for layer in self.layers:
            string += f"\n{layer.__class__.__name__} layer - "
            string += f"input_shape: {layer.inputs} "
            string += f"- {layer.neurons} neurons"
        return string


# %% Valid Usage 1

model = Sequential(name="Testing")
model.add(Dense(neurons=3, inputs=4))
model.add(Dense(neurons=1, inputs=3))

X = np.array([1, 2, 3, 4])

print(model.feed_forward(X))

# %% Valid Usage 2
layers = [
    Dense(neurons=3, inputs=4),
    Dense(neurons=1, inputs=3),
]

model2 = Sequential(name="Testing 2", layers=layers)
print(model2.summary())

X = np.array([1, 2, 3, 4])

print(model2.feed_forward(X))
