"""Small neural-network building blocks backed by :mod:`engine.value`."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Literal

from .value import Scalar, Value, Vector, sum_values


Activation = Literal["linear", "relu", "sigmoid"]


class Neuron:
    def __init__(self, weights: Sequence[Value], bias: Value | Scalar = 0.0) -> None:
        self.weights = list(weights)
        self.bias = bias if isinstance(bias, Value) else Value(bias, label="b")

    def __call__(self, inputs: Sequence[Value], activation: Activation = "relu") -> Value:
        if len(inputs) != len(self.weights):
            raise ValueError(
                f"expected {len(self.weights)} inputs, received {len(inputs)}"
            )
        output = self.bias + sum_values(
            input_ * weight for input_, weight in zip(inputs, self.weights)
        )
        if activation == "linear":
            return output
        if activation == "relu":
            return output.relu()
        if activation == "sigmoid":
            return output.sigmoid()
        raise ValueError(f"unsupported activation: {activation}")

    def parameters(self) -> Vector:
        return [*self.weights, self.bias]

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.grad = 0.0


def default_neuron(
    input_size: int,
    label: str = "w",
    rng: random.Random | None = None,
) -> Neuron:
    if input_size < 1:
        raise ValueError("input_size must be positive")
    rng = rng or random
    weights = [
        Value(rng.uniform(-1, 1), label=f"{label}{index}")
        for index in range(input_size)
    ]
    return Neuron(weights)


class Layer:
    def __init__(self, neurons: Sequence[Neuron], activation: Activation = "relu") -> None:
        if not neurons:
            raise ValueError("a layer requires at least one neuron")
        self.neurons = list(neurons)
        self.activation = activation

    def __call__(self, inputs: Sequence[Value]) -> Vector:
        return [neuron(inputs, self.activation) for neuron in self.neurons]

    def parameters(self) -> Vector:
        return [
            parameter for neuron in self.neurons for parameter in neuron.parameters()
        ]

    def zero_grad(self) -> None:
        for neuron in self.neurons:
            neuron.zero_grad()


def create_layer(
    input_size: int,
    output_size: int,
    activation: Activation = "relu",
    rng: random.Random | None = None,
) -> Layer:
    if output_size < 1:
        raise ValueError("output_size must be positive")
    return Layer(
        [
            default_neuron(input_size, label=f"w{index}_", rng=rng)
            for index in range(output_size)
        ],
        activation,
    )


class MLP:
    def __init__(self, layers: Sequence[Layer]) -> None:
        if not layers:
            raise ValueError("an MLP requires at least one layer")
        self.layers = list(layers)

    def __call__(self, inputs: Sequence[Value]) -> Vector:
        output = list(inputs)
        for layer in self.layers:
            output = layer(output)
        return output

    def parameters(self) -> Vector:
        return [
            parameter for layer in self.layers for parameter in layer.parameters()
        ]

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.grad = 0.0


def create_mlp(
    input_size: int,
    layer_sizes: Sequence[int],
    activations: Sequence[Activation] | None = None,
    seed: int | None = None,
) -> MLP:
    """Create an MLP whose final activation defaults to ``linear``."""
    if not layer_sizes:
        raise ValueError("layer_sizes must contain at least one output size")
    if activations is None:
        activations = ["relu"] * (len(layer_sizes) - 1) + ["linear"]
    if len(activations) != len(layer_sizes):
        raise ValueError("activations and layer_sizes must have the same length")

    rng = random.Random(seed)
    layers: list[Layer] = []
    previous_size = input_size
    for output_size, activation in zip(layer_sizes, activations):
        layers.append(create_layer(previous_size, output_size, activation, rng))
        previous_size = output_size
    return MLP(layers)
