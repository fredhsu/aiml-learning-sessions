"""Canonical scalar reverse-mode autodiff engine and Value tensor helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Literal, Self


Scalar = float | int


class Value:
    """A scalar value in a reverse-mode autodiff computation graph."""

    def __init__(
        self,
        data: Scalar,
        prev: Iterable[Self] = (),
        label: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self.label = label
        self._prev = tuple(prev)
        self._backward = lambda: None

    def __repr__(self) -> str:
        label = f"{self.label}: " if self.label else ""
        return f"Value({label}data={self.data}, grad={self.grad})"

    @staticmethod
    def _coerce(other: Scalar | Self) -> Self:
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other: Scalar | Self) -> Self:
        other_value = self._coerce(other)
        out = Value(self.data + other_value.data, (self, other_value), "+")

        def backward() -> None:
            self.grad += out.grad
            other_value.grad += out.grad

        out._backward = backward
        return out

    def __radd__(self, other: Scalar) -> Self:
        return self + other

    def __mul__(self, other: Scalar | Self) -> Self:
        other_value = self._coerce(other)
        out = Value(self.data * other_value.data, (self, other_value), "*")

        def backward() -> None:
            self.grad += other_value.data * out.grad
            other_value.grad += self.data * out.grad

        out._backward = backward
        return out

    def __rmul__(self, other: Scalar) -> Self:
        return self * other

    def __neg__(self) -> Self:
        return self * -1.0

    def __sub__(self, other: Scalar | Self) -> Self:
        return self + (-self._coerce(other))

    def __rsub__(self, other: Scalar) -> Self:
        return self._coerce(other) - self

    def __truediv__(self, other: Scalar | Self) -> Self:
        return self * self._coerce(other) ** -1

    def __rtruediv__(self, other: Scalar) -> Self:
        return self._coerce(other) / self

    def __pow__(self, exponent: Scalar) -> Self:
        exponent = float(exponent)
        out = Value(self.data**exponent, (self,), "**")

        def backward() -> None:
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = backward
        return out

    def square(self) -> Self:
        return self**2

    def exp(self) -> Self:
        out = Value(math.exp(self.data), (self,), "exp")

        def backward() -> None:
            self.grad += out.data * out.grad

        out._backward = backward
        return out

    def log(self) -> Self:
        if self.data <= 0:
            raise ValueError("log is defined only for positive Value.data")
        out = Value(math.log(self.data), (self,), "log")

        def backward() -> None:
            self.grad += out.grad / self.data

        out._backward = backward
        return out

    def relu(self) -> Self:
        out = Value(max(0.0, self.data), (self,), "relu")

        def backward() -> None:
            self.grad += (self.data > 0.0) * out.grad

        out._backward = backward
        return out

    def sigmoid(self) -> Self:
        # This branch avoids overflow for large-magnitude inputs.
        if self.data >= 0:
            data = 1.0 / (1.0 + math.exp(-self.data))
        else:
            exp_x = math.exp(self.data)
            data = exp_x / (1.0 + exp_x)
        out = Value(data, (self,), "sigmoid")

        def backward() -> None:
            self.grad += out.data * (1.0 - out.data) * out.grad

        out._backward = backward
        return out

    def _topological_order(self) -> list[Self]:
        topo: list[Value] = []
        visited: set[Value] = set()

        def visit(value: Value) -> None:
            if value in visited:
                return
            visited.add(value)
            for parent in value._prev:
                visit(parent)
            topo.append(value)

        visit(self)
        return topo

    def backward(self) -> None:
        """Accumulate gradients of this scalar with respect to its ancestors."""
        topo = self._topological_order()
        self.grad = 1.0
        for value in reversed(topo):
            value._backward()

    def zero_grad(self) -> None:
        """Clear gradients on every value reachable from this value."""
        for value in self._topological_order():
            value.grad = 0.0


Vector = list[Value]
Matrix = list[Vector]


def values(data: Iterable[Scalar], label: str = "") -> Vector:
    """Create a vector of leaf Values."""
    return [Value(value, label=label) for value in data]


def sum_values(values_: Iterable[Value]) -> Value:
    total = Value(0.0, label="sum")
    for value in values_:
        total = total + value
    return total


def logsumexp(logits: Sequence[Value]) -> Value:
    """Stable log(sum(exp(logits)))."""
    if not logits:
        raise ValueError("logsumexp requires at least one logit")
    maximum = max(logit.data for logit in logits)
    return sum_values((logit - maximum).exp() for logit in logits).log() + maximum


def cross_entropy_from_logits(logits: Sequence[Value], target_index: int) -> Value:
    if not 0 <= target_index < len(logits):
        raise IndexError("target_index is outside the logits vector")
    return logsumexp(logits) - logits[target_index]


def matmul(inputs: Matrix, weights: Matrix) -> Matrix:
    """Matrix multiplication over Value scalars."""
    if not inputs or not weights or not inputs[0] or not weights[0]:
        raise ValueError("matmul requires non-empty matrices")
    input_width = len(inputs[0])
    if any(len(row) != input_width for row in inputs):
        raise ValueError("inputs must be rectangular")
    if any(len(row) != len(weights[0]) for row in weights):
        raise ValueError("weights must be rectangular")
    if input_width != len(weights):
        raise ValueError("matmul dimensions do not match")

    weight_columns = list(zip(*weights))
    return [
        [
            sum_values(value * weight for value, weight in zip(row, column))
            for column in weight_columns
        ]
        for row in inputs
    ]


def add_bias(matrix: Matrix, bias: Sequence[Value]) -> Matrix:
    if not matrix or not matrix[0]:
        raise ValueError("add_bias requires a non-empty matrix")
    width = len(matrix[0])
    if len(bias) != width:
        raise ValueError("bias width does not match matrix width")
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix must be rectangular")
    return [
        [value + bias[column] for column, value in enumerate(row)] for row in matrix
    ]


def sum_axis(matrix: Matrix, axis: Literal[0, 1] = 0) -> Vector:
    if not matrix or not matrix[0]:
        raise ValueError("sum_axis requires a non-empty matrix")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix must be rectangular")
    if axis == 0:
        return [sum_values(row[column] for row in matrix) for column in range(width)]
    return [sum_values(row) for row in matrix]


def sum_all(matrix: Matrix) -> Value:
    return sum_values(value for row in matrix for value in row)


def mean_all(matrix: Matrix) -> Value:
    if not matrix or not matrix[0]:
        raise ValueError("mean_all requires a non-empty matrix")
    return sum_all(matrix) / (len(matrix) * len(matrix[0]))
