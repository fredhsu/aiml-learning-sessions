"""Canonical NumPy optimizers for the from-scratch experiments.

Each optimizer receives flat (or otherwise identically shaped) parameter and
gradient arrays and returns a new parameter array. Optimizer state is held by
the optimizer instance, never by the model or parameter array.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _arrays(params: ArrayLike, grads: ArrayLike) -> tuple[FloatArray, FloatArray]:
    parameters = np.asarray(params, dtype=float)
    gradients = np.asarray(grads, dtype=float)
    if parameters.shape != gradients.shape:
        raise ValueError(
            f"parameter shape {parameters.shape} does not match gradient shape {gradients.shape}"
        )
    return parameters, gradients


def _validate_hyperparameters(
    learning_rate: float,
    weight_decay: float = 0.0,
) -> None:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")


class Optimizer(ABC):
    """Interface shared by optimizers that operate on NumPy parameter arrays."""

    @abstractmethod
    def step(self, params: ArrayLike, grads: ArrayLike) -> FloatArray:
        """Return parameters after one update using gradients at ``params``."""

    def reset_state(self) -> None:
        """Reset state accumulated across calls to :meth:`step`."""


class SGD(Optimizer):
    """Gradient descent with optional decoupled weight decay."""

    def __init__(self, learning_rate: float, weight_decay: float = 0.0) -> None:
        _validate_hyperparameters(learning_rate, weight_decay)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def step(self, params: ArrayLike, grads: ArrayLike) -> FloatArray:
        parameters, gradients = _arrays(params, grads)
        decay = 1.0 - self.learning_rate * self.weight_decay
        return decay * parameters - self.learning_rate * gradients


class MomentumSGD(Optimizer):
    """SGD with an exponential moving average of gradients.

    Weight decay is decoupled: it shrinks parameters directly and is not added
    to the gradient/velocity accumulator.
    """

    def __init__(
        self,
        learning_rate: float,
        beta: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        _validate_hyperparameters(learning_rate, weight_decay)
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        self.learning_rate = learning_rate
        self.beta = beta
        self.weight_decay = weight_decay
        self.velocity: FloatArray | None = None

    def step(self, params: ArrayLike, grads: ArrayLike) -> FloatArray:
        parameters, gradients = _arrays(params, grads)
        if self.velocity is None:
            self.velocity = np.zeros_like(parameters)
        elif self.velocity.shape != parameters.shape:
            raise ValueError("parameter shape changed after momentum was initialized")

        self.velocity = self.beta * self.velocity + gradients
        decay = 1.0 - self.learning_rate * self.weight_decay
        return decay * parameters - self.learning_rate * self.velocity

    def reset_state(self) -> None:
        self.velocity = None


class Adam(Optimizer):
    """Adam with optional bias correction.

    This is deliberately Adam rather than AdamW: the historical experiments
    only used decoupled decay for SGD and momentum, not for Adam.
    """

    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        correction: bool = True,
    ) -> None:
        _validate_hyperparameters(learning_rate)
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1)")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.correction = correction
        self.first_moment: FloatArray | None = None
        self.second_moment: FloatArray | None = None
        self.step_count = 0

    def step(self, params: ArrayLike, grads: ArrayLike) -> FloatArray:
        parameters, gradients = _arrays(params, grads)
        if self.first_moment is None:
            self.first_moment = np.zeros_like(parameters)
            self.second_moment = np.zeros_like(parameters)
        elif self.first_moment.shape != parameters.shape:
            raise ValueError("parameter shape changed after Adam was initialized")

        self.first_moment = (
            self.beta1 * self.first_moment + (1.0 - self.beta1) * gradients
        )
        self.second_moment = (
            self.beta2 * self.second_moment + (1.0 - self.beta2) * gradients**2
        )
        self.step_count += 1

        if self.correction:
            first_moment = self.first_moment / (1.0 - self.beta1**self.step_count)
            second_moment = self.second_moment / (1.0 - self.beta2**self.step_count)
        else:
            first_moment = self.first_moment
            second_moment = self.second_moment

        return parameters - self.learning_rate * first_moment / (
            np.sqrt(second_moment) + self.epsilon
        )

    def reset_state(self) -> None:
        self.first_moment = None
        self.second_moment = None
        self.step_count = 0
