"""Closed-resource Phase 0 retrieval diagnostic.

Implement this attempt before opening Session 1 artifacts or receiving tests.
"""

from types import new_class

import jax
import jax.numpy as jnp

# B, D, C = 7, 4, 5


def init_params(key: jax.Array, n_features: int, n_classes: int):
    """Return a parameter pytree satisfying the derived W/b contracts."""
    # Create random values for weights (n_features,n_classes) and biases
    weights = jax.random.normal(key, (n_features, n_classes))
    biases = jax.random.normal(key, (5,))
    return {"W": weights, "b": biases}


def linear_logits(params, x: jax.Array) -> jax.Array:
    """Return logits satisfying the derived batch/class shape contract."""
    W = params["W"]
    b = params["b"]

    return x @ W + b


def stable_cross_entropy(params, x: jax.Array, y: jax.Array) -> jax.Array:
    """Return stable mean multiclass cross-entropy."""
    logits = linear_logits(params, x)
    normalizer = jax.nn.logsumexp(logits, axis=1, keepdims=True)
    shifted = y - normalizer
    return jnp.mean(shifted, axis=1)


def sgd_update(params, x: jax.Array, y: jax.Array, learning_rate: float):
    """Return (new_params, pre_update_loss) from one pytree SGD update."""
    # will use value_and_grad here
    # I'm unclear on the exact way to use value_and_grad
    pre_update_loss, grad = jax.value_and_grad(stable_cross_entropy, params)
    new_params = params - learning_rate * params * grad

    return (new_params, pre_update_loss)


def make_fixture(key: jax.Array):
    """Return fixed-seed x and integer y satisfying the B/D/C contract."""
    x = jax.random.normal(key, (7, 4))
    y = jnp.array([0, 0, 1, 1, 2, 3, 4])

    return (x, y)


def main():
    """Construct the fixture and parameters, then verify one update."""
    # x should be input of (B,D)
    # y should be output of (B) with values that are classes [0,1,2,3,4]

    n_batches = 7
    n_features = 4
    n_classes = 5
    key = jax.random.key(seed=7)

    x, y = make_fixture(key)
    assert x.shape == (n_batches, n_features)
    assert y.shape == (n_batches,)

    params = init_params(key, n_features, n_classes)
    (new_params, pre_update_loss) = sgd_update(params, x, y, 0.1)


if __name__ == "__main__":
    main()
