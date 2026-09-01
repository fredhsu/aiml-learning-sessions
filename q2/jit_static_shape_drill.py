"""Faded JIT/static-shape drill.

Complete every TODO from the stated shape contract before running this file.
Do not inspect earlier classifier implementations.
"""

import jax
import jax.numpy as jnp

B, D, C = 5, 3, 4


def init_params(key: jax.Array):
    """Return {'W': (D, C), 'b': (C,)}."""
    key0, key1 = jax.random.split(key, 2)
    weights = jax.random.normal(key0, (D, C))
    biases = jax.random.normal(key1, (C,))
    return {"W": weights, "b": biases}


def one_logits(params, x_i: jax.Array) -> jax.Array:
    """Map x_i: (D,) to logits: (C,)."""
    return x_i @ params["W"] + params["b"]


# TODO: wrap one_logits with jax.vmap so parameters stay shared and only x's
#       leading (batch) axis is mapped. Write in_axes yourself; do not guess it
#       by running the file. Record in the notes what in_axes=(0, 0) and
#       in_axes=(None, None) would each do before you choose.
batched_logits = jax.vmap(one_logits, in_axes=(1, None))


def stable_loss(params, x: jax.Array, y: jax.Array) -> jax.Array:
    """Return mean stable multiclass CE; x: (B, D), y: (B,)."""
    logits = batched_logits(params, x)
    out = jax.nn.logsumexp(logits, axis=1)
    return jnp.mean(out.sum())


@jax.jit
def update(params, x: jax.Array, y: jax.Array, learning_rate: float):
    """Return (new_params, pre_update_loss) with unchanged pytree shapes."""
    values, grads = jax.value_and_grad(stable_loss)(params, x, y)
    new_params = jax.tree.map(
        lambda p, grad: (p["W"] - grad * learning_rate, p["b"] - grad * learning_rate),
        params,
        grads,
    )
    return (new_params, values)


@jax.jit
def bad_unique_count(y: jax.Array):
    """Intentionally invalid: run separately and record the exact error."""
    return jnp.unique(y).shape[0]


def main():
    key = jax.random.key(0)
    key_x, key_params, key_x2 = jax.random.split(key, 3)
    x = jax.random.normal(key_x, (B, D))
    y = jnp.array([0, 3, 1, 2, 3], dtype=jnp.int32)
    params = init_params(key_params)

    logits = batched_logits(params, x)
    assert logits.shape == (B, C)
    loss = stable_loss(params, x, y)
    assert loss.shape == (B,)
    # TODO: apply update, then assert loss and every parameter leaf are finite
    #       and preserve their initial shapes.
    new_params, new_loss = update(params, x, y, 0.01)
    assert new_loss.shape == (B,)
    assert new_params.shape == (D, C)

    x2 = jax.random.normal(key_x2, (8, D))
    y2 = jnp.array([0, 3, 1, 2, 3], dtype=jnp.int32)
    print("Using different JIT shape signature")
    new_params2, new_loss2 = update(params, x2, y2, 0.01)
    assert new_loss2.shape == (8,)
    assert new_params2.shape == (D, C)

    count = bad_unique_count(y)
    print(count)
    # TODO: independent reference check — the tutor authored the task and the
    #       tests, so neither is external evidence. Compare stable_loss against
    #       optax.softmax_cross_entropy_with_integer_labels (mean-reduced), and
    #       compare grads["W"] against a central finite-difference estimate.
    #       Print both max absolute differences. Predict each before running.
    # ! I don't know how to use optax.softmax_cross_entropy_with_integer_labels


if __name__ == "__main__":
    main()
