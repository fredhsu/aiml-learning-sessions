"""Faded JIT/static-shape drill.

Complete every TODO from the stated shape contract before running this file.
Do not inspect earlier classifier implementations.
"""

import jax
from jax.errors import ConcretizationTypeError
import jax.numpy as jnp
import optax

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
batched_logits = jax.vmap(one_logits, in_axes=(None, 0))


def stable_loss(params, x: jax.Array, y: jax.Array) -> jax.Array:
    """Return mean stable multiclass CE; x: (B, D), y: (B,)."""
    logits = batched_logits(params, x)
    normalizer = jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    log_probs = logits - normalizer
    correct = jnp.take_along_axis(log_probs, y[:, None], axis=-1)
    return -jnp.mean(correct)


@jax.jit
def update(params, x: jax.Array, y: jax.Array, learning_rate: float):
    """Return (new_params, pre_update_loss) with unchanged pytree shapes."""
    values, grads = jax.value_and_grad(stable_loss)(params, x, y)
    new_params = jax.tree.map(
        lambda p, g: p - learning_rate * g,
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
    assert loss.shape == ()
    # TODO: apply update, then assert loss and every parameter leaf are finite
    #       and preserve their initial shapes.
    new_params, new_loss = update(params, x, y, 0.01)
    assert new_loss.shape == ()

    x2 = jax.random.normal(key_x2, (8, D))
    y2 = jnp.array([0, 3, 1, 2, 3, 0, 3, 2], dtype=jnp.int32)
    print("Using different JIT shape signature")
    new_params2, new_loss2 = update(params, x2, y2, 0.01)
    _, grads = jax.value_and_grad(stable_loss)(params, x, y)
    assert new_loss2.shape == ()
    for value in (loss, new_loss, new_loss2):
        assert bool(jnp.isfinite(value))
    for p, g, updated, updated2 in zip(
        jax.tree.leaves(params),
        jax.tree.leaves(grads),
        jax.tree.leaves(new_params),
        jax.tree.leaves(new_params2),
    ):
        assert p.shape == g.shape == updated.shape == updated2.shape
        assert bool(jnp.all(jnp.isfinite(g)))
        assert bool(jnp.all(jnp.isfinite(updated)))
        assert bool(jnp.all(jnp.isfinite(updated2)))

    try:
        count = bad_unique_count(y)
    except ConcretizationTypeError as e:
        print(f"An ConcretizationTypeError occurred: {e}")

    optax_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
    loss_difference = jnp.abs(loss - optax_loss)
    assert loss_difference <= 1e-6
    # central differences for every W[i,j] using h=1e-2
    h = 1e-2
    fd_grads_W = zeros_like(params["W"])
    for i in range(params["W"].shape[0]):
        for j in range(params["W"].shape[1]):
            e = jnp.zeros_like(params["W"])
            e = e.at[i, j].set(1.0)
            he = h * e
            p1 = {"W": params["W"] + he, "b": params["b"]}
            l1 = stable_loss(p1, x, y)
            p2 = {"W": params["W"] - he, "b": params["b"]}
            l2 = stable_loss(p2, x, y)
            fd_grads_W = fd_grads_W.at[i, j].set((l1 - l2) / (2 * h))

    max_diff = jnp.max(jnp.abs(grads["W"] - fd_grads_W))
    # compare resulting (D,C) matrix to grads['W']
    # print both max diff and assert tolerance
    print(f"loss difference: {loss_difference}, max diff: {max_diff}")
    assert bool(max_diff <= 2e-4)


if __name__ == "__main__":
    main()
