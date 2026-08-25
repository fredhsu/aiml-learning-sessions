"""Session 1 scaffold: deterministic 3-class linear classifier in pure JAX.

Complete each TODO without consulting a full classifier implementation.  The
contract and the traced worked example are in the coaching message.
"""

import jax
import jax.numpy as jnp


def make_dataset(key: jax.Array, n_per_class: int = 32):
    """Return linearly separable x: (3*n_per_class, 2), y: (3*n_per_class,)."""
    # TODO: split key; sample three 2-D Gaussian clusters around the centres
    # [[-2, -1], [2, -1], [0, 2]], each with standard deviation 0.35.
    # Concatenate them and return integer labels 0, 1, 2.
    k0, k1, k2 = jax.random.split(key, 3)
    cluster_key = jnp.array([k0, k1, k2])
    centers = jnp.array([[-2.0, -1.0], [2.0, -1], [0.0, 2.0]])
    key_center = zip(cluster_key, centers)

    def gen_cluster(key_center):
        (key, center) = key_center
        return center + 0.35 * jax.random.normal(key, (n_per_class, 2))

    clusters = list(map(gen_cluster, key_center))
    x = jnp.concatenate([clusters[0], clusters[1], clusters[2]])
    y = jnp.concatenate(
        [jnp.full(n_per_class, 0), jnp.full(n_per_class, 1), jnp.full(n_per_class, 2)]
    )
    return x, y


def init_params(key: jax.Array, n_features: int, n_classes: int):
    """Return {'W': (n_features, n_classes), 'b': (n_classes,)}."""
    # TODO: small normal W (scale 0.01); zero b. Do not reuse a consumed key.
    scale = 0.01
    key, subkey = jax.random.split(key)
    W = jax.random.normal(subkey, (n_features, n_classes)) * scale
    b = jnp.zeros(n_classes)
    return {"W": W, "b": b}


def linear_logits(params, x: jax.Array) -> jax.Array:
    """Return x @ W + b, with shape (batch, n_classes)."""
    W = params["W"]
    b = params["b"]
    return x @ W + b


def cross_entropy(params, x: jax.Array, y: jax.Array) -> jax.Array:
    """Mean negative log likelihood, stably computed from logits.

    Do not form softmax probabilities and then take log. Use logsumexp along
    the class axis, retain its dimension for subtraction, then gather the
    correct-class log probabilities with take_along_axis.
    """
    z = linear_logits(params, x)

    log_probs = z - jax.nn.logsumexp(z, axis=1, keepdims=True)
    correct = jnp.take_along_axis(log_probs, y[:, None], axis=1)
    loss = jnp.mean(-correct)
    return loss


def update(params, x: jax.Array, y: jax.Array, learning_rate: float):
    """Return (new_params, pre_update_loss) using one full-batch SGD step."""
    # TODO: jax.value_and_grad(cross_entropy); tree-map p - lr * grad.
    (loss, grads) = jax.value_and_grad(cross_entropy)(params, x, y)
    new_params = jax.tree.map(
        lambda parameter, gradient: parameter - learning_rate * gradient,
        params,
        grads,
    )
    return new_params, loss


def train(key: jax.Array, steps: int = 200, learning_rate: float = 0.2):
    """Train and return (params, x, y, initial_loss, final_loss)."""
    # TODO: split a root key into independent data and parameter keys.
    # Record initial loss. Apply update exactly `steps` times. Record final loss.
    data_key, param_key = jax.random.split(key, 2)
    x, y = make_dataset(data_key)
    n_features = x.shape[1]
    n_classes = len(jnp.unique(y))
    params = init_params(param_key, n_features, n_classes)
    initial_loss = cross_entropy(params, x, y)
    for i in range(steps):
        params, _ = update(params, x, y, learning_rate)

    final_loss = cross_entropy(params, x, y)
    return (params, x, y, initial_loss, final_loss)
