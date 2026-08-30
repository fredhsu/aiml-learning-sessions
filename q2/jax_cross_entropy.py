import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import jax
    import jax.numpy as jnp

    return jax, jnp, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cross entropy using JAX

    Shape check:
    B = Batch size
    D = Number of features
    C = Number of classes

    Input x: (B, D)
    Parameters W: (D, C)
    Bias b: (C)
    Output y: (B) - will be integer class IDs
    """)
    return


@app.cell
def _(jax, jnp):
    def make_dataset(key: jax.Array, n_per_class: int = 32):
        """Return linearly separable x: (3*n_per_class, 2), y: (3*n_per_class,)."""
        # split key; sample three 2-D Gaussian clusters around the centres
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


    return init_params, make_dataset


@app.cell
def _(init_params, jax, make_dataset):
    x, y = make_dataset(jax.random.key(7), n_per_class=4)

    n_features = x.shape[
        1
    ]  # The number of columns, with this 2D example it should be 2 based on the second dimension of x
    n_classes = 3  # the integer labels, here it is [0, 1, 2]
    params = init_params(jax.random.key(1), n_features, n_classes)
    return n_classes, params, x, y


@app.cell
def dataset_scatterplot(n_classes, x, y):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for _class_id in range(n_classes):
        _mask = y == _class_id
        ax.scatter(
            x[_mask, 0],
            x[_mask, 1],
            label=f"Class {_class_id}",
            s=70,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.7,
        )

    ax.set(
        title="Synthetic classification dataset",
        xlabel="Feature 1",
        ylabel="Feature 2",
    )
    ax.legend(title="Label")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig
    return


@app.cell
def _(jax):
    def linear_logits(params, x:jax.Array) -> jax.Array:
        """x is (B,D), W is (D,C), b : (C,)"""
        W = params["W"]
        b = params["b"]
        return x @ W + b # (B,C)


    return (linear_logits,)


@app.cell
def _(linear_logits, params, x):
    def try_linear_logits():
        Z = linear_logits(params, x)
        print(Z.shape)
    try_linear_logits()
    return


@app.cell
def _(jax, jnp, linear_logits, params, x, y):
    def cross_entropy(params, x: jax.Array, y: jax.Array) -> jax.Array:
        """ Cross entropy will be the mean of the NLL """
        z = linear_logits(params, x)
        # Normalize Z by shifting by log(sum(e^z))
        # Taking logsumexp along the class axis (columns) to get the normalizer
        # Keep the dimensions so we can broadcast the subtraction for each class

        log_normalizer = jax.nn.logsumexp(z, axis=1, keepdims=True) # Shape (B, 1)

        # z: (B,C), log_normalizer: (B, 1) so we broadcast the logsumexp to each class column C
        log_probs = z - log_normalizer

        # Now we want to get just the logprobs for the correct column, so we use 
        # the values of y, converted to a column by adding a new axis to the array
        # then take_along_axis will select the correct column value. Use axis=1 to do this
        # across the columns
        correct = jnp.take_along_axis(log_probs, y[:,jnp.newaxis], axis=1) # (B,1)

        # Now get the loss by finding the mean of the negative values
        loss = jnp.mean(-correct) # scalar
        return loss

    cross_entropy(params, x, y)
    return (cross_entropy,)


@app.cell
def _(cross_entropy, init_params, jax, jnp):
    def make_dataset2(key: jax.Array, n_per_class: int = 32):
        """Return linearly separable x: (4*n_per_class, 3), y: (4*n_per_class,)."""
        # split key; sample three 3-D Gaussian clusters around the centres
        # Concatenate them and return integer labels 0, 1, 2, 3.
        k0, k1, k2, k3 = jax.random.split(key, 4)
        cluster_key = jnp.array([k0, k1, k2, k3])
        centers = jnp.array([[-2.0, -1.0, 0.0], [2.0, -1, 2.0], [0.0, 2.0, -1.0], [-1.0, -2.0, -2.0]])
        key_center = zip(cluster_key, centers)

        def gen_cluster(key_center):
            (key, center) = key_center
            return center + 0.35 * jax.random.normal(key, (n_per_class, 3))

        clusters = list(map(gen_cluster, key_center))
        x = jnp.concatenate([clusters[0], clusters[1], clusters[2], clusters[3]])
        y = jnp.concatenate(
            [jnp.full(n_per_class, 0), jnp.full(n_per_class, 1), jnp.full(n_per_class, 2), jnp.full(n_per_class, 3)], 
        )
        return x, y

    def test_cross_entropy_2():
        #
        x, y = make_dataset2(jax.random.key(7), n_per_class=6)
        assert x.shape == (24,3)
        assert y.shape == (24,)
        n_features = x.shape[1]  
        n_classes = 4  # the integer labels, here it is [0, 1, 2, 3]
        params = init_params(jax.random.key(1), n_features, n_classes)
        loss = cross_entropy(params, x, y)

    test_cross_entropy_2()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Update task

    Implement `update(params, x, y, learning_rate)` using `jax.value_and_grad`
    and a pytree SGD update. Return both the updated parameters and the pre-update
    loss. Before running it, predict whether one step should lower this batch's
    loss.
    """)
    return


@app.cell
def _(cross_entropy, jax):
    def update(params, x: jax.Array, y: jax.Array, learning_rate: float):
        (loss, grads) = jax.value_and_grad(cross_entropy)(params, x, y)
        new_params = jax.tree.map(
            lambda parameter, gradient: parameter - learning_rate * gradient,
            params,
            grads,
        )
        return new_params, loss

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
