import jax
import jax.numpy as jnp
import pytest

from session_01_linear_classifier import (
    cross_entropy,
    init_params,
    linear_logits,
    make_dataset,
    train,
    update,
)


def test_dataset_is_deterministic_and_has_expected_contract():
    x1, y1 = make_dataset(jax.random.key(7), n_per_class=4)
    x2, y2 = make_dataset(jax.random.key(7), n_per_class=4)

    assert x1.shape == (12, 2)
    assert y1.shape == (12,)
    assert jnp.issubdtype(y1.dtype, jnp.integer)
    assert jnp.array_equal(x1, x2)
    assert jnp.array_equal(y1, y2)
    assert jnp.array_equal(jnp.bincount(y1, length=3), jnp.array([4, 4, 4]))


def test_logits_and_gradients_have_parameter_shapes():
    params = init_params(jax.random.key(1), n_features=2, n_classes=3)
    x = jnp.ones((5, 2))
    y = jnp.array([0, 1, 2, 0, 1])
    logits = linear_logits(params, x)
    loss, grads = jax.value_and_grad(cross_entropy)(params, x, y)

    assert logits.shape == (5, 3)
    assert jnp.isfinite(loss)
    assert grads["W"].shape == (2, 3)
    assert grads["b"].shape == (3,)
    assert jnp.all(jnp.isfinite(grads["W"]))
    assert jnp.all(jnp.isfinite(grads["b"]))

    new_params, reported_loss = update(params, x, y, learning_rate=0.2)
    assert jnp.isfinite(reported_loss)
    assert not jnp.array_equal(new_params["W"], params["W"])
    assert float(cross_entropy(new_params, x, y)) < float(reported_loss)


def test_cross_entropy_is_stable_for_extreme_logits():
    params = {
        "W": jnp.array([[1000.0, -1000.0, 0.0], [0.0, 0.0, 0.0]]),
        "b": jnp.zeros(3),
    }
    x = jnp.array([[1.0, 0.0], [0.0, 0.0]])
    y = jnp.array([0, 2])
    loss = cross_entropy(params, x, y)

    # First example has loss ~= 0; second has uniform-class loss log(3).
    assert jnp.isfinite(loss)
    assert float(loss) == pytest.approx(float(jnp.log(3.0) / 2), abs=1e-6)


def test_sgd_reduces_loss_learns_and_is_reproducible():
    key = jax.random.key(42)
    params, x, y, initial_loss, final_loss = train(key)
    logits = linear_logits(params, x)
    accuracy = jnp.mean(jnp.argmax(logits, axis=1) == y)
    params_2, _, _, initial_loss_2, final_loss_2 = train(key)

    assert float(final_loss) < float(initial_loss) * 0.15
    assert float(accuracy) >= 0.98
    assert float(initial_loss_2) == pytest.approx(float(initial_loss), abs=1e-7)
    assert float(final_loss_2) == pytest.approx(float(final_loss), abs=1e-7)
    assert jnp.allclose(params_2["W"], params["W"])
    assert jnp.allclose(params_2["b"], params["b"])

