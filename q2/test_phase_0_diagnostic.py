import jax
import jax.numpy as jnp
import pytest

from phase_0_diagnostic_attempt import (
    B,
    C,
    D,
    init_params,
    make_fixture,
    sgd_update,
    stable_cross_entropy,
)
from phase_0_seeded_variants import (
    variant_1_global_reduction,
    variant_2_batch_shaped_params,
    variant_3_jitted_loss,
)


def zero_params():
    return {"W": jnp.zeros((D, C)), "b": jnp.zeros((C,))}


def extreme_fixture():
    params = {
        "W": jnp.array(
            [[1000.0, -1000.0, 0.0, 0.0, 0.0], [0.0] * 5, [0.0] * 5, [0.0] * 5]
        ),
        "b": jnp.zeros(C),
    }
    x = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    y = jnp.array([0, 4])
    return params, x, y


def test_task_1_fixture_parameter_gradient_and_update_contract():
    x, y = make_fixture(jax.random.key(11))
    params = init_params(jax.random.key(12), D, C)
    loss, grads = jax.value_and_grad(stable_cross_entropy)(params, x, y)

    assert x.shape == (B, D)
    assert y.shape == (B,)
    assert params["W"].shape == (D, C)
    assert params["b"].shape == (C,)
    assert loss.shape == ()
    assert grads["W"].shape == (D, C)
    assert grads["b"].shape == (C,)
    assert jnp.isfinite(loss)
    assert jnp.all(jnp.isfinite(grads["W"]))
    assert jnp.all(jnp.isfinite(grads["b"]))

    starting = zero_params()
    before = stable_cross_entropy(starting, x, y)
    updated, reported = sgd_update(starting, x, y, learning_rate=0.1)
    after = stable_cross_entropy(updated, x, y)
    assert float(reported) == pytest.approx(float(before), abs=1e-7)
    assert float(after) < float(before)


def test_task_1_cross_entropy_is_stable_and_scalar():
    params, x, y = extreme_fixture()
    loss = stable_cross_entropy(params, x, y)

    assert loss.shape == ()
    assert jnp.isfinite(loss)
    assert float(loss) == pytest.approx(float(jnp.log(5.0) / 2), abs=1e-6)


def test_variant_1_normalises_each_example_independently():
    params, x, y = extreme_fixture()
    loss = variant_1_global_reduction(params, x, y)
    assert loss.shape == ()
    assert float(loss) == pytest.approx(float(jnp.log(5.0) / 2), abs=1e-6)


def test_variant_2_uses_feature_and_class_dimensions():
    x, _ = make_fixture(jax.random.key(3))
    params = variant_2_batch_shaped_params(jax.random.key(4), x, C)
    assert params["W"].shape == (D, C)
    assert params["b"].shape == (C,)


def test_variant_3_has_no_data_dependent_shape_inside_jit():
    x, y = make_fixture(jax.random.key(5))
    params = zero_params()
    loss = variant_3_jitted_loss(params, x, y)
    reference = stable_cross_entropy(params, x, y)
    assert loss.shape == ()
    assert jnp.isfinite(loss)
    assert float(loss) == pytest.approx(float(reference), abs=1e-6)
