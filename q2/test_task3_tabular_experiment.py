import jax
import jax.numpy as jnp
import pytest

import task3_tabular_experiment as exp


def test_dataset_and_split_contracts_are_deterministic_and_disjoint():
    x1, y1 = exp.make_dataset(jax.random.key(1))
    x2, y2 = exp.make_dataset(jax.random.key(1))
    train1, test1 = exp.split_indices(jax.random.key(2), exp.N_ROWS)
    train2, test2 = exp.split_indices(jax.random.key(2), exp.N_ROWS)

    assert x1.shape == (240, 4)
    assert y1.shape == (240,)
    assert jnp.issubdtype(x1.dtype, jnp.floating)
    assert jnp.issubdtype(y1.dtype, jnp.integer)
    assert jnp.array_equal(jnp.bincount(y1, length=2), jnp.array([180, 60]))
    assert jnp.array_equal(x1, x2)
    assert jnp.array_equal(y1, y2)

    assert train1.shape == (168,)
    assert test1.shape == (72,)
    assert jnp.array_equal(train1, train2)
    assert jnp.array_equal(test1, test2)
    assert len(jnp.unique(train1)) == 168
    assert len(jnp.unique(test1)) == 72
    assert len(jnp.intersect1d(train1, test1)) == 0
    assert jnp.array_equal(
        jnp.sort(jnp.concatenate([train1, test1])), jnp.arange(exp.N_ROWS)
    )


def test_standardizer_fits_featurewise_and_transforms_without_refitting():
    x_train = jnp.array(
        [[1.0, 10.0, -2.0, 4.0], [3.0, 14.0, 0.0, 8.0], [5.0, 18.0, 2.0, 12.0]]
    )
    x_test = jnp.array([[7.0, 22.0, 4.0, 16.0]])
    mean, std = exp.fit_standardizer(x_train)
    train_scaled = exp.transform_standardizer(x_train, mean, std)
    test_scaled = exp.transform_standardizer(x_test, mean, std)

    assert mean.shape == (4,)
    assert std.shape == (4,)
    assert jnp.allclose(jnp.mean(train_scaled, axis=0), jnp.zeros(4), atol=1e-6)
    assert jnp.allclose(jnp.std(train_scaled, axis=0), jnp.ones(4), atol=1e-6)
    # If the test row had been refitted independently, it would transform to zero.
    assert not jnp.allclose(test_scaled, jnp.zeros_like(test_scaled))


def test_majority_baseline_and_metrics_match_manual_reference():
    y_train = jnp.array([0, 0, 0, 1])
    assert jnp.array_equal(exp.majority_predict(y_train, 3), jnp.array([0, 0, 0]))

    y_true = jnp.array([0, 0, 0, 0, 1, 1])
    y_pred = jnp.array([0, 0, 0, 1, 1, 0])
    recall_0, recall_1 = exp.class_recalls(y_true, y_pred)
    score = exp.balanced_accuracy(y_true, y_pred)

    assert float(recall_0) == pytest.approx(3 / 4)
    assert float(recall_1) == pytest.approx(1 / 2)
    assert float(score) == pytest.approx((3 / 4 + 1 / 2) / 2)

    with pytest.raises(ValueError):
        exp.class_recalls(jnp.array([0, 0]), jnp.array([0, 1]))


def test_run_experiment_fits_preprocessing_on_training_rows_and_is_reproducible(monkeypatch):
    fit_row_counts = []
    original_fit = exp.fit_standardizer

    def recording_fit(x_train):
        fit_row_counts.append(x_train.shape[0])
        return original_fit(x_train)

    monkeypatch.setattr(exp, "fit_standardizer", recording_fit)
    result1 = exp.run_experiment(seed=7)
    result2 = exp.run_experiment(seed=7)

    assert fit_row_counts == [168, 168]
    assert result1["n_train"] == 168
    assert result1["n_test"] == 72
    assert sum(result1["train_class_counts"].values()) == 168
    assert sum(result1["test_class_counts"].values()) == 72
    assert result1["scaler_fit_rows"] == 168
    assert float(result1["majority_balanced_accuracy"]) == pytest.approx(0.5)
    assert float(result1["learned_balanced_accuracy"]) >= 0.80
    assert float(result1["learned_balanced_accuracy"]) >= (
        float(result1["majority_balanced_accuracy"]) + 0.25
    )
    assert bool(result1["target_met"])

    for key in (
        "majority_balanced_accuracy",
        "learned_balanced_accuracy",
        "learned_recall_class_0",
        "learned_recall_class_1",
    ):
        assert float(result2[key]) == pytest.approx(float(result1[key]), abs=1e-7)
    assert result2["train_class_counts"] == result1["train_class_counts"]
    assert result2["test_class_counts"] == result1["test_class_counts"]
