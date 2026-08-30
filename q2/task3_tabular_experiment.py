"""Task 3: leakage-safe, fixed-seed tabular-baseline vertical slice.

Complete the first attempt before running it or receiving verification checks.
"""

import jax
import jax.numpy as jnp


N_ROWS, N_FEATURES = 240, 4
N_TRAIN, N_TEST = 168, 72


def make_dataset(key: jax.Array):
    """Return an intentionally imbalanced binary X: (240, 4), y: (240,)."""
    k0, k1 = jax.random.split(key, 2)
    centers = jnp.array(
        [[-2.0, -2.0, -2.0, -2.0], [2.0, 2.0, 2.0, 2.0]]
    )  # Two clusters for the binary case

    def gen_cluster(key_center, num_rows):
        (key, center) = key_center
        return center + 0.35 * jax.random.normal(key, (num_rows, N_FEATURES))

    num_0_rows = round(0.75 * N_ROWS)
    num_1_rows = N_ROWS - num_0_rows

    x0 = gen_cluster((k0, centers[0]), num_0_rows)
    x1 = gen_cluster((k1, centers[1]), num_1_rows)
    X = jnp.concat([x0, x1])
    y0 = jnp.zeros((num_0_rows), dtype=int)
    y1 = jnp.ones((num_1_rows), dtype=int)
    y = jnp.concat([y0, y1])
    assert X.shape == (N_ROWS, N_FEATURES)
    assert y.shape == (N_ROWS,)

    return (X, y)


def split_indices(key: jax.Array, n_rows: int):
    """Return disjoint train/test index arrays of lengths 168 and 72."""
    _, subkey = jax.random.split(key)

    split_idx = round(n_rows * 0.7)

    shuffled_indices = jax.random.permutation(subkey, n_rows)

    train_indices = shuffled_indices[:split_idx]
    test_indices = shuffled_indices[split_idx:]
    assert train_indices.shape == (N_TRAIN,)
    assert test_indices.shape == (N_TEST,)
    return (train_indices, test_indices)


def fit_standardizer(x_train: jax.Array):
    """Fit and return featurewise training-only mean and standard deviation."""
    mean = jnp.mean(x_train, axis=0)
    std = jnp.std(x_train, axis=0) + 1e-9
    assert mean.shape == (N_FEATURES,)
    assert std.shape == (N_FEATURES,)
    return mean, std


def transform_standardizer(x: jax.Array, mean: jax.Array, std: jax.Array):
    """Apply already-fitted standardisation without fitting new statistics."""
    result = (x - mean) / std
    assert result.shape == x.shape
    return result


def majority_predict(y_train: jax.Array, n_test: int):
    """Predict the training-set majority class for every test example."""
    num_zeros = jnp.sum(y_train == 0)
    num_ones = jnp.sum(y_train == 1)
    majority_test = []
    if num_ones > num_zeros:
        majority_test = [1] * n_test
    else:
        majority_test = [0] * n_test
    majority_test = jnp.array(majority_test)
    assert majority_test.shape == (n_test,)
    return majority_test


def balanced_accuracy(y_true: jax.Array, y_pred: jax.Array):
    """Return mean recall across the two classes."""
    # balanced_accuracy(y_true, y_pred) -> scalar
    recall0, recall1 = class_recalls(y_true, y_pred)
    mean = jnp.mean(jnp.array([recall0, recall1]))
    assert mean.shape == ()
    return mean


def class_recalls(y_true: jax.Array, y_pred: jax.Array):
    """Return (recall_class_0, recall_class_1)."""
    # class_recalls(y_true, y_pred) -> recall_0, recall_1
    num0 = jnp.sum((y_true == 0) & (y_pred == 0))
    denom0 = jnp.sum(y_true == 0)

    num1 = jnp.sum((y_true == 1) & (y_pred == 1))
    denom1 = jnp.sum(y_true == 1)
    if denom0 == 0 or denom1 == 0:
        raise ValueError("absent data")
    recall_0 = num0 / denom0
    recall_1 = num1 / denom1

    return (recall_0, recall_1)


def train_linear_model(key: jax.Array, x_train: jax.Array, y_train: jax.Array):
    """Train a fixed-hyperparameter binary linear model and return parameters."""
    # train_linear_model(...) -> params
    scale = 0.01
    lr = 0.1
    key, subkey = jax.random.split(key)
    weights = jax.random.normal(subkey, (N_FEATURES, 2)) * scale
    params = {"W": weights, "b": jnp.zeros(2)}

    def cross_entropy(params, x: jax.Array, y: jax.Array) -> jax.Array:
        """Mean negative log likelihood, stably computed from logits.

        Do not form softmax probabilities and then take log. Use logsumexp along
        the class axis, retain its dimension for subtraction, then gather the
        correct-class log probabilities with take_along_axis.
        """
        z = x @ params["W"] + params["b"]

        log_probs = z - jax.nn.logsumexp(z, axis=1, keepdims=True)
        correct = jnp.take_along_axis(log_probs, y[:, None], axis=1)
        loss = jnp.mean(-correct)
        return loss

    for i in range(20):
        (_, grad) = jax.value_and_grad(cross_entropy)(params, x_train, y_train)
        new_params = jax.tree.map(lambda p, g: p - lr * g, params, grad)
        params = new_params

    return params


def predict(params, x: jax.Array):
    """Return binary integer predictions with shape (batch,)."""
    # predict(params, X) -> integer labels
    preds = x @ params["W"] + params["b"]
    return jnp.argmax(preds, axis=1)


def run_experiment(seed: int = 0):
    """Run the complete split-before-fit experiment and return a result dict."""
    key = jax.random.PRNGKey(seed)
    data_key, split_key, train_key = jax.random.split(key, 3)
    x, Y = make_dataset(data_key)
    train_indx, test_indx = split_indices(split_key, N_ROWS)
    x_train = x[train_indx]
    mean, std = fit_standardizer(x_train)
    x_train = transform_standardizer(x_train, mean, std)
    y_train = Y[train_indx]

    x_test = x[test_indx]
    x_test = transform_standardizer(x_test, mean, std)
    y_test = Y[test_indx]
    params = train_linear_model(train_key, x_train, y_train)
    preds = predict(params, x_test)
    majority = majority_predict(y_train, N_TEST)
    recall0, recall1 = class_recalls(y_test, preds)
    train_class_counts = {
        "0": int(jnp.sum(y_train == 0)),
        "1": int(jnp.sum(y_train == 1)),
    }

    test_class_counts = {
        "0": int(jnp.sum(y_test == 0)),
        "1": int(jnp.sum(y_test == 1)),
    }
    learned_balanced_accuracy = balanced_accuracy(y_test, preds)
    majority_balanced_accuracy = balanced_accuracy(y_test, majority)
    target_met = bool(
        (learned_balanced_accuracy >= 0.80)
        & (learned_balanced_accuracy >= majority_balanced_accuracy + 0.25)
    )
    return {
        "seed": seed,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "train_class_counts": train_class_counts,
        "test_class_counts": test_class_counts,
        "scaler_fit_rows": len(x_train),  # must be n_train
        "majority_class": majority[0],
        "majority_balanced_accuracy": majority_balanced_accuracy,
        "learned_balanced_accuracy": learned_balanced_accuracy,
        "learned_recall_class_0": recall0,
        "learned_recall_class_1": recall1,
        "target_met": target_met,
    }


if __name__ == "__main__":
    results = run_experiment(seed=7)
    for name, value in results.items():
        print(f"{name}: {value}")
