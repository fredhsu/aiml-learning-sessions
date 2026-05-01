# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.3",
#     "matplotlib==3.10.9",
#     "numpy==2.4.4",
#     "scikit-learn==1.8.0",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import math
    import random
    from dataclasses import dataclass
    from collections.abc import Callable
    from itertools import batched
    from typing import Self, override
    import numpy as np
    from pprint import pprint

    return Callable, Self, dataclass, math, np, override, pprint, random


@app.cell
def _(dataclass, np):
    @dataclass(frozen=True)
    class TrainedRun:
        flat_init: np.ndarray
        flat_sgd: np.ndarray
        flat_adam: np.ndarray
        features: np.ndarray
        labels: np.ndarray
        sgd_losses: np.ndarray
        adam_losses: np.ndarray
        schema: list[dict]

    def load_trained_run(path, schema):
        with np.load(path) as data:
            return TrainedRun(
                flat_init=data["flat_init"],
                flat_sgd=data["flat_sgd"],
                flat_adam=data["flat_adam"],
                features=data["X"],
                labels=data["y"],
                sgd_losses=data["sgd_losses"],
                adam_losses=data["adam_losses"],
                schema=schema,
            )

    return (load_trained_run,)


@app.cell
def _(Callable, Self, math, override):

    class Value:
        data: float
        grad: float
        label: str
        _prev: list[Self]
        _backward: Callable[..., None]

        def __init__(self, data: float, prev: list[Self], label: str):
            self.data = data
            self.grad = 0.0
            self._prev = prev
            self._backward = lambda: None
            self.label = label

        @override
        def __repr__(self):
            return f"label: {self.label} data: {self.data}, grad: {self.grad}, prev: {self._prev}"

        def __add__(self, other: float | int | Self):
            other_value = other if isinstance(other, Value) else Value(float(other), [], "")
            out = Value(
                data=self.data + other_value.data, prev=[self, other_value], label="+"
            )

            def _backward():
                self.grad += out.grad
                other_value.grad += out.grad

            out._backward = _backward
            return out

        def __mul__(self, other: float | int | Self):
            other_value = other if isinstance(other, Value) else Value(float(other), [], "")
            out = Value(
                data=self.data * other_value.data, prev=[self, other_value], label="*"
            )

            def _backward():
                self.grad += other_value.data * out.grad
                other_value.grad += self.data * out.grad

            out._backward = _backward
            return out

        def __truediv__(self, other: float | int | Self):
            other_value = other if isinstance(other, Value) else Value(float(other), [], "")
            return self * other_value**-1

        def __pow__(self, other: int):
            out = Value(data=self.data**other, prev=[self], label="^")

            def _backward():
                self.grad += other * self.data ** (other - 1) * out.grad

            out._backward = _backward
            return out

        def exp(self):
            out = Value(data=math.exp(self.data), prev=[self], label="exp")

            def _backward():
                self.grad += math.exp(self.data) * out.grad

            out._backward = _backward
            return out

        def log(self):
            out = Value(data=math.log(self.data), prev=[self], label="log")

            def _backward():
                self.grad += self.data**-1 * out.grad

            out._backward = _backward
            return out

        def relu(self):
            out = self.data if self.data > 0 else 0
            out = Value(out, [self], label="relu")

            def _backward():
                self.grad += (self.data > 0) * out.grad

            out._backward = _backward

            return out

        def sigmoid(self):
            data = 1.0 / (1.0 + math.exp(-self.data))
            out = Value(data, prev=[self], label="sigmoid")

            def _backward():
                self.grad += data * (1 - data) * out.grad

            out._backward = _backward
            return out

        def backward(self):
            topo: list[Value] = []
            visited: set[Value] = set()

            def build_graph(v: Value):
                if v not in visited:
                    visited.add(v)
                    for p in v._prev:
                        build_graph(p)
                    topo.append(v)

            build_graph(self)
            self.grad = 1
            for v in reversed(topo):
                v._backward()

        # week 2 session 1: 12-JAN
        def square(self):
            out = self.data**2
            out = Value(out, [self], label="square")

            def _backward():
                self.grad += 2 * self.data * out.grad

            out._backward = _backward
            return out

        def __neg__(self):
            return self * Value(-1, [], label="neg")

        def __sub__(self, other: float | int | Self):
            other_value = other if isinstance(other, Value) else Value(float(other), [], "")
            out = Value(self.data - other_value.data, [self, other_value], label="-")

            def _backward():
                self.grad += out.grad
                other_value.grad -= out.grad  # Fixed: subtraction gradient

            out._backward = _backward
            return out

        def __radd__(self, other: float | int | Self):
            return self + other


    return (Value,)


@app.cell
def _(Value):
    type Vector = list[Value]
    return (Vector,)


@app.cell
def _(Value, Vector):

    def sum_values(vs: Vector) -> Value:
        out = Value(0.0, [], "0")
        for v in vs:
            out = out + v
        return out


    def logsumexp(zs: Vector) -> Value:
        m = max(z.data for z in zs)
        exp = [(z - m).exp() for z in zs]
        return sum_values(exp).log() + m


    def cross_entropy_from_logits(logits: Vector, y_index: int) -> Value:
        return logsumexp(logits) - logits[y_index]

    return (cross_entropy_from_logits,)


@app.cell
def _(Value, Vector, math, override):

    class Neuron:
        weights: Vector
        bias: Value

        def __init__(self, weights: Vector, bias: float) -> None:
            self.weights = weights
            self.bias = Value(bias, [], "b")

        def __call__(self, inputs: Vector, non_lin: str = "relu"):
            weighted = [x * w for (x, w) in zip(inputs, self.weights)]
            out = self.bias
            for n in weighted:
                out = out + n

            if non_lin == "sigmoid":
                return out.sigmoid()
            if non_lin == "relu":
                return out.relu()
            else:
                # return raw pre-activation
                return out

        @override
        def __repr__(self):
            return f"Neuron:{[(n.label, n.data, n.grad) for n in self.parameters()]}"

        def zero_gradients(self):
            for w in self.weights:
                w.grad = 0.0
            self.bias.grad = 0.0

        def parameters(self):
            return self.weights + [self.bias]

        def print_weight_gradient_norm(self):
            sum_sq = sum([w.grad**2 for w in self.weights])
            norm = math.sqrt(sum_sq)
            print(f"Neuron gradient norm: |∇_w L|₂ = {norm:.4f} (weights only)")

    return (Neuron,)


@app.cell
def _(Neuron, Value, random):

    def default_neuron(nin: int, label: str = "w"):
        weights = [Value(random.uniform(-1, 1), [], label + str(i)) for i in range(nin)]
        bias = 0.0
        return Neuron(weights, bias)

    return (default_neuron,)


@app.cell
def _(Neuron, Vector, default_neuron, override):
    class Layer:
        neurons: list[Neuron]
        non_lin: str

        def __init__(self, neurons: list[Neuron], non_lin: str = "relu") -> None:
            self.neurons = neurons
            self.non_lin = non_lin

        def __call__(self, x: Vector):
            out = [n(x, self.non_lin) for n in self.neurons]
            return out

        @override
        def __repr__(self):
            return f"Layer of [{', '.join(str(len(n.weights)) for n in self.neurons)}]"

        def zero_gradients(self):
            for n in self.neurons:
                n.zero_gradients()

        def parameters(self):
            return [p for n in self.neurons for p in n.parameters()]

        def flat_parameters(self):
            weights_flat=[w.data for n in self.neurons for w in n.weights]
            biases_flat = [n.bias.data for n in self.neurons]
            return weights_flat + biases_flat

        def schema(self):
            n_in = len(self.neurons[0].weights)
            n_out = len(self.neurons)
            return {
                "W_shape": (n_out, n_in), 
                "b_shape": (n_out,),
                "activation":self.non_lin,
            }

    def create_layer(nin: int, nout: int, non_lin: str = "relu"):
        neurons = [default_neuron(nin, "w" + str(i)) for i in range(nout)]
        return Layer(neurons, non_lin=non_lin)

    return Layer, create_layer


@app.cell
def _(Layer, Vector, np):
    class MLP:
        layers: list[Layer]

        def __init__(self, layers: list[Layer]):
            self.layers = layers  # [::-1]

        def __call__(self, x: Vector):
            for layer in self.layers:
                x = layer(x)
            return x

        def flat_parameters(self):
            """Return all weights (row-major: neuron 0's incoming weights, then neuron 1's, ...) followed by all biases."""
            flat = []
            schema = []
            for layer in self.layers:
                flat.extend(layer.flat_parameters())
                schema.append(layer.schema())
            return np.array(flat), schema

    def unflatten(flat: np.ndarray, schema: list[dict]) -> list[dict]:
        """Convert flat parameter vector + schema into a list of per-layer dicts: [{"W": ndarray, "b": ndarray, "activation": str}, ...] """
        triples = []
        cursor = 0
        for layer in schema:
            (n_out, n_in) = layer["W_shape"]
            n_w = n_out * n_in
            (b_shape,) = layer["b_shape"]

            W = flat[cursor : cursor + n_w].reshape(n_out, n_in).copy()
            cursor += n_w
            b = flat[cursor : cursor + b_shape].copy()
            cursor += b_shape

            triples.append(
                {
                    "W":W,
                    "b":b,
                    "activation":layer["activation"]
                }
            )
        assert cursor == len(flat), f"Schema doesn't match flat length: {cursor} vs {len(flat)}"
        return triples



    return MLP, unflatten


@app.cell
def _(MLP, create_layer):
    def make_two_moons_mlp():
        return MLP([
            create_layer(2, 16, non_lin="relu"),
            create_layer(16, 16, non_lin="relu"),
            create_layer(16, 2, non_lin="None"),
        ])

    def two_moons_schema():
        _, schema = make_two_moons_mlp().flat_parameters()
        return schema

    return make_two_moons_mlp, two_moons_schema


@app.cell
def _(MLP, create_layer, random, unflatten):
    def check_schema():
        # Build a fresh MLP
        random.seed(42)
        mlp = MLP([
            create_layer(2, 16, non_lin="relu"),
            create_layer(16, 16, non_lin="relu"),
            create_layer(16, 2, non_lin="None"),
        ])

        # Flatten
        flat, schema = mlp.flat_parameters()
        print(f"Flat length: {len(flat)}")  # should be 2*16 + 16 + 16*16 + 16 + 16*2 + 2 = 354

        # Unflatten
        layers = unflatten(flat, schema)

        # Spot-check: pick a known parameter, find it in both representations
        # Layer 0, neuron 3, weight 1 (incoming from input dim 1)
        expected = mlp.layers[0].neurons[3].weights[1].data
        actual = layers[0]["W"][3, 1]
        print(f"Layer 0 neuron 3 weight 1: expected {expected}, got {actual}")
        assert abs(expected - actual) < 1e-12

        # Layer 0, neuron 5, bias
        expected_b = mlp.layers[0].neurons[5].bias.data
        actual_b = layers[0]["b"][5]
        print(f"Layer 0 neuron 5 bias: expected {expected_b}, got {actual_b}")
        assert abs(expected_b - actual_b) < 1e-12

        # Layer 2 (output), neuron 0, weight 7
        expected = mlp.layers[2].neurons[0].weights[7].data
        actual = layers[2]["W"][0, 7]
        print(f"Layer 2 neuron 0 weight 7: expected {expected}, got {actual}")
        assert abs(expected - actual) < 1e-12

    check_schema()
    return


@app.cell
def _(MLP, create_layer, pprint, random, unflatten):
    def check_schema2():
        random.seed(42)
        TRAIN_DATA = [
            ([-2.0, -1.0], 0),
            ([-1.5, -1.2], 0),
            ([0.0, 2.0], 1),
            ([0.5, 1.8], 1),
            ([2.0, -1.0], 2),
            ([1.5, -1.3], 2),
        ]
        VAL_DATA = [
            ([-2.2, -0.8], 0),
            ([-0.5, 2.2], 1),
            ([2.2, -0.7], 2),
        ]

        xs = [x for x, _ in TRAIN_DATA]
        ys = [y for _, y in TRAIN_DATA]

        layers = [create_layer(2, 4), create_layer(4, 4, non_lin="None")]

        mlp = MLP(layers)
        params, schema = mlp.flat_parameters()
        pprint(params)
        pprint(schema)
        print("now unflatten")
        triples = unflatten(params, schema)
        pprint(triples)
    check_schema2()
    return


@app.cell
def _(np, unflatten):
    def forward_loss_np(flat: np.ndarray, schema: list[dict], 
                        X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute average cross-entropy loss for the MLP defined by (flat, schema)
        on inputs X (shape (N, n_in)) and integer targets y (shape (N,)).
        Returns a Python float.
        """
        layers = unflatten(flat, schema)

        # Forward pass
        A = X
        for i, layer in enumerate(layers):
            Z = A @ layer["W"].T + layer["b"]   # shape (N, n_out)
            if layer["activation"] == "relu":
                A = np.maximum(Z, 0.0)
            elif layer["activation"] == "sigmoid":
                A = 1.0 / (1.0 + np.exp(-Z))
            else:
                A = Z   # output layer, no activation

        logits = A   # final pre-activation = logits, shape (N, K)

        # Cross-entropy with log-sum-exp trick, batched
        m = logits.max(axis=1, keepdims=True)              # shape (N, 1)
        log_sum_exp = m.squeeze(1) + np.log(np.exp(logits - m).sum(axis=1))   # shape (N,)
        correct_class_logits = logits[np.arange(len(y)), y]  # shape (N,)
        per_sample_loss = log_sum_exp - correct_class_logits

        return float(per_sample_loss.mean())

    return (forward_loss_np,)


@app.cell
def _(
    MLP,
    Value,
    create_layer,
    cross_entropy_from_logits,
    forward_loss_np,
    np,
    random,
):
    def check_forward():
        random.seed(42)
        mlp = MLP([
            create_layer(2, 16, non_lin="relu"),
            create_layer(16, 16, non_lin="relu"),
            create_layer(16, 2, non_lin="None"),
        ])

        # Small test batch
        X_test = np.array([[0.5, -0.3], [1.2, 0.8], [-0.7, 0.1]])
        y_test = np.array([0, 1, 0])

        # numpy forward
        flat, schema = mlp.flat_parameters()
        loss_np = forward_loss_np(flat, schema, X_test, y_test)

        # Value-engine forward
        total = 0.0
        for x_row, y_i in zip(X_test, y_test):
            xs = [Value(float(v), [], "x") for v in x_row]
            logits = mlp(xs)
            loss = cross_entropy_from_logits(logits, int(y_i))
            total += loss.data
        loss_ve = total / len(X_test)

        print(f"numpy loss:        {loss_np:.10f}")
        print(f"Value-engine loss: {loss_ve:.10f}")
        print(f"Difference:        {abs(loss_np - loss_ve):.2e}")
        assert abs(loss_np - loss_ve) < 1e-10

    check_forward()
    return


@app.cell
def _():
    from sklearn.datasets import make_moons
    import matplotlib.pyplot as plt

    def plot_moons():
        X, y = make_moons(n_samples=500, noise=0.20, random_state=42)
        # X shape: (500, 2), y shape: (500,) with values in {0, 1}
        plt.figure(figsize=(8, 5))
        plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Moon 1', alpha=0.7)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Moon 2', alpha=0.7)

        # Formatting
        plt.title("Two Moons Dataset Visualization")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()

    return make_moons, plt


@app.cell
def _(np):
    from abc import ABC, abstractmethod
    class Optimizer(ABC):
        @abstractmethod
        def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
            """Take in current parameter vector and gradient vector, return updated parameter vector."""
            pass

    return (Optimizer,)


@app.cell
def _(Optimizer):
    class SGD(Optimizer):
        def __init__(self, learning_rate, weight_decay=0.0):
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay

        def step(self, params, grads):
            return (1 - self.learning_rate * self.weight_decay) * params - self.learning_rate * grads

    return (SGD,)


@app.cell
def _(Optimizer, np):
    class MomentumSGD(Optimizer):
        def __init__(self, learning_rate, beta, weight_decay=0.0):
            self.learning_rate = learning_rate
            self.beta = beta
            self.weight_decay = weight_decay
            self.v = None  # initialized on first step

        def step(self, params, grads):
            if self.v is None:
                self.v = np.zeros_like(params)
            self.v = self.beta * self.v + grads
            return (1 - self.learning_rate * self.weight_decay) * params - self.learning_rate * self.v

    return (MomentumSGD,)


@app.cell
def _(Optimizer, np):
    class Adam(Optimizer):
        def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, correction=True):
            self.learning_rate = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.correction = correction
            self.m = None
            self.v = None
            self.step_counter = 1

        def step(self, params, grads):
            if self.m is None:
                self.m = np.zeros_like(params)
                self.v = np.zeros_like(params)
            self.m = self.beta1 * self.m + (1 - self.beta1) * grads
            self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2

            if self.correction:
                m_hat = self.m / (1 - self.beta1 ** self.step_counter)
                v_hat = self.v / (1 - self.beta2 ** self.step_counter)
            else:
                m_hat = self.m
                v_hat = self.v

            new_params = params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            self.step_counter += 1
            return new_params

    return (Adam,)


@app.cell
def _(SGD, np):
    def check_optimizer():
        def f(p): return p[0]**2 + 10 * p[1]**2
        def grad_f(p): return np.array([2*p[0], 20*p[1]])

        p = np.array([-3.0, 1.0])
        sgd = SGD(0.09)
        for _ in range(15):
            p = sgd.step(p, grad_f(p))
            print(p)
    check_optimizer()
    return


@app.cell
def _(Value, cross_entropy_from_logits, np):
    def train_mlp(mlp, X, y, optimizer, n_epochs, batch_size, seed=42):
        rng = np.random.default_rng(seed)
        N = len(X)

        value_params = [p for layer in mlp.layers for p in layer.parameters()]

        epoch_losses = []

        for epoch in range(n_epochs): 
            # Shuffle indicies
            idx = rng.permutation(N)

            epoch_loss = 0.0
            n_batches = 0

            for batch_start in range(0, N, batch_size):
                batch_idx = idx[batch_start : batch_start + batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                for p in value_params:
                    p.grad = 0.0

                batch_loss_value = Value(0.0, [], "loss")
                for x_row, y_i in zip(X_batch, y_batch):
                    xs = [Value(float(v), [], "x") for v in x_row]
                    logits = mlp(xs)
                    loss_i = cross_entropy_from_logits(logits, int(y_i))
                    batch_loss_value = batch_loss_value + loss_i

                batch_loss_value = batch_loss_value / len(X_batch)
                batch_loss_value.backward()

                params_np = np.array([p.data for p in value_params])
                grads_np = np.array([p.grad for p in value_params])

                new_params_np = optimizer.step(params_np, grads_np)

                for p, new_val in zip(value_params, new_params_np):
                    p.data = float(new_val)

                epoch_loss += batch_loss_value.data
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            epoch_losses.append(avg_loss)

            if epoch % 10 == 0 or epoch == n_epochs - 1:
                print(f"Epoch {epoch:3d} | loss {avg_loss:.4f}")

        return epoch_losses

    return (train_mlp,)


@app.cell
def _(MLP, MomentumSGD, create_layer, make_moons, train_mlp):
    def small_test_run():
        mlp = MLP([
            create_layer(2, 16, non_lin="relu"),
            create_layer(16, 16, non_lin="relu"),
            create_layer(16, 2, non_lin="None"),
        ])
        X, y = make_moons(n_samples=50, noise=0.20, random_state=42)

        optimizer = MomentumSGD(learning_rate=0.1, beta = 0.9)
        n_epochs = 20
        batch_size = 10
        print(train_mlp(mlp, X, y, optimizer, n_epochs, batch_size))


    return


@app.cell
def _(Adam, SGD, make_moons, make_two_moons_mlp, np, random, train_mlp):
    def full_two_moons_training():
        # Two-moons dataset
        X, y = make_moons(n_samples=500, noise=0.20, random_state=42)
        X = X.astype(np.float64)
        y = y.astype(np.int64)

        # Save the same init for both models by reseeding
        def fresh_mlp():
            random.seed(42)
            return make_two_moons_mlp()
        # train with SGD
        mlp_sgd = fresh_mlp()
        flat_init, schema = mlp_sgd.flat_parameters()
        sgd_losses = train_mlp(mlp_sgd, X, y, optimizer=SGD(learning_rate=0.05), n_epochs=200, batch_size=32, seed=42)
        flat_sgd, _ = mlp_sgd.flat_parameters()

        # train with Adam
        mlp_adam = fresh_mlp()
        # init verification that models are starting from same init

        mlp_adam_check = fresh_mlp()
        flat_check, _ = mlp_adam_check.flat_parameters()
        assert np.allclose(flat_init, flat_check), "Initialization not reproducible!"

        adam_losses = train_mlp(mlp_adam, X, y, optimizer=Adam(learning_rate=0.01), n_epochs=200, batch_size=32, seed=42)
        flat_adam, _ = mlp_adam.flat_parameters()

        assert not np.allclose(flat_sgd, flat_adam), "Final params are the same, but should be different"

        np.savez("w10_trained_models.npz", 
             flat_init=flat_init, flat_sgd=flat_sgd, flat_adam=flat_adam,
             X=X, y=y, sgd_losses=sgd_losses, adam_losses=adam_losses)
        # Print final losses
        print(f"\nSGD  final loss: {sgd_losses[-1]:.4f}")
        print(f"Adam final loss: {adam_losses[-1]:.4f}")

    full_two_moons_training()
    return


@app.cell
def _(load_trained_run, two_moons_schema):
    trained_run = load_trained_run("w10_trained_models.npz", two_moons_schema())
    return (trained_run,)


@app.cell
def _(forward_loss_np, np, plt, trained_run):
    def loss_slice_1d(flat_center, schema, X, y, direction, alphas):
        """Evaluate loss at center + alpha * direction for each alpha."""
        return np.array([forward_loss_np(flat_center + a * direction, schema, X, y) 
                         for a in alphas])

    def plot_sgd_loss(run):
        # Use the SGD-trained model as the center
        rng = np.random.default_rng(0)
        direction_naive = rng.standard_normal(len(run.flat_sgd))   # N(0, I), no normalization
        alphas = np.linspace(-1.0, 1.0, 51)

        losses_naive = loss_slice_1d(
            run.flat_sgd,
            run.schema,
            run.features,
            run.labels,
            direction_naive,
            alphas,
        )
        print(np.linalg.norm(run.flat_sgd))
        print(np.linalg.norm(direction_naive))
        plt.figure(figsize=(8, 4))
        plt.plot(alphas, losses_naive)
        plt.axvline(0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel("α (step along random direction)")
        plt.ylabel("loss")
        plt.title("1D loss slice — naive random direction (no normalization)")
        plt.grid(alpha=0.3)
        plt.savefig("w10_1d_slice_naive.png", dpi=120, bbox_inches='tight')
        plt.show()


    plot_sgd_loss(trained_run)
    return (loss_slice_1d,)


@app.cell
def _(np):
    def filter_normalize(direction_flat, params_flat, schema):
        """
        Rescale `direction_flat` per-filter to match the per-filter norms of `params_flat`.
        Bias entries in direction are set to zero.
        Returns a new flat array (same shape as direction_flat).
        """

        flat = []
        cursor = 0

        for layer in schema:
            (n_out, n_in) = layer["W_shape"]
            n_w = n_out * n_in
            (b_shape,) = layer["b_shape"]

            W = params_flat[cursor : cursor + n_w].reshape(n_out, n_in)
            W_norms = np.linalg.norm(W, axis=1, keepdims=True)
            D = direction_flat[cursor : cursor + n_w].reshape(n_out, n_in)
            D_norms = np.linalg.norm(D, axis=1, keepdims=True) + 1e-10
            cursor += n_w
            D = D * W_norms/D_norms
            flat.append(D.flatten())
            b = params_flat[cursor : cursor + b_shape]
            b = np.zeros_like(b)
            cursor += b_shape

            flat.append(b)

        return np.concatenate(flat)

    return (filter_normalize,)


@app.cell
def _(filter_normalize, loss_slice_1d, np, trained_run):
    rng = np.random.default_rng(0)
    direction_naive = rng.standard_normal(len(trained_run.flat_sgd))
    direction_norm = filter_normalize(
        direction_naive, trained_run.flat_sgd, trained_run.schema
    )
    alphas = np.linspace(-1.0, 1.0, 51)
    losses_naive = loss_slice_1d(
        trained_run.flat_sgd,
        trained_run.schema,
        trained_run.features,
        trained_run.labels,
        direction_naive,
        alphas,
    )
    return alphas, direction_norm, losses_naive


@app.cell
def _(filter_normalize, np, trained_run):
    def filter_normalizie_check(run):
        rng = np.random.default_rng(0)
        direction_naive = rng.standard_normal(len(run.flat_sgd))
        direction_norm = filter_normalize(direction_naive, run.flat_sgd, run.schema)

        print(f"Naive direction norm:      {np.linalg.norm(direction_naive):.3f}")
        print(f"Normalized direction norm: {np.linalg.norm(direction_norm):.3f}")
        print(f"Trained weights norm:      {np.linalg.norm(run.flat_sgd):.3f}")

        # Check per-filter norms in layer 0
        n_out, n_in = run.schema[0]["W_shape"]
        W0 = run.flat_sgd[:n_out*n_in].reshape(n_out, n_in)
        D0 = direction_norm[:n_out*n_in].reshape(n_out, n_in)
        print(f"\nLayer 0 per-filter norms (W vs D):")
        for j in range(n_out):
            print(f"  filter {j}: |W|={np.linalg.norm(W0[j]):.4f}, |D|={np.linalg.norm(D0[j]):.4f}")

    filter_normalizie_check(trained_run)
    return


@app.cell
def _(alphas, direction_norm, loss_slice_1d, losses_naive, plt, trained_run):
    def plot_normalized(run):
        losses_norm = loss_slice_1d(
            run.flat_sgd,
            run.schema,
            run.features,
            run.labels,
            direction_norm,
            alphas,
        )

        # Plot both naive and normalized side by side for direct comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=False)

        axes[0].plot(alphas, losses_naive)
        axes[0].axvline(0, color='k', linestyle='--', alpha=0.3)
        axes[0].set_xlabel("α")
        axes[0].set_ylabel("loss")
        axes[0].set_title("Naive random direction")
        axes[0].grid(alpha=0.3)

        axes[1].plot(alphas, losses_norm)
        axes[1].axvline(0, color='k', linestyle='--', alpha=0.3)
        axes[1].set_xlabel("α")
        axes[1].set_ylabel("loss")
        axes[1].set_title("Filter-normalized direction")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("w10_1d_slice_comparison.png", dpi=120, bbox_inches='tight')
        plt.show()

    plot_normalized(trained_run)
    return


@app.cell
def _(filter_normalize, forward_loss_np, np, trained_run):
    def loss_slice_2d(flat_center, schema, X, y, d1, d2, alphas, betas):
        """Evaluate loss at center + alpha*d1 + beta*d2 over a grid."""
        losses = np.zeros((len(alphas), len(betas)))
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                losses[i, j] = forward_loss_np(
                    flat_center + a * d1 + b * d2, schema, X, y
                )
            if i % 5 == 0:
                print(f"  row {i+1}/{len(alphas)}")
        return losses


    def _():

        # Two independent random directions, each filter-normalized
        rng = np.random.default_rng(0)
        d1_naive = rng.standard_normal(len(trained_run.flat_sgd))
        d2_naive = rng.standard_normal(len(trained_run.flat_sgd))

        d1 = filter_normalize(d1_naive, trained_run.flat_sgd, trained_run.schema)
        d2 = filter_normalize(d2_naive, trained_run.flat_sgd, trained_run.schema)

        # Grid: 25x25 = 625 evaluations. At ~0.04s each, ~25s total.
        alphas_2d = np.linspace(-1.0, 1.0, 25)
        betas  = np.linspace(-1.0, 1.0, 25)

        print("Computing 2D slice...")
        losses_2d = loss_slice_2d(
            trained_run.flat_sgd,
            trained_run.schema,
            trained_run.features,
            trained_run.labels,
            d1,
            d2,
            alphas_2d,
            betas,
        )
        print(f"d1·d2 / (|d1| |d2|) = {d1 @ d2 / (np.linalg.norm(d1) * np.linalg.norm(d2)):.4f}")
        return print(f"Done. Loss range: [{losses_2d.min():.4f}, {losses_2d.max():.4f}]")


    _()
    return (loss_slice_2d,)


@app.cell
def _(filter_normalize, loss_slice_2d, np, plt, trained_run):
    def plot_2d_loss():
        rng = np.random.default_rng(0)
        d1_naive = rng.standard_normal(len(trained_run.flat_sgd))
        d2_naive = rng.standard_normal(len(trained_run.flat_sgd))

        d1 = filter_normalize(d1_naive, trained_run.flat_sgd, trained_run.schema)
        d2 = filter_normalize(d2_naive, trained_run.flat_sgd, trained_run.schema)
        alphas = np.linspace(-1.0, 1.0, 25)
        betas  = np.linspace(-1.0, 1.0, 25)
        losses_2d = loss_slice_2d(
            trained_run.flat_sgd,
            trained_run.schema,
            trained_run.features,
            trained_run.labels,
            d1,
            d2,
            alphas,
            betas,
        )
        A, B = np.meshgrid(alphas, betas, indexing='ij')

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Linear scale
        cs1 = axes[0].contourf(A, B, losses_2d, levels=20, cmap='viridis')
        axes[0].contour(A, B, losses_2d, levels=20, colors='white', alpha=0.3, linewidths=0.5)
        axes[0].plot(0, 0, 'r*', markersize=15, label='trained minimum')
        axes[0].set_xlabel('α (direction 1)')
        axes[0].set_ylabel('β (direction 2)')
        axes[0].set_title('Linear scale')
        axes[0].legend()
        plt.colorbar(cs1, ax=axes[0])

        # Log scale (log(1 + loss) to handle near-zero values)
        cs2 = axes[1].contourf(A, B, np.log1p(losses_2d), levels=20, cmap='viridis')
        axes[1].contour(A, B, np.log1p(losses_2d), levels=20, colors='white', alpha=0.3, linewidths=0.5)
        axes[1].plot(0, 0, 'r*', markersize=15, label='trained minimum')
        axes[1].set_xlabel('α (direction 1)')
        axes[1].set_ylabel('β (direction 2)')
        axes[1].set_title('log(1 + loss) — reveals structure near minimum')
        axes[1].legend()
        plt.colorbar(cs2, ax=axes[1])

        plt.tight_layout()
        plt.savefig("w10_2d_slice_sgd.png", dpi=120, bbox_inches='tight')
        plt.show()

        # Min at center?
        min_idx = np.unravel_index(losses_2d.argmin(), losses_2d.shape)
        print(f"Min at grid index: {min_idx} (center is ({len(alphas)//2}, {len(betas)//2}))")
        print(f"Loss at center: {losses_2d[len(alphas)//2, len(betas)//2]:.4f}")
        print(f"Loss min:       {losses_2d.min():.4f}")

    plot_2d_loss()
    return


if __name__ == "__main__":
    app.run()
