import math
import random
from collections.abc import Callable
from itertools import batched
from typing import Self, override

## Week 3 Session 2 (60–90m) — Code from scratch: add BCE/CE + batch training + gradient check
# **Goal:** Your code stops being “it kinda works” and starts being *trustworthy*.

# ### Work plan (60–75m)
# 1. Implement the loss route you chose:
#    - Route A: sigmoid + BCE
#    - Route B: softmax + CE (softmax can be implemented stably via log-sum-exp)
# 2. Add mini-batch support (minimal version):
#    - Either loop samples and accumulate grads, or average loss over batch then backprop once (depending on your design).
# 3. Add a **finite-difference gradient check** utility:
#    - Pick one parameter $p$
#    - Approx: $\frac{\partial L}{\partial p} \approx \frac{L(p+\epsilon)-L(p-\epsilon)}{2\epsilon}$
#    - Compare to autograd gradient; print relative error
# 4. “Torture test”:
#    - Run grad-check on 2–3 randomly chosen params each run (small network, small data).

# ### Output (last 10m)
# - Commit code.
# - Note: **Week3_S2_Code_Notes**
#   - “What I implemented”
#   - “Known limitations”
#   - “Next obvious refactor”


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
        data = data = 1.0 / (1.0 + math.exp(-self.data))
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
        return other + self


type Vector = list[Value]


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


def default_neuron(nin: int, label: str = "w"):
    weights = [Value(random.uniform(-1, 1), [], label + str(i)) for i in range(nin)]
    bias = 0.0
    return Neuron(weights, bias)


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


def create_layer(nin: int, nout: int, non_lin: str = "relu"):
    neurons = [default_neuron(nin, "w" + str(i)) for i in range(nout)]
    return Layer(neurons, non_lin=non_lin)


class MLP:
    layers: list[Layer]

    def __init__(self, layers: list[Layer]):
        self.layers = layers  # [::-1]

    def __call__(self, x: Vector):
        for layer in self.layers:
            x = layer(x)
        return x


def mini_batch(mlp: MLP, xs, ys, bs=4, learning_rate=0.1):
    # 2. Add mini-batch support (minimal version):
    #    - Either loop samples and accumulate grads, or average loss over batch then backprop once (depending on your design).
    total_set = zip(xs, ys)
    batches = list(batched(total_set, bs))
    total_loss = 0.0
    for i, batch in enumerate(batches):
        batch_loss = Value(0.0, [], "loss")
        for xs, y in batch:
            xs = [Value(x, [], "x") for x in xs]
            # ys = [[0], [1], [0], [1]]
            logits = mlp(xs)
            # get cross entropy loss
            loss = cross_entropy_from_logits(logits, y)
            batch_loss = batch_loss + loss
            # zero all gradients
        for layer in mlp.layers:
            layer.zero_gradients()
        batch_loss = batch_loss / len(batch)  # average loss
        print(f"batch {i}: loss: {batch_loss.data}")
        batch_loss.backward()
        # update
        for layer in mlp.layers:
            for p in layer.parameters():
                p.data -= p.grad * learning_rate
        total_loss += batch_loss.data
    return total_loss


def w3_numerical_grad(mlp):
    eps = 1e-4
    input_data = [0, 1]
    target = 1

    # Zero all gradients first to start fresh
    for layer in mlp.layers:
        layer.zero_gradients()

    # First forward pass to get analytical gradient
    xs = [Value(x, [], "x") for x in input_data]
    logits = mlp(xs)  # Vector of length K
    loss = cross_entropy_from_logits(logits, target)

    loss.backward()

    # Store the analytical gradient and weight value
    weight_to_check = mlp.layers[1].parameters()[2]
    analytical_gradient = weight_to_check.grad
    original_weight_value = weight_to_check.data

    print(f"Weight: {weight_to_check}")
    print(f"Analytical gradient: {analytical_gradient}")

    # Perturb weight up
    weight_to_check.data = original_weight_value + eps
    xs_plus = [Value(x, [], "x") for x in input_data]  # Fresh values
    out_plus = mlp(xs_plus)
    loss_plus = cross_entropy_from_logits(out_plus, target)

    # Perturb weight down
    weight_to_check.data = original_weight_value - eps
    xs_minus = [Value(x, [], "x") for x in input_data]  # Fresh values
    out_minus = mlp(xs_minus)
    loss_minus = cross_entropy_from_logits(out_minus, target)

    # Restore original weight
    weight_to_check.data = original_weight_value

    # Compute numerical gradient using centered difference
    numerical_gradient = (loss_plus.data - loss_minus.data) / (2 * eps)

    print(f"Numerical gradient: {numerical_gradient}")
    print(f"Difference: {abs(analytical_gradient - numerical_gradient)}")

    # Use relative error for better comparison
    relative_error = abs(analytical_gradient - numerical_gradient) / (
        abs(analytical_gradient) + abs(numerical_gradient) + 1e-8
    )
    print(f"Relative error: {relative_error}")
    print(f"Match (relative error < 1e-5): {relative_error < 1e-5}")


def train_step(xs_raw: list[float], y_index: int, mlp: MLP, lr: float):
    xs = [Value(x, [], "x") for x in xs_raw]
    logits = mlp(xs)  # Vector of length K
    loss = cross_entropy_from_logits(logits, y_index)

    # zero grads
    for layer in mlp.layers:
        layer.zero_gradients()

    loss.backward()

    # SGD update
    for layer in mlp.layers:
        for p in layer.parameters():
            # logging parameter update magnitudes
            update_mag = lr * p.grad
            print(f"param {p.label} update -= {update_mag:.4f}")

            p.data -= lr * p.grad
        for i, neuron in enumerate(layer.neurons):
            print(f"Neuron {i}")
            neuron.print_weight_gradient_norm()

    return loss.data


def main():
    # random.seed(42)
    random.seed(142)
    TRAIN_DATA = [
        ([-2.0, -1.0], 0),
        ([-1.5, -1.2], 0),
        ([-2.2, -0.8], 0),
        ([0.0, 2.0], 1),
        ([0.5, 1.8], 1),
        ([-0.5, 2.2], 1),
        ([2.0, -1.0], 2),
        ([1.5, -1.3], 2),
        ([2.2, -0.7], 2),
    ]

    xs = [x for x, _ in TRAIN_DATA]
    ys = [y for _, y in TRAIN_DATA]

    layers = [create_layer(2, 4), create_layer(4, 4, non_lin="None")]
    for layer in layers:
        print(layer)
        print(layer.neurons)
        print()
    mlp = MLP(layers)

    # print("===== grad check (before training) =====")
    # w3_numerical_grad(mlp)
    # print()
    for i in range(1, 5):
        print(f"\n== Run number {i} ==")
        loss = train_step(xs_raw=xs[0], y_index=0, mlp=mlp, lr=0.1)
        print(f"== loss: {loss} ==\n")
    # for _ in range(50):
    #     epoch_loss = mini_batch(mlp, xs, ys)
    #     print(f"Epoch loss: {epoch_loss}\n")

    # for x, y in TRAIN_DATA:
    #     print(f"input: {x}, true: {y}, predict: {predict(x, mlp)}")


def predict(xs_raw, mlp):
    xs = [Value(x, [], "x") for x in xs_raw]
    logits = mlp(xs)
    return max(range(len(logits)), key=lambda i: logits[i].data)


if __name__ == "__main__":
    main()
