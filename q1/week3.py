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
        return f"Neuron({len(self.weights)})"

    def zero_gradients(self):
        for w in self.weights:
            w.grad = 0.0
        self.bias.grad = 0.0

    def parameters(self):
        return self.weights + [self.bias]


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
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

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


def mini_batch(mlp: MLP, xs, ys, bs=4):
    # 2. Add mini-batch support (minimal version):
    #    - Either loop samples and accumulate grads, or average loss over batch then backprop once (depending on your design).
    total_set = zip(xs, ys)
    batches = batched(total_set, bs)

    for batch in batches:
        for xs, y in batch:
            xs = [Value(x, [], "x") for x in xs]
            # ys = [[0], [1], [0], [1]]
            out = mlp(xs)
            # get cross entropy loss
            # loss = (out[0] - Value(ys, [], "y")).square()  # loss is MSE
            # print(f"\nloss: {loss.data}")
            # zero all gradients
            for layer in layers:
                layer.zero_gradients()
            loss.backward()
            # update
            for layer in layers:
                for p in layer.parameters():
                    # print(f"previous: {p.data}")
                    p.data -= p.grad * learning_rate
                    # print(f"updated: {p.data}")


def w2_numerical_grad():
    layers = [create_layer(2, 4), create_layer(4, 1, non_lin="sigmoid")]
    mlp = MLP(layers)

    eps = 1e-4
    input_data = [0, 1]
    target = 1

    # First forward pass to get analytical gradient
    xs = [Value(x, [], "x") for x in input_data]
    out = mlp(xs)
    loss = (out[0] - Value(target, [], "y")).square()
    loss.backward()

    # Store the analytical gradient and weight value
    weight_to_check = mlp.layers[0].parameters()[4]
    analytical_gradient = weight_to_check.grad
    original_weight_value = weight_to_check.data

    print(f"Weight: {weight_to_check}")
    print(f"Analytical gradient: {analytical_gradient}")

    # Perturb weight up
    weight_to_check.data = original_weight_value + eps
    xs_plus = [Value(x, [], "x") for x in input_data]  # Fresh values
    out_plus = mlp(xs_plus)
    loss_plus = (out_plus[0] - Value(target, [], "y")).square()

    # Perturb weight down
    weight_to_check.data = original_weight_value - eps
    xs_minus = [Value(x, [], "x") for x in input_data]  # Fresh values
    out_minus = mlp(xs_minus)
    loss_minus = (out_minus[0] - Value(target, [], "y")).square()

    # Restore original weight
    weight_to_check.data = original_weight_value

    # Compute numerical gradient using centered difference
    numerical_gradient = (loss_plus.data - loss_minus.data) / (2 * eps)

    print(f"Numerical gradient: {numerical_gradient}")
    print(f"Difference: {abs(analytical_gradient - numerical_gradient)}")
    print(f"Match: {abs(analytical_gradient - numerical_gradient) < 1e-5}")


def xor():
    print("\nxor")
    xss = [[0, 0], [0, 1], [1, 1], [1, 0]]
    yss = [0, 1, 0, 1]
    learning_rate = 0.1
    layers = [create_layer(2, 8), create_layer(8, 1, non_lin="sigmoid")]
    mlp = MLP(layers)
    print(mlp.layers)
    for k in range(200):
        print(f"iteration: {k}")
        for xs, ys in zip(xss, yss):
            xs = [Value(x, [], "x") for x in xs]
            # ys = [[0], [1], [0], [1]]
            out = mlp(xs)
            loss = (out[0] - Value(ys, [], "y")).square()  # loss is MSE
            print(f"\nloss: {loss.data}")
            # zero all gradients
            for layer in layers:
                layer.zero_gradients()
            loss.backward()
            # update
            for layer in layers:
                for p in layer.parameters():
                    # print(f"previous: {p.data}")
                    p.data -= p.grad * learning_rate
                    # print(f"updated: {p.data}")


def two_layer_simple():
    xs = [Value(x, [], "x") for x in [1.0, 2.0, 3.0]]
    w1 = [Value(w, [], "w") for w in [2.0, 4.0, 6.0]]

    bias = 5.0
    neuron = Neuron(weights=w1, bias=bias)
    print(neuron(xs))
    n2 = default_neuron(3)
    print(n2)
    for n in n2.weights:
        print(n)

    # Testing with simple two layer
    # x = 2, y = 2
    x = Value(data=2, prev=[], label="x")
    y = [Value(data=2.0, prev=[], label="y")]
    z1 = Layer([Neuron([Value(0.1, [], label="w1")], 0.1)])
    # a1 = ReLU(z1) = .1*x+.1 = .3
    a1 = z1([x])
    print(f"a1: {a1}")
    # z2 = W2*a1 + b2 = .5*a1 + .1 = .25
    z2 = Layer([Neuron([Value(0.5, [], label="w2")], 0.1)], non_lin="sigmoid")
    # y_hat=sigmoid(z2) = .562
    y_hat = z2(a1)
    print(f"\ny_hat: {y_hat[0]}")
    # loss = (y_hat-y)^2 = (.562 - 2)^2 = 2.067
    loss = (y_hat[0] - y[0]) ** 2

    # print(y_hat)
    print(loss)
    loss.backward()
    print()
    print(loss)
    # Print the w2 values, gradient should be \frac{\partial L}{\partial w_2}=(\hat{y}-y)\hat{y}(1-\hat{y})a_1 * 2 = -0.212
    # w2 happens to be the last element of the z2 layer
    print("\nw2.grad should be ~ -0.212")
    print(z2.neurons[-1].weights)


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
            p.data -= lr * p.grad

    return loss.data


def main():
    random.seed(42)
    # print("week 2")

    # xor()

    # w2_numerical_grad()
    #
    # Week 3


if __name__ == "__main__":
    main()
