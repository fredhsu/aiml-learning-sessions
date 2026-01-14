from collections.abc import Callable
from typing import Self


class Value:
    data: float
    grad: float
    _prev: list[Self]
    _backward: Callable[..., None]

    def __init__(self, data: float = 0.0, prev: list[Self] = []):
        self.data = data
        self.grad = 0.0
        self._prev = prev
        self._backward = lambda: None

    def __str__(self):
        return f"data: {self.data}, prev: {self._prev}"

    def __add__(self, other: Self):
        out = Value(data=self.data + other.data, prev=[self, other])

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Self):
        out = Value(data=self.data * other.data, prev=[self, other])

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    # Session 4
    def relu(self):
        out = self.data if self.data > 0 else 0
        out = Value(out, [self])

        def _backward():
            self.grad += (out.data > 0) * out.grad

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
        out = Value(out, [self])

        def _backward():
            self.grad += 2 * self.data * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        out = self.data - other.data
        out = Value(out, [self, other])

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out


def w2s2():
    w = Value(0)
    b = Value(0)
    xs = [Value(1), Value(2), Value(3)]
    ys = [Value(2), Value(4), Value(6)]

    learning_rate = 0.05
    for step in range(40):
        loss = Value(0)
        for x, y in zip(xs, ys):
            y_hat = w * x + b
            loss += (y_hat - y).square()
        loss.backward()
        # we should divide loss by N for the accruate total loss
        print(f"w.grad:{w.grad}, b.grad:{b.grad}")
        w.data -= learning_rate * w.grad
        b.data -= learning_rate * b.grad

        w.grad = 0.0
        b.grad = 0.0
        print(f"step: {step}, w.data:{w.data}, b.data:{b.data}, loss: {loss.data}")
        print()


def analytical_grad_s2():
    x = Value(2.0)
    y = Value(4.0)
    w = Value(1.3)
    b = Value(-0.7)

    y_hat = w * x + b
    loss = (y_hat - y).square()
    loss.backward()
    print(f"w.grad:{w.grad}, b.grad:{b.grad}")


def numerical_grad_s2():
    x = Value(2.0)
    y = Value(4.0)

    eps = 1e-4
    # w + eps
    # w_p = Value(1.3 + eps)
    w_p = Value(1.3)
    b_p = Value(-0.7 + eps)
    y_hat_p = w_p * x + b_p
    loss_p = (y_hat_p - y).square()

    # small changes to parameters
    # w_m = Value(1.3 - eps)
    w_m = Value(1.3)
    b_m = Value(-0.7 - eps)
    y_hat_m = w_m * x + b_m
    loss_m = (y_hat_m - y).square()

    grad_w_numeric = (loss_p.data - loss_m.data) / (2 * eps)
    grad_b_numeric = (loss_p.data - loss_m.data) / (2 * eps)
    print(f"grad_w_numeric: {grad_w_numeric}")
    print(f"grad_b_numeric: {grad_b_numeric}")


def main():
    # Week 2 Session 2
    w2s2()
    analytical_grad_s2()
    numerical_grad_s2()
    # x = Value(2.0)
    # y = Value(3.0)
    # z = x * y + y
    # z.backward()
    # print(f"z.data={z.data}")
    # print(f"dz/dx={x.grad}")
    # print(f"dz/dy={y.grad}")

    # x = Value(4.0)
    # y = Value(3.0)
    # z = x * x + y * y
    # z.backward()
    # print()
    # print(f"z.data={z.data}")
    # print(f"dz/dx={x.grad}")
    # print(f"dz/dy={y.grad}")

    # x = Value(4.0)
    # y = Value(3.0)
    # z = x * y * y + y * y
    # z.backward()
    # print()
    # print(f"z.data={z.data}")
    # print(f"dz/dx={x.grad}")
    # print(f"dz/dy={y.grad}")

    # # Session 4
    # # Case A: Gradient flows
    # x = Value(2.0)
    # y = x.relu()
    # y.backward()
    # print()
    # print(f"y.data = {y.data}")
    # print(f"x.grad = {x.grad}")

    # # Case B: Gradient dies
    # x = Value(-2.0)
    # y = x.relu()
    # y.backward()
    # print()
    # print(f"y.data = {y.data}")
    # print(f"x.grad = {x.grad}")

    # # Chain ReLU
    # x = Value(1.0)
    # w = Value(2.0)
    # z = (x * w).relu()
    # z.backward()
    # print()
    # print(f"z.data = {z.data}")
    # print(f"x.grad = {x.grad}")
    # print(f"w.grad = {w.grad}")

    # x = Value(1.0)
    # w = Value(2.0)
    # z = (x * w).relu()
    # z.backward()
    # print()
    # print(f"z.data = {z.data}")
    # print(f"x.grad = {x.grad}")
    # print(f"w.grad = {w.grad}")

    # x = Value(-1.0)
    # w = Value(2.0)
    # z = (x + w).relu()
    # z.backward()
    # print()
    # print(f"z.data = {z.data}")
    # print(f"x.grad = {x.grad}")
    # print(f"w.grad = {w.grad}")

    # week 2 session 1: 12-JAN
    # x = Value(2.0)
    # y = Value(4.0)
    # w = Value(-1.0)

    # y_hat = w * x
    # loss = (y_hat - y).square()
    # loss.backward()
    # print(f"loss: {loss.data}")
    # print(f"w.grad: {w.grad}")

    # learning_rate = 0.1
    # w.data -= learning_rate * w.grad

    # y_hat = w * x
    # loss = (y_hat - y).square()
    # loss.backward()
    # print(f"loss: {loss.data}")
    # print(f"w.grad: {w.grad}")

    # for step in range(10):
    #     y_hat = w * x
    #     loss = (y_hat - y).square()
    #     loss.backward()
    #     w.data -= learning_rate * w.grad
    #     w.grad = 0.0
    #     print(f"step: {step}, w.data:{w.data}, loss: {loss.data}")

    # print(f"loss: {loss.data}")


if __name__ == "__main__":
    main()
