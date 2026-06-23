"""
# Week 12 Session 2 Code - 1D convolution forward/backward pass

Standalone mini implementation using cross correlation convention, stride 1.
Not wired into the Value object for now, will address this later.

## Predictions
1. Forward sanity (x = ones(5), k = [1/3,1/3,1/3], valid, stride 1):
       First guess: [1, 1]  length 2.
       Corrected against the S1 output-length formula (n - m + 1 = 5-3+1 = 3):
       [1, 1, 1]  length 3.
2. Grad-check passing threshold (centered finite differences): rel err ~1e-8.
3. Most likely bug: off-by-one in the valid range.

## Results
Forward: [1,1,1] len 3 confirmed. Input of length 6 gave 4 outputs, boundary holds.
Backward: Derived both gradients, grad-check passes, both dx and dk are under 1e-8 threshold error

Derived gradients (for reference):
    dL/dk[j] = sum_{i=0}^{n-m}              g[i] * x[i+j]
    dL/dx[t] = sum_{i=max(0,t-m+1)}^{min(n-m,t)} g[i] * k[t-i]

## Debrief
- Predicted off-by-one bug was correct, and hit it twice at range boundaries:
  a. Allocating the list for forward: `[0.0]*m` initially, but it should be `[0.0]*(n-m+1)`
  b. Backward dx loop I initially did not have the right upper bound on range, it excluded the top term. It needed to be `min(n-m,t) + 1`.
- So I ended up commiting the error that I predicted would be a challenge. I need to think about boundaries more.
- Did not hit a conv/cross-correlation flip bug, stayed consistent with cross-correlation.
- Grad check was better than the predicted error.

"""

import numpy as np


def conv1d_backward(x, k, g):
    n = len(x)
    m = len(k)
    dx = [0.0] * n
    dk = [0.0] * m

    for j in range(m):
        for i in range(0, n - m + 1):
            dk[j] += g[i] * x[i + j]

    for t in range(n):
        for i in range(max(0, t - m + 1), min(n - m, t) + 1):
            dx[t] += g[i] * k[t - i]
    return (dx, dk)


def conv1d_forward(x, k):
    # Note this uses cross-correlation as a convetion instead of true convolution
    n = len(x)
    m = len(k)
    outputlen = n - m + 1
    y = [0.0] * outputlen
    for i in range(outputlen):
        for j in range(m):
            y[i] += x[i + j] * k[j]

    return y


def check_gradients(x, k, seed=0, eps=1e-5):
    rng = np.random.default_rng(seed)
    n, m = len(x), len(k)
    out_len = n - m + 1

    # Random weights w define a scalar loss L = sum_i w[i] * y[i].
    w = rng.standard_normal(out_len)

    def loss(x, k):
        y = conv1d_forward(x, k)
        return sum(w[i] * y[i] for i in range(out_len))

    # Analytic gradients. You need to supply g = dL/dy here.
    g = w  # what is dL/dy for L = sum_i w[i]*y[i]?
    dx, dk = conv1d_backward(x, k, g)

    # Numerical gradient for x via centered differences.
    dx_num = [0.0] * n
    for t in range(n):
        xp, xm = list(x), list(x)
        xp[t] += eps
        xm[t] -= eps
        dx_num[t] = (loss(xp, k) - loss(xm, k)) / (2 * eps)

    # Numerical gradient for k.
    dk_num = [0.0] * m
    for j in range(m):
        kp, km = list(k), list(k)
        kp[j] += eps
        km[j] -= eps
        dk_num[j] = (loss(x, kp) - loss(x, km)) / (2 * eps)

    def rel_err(a, b):
        a, b = np.array(a), np.array(b)
        denom = np.maximum(1e-12, np.abs(a) + np.abs(b))
        return np.max(np.abs(a - b) / denom)

    print("dx rel err:", rel_err(dx, dx_num))
    print("dk rel err:", rel_err(dk, dk_num))


def main():
    x = [1.0, 1.0, 1.0, 1.0, 1.0]
    k = [0.33, 0.33, 0.33]
    y = conv1d_forward(x, k)
    print(y)
    valid = [1.0, 1.0, 1.0]
    assert np.allclose(y, valid, 0.1, 0.1)
    x = [1.0] * 6
    k = [0.33, 0.33, 0.33]
    y = conv1d_forward(x, k)
    print(y)
    valid = [1.0, 1.0, 1.0, 1.0]
    assert np.allclose(y, valid, 0.1, 0.1)

    check_gradients(x, k, seed=0, eps=1e-5)


if __name__ == "__main__":
    main()
