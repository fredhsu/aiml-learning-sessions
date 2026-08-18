"""
# Week 12 Session 2 Code - 1D convolution forward/backward pass

Standalone mini-implementation using cross-correlation convention (no filter
flip), valid convolution, stride 1. Intentionally not wired into the Value
engine; that integration is a later cleanup task.

## Predictions
1. Forward sanity (x = ones(5), k = [1/3,1/3,1/3], valid, stride 1):
       First guess: [1, 1]  length 2.
       Corrected against the S1 output-length formula (n - m + 1 = 5-3+1 = 3):
       [1, 1, 1]  length 3.
2. Grad-check passing threshold (centered finite differences): rel err ~1e-8.
3. Most likely bug: off-by-one in the valid range.

## Results
Forward:  [1,1,1] len 3 confirmed; length-6 input -> 4 outputs, boundary holds.
Backward: derived both gradients from the Toeplitz view, grad-check passes.
       dx rel err: 2.6e-10
       dk rel err: 2.3e-12   (both well under the 1e-8 threshold)

## Calibration Debrief
- Predicted bug (off-by-one) was correct, and it bit TWICE, both at valid-range
  boundaries:
    (a) forward allocation `[0.0]*m` should be `[0.0]*(n-m+1)` — invisible in the
        n=5,m=3 case because m == n-m+1; exposed by a length-6 input.
    (b) backward dx loop used an exclusive `range` upper bound where the
        derivation has an INCLUSIVE bound: needs `min(n-m, t) + 1`. Dropped the
        top term (e.g. x[0] got zero gradient) until fixed.
- Sharper signal than "I predicted my bug": I also COMMITTED the off-by-one in my
  own forward prediction ([1,1] before the formula corrected me). Boundary
  reasoning is genuinely soft — confirmed in both the prediction and the code.
  Carry this to Week 13.
- No conv/cross-correlation flip bug. Stayed consistent with cross-correlation
  forward; the "flip" appeared correctly AS MATH (the k[t-i] index reversal in
  dx) rather than as a bug.
- Grad-check landed at 1e-10..1e-12, slightly better than the predicted 1e-8 —
  normal for centered differences at eps=1e-5.

Derived gradients (for reference):
    dL/dk[j] = sum_{i=0}^{n-m}              g[i] * x[i+j]
    dL/dx[t] = sum_{i=max(0,t-m+1)}^{min(n-m,t)} g[i] * k[t-i]
"""

import numpy as np


def conv1d_forward(x, k):
    n = len(x)
    m = len(k)
    outputlen = n - m + 1
    y = [0.0] * outputlen
    for i in range(outputlen):
        for j in range(m):
            y[i] += x[i + j] * k[j]
    return y


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


def check_gradients(x, k, seed=0, eps=1e-5):
    rng = np.random.default_rng(seed)
    n, m = len(x), len(k)
    out_len = n - m + 1

    # Random weights w define a scalar loss L = sum_i w[i] * y[i].
    w = rng.standard_normal(out_len)

    def loss(x, k):
        y = conv1d_forward(x, k)
        return sum(w[i] * y[i] for i in range(out_len))

    # dL/dy[i] = w[i] for L = sum_i w[i] * y[i].
    g = list(w)
    dx, dk = conv1d_backward(x, k, g)

    dx_num = [0.0] * n
    for t in range(n):
        xp, xm = list(x), list(x)
        xp[t] += eps
        xm[t] -= eps
        dx_num[t] = (loss(xp, k) - loss(xm, k)) / (2 * eps)

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
    # Forward checks
    x = [1.0, 1.0, 1.0, 1.0, 1.0]
    k = [1 / 3, 1 / 3, 1 / 3]
    y = conv1d_forward(x, k)
    print(y)
    assert np.allclose(y, [1.0, 1.0, 1.0])

    x = [1.0] * 6
    y = conv1d_forward(x, k)
    print(y)
    assert np.allclose(y, [1.0, 1.0, 1.0, 1.0])  # boundary exercised: 4 outputs

    # Gradient checks
    rng = np.random.default_rng(1)
    x = list(rng.standard_normal(7))
    k = list(rng.standard_normal(3))
    check_gradients(x, k)


if __name__ == "__main__":
    main()
