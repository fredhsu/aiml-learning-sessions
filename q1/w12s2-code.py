"""
<<<<<<< HEAD
# Week 12 Session 2 Code - 1D convolution forward/backward pass

Standalone mini implementation using cross correlation convention, stride 1.
Not wired into the Value object for now, will address this later.

## Predictions
=======
Week12_S2_Conv_Code — 1D convolution forward/backward, pure Python

Q1 Week 12, Session 2. Standalone mini-implementation: cross-correlation
convention (no filter flip), valid conv, stride 1. Intentionally NOT wired
into the Value engine — that integration is a Week 14 cleanup task.

================================================================================
PREDICTIONS (logged before writing the code)
================================================================================
>>>>>>> dfaf04c (working on w12s4)
1. Forward sanity (x = ones(5), k = [1/3,1/3,1/3], valid, stride 1):
       First guess: [1, 1]  length 2.
       Corrected against the S1 output-length formula (n - m + 1 = 5-3+1 = 3):
       [1, 1, 1]  length 3.
2. Grad-check passing threshold (centered finite differences): rel err ~1e-8.
3. Most likely bug: off-by-one in the valid range.

<<<<<<< HEAD
## Results
Forward: [1,1,1] len 3 confirmed. Input of length 6 gave 4 outputs, boundary holds.
Backward: Derived both gradients, grad-check passes, both dx and dk are under 1e-8 threshold error
=======
================================================================================
RESULTS (reviewed against predictions)
================================================================================
Forward:  [1,1,1] len 3 confirmed; length-6 input -> 4 outputs, boundary holds.
Backward: derived both gradients from the Toeplitz view, grad-check passes.
       dx rel err: 2.6e-10
       dk rel err: 2.3e-12   (both well under the 1e-8 threshold)

CALIBRATION DEBRIEF
-------------------
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
>>>>>>> dfaf04c (working on w12s4)

Derived gradients (for reference):
    dL/dk[j] = sum_{i=0}^{n-m}              g[i] * x[i+j]
    dL/dx[t] = sum_{i=max(0,t-m+1)}^{min(n-m,t)} g[i] * k[t-i]
<<<<<<< HEAD

## Debrief
- Predicted off-by-one bug was correct, and hit it twice at range boundaries:
  a. Allocating the list for forward: `[0.0]*m` initially, but it should be `[0.0]*(n-m+1)`
  b. Backward dx loop I initially did not have the right upper bound on range, it excluded the top term. It needed to be `min(n-m,t) + 1`.
- So I ended up commiting the error that I predicted would be a challenge. I need to think about boundaries more.
- Did not hit a conv/cross-correlation flip bug, stayed consistent with cross-correlation.
- Grad check was better than the predicted error.

=======
>>>>>>> dfaf04c (working on w12s4)
"""

import numpy as np


<<<<<<< HEAD
=======
def conv1d_forward(x, k):
    n = len(x)
    m = len(k)
    outputlen = n - m + 1
    y = [0.0] * outputlen
    for i in range(outputlen):
        for j in range(m):
            y[i] += x[i + j] * k[j]
    return y


>>>>>>> dfaf04c (working on w12s4)
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


<<<<<<< HEAD
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


=======
>>>>>>> dfaf04c (working on w12s4)
def check_gradients(x, k, seed=0, eps=1e-5):
    rng = np.random.default_rng(seed)
    n, m = len(x), len(k)
    out_len = n - m + 1

    # Random weights w define a scalar loss L = sum_i w[i] * y[i].
    w = rng.standard_normal(out_len)

    def loss(x, k):
        y = conv1d_forward(x, k)
        return sum(w[i] * y[i] for i in range(out_len))

<<<<<<< HEAD
    # Analytic gradients. You need to supply g = dL/dy here.
    g = w  # what is dL/dy for L = sum_i w[i]*y[i]?
    dx, dk = conv1d_backward(x, k, g)

    # Numerical gradient for x via centered differences.
=======
    # dL/dy[i] = w[i] for L = sum_i w[i] * y[i].
    g = list(w)
    dx, dk = conv1d_backward(x, k, g)

>>>>>>> dfaf04c (working on w12s4)
    dx_num = [0.0] * n
    for t in range(n):
        xp, xm = list(x), list(x)
        xp[t] += eps
        xm[t] -= eps
        dx_num[t] = (loss(xp, k) - loss(xm, k)) / (2 * eps)

<<<<<<< HEAD
    # Numerical gradient for k.
=======
>>>>>>> dfaf04c (working on w12s4)
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
<<<<<<< HEAD
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
=======
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
>>>>>>> dfaf04c (working on w12s4)


if __name__ == "__main__":
    main()
