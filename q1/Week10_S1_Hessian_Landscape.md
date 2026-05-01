# Week 10 Session 1 - Math - Hessian Landscape

## Multivariate second-order Taylor expansion

Starting with the univariate second-order expansion:

$$
f(w+\Delta w) = f(w) + f'(w) \Delta w + \frac{1}{2} f''(w) (\Delta w)^2
$$

Then given the following shapes for a multivariate function:

$$
w: n x 1
\nabla f(w) : n x 1
H : n x n
\Delta w : n x 1
$$

The multivariate second-order Taylor expansion is:

$$
f(w+\Delta w) = f(w) + \nabla f(w)^T \Delta w + \frac{1}{2} \Delta w^T H \Delta w
$$

At a minimum $ \nabla f(w^\*) = 0$ so the first-order term vanishes and the second-order term will dominate

Since the Hessian matrix is symmetric, by the [[spectral theorem]] the eigenvalues are real and the eigenvectors are orthogonal.

$$
H = Q \Lambda Q^T \\
u = Q^T \Delta w \\
\\
\frac{1}{2} \Delta w^T H \Delta w = \frac{1}{2} \Delta w^T Q \Lambda Q^T \Delta w = \frac{1}{2} (Q^T \Delta w)^T \Lambda (Q^T \Delta w) = \frac{1}{2}u^T \Lambda u = \frac{1}{2} \sum \lambda_i u_i^2
$$

Here Q is one rotation, and the result of that rotation is the quadratic decouples into a sum of per-direction parabolas.

## Connection to Week 7

The above decomposition is the same as when we used [[eigendecomposition]] for analyzing the different contributions to momentum in week 7.

## Sharp vs flat

### Per-direction

Use the eigenvalues

### Whole-minimum

Use scalar summaries

- $\lamda_{\max}$ : sharpness of the single steepest direction
- [[Trace]] : average curvature (sum of all eignevalues)
- [[Determinant]] : product of eigenvalues
- [[Condition Number]] : [[anisotropy]] of the basin

Different scalar summaries can rank minima differently, and they differ in how easily they can be manipulated by reparameterization. Condition number is scale-invariant in the simplest sense (ratio of eigenvalues), while $\lambda\_{\max}, trace, and determinant are not.

## Pre-reading conjecture

### Will flat minima generalize better?

My guess is that they will, since you can take larger steps away from the minima without drastically changing the behavior.

### Reparameterization

- Is sharpness a property of the function the network computes, or a property of the parameterization chosen?
- Hessian changes when rescaling the function weights, even though we have the same absolute perturbation, it would have a different effect on the loss and sharpness of the landscape.
- under $W_1 \to cW_1, W_2 \to W_2 / c$ in a ReLU network, function is preserved but Hessian entries rescale; $\lambda_{\max}$ can be made arbitrarily large.
- The implication is that Hessian-based sharpness is not reparameterization-invariatn, so it can't be the direct causal mechanism for generalization.
- My initial intuition was that the Hessian would be unchanged. It isn't. Open question for S3: does $\kappa$ (condition number) survive this attack?
