# Gradient Descent Is Steepest Descent Under the Euclidean Metric

## Outline

1. Taylor expansion -> Linear map $df_x$ approximation
2. Optimization - want to take a small step that will make $df_x$ as negative as possible
3. Constrained minimization of that linear map - need a size constraint to find a useful solution
4. Choice of norm/inner product - choosing a metric to define 'size' of step
5. Representation of the linear map as a gradient vector - when we choose inner product, we represent $df_x$ as a vector: $\nabla f$
6. Orthogonality of level sets as a consequence - gives us 'steepest' descent.

## Writeup

For gradient descent, we want to find which direction will reduce a function the most for a small step $\epsilon$. This can be defined as the following constrained optimization problem:

$$
\min_{|\Delta x| \le \varepsilon} df_x(\Delta x)
$$

Here $df_x$ is the derivative of $f$ at $x$, viewed as a linear map (covector) acting on a direction $\Delta x$.

The derivative $df_x$ is a covector: a linear map that consumes a vector and produces a scalar rate of change. The "steepest" direction is dependent on how we measure the size of a step, i.e. our metric. Once we choose an inner product (and therefore our metric), such as the Euclidean norm with the dot product, we lock in how the rest of the calculations will behave and the covector is represented by the gradient vector.

We can use the Taylor expansion to give a linear approximation and to identify what we can change and what we can't. In this case we drop the higher order term which is negligible when $\Delta x$ is small.

$$
f(x + \Delta x) \approx f(x) + df_x(\Delta x)
$$

Under the Euclidean inner product, $df_x$ becomes:

$$
df_x(\Delta x) = \nabla f(x)^T \Delta x
$$

Since $f(x)$ is fixed at the current point, we can only adjust $\nabla f(x)^T \Delta x$ (the change/step we can take).

If we were to minimize this function without a constraint, then there would be no limit, we would take an arbitrarily large step and it would go to $-\infty$. So we must impose a constraint, in this case $||\Delta x|| <= \epsilon $ Solving the constrained minimization problem under the Euclidean norm yields a step in the direction of the negative gradient.
Geometrically, this direction is normal to the level set of $f$ at $x$, explaining why gradient descent moves orthogonally to level sets.
By choosing to use the inner product, and therefore Euclidean metric, we have also identifed the norm: $||v|| = \sqrt{\langle v , v \rangle}$. Geometrically this will be a sphere. Without this we cannot properly define the gradient, it would be an abstract covector. Other metrics and norms can be chosen to achieve different results.

The final result for GD with Euclidean norm optimal step is in the direction defined by the negative of the gradient, multiplied by $\epsilon$.

## The Question

What direction $\Delta x$ (with $\|\Delta x\| \leq \epsilon$) minimizes
f(x + Δx) the most?

## The Math

After choosing the Euclidean inner product, the derivative covector $df_x$ is represented by the gradient vector $\nabla f(x)$, allowing the constrained optimization problem to be written in terms of inner products and norms.

Taylor expansion: f(x + Δx) ≈ f(x) + ∇f(x)ᵀΔx + O(||Δx||²)

For small ε, minimize the linear term:
min\_{||Δx||₂ ≤ ε} ∇f(x)ᵀΔx

By Cauchy-Schwarz: ∇f(x)ᵀΔx ≥ -||∇f(x)||₂ · ||Δx||₂

Equality when: Δx = -ε · ∇f(x)/||∇f(x)||₂

## The Geometric Picture

- Constraint ||Δx||₂ ≤ ε defines a sphere around x
- ∇f(x) is normal to level set
- "Steepest descent" = most negative inner product with ∇f(x)
- Under Euclidean metric: this is the gradient direction itself

## The Choice That Matters

Different norm → different "steepest" direction:

- L₁ norm (||Δx||₁ ≤ ε): coordinate descent behavior
- L∞ norm: equal step in all coordinates
