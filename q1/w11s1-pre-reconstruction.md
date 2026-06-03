---
created: 2026-05-19T00:32:05Z
id: 019e3da5-a088-7571-9fe2-8dce6c9c3be1
---
# Week 10 Session 1

## Cold reconstruction

1. 2, the loss is a quadratic function, it will be the points where the weights will intercept the loss axis at zero.
   Incorrect, need to think in linear algebra terms, actually infinitely many, and 90 dimensional continuum.

2. gradient is w(Xw-y), update will subtract learning rate\*gradient from the function
   Wrong, think about shapes, should be X^T(Xw-y)
   update is then w\_{k+1} = w_k - \alpha X^T (Xw_k -y)
   Every GD update vector lives in the span/rowspace(X)

3. Staring from w_0=0, all the GD updates will live in the rowspace(X)

And the full chain now connects:

GD from w0=0w*0 = 0
w0​=0 is trapped in rowspace(X)\text{rowspace}(X)
rowspace(X) forever (we proved this).
GD converges to \_a* zero-loss solution (assume small enough α\alpha
α).
The only zero-loss solution in rowspace(X)\text{rowspace}(X)
rowspace(X) is the min-norm solution w∗w^_
w∗.
Therefore GD from zero converges to w∗w^_
w∗. ∎

1. What changes with w0≠0? Predict: if you start at a generic random w0w_0
   w0​, will GD still converge to the min-norm solution? If not, what does it converge to?

The intersection of points in a R^N subspace of dimension a, and an affine subspace dimension b is a+b-N, for example with a row subspace a=10, solutions set b=90, ambient N=100, the intersection dimension is 0, a single point.

With a nonzero initialization, the solution ends up shifted by the null space component equivalent to the initialization.

Any initialization within the rowspace(X) converges to min-norm solution, zero is the most convenient row-space point.
Outside of those, it will converge to a solution, with zero loss, but will be shifted by the initialization.

1. The Dinh et al. argument said static Hessian-sharpness is parameterization-dependent and can't explain generalization. Is the row-space argument parameterization-dependent? Does it rely on the Euclidean inner product anywhere?

It shows up when we used the inner product with the Pythagorean Theorem and show that we will hit the min norm. The inner product was used to decompose into rowspace(X) and null(X) where they are orthogonal.

What surprised me:
How the choice of initialization related to the minimum we ended up at. That a zero initialization weight would bring us to the min norm, whereas others would get us to a min that may be shifted based on the values of the initialization.

What I got right:
Some of the carryover concepts relating to inner products.

What was wrong:
My initial breakdown of the gradient from a linear algebra based and row-space perspective. I did not have the right conceptual idea of rowspace.

Weak spot identified: Reflex for the four fundamental subspaces (rowspace, column space, null space, left null space) and their orthogonal-complement relationships. Worth a brief refresher before Week 12 (Toeplitz / structured linear operators) where this picture returns. Reference: Strang, Introduction to Linear Algebra, Ch. 3.
