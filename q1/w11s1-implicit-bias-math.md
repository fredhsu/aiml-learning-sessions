---
created: 2026-05-19T00:32:13Z
id: 019e3da5-c23e-7873-ae95-2d7f2aa1a70e
---

# Week 11 Session 1 - Implicit Bias Math

## Theorem statement (Result)

For an underdetermined linear regression with $X \in \mathbf{R}^{n \times p}, p > n$, full row rank, gradient descent initialized at $w_0=0$ with an appropriately small step size will converge to the minimum-norm solution $w^*=X^T(XX^T)^{-1}y$.

## Setup

The goal of this session is to build an understanding of which solution will [[Gradient Descent]] pick given an overparameterized model. We will use an example where we are finding solutions in $w \in \mathbf{R}^{100}$, where $n=10, p=100$ so $X=10 \times 100$. So the solution set here is a 90-dimensional affine subspace of $\mathbf{R}^{100}$, since each of the 10 equations will remove one dimension, leaving 100-10=90 degrees of freedom. There would be infinitely many possible solutions of $L(w)=0$. The set of all possible linear combinations of the rows of X is the rowspace(X) i.e. the span, which in this case is a 10-dimensional subspace.

## The row-space lemma (4 numbered steps)

1. $\nabla L = X^T(Xw - y)$
2. The update vector is $-\alpha X^T r_k$ where $r_k = Xw_k - y$. Since the columns of $X^T$ are the rows of X, $X^Tr_k$ is a linear combination of those rows, and therefore a vector in rowspace(X)
3. $w_0 = 0 \in \text{rowspace}(X)$
4. By induction, $w_k \in \text{rowspace}(X)$

## The min-norm characterization

### Orthogonal decomposition

We can use orthogonal decomposition for any vector $w \in \mathbf{R}^{100}$ to split into $w=w_{\parallel} + w_{\perp}$ where $w_{\parallel} \in \text{rowspace(X)}$ and $w_{\perp} \in \text{null(X)}$ and these two subspaces are orthogonal.

### Pythagorean Theorem/Euclidean inner product

Using Pythagoras with the orthogonal vectors we can see that $||w||^2 = ||w_{\parallel}||^2 + ||w_{\perp}||^2$ since we know they are perpendicular under the Euclidean inner product.

This gives us a unique solution in row space, which is
the minimum-norm solution $w^*$.

Since $w^*$ is rowspace(X) we can rewrite it as $w^*=X^T \beta$ for $\beta \in \mathbf{R}^n$. Substituting that into the function $Xw^*=y$, gives $XX^T \beta = y$, making $\beta=(XX^T)^{-1} y$, and finally we have a closed form: $w^* = X^T(XX^T)^{-1}y$

## Gradient Descent from zero to min-norm

Now we can put it all together, beginning with initializing GD with zeros:

1. Starting with $w_0 = 0$ and a gradient of $X^T(Xw-y)$, we will stay within rowspace(X).
2. GD will converge to a zero-loss solution
3. Using orthogonal decomposition, Euclidean inner product, and Pythagoras we see that the only solution in the rowspace is the min-norm.
4. Therefore GD initialized at zero will converge to the min-norm, $w^*$

## Initialization caveat

If $w_0 \neq 0$, we decompose $w_0=w_{0,\parallel} + w_{0,\perp}$. The row-space component continues with GD as before, which will end up on a row-space vector that will satisfy $Xw=y$, namely $w^*$. However the null-space component doesn't change since all the GD updates happen in rowspace(X). So $w_{\perp}$ never changes from the initial value. This means:
$$w_{\infty} = w^* + w_{0,\perp}$$
A random initialization will very likely have a non-zero null-space initialization, so it will end up with a larger-norm solution than the zero initialization. This shows that the implicit bias of GD is dependent on the initialization.

## Metric remark

The Euclidean inner product enters in two places:

1. The orthogonal decomposition $\mathbb{R}^p = \text{rowspace}(X) \oplus \text{null}(X)$ is orthogonal under the Euclidean inner product. Under a different inner product $\langle u, v \rangle_M = u^T M v$, these subspaces are no longer orthogonal, and the Pythagorean split doesn't apply in the standard form.
2. The form of the GD update. The "gradient" $\nabla L = X^T(Xw - y)$ is the gradient under the Euclidean metric. Under metric $M$, the steepest-descent direction is $M^{-1} X^T(Xw - y)$, which is no longer in rowspace(X), it's in $M^{-1} \cdot \text{rowspace}(X)$

The row-space lemma no longer holds under a non-Euclidean metric, and GD lands on the M-min-norm solution instead. There's nothing special about "min-norm" it depends on an implicit metric choice.

**Forward to S2**: Adam's per-coordinate rescaling acts like an evolving diagonal metric, so Adam from zero should not recover the Euclidean min-norm solution. Concrete prediction to verify experimentally.
Cross-link: [[w5s1-gd-euclidean]]

## My weak spot

Reflex for the four fundamental subspaces (rowspace, column space, null space, left null space) and their orthogonal-complement relationships. Worth a brief refresher before Week 12 (Toeplitz / structured linear operators) where this picture returns. Reference: Strang, Introduction to Linear Algebra, Ch. 3

## Implicit Bias

Implicit bias is something that the optimization algorthim (GD), not the loss function, picks from a continuum of equally good solutions. The loss function doesn't discriminate among the 90-dimensional flat of zero-loss solutions, it is a combination of the algorithm, initialization, and metric that jointly pick a specific point. The optimizer itself is a regularizer, even without an explicit regularization term in the objective.
