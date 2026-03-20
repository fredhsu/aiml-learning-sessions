# Week 8 S1 Regularization_Math

L2 regularization (ridge regression) adds a quadratic penalty to the loss, controlled by the hyperparameter $\lambda$, to discourage large weights and help reduce overfitting.

## Algebraic derivation

Regularized Loss: $$\tilde{L}(w) = L(w) + \frac{\lambda}{2} ||w||_2^2$$
Gradient: $$\nabla \tilde{L}(w) = \nabla L(w) + \lambda w$$
Weight decay update:

$$
\begin{aligned}
w^{t+1} = w^t - \alpha(\nabla L(w^t) + \lambda w^t) \\
= (1-\alpha \lambda)w^t - \alpha \nabla L(w^t)
\end{aligned}
$$

This shows the effect of weight decay: at each step the weights are shrunk toward zero by the factor $(1-\alpha \lambda)$, then updated using the ordinary loss gradient. This factor also creates a constraint $(1-\alpha \lambda) > 0$ to avoid divergence.

## Geometric picture

The additional weight decay adds a bowl centered at the origin, increasing the curvature in every direction. This has a stronger effect in the small eigenvalue directions because the curvature of $L$ is weak, so $\lambda$ dominates the denominator of the regularized minimum $(\lambda_i + \lambda)$ pulling it more towards zero.
![Loss plot with bowl](w7s2-loss-plot.png)

## Bayesian framing

From a Bayesian perspective we use the MAP (Maximum a posteriori) estimation and the Gaussian prior for the distribution of the weights. Maximizing the posterior is equivalent to minimizing the loss plus a penalty. $\lambda$ has a probabilistic interpretation that is inverse to the variance. The bigger $\lambda$ is, the stronger the belief that the weights should be near zero. This can be derived starting with $P(w|D)$ and using $P(w)=\mathcal{N}(0,\frac{1}{\lambda}I)$.
