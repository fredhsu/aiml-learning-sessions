# Momentum Math

## Heavy ball update

$$
\begin{align}
z^{k+1}&=\beta z^k + \nabla f(w^k) \\
w^{k+1}&=w^k - \alpha z^{k+1}
\end{align}
$$

where $z^{k+1}$ is the accumulation/velocity (exponentially weighted sum) and $\beta$ is how much [[momentum]]/history is kept. Note we are using this accumulation form where we absorb the $(1-\beta)$ into $\alpha$ vs the 'normalized form' from [[2026-02-28 Session W6S3 Reading about Momentum]].

## [[Nesterov]] update

$$
\begin{align}
z^{k+1}&=\beta z^k + \nabla f(w^k-\alpha \beta z^k) \\
w^{k+1}&=w^k - \alpha z^{k+1}
\end{align}
$$

The weight update is the same, but the $z$ update uses the [[gradient]] at the look-ahead point that reflects where the next step is about to go, so the correction will pull back before the step is committed. This helps when the momentum would cause the update to overshoot.

## Effective memory length of momentum

Solving $\beta^t = \frac{1}{e}$ for $t$ gives $t=\frac{-1}{ln(\beta)}$ as a measure of how far back the momentum calculations will carry. As $\beta \to 0$ the memory becomes only a single step. As $\beta \to 1$ the step will go towards $\infty$ making the updates slow to adjust.

## Why $\sqrt \kappa$ instead of $\kappa$

Momentum accumulates the consistent [[gradient]]s in the minimum [[eigenvalue]] direction, speeding up convergence in this direction by a factor $sqrt(\kappa)$ instead of $\kappa$. This is due to the difference in optimal convergence rate which is constrained by $\lambda_{min}$ . For vanilla GD it is $\frac{\kappa - 1}{\kappa + 1}$ roughly $1-\frac{2}{\kappa}$ for large $\kappa$. With momentum this becomes $\frac{\sqrt \kappa - 1}{\sqrt \kappa + 1}$ and $1-\frac{2}{\sqrt \kappa}$. So momentum effectively reduces the iteration complexity from $\kappa \to \sqrt \kappa$.

Note using the [[conditioning number]] assumes we are working with strongly convex smooth functions such as a quadratic.
