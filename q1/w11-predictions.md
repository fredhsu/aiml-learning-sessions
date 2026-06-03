---
created: 2026-06-02T22:41:37Z
id: 019e8a7f-e275-7d83-af43-9f7a9aba793f
---
## Predictions

    1. GD from zero. Run GD from $w_0 = 0$ until $\|Xw - y\| < 10^{-8}$ on a random n=10, p=100 problem. What do you expect for:
    - ∥wGD∥/∥w∗∥ (ratio of GD's solution norm to the closed-form min-norm)? 1
    - ∥wGD−w∗∥/∥w∗∥ (relative error in the solution itself)? < $10^{-8}$

    2. GD from random init.Now initialize $w_0 \sim \mathcal{N}(0, \sigma^2 I)$ with, say,$\sigma = 0.1$. Your S1 says $\infty = w^* + w_{0,\perp}$. What do you expect for:

    - $\|w_\infty - w^*\|$ - should this match $\|w_{0,\perp}\|$ to high precision? yes
    - $\|w_\infty\|$ vs $\|w^*\|$ — which is bigger and by how much (in terms of $\|w_{0,\perp}\|$)? $w_{\infty}$ is bigger by $w_{0, \perp}$
    - Does the training loss at convergence differ between zero-init and random-init runs? no

    3. Adam from zero.
    Your S1 forward note says Adam's per-coordinate rescaling acts as an evolving diagonal metric, so Adam-from-zero should not recover the Euclidean min-norm. But you need to be specific about what you expect:

    Will Adam-from-zero converge to a zero-loss solution? (Yes/no — and why?) Yes, it still uses GD to drive to a zero loss solution.
    If yes, will $\|w_{\text{Adam}}\|$ be larger or smaller than $\|w^*\|$? Or could it go either way? Could go either way as it will adjust on the fly.
    Does Adam's solution stay in rowspace(X)? Sketch the argument in one sentence. No, since it provides bias correction it may deviate from rowspace(X)

    Bonus prediction (Momentum): Does MomentumSGD from zero recover min-norm, or not? Reason from the update rule before running.
    Yes, the update rule still stays within rowspace(X)

    ## Corrections to predictions:
    3.
    - Yes, but because least squares is a convex problem, and any reasonable descent method will get to the zero loss solution, not because it is GD.
    - Adam's updates move closer to sign(X^T y), and the sign direction is not always in rowspace(X). It will still converge to *some* zero-loss solution, but not the min-norm one. The norm size will be larger.
    - Yes, but because it acts as a diagonal preconditioner, multiplying rowspace by a non-identity diagonal matrix that usually produces a vector outside of rowspace.
    3b. I expect $\|w_{Adam}\$ to be larger, and should be more concentrated.
