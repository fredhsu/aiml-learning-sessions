---
created: 2026-05-21T20:46:58Z
id: 019e4c4a-9a62-7bf1-8cf4-c754c318bb5d
---
# Week 11 Session 2 - Implicit Bias Code

## Setup

Underdetermined linear regression: $X \in \mathbb{R}^{10 \times 100}$ with iid Gaussian entries scaled by $1/\sqrt{p}$, $y = X w_{\text{true}}$ for a random $w_{\text{true}}$. Solution set is a 90-dimensional affine subspace (100-10=90). Originally I did not have the scaling, but added it because the norm was too large and I could not see the effects of the Pythagorean identity.

Closed-form min-norm solution: $w^* = X^T(XX^T)^{-1} y$, computed via `np.linalg.solve` rather than explicit inverse for efficiency. $\|w^*\|_2 = 2.65$, $\|w^*\|_\infty = 0.82$.

Step size for SGD/MomentumSGD: $\alpha = 0.9$, safely below $2/\lambda_{\max}(XX^T)$. Adam needs a smaller step ($\alpha = 0.01$) because its per-coordinate normalization changes the effective step scale, when I used the same $\alpha$ as GD it diverges. **Adam and GD are not directly comparable at the same learning rate**.

## Results table

| Run         | Init                  | Steps to tol=1e-8 | $\|w_\infty\|_2$ | $\|w_\infty\|_\infty$ | $\|w_\infty - w^*\|$ | Rowspace leak   |
| ----------- | --------------------- | ----------------- | ---------------- | --------------------- | -------------------- | --------------- |
| SGD         | zero                  | 32                | 2.65             | 0.82                  | $7.6 \times 10^{-9}$ | $\sim 10^{-15}$ |
| SGD         | random ($\sigma=0.1$) | 345               | 2.84             | 0.67                  | 1.02                 | 1.02            |
| MomentumSGD | zero                  | 354               | 2.65             | 0.82                  | $6.5 \times 10^{-9}$ | $\sim 10^{-15}$ |
| Adam        | zero                  | 355               | 3.24             | 0.50                  | 1.87                 | 1.87            |

## Theorems confirmed numerically

**GD from zero → min-norm.** The S1 row-space argument predicts $w_\infty = w^*$ exactly. Measured: $\|w_\infty - w^*\| / \|w^*\| = 7.6 \times 10^{-9}$ at the convergence tolerance, basically zero as expected. Ratio $\|w_\infty\| / \|w^*\| = 1.000000$. These two tests confirm the prediction.

**GD from random init →** $w^* + w_{0,\perp}$. The S1 corollary predicts that the nullspace component of $w_0$ is preserved and the rowspace component evolves to $w^*$. The full check is the _vector_ equality, not just norm equality:

$\|w_\infty - w^* - w_{0,\perp}\| = 1.19 \times 10^{-8}$

Pythagorean identity holds at the same precision:

$\|w_\infty\|^2 - \|w^*\|^2 - \|w_{0,\perp}\|^2 = -1.15 \times 10^{-8}$

Both are at the residual tolerance, the theorems hold to whatever precision GD converges to.

**MomentumSGD from zero → min-norm.** The momentum buffer is a running sum of past gradients, each in $\text{rowspace}(X)$. Sum of rowspace vectors is in rowspace. By induction $w_k \in \text{rowspace}(X)$ for all $k$, so the row-space lemma applies unchanged and MSGD converges to the same min-norm solution as GD. Confirmed numerically, I got the same $\|w_\infty\|$ and relative error as SGD-from-zero.

## The Adam finding: equalization, not concentration

Adam-from-zero reaches a zero-loss solution but lands on a _different point_ of the zero-loss subspace than GD does:

* $\|w_{\text{Adam}}\|_2 = 3.24$ — 22% larger than $\|w^*\|_2$.

* $\|w_{\text{Adam}}\|_\infty = 0.50$ — 40% **smaller** than $\|w^*\|_\infty = 0.82$.

* Rowspace leak: 1.87 (vs $\sim 10^{-15}$ for GD), Adam's solution sits substantially in $\text{null}(X)$.

For Adam to have larger $\ell_2$ but smaller $\ell_\infty$, its mass must be spread across more coordinates at moderate magnitudes. This is the opposite of sparsity. The companion plot (`Coordinate magnitudes, sorted descending`) makes this visible: $w^*$ concentrates mass in the top 10–15 coordinates and decays; $w_{\text{Adam}}$ is nearly flat across all 100 coordinates.

Top 5 coordinate magnitudes:

* $w^*$: $[0.52, 0.55, 0.70, 0.74, 0.82]$ - graded.

* $w_{\text{Adam}}$: $[0.41, 0.43, 0.44, 0.47, 0.50]$ - nearly uniform.

### Mechanism

Adam's update is $\alpha \cdot \hat{m}_i / (\sqrt{\hat{v}_i} + \epsilon)$. The denominator is roughly the running RMS of the gradient at coordinate $i$, so coordinates with large past gradients get scaled down and coordinates with small past gradients get scaled up. This _equalizes_ effective step sizes across coordinates. Over many steps the cumulative effect is a solution where every coordinate carries roughly equal magnitude, instead of GD's pattern where a few coordinates are siginficant and the rest stay small.

### Connection to the metric thread

In Week 5 the optimizer's metric determined the _step direction_. Here the metric determines the _solution shape_. The S1 row-space lemma fails for Adam because the per-coordinate diagonal preconditioner is not the identity. Multiplying a rowspace vector by a non-trivial diagonal matrix produces a vector outside rowspace.

GD's implicit bias is "minimize $\ell_2$ norm", by the row-space lemma. Adam's implicit bias has no clean theorem characterizing the norm Adam minimizes. Empirically on this problem, the bias is toward _coordinate equalization_, which is what the diagonal rescaling does to gradients. The takeaway for the metric thread: different optimizers don't just converge to different points, they converge to _qualitatively differently-shaped_ solutions.

## Side observation: momentum overshoots, still works

The plot for MomentumSGD oscillates drastically with $\|w_k\|$ peaking at $\sim 5$ (almost twice $w^*$) before settling down. It takes 354 steps to converge vs 32 for plain SGD. On a well-conditioned problem there is no curvature mismatch for momentum to compensate, so it just overshoots without acceleration benefit. The opposite of the Week 7 ravine experiments where momentum was the right tool. *The value of using momentum depends on the problem geometry, it is not unconditional.*

## Open question for S3

The row-space lemma is fundamentally linear: gradients of a linear loss are linear combinations of rows of $X$. For deep networks the gradient is not in any clean "row space," so the proof does not apply. However, deep networks do generalize from GD, which suggests something similiar to implicit bias is at work, just not one we can characterize cleanly. S3 (Neyshabur et al.) is about how this argument extends or fails to extend beyond the linear case.

## Cross-links

* \[\[Week11\_S1\_ImplicitBias\_Math]] — the row-space theorem these experiments verify

* \[\[Week5\_S1\_GD\_Euclidean]] — the metric framing that connects GD's bias to the Euclidean inner product

* \[\[Week9\_S4\_OptimizerCheatSheet]] — Adam's per-coordinate rescaling, now visible as a deviation from min-norm

* \[\[Week7\_S2\_Momentum\_Code]] — momentum on ravines, where the overshoot was useful (contrast)

