---
created: 2026-05-27
id: 019e6aa1-8d21-7ba1-aee5-b47ddd2c265f
status: draft
tags:
  - explainer
  - optimization
  - implicit-bias
  - generalization
  - week11
title: Implicit Bias of Gradient Descent — A Working Explainer
---

# Implicit Bias of Gradient Descent

## The puzzle

Modern neural networks have more parameters than training examples. By conventional wisdom, they should be overfitting, but instead they generalize surprisingly well.

When given random training data with random labels, these networks are able to memorize random labels to zero training loss. This shows the constraint is not the model class. The same network given real data not only finds a fit to the data, but it fits in a useful way allowing for generalization behavior. Something other than the model class is providing this useful solution.

The something is the optimization algorithm. Although there are many possible solutions that minimize training loss, gradient descent systematically chooses a path to a specific one that happens to be a good one. GD does this without any specific external regularization. There is an implicit bias that is a part of the algorithm leading the way to this good solution.

This explainer uses linear regression as a simplified example of implicit bias and examines what does and does not generalize to a neural network.

## Why the question is well-posed

Starting with a least-squares problem: $\min_w \|Xw - y\|^2$ with $X \in \mathbb{R}^{n \times p}$ and $p > n$, we have a solution set of dimension $p - n$. Every point in this subspace achieves zero training error. The training loss alone does not distinguish among them.

Three things could be doing the selection:

1. **The model class.** Restrict $w$ to some smaller set (regularization, parameter sharing, sparsity constraints). The loss function would then have a unique minimizer.
2. **The data.** Some property of $X$ and $y$ pins down a preferred solution.
3. **The algorithm.** The optimizer's trajectory, starting from some initialization, terminates at one specific point in the solution set.

In the absence of explicit regularization, options 1 and 2 do nothing — the solution set is what it is. The algorithm is the only remaining selector. What we want to know is: _which_ point does gradient descent pick, and _why_?

## The clean linear case

### Theorem

For $\min_w \|Xw - y\|^2, X \in \mathbb{R}^{n \times p}, p > n$, and full row rank, gradient descent from $w_0=0$ with step size $\alpha < 2/\lambda_{\max}(X^TX)$
converges to the minimum $\ell_2$ norm solution $w^{*}= X^T(XX^T)^{-1}y$. Starting from arbitrary $w_0$, gradient descent converges to $w^{*} + w_{0,\perp}$ , where $w_{0,\perp}$ is the projection of $w_0$ onto $\text{null}(X)$.

### Proof

_Rowspace Lemma:_

1. $\nabla L = X^T (Xw-y)$ - Gradient of the loss
2. Update vector: $\alpha X^T r_k$ where $r_k = Xw_k -y$ since the columns of $X^T$ are the rows of $X$, $X^T r_k$ is a linear combination of the rows of X, so it is within the rowspace of $X$.
3. $w_0 = 0 \in \text{rowspace}(X)$
4. $w_k \in \text{rowspace}(X)$ by induction

By staying within rowspace, we imply that we will reach a min-norm. This is because the rowspace is orthogonal to the nullspace, so if $w_k \in \text{rowspace}(X)$ and $Xw_k = y$ then $w_k$ is the unique point in rowspace that is on the zero-loss subspace, and that is exactly $w^*$.

This was shown empirically by comparing the closed form solution to the results from running gradient descent on a linear regression. From my experiments:

- GD from zero: $||w_\infty - w^*|| / ||w^*|| = 7.6 \times 10^{-9}$ within tolerance.

- GD from random init: $||w_\infty - w^* - w_{0,\perp}|| = 1.2 \times 10^{-8}$ at the _vector_ level

- The Pythagorean identity holds to the same precision

### The geometric picture

- $\mathbb{R}^p$ decomposes as $\text{rowspace}(X) \oplus \text{null}(X)$ (orthogonal split under $\ell_2$)

- The zero-loss set $\{w : Xw = y\}$ is an affine subspace parallel to $\text{null}(X)$

- It crosses $\text{rowspace}(X)$ at exactly one point: $w^*$

- GD lives in rowspace, so it lands at that crossing point

- Random init shifts the starting point off the rowspace by some nullspace amount, and GD can only move within the rowspace direction, so the nullspace offset is preserved verbatim

## The metric thread

In week five, we talked about GD using the steepest descent under the Euclidean inner product as the metric. This metric is what determines the 'downhill' direction. With Adam as the optimizer, we have a different metric, and therefore a different downhill direction. Adam's update uses per-coordinate division by $1/\sqrt{v}$, which is the same as using a diagonal preconditioner $D^{-1}$ where $D=\text{diag}(\sqrt{v_i})$. By multiplying a rowspace vector by a diagonal matrix that isn't $I$, we travel outside of rowspace, so the Adam updates are not in rowspace and we can no longer use the induction from the rowspace lemma. As a result, Adam can land anywhere on the zero-loss subspace, including somewhere in the nullspace direction.
From the experiments in S2, we found different norms between Adam and GD, and a rowspace leak:

- $||w_{Adam}||_2 = 3.24$ vs $||w^*||_2 = 2.65$ - 22% larger

- $||w_{Adam}||_\infty = 0.50$ vs $||w^*||_\infty = 0.82$ - 40% smaller

- rowspace leak: 1.87 (GD had $\sim 10^{-15}$)

The result is that Adam not only lands at a different point, but has a different shaped solution.

## The leap to deep learning

The Zhang, et al. paper showed that deep networks can fit random labels to zero training error, which means their function class is rich enough to fit anything. So the function class cannot be what's stopping them from overfitting on real data. This means that classical bounds on complexity (VC dimension, Rademacher complexity) that only constrain what a function class can do are not useful.
The Neyshabur paper argues that the algorithm itself is the source of inductive bias by using empirical evidence of larger networks not hurting generalization. This is consistent with norm based, rather than dimension based constraints. Using linear as an example, trace norm, not rank, provides the bounding factor for test loss behavior. By the analogy, some undefined norm, not the network width, is likely the bounding factor for deep networks. However, for deep neural networks it is still an open question as to what specific norm is providing the constraint.
Week 10's discussion showed that the geometry of the loss surface at the minimum does not determine generalization. Implicit bias is a dynamic explanation, not the geometry around the point, but the path taken.

## What this changes about optimization

The optimizer is not just a tool for finding "the" minimum. When using an overparameterized problem, there is an infinite number of minima and the optimizer is choosing one. This makes the choice of optimizer a modeling decision instead of an engineering choice. Choosing between SGD, Adam, or AdamW decides the kind of solution you want, not just how to find it.

Looking at AdamW from this new perspective, we see how the decoupled weight decay is an explicit correction to a small norm from Adam's implicit bias when the bias points the wrong way for generalization. This is shown with AdamW often beating Adam when generalization matters.

## Honest limits

The arguments are proven only for linear systems. They do not directly apply to networks with non-linearities. Even when looking at the linear case, the min-norm result is only valid assuming:

- zero init

- Step size below $2/\lambda_{\max} (X^T X)$

- iterating to convergence

- vanilla SGD

The analogy is proposed and empirically shown not proven. Adam's deviation from min-norm is shown empirically, but no clean theorem characterizes which solution Adam picks. The metric framing above (Adam = diagonal preconditioner = different geometry) explains _why_ it deviates but not _which way_. Although the conclusion is that the algorithm's bias aims to minimize a norm, it is an open problem to define what exact norm it is.

## Forward connections

These connections will appear in later sessions:

- RL policy gradients and natural gradients (Q3)

- Architecture-as-bias — convolution's translation equivariance

- attention's learned bias (Q2)

---

## References

- [[Week11_S1_ImplicitBias_Math]] — the row-space theorem

- [[Week11_S2_ImplicitBias_Code]] — numerical verification, including Adam's deviation

- [[Week5_S1_GD_Euclidean]] — GD as steepest descent under Euclidean metric

- [[Week9_S4_OptimizerCheatSheet]] — Adam's per-coordinate rescaling

- Zhang et al., 2017 — _Understanding Deep Learning Requires Rethinking Generalization_, arXiv:1611.03530

- Neyshabur, Tomioka, Srebro, 2014 — _In Search of the Real Inductive Bias_, arXiv:1412.6614

- Soudry et al., 2018 — _The Implicit Bias of Gradient Descent on Separable Data_, arXiv:1710.10345 (deeper alternative; not read this week)
