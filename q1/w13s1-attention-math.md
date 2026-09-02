---
created: 2026-07-12T06:17:27
id: 32f5cc63-12bc-4567-9ec3-2ca682ec8469
tags:
- week-13
- attention
- inner-products
- metric-thread
- softmax
title: Week 13 S1 Attention Math
---

# Week 13 S1 Attention Math

Attention at its core is just a weighted average of the **values**. The queries and keys only produce the weights.

## Derivation

Beginning with a single query retrieval:

$$
\text{attn}(q, K, V) = \sum_i \alpha_i v_i
$$

with $\alpha = \text{softmax}(\text{scores} / \sqrt{d_k})$ and $\text{score}_i = q^\top k_i$. The softmax outputs are between 0 and 1 and sum to 1. The values are multiplied by these weights and summed to compute the output.

(Note the scores themselves are unbounded reals — it is the *softmax outputs* that live in $[0,1]$.)

Lifting to matrices, all queries at once:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

The softmax is applied **row-wise**: each query's weights form a distribution over the keys.

## Variance argument for $\sqrt{d_k}$

Starting from $\text{Var}(q^\top k) = \text{Var}\!\left(\sum_{j=1}^{d_k} q_j k_j\right)$, with $q, k$ having i.i.d. entries of mean 0 and variance 1:

1. Independent terms add: the $q_j k_j$ are independent across $j$, so the variance of the sum is the sum of the variances.
2. Each term has variance $\mathbb{E}[q_j^2]\,\mathbb{E}[k_j^2] - (\mathbb{E}[q_j k_j])^2 = 1 \cdot 1 - 0 = 1$.
3. Therefore $\text{Var}(q_j k_j) = 1$ and $\text{Var}(q^\top k) = d_k$.

We want to scale the score to have a standard deviation of 1, so we divide by the square root of the variance — $\sqrt{d_k}$, **not** $d_k$. (Dividing by the variance would over-shrink the scores to variance $1/d_k$.)

The reason behind this scaling: larger scores cause softmax to saturate, making the weights act like a one-hot encoding. This makes the softmax Jacobian collapse and the gradients vanish. The $\sqrt{d_k}$ divisor holds the score spread at 1 regardless of head dimension, keeping softmax in its responsive regime.

## Bilinear form — the metric thread

$$
\text{score}(x_i, x_j) = (W_Q x_i)^\top (W_K x_j) = x_i^\top \left(W_Q^\top W_K\right) x_j
$$

Writing $M = W_Q^\top W_K$, the score is a **bilinear form** — linear in each argument separately — and it generalizes the Euclidean inner product $x_i^\top x_j$, which is just the special case $M = I$.

This is where the metric thread arrives. In previous weeks the metric was taken as *given* and we asked what followed from it: the gradient as a covector under the Euclidean metric (Week 4), each optimizer as a different metric on parameter space (Weeks 7–9), the min-norm solution depending on the Euclidean inner product in two places (Week 11). Here the network **learns** $M$ from data. Same structure — an inner product decides what counts as "similar" — but the metric is no longer fixed to the identity.

Cross-links: [[w4s1-gradients-levelsets]] · [[w5s1-gd-euclidean]]

## Asymmetry reason for separate Q/K

With a shared $W$, the score is $x_i^\top W^\top W x_j$. But $W^\top W$ is symmetric, which forces $\text{score}(i,j) = \text{score}(j,i)$ — token $i$ attending to $j$ would always equal $j$ attending to $i$.

With separate projections, the score is $x_i^\top W_Q^\top W_K x_j$, and $W_Q^\top W_K$ is a general matrix — asymmetry allowed. Attention relationships usually *are* asymmetric (an adjective attending to its noun is not the same as the noun attending back), so this matters.

## Routing vs. payload for separate V

The query/key pair does the **routing** of the attention — determining *how much* to attend. $V$ is the **payload** — *what actually gets retrieved*.

They are separate because "what I match on" and "what I retrieve" are different questions. A search query matches on keywords but returns document contents, not the keywords again. A separate $W_V$ lets the network learn what information flows forward independently of what it matched on.

## Things to watch going into S2

- **Softmax axis:** row-wise over keys — each query's weights sum to 1. (Self-flagged weak spot.)
- **Shapes:** $QK^\top$ is $n_q \times n_k$; output is $n_q \times d_v$. These become the S2 assertions.
- **Softmax stabilization:**
  $$
  \text{softmax}(z)_i = \frac{\exp(z_i - \max_j z_j)}{\sum_j \exp(z_j - \max_j z_j)}
  $$
  This is an **identity, not an approximation**: the constant factor $e^{-\max_j z_j}$ appears in both numerator and denominator and cancels exactly. The result is unchanged, but the exponent stays bounded above by 0, preventing overflow.

## Cold-pass calibration

Three gaps surfaced between the cold reconstruction and the finished derivation.

1. **The softmax was missing from the cold formula.** I wrote $QK^\top V$. The regrouping $QK^\top V = Q(K^\top V)$ shows why that fails: with $K^\top V$ a fixed matrix, the whole thing collapses to a plain linear map of the queries and the content-dependence — the entire point of attention — vanishes. The nonlinearity is *necessary*, not decorative.

2. **The $\mathbb{E}[X^2]$-vs-$\mathbb{E}[X]$ moment reflex was rusty.** The variance derivation stalled until I had the $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$ scaffold in front of me; I initially substituted $\mathbb{E}[X]$ where $\mathbb{E}[X^2]$ was needed. Not a boundary/off-by-one error — a "which moment does variance need" gap. Worth a refresher before any week that leans on second moments.

3. **Vocabulary-vs-derivation slip — second week running.** In the first draft of this note the "asymmetry" and "routing/payload" headings had each other's content. This is the same failure mode as the equivariance/invariance label errors in Week 12, and the same fix worked: derive the object first, then read the label off it, rather than retrieving the word from memory. Two weeks in a row makes this a **standing pattern**, not a one-off — build an explicit label-check into S4 when writing the explainer.

**On the credit side:** the cold shapes were right ($QK^\top$ as $n_q \times n_k$, output as $n_q \times d_v$), and the $\sqrt{d_k}$ result now has two independent confirmations — the cold derivation and Manning's CS224n treatment.

## Forward

- **S2:** implement `scaled_dot_product_attention`; the three "things to watch" above are the assertions.
- **S3:** pre-reading conjecture is now committed — *Q/K build the addressing metric, V is the content being addressed.* Test it against Vaswani, and check whether the paper's $\sqrt{d_k}$ justification matches the derivation above.
- **S4:** the bilinear-form section is the spine of the explainer. Forward to [[w12s4-conv-explainer]] — conv hard-codes connectivity, attention learns it.
