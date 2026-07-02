---
created: 2026-06-08T23:20:09Z
id: 21ede718-4f53-4e07-81e4-57159396cd9a
---

# Week 12 Session 1 - Convolution as a Structured Linear Operator

## The operation (cross-correlation convention)

1D conv, input $x \in \mathbb{R}^n$, filter $k \in \mathbb{R}^m$. A single output entry:
$$y_i = \sum_{j=1}^{m} x_{j+i-1}\, k_j$$
The filter index $j$ runs $1 \ldots m$ and is **fixed** (the filter is the same at every position); the input read-position $j+i-1$ **slides** with the output index $i$. The two indices must be decoupled — $x$ and $k$ cannot share an index, or the filter would slide too.

Convention: this is **cross-correlation** (taps in increasing order, $k_1$ on the leftmost input). True convolution flips the kernel ($k_{m-j+1}$). Since $k$ is learned the distinction is immaterial in practice; what ML calls "conv" is cross-correlation.

> **The fused error to avoid.** The *number of outputs* ($n-m+1$, indexed by $i$) is not the *number of terms per output* ($m$, indexed by $j$). They happened to coincide in the $n=5, m=3$ toy case (both 3), which masks the distinction. Counting the toy case by hand — positions vs. multiplications — separates them.

## The Toeplitz form

$y = Kx$ with $K \in \mathbb{R}^{(n-m+1)\times n}$. Each row is the filter shifted one step right, zeros elsewhere. For $n=5, m=3$:
$$K = \begin{bmatrix} k_1 & k_2 & k_3 & 0 & 0 \\ 0 & k_1 & k_2 & k_3 & 0 \\ 0 & 0 & k_1 & k_2 & k_3 \end{bmatrix}$$
Rows count outputs ($n-m+1 = 3$); columns count inputs ($n = 5$). **Banded Toeplitz**: $K_{ij}$ depends only on $j - i$, so values are constant along diagonals, and the band is narrow (only $m$ nonzero diagonals). A dense-looking $\mathbb{R}^5 \to \mathbb{R}^3$ map pinned down by just 3 numbers.

> **Reflex check.** For $Kx$ to be defined, $K$'s column count must equal $\dim x$. A $5\times 3$ matrix times a length-5 vector is undefined — run this dimension check every time you write $y = Kx$.

## Output size

Derive from the index constraint, don't recall. The largest input index touched is $m + i_{\max} - 1$ (at $j=m$), and it must not exceed $n$:
$$m + i_{\max} - 1 \le n \implies i_{\max} = n - m + 1$$
That's valid conv, stride 1, no padding. General formula:
$$\text{out} = \frac{n - m + 2p}{s} + 1$$
- **Padding** adds $2p$ to the effective input length ($p$ each side). It *grows* the input, so the output grows — sign is $+$, not $-$.
- **Stride** divides the *span* (number of steps between first and last position) by $s$. The $+1$ counts the always-present starting position and stays **outside** the division.

Checks: $(5-3)/1 + 1 = 3$; $(5-3)/2 + 1 = 2$ (stride-2 starts at positions $\{1, 3\}$).

## Four fundamental subspaces of $K$ (Strang Ch. 3 — closes the W11 gap)

$K : \mathbb{R}^n \to \mathbb{R}^{n-m+1}$. For a generic (non-degenerate) filter, $\text{rank}\,r = n-m+1 =$ number of rows = **full row rank** (the max possible for a wide matrix; the map is onto). It is *not* full column rank — 3 rows can't make 5 columns independent.

For $n=5, m=3$ ($r = 3$):

| Subspace | Lives in | Dim | Meaning |
|---|---|---|---|
| Row space $C(K^T)$ | $\mathbb{R}^5$ | $r = 3$ | input directions the layer responds to |
| Null space $N(K)$ | $\mathbb{R}^5$ | $n - r = 2$ | inputs sent to zero — the **blind spot** |
| Column space $C(K)$ | $\mathbb{R}^3$ | $r = 3$ | all achievable outputs ($= \mathbb{R}^3$, onto) |
| Left null space $N(K^T)$ | $\mathbb{R}^3$ | $0$ | trivial |

Bookkeeping: input space $3 + 2 = 5$, output space $3 + 0 = 3$. **A subspace's dimension is not the dimension of the ambient space it sits in** — the row space is a 3-dim plane *inside* $\mathbb{R}^5$, not 5-dim.

This is the *same* orthogonal decomposition as Week 11 ($\mathbb{R}^p = \text{rowspace} \oplus \text{null}$), now applied to a **structured operator $K$** instead of a data matrix $X$.

### Null space = blind spot (concrete)

Box filter $k = (1,1,1)$, so $K$ sums each length-3 window. $Kx = 0$ requires every consecutive triple to sum to zero:
$$x_1+x_2+x_3 = 0,\quad x_2+x_3+x_4 = 0,\quad x_3+x_4+x_5 = 0$$
Subtracting adjacent equations gives $x_4 = x_1,\ x_5 = x_2$; the first equation then forces $x_3 = -x_1 - x_2$ (so $x_3$ is *not* free — subtraction alone drops a constraint, re-impose one original equation). General null vector ($x_1, x_2$ free → dim 2):
$$x = (x_1,\ x_2,\ -x_1 - x_2,\ x_1,\ x_2)$$
e.g. $(1, -1, 0, 1, -1)$ — it **oscillates**. An averaging filter passes smooth (low-frequency) content and annihilates fast (high-frequency) content: it is a **low-pass filter**, and $N(K)$ is literally the set of high-frequency signals it discards. The four-subspaces picture is not an abstraction — $N(K)$ is the concrete set of inputs this layer throws away.

## Weight sharing as a constraint

Dense $\mathbb{R}^5 \to \mathbb{R}^3$: $3 \times 5 = 15$ independent weights (general $n(n-m+1)$, roughly $n^2$). Conv: $m = 3$ free numbers. Two constraints collapse 15 cells to 3 degrees of freedom:

1. **Tying** — nonzero values repeat along diagonals (Toeplitz): the same $k_j$ is reused at every position → *parameter sharing*.
2. **Zeros** — off-band cells locked to 0, never free: each output sees only a local window → *locality*. (The zeros don't zero the output; they exclude out-of-window inputs from that output's sum.)

These two constraints **are** the inductive bias, one data assumption each:

- **Tying → translation equivariance.** One shared detector means the response depends on the *pattern*, not its absolute position. Shift the input by $t$ and the output shifts by $t$: same response, different output location. Justified iff a useful pattern can appear anywhere in the signal. (Contrast: a dense layer gives every position its own weights, so an edge at position 1 vs. position 3 is, to the network, two unrelated events.)
- **Zeros → locality.** Relevant dependencies are short-range; distant inputs are irrelevant to a given output.

> **Equivariance ≠ invariance.** Equivariance: the output *tracks* the shift. Invariance: the output *ignores* the shift entirely. Convolution gives equivariance; pooling later converts it to invariance — watch for exactly where that step happens in the S3 reading.

## 2D (conceptual only)

The filter becomes 2D and $K$ becomes **doubly-block-Toeplitz**. Same principle (diagonal tying + local support); not derived here.

## Forward to S2 (open thread — deliberately left to open S2)

The **gradient-summing structure** of $\partial L / \partial k$. Because each $k_j$ is shared across multiple rows of $K$, its gradient sums the upstream signal over every position where it was applied:
$$\frac{\partial L}{\partial k_j} = \sum_i \frac{\partial L}{\partial y_i}\, x_{j+i-1}$$
This sum is the signature of weight sharing and is exactly what the S2 backward pass must implement. Derive it on paper first, then code forward/backward and finite-difference grad-check **both** $\nabla_x L$ and $\nabla_k L$.

Two Jacobians to keep straight: $\partial y / \partial x = K$ (the Toeplitz matrix itself) vs. $\partial L / \partial k$ (the summed quantity above).

## Cold-pass calibration (where intuition missed)

- **Prompts 1–3 were one fused error**: output-count conflated with terms-per-output. Resolved only by counting the $n=5, m=3$ case by hand.
- **$\dim K$** written as $n \times m$ (transposed) twice — the $Kx$ inner-dimension check catches it immediately.
- **Output formula**: dropped the $+1$ and got the padding sign backwards. Recurring sign slip — padding *adds* to the effective length.
- **Subspaces**: confused rectangular full-rank terminology (full rank $= \min(\text{rows}, \text{cols})$) and subspace-dim vs. ambient-dim (wrote row space $= 5$).
- **Equivariance**: first said "detects differently," which contradicts the sharing justification; corrected to *same detection, shifted location*.

## Cross-links

- [[Week11_S1_ImplicitBias_Math]] — the rowspace/nullspace decomposition, reused here on a structured operator. (The W11 "weak spot" note flagged the four subspaces for refresh before W12; now closed.)
- [[Week5_S1_GD_Euclidean]] — the metric thread. The four-subspaces orthogonality is Euclidean, as in W11.
- [[Week4_...]] — the parameterized linear map *(fill in exact title)*. Convolution is that story with a structural constraint added.
- **Forward → Week 13 (attention):** convolution *hard-codes* its connectivity (a fixed band); attention *learns* it (content-dependent mixing weights). Both are linear maps from inputs to outputs; they differ in how the mixing weights are chosen — and the query-key inner product is where the metric thread sharpens.
