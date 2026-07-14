---
created: 2026-07-03T14:49:21Z
id: fb3d27e0-e511-4278-ac92-74eb8ba487e8
tags:
  - explainer
  - convolution
  - inductive-bias
  - linear-map
  - toeplitz
title: Convolution as a Structured Linear Operator
---

# Convolution as a Structured Linear Operator

Convolution creates a structural inductive bias by using a sliding filter across
a multi-dimensional dataset such as an image. The filter defines an area of
interest that is slid across the input and matrix multiplied to produce a new
output map. This is the parameterized linear map from Week 4, but now with a
structural constraint [[w4s1-gradients-levelsets]]. The operation can also be
described by a banded Toeplitz matrix.

Most deep-learning libraries implement cross-correlation rather than mathematical convolution. The filter coefficients are applied in their original order rather than reversed. [w12s2-code.py] takes a similar approach. The Toeplitz structure, weight-sharing behavior, and inductive-bias arguments are the same, so this note follows the common convention of calling the operation convolution.

Here is an example for the 1D case: the
flattened input $x$ is $[1,0,0,0,1]$ and the filter is $[1,0,0]$. We start by
applying the filter to the first three elements and take the dot product to
produce the first element of the output:

```
INPUT GRID (5x1)     FILTER (3x1)            OUTPUT MAP (3x1)
+-----+               +-----+             +-----+
| *1* |               |  1  |             |  1  |
| *0* |       X       |  0  |    ====>    |  .  |
| *0* |               |  0  |             |  .  |
|  0  |               +-----+             +-----+
|  1  |
+-----+
```

Then we slide the filter down and get the next value:

```
+-----+               +-----+             +-----+
|  1  |               |  1  |             |  1  |
| *0* |       X       |  0  |    ====>    |  0  |
| *0* |               |  0  |             |  .  |
| *0* |               +-----+             +-----+
|  1  |
+-----+
```

Then repeat:

```
+-----+               +-----+             +-----+
|  1  |               |  1  |             |  1  |
|  0  |       X       |  0  |    ====>    |  0  |
| *0* |               |  0  |             |  0  |
| *0* |               +-----+             +-----+
| *1* |
+-----+
```

A Toeplitz matrix has the same value across each diagonal. For a 3-tap filter
$[a,b,c]$ acting on a length-5 input, the operator is $3\times5$ (output length
$n-m+1 = 5-3+1 = 3$):

$$
K=\begin{bmatrix}
a & b & c & 0 & 0\\
0 & a & b & c & 0 \\
0 & 0 & a & b & c \\
\end{bmatrix}
$$

The three nonzero diagonals are the filter taps and the off-band zeros are locality.
The resulting output is $y = Kx$, where $K$ is the Toeplitz matrix built from the
filter and $x$ is the flattened input. [[w12s1-conv-math]]

This example uses a stride of 1 and no padding. The general output length formula is:

$$
n_{out} = \lfloor \frac{n+2p-m}{s} \rfloor + 1
$$

For our example, n=5, m=3, p=0, and s=1:

$$
n_{out}=\frac{5-3}{1}+1 = 3
$$

## Weight sharing as parameter tying

Convolution combines two constraints. Locality reduces a dense $n(n-m+1)$ parameter map to a locally connected map with $m(n-m+1)$ parameters. Weight sharing then ties the $m$ weights across all output positions, reducing the number of independent parameters to $m$. The benefits are reduced computational load from the locality, and reduced storage by weight sharing.
When building the gradient for backprop, the resulting Jacobian has values that appear in many rows,
so $\partial L / \partial k$ sums the contribution across all the positions each
tap is used in. This was verified in the code example, where the finite-difference
grad-check reached $<10^{-5}$ relative error on both $\nabla_x L$ and
$\nabla_k L$, comparing the tied Jacobian against the gradient-summing structure.

This constraint forms the specific shape the layer can represent, and _is_ the
inductive bias of the function. So the _inductive_ bias is based on the
structure, in contrast to the _implicit_ bias that comes from the algorithm
[[w11s1-implicit-bias-math]].

Because this valid convolution maps $\mathbb{R}^5$ to $\mathbb{R}^3$, any single filter has a null space of at least two, so it will discard some input directions. The discarded directions depend on the filter coefficients. Using several filter stacks can reduce the shared blind subspace. The same structure that lets the layer see local patterns prevents it from seeing others. For example with $k=[1,1,1]$, the blind directions are inputs where every length-three sliding window sums to zero, not all inputs whose total sum is zero.

## The four subspaces

The matrix K has four fundamental subspaces:

1. **Input space — $\text{null}(K)$:** the blind spot of the filter caused by the
   structure of the convolution, inputs that produce zero output.
2. **Input space — $\text{row}(K)$:** the input directions the layer responds to,
   the orthogonal complement of the blind spot $\text{null}(K)$.
3. **Output space — $C(K)$:** the column space, i.e. the achievable outputs.
4. **Output space — $N(K^\top)$:** the left null space, outputs that cannot be
   produced.

This is the same four-subspace decomposition as Week 11, but now applied to a
structured operator instead of a matrix of data. [[w11s1-implicit-bias-math]]

## The inductive bias

The inductive bias of convolution can be seen in the equivariance and invariance
properties. Translation equivariance makes the output track a shift in the input:

$$f(\text{shift}_s\, x) = \text{shift}_s\, f(x)$$

So shifting then detecting is the same as detecting then shifting. This
equivariance does not hold generally (it does not work for rotation or scaling)
and is a result of weight sharing.

Local pooling provides invariance, identifying a feature regardless of its
absolute position. Pooling "summarizes" a local area, producing a single value
for a given region. For instance, a $3\times3$ max-pool over a region returns the
maximum value of that region:

```
+-------+
| 1 2 3 |
| 0 5 1 |  ==>  7
| 7 2 1 |
+-------+
```

Pooling can be effective at reducing the dimensionality of the data.

Depth stacking leads to compositionality, exposing hierarchical structure. For
instance: edges → parts → objects. The receptive field grows as
$1 + L(k-1)$ for $L$ stacked width-$k$ layers, stride 1, and dilation 1.

One limitation of these bias properties: discarding the location information
through invariance can be an issue for some use cases. For example, applying
convolution and pooling to a weather map may identify storms but lose the
location of the storm. The population affected by the storm needs the location
for the information to be useful, but that information may have been lost by
pooling. Local pooling will coarsen the storm's position, and global pooling discards the location almost entirely. (Note the division of labor: the equivariant conv layer _preserves_
position, pooling throws it away)

## Forward to attention

Looking forward, we will see how the hard-coded connectivity of convolution
compares to the learned connectivity of attention. Both opertations produce weighted sums of values. Convolution uses learned weights tied to fixed relative offsets. Attention computes its weights from query-key similarities of each input. For fixed queries and keys, attention is linear in the avlues ,but the full attention operation is content-dependent and nonlinear. [[w13s1-attention-math]]
