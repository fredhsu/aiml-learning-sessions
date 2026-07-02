Convolution acts as a structural description to the implicit bias provided by using a sliding filter across a multi-dimensional dataset such as an image. The filter defines an area of interest that is slid across the input and matrix multiplied by a K-banded Toeplitz matrix.

```
INPUT GRID (5x1)     FILTER (1x3)            OUTPUT MAP (3x3)
+-----+               +-----+             +-------------+
| *1* |               |  1  |             |  1   .   .  |
| *0* |       X       |  0  |    ====>    |  .   .   .  |
| *0* |               |  0  |             |  .   .   .  |
|  0  |               +-----+             +-------------+
|  1  |
+-----+
```

Toeplitz matrix has the same value across diagonals:

$$
\begin{bmatrix}
a & 0 & 0 \\
0 & a & 0 \\
0 & 0 & a \\
\end{bmatrix}
$$

The resulting output is: $y=Kx$ where K is the Toeplitz matrix built from the filter and x is the flattened input.

One of the features of convolution networks is weight sharing, which reduces the memory and computational load. Without weight sharing the parameter count would be $n(n-m+1)$, but with weight sharing it is constrained to m. The resulting Jacobian has values that appear in many rows of $\partial L / \partial k$ which sums the contribution across all the positions it is used in. This constraint forms the specific shape the layer can represent and forms the inductive bias of the function. So the inductive bias is based on the structure, in contrast to the _implicit_ bias that is from the algorithm (as seen in week 11 GD).

The result of applying the convolutional layer is four subspaces:

1. Input space : null(k) the blind spot of the filter caused by the structure of the convolution, inputs that produce zero output
2. Input space : rowspace(k) by applying linear combinations we start with an initial and stay within rowspace to build C(k)
3. Output space : C(k) the possible outputs
4. Output space : N(K^T) nullspace, outputs that cannot be produced

The inductive bias of the convolution is seen in the equivariance and invariance properties. Translation equivariance makes the output track the shift in input: $f(\text{shift x}\_s) = \text{shift}\_s f(x) $, so shifting then detecting is the same as detecting then shifting. This equivariance does not hold generally (does not work for rotation or scaling) and is a is a result of weight sharing.

Local pooling provides invariance, and identifies the feature regardless of the absolute position.

Depth stacking leads to compositionality, exposing hierarchical structure. For instance: edges -> parts -> objects, with a receptive field that grows as $1+L(k-1)$. Max pooling is one of the most common pooling mechanisms. Pooling can be effective at reducing the dimensionality of the data.

Invariance incurs a limit to the problems convolution can address. For example, applying convolution and pooling to a weather map may identify storms, but lose the location of the storm. The population that is affected by the storm needs the location for the information to be useful, but that information has been discarded.

Looking forward, we will see how the hard-coded connectivity of convolution compares to the learned connectivity of attention. Both provide linear maps from values to inputs, but differ in how the weights are set.
