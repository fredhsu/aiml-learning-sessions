# Matrix Gradients for Y = XW + b (Batched Linear Layer)

Working through the matrix gradients, we start with defining the input shapes for X, W, and b. Assuming we're using batched training:

- X has shape (B, D)
- W has shape (D, K)
- b has shape (K,) (broadcast across batch)
- Output $Y = XW + b$ has shape (B, K)

Where:

- B: Batch size
- D: # features
- K: # Neurons/outputs

We’ll treat the upstream gradient (covector) as:

- $G = \frac{\partial L}{\partial Y}$ with shape (B, K)

## Shape contract for each op, forward and backward

| operation | input shapes           | output shape              | gradient shapes returned                                                         |
| --------- | ---------------------- | ------------------------- | -------------------------------------------------------------------------------- |
| $Y_0=XW$  | X: (B,D) <br> W: (D,K) | $Y_0$: (B,D)(D,K)->(B,K)  | $\frac{\partial L}{\partial X}$:(D,K) <br> $\frac{\partial L}{\partial W}$:(B,D) |
| $Y=Y_0+b$ | $Y_0$: (B,K), b: (K,)  | $Y$: (B,K) + (K) -> (B,K) | $\frac{\partial L}{\partial b}$:(K,)                                             |

Remember from the whole build up with covectors to this point, $df_X : T_X \to T_Y$ it is not a matrix but a linear map, so it does not have a defined shape. If we wrote out the full Jacobian it would be (BK)x(BD), very large. Instead we apply the adjoint to the linear map $df_X$. Since $df_X$ maps $(B,D) \to (B,K)$, the adjoint maps $(B,K) \to (B,D)$ which is what gives us the shape for $\frac{\partial L}{\partial X}$

So when we apply the inner product on the backward pass we use the Frobenius inner product:
$$\langle A,B \rangle := \sum_{i,j} A_{ij} B{ij} = tr(A^\top B)$$
We want to go from:
$$\langle G, \Delta Y_0 \rangle = \langle G, \Delta XW \rangle$$
to the form:
$$\langle (something), \Delta X \rangle$$
which aligns with our inner product definition for the covector.

1. $tr(WG^\top \Delta X)=tr((WG^\top \Delta X)^\top)=tr(\Delta X^\top (WG^\top)^\top)=tr(\Delta X^\top (GW^\top))$
2. $M=GW^\top : (B,K)(K,D)=(B,D)$

So on forward pass we have: $\Delta Y_0 = \Delta XW$

On the backward pullback, we have M such that $\langle G,\Delta Y_0 \rangle = \langle M, \Delta X \rangle$ is $M=GW^\top$

Which matches up with how we calculate the vjp, upstream gradient * adjoint of the Jacobin. For $\Delta X$, $W^\top$ is the adjoint of the Jacobian. With batching we're doing one VJP per row.

- JVP: $\Delta X \mapsto \Delta XW$.
- VJP: $G \mapsto GW^\top$.
- Same linear map, opposite direction, adjoint in the middle.
