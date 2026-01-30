# Week 2 Session 3 Writeup

Backpropagation is a specific algorithm that implements reverse mode autodifferentiation and is used to find the gradients for parameters in a neural network. It operates on a computational graph (a directed acyclic graph), caching the intermediate activations during the forward pass (essentially a composition of functions). Once the forward pass is complete, backprop uses the chain rule and reverse topological ordering of the graph to calculate the derivatives of the final output with respect to the intermediate nodes and parameters of the network in a single backward pass by pushing the adjoints (sensitivities with respect to loss). The adjoints are pushed at each node by multiplying the upstream gradient by the local Jacobian (calculated using the cached activations). This leverages the Vector-Jacobian Product (VJP), making reverse mode efficient when there's a single scalar output (loss) and high-dimensional parameter space. The scalar loss starts the backward pass with an initial adjoint $\frac{\partial L}{\partial L} = 1.0$. The VJP avoids constructing the full Jacobian. For element-wise operations, the Jacobian is diagonal, so the VJP reduces to a Hadamard product; for matmul, closed-form expressions can be used to avoid explicit Jacobian construction. Mathematically, the forward pass composes functions to produce the output, while the backward pass composes their adjoint linearizations. In comparison, forward mode autodiff would require a pass for every parameter.

Backpropagation is reverse-mode automatic differentiation of a scalar functional defined by a composition of functions, implemented by composing adjoint linearizations in reverse topological order.

Adjoint: in practical terms it is the transpose, which we end up using that because we are moving backward:
Linear map: A: V -> W
Adjoint: A^_: W^_ -> V^_
A pushes vectors forward
A_ pull gradients (covectors) backward

Forward mode uses J_f x
Reverse mode uses x=J_f^T y

Each node:
𝑥 → 𝑓 𝑦 x f ​ y

induces two maps:

Forward: 𝑓 ∗ : 𝑇 𝑥 → 𝑇 𝑦 f ∗ ​ :T x ​ →T y ​

Backward: 𝑓 ∗ : 𝑇 𝑦 ∗ → 𝑇 𝑥 ∗ f ∗ :T y ∗ ​ →T x ∗ ​

Backprop computes: ∂ 𝐿 ∂ 𝑥 = 𝑓 ∗ ( ∂ 𝐿 ∂ 𝑦) ∂x ∂L ​ =f ∗ ( ∂y ∂L ​)

<https://chatgpt.com/s/t_6972d09aa53c8191aedd186a48f70a19>

VJP: when using multivariate chain rule with a single output, we can use VJP to find the gradients more efficiently than calculating the full Jacobian which would be sparsely populated. With a single output the Jacobian becomes a row vector and the gradient of the weights is the outer product of the upstream gradient and the input:
$$ \nabla_W L = J_W^T \cdot (\nabla_y L) $$

Where y is the output, $\nabla_yL$ is the upstream gradient (signal of sensitivity of loss to output), $J_W = \frac{\partial y}{\partial W}$ is the Jacobian representing the local sensitivity of the layer.

For a linear layer this can be made more efficient with an outer product that avoids calculating the actual Jacobian:
$$ \nabla_W L = (\nabla_y L)x^T$$
This effectively calculates the numerical result of $J_W^T v$ where v is $\nabla_y L$ without building the actual Jacobian.

Clarificaion:
Neilsen refers to "error" as a more physical and historical reference to what is mathematically the adjoint. Backprop and NN predate autodiff theory, and error is a bit more intuitive but mathematically it is a cotangent vector/sensitivity being passed as an ajdoint. In his terms the \delta "error" is the adjoint
