# Read actively: gradients, metrics, and optimization bias

## Reading chapter 4 Deep Learning

pg. 82 - discussion on partial derivatives, gradient (vector containing all partial derivatives), directional derivative, unit vector $u$(slope in direction $u$)

- The dot product $\nabla f(x) \cdot v$ is a directional derivative and local change: $D_v f(x) = \nabla f(x) \cdot v$
- The unit vector gets multiplied by the gradient to define update directions and step size: $-\eta \nabla f(x)$ or $|\nabla f(x)|\hat{u}$
- By using the unit vector we can focus on direction, since the dot product mixes direction and magnitude.
- The dot product $\nabla f \cdot v$ is a covector acting on a vector
- Writing $\nabla f$ as a vector already assumes an inner product
- The unit vector is how you separate direction from magnitude
  The vector you choose to dot with the gradient is a unit direction you’re probing, while the gradient itself encodes both the direction of steepest change and the magnitude of that change.

The gradient vector $\nabla f(x)$ is the inner-product representation of the derivative covector $df_x$, and its action on a tangent vector $v$ is given by $\langle \nabla f(x), v\rangle$.

### 4.4 Constrained Optimization

In constrained optimization, optimality means the derivative covector vanishes on all tangent directions permitted by the constraint, which forces it to lie in the span of the constraint’s covectors.

I think I have a clearer picture on the covector formulas:

$$
df_x(v)=\langle \nabla f, v \rangle
$$

$$
\langle \nabla f, v \rangle = 0 \\
$$

The components of the gradient in standard coordinates are the partial derivatives, and the gradient is a representation of the covector based on the chosen inner product. The gradient always points in the direction normal to the level set. So what we are working through is finding the vector $v$ that is the tangent vector so that when we take the inner product it is equal to zero, the covector is fixed. The covector is the abstract linear mapping of the derivative, which can be realized as a vector form when we use the inner product to define the vector being applied.

Level sets are defined by a covector; normals appear only after choosing an inner product.

**revised**

The covector $df_x$ is fixed at a point. By choosing an inner product, we represent this covector as a gradient vector $\nabla f(x)$. When we take the dot product $\langle \nabla f, v\rangle$, the gradient is fixed and we vary the tangent vector $v$, which represents possible directions of motion. Tangent vectors that make this dot product zero correspond to directions along which the function does not change. In backpropagation, covectors are what get pulled back through the graph, but in practice they are represented as gradients, so their action via inner products is implicit.

## Questions

How does this Geometric approach apply to Jacboian/Hessian?
