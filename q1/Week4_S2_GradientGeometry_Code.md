# Gradient Geometry code

- The backwards functions that calculate the gradient, use the covector implemented as `out.grad` where the inner product has been implicitly applied to produce a value for the gradient.
- In some cases the gradient norm is zero for a neuron, indicating that it will not change, geometrically this would mean we move along the level set for those parameters.
- What surprised me was how many neurons in such a small network were deactivated. I'm assuming this is due to vanishing gradients.
