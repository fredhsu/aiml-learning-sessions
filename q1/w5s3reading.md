The reading focuses on a quadratic use case, so the discussion that follows is based around that example. Curvature of the surface (the Hessian) limits how fast GD can move because the shape of the level sets / contour lines determine the descent path. This is directly related to the condition number which is calculated as the ratio of largest to smallest eigenvalues of the Hessian $\kappa = \frac{\lambda_{max}}{\lambda_{min}}$ . If $\kappa = 1$ then the contour lines are circles and the gradient will be pointing directly at the minimum which would allow you to go minimize in a single step given the right learning rate. A large condition number means the "bowl" is elongated leading to a narrow ravine. This creates a less than ideal path for gradient descent with steep walls on the sides and a gentle slope along the "valley floor". GD follows the negative gradient each step, following a zigzag path jumping back and forth across the ravine rather than along the axis of the valley towards the minimum. The axis here is defined by the eigenvectors, but the gradient does not normally follow the eigenvector because the large $\lambda_{max}$ dominates the gradient. This is a result of using the Euclidean metric (which views the world as a sphere), which does not take into account the steepness of the walls. A different metric could account for this, making a step along the gentle slope worth more than a steep one.

To decode the eigenvectors and values a bit more:

- One eigenvector points up the steepest walls.
- The other points along the axis of the valley towards the minimum.
- Eigenvalues indicate the amount of curvature along their eigenvectors.
- Large $\lambda$ is a wall with steep slope, if you make a large step down this wall, the next step will flip direction.
- Small $\lambda$ is the valley floor, you can make large steps here safely, but GD can't tell.

A small learning rate is needed to avoid divergence along the large eigenvalue direction $\alpha \lt \frac{2}{\lambda_{max}}$ which defines 'trust' region constraint. However, this constraint slows our progress along the valley axis where it matters most.

Steepest only makes sense with the context of a metric, we need the metric to define what 'steep' means. This example uses the Euclidean metric, and shows we can fall into a trap when the conditioning number is large.

We can try to address this weakness of GD by changing the inner product to make the level sets circular in the new coordinates, preconditioning the coordinates to get $\kappa$ as close to 1 as possible.
