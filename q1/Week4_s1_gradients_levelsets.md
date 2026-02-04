# Week 4 Session 1

## Derivative as linear map

A covector is a linear map that consumes a vector and produces a scalar output. It serves as a function, not actual values. The derivative of a function $f : \mathbb{R}^n \to \mathbb{R}$ at a point $x$ is $df_x : \mathbb{R}^n \to \mathbb{R}$, which is a covector. Here the covector takes a tangent vector and outputs the rate of change.

## Level set

A level set describes all the values that produce the same output for a function. Similar to a contour map where all the contour lines remain at a similar altitude. Tangent vectors to the level set will move along the level set, so their rate of change is zero. This shows that $df_x(v)=0$ for all tangent vectors $v$ to the level set.

## Normals

If we represent the covector $df_x$ using an inner product, then for any tangent vector $v$ we have:

$$
df_x(v)=\langle \nabla f, v \rangle \\
\langle \nabla f, v \rangle = 0 \\
$$

since we know $v$ is tangent to the level set if we find a gradient that gives a zero inner product with $v$ then we know it is normal to the level set since a inner product of zero indicates they are orthogonal. This means the gradient would be the direction steepest ascent since it goes in the direction normal to the level set.

## Inner product choice

By choosing an inner product, we choose how to represent a covector as a vector. The set of vectors for which the covector is zero defines the tangent space to the level set, while the chosen inner product determines which vector we call the gradient.

## Worked example

Starting with $f(x,y) = x^2 + 2y^2$

$$
df_{x,y}(v_1,v_2)=2xv_1+4yv_2
$$

Choosing a point $(x_0, y_0)$ and setting equal to zero to find tangents

$$
df_{x_0,y_0}(v_1,v_2)=2x_0v_1+4y_0v_2 = 0
$$

Now pick a specific value $(x_0, y_0)=(2,4)$

$$
2(2)v_1 + 4(4)v_2 = 0 \\
v_1=-4v_2
$$

Then we can pick a specific $v_1 = -4$ which gives the vector $(-4,1)$, so moving in the direction $(-4,1)$ will stay along the level set (no change). Now to find the gradient $\nabla f(x,y)=(2x, 4y)$ at point $(2,4)$ gives $(4,16)$ if we take the inner(dot) product with the tangent vector:

$$
(4,16) \dot (-4,1) \\
(4 * -4) + (16 * 1) = 0
$$

Since their dot product is zero, they are orthogonal.

## Why gradients aren't really vectors

Gradients are actually covectors, since they are linear maps from vectors to a scalars. When you choose an inner product and coordinate, they can then be represented as vectors. This happens intrinsically in most ML code, so we don't work with the covectors directly.

In backprop, we propagate covectors backward through the graph and only later represent them as vectors via an implicit inner product.
