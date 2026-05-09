# Week 10 Session 3 - Reading

## Pre-reading

1. What can a 2D slice of a 354-dim landscape not show you?
   a. Any other dimensions outside of the two chosen
   b. The full picture of the landscape
   c. 1D slices of either of the two dimensions

2. From your own S2 experience, what did filter normalization actually fix?

   Scaling issues when the weights of the trained network are much smaller than the data

3. What's your prediction for the Dinh et al. argument?

   Rescaling the layers can have a noticeable effect on model performance when there is a big difference in the scale of the data and weights without changing the function behavior.

## Li et al. - Visualizing the Loss Landscape of Neural Nets

- Filter norm is introduced as a method for producing better plots that handle the fact that networks with homogeneous activations (ReLU) have rescaling invariance and can have large differences in scale.
- After applying filter norm, graphs have cleaner plots of sharpness and flatness, allowing for better comparison and interpretation, with examples showing the improved training of deep networks when skip connections are used. The filter makes use of the rescaling invariance noted above.
- Filter normalization produces plots that are invariant under the same function-preserving rescaling that ReLU networks themselves exhibit.
- Empirically flatter, less chaotic landscape geometry correlates to better generalization. (This is correlation not causation, the Dinh et al. paper will attack the causal side)
- The paper points out that the reduced dimension plots can show that a graph has non-convex regions, but if it shows a convex region you cannot assume the entire landscape is convex.

## Dinh et al. - Sharp Minima Can Generalize For Deep Nets

- This paper disputes the notion that flatness around the minima of a [[ReLU]] based network is a good measure of generalization.
- It uses reparameterization to transform the landscape, while preserving the function exactly (same outputs, same loss on every sample). Only the [[Hessian]], which depends on weight coordinates, changes.
- By reparameterizing, the authors are able to change the sharpness of the landscape, making the area around a minima much sharper without changing the generalization.
- It does not claim that this refutation holds for networks that are not based on ReLU as the activation function.
- The explicit construction relies on ReLU homogeneity, but the more general point is that Hessian [[eigenvalue]]s are parameterization-dependent and generalization is parameterization-invariant.

## Post-reading reflection

My S1 conjecture about large steps around flat-minima does not hold up directly, since we can rescale and change what it means to have a larger step. Where it does hold up is when you think about it in function space rather than the landscape of the weights.

Pre-reading answer 2 was meaningfully wrong: I framed filter normalization as a "weights vs. data" scale issue, but it's actually about "random direction vs. weights" scale matching, motivated by ReLU rescaling invariance. The data is not part of the picture.

Pre-reading answer 3 was also wrong: I predicted reparameterization would affect "model performance," but the paper's whole force is that it changes the Hessian without changing function, loss, or generalization at all. The dissociation between weight-space geometry and function-space behavior is the entire substance of the argument.

With a zero-loss manifold, there are a number of potential choices GD could take, some that generalize well, others that are memorizing the training data. Dinh et al. show that Hessian-based sharpness in weight space is parameterization-dependent and cannot, on its own, explain generalization. This pushes the explanation away from the static geometry of the minimum and toward the dynamics of how GD reaches it, answering which point on a zero-loss manifold gets selected, and why. Week 11 takes up this directly: among all the zero-training-loss solutions in an overparameterized network, why does GD systematically land on ones that generalize? The answer involves the trajectory of GD itself — what's known as implicit bias.
