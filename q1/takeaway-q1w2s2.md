# Takeaway

We have to do a summation of the per-example gradients since the total loss is the sum of the per-example losses and the differentiation is linear. We can ignore the constant 1/N used in finding the average of the squared loss since it does not affect the direction of the gradient and is a constant value.

I still have some error when doing manual derivations of the graph, namely in the re-substitution of intermediate variables in the chain rule.

I gained confidence in the backprop method by doing numeric differentiation with a finite difference as a gradient check and found the difference between the backprop gradients through analytical gradient and the numeric results were within a very small margin of error. Apparently this is a method used in by researchers and framework authors to verify results.

## Anchor Sentence

Because the total loss is defined as the sum of per-example losses, and differentiation is linear, the gradient of the total loss is the sum of the per-example gradients.
