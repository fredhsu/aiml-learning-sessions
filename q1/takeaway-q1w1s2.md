# Takeaway

I was surprised by the trouble I had making the topo graph work correctly. That is still fuzzy. I also forgot the proper way to handle the backward function, namely to do the gradient updates as part of the backward call.

The basic structure of the class and the operators was mostly simple.

The overall process is simple, calculate local values and prepare local derivatives (i.e. define the backward function with the correct partial derivatives), then do the gradient calculation on the backward pass. One of the keys is that the backward pass must be done in topological order.

## Anchor Sentence

Autodiff is just local derivatives plus a graph traversal that respects dependency order.
