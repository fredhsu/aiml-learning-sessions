---
created: 2026-08-14T20:48:58Z
id: 01a00208-fc1c-7e18-92d8-251297f50c45
---

# Week 13 Session 4 - Attention Explainer

The focus on this week has been attention and the transformer architecture. This brings together two previous ideas:

1. Differentiability - If we use argmax and retrieve, it isn't differentiable because it has a gradient of zero almost everywhere. But using a softmax weight average of all values produces a smoother landscape and is differentiable. This makes attention trainable.
2. Metric - The score weights (QK) are learned using a bilinear form $M=W_Q^T W_K$, so we learn the metric from the data. Compared with the inner product discussion in week 4, we are using the data to learn the metric instead of using I.

From week 12's convolution discussion we see some similarities between convolution and attention. Both produce linear maps: $AV$, but they set A differently. Conv fixes A as a Toeplitz band, the structure forms the bias. Attention uses $A=softmax(scores)$ and learns the structure from the content.

## 1. Hard lookup -> Soft lookup

The softmax function acts like a soft argmax, which in turn makes it differentiable. When creating a copy-task using a query acting as a shift $q_i = p_{i-1}$ mask for rows $i \ge 1$ the output was roughly 19 times closer to the sequence mean than to the target value it should've copied. This produces the right value if we were using argmax (the right key was the largest), but not much of a score difference. Since the winning key was just slightly larger than the other keys, softmax spreads the weight across all the keys near the mean. This shows that it is genuinely 'softer' than an argmax. However, even though we had the right pattern it isn't sharp enough. We can use a temperature 'c' to push things closer to a hard lookup. As $c$ approaches $\infty$ softmax becomes like argmax. So we're looking at how high we need to set the temperature to push the correct entries above 0.9. For this example using $x$ as the logit gap:

$$
e^x/(e^x+7) > 0.9  ⇒  e^x > 63  ⇒  x = c/√8 > ln 63  ⇒  c ≳ 11.72
$$

So using a temperature threshold around 11.72 produces a strong enough difference for the soft lookup.

## 2. Learned metric

From session 1, we found that score calculation can be made a bilinear form:

$$
score(x_i, x_j)=(W_Q x_i)^T (W_K x_j) = x_i^T M x_j
$$

with $M=W_Q^T W_K$.

This bilinear form is a generalized inner product with a learned metric. In contrast, in Weeks 4-5 we chose $I$ as a fixed metric inner product which defined the geometry. If we were to use the same matrices for $W_Q$ and $W_K$, then we'd have a sufficient condition for symmetric $M: M=W^T W$ and therefore $x_i^T M x_j = x_j^T M x_i$. A symmetric M gives symmetric scores and weighs our attention the same in both directions. In other words, if we used a symmetric matrix for M, then i would attend to j the same as j attends to i. For this use case we want asymmetry, so we use different $W_Q$ and $W_K$ allowing i and j to weigh each other independently. Note that this is different than the asymmetry we see in the A matrix, which is often asymmetric due to the row softmax.

Additionally session 1 showed why we divide the score by $\sqrt{d_k}$ to keep variance controlled to prevent softmax from saturating.

## 3. Attention vs Convolution

Let's dig deeper into the comparison we raised in the intro between convolution and attention. In both cases we have the output = $AV$, and linear in $V$, so the difference lies in $A$.

- Convolution: A is fixed as a Toeplitz band, and the weights are the same regardless of the input. The locality of the calculations is hard-coded as a bias.
- Attention: $A=softmax(scores)$, is now dependent on the input. The filter/connectivity is now chosen per input.
  - The cost of this additional flexibility is the [[translation equivariance]] we got for free with convolution is no longer there until you add it.
  - Note: The softmax used in A also means it is nonlinearly dependent on x. So we cannot claim that both cases are linear in A, only that they are linear in V.

So we can see convolution as a special case where A is set to a fixed band, and no longer dependent on the input. Or conversely that attention is convolution with the connectivity learned from the input data.

## 4. Multi-head

One issue with attention is that having a single head will produce one weighted average of the V rows. So we get one averaged out value vector per query and lose patterns from different subspaces. To address this the paper uses multi-head attention.

Multi-head attention uses $h$ heads in parallel to calculate attention in multiple subspaces (of dimension $d/h$), then concatenate them. This allows the individual heads to capture the different relationships without averaging them all together. Now each head has its own set of weights $W_Q, W_K, W_V$ resulting in a different learned metric $M_h$.

One correction from my earlier prediction was I thought this was positional, but instead its to attend to different representation subspaces. I had also mistakenly attributed the use of multi-head attention for computational speed up.

## 5. Positional Encoding

Since we use a weighted average for attention it is permutation-equivariant, so order must be added back. In the transformer architecture it is added to the token embeddings (not concatenated), which means the encoded positional information must have the same dimension as $d_{model}$. The encoding used in this paper is sinusoidal $PE(pos + k)$, which is a linear function of $PE(pos)$ since the linear map only depends on the offset $k$, not the absolute position, and the relative positions can be expressed as a fixed linear transform. This provides a more efficient encoding than a naive implementation of integer indexes, and empirically close to learned encodings.

## 6. Forward hooks

- How does Vision Transformers handle 2D positional encoding? Does the linear-offset / relative-position structure survive two axes, or does ViT just learn it?
- Does multi-head relate to conv-channels?
- Residual connections in the transformer → [[w10S3_Read_Landscape]] — skip connections and loss-landscape smoothing
