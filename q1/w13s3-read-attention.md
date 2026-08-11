---
created: 2026-08-08T15:57:46Z
id: 019fe218-38e9-7339-82f2-f5af1452beab
---

# Week 13 Session 3 - Reading Attention is all you need
## Why separate Q/K/V?
### Pre-reading 
- The query-key score is directional - provides a weighted score for the lookup of V.
- The score is how much to attend, the value is the actual target being attended to
- The paper does not go into specifics on the reasoning behind the different maps, however my work in [[w13s1]] goes deeper than the paper.
### Separate Q and K
- Having a separate $W_Q, W_K$ allows the score matrix to be asymmetric. 
- $S_{i,j} = x_i^T M x_j$ is a bilinear form with $M=W_Q^T W_K$. This is related to the thread on metrics and inner products from [[w4s2]].
- This asymmetry allows $i$ to attend to $j$ without $j$ attending to $i$
- My initial read on this was incorrect, I was assigning the asymmetry to the attention matrix A, not hte score matrix S. A is asymmetric in general since it comes after the row-wise softmax, and its asymmetry is independent of S.

## Why Multi-head?
### Pre-reading
- I thought it was related to speed and efficiency, this was an incorrect assumption.
- However if you take h heads of dimension d/h it is roughly equal to one head of dimension d, so there is little compute savings.
### Multi-head attends to different subspaces
- Multiple heads provides attention to different subspaces. With a single head, the value vectors are averaged and lose the distinct patterns of the different subspaces. Each query's output is a softmax weighted average of the V rows, so one head results in one averaged value vector. Multi-head computes the different weighted averages in parallel and concatenates them.
### My additional thoughts
- This is similar to how multiple convolutional channels work.
- I had also made an error here thinking that the multiple heads addressed different positions, not different subspaces.

## Why positional encoding?
### Pre-reading
- Positional encoding is trying to solve the issue of permutation equivariance. A naive way to handle it would be to just index the different positions with an integer. I did recall that sinusoidal functions were another method.
### Methods used in the paper
- In section 3.5 they highlight the use of sinusoidal encoding: $PE(pos + k)$ is a linear function of $PE(pos)$, and the map is a rotation that only depends on the offset k (angle addition).
- The encoding is added to, not concatenated with the token embeddings. Since we are adding the positional embedding, it must match the dimension of $d_{model}$.
- Learned embeddings were considered by the paper, but they decided to stay with sinusoidal since they had near identical results, and had some thoughts on extrapolation benefits of the sinusoidal.
### Adding vs concatenation
- Adding will combine the content and position in the same $d_{model}$ coordinates, compared to concatenation assigning different slots. This means that its up to the model behavior downstream to understand and disentangle the combined values.

## Calibration log
Here are some areas that I got mixed up to keep clear in the future. These may be good candidates for expanding into flashcards. I have had the right intuition attached to the wrong object multiple times but corrected it when revisiting the specific expression.
- A vs S matrices
- Positions vs subspaces
- When to concat vs add
### Prediction hits
- $\sqrt(d_k)$ confirmed
- permutation equivariance framing for positional encoding
### Prediction misses
- Multi-head attention justification - my guess on speed was disproven when the flat FLOP count was compared.

## Two Q2 questions + one backward link
- How does positional encoding generalize to 2D/grid data (images, ViT)?
  Does the linear-offset / relative-position structure survive two axes, or does
  ViT just learn it?
- Does the multi-head ≈ conv-channels analogy actually hold up?
- Residual connections in the transformer → [[w10s3]] — skip connections and loss-landscape smoothing
