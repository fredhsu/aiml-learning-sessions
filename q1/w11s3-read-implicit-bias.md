---
created: 2026-05-22T17:58:50Z
id: 019e50d7-0a93-7941-b36d-31c98d259959
---

# Week 11 Session 3 - Implicit Bias

## Pre reading questions

1. Your hypothesis. Does GD always prefer "simple" solutions? What does "simple" even mean when the model isn't linear? (You've now seen what "simple" means in the linear case — min-ℓ2\ell_2
   min-ℓ2\ell_22​-norm. What's the analogue for a deep network?)
   I think it will prefer simple solutions, with the definition of simple being minimized error.
   -- Revised
   My guess at what might be 'simple' is a small weight norm.

2. A prediction. Will the Neyshabur et al. argument generalize cleanly from linear regression to deep networks, or will it require new ideas? Why?

   No, it will require new ideas since the definition of simple and the calculations will be different. Since deep networks will introduce non-linearities, the math does not directly apply.

3. A connection. How does the implicit-bias story relate to the Dinh et al. counterexample from Week 10? (Hint from the plan: Dinh et al. ruled out one static explanation for generalization; implicit bias is a dynamic candidate.)
   We have a more straightforward / simple use case with the linear example to show how GD and minimization/generalization can be calculated, similar to the static analysis. However, these properties no longer hold for analyzing the deep network, even though some of the empirical results may imply this is the case.
   -- Revised
   Dinh, et al. removed the static explanation by showing a counterexample to using the geometry of the loss landscape as a way to determine generalization, and forced the discussion to a find a dynamic one. Implicit bias is one such dynamic explanation, looking at the path we take rather than just the final destination.

4. From your experiment. You just saw that Adam lands on a different solution than GD on the linear problem — equalized rather than min-norm. What do you predict Neyshabur et al. will say about the role of the specific algorithm choice (not just "use an algorithm"), based on the title alone?

The algorithm choice will dictate implicit regularization, which will be the true measure of generalization, and is dictated by the algorithm choice instead of looking at implicit bias which was shown to be influenced by either regularization algorithm or initial values.

## Post reading Neyshabur paper

1. What empirical phenomenon does Section 2 demonstrate, and how does it relate to the Zhang et al. memorization result?
   Increasing the size of the network well beyond what is needed to get to zero training loss continues to improve the testing loss even after the network has likely memorized the training data. The connection to Zhang et al paper is that after memorization, growing the network doesn't hurt generalization.

2. What's the matrix factorization analogy? In one sentence: what's the parallel to deep learning that the authors are drawing?
   With matrix factorization it was shown that the trace norm, not the rank is what should be bounded for improving generalization. The analogy is that for deep learning the rank is like the network width, and so there is a norm that should be the bounding factor.

3. What specific quantity do they propose as the implicit regularizer? Is it a specific norm? A vague "complexity measure"? Something else?
   They do not propose a specific norm, only show that it is some norm that should be the bounding factor. Which exact norm is left as an open question.

4. What assumptions does their argument depend on? (Specific architecture? Specific optimizer? Specific loss function?)
   Single hidden layer ReLU, with SGD as the optimizer. No weight decay, momentum, or Adam. Loss is based on cross entropy with truncation. The implicit bias explanation is suggested here, but still not proven.

5. Updated answers to your pre-reading questions. Especially Q1 (what's "simple" for a deep network?) — did the paper sharpen or change your answer?
   For deep networks, simple would be the low norm of the function, not a small number of parameters. The width of the network can be as big as you want, but some norm needs to be bounded. More specifically it is a scale-invariant norm.

## Post reading Zhang et al

This paper ruled out the classical function-class explanation. Capacity isn't the binding constraint, so the constraint is something else, and implicit bias is a possibility.
