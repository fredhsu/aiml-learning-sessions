---
title: Q1 Index
tags:
  - index
---

# Q1 Index

> Map of Q1 by topic and session artifact. `—` means a session intentionally produced no note; **Not captured** means its planned output is absent from the vault; **Not yet** marks a planned Week 14 output.

## Quarter thread

**Through-line:** Backpropagation computes covector gradients; a chosen metric turns them into update directions. Optimization methods change that effective geometry and thereby their trajectories and implicit biases. Convolution and attention then instantiate two connectivity patterns: a fixed structured map and an input-dependent learned map.

**Recurring questions:**

- What are the shapes and constraints of the problem?
- Which biases do different operations introduce, and how can they be corrected or exploited?
- Which metric or preconditioner determines the update direction, and what learning behavior follows?

## Week-by-week artifact map

| Week | Topic / central question                                                            | S1 — Math                                   | S2 — Code                                                                           | S3 — Reading                       | S4 — Explainer / synthesis                                                       | Key result, correction, or open question                                                                                   |
| ---- | ----------------------------------------------------------------------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------- | -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| 1    | Basic backprop and autodiff                                                         | —                                           | [[w1s2-plan]] · [[w1s2-takeaway]]                                                   | —                                  | —                                                                                | Establish the first learning loop and reverse-mode vocabulary.                                                             |
| 2    | Backprop and VJPs                                                                   | [[w2s1-plan]] _(plan; output not captured)_ | [[w2s2-takeaway]] · [[code/experiments/w2s2-code.py]]                               | [[backprop-explainer]]             | **Not captured**                                                                 | Reverse mode propagates adjoints as VJPs, making scalar-loss gradients cheap.                                              |
| 3    | Cross-entropy and softmax                                                           | **Not captured**                            | [[w3s2-code-notes]] · [[code/experiments/w3-code.py]]                               | **Not captured**                   | [[backprop-explainer#Week 3 updates]]                                            | Stable loss layers and finite-difference checks make the learning loop testable.                                           |
| 4    | Gradients are covectors                                                             | [[w4s1-gradients-levelsets]]                | [[w4s2-gradient-geometry-code]] · [[code/experiments/w4-code.py]]                   | [[w4s3-read-gradient-geometry]]    | [[backprop-explainer#Week 4 Updates]]                                            | A derivative is a covector; its gradient-vector representation depends on the metric.                                      |
| 5    | Geometric view of gradient descent                                                  | [[w5s1-gd-euclidean]]                       | [[w5s2-code]] · [[w5s2-v2]] · [[code/experiments/w5s2-code.py]]                     | [[w5s3-reading]]                   | [[backprop-explainer-v04]]                                                       | Condition number explains ravine zig-zag; “steepest” is metric-dependent.                                                  |
| 6    | Batched linear layers and momentum                                                  | [[w6s1-matrix-gradients]]                   | [[code/experiments/w6s2-code.py]]                                                   | [[w6s3-momentum-reading]]          | [[code/experiments/w6s4-code.py]] · [[code/experiments/w6s4-momentum-visual.py]] | Shape contracts prevent silent broadcasting bugs; momentum is an EMA that damps ravine oscillation.                        |
| 7    | Momentum optimizer                                                                  | [[w7s1-momentum-math]]                      | [[w7s2-optimizer-code]] · [[code/experiments/w7s2-code.py]]                         | [[w7s3-code]]                      | [[w7-backprop-explainer]]                                                        | Momentum cancels oscillating components while preserving consistent progress down a valley.                                |
| 8    | Weight decay and regularization                                                     | [[w8s1-regularization-math]]                | [[w8s2-train-val-code]] · [[code/experiments/w8s2-code.py]]                         | [[w8s3-read-training-diagnostics]] | [[w8s4-diagnostic-checklist]]                                                    | Separate training loss from validation behavior; use diagnostics before changing hyperparameters.                          |
| 9    | Adam optimizer                                                                      | [[w9s1-adam-math]]                          | [[w9s2-adam-code]] · [[code/experiments/w9s2-code.py]]                              | [[w9s3-read-adam-w]]               | [[w9s4-optimizer-cheat-sheet]]                                                   | Adam adapts a diagonal preconditioner; AdamW decouples weight decay from the adaptive update.                              |
| 10   | Hessian landscapes                                                                  | [[w10s1-hessian-landscape]]                 | [[w10s2-landscape-viz-code]] · [[code/experiments/w10s2-code.py]]                   | [[w10s3-read-landscape]]           | [[w10s4-writeup]]                                                                | The Adam comparison slice is still a generate-or-discharge debt; landscape slices are only partial evidence.               |
| 11   | Implicit bias                                                                       | [[w11s1-implicit-bias-math]]                | [[w11s2-implicit-bias-code]] · [[code/experiments/w11s2-code.py]]                   | [[w11s3-read-implicit-bias]]       | [[w11s4-implicit-bias-explainer]]                                                | GD from zero selects the minimum-norm solution; Adam showed **equalization, not concentration**.                           |
| 12   | Convolutional networks                                                              | [[w12s1-conv-math]]                         | [[code/mini/conv1d.py]] · [[code/experiments/w12s2-code.py]]                        | [[w12s3-read-cnn]]                 | [[w12s4-conv-explainer]]                                                         | Convolution is translation-**equivariant**, not invariant; its connectivity is a fixed banded map.                         |
| 13   | Attention                                                                           | [[w13s1-attention-math]]                    | **Not captured** _(the plan describes an implementation, but none is in the vault)_ | [[w13s3-read-attention]]           | [[w13s4-attention-explainer]]                                                    | Keep score-matrix \(S\) asymmetry distinct from row-softmax \(A\) asymmetry; assert \(A \in \mathbb{R}^{n_q \times n_k}\). |
| 14   | [[w14-lesson-plan]] — Quarter wrap: cleanup, calibration, synthesis, and Q2 handoff | This index _(repo and vault cleanup)_       | **Not yet** — closed-book self-test                                                 | **Not yet** — Q1 synthesis         | **Not yet** — Q2 launch note                                                     | Close or discharge the Adam-slice and artifact-capture debts before Q2.                                                    |

## Topic map

### Backpropagation and autodiff

- **Weeks:** 1–3
- **Core artifacts:** [[backprop-explainer]] · [[w3s2-code-notes]] · [[code/engine/value.py]]
- **Core notes:** Reverse-mode autodiff evaluates vector–Jacobian products on a computational graph, efficiently producing gradients of a scalar loss with respect to many parameters.
- **What to revisit:** Re-derive a VJP and explain why reverse mode avoids materializing the full Jacobian.

### Geometry and metrics

- **Weeks:** 4–6
- **Core artifacts:** [[w4s1-gradients-levelsets]] · [[w5s1-gd-euclidean]] · [[w6s3-momentum-reading]]
- **Core notes:** The derivative is a covector. An inner product identifies it with a gradient vector, so the geometry and parameterization determine what counts as steepest descent.
- **What to revisit:** Reconstruct the covector-to-gradient conversion and relate condition number to ravine dynamics.

### Optimization and implicit bias

- **Weeks:** 6–9
- **Core artifacts:** [[w7s1-momentum-math]] · [[w8s4-diagnostic-checklist]] · [[w9s4-optimizer-cheat-sheet]]
- **Core notes:** Momentum and adaptive methods alter the effective update geometry and trajectory. Regularization and optimization choices affect which solutions are selected.
- **What to revisit:** Distinguish a change to the update rule or effective conditioning from a change to the loss landscape itself.

### Generalization and loss landscapes

- **Weeks:** 10–11
- **Core artifacts:** [[w10s3-read-landscape]] · [[w10s4-writeup]] · [[w11s4-implicit-bias-explainer]]
- **Core notes:** Landscape visualizations build intuition but are partial slices. Implicit bias explains what an optimizer selects among many fitting solutions.
- **What to revisit:** State the limits of a loss slice and re-derive why zero-initialized GD yields a minimum-norm solution.

### Architectural building blocks

- **Weeks:** 12–13
- **Core artifacts:** [[w12s4-conv-explainer]] · [[code/mini/conv1d.py]] · [[w13s4-attention-explainer]]
- **Core notes:** Both operations are conditionally linear in \(V\) when \(A\) is fixed. Convolution fixes \(A\) as a structured banded map; attention derives \(A\) from the input through learned similarity.
- **What to revisit:** Derive the attention shapes and explain why attention is not globally linear in its input.

### Quarter wrap and Q2 handoff

- **Week:** 14
- **Core artifact:** [[w14-lesson-plan]]
- **Core notes:** Cleanup makes the Q1 work reproducible; calibration, synthesis, and explicit Q2 choices turn it into a usable foundation.
- **What to revisit:** Complete the self-test, record misses honestly, and decide the NumPy-versus-PyTorch boundary for Q2.
