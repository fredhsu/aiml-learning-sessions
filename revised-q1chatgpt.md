# Q1 Revised Weekly Plan (Weeks 4–13)

_Status: you’re currently in Week 4._  
**Q1 spine:** foundations + backprop + optimization + “trustworthy training”  
**Light-touch threads (Q1):** CNN-perception preview + RL grammar preview + robotics loop thinking

## Working assumptions from your progress so far

You’ve already spent serious reps on:

- Backprop as reverse-mode autodiff / VJP on a computational graph
- Cross-entropy + softmax derivatives, and why “CE-from-logits” is the stable route
- LogSumExp stability intuition
- Finite-difference gradient checking and what “close enough” means
- Bernoulli log-likelihood → derivative → MLE (and the BCE connection)
- Jacobian vs VJP compute/shape thinking (vectors/covectors, adjoints)

So the rest of Q1 is about **consolidating into a usable, tested mini-framework**, plus small previews of CNN/RL/robotics _without derailing the main spine_.

---

# Week 4 — Consolidate: CE-from-logits + LogSumExp + Gradient Check (Finish strong)

**Focus:** make your loss + backward pass stable and verifiable.

### Deliverables

- Stable **cross-entropy-from-logits** implementation (binary _or_ multiclass; pick one and do it right).
- A gradient-check harness that you can point at _any_ tiny graph.
- A short note: “LogSumExp: what it fixes and why max-shift works.”

### Sessions

**Session 1 (Math & derivations)**

- Re-derive CE-from-logits end-to-end (include the stability step explicitly).
- Write down the gradient you expect wrt logits: softmax(logits) − one_hot(label).

**Session 2 (Code from scratch)**

- Implement CE-from-logits with max-shift + LogSumExp.
- Backward: verify the logits gradient matches the derivation.

**Session 3 (Read actively)**

- Re-read your backprop references and rewrite them in your own notation:
  - Nielsen’s $\delta$ idea + “error at layer” framing (preview the translation you’ll formalize later).

**Session 4 (Write/teach)**

- 1–2 pages: “Why CE-from-logits beats softmax then CE” + “LogSumExp in one diagram.”

**Optional Session 5 (Non-technical)**

- One short reading on falsifiability / model testing; connect to gradient checks and unit tests.

**Checkpoint**

- Gradient check passes on a tiny random net (small epsilon, multiple random seeds).
- You have at least 2 “regression tests” for past numerical issues (overflow/underflow).

---

# Week 5 — Mini-batching + BCE/Bernoulli + Logistic Regression Bridge

**Focus:** turn “single-example training” into “real training loop,” and unify BCE with Bernoulli likelihood.

### Deliverables

- Mini-batch training (mean loss + mean gradient) + a clean update step.
- A derivation note connecting:
  - Bernoulli log-likelihood ⇄ BCE
  - logistic regression as a linear model + sigmoid + BCE

### Sessions

**Session 1**

- Derive batch loss: average of per-example losses → gradient is average of per-example gradients.
- Show explicitly why averaging keeps learning rate meaning stable across batch sizes.

**Session 2**

- Implement mini-batch accumulation:
  - Option A: accumulate grads per sample then average
  - Option B: compute mean loss then one backprop (depending on your graph design)

**Session 3**

- Read: skim relevant parts of :contentReference[oaicite:0]{index=0} that discuss cost functions + backprop intuition.
- Add “δ notation” breadcrumbs you’ll formalize later.

**Session 4**

- Write: “BCE as negative log-likelihood of Bernoulli” + “Why MLE leads to cross-entropy.”

**Optional Session 5**

- Philosophy note: “What counts as evidence my code is correct?” (gradient check, invariants, tests).

**Checkpoint**

- You can train a logistic regression model on a toy dataset and see loss decrease reliably.

---

# Week 6 — Vectorization & Shapes: Linear Layers, MatMul, Broadcasting Discipline

**Focus:** move from scalar micrograd vibes → vector/matrix code you can scale.

### Deliverables

- A robust `Linear` layer (MatMul + bias) with correct backward.
- A “shape contract” doc: each op states input/output shapes + backward shapes.

### Sessions

**Session 1**

- Re-derive gradients for:
  - `Y = XW + b` (matrix calculus, but keep it practical)
- Tie back to VJP: “upstream grad” times local Jacobian (implicitly).

**Session 2**

- Implement `matmul`, `sum(axis=...)`, broadcasting add, and their backwards.
- Add tests that specifically catch shape bugs (the silent killers).

**Session 3**

- Read: one short piece on Jacobians vs VJPs (or re-read your own notes).
- Summarize: when Jacobian is useful conceptually vs computationally.

**Session 4**

- Write a one-page cheat sheet:
  - “Gradients are covectors” → “we represent them as arrays using an inner product” (your language).

**Optional Session 5**

- Short note: “Why engineering uses conventions (row/column) and math cares about maps.”

**Checkpoint**

- You can run a 2-layer MLP with batched inputs without shape explosions.

---

# Week 7 — Optimization Mechanics: SGD, Momentum, (Optional) Adam + Why They Work

**Focus:** not just “use Adam,” but “I can explain why momentum helps.”

### Deliverables

- Implement SGD + momentum (Adam optional if time/energy).
- A tiny experiment comparing convergence on the same toy problem.

### Sessions

**Session 1**

- Derive momentum update as an exponential moving average of gradients.
- Intuition: smoothing noise + accelerating along consistent descent directions.

**Session 2**

- Code: optimizer abstraction + SGD + momentum.
- Add a simple logger: loss per step, maybe gradient norm.

**Session 3**

- Read: short optimization notes (or revisit your existing quarterly plan section).
- Extract 3 rules-of-thumb you actually believe (with reasons).

**Session 4**

- Write: “Optimization as a dynamical system” (tiny, clear, practical).

**Optional Session 5**

- Philosophy thread: “implicit bias” + why optimization choices affect solutions.

**Checkpoint**

- Same model trains noticeably better with momentum than plain SGD on at least one toy task.

---

# Week 8 — Trustworthy Training Loop: Metrics, Overfitting, Regularization (Minimal)

**Focus:** training isn’t “loss goes down,” it’s “generalization + diagnostics.”

### Deliverables

- Train/val split + accuracy (or relevant metric).
- One regularization tool: L2 weight decay (simple, effective).

### Sessions

**Session 1**

- Derive L2 regularization gradient: add `λ ||W||^2` → gradient adds `2λW`.

**Session 2**

- Implement weight decay option in optimizer or loss.
- Add train/val evaluation loop.

**Session 3**

- Read: a short section from :contentReference[oaicite:1]{index=1} on regularization/overfitting (skim is fine).

**Session 4**

- Write: “How I know if training is lying to me” (overfitting signals, data leakage, metric mismatch).

**Optional Session 5**

- Non-technical reflection: “Why ‘measurement’ is the soul of science.”

**Checkpoint**

- You can intentionally overfit a tiny dataset and recognize it immediately.

---

# Week 9 — Exponential Family Unification: Bernoulli & Categorical → Softmax Regression

**Focus:** connect likelihood, log-partition, and the losses you implement.

### Deliverables

- A clean derivation note:
  - exponential family form
  - log-partition `A(η)` and why its gradient gives expectations
- Implement softmax regression (multiclass logistic regression) using your stable CE-from-logits.

### Sessions

**Session 1**

- Re-derive: categorical likelihood → softmax + cross-entropy.
- Emphasize the “normalization constant” role (partition function vibe).

**Session 2**

- Code: softmax regression trainer + evaluation on a toy multiclass dataset.

**Session 3**

- Read: revisit your own exponential family notes; patch the gaps (this is where precision matters).

**Session 4**

- Write: “Loss functions as negative log-likelihood” (with Bernoulli + categorical examples).

**Optional Session 5**

- Short philosophy/history note: “What is a ‘model class’ and what does MLE really claim?”

**Checkpoint**

- Your multiclass model trains and gradients pass checks on small randomized inputs.

---

# Week 10 — CNN Preview: Convolution as a Linear Operator (Tiny, Correct, Tested)

**Focus:** introduce convolution without turning Q1 into a vision quarter.

### Deliverables

- A minimal 1D (or tiny 2D) convolution forward pass.
- Gradient check for conv weights and inputs (even if slow + loop-based).

### Sessions

**Session 1**

- Understand conv as:
  - sliding dot-products
  - weight sharing
  - (conceptually) multiplication by a sparse Toeplitz-like matrix

**Session 2**

- Code a simple conv (loops are fine).
- Backward: either derive it explicitly or implement via your existing ops and verify with gradient check.

**Session 3**

- Read: a short CNN primer (no need to go deep yet).
- Write down 3 “CNN inductive bias” bullets you can explain (locality, translation equivariance, parameter sharing).

**Session 4**

- Write: “Convolution in plain linear algebra language.”

**Optional Session 5**

- Robotics connection note: why locality/translation equivariance matches sensors and perception.

**Checkpoint**

- Conv gradients numerically check out on a tiny random input.

---

# Week 11 — Tiny CNN Toy Task + “Perception → Decision” Sketch

**Focus:** one miniature perception experiment + a robotics-style pipeline sketch.

### Deliverables

- Train a tiny CNN on a tiny dataset (even synthetic shapes).
- A one-page pipeline diagram: sensor → model → confidence → downstream action.

### Sessions

**Session 1**

- Minimal theory: pooling vs striding, and what invariance really means (and what it doesn’t).

**Session 2**

- Train the tiny CNN (don’t chase accuracy; chase understanding + debugging).

**Session 3**

- Read: short section on evaluation in vision (confusion matrix, common failure patterns).

**Session 4**

- Write: “What my CNN gets wrong and why” + pipeline sketch.

**Optional Session 5**

- Robotics glue: latency + why batching isn’t always your friend in real-time loops.

**Checkpoint**

- You can point to concrete failure cases and propose a fix (data, architecture, regularization).

---

# Week 12 — RL Grammar Preview: Bandits → MDP Intuition + Reward Gotchas

**Focus:** introduce RL _conceptually and minimally_ so Q3 doesn’t feel alien.

### Deliverables

- Implement a multi-armed bandit (ε-greedy) + plot/print average reward over time.
- A note: “Reward is not the task” (reward hacking / misalignment in miniature).

### Sessions

**Session 1**

- Define: policy, value, return, exploration vs exploitation.
- Bandit math: expected value and why exploration matters.

**Session 2**

- Code: bandit + ε-greedy; optionally compare to UCB.

**Session 3**

- Read: intro chapter snippets from :contentReference[oaicite:2]{index=2} (just enough to get the vocabulary clean).

**Session 4**

- Write: “How RL differs from supervised learning” (data distribution, feedback loops, non-stationarity).

**Optional Session 5**

- Robotics connection: sense → decide → act loop + why delay/partial observability matters.

**Checkpoint**

- You can explain (in your own words) why “offline supervised training” doesn’t directly solve control.

---

# Week 13 — Q1 Wrap: Polish, Document, Teach (Lock the foundation)

**Focus:** turn the quarter into a coherent artifact you’ll reuse in Q2–Q4.

### Deliverables

- “Micrograd+” repo cleanup:
  - stable CE-from-logits
  - batching
  - gradient check harness
  - tests + README
- Write-up: **“Backprop explained to my future self”** (include δ notation translation)
- A Q2 launchpad note: “What I’m carrying forward into attention + representations.”

### Sessions

**Session 1**

- Re-derive one full path cleanly (end-to-end): logits → loss → parameter gradients.
- Make it readable, not just correct.

**Session 2**

- Refactor code + add 3–5 meaningful tests (not just “it runs”).

**Session 3**

- Read: your own notes. Seriously. Patch weak spots, remove contradictions.

**Session 4**

- Write/teach: final explainer + diagrams (computational graph, VJP flow, δ notation mapping).

**Optional Session 5**

- Non-technical synthesis note: “What changed in how I think about ‘knowing’ something?”

**Checkpoint**

- You can hand a friend your repo + doc and they can run it and understand the core ideas.

---

## Q1 Success Criteria (end-of-quarter)

By the end of Week 13, you should be able to:

- Implement and _trust_ CE-from-logits (stable), batching, and backprop in your engine
- Explain backprop in both “computational graph/VJP” language and Nielsen’s $\delta$ notation
- Debug shape and stability issues systematically
- Have light-but-real footholds in:
  - convolution (with gradient checks)
  - RL vocabulary + a tiny bandit experiment
  - robotics loop framing (sense → decide → act)
