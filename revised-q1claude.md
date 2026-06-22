# Q1 Revised Plan: Weeks 5–14 (claude)

> Building from Weeks 2–4 foundation (backprop, VJP, gradients as covectors)

---

## Overview

| Phase                    | Weeks | Theme                                 | Explainer Version |
| ------------------------ | ----- | ------------------------------------- | ----------------- |
| Foundations (existing)   | 2–4   | Backprop, VJP, gradients as covectors | v0.1 → v0.3       |
| Optimization as Geometry | 5–7   | GD, momentum, Adam as metric choices  | v0.4 → v0.6       |
| Generalization           | 8–10  | Loss landscapes, implicit bias, noise | v0.7 → v0.9       |
| Architecture Motivations | 11–12 | CNNs, RNNs, the bottleneck            | v0.10 → v0.11     |
| Attention Warm-Up        | 13–14 | Soft addressing, Q1 synthesis         | v0.12 → v1.0      |

---

# Phase 2: Optimization as Geometry (Weeks 5–7)

**Purpose:** Apply your covector/level-set intuition to understand _why_ optimizers differ—not just _how_.

---

## Week 5: Gradient Descent as Geometric Flow

### Week 5 outcome targets (ship by week's end)

- **Conceptual mastery:** GD as following the gradient flow on a manifold; why step size is constrained by curvature.
- **Math:** Derive GD from Taylor expansion; connect to your level-set picture.
- **Code:** Visualize GD trajectories on 2D surfaces; see where geometry predicts behavior.
- **Teaching artifact:** Explainer section: **"GD as steepest descent under the Euclidean metric"**

---

### Session 1 (60–90m) — Math: GD from first principles

**Goal:** Make GD feel like a _choice_, not the only option.

#### Work (55–75m)

1. **Taylor expansion view**
   - $f(x + \Delta x) \approx f(x) + \nabla f(x)^\top \Delta x + \frac{1}{2} \Delta x^\top H \Delta x$
   - What does "steepest descent" mean? Steepest _with respect to what norm_?

2. **The constrained optimization view**
   - GD solves: $\min_{\|\Delta x\| \leq \epsilon} f(x) + \nabla f(x)^\top \Delta x$
   - Solution: $\Delta x \propto -\nabla f(x)$ (under Euclidean norm)

3. **Connect to Week 4**
   - The gradient is a covector; the norm comes from the inner product
   - Different inner product → different "steepest" direction

4. **Learning rate as trust region**
   - Why does LR depend on curvature? When does the linear approximation break?

#### Output (last 10m)

Create note: **Week5_S1_GD_Geometry**

- Derivation of GD as constrained optimization
- One paragraph: "What changes if we use a different metric?"

---

### Session 2 (60–90m) — Code: GD trajectories on loss surfaces

**Goal:** See the geometry you derived.

#### Work (60–75m)

1. **Build 2D test surfaces**
   - Quadratic bowl: $f(x,y) = x^2 + y^2$
   - Ravine: $f(x,y) = x^2 + 10y^2$ (or $50y^2$)
   - Saddle: $f(x,y) = x^2 - y^2$
   - Rosenbrock (optional): $f(x,y) = (1-x)^2 + 100(y-x^2)^2$

2. **Implement vanilla GD** (if not already in your framework)

3. **Visualize**
   - Contour plots with trajectory overlay
   - Vary LR: find divergence threshold, slow convergence, sweet spot

4. **Observe**
   - On the ravine: GD oscillates perpendicular to the valley. Why? (Hint: eigenvalues of Hessian)

#### Output (last 10m)

Create note: **Week5_S2_GD_Visualization**

- 3 surface plots with trajectories
- "The ravine problem" — why condition number matters

---

### Session 3 (60–90m) — Read: curvature, condition number, and preconditioning

**Goal:** Understand why "just use a smaller LR" isn't the answer.

#### Reading focus

- Condition number of Hessian
- Why ill-conditioned problems make GD slow
- Preconditioning as changing the metric

#### Active reading workflow

1. Before: Write what you _think_ condition number means for optimization.
2. During: Find where the Hessian eigenvalues enter the convergence rate.
3. After: Rewrite the key bound in your notation.

#### Output (last 10m)

Create note: **Week5_S3_Read_Conditioning**

- The convergence rate formula (how LR and condition number interact)
- 5 questions for Week 6 (e.g., "What if we adapted the metric locally?")

---

### Session 4 (60–90m) — Write: Explainer v0.4

**Goal:** Connect GD to your gradient geometry from Week 4.

#### Add sections

1. **GD as steepest descent** — under what metric?
2. **Why learning rate is bounded** — curvature and trust regions
3. **The ravine problem** — with your visualization

#### Output

**Backprop_Explainer_v0.4**

---

### Optional Session 5 — Non-technical: optimization in nature/engineering

- Gradient flows in physics (heat equation, diffusion)
- Or: coordinate systems in physics (why Lagrangian mechanics uses generalized coordinates)

---

### Week 5 "done" checklist

- [ ] I can derive GD as constrained optimization under Euclidean norm.
- [ ] I can explain why learning rate depends on curvature.
- [ ] I visualized GD trajectories and observed the ravine problem.
- [ ] My explainer now includes the geometric view of GD.

---

## Week 6: Momentum as Geometry

### Week 6 outcome targets (ship by week's end)

- **Conceptual mastery:** Momentum as averaging gradients over time; why it helps with ravines.
- **Math:** Derive momentum; connect to the "heavy ball" ODE.
- **Code:** Add momentum to your optimizer; compare trajectories.
- **Teaching artifact:** Section: **"Momentum smooths the gradient field"**

---

### Session 1 (60–90m) — Math: momentum derivation

**Goal:** See momentum as a _principled_ fix, not a hack.

#### Work (55–75m)

1. **The problem momentum solves**
   - Review: GD oscillates on ravines because gradient points across the valley
   - What if we averaged recent gradients? The across-valley components cancel.

2. **Momentum update**
   - $v_{t+1} = \beta v_t + \nabla f(x_t)$
   - $x_{t+1} = x_t - \alpha v_{t+1}$
   - What does $\beta$ control? (exponential moving average window)

3. **Heavy ball interpretation**
   - View as discretized ODE: $\ddot{x} + \gamma \dot{x} + \nabla f(x) = 0$
   - Physical intuition: ball rolling with friction

4. **Nesterov's twist**
   - Evaluate gradient at "lookahead" position
   - Why this helps: gradient is more accurate for where you'll end up

#### Output (last 10m)

Create note: **Week6_S1_Momentum_Math**

- Momentum update derived from averaging
- One paragraph: "Momentum as a low-pass filter on gradients"

---

### Session 2 (60–90m) — Code: momentum on your test surfaces

**Goal:** See momentum fix the ravine problem.

#### Work (60–75m)

1. Implement momentum (and optionally Nesterov)
2. Run on same surfaces from Week 5
3. Compare trajectories: GD vs momentum vs Nesterov
4. Experiment: vary $\beta$. What happens at $\beta = 0$? At $\beta = 0.99$?

#### Output (last 10m)

Create note: **Week6_S2_Momentum_Code**

- Side-by-side trajectory plots
- "When momentum hurts" (if you find cases)

---

### Session 3 (60–90m) — Read: momentum, acceleration, and the geometry of optimization

**Goal:** Deepen the connection between discrete updates and continuous dynamics.

#### Reading focus (pick one)

- Sutskever's thesis on momentum
- Or: "Why Momentum Really Works" (Goh, distill.pub-style)
- Or: Nesterov's acceleration in convex optimization

#### Output (last 10m)

Create note: **Week6_S3_Read_Momentum**

- Key insight in your words
- Connection to Week 5's curvature story

---

### Session 4 (60–90m) — Write: Explainer v0.5

#### Add sections

1. **Why momentum works** — averaging cancels oscillations
2. **Momentum as dynamics** — the heavy ball picture
3. **Nesterov's lookahead** — one paragraph

#### Output

**Backprop_Explainer_v0.5**

---

### Week 6 "done" checklist

- [ ] I can derive momentum from the averaging perspective.
- [ ] I can explain the heavy ball ODE interpretation.
- [ ] I implemented momentum and saw it fix the ravine problem.
- [ ] My explainer now includes momentum and Nesterov.

---

## Week 7: Adaptive Methods and Metric Learning

### Week 7 outcome targets (ship by week's end)

- **Conceptual mastery:** Adam as approximate diagonal preconditioning; why per-parameter LR helps.
- **Math:** Derive RMSprop and Adam; connect to your Week 4 metric story.
- **Code:** Implement Adam; test on your surfaces + a small net.
- **Teaching artifact:** Section: **"Adaptive methods as learned metrics"**

---

### Session 1 (60–90m) — Math: from preconditioning to Adam

**Goal:** See Adam as GD under a _learned_ diagonal metric.

#### Work (55–75m)

1. **The ideal: Newton's method**
   - Update: $x_{t+1} = x_t - H^{-1} \nabla f(x_t)$
   - This is GD under the metric defined by $H$
   - Problem: computing $H^{-1}$ is expensive

2. **Diagonal approximation**
   - What if we just scaled each coordinate by its "typical gradient magnitude"?
   - RMSprop: $v_t = \beta v_{t-1} + (1-\beta) g_t^2$, update $\propto g_t / \sqrt{v_t}$

3. **Adam = RMSprop + momentum**
   - First moment (momentum): $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
   - Second moment (RMSprop): $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
   - Bias correction: $\hat{m}_t = m_t / (1 - \beta_1^t)$, $\hat{v}_t = v_t / (1 - \beta_2^t)$
   - Update: $x_{t+1} = x_t - \alpha \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$

4. **Connect to Week 4**
   - Dividing by $\sqrt{v_t}$ is like changing the inner product per-coordinate
   - Adam implicitly learns a diagonal metric

#### Output (last 10m)

Create note: **Week7_S1_Adam_Derivation**

- Adam update with bias correction, fully derived
- "Adam as diagonal preconditioning" — one paragraph

---

### Session 2 (60–90m) — Code: implement Adam, test thoroughly

#### Work (60–75m)

1. Implement Adam from scratch (with bias correction)
2. Test on 2D surfaces — compare to GD and momentum
3. Test on a small neural net (your XOR net, or slightly larger)
4. Experiment: what happens without bias correction? Without the $\epsilon$ in denominator?

#### Output (last 10m)

Create note: **Week7_S2_Adam_Code**

- Trajectory comparisons
- "Where Adam shines vs. struggles"

---

### Session 3 (60–90m) — Read: Adam's pathologies and fixes

**Goal:** Understand when the diagonal approximation fails.

#### Reading focus (pick one)

- Reddi et al., "On the Convergence of Adam" (non-convergence examples)
- Or: AdamW paper ("Decoupled Weight Decay Regularization")
- Or: comparison of Adam variants

#### Output (last 10m)

Create note: **Week7_S3_Read_Adam_Pathologies**

- One concrete failure case
- Why AdamW exists

---

### Session 4 (60–90m) — Write: Explainer v0.6 + Optimizer Cheatsheet

#### Add sections

1. **The preconditioning story** — from Newton to Adam
2. **When to use what** — GD (baseline), momentum (ravines), Adam (sparse gradients, varied scales)

#### Output

- **Backprop_Explainer_v0.6**
- **Optimizer_Cheatsheet** (1 page)

---

### Week 7 "done" checklist

- [ ] I can derive Adam from the preconditioning perspective.
- [ ] I can explain why bias correction matters.
- [ ] I implemented Adam and tested on surfaces + neural net.
- [ ] I have an optimizer cheatsheet I trust.

---

# Phase 3: Loss Landscapes & Generalization (Weeks 8–10)

**Purpose:** Move from "does it converge?" to "what kind of solution does it find?"

---

## Week 8: Loss Landscape Geometry

### Week 8 outcome targets (ship by week's end)

- **Conceptual mastery:** What "sharp" vs "flat" minima mean; why we care.
- **Math:** Hessian eigenvalues at minima; connection to generalization claims.
- **Code:** Visualize loss landscapes for a small trained net.
- **Teaching artifact:** Section: **"The geometry of solutions"**

---

### Session 1 (60–90m) — Math: characterizing minima

#### Work (55–75m)

1. **Local geometry at a minimum**
   - Taylor expansion: $f(x^* + \delta) \approx f(x^*) + \frac{1}{2} \delta^\top H \delta$
   - Hessian eigenvalues determine "sharpness"

2. **Sharp vs flat**
   - Sharp: large eigenvalues, small perturbations increase loss quickly
   - Flat: small eigenvalues, robust to perturbations

3. **The generalization hypothesis**
   - Claim: flat minima generalize better (robust to parameter perturbations ≈ robust to data perturbations?)
   - Caveats: this is contested; depends on parameterization

#### Output (last 10m)

Create note: **Week8_S1_Minima_Geometry**

- Hessian eigenvalue interpretation
- "Sharp vs flat" with caveats

---

### Session 2 (60–90m) — Code: visualize loss landscapes

#### Work (60–75m)

1. Train a small net (e.g., 2-layer MLP on a small dataset)
2. Implement 1D or 2D loss landscape visualization:
   - Pick two directions (random, or PCA of training trajectory)
   - Evaluate loss along those directions
   - Plot as 1D curve or 2D contour
3. Compare: SGD-trained vs Adam-trained (if time)

#### Output (last 10m)

Create note: **Week8_S2_LossLandscape_Viz**

- Loss landscape plots
- Observations about sharpness

---

### Session 3 (60–90m) — Read: Li et al. "Visualizing the Loss Landscape"

**Focus:** Extract the methodology and key claims.

#### Output (last 10m)

Create note: **Week8_S3_Read_Landscape**

- How they choose directions (filter normalization)
- Key figure interpretations

---

### Session 4 (60–90m) — Write: Explainer v0.7

#### Add section

"The geometry of solutions" — sharp vs flat, with your visualizations

#### Output

**Backprop_Explainer_v0.7**

---

### Week 8 "done" checklist

- [ ] I can explain what Hessian eigenvalues tell us about a minimum.
- [ ] I visualized loss landscapes for a trained net.
- [ ] I understand the flat minima hypothesis and its caveats.

---

## Week 9: Implicit Bias of Gradient Descent

### Week 9 outcome targets (ship by week's end)

- **Conceptual mastery:** GD finds specific solutions among many possible ones; this is "free regularization."
- **Math:** GD on underdetermined linear regression finds minimum-norm solution.
- **Code:** Verify implicit bias experimentally.
- **Teaching artifact:** Section: **"What GD gives you for free"**

---

### Session 1 (60–90m) — Math: minimum norm solutions

#### Work (55–75m)

1. **Setup:** Linear regression with $p > n$ (more parameters than data points)
   - Infinitely many solutions with zero training loss
   - Which one does GD find?

2. **Derivation:** GD initialized at zero finds minimum $\ell_2$ norm solution
   - Key insight: gradients live in the row space of $X$
   - If you start at zero, you stay in the row space
   - Final solution: $w^* = X^\top (XX^\top)^{-1} y$ (pseudoinverse)

3. **Implication:** The optimizer is implicitly regularizing
   - You didn't ask for small weights, but you got them

#### Output (last 10m)

Create note: **Week9_S1_ImplicitBias_Math**

- Derivation of minimum-norm solution
- "Implicit regularization" intuition

---

### Session 2 (60–90m) — Code: verify implicit bias

#### Work (60–75m)

1. Create underdetermined linear regression problem
   - $n = 10$ data points, $p = 100$ parameters
2. Run GD from zero initialization until convergence
3. Compare to closed-form minimum-norm solution (pseudoinverse)
4. Try different initializations—does the solution change?

#### Output (last 10m)

Create note: **Week9_S2_ImplicitBias_Code**

- Comparison plots
- "What changes with different init"

---

### Session 3 (60–90m) — Read: implicit bias in neural nets

#### Reading focus (pick one)

- Implicit bias in logistic regression (max-margin)
- Or: implicit bias in deep linear networks
- Or: Gunasekar et al. on implicit regularization

#### Output (last 10m)

Create note: **Week9_S3_Read_ImplicitBias**

- Key claims about neural net implicit bias
- Open questions

---

### Session 4 (60–90m) — Write: Explainer v0.8

#### Add section

"What GD gives you for free" — implicit bias story

#### Output

**Backprop_Explainer_v0.8**

---

### Week 9 "done" checklist

- [ ] I can derive why GD finds minimum-norm solutions in linear regression.
- [ ] I verified this experimentally.
- [ ] I understand how this extends (partially) to neural nets.

---

## Week 10: Batch Size, Noise, and Generalization

### Week 10 outcome targets (ship by week's end)

- **Conceptual mastery:** SGD noise as implicit regularization; batch size tradeoffs.
- **Math:** Gradient variance as a function of batch size.
- **Code:** Experiments varying batch size.
- **Teaching artifact:** Section: **"The role of noise in optimization"**

---

### Session 1 (60–90m) — Math: SGD as noisy GD

#### Work (55–75m)

1. **Stochastic gradient**
   - $\hat{g} = \frac{1}{B} \sum_{i \in \text{batch}} \nabla f_i(x)$
   - $\mathbb{E}[\hat{g}] = \nabla f(x)$, but $\text{Var}(\hat{g}) \propto \sigma^2 / B$

2. **Noise as regularizer**
   - Small batches → more noise → harder to stay in sharp minima
   - Connection to flat minima hypothesis

3. **Linear scaling rule**
   - Claim: if you increase batch size by $k$, increase LR by $k$
   - Intuition: keep the "noise scale" constant
   - When this works, when it breaks

#### Output (last 10m)

Create note: **Week10_S1_SGD_Noise**

- Variance formula
- Linear scaling rule with caveats

---

### Session 2 (60–90m) — Code: batch size experiments

#### Work (60–75m)

1. Train same net with batch sizes: 1, 16, 64, 256, full batch
2. Track: training loss, test loss, gradient variance (sample it)
3. Test linear scaling rule: does 4× batch with 4× LR match original?

#### Output (last 10m)

Create note: **Week10_S2_BatchSize_Code**

- Loss curves for different batch sizes
- Linear scaling rule results

---

### Session 3 (60–90m) — Read: large-batch training

#### Reading focus

- Goyal et al. "Accurate, Large Minibatch SGD"
- Or: "Don't Decay the Learning Rate, Increase the Batch Size"

#### Output (last 10m)

Create note: **Week10_S3_Read_LargeBatch**

- Practical recipes for large-batch training
- Warmup schedules

---

### Session 4 (60–90m) — Write: Explainer v0.9

#### Add section

"The role of noise in optimization" — batch size, variance, generalization

#### Output

**Backprop_Explainer_v0.9**

---

### Week 10 "done" checklist

- [ ] I can explain why small batches might help generalization.
- [ ] I ran batch size experiments and observed the effects.
- [ ] I understand the linear scaling rule and its limits.

---

# Phase 4: Architecture Motivations (Weeks 11–12)

**Purpose:** Understand _why_ specific architectures exist—what problems they solve. Bridge to Q2.

---

## Week 11: From MLPs to Structured Architectures

### Week 11 outcome targets (ship by week's end)

- **Conceptual mastery:** Why MLPs fail on images/sequences; what inductive biases fix this.
- **Math:** Parameter counting; translation equivariance.
- **Code:** Implement minimal conv layer (forward pass).
- **Teaching artifact:** Section: **"Why structure matters"**

---

### Session 1 (60–90m) — Math: the curse of unstructured models

#### Work (55–75m)

1. **MLPs on images**
   - 224×224×3 input → first layer has 150K+ parameters per hidden unit
   - No built-in translation invariance: cat in corner ≠ cat in center

2. **Convolutions as constrained MLPs**
   - Weight sharing: same filter everywhere
   - Locality: each output depends on small patch
   - Derive: how many parameters in a 3×3 conv with 64 filters?

3. **Equivariance**
   - Definition: if input shifts, output shifts the same way
   - $f(T(x)) = T(f(x))$ for translation $T$
   - Why this is a good inductive bias for images

#### Output (last 10m)

Create note: **Week11_S1_Structure_Math**

- Parameter count comparison: MLP vs conv
- Equivariance definition and intuition

---

### Session 2 (60–90m) — Code: minimal conv

#### Work (60–75m)

1. Implement 2D convolution forward pass (no libraries)
   - Nested loops are fine; understand the indexing
2. Verify against a library implementation (e.g., `np.convolve` or PyTorch)
3. (Optional) Implement max pooling

#### Output (last 10m)

Create note: **Week11_S2_Conv_Code**

- Your conv implementation
- "What I learned from the indexing"

---

### Session 3 (60–90m) — Read: inductive biases in deep learning

#### Reading focus

- Battaglia et al. "Relational inductive biases, deep learning, and graph networks"
- Or: a good survey on CNN design principles

#### Output (last 10m)

Create note: **Week11_S3_Read_InductiveBias**

- Types of inductive biases (locality, weight sharing, etc.)
- How architecture encodes assumptions

---

### Session 4 (60–90m) — Write: Explainer v0.10

#### Add section

"Why structure matters" — from MLPs to CNNs

#### Output

**Backprop_Explainer_v0.10**

---

### Week 11 "done" checklist

- [ ] I can explain why MLPs are inefficient for images.
- [ ] I implemented convolution from scratch.
- [ ] I understand equivariance and its value.

---

## Week 12: Sequences, Recurrence, and the Bottleneck Problem

### Week 12 outcome targets (ship by week's end)

- **Conceptual mastery:** Why sequences are hard; what RNNs do; where they fail.
- **Math:** RNN forward pass; backprop through time; vanishing gradients.
- **Code:** Implement minimal RNN.
- **Teaching artifact:** Section: **"The seq2seq bottleneck"** — setup for attention.

---

### Session 1 (60–90m) — Math: RNNs and BPTT

#### Work (55–75m)

1. **Why sequences are different**
   - Variable length; order matters
   - MLPs can't handle "the cat sat" vs "sat the cat"

2. **RNN mechanics**
   - $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$
   - Hidden state carries information forward
   - Parameters shared across time

3. **Backprop through time**
   - Unroll the computation graph
   - Derive gradient for 3 timesteps on paper
   - $\frac{\partial L}{\partial W_h}$ involves sum over all timesteps

4. **Vanishing/exploding gradients**
   - Gradient involves products: $\prod_t \frac{\partial h_t}{\partial h_{t-1}}$
   - If spectral radius < 1: vanishing. If > 1: exploding.
   - Why long-range dependencies are hard

#### Output (last 10m)

Create note: **Week12_S1_RNN_Math**

- RNN equations
- BPTT derivation for 3 steps
- Vanishing gradient explanation

---

### Session 2 (60–90m) — Code: minimal RNN

#### Work (60–75m)

1. Implement vanilla RNN forward pass
2. (Optional) Implement BPTT manually for a few steps
3. Train on tiny sequence task:
   - Echo task: output = input delayed by k steps
   - Or: character-level next-char prediction on tiny corpus

#### Output (last 10m)

Create note: **Week12_S2_RNN_Code**

- Your RNN implementation
- Training curves on toy task

---

### Session 3 (60–90m) — Read: seq2seq and the bottleneck

#### Reading focus

- Bahdanau et al. (2014) "Neural Machine Translation by Jointly Learning to Align and Translate" — sections 1–3
- Focus on: What problem does attention solve?

#### Key questions to answer

- Why is encoding a whole sentence into one vector bad?
- What information is lost?
- How does attention fix this?

#### Output (last 10m)

Create note: **Week12_S3_Read_Seq2Seq**

- "The problem attention was invented to solve"
- Diagram: encoder-decoder with bottleneck

---

### Session 4 (60–90m) — Write: Explainer v0.11

#### Add section

"From MLPs to RNNs to... what's missing?" — the seq2seq bottleneck, setup for Q2

#### Output

**Backprop_Explainer_v0.11**

---

### Week 12 "done" checklist

- [ ] I can derive BPTT for a short sequence.
- [ ] I understand why vanishing gradients happen in RNNs.
- [ ] I implemented a minimal RNN.
- [ ] I can explain the seq2seq bottleneck problem.

---

# Phase 5: Attention Warm-Up (Weeks 13–14)

**Purpose:** Arrive at Q2 with attention intuition in place.

---

## Week 13: Attention as Soft Addressing

### Week 13 outcome targets (ship by week's end)

- **Conceptual mastery:** Attention as learnable, content-based retrieval.
- **Math:** Bahdanau attention; dot-product attention; scaling.
- **Code:** Implement scaled dot-product attention.
- **Teaching artifact:** Section: **"Attention as soft lookup"**

---

### Session 1 (60–90m) — Math: attention mechanisms

#### Work (55–75m)

1. **The key insight**
   - Instead of one summary vector, compute weighted combination of all encoder states
   - Weights depend on _what you're looking for_ (query) and _what's available_ (keys)

2. **Bahdanau (additive) attention**
   - Alignment score: $e_{ij} = v^\top \tanh(W_q q_i + W_k k_j)$
   - Attention weights: $\alpha_{ij} = \text{softmax}_j(e_{ij})$
   - Context vector: $c_i = \sum_j \alpha_{ij} v_j$

3. **Dot-product attention**
   - Alignment score: $e_{ij} = q_i^\top k_j$
   - Simpler, faster (matrix multiplication)
   - Scaling: divide by $\sqrt{d_k}$ to prevent softmax saturation

4. **Connect to your geometry**
   - Dot product measures similarity in embedding space
   - Softmax gives a probability distribution over positions
   - Attention is differentiable "lookup"

#### Output (last 10m)

Create note: **Week13_S1_Attention_Math**

- Both attention formulas
- "Why scale by $\sqrt{d_k}$"

---

### Session 2 (60–90m) — Code: implement attention

#### Work (60–75m)

1. Implement scaled dot-product attention from scratch

   ```
   def attention(Q, K, V):
       scores = Q @ K.T / sqrt(d_k)
       weights = softmax(scores)
       return weights @ V
   ```

2. Test on synthetic tasks:
   - Copy task: can attention learn to copy?
   - Selective copy: copy only certain positions
   - Sort by value: attend to positions in order of their values

3. Visualize attention weights as heatmaps

#### Output (last 10m)

Create note: **Week13_S2_Attention_Code**

- Your attention implementation
- Attention weight visualizations
- "What attention learned on toy tasks"

---

### Session 3 (60–90m) — Read: "Attention Is All You Need" (sections 1–3)

#### Focus

- Just the attention mechanism, not the full Transformer yet
- Multi-head attention: why multiple heads?
- Q/K/V projections: why project before attention?

#### Output (last 10m)

Create note: **Week13_S3_Read_AIAYN**

- Q/K/V interpretation
- Multi-head attention intuition
- Questions for Q2 (e.g., "What does each head learn?")

---

### Session 4 (60–90m) — Write: Explainer v0.12

#### Add section

"Attention explained" — from bottleneck problem to soft lookup

#### Output

**Backprop_Explainer_v0.12**

---

### Week 13 "done" checklist

- [ ] I can derive both additive and dot-product attention.
- [ ] I understand why we scale by $\sqrt{d_k}$.
- [ ] I implemented attention and tested on toy tasks.
- [ ] I have questions ready for Q2 (multi-head, Transformers).

---

## Week 14: Q1 Synthesis

### Week 14 outcome targets (ship by week's end)

- **Review:** Walk through all Q1 work; identify what you own vs. what's shaky.
- **Write:** Complete Q1 synthesis document.
- **Prep:** Set up Q2 Week 1.

---

### Session 1 (60–90m) — Review all Q1 outputs

#### Work

- Re-read your notes from Weeks 2–13
- For each week, mark:
  - ✓ Solid: I can explain this without notes
  - ? Shaky: I get the idea but couldn't reproduce it
  - ✗ Gap: I need to revisit this

#### Output

Create note: **Week14_S1_Q1_Audit**

- List of ✓/? /✗ by topic
- Top 3 things to revisit

---

### Session 2 (60–90m) — Write: "Backprop to Attention" narrative

#### Work

Write a 2–3 page narrative connecting the quarter:

- Start: What is a gradient? (Week 2–4)
- Middle: How do we use gradients? (Week 5–10)
- Bridge: Why do architectures matter? (Week 11–12)
- End: What problem does attention solve? (Week 13)

#### Output

Create note: **Week14_S2_Q1_Narrative**

---

### Session 3 (60–90m) — Polish: Explainer final Q1 version

#### Work

- Consolidate v0.12 into **Backprop_Explainer_v1.0**
- Add introduction: "What this document covers"
- Add conclusion: "What's next (Q2 preview)"
- Create one "best diagram" that captures the whole Q1 story

#### Output

**Backprop_Explainer_v1.0** (final Q1 version)

---

### Session 4 (60–90m) — Q2 setup

#### Work

1. Review Q2 goals (Transformers, representations, multi-head attention)
2. Skim full "Attention Is All You Need" paper
3. Write: "Questions I'm bringing to Q2"
   - What does each attention head learn?
   - Why positional encoding?
   - How do residual connections + layer norm interact with optimization?

#### Output

Create note: **Week14_S4_Q2_Setup**

- Q2 week 1 plan
- Questions to answer

---

### Week 14 "done" checklist

- [ ] I audited all Q1 work and identified gaps.
- [ ] I wrote the Q1 narrative connecting all topics.
- [ ] I have Backprop_Explainer_v1.0 complete.
- [ ] I'm ready for Q2 with clear questions.

---

# Q1 Final Deliverables Summary

| Deliverable                 | Description                                                           |
| --------------------------- | --------------------------------------------------------------------- |
| **Backprop_Explainer_v1.0** | Complete document: gradients → optimizers → architectures → attention |
| **Optimizer_Cheatsheet**    | 1-page reference: when to use GD/momentum/Adam                        |
| **Code Repository**         | Autograd engine, optimizers, conv, RNN, attention                     |
| **Weekly Notes**            | ~40 session notes documenting your learning                           |
| **Q1 Audit**                | Honest assessment of what you own vs. gaps                            |
| **Q2 Setup**                | Questions and plan for Transformer deep-dive                          |

---

# Appendix: Suggested Reading by Week

| Week | Primary Reading                                                           |
| ---- | ------------------------------------------------------------------------- |
| 5    | Boyd & Vandenberghe, Convex Optimization (Ch. 9, unconstrained)           |
| 6    | Sutskever thesis (momentum sections) or Goh's "Why Momentum Really Works" |
| 7    | Reddi et al. "On the Convergence of Adam"                                 |
| 8    | Li et al. "Visualizing the Loss Landscape of Neural Nets"                 |
| 9    | Gunasekar et al. on implicit regularization                               |
| 10   | Goyal et al. "Accurate, Large Minibatch SGD"                              |
| 11   | Battaglia et al. "Relational inductive biases"                            |
| 12   | Bahdanau et al. (2014) sections 1–3                                       |
| 13   | Vaswani et al. "Attention Is All You Need" sections 1–3                   |
