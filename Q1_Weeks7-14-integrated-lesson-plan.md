# Q1 · Weeks 7–14 · Integrated Lesson Plan

## Course Sources

| Abbrev     | Course                                                   | Access                                                                                |
| ---------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **CS4787** | Cornell: Principles of Large-Scale ML (De Sa)            | Lecture PDFs public via Spring 2019 offering: `cs.cornell.edu/courses/cs4787/2019sp/` |
| **CS231n** | Stanford: Deep Learning for Computer Vision              | Notes: `cs231n.github.io` · Assignments: `cs231n.github.io/assignments2024/`          |
| **CS224n** | Stanford: NLP with Deep Learning (Manning)               | 2024 YouTube lectures · Slides + assignments: `web.stanford.edu/class/cs224n/`        |
| **Bottou** | Bottou et al., _Optimization Methods for Large-Scale ML_ | `arxiv.org/pdf/1606.04838v1.pdf` (free)                                               |
| **GBC**    | Goodfellow, Bengio & Courville, _Deep Learning_          | `deeplearningbook.org` (free)                                                         |
| **SB**     | Shalev-Shwartz & Ben-David, _Understanding ML_           | `cs.huji.ac.il/~shais/UnderstandingMachineLearning/` (free)                           |

Existing references (Boyd & Vandenberghe, Shewchuk, Grosse CSC421, Distill.pub) remain active.

---

## Week 7 — SGD + Momentum (ship an optimizer API)

### Outcome targets

- **Conceptual:** Why momentum improves conditioning from $O(\kappa)$ to $O(\sqrt{\kappa})$, and how Nesterov differs from heavy-ball.
- **Code:** `Optimizer` abstraction with `SGD` and `MomentumSGD`; one plot showing momentum clearly beating vanilla GD on a ravine.
- **Teaching artifact:** 1–2 page note in explainer: **"Momentum as EMA / heavy ball"** with convergence comparison plot.

---

### Session 1 (60–90m) — Math: formalize the momentum update

**Goal:** Go from Week 6's EMA intuition to a formal optimizer abstraction with convergence guarantees.

#### Prep (5m)

Pull up your Week 6 notes (w6s3.md). You have the accumulator form, normalized form, and the unrolled recurrence. This session makes it rigorous.

#### Work (55–75m)

1. **Write the full momentum update as a dynamical system**
   - State: $(w^k, z^k)$. Update: $z^{k+1} = \beta z^k + \nabla f(w^k)$, then $w^{k+1} = w^k - \alpha z^{k+1}$
   - Identify where $\alpha$ and $\beta$ enter and what each controls
2. **Nesterov momentum**
   - Write the "lookahead" variant: evaluate gradient at $w^k - \alpha \beta z^k$ instead of $w^k$
   - Geometric intuition: Nesterov corrects the overshoot _before_ the gradient step
3. **Convergence sketch on the 2D quadratic**
   - Using $f(w) = \frac{1}{2} w^T H w$ with $H = \text{diag}(\lambda_{\max}, \lambda_{\min})$
   - Show that momentum's effective convergence rate depends on $\sqrt{\kappa}$ not $\kappa$
   - Reference: CS4787 L7 (2019) derives this — skim the PDF _after_ attempting the derivation yourself

#### Output (last 10m)

Create note: **Week7_S1_Momentum_Math**

- The two-form dynamical system (heavy-ball and Nesterov)
- One paragraph: why $\sqrt{\kappa}$ instead of $\kappa$

---

### Session 2 (60–90m) — Code: implement optimizers + ravine comparison

**Goal:** Ship working `SGD` and `MomentumSGD` optimizers and see the difference on your ravine.

#### Work plan (60–75m)

1. **Define an `Optimizer` abstraction**
   - Interface: `step(parameters)`, `zero_grad()`
   - Each optimizer stores its own state (velocity buffer for momentum)
2. **Implement `SGD` and `MomentumSGD`**
   - Check your pseudocode against Bottou et al. Ch. 7 (p. 41–47 in the arXiv PDF)
3. **Run on your ravine toy** ($f(x,y) = x^2 + 10y^2$ from `w6s2.py`)
   - Same start point, same learning rate, compare trajectories
   - Try $\beta \in \{0.5, 0.9, 0.99\}$
4. **Plot: loss vs steps** for SGD and Momentum side by side

#### What to observe

- How momentum smooths the zigzag across the ravine
- What happens when $\beta$ is too high (slow turn response) vs. too low (reverts to vanilla GD)
- Whether the momentum trajectory overshoots the minimum and comes back

#### Output (last 10m)

Create note: **Week7_S2_Optimizer_Code**

- Working optimizer code
- One comparison plot with brief annotation

---

### Session 3 (60–90m) — Read: CS4787 momentum lecture + convergence analysis

**Goal:** Get the convergence-rate story from a formal source and extract practical rules.

#### Reading material

- **Primary:** CS4787 Lecture 7 (Spring 2019): "Accelerating SGD with momentum"
  - URL: `cs.cornell.edu/courses/cs4787/2019sp/notes/lecture7.pdf`
  - Covers: condition number recap → 2D quadratic analysis → heavy-ball method → Chebyshev iteration → Nesterov's accelerated method
  - Demo notebook: `cs.cornell.edu/courses/cs4787/2019sp/notes/Notebook7.ipynb`
- **Secondary (already read):** Distill "Why Momentum Really Works" — revisit the eigendecomposition section with fresh eyes after the CS4787 convergence analysis
- **Deep dive (optional):** Bottou et al. Ch. 7 for the full SGD + momentum convergence proof

#### Active reading workflow

1. Before reading: write your current best 3-sentence explanation of "why momentum helps"
2. During reading: note every place the condition number $\kappa$ appears in a convergence bound
3. After reading: update your 3-sentence explanation — did the convergence-rate story change anything?

#### Output (last 10m)

Create note: **Week7_S3_Read_Momentum**

- 3 rules for setting momentum hyperparameters (extracted from the reading)
- 1 connection: how does the Chebyshev iteration view relate to the eigendecomposition view from Distill?

---

### Session 4 (60–90m) — Write: explainer momentum section + plots

**Goal:** Add a momentum section to your backprop explainer that you could teach from.

#### Add to explainer

1. **Momentum as EMA of gradients** — the recurrence, the exponential weighting, what $\beta$ controls
2. **Why it helps on ravines** — geometric picture: oscillating components cancel, consistent components reinforce
3. **The convergence improvement** — $O(\kappa) \to O(\sqrt{\kappa})$, what this means in practice
4. **Nesterov vs. heavy-ball** — one paragraph on the lookahead correction
5. **Include your comparison plot** from S2

#### Output

- Updated doc: **Backprop_Explainer_v0.5** (or next version)

---

### Self-test

Derive the convergence rate improvement from momentum on the 2D quadratic yourself. If you can show why the rate goes from $O(\kappa)$ to $O(\sqrt{\kappa})$, you've internalized the core result.

### Week 7 "done" checklist

- [ ] I can write the heavy-ball and Nesterov updates from memory.
- [ ] I can explain why momentum improves convergence rate on ill-conditioned problems.
- [ ] My optimizer code runs and produces a clear comparison plot.
- [ ] My explainer has a momentum section with geometric + convergence-rate framing.

---

## Week 8 — Train/val + metrics + weight decay

### Outcome targets

- **Conceptual:** L2 regularization as a gradient contribution; what train/val divergence tells you and what it doesn't.
- **Code:** Train/val split, accuracy metric, L2 weight decay toggle.
- **Teaching artifact:** Short "training lies" diagnostic checklist.

---

### Session 1 (60–90m) — Math: L2 regularization as gradient modification

**Goal:** Derive the gradient contribution of L2 regularization and understand weight decay geometrically.

#### Prep (5m)

Read the regularization section of CS231n "Neural Networks Part 2" (`cs231n.github.io/neural-networks-2/`) — skim the L2 and dropout sections (15 min). Then close it and derive from scratch.

#### Work (55–75m)

1. **L2 regularization objective**
   - Write $\tilde{L}(w) = L(w) + \frac{\lambda}{2} \|w\|^2$
   - Take the gradient: $\nabla \tilde{L} = \nabla L + \lambda w$
   - The update becomes $w \leftarrow w(1 - \alpha\lambda) - \alpha \nabla L(w)$ — weight _decay_
2. **Geometric interpretation**
   - L2 adds a spherical penalty centered at the origin
   - The optimal point is pulled toward zero — by how much? Depends on the eigenvalues of the Hessian
   - Connection to your metric story: L2 is equivalent to a Gaussian prior on weights (Bayesian framing)
3. **What weight decay does NOT do**
   - It does not prevent overfitting in all cases
   - It biases toward small-norm solutions — is this always good?
4. **Preview: AdamW vs. Adam + L2** (just flag the distinction; full treatment in Week 9)

#### Output (last 10m)

Create note: **Week8_S1_Regularization_Math**

- Derivation of the weight decay update
- One paragraph on the geometric interpretation

---

### Session 2 (60–90m) — Code: eval loop, metrics, weight decay

**Goal:** Add training infrastructure: train/val split, accuracy, and L2 decay.

#### Work plan (60–75m)

1. **Train/val split**
   - Split your training data (e.g., the 3-class dataset from `w6s2.py`) into train and validation
   - Run training, log both train loss and val loss per epoch
2. **Accuracy metric**
   - Implement classification accuracy on the val set after each epoch
3. **L2 weight decay**
   - Add a `weight_decay` parameter to your `SGD` optimizer
   - Implement as $w \leftarrow w(1 - \alpha\lambda)$ before the gradient step
4. **Experiment:** Train with and without decay, compare train/val curves

#### Stretch (if time): CS231n Assignment 2 — Dropout section

- `cs231n.github.io/assignments2024/assignment2/`
- The dropout forward/backward implementation is clean and well-tested
- Treat this as a bonus, not a requirement for this week

#### Output (last 10m)

Create note: **Week8_S2_TrainVal_Code**

- Train/val loss curves with and without weight decay
- 2 observations about what changed

---

### Session 3 (60–90m) — Read: CS231n training diagnostics

**Goal:** Learn the practical art of diagnosing training pathologies.

#### Reading material

- **Primary:** CS231n "Neural Networks Part 3: Learning and Evaluation"
  - URL: `cs231n.github.io/neural-networks-3/`
  - Covers: gradient checks, sanity checks, babysitting the learning process, loss curve interpretation, hyperparameter search (random vs. grid)
- **Secondary:** CS4787 Lecture 13 (2019): "Early stopping and batch normalization"
  - URL: `cs.cornell.edu/courses/cs4787/2019sp/notes/lecture13.pdf`
  - Adds the theoretical angle on early stopping as implicit regularization

#### Active reading workflow

1. Before reading: list 3 things you currently check when training seems broken
2. During reading: flag every diagnostic that you don't currently use
3. After reading: write a prioritized checklist of "first things to check"

#### Output (last 10m)

Create note: **Week8_S3_Read_TrainingDiagnostics**

- Your prioritized diagnostic checklist (this becomes the deliverable for S4)
- 2 new diagnostics you hadn't thought of before

---

### Session 4 (60–90m) — Write: "training lies" diagnostic note

**Goal:** Produce a one-page reference for what can go wrong during training and how to detect it.

#### Structure

1. **Loss not decreasing** — learning rate too high? Gradients exploding? Bug in loss computation?
2. **Train loss decreasing, val loss not** — overfitting, but is it regularization-solvable or data-solvable?
3. **Both decreasing but slowly** — learning rate too low? Bad initialization? Poor conditioning?
4. **Loss NaN/Inf** — numerical instability, check for log(0), overflow in exp
5. **Accuracy flat despite loss decreasing** — calibration issue, wrong metric, class imbalance

#### Output

- Note: **Week8_Diagnostic_Checklist**

### Week 8 "done" checklist

- [ ] I can derive the weight decay update from L2 regularization.
- [ ] My training loop has train/val split, accuracy logging, and a weight decay toggle.
- [ ] I have a diagnostic checklist I could hand to someone.

---

## Week 9 — Adam

### Outcome targets

- **Conceptual:** Adam as momentum + diagonal preconditioning; why AdamW ≠ Adam + L2; connection back to the metric/inner product story.
- **Code:** Adam with bias correction; 3-way comparison (SGD vs. Momentum vs. Adam).
- **Teaching artifact:** 1-page "when to use what" optimizer cheatsheet.

---

### Session 1 (60–90m) — Math: Adam as diagonal preconditioner

**Goal:** Derive Adam from first principles as momentum + adaptive per-parameter scaling.

#### Prep (5m)

Read CS4787 Lecture 8 (2019): "Preconditioning and adaptive learning rates"

- URL: `cs.cornell.edu/courses/cs4787/2019sp/notes/lecture8.pdf`
- Skim sections on preconditioning and AdaGrad before diving in. This frames Adam as a _metric choice_ — directly continuing your Weeks 4–5 thread.

#### Work (55–75m)

1. **AdaGrad → RMSProp → Adam progression**
   - AdaGrad: accumulate squared gradients, divide step by $\sqrt{G_t + \epsilon}$
   - Problem: AdaGrad's accumulator only grows → learning rate goes to zero
   - RMSProp: use EMA of squared gradients instead (fixes the monotonic decay)
   - Adam: RMSProp + momentum (first moment) with bias correction
2. **Write Adam's full update** (from Kingma & Ba, 2014, Figure 1)
   - $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
   - $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
   - $\hat{m}_t = m_t / (1 - \beta_1^t)$, $\hat{v}_t = v_t / (1 - \beta_2^t)$
   - $w_{t+1} = w_t - \alpha \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$
3. **Why bias correction matters**
   - At $t=1$: $m_1 = (1-\beta_1) g_1$, which is much smaller than $g_1$. The correction compensates.
   - Derive: $E[m_t] = (1-\beta_1^t) E[g_t]$ (assuming stationary gradients)
4. **Adam as a metric choice**
   - The $1/\sqrt{v_t}$ scaling is a _diagonal approximation to the inverse Hessian_
   - This changes the effective inner product per-parameter — exactly the preconditioning story from CS4787 L8
   - Connection: Adam implicitly chooses a different metric than Euclidean GD. Your Week 5 insight that "steepest descent depends on the metric" applies here concretely.

#### Output (last 10m)

Create note: **Week9_S1_Adam_Math**

- Full Adam update with bias correction
- One paragraph: "Adam as diagonal preconditioning"

---

### Session 2 (60–90m) — Code: implement Adam + 3-way comparison

**Goal:** Working Adam implementation; side-by-side comparison with SGD and Momentum.

#### Work plan (60–75m)

1. **Implement Adam** in your optimizer framework
   - Store $m$ and $v$ per parameter, track iteration count for bias correction
   - Verify against the pseudocode in Kingma & Ba (Figure 1)
2. **Run 3-way comparison** on your toy problem
   - SGD vs. Momentum vs. Adam, same task, same initial point
   - Plot: loss vs. steps for all three
3. **Ablations**
   - Bias correction on vs. off — observe the first ~10 steps closely
   - $\epsilon = 10^{-8}$ vs. $\epsilon = 10^{-1}$ — what breaks?
   - Try Adam on the isotropic bowl vs. the ravine — where does it shine vs. overkill?

#### Output (last 10m)

Create note: **Week9_S2_Adam_Code**

- Working Adam implementation
- 3-way comparison plot
- 1–2 sentences per ablation result

---

### Session 3 (60–90m) — Read: AdamW and optimizer failure modes

**Goal:** Understand where Adam fails and why AdamW exists.

#### Reading material

- **Primary:** Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2019)
  - Key insight: Adam + L2 regularization ≠ Adam + weight decay. L2 modifies the gradient before Adam's adaptive scaling, which distorts the regularization. Decoupled weight decay applies the decay _after_ the Adam step.
- **Secondary:** CS4787 Lecture 9 (2019): "Variance reduction and averaging"
  - URL: `cs.cornell.edu/courses/cs4787/2019sp/notes/lecture9.pdf`
  - Nice to have: gives you SVRG and Polyak averaging, which appear in RL policy gradient literature later
- **Optional deep dive:** Wilson et al., "The Marginal Value of Adaptive Gradient Methods in Machine Learning" (2017) — argues SGD+momentum generalizes better than Adam in some regimes

#### Active reading workflow

1. Before reading: predict — should weight decay go _inside_ or _outside_ the Adam update?
2. During reading: trace the math showing where L2 regularization gets rescaled by $1/\sqrt{v_t}$
3. After reading: update your optimizer code to support AdamW as a variant

#### Output (last 10m)

Create note: **Week9_S3_Read_AdamW**

- One paragraph: why L2 + Adam ≠ weight decay + Adam
- Decision criteria: when to use Adam vs. AdamW vs. SGD+momentum

---

### Session 4 (60–90m) — Write: optimizer cheatsheet

**Goal:** Produce a 1-page reference for choosing and tuning optimizers.

#### Structure

1. **SGD** — when to use, typical $\alpha$ ranges, failure modes
2. **SGD + Momentum** — when to prefer over vanilla SGD, $\beta$ guidance, learning rate schedules
3. **Adam** — defaults ($\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$), when it shines, when it struggles
4. **AdamW** — when to use instead of Adam + L2
5. **The metric connection** — all optimizers as different choices of effective inner product
6. **Quick reference table** with default hyperparameters

Cross-reference CS231n's practical advice: Adam as robust default, SGD+Nesterov for best generalization when tuned, learning rate decay common with SGD but less so with Adam.

#### Output

- Note: **Week9_Optimizer_Cheatsheet**

### Self-test

Implement Adam from the original paper's pseudocode (Figure 1) and verify your implementation produces identical iterates to the CS231n notes' Adam code snippet for a toy problem. Then ablate bias correction and epsilon.

### Week 9 "done" checklist

- [ ] I can derive Adam's update and explain each component's role.
- [ ] I can explain why Adam is a diagonal preconditioner and connect it to metric choice.
- [ ] I know when to use AdamW vs. Adam + L2 and why.
- [ ] My comparison plot clearly shows the tradeoffs between SGD, Momentum, and Adam.

---

## Week 10 — Loss landscape slices

### Outcome targets

- **Conceptual:** What the Hessian tells you locally; what sharp/flat minima mean (and don't mean).
- **Code:** 1D and/or 2D parameter-direction loss slice plots.
- **Teaching artifact:** Note on sharp/flat minima with interpretation caveats.

---

### Session 1 (60–90m) — Math: Taylor expansion and the Hessian local picture

**Goal:** Formalize what "loss landscape shape" means at a point.

#### Work (55–75m)

1. **Second-order Taylor expansion**
   - $f(w + \Delta w) \approx f(w) + \nabla f(w)^T \Delta w + \frac{1}{2} \Delta w^T H \Delta w$
   - At a minimum: $\nabla f = 0$, so the shape is governed by $H$
2. **Eigendecomposition of $H$**
   - Large eigenvalues → sharp curvature (narrow valley)
   - Small eigenvalues → flat directions
   - Zero eigenvalues → degenerate directions (symmetries)
3. **What "sharp" and "flat" minima mean**
   - Sharp: large eigenvalues of $H$ → high curvature → small perturbations increase loss quickly
   - Flat: small eigenvalues → loss changes slowly → potentially better generalization (Hochreiter & Schmidhuber, 1997)
   - Caveats: the relationship between flatness and generalization is subtle and contested
4. **Why visualizing helps**
   - In high dimensions you can't see the full landscape, but 1D/2D slices along chosen directions reveal structure

#### References

- Boyd & Vandenberghe Ch. 9 (second-order methods) — you already have this
- GBC Ch. 8.2 (challenges in optimization: ill conditioning, saddle points, local minima)

#### Output (last 10m)

Create note: **Week10_S1_Hessian_Landscape**

---

### Session 2 (60–90m) — Code: loss slice visualization

**Goal:** Implement 1D and 2D loss slice plots.

#### Work plan (60–75m)

1. **1D slice:** Pick a direction $d$ in parameter space (e.g., the gradient direction, or a random direction). Evaluate $f(w^* + \alpha d)$ for a range of $\alpha$ values. Plot loss vs. $\alpha$.
2. **2D slice:** Pick two orthogonal directions $d_1, d_2$. Evaluate $f(w^* + \alpha d_1 + \beta d_2)$ on a grid. Plot as contour or surface.
3. **Filter-normalized directions** (from Li et al.): normalize each direction vector to have the same norm as the corresponding parameter tensor. This prevents large layers from dominating the visualization.
4. **Run on your MLP** — compare: random init vs. trained, different optimizers

#### Output (last 10m)

Create note: **Week10_S2_LandscapeViz_Code**

- 1D slice plot at trained minimum
- 2D contour plot (if time)

---

### Session 3 (60–90m) — Read: landscape visualization + overparameterization

**Goal:** Understand what loss landscape visualizations do and don't tell you.

#### Reading material

- **Primary:** Li et al., "Visualizing the Loss Landscape of Neural Nets" (2018)
  - Key method: filter-normalized random directions for fair comparison across architectures
  - Key finding: skip connections dramatically smooth the loss landscape
- **Secondary:** CS4787 Lecture 11 (2019): "Deep neural networks"
  - URL: `cs.cornell.edu/courses/cs4787/2019sp/notes/lecture11.pdf`
  - Bridges from the convex optimization world to the overparameterized regime where landscapes behave differently

#### Active reading workflow

1. Before reading: predict — does a deeper network have a smoother or rougher loss landscape?
2. During reading: note every caveat about what visualizations can't capture
3. After: list 3 things a 2D slice _cannot_ tell you about a high-dimensional landscape

#### Output (last 10m)

Create note: **Week10_S3_Read_Landscape**

---

### Session 4 (60–90m) — Write: sharp/flat minima interpretation

**Goal:** Write a note that honestly presents what we know and don't know about landscape geometry.

#### Structure

1. What the Hessian tells you at a point
2. Sharp vs. flat: the hypothesis and the evidence
3. Caveats: reparametrization can change flatness without changing the function (Dinh et al., 2017)
4. What your visualizations showed (from S2)

#### Output

- Note: **Week10_Landscape_Interpretation**

### Week 10 "done" checklist

- [ ] I can explain what the Hessian eigenvalues tell you about local geometry.
- [ ] I have working loss slice visualizations.
- [ ] I can state the sharp/flat minima hypothesis _and_ its caveats.

---

## Week 11 — Implicit bias of GD

### Outcome targets

- **Conceptual:** In the underdetermined ($p > n$) setting, GD from zero initialization converges to the minimum-norm solution. This is a _choice_ made by the algorithm, not by the objective.
- **Code:** Underdetermined linear regression experiment showing GD → min-norm.
- **Teaching artifact:** Short explainer: "GD chooses among infinite solutions."

---

### Session 1 (60–90m) — Math: why GD stays in the row space

**Goal:** Prove that GD from $w_0 = 0$ on an underdetermined linear system converges to the min-norm solution.

#### Work (55–75m)

1. **Setup:** $X w = y$ with $p > n$ (more parameters than equations). Infinitely many solutions.
2. **Key observation:** GD updates $w_{k+1} = w_k - \alpha X^T(Xw_k - y)$. Starting from $w_0 = 0$, every update is a linear combination of rows of $X$. So $w_k \in \text{rowspace}(X)$ for all $k$.
3. **The min-norm solution:** The minimum-norm solution $w^* = X^T(XX^T)^{-1}y$ also lies in the row space. Since GD never leaves the row space, it converges to $w^*$.
4. **Why this matters:** The loss function doesn't prefer $w^*$ — every solution achieves zero loss. The algorithm's _dynamics_ select a particular solution. This is "implicit bias."

#### References

- Grosse's CSC421 notes (if they cover this)
- SB Ch. 12–14 for the generalization context
- For deeper theory: Neyshabur et al., "In Search of the Real Inductive Bias" (2014)

#### Output (last 10m)

Create note: **Week11_S1_ImplicitBias_Math**

- The row-space argument (3–5 steps)
- One paragraph: what "implicit bias" means

---

### Session 2 (60–90m) — Code: min-norm experiment

**Goal:** Demonstrate GD → min-norm experimentally.

#### Work plan (60–75m)

1. **Setup:** Generate random $X \in \mathbb{R}^{n \times p}$ with $p \gg n$ (e.g., $n=10$, $p=100$). Random $y$.
2. **Run GD** from $w_0 = 0$ until convergence (loss ≈ 0).
3. **Compare:** $\|w_{GD}\|$ vs. $\|w_{min-norm}\|$ where $w_{min-norm} = X^T(XX^T)^{-1}y$ (compute with numpy).
4. **Vary initialization:** What happens from $w_0 \neq 0$? From random init?
5. **Vary optimizer:** Does momentum GD still converge to min-norm? What about Adam?

#### Output (last 10m)

Create note: **Week11_S2_ImplicitBias_Code**

- Norm comparison table
- 1–2 observations about non-zero init and different optimizers

---

### Session 3 (60–90m) — Read: implicit regularization in deep learning

**Goal:** Understand how implicit bias extends beyond linear models.

#### Reading material (pick one)

- **Option A (accessible):** Neyshabur et al., "In Search of the Real Inductive Bias: On the Role of Implicit Regularization in Deep Learning" (2014)
- **Option B (deeper):** Soudry et al., "The Implicit Bias of Gradient Descent on Separable Data" (2018) — shows GD converges to max-margin (SVM) solution for classification
- **Option C (survey-style):** Bottou et al. Ch. 4–5 on SGD's noise as implicit regularization

#### Active reading workflow

1. Before reading: state your hypothesis — does GD always prefer "simple" solutions?
2. During reading: note every assumption the proofs require (linearity? convexity? initialization?)
3. After: write one question about what happens in non-linear (deep) models

#### Output (last 10m)

Create note: **Week11_S3_Read_ImplicitBias**

---

### Session 4 (60–90m) — Write: "GD chooses among infinite solutions"

**Goal:** Produce a short explainer of implicit bias that connects back to the optimization narrative.

#### Structure

1. The setup: underdetermined systems have infinitely many solutions
2. GD from zero → min-norm: the row-space argument
3. What this means: the optimizer is a regularizer, even without explicit regularization
4. Forward connection: in deep learning, the implicit bias story is richer and less understood
5. Your experimental results from S2

#### Output

- Note: **Week11_ImplicitBias_Explainer**

### Week 11 "done" checklist

- [ ] I can prove that GD from zero converges to the min-norm solution for underdetermined linear regression.
- [ ] My experiment confirms this numerically.
- [ ] I can explain "implicit bias" as a concept and why it matters for deep learning.

---

## Week 12 — Convolution mini-implementation

**Scope:** Standalone pure-Python (loops OK), tiny sizes, grad-check via finite differences. No need to integrate into your `Value` engine.

### Outcome targets

- **Conceptual:** Convolution as a structured linear operator with weight sharing; the Toeplitz view.
- **Code:** Minimal conv forward (1D or tiny 2D) with finite-diff gradient check.
- **Teaching artifact:** 1-page note on "conv as structured linear operator."

---

### Session 1 (60–90m) — Math: convolution as structured matrix multiply

**Goal:** Understand conv as a special case of matrix multiplication with shared weights.

#### Prep (10m)

Read the CS231n "Convolutional Neural Networks" notes: `cs231n.github.io/convolutional-networks/`
Focus on: spatial arrangement (stride, padding, output size formulas), parameter sharing, and the "filter slides across the input" framing.

#### Work (50–70m)

1. **1D convolution as sliding dot product**
   - Input: $x \in \mathbb{R}^n$, filter: $k \in \mathbb{R}^m$, output: $y_i = \sum_j k_j \cdot x_{i+j}$
   - Write this as $y = Kx$ where $K$ is a Toeplitz (banded diagonal) matrix
2. **Weight sharing as a Jacobian constraint**
   - A fully connected layer: every entry of $W$ is independent
   - A conv layer: $W$ is Toeplitz — same weights appearing in multiple rows
   - The Jacobian $\frac{\partial y}{\partial x}$ is the same Toeplitz matrix $K$
   - The gradient $\frac{\partial L}{\partial k}$ sums contributions from all positions where $k$ was applied
3. **Output size formula:** $(n - m + 2p) / s + 1$ where $p$ = padding, $s$ = stride
4. **Extension to 2D** (conceptual): the filter becomes 2D, the Toeplitz matrix becomes a doubly-block-Toeplitz matrix, but the principle is identical

#### References

- CS231n conv notes (primary geometric introduction)
- GBC Ch. 9 (formal treatment, Toeplitz view)

#### Output (last 10m)

Create note: **Week12_S1_Conv_Math**

- The Toeplitz matrix view with a small worked example
- Output size formula

---

### Session 2 (60–90m) — Code: implement conv + grad check

**Goal:** Working conv forward and backward with gradient verification.

#### Option A: Roll your own (matches your "build from scratch" style)

1. **1D conv forward:** Implement with explicit loops
2. **1D conv backward:** Derive $\frac{\partial L}{\partial k}$ and $\frac{\partial L}{\partial x}$ from the Toeplitz view
3. **Gradient check:** Finite difference verification for both $\nabla_k L$ and $\nabla_x L$

#### Option B: CS231n Assignment 2 — Conv Nets section (better test harness)

- URL: `cs231n.github.io/assignments2024/assignment2/`
- The "Convolutional Networks" notebook has you implement 2D conv forward and backward in numpy
- Provides numerical gradient checking utilities and CIFAR-10 to test on
- **Scope control:** Only do the conv layer forward/backward sections. Skip PyTorch and Network Visualization.

#### Output (last 10m)

Create note: **Week12_S2_Conv_Code**

- Working implementation (1D or 2D)
- Gradient check results (pass/fail with error magnitude)

---

### Session 3 (60–90m) — Read: CNN inductive bias

**Goal:** Understand what assumptions convolutions encode and why they work for spatial data.

#### Reading material (pick one)

- **Option A:** CS231n "Understanding and Visualizing CNNs" (`cs231n.github.io/understanding-cnn/`) — tSNE embeddings, deconvnets, what CNNs learn at each layer
- **Option B:** GBC Ch. 9.1–9.4 — formal treatment of convolution, pooling, and the inductive bias of translation equivariance
- **Option C (shorter):** The relevant sections of the LeCun et al. "Gradient-Based Learning Applied to Document Recognition" (1998) — the original motivation

#### Active reading workflow

1. Before reading: list 3 properties of images that make convolution a good prior
2. During reading: note where translation equivariance is explicitly invoked
3. After: write one sentence on what kinds of problems convolution would be a _bad_ prior for

#### Output (last 10m)

Create note: **Week12_S3_Read_CNN**

---

### Session 4 (60–90m) — Write: conv as structured linear operator

**Goal:** Write a note connecting convolution back to your linear algebra framework.

#### Structure

1. Convolution as constrained matrix multiply (Toeplitz structure)
2. Weight sharing as parameter tying → structured Jacobian
3. The inductive bias: translation equivariance, locality
4. Forward connection to attention (Week 13): attention _learns_ the connectivity pattern that conv hard-codes

Include one diagram (ASCII or hand-drawn): the filter sliding across the input, aligned with the corresponding rows of the Toeplitz matrix.

#### Output

- Note: **Week12_Conv_Explainer**

### Week 12 "done" checklist

- [ ] I can explain convolution as a Toeplitz matrix and derive the output size formula.
- [ ] My implementation passes gradient checks.
- [ ] I can articulate what inductive bias convolution encodes (locality, translation equivariance, weight sharing).

---

## Week 13 — Attention mini-implementation

**Scope:** Pure Python, softmax stabilization, attention weight visualization. Standalone — no need to integrate into your `Value` engine.

### Outcome targets

- **Conceptual:** Attention as differentiable soft lookup; why $\sqrt{d}$ scaling; connection to inner products and metrics.
- **Code:** Scaled dot-product attention with softmax stabilization + heatmap.
- **Teaching artifact:** 1-page note on "attention as differentiable lookup / soft addressing."

---

### Session 1 (60–90m) — Math: derive attention from first principles

**Goal:** Derive scaled dot-product attention and understand each design choice.

#### Prep (15–20m)

Watch CS224n Lecture 8 (2024 YouTube): "Self-Attention and Transformers"

- Full playlist: search "CS224N 2024" on YouTube
- The lecture is ~80 min; watch at 1.5x and focus on the 30-minute block where Manning derives attention. Skip NLP-specific application examples.

#### Work (40–60m)

1. **The retrieval analogy**
   - Query $q$: "what am I looking for?"
   - Keys $k_i$: "what does position $i$ contain?"
   - Values $v_i$: "what does position $i$ contribute?"
   - Attention is a weighted average of values, where weights come from query-key similarity
2. **Scaled dot-product attention**
   - $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$
   - The inner product $q \cdot k$ measures similarity — this IS the same inner product story from Week 4
   - Softmax converts similarities to a probability distribution
3. **Why $\sqrt{d_k}$ scaling**
   - If $q$ and $k$ have entries drawn from $\mathcal{N}(0, 1)$, then $q \cdot k$ has variance $d_k$
   - Large variance → softmax saturates → gradients vanish
   - Dividing by $\sqrt{d_k}$ normalizes the variance to 1
4. **Softmax stabilization**
   - $\text{softmax}(z)_i = \exp(z_i - \max(z)) / \sum_j \exp(z_j - \max(z))$
   - Subtracting the max prevents overflow without changing the result
5. **Multi-head attention** (conceptual only)
   - Multiple parallel attention operations with different learned projections
   - Each head can attend to different types of relationships

#### References

- CS224n Lecture 8 (2024) — primary derivation
- Vaswani et al., "Attention Is All You Need" (2017), sections 1–3 — the original

#### Output (last 10m)

Create note: **Week13_S1_Attention_Math**

- Full derivation of scaled dot-product attention
- One paragraph: the $\sqrt{d_k}$ argument
- One paragraph: connection to inner products / metric choice from Week 4–5

---

### Session 2 (60–90m) — Code: implement attention + visualize

**Goal:** Working scaled dot-product attention in pure numpy/Python, with a heatmap of attention weights.

#### Work plan (60–75m)

1. **Implement `scaled_dot_product_attention(Q, K, V)`**
   - Compute $QK^T / \sqrt{d_k}$
   - Apply softmax with numerical stabilization (subtract max)
   - Multiply by $V$
   - Return both the output and the attention weights (for visualization)
2. **Toy task:** Create a simple sequence where attention should learn an obvious pattern
   - E.g., a copying task where position $i$ should attend to position $i-1$
   - Or a "find the maximum" task
3. **Visualize attention weights** as a heatmap (matplotlib `imshow`)
   - Rows = query positions, columns = key positions
   - Brighter = higher attention weight

#### Stretch: CS224n later assignments

The 2025 default final project involves implementing GPT-2 components. This is a better fit for Q2 than Week 13.

#### Output (last 10m)

Create note: **Week13_S2_Attention_Code**

- Working attention function
- Attention heatmap on toy task

---

### Session 3 (60–90m) — Read: "Attention Is All You Need"

**Goal:** Read the original transformer paper with enough background to understand the design choices.

#### Reading material

- **Primary:** Vaswani et al., "Attention Is All You Need" (2017), sections 1–3
  - Focus questions: Why Q/K/V instead of a single similarity function? Why multi-head? Why positional encoding?
- **Secondary:** Jay Alammar, "The Illustrated Transformer" — best visual walkthrough of the full architecture
- **Tertiary:** CS224n lecture notes on attention (available on the course website)

#### Active reading workflow

1. Before reading: write your best answer to "why separate Q, K, V matrices?"
2. During reading: mark every design decision and its stated justification
3. After reading: update your answer. Does the paper's justification convince you?

#### Output (last 10m)

Create note: **Week13_S3_Read_Attention**

- Answer to "why Q/K/V?"
- 3 design decisions and their justifications
- 2 questions for Q2

---

### Session 4 (60–90m) — Write: attention as differentiable lookup

**Goal:** Write a note that explains attention as a generalization of hard lookup, connecting back to your optimization/metric thread.

#### Structure

1. **Hard lookup vs. soft lookup**
   - Hard: retrieve value at exact index → not differentiable
   - Soft: weighted average of all values → differentiable, trainable
2. **Attention weights as similarity under a learned metric**
   - The Q and K projections define a learned bilinear form: $\text{similarity}(x_i, x_j) = (W_Q x_i)^T (W_K x_j)$
   - This IS an inner product in the projected space — the same structure as your Week 4–5 work
3. **Softmax as soft argmax** — converts similarities to a distribution
4. **Why this matters:** attention lets the model learn _which_ positions are relevant, unlike conv which hard-codes locality
5. **Forward connection:** in Q2, you'll see attention applied to vision (ViT), RL, and robotics

#### Output

- Note: **Week13_Attention_Explainer**

### Week 13 "done" checklist

- [ ] I can derive scaled dot-product attention and explain the $\sqrt{d_k}$ scaling.
- [ ] I can implement attention with numerically stable softmax.
- [ ] I can explain why Q/K/V are separate projections.
- [ ] I can connect attention weights back to the inner product / metric framework.

---

## Week 14 — Q1 wrap / polish (buffer week)

### Outcome targets

- **Repo:** Clean, tested, reproducible.
- **Synthesis:** A final writeup telling the "backprop → optimization → generalization" story arc.
- **Forward:** Q2 launch questions.

---

### Session 1 (60–90m) — Repo cleanup + tests

1. Ensure every major piece of code runs cleanly
2. Add a README with "how to reproduce" instructions
3. Verify gradient checks still pass
4. Optional: add a few unit tests for your optimizer implementations

---

### Session 2 (60–90m) — Synthesis writeup

Write a 2–3 page document telling the Q1 story arc:

1. **Backprop** (Weeks 1–3): reverse-mode autodiff, VJPs, the computational graph
2. **Geometry** (Weeks 4–5): gradients are covectors, steepest descent depends on the metric, condition numbers and ravines
3. **Optimization** (Weeks 6–9): momentum as EMA, Adam as diagonal preconditioning, all optimizers as metric choices
4. **Generalization** (Weeks 10–11): loss landscapes, implicit bias, what the optimizer gives you for free
5. **Building blocks** (Weeks 12–13): conv as structured linear operator, attention as differentiable lookup
6. **The thread:** the inner product / metric appears everywhere — in defining gradients, in choosing descent directions, in adaptive optimizers, and in attention similarity

---

### Session 3 (60–90m) — Self-test (optional)

Pick one of these as a check on Q1 mastery:

- **CS231n Assignment 1:** Implement kNN, SVM, Softmax, and a 2-layer net from scratch. If you can do it comfortably in one session, Q1 is solid.
- **Derivation challenge:** Starting from $f(x) = \frac{1}{2} x^T A x - b^T x$, derive GD, momentum, and Adam updates. Show how each one changes the effective condition number.

---

### Session 4 (60–90m) — Q2 planning

Write 5–10 questions that Q1 leaves open and that Q2 (perception + representation learning) should answer:

- How do CNNs learn hierarchical features? (Week 12 left this as a forward pointer)
- What does attention buy you over convolution for images? (ViT)
- How do representation learning objectives (contrastive, generative) connect to the optimization story?
- Where do the ideas from Q1 (metrics, implicit bias, preconditioning) show up in modern architectures?

#### Output

- Note: **Q2_Launch_Questions**

### Week 14 "done" checklist

- [ ] Repo is clean and reproducible.
- [ ] Synthesis writeup connects the Q1 arc.
- [ ] Q2 questions are concrete enough to guide planning.

---

## Resource Quick Reference

### CS4787 (Cornell) — Public lecture PDFs

All at `cs.cornell.edu/courses/cs4787/2019sp/notes/lectureN.pdf`

| Lecture | Topic                          | Your Week    |
| ------- | ------------------------------ | ------------ |
| L4      | GD, conditioning               | Review       |
| L5      | SGD                            | 7            |
| L7      | Momentum, Nesterov             | 7            |
| L8      | Preconditioning, AdaGrad, Adam | 9            |
| L9      | Variance reduction, averaging  | 9 (optional) |
| L11     | Deep neural networks           | 10–11        |
| L13     | Early stopping, batch norm     | 8 (optional) |

### CS231n (Stanford) — Notes + assignments

| Resource                                                                 | Your Week |
| ------------------------------------------------------------------------ | --------- |
| `cs231n.github.io/neural-networks-2/` (regularization, init, batch norm) | 8         |
| `cs231n.github.io/neural-networks-3/` (training diagnostics, optimizers) | 7–9       |
| `cs231n.github.io/convolutional-networks/` (conv layers, architectures)  | 12        |
| Assignment 2: Conv forward/backward in numpy                             | 12        |

### CS224n (Stanford) — Attention + Transformers

| Resource                                               | Your Week       |
| ------------------------------------------------------ | --------------- |
| Lecture 8 (YouTube 2024): Self-attention, Transformers | 13              |
| Lecture 9 (YouTube 2024): Pretraining, GPT, BERT       | 13 / Q2 preview |

### Textbooks (free online)

| Book                                                                                                  | Chapters   | Weeks   |
| ----------------------------------------------------------------------------------------------------- | ---------- | ------- |
| Bottou et al., _Optimization Methods for Large-Scale ML_ (`arxiv.org/pdf/1606.04838v1.pdf`)           | Ch. 3–5, 7 | 7–9     |
| Shalev-Shwartz & Ben-David, _Understanding ML_ (`cs.huji.ac.il/~shais/UnderstandingMachineLearning/`) | Ch. 12–14  | 7–11    |
| Goodfellow et al., _Deep Learning_ (`deeplearningbook.org`)                                           | Ch. 8, 9   | 7–9, 12 |

### Papers

| Paper                                                               | Week |
| ------------------------------------------------------------------- | ---- |
| Kingma & Ba, "Adam" (2014)                                          | 9    |
| Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2019) | 9    |
| Li et al., "Visualizing the Loss Landscape of Neural Nets" (2018)   | 10   |
| Neyshabur et al., "Implicit Regularization in Deep Learning" (2014) | 11   |
| Vaswani et al., "Attention Is All You Need" (2017)                  | 13   |
