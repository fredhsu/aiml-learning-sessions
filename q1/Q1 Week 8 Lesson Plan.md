# Q1 · Week 8 · Session-by-Session Lesson Plan

## Carry-forward from Week 7

Two open issues flagged in `backprop_explainer-week7.md` (v0.6) that belong in S4 this week:

- **Typo:** "gradietns" — find and fix.
- **Conceptual imprecision:** The Week 6 explainer attributed oscillation to eigenvalues directly. The precise claim is: _gradient components along high-curvature eigendirections flip sign_. The eigenvalues measure the curvature; the sign-flipping is a property of the gradient components projected onto those eigendirections. Correct the phrasing in the momentum section before writing Week 8 additions.

---

## Week 8 outcome targets (ship by week's end)

- **Conceptual:** L2 regularization as a gradient contribution and as a geometric constraint; what train/val divergence tells you and what it doesn't; early stopping as an implicit form of regularization.
- **Math:** Clean derivation of the weight-decay update from the regularized objective; connection to the Bayesian (Gaussian prior) framing; understanding of where and why the connection between L2 regularization and weight decay breaks when combined with adaptive optimizers like Adam (preview of Week 9).
- **Code:** Working train/val split, per-epoch accuracy logging, and L2 weight decay toggle integrated into your optimizer framework from Week 7.
- **Teaching artifact:** A one-page "training lies" diagnostic checklist covering the five canonical failure modes.

---

## Session 1 (60–90m) — Math: L2 regularization as gradient modification

**Goal:** Derive weight decay from first principles and understand what L2 regularization actually constrains geometrically.

### Prep — cold reconstruction first (10m)

Before opening any notes or the CS231n page, write answers to these three questions from memory:

1. What is the modified loss function when L2 regularization with strength $\lambda$ is added to a base loss $L(w)$?
2. What does the gradient of that modified loss look like, compared to the unregularized gradient?
3. In the weight update rule, what changes?

Write a few lines for each — even rough. The goal is to surface what you already know and locate your uncertainty before the reading shapes your thinking.

After writing, skim the regularization section of CS231n "Neural Networks Part 2" (`cs231n.github.io/neural-networks-2/`) — L2 and dropout only, approximately 15 minutes. Then close it.

### Work (50–65m)

**1. Derive the weight decay update (20m)**

Start from the regularized objective:
$$\tilde{L}(w) = L(w) + \frac{\lambda}{2} \|w\|^2$$

Take the gradient. Then write out the full gradient descent update step for $\tilde{L}$, and algebraically rearrange it into the weight decay form. The target form should make the "decay" visible as a multiplicative shrinkage of $w$ before the gradient step lands.

Check: does the factor out front tell you what fraction of the weight survives each step? What does this imply about the long-run behavior if the gradient $\nabla L$ were zero?

**2. Geometric interpretation (15m)**

The L2 penalty $\frac{\lambda}{2}\|w\|^2$ adds a spherical bowl to the loss surface, centered at the origin. Think through these questions without looking anything up:

- In 2D, sketch (mentally or on paper) what the combined loss surface looks like when the original loss has a minimum far from the origin. Where does the regularized minimum land?
- The location of the regularized minimum depends on the curvature of $L$ in each direction — specifically, on the eigenvalues of the Hessian $H$ of $L$. Try to see why: in directions where $L$ is steep (large $\lambda_i$), the original loss "wins" and pulls the solution less; in flat directions (small $\lambda_i$), the L2 bowl dominates more. Does this match your intuition?
- This is exactly the conditioning story from Week 5, appearing again: the Hessian eigenvalues determine how regularization redistributes weight across directions.

**3. Bayesian framing (10m)**

L2 regularization has a probabilistic interpretation: it is equivalent to placing an isotropic Gaussian prior $p(w) = \mathcal{N}(0, \frac{1}{\lambda} I)$ on the weights and doing MAP estimation. Write out the log-posterior maximization and verify that it reduces to minimizing $\tilde{L}(w)$.

This is worth having in your notes not because you'll use Bayesian reasoning constantly, but because it clarifies what L2 _actually_ encodes as a belief: "weights should be small unless the data forces them not to be." It also previews why L2 regularization is not the same as weight decay for adaptive optimizers — the Gaussian prior assumption interacts with the adaptive scaling.

**4. Flag the AdamW distinction (5m)**

Note, don't derive yet: when the gradient update is scaled per-parameter (as in Adam), adding L2 to the loss causes the regularization to be _inconsistently_ scaled — parameters with large gradient variance get less regularization than parameters with small variance, which distorts the prior. Decoupled weight decay (AdamW) fixes this by applying the decay _outside_ the adaptive scaling. This is Week 9's S3 topic; just flag it now.

### Output (last 10m)

Create note: **Week8_S1_Regularization_Math**

Include:

- The derivation: regularized loss → gradient → weight decay update (the algebraic steps, not just the result)
- One paragraph: the geometric picture (spherical bowl + Hessian eigenvalue interaction)
- Two sentences: the Bayesian framing and what it says about the implicit assumption
- One flagged open question: the AdamW distinction (to be resolved in Week 9)

---

## Session 2 (60–90m) — Code: eval loop, metrics, and weight decay

**Goal:** Add the minimum viable training infrastructure to your existing framework: train/val split, per-epoch accuracy, and L2 weight decay. Observe what changes when regularization is on vs. off.

### Prep (5m)

Pull up `w6s2.py`. The 3-class toy dataset, `MLP`, and `mini_batch` training loop are your starting point. You'll be extending, not rewriting.

### Work plan (55–70m)

**1. Train/val split (10m)**

Split the 9-sample dataset into train (6–7 samples) and val (2–3 samples). With a dataset this small, the split is almost symbolic — that's fine, the goal is to wire the infrastructure correctly.

Run training and log both train loss and val loss after each epoch (or every N steps). Confirm both are accessible before moving on.

**2. Accuracy metric (10m)**

Implement classification accuracy on the val set. After each epoch, run a forward pass on val samples (no gradient computation needed), take the argmax of the logits, and compare to the true labels. Log this alongside the loss.

One thing to notice: loss can decrease while accuracy stays flat (or vice versa, especially early). This is a real diagnostic and it should show up in your plots.

**3. L2 weight decay in your optimizer (20m)**

Add a `weight_decay` parameter to `SGD` (default `0.0`). Implement the decay step:
$$w \leftarrow w(1 - \alpha \lambda)$$
This should happen _before_ the gradient step, or equivalently be folded into the update as:
$$w \leftarrow w - \alpha(\nabla L + \lambda w)$$

Both formulations are equivalent — verify this algebraically before coding, then choose one.

Then add the same parameter to `MomentumSGD`. Does the weight decay interact with the velocity accumulator? It shouldn't (the decay is applied to parameters, not to the velocity buffer), but make sure your implementation reflects this.

**4. Experiment (15m)**

Run four conditions on the same starting point and same number of epochs:

- SGD, no decay
- SGD, with decay ($\lambda = 0.01$ or $0.001$, tune to taste)
- Momentum, no decay
- Momentum, with decay

Log train loss, val loss, and val accuracy for each. With a 9-sample toy dataset you won't see generalization differences — but you _will_ see the weight norms shrink and the loss surface shift. That's enough to verify the implementation is correct.

One diagnostic to run: after training with vs. without decay, print `sum(p.data**2 for layer in mlp.layers for p in layer.parameters())` — the total squared weight norm. It should be substantially smaller with decay on.

**5. Stretch — Dropout (optional, if 15–20 min remain)**

The CS231n Assignment 2 dropout section (`cs231n.github.io/assignments2024/assignment2/`) is a clean numpy implementation of dropout forward/backward. If you have time, implement a minimal `Dropout` pass in your `Value`-engine or as a standalone numpy function. Verify that setting `p=0.0` (no dropout) matches the non-dropout path exactly.

This is genuinely optional — Week 8's required S2 output is the weight decay experiment.

### Output (last 10m)

Create note: **Week8_S2_TrainVal_Code**

Include:

- The extended `SGD` / `MomentumSGD` code snippets (or link to your file)
- Train/val loss curves for the four experimental conditions (rough matplotlib plots are fine)
- Two concrete observations: one about the weight norm, one about something that surprised you or didn't behave as expected

---

## Session 3 (60–90m) — Read: CS231n training diagnostics + CS4787 early stopping

**Goal:** Build a mental model of what training pathologies look like and how to diagnose them efficiently. This session's output becomes the raw material for S4.

### Prep — pre-reading exercise (5m)

Before opening anything, write a list of 3 things you currently check when training seems broken. Be specific — not "check the loss" but "check whether train loss decreases on the first 5 steps with a tiny learning rate." These three items are your baseline; you'll revisit them at the end.

### Reading material

**Primary — CS231n "Neural Networks Part 3: Learning and Evaluation"**
URL: `cs231n.github.io/neural-networks-3/`

This is one of the best practical training guides available. Read the following sections actively (not just skimming):

- _Gradient checks_ — the conditions under which numerical gradient checks are reliable and where they mislead you
- _Sanity checks before training_ — the two fast checks that rule out implementation bugs (initialized loss at random chance? loss decreases on overfit to one batch?)
- _Babysitting the learning process_ — what loss curves, accuracy curves, and update-to-weight ratio plots tell you
- _Hyperparameter optimization_ — random search vs. grid search, why random search is usually better

**Secondary — CS4787 Lecture 13 (Spring 2019): "Early stopping and batch normalization"**
URL: `cs.cornell.edu/courses/cs4787/2019sp/notes/lecture13.pdf`

Skim for the early stopping section only (batch norm is Week 8 optional / later). The key point from CS4787: early stopping is not just a practical trick — it has a theoretical relationship to L2 regularization. For a quadratic loss, stopping early and regularizing with $\lambda$ can produce the same solution under the right conditions. This framing connects Week 8 back to the implicit bias thread you'll pick up in Week 11.

### Active reading workflow

1. Mark every diagnostic technique in CS231n Part 3 that you _don't currently use_ in your training loops.
2. For each marked item: write one sentence on why it would have caught a bug or pathology you've encountered (real or hypothetical).
3. After finishing both readings: go back to your pre-reading list of 3 checks. Update it — what would you add? What ordering changes?

### Output (last 10m)

Create note: **Week8_S3_Read_TrainingDiagnostics**

Include:

- Your pre-reading list (unedited) and your post-reading updated list
- Two diagnostics you hadn't thought of before, with a one-sentence note on _when_ each applies
- One sentence on the early stopping ↔ L2 regularization relationship from CS4787 L13 — flag this as a forward connection to Week 11 implicit bias

---

## Session 4 (60–90m) — Write: "training lies" diagnostic checklist + explainer updates

**Goal:** Produce a one-page reference for training pathologies _and_ fix the two carry-forward issues in your backprop explainer.

### Part A — Fix carry-forward issues (15m)

Open `backprop_explainer-week7.md` (or your current working version). Before writing anything new:

1. **Fix the typo:** Search for "gradietns" and correct to "gradients."
2. **Sharpen the oscillation language:** In the momentum section, locate wherever oscillation is attributed to eigenvalues and tighten it. The precise statement is: _gradient components projected onto high-curvature eigendirections tend to flip sign each step, because the curvature in that direction is large enough that the gradient overshoots_. The eigenvalues are the _cause_ (they quantify curvature), but the sign-flipping is a property of the gradient components, not the eigenvalues themselves.

These are small changes but they matter — your explainer is something you're building to be correct, not just directionally right.

### Part B — Write the diagnostic checklist (45m)

Produce a note structured around five canonical failure modes. For each: what you observe, the most likely cause, and the first thing to check.

**1. Loss not decreasing from the start**

- Possible causes: learning rate too high (gradient explosion), learning rate too low (effectively zero movement), bug in the loss computation, bug in the backward pass (gradients not flowing)
- First check: run the "overfit one sample" sanity check — if you can't get train loss to zero on a single sample with a generous LR, there's a bug

**2. Train loss decreasing but val loss not improving (or diverging)**

- Possible causes: overfitting (model capacity exceeds what the data can constrain), data leakage (val set accidentally contaminated), wrong metric (loss decreasing but in a direction that doesn't generalize), the dataset is too small for the split to be meaningful
- First check: is the val loss increasing monotonically, or just not keeping pace with train? The shape of the divergence tells you something

**3. Both losses decreasing, but very slowly**

- Possible causes: learning rate too low, poor initialization (saturated activations, zero gradients), high condition number (the conditioning story from Weeks 5–7), bad data scaling
- First check: plot the update-to-weight ratio (CS231n recommends this around $10^{-3}$); a ratio far below that signals the LR is too small relative to weight magnitudes

**4. Loss becomes NaN or Inf**

- Possible causes: learning rate too high → divergence, log(0) in cross-entropy, exp overflow in softmax without stabilization, gradient explosion in a deep network
- First check: add gradient norm logging; if norms are exploding in early steps, reduce LR or clip gradients

**5. Accuracy flat despite loss decreasing**

- Possible causes: loss and accuracy can decouple when the model is miscalibrated (e.g., confidently wrong), class imbalance dominates, or the decision boundary isn't moving even though the loss surface is being minimized in a direction that doesn't affect classification
- First check: examine the per-class accuracy separately; a flat overall accuracy often means one class is being ignored

**Additional notes to include:**

- The "two fast sanity checks" from CS231n Part 3 (worth memorizing): (a) does random-init loss match theoretical chance-level loss for your task? (b) can you overfit a single training batch to near-zero loss?
- The update-to-weight ratio heuristic: $\frac{\alpha \|\nabla L\|}{\|w\|} \approx 10^{-3}$ as a rough guide
- The early stopping connection: early stopping has a regularization effect analogous to L2; stopping epoch $T$ roughly corresponds to a regularization strength $\lambda \sim \frac{1}{\alpha T}$ for quadratic losses

### Part C — Add a Week 8 section to the explainer (15m)

Add a short section to `backprop_explainer-week7.md` (which will become v0.6.1 or your numbering convention):

**Week 8: Regularization and diagnostics**

Contents:

1. L2 regularization as weight decay — the derivation in one paragraph, the geometric picture in one sentence
2. What weight decay does and does not guarantee (small-norm solution, not necessarily generalizable)
3. The Bayesian framing: one sentence
4. Early stopping as implicit regularization: the forward pointer to Week 11

### Output

Two deliverables:

- Note: **Week8_Diagnostic_Checklist** (the five-failure-mode reference, standalone)
- Updated explainer: **backprop_explainer-v0.6.1** (or your versioning) with typo fixed, oscillation claim sharpened, and Week 8 regularization section appended

---

## Forward connections to seed

These are worth jotting in your Week 8 notes so they're explicit when you get there:

- **Week 9:** Adam + L2 ≠ AdamW. Weight decay only decouples correctly when applied _outside_ the adaptive scaling. The geometric picture from S1 (L2 as a Gaussian prior) breaks down when the effective metric is per-parameter adaptive.
- **Week 11:** Early stopping and L2 regularization produce similar solutions on quadratic losses. Both are forms of implicit bias: they select among solutions not by the loss function alone but by the dynamics and constraints of the optimization procedure.
- **Q3 (RL):** Weight decay reappears as a common regularization technique for policy networks. The diagnostic checklist directly applies to RL training, which has additional pathologies (reward sparsity, non-stationarity) layered on top.

---

## Week 8 "done" checklist

- [ ] I can derive the weight decay update from $\tilde{L}(w) = L(w) + \frac{\lambda}{2}\|w\|^2$ without looking anything up.
- [ ] I can explain geometrically what L2 regularization does to the loss surface and where the regularized minimum lands relative to the unregularized one.
- [ ] I know the Bayesian framing (Gaussian prior / MAP) and can state the one sentence version.
- [ ] I can flag the AdamW distinction without deriving it yet.
- [ ] My training loop has train/val loss logging, per-epoch accuracy, and a weight decay toggle that visibly changes the weight norm.
- [ ] I have the two CS231n sanity checks memorized and can state the update-to-weight ratio heuristic.
- [ ] The typo and oscillation imprecision in the backprop explainer are fixed.
- [ ] My diagnostic checklist covers the five failure modes with cause + first-check for each.
