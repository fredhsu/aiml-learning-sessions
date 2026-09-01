---
created: 2026-05-16
id: 019e3da5-a07e-78b3-b063-ac66b098141a
tags:
- lesson-plan
- optimization
- generalization
- implicit-bias
title: Q1 · Week 11 · Session-by-Session Lesson Plan
---

# Q1 · Week 11 · Session-by-Session Lesson Plan

## Context: Where You're Coming From

Week 10 closed strong. You built filter-normalized 1D and 2D loss-slice plots on a trained MLP, read Li et al. and Dinh et al. against your own visualizations, and wrote a six-section interpretation note that took the sharp/flat hypothesis seriously while presenting Dinh et al.'s reparameterization argument as the counterweight. The forward hook you wrote at the end of S4 named the question Week 11 picks up directly:

> Among all the zero-training-loss solutions in an overparameterized network, why does GD systematically land on ones that generalize? The answer involves the trajectory of GD itself — what's known as **implicit bias**.

That's the thread.

Week 10 was about *static* geometry: what the landscape looks like at a point. Dinh et al. forced you to admit that static Hessian-sharpness can't carry the generalization story on its own — reparameterization breaks it. Week 11 pivots to *dynamic* properties: the trajectory GD takes and what it implicitly selects for. This is the cleanest possible follow-on, and the math sits at a sweet spot — the underdetermined linear regression case is provable in a page or two, and the experiment is small enough you can build it from scratch in S2 without a course assignment.

The metric thread continues but takes a different form. In Weeks 5–9, "choice of metric" determined the *step direction*. This week, the inner product shows up in a different place: as the geometry the *solution set* is measured against. GD doesn't pick the L1-minimum solution or the L∞-minimum solution; from $w_0 = 0$ it picks the L2-minimum solution. Why L2? Because the row-space argument is fundamentally a statement about orthogonal projection in the Euclidean inner product. This is the same metric framing you've used all quarter, applied at a different layer of the problem.

No open carry-forward issues from Week 10. The backprop explainer stays at v0.7 — this week's primary writing artifact is a standalone explainer on implicit bias.

---

## Week 11 outcome targets (ship by week's end)

- **Conceptual:** What "implicit bias" means precisely; why GD from $w_0 = 0$ on underdetermined linear regression picks the minimum-norm solution; what changes with non-zero initialization; why this matters for deep networks even though the linear case is the only one with a clean proof.
- **Math:** The row-space argument (GD iterates stay in $\text{rowspace}(X)$ when started at zero). Connection to the orthogonal decomposition $\mathbb{R}^p = \text{rowspace}(X) \oplus \text{null}(X)$. Why this picks out $w^* = X^T(XX^T)^{-1}y$.
- **Code:** Underdetermined linear regression with $p \gg n$. Verify GD → min-norm numerically. Ablate: initialization (zero vs. random), optimizer (GD vs. momentum vs. Adam). Build a small comparison table of $\|w_{\text{final}}\|$ values.
- **Teaching artifact:** Standalone explainer note titled "GD chooses among infinite solutions." Connect back to the Week 10 forward hook explicitly so the two notes form a pair in your Zettelkasten.

---

## Session 1 (60–90m) — Math: why GD from zero converges to min-norm

**Goal:** Prove the row-space result cleanly enough that you can write it down without reference. Understand precisely what role the zero initialization plays — and what breaks if you change it.

### Cold reconstruction prompts (do these before reading anything — 10–15m)

Write your answers in a fresh note before you look at any reference material. These are calibration anchors for the post-S1 review.

1. **The setup.** Linear regression with $X \in \mathbb{R}^{n \times p}$, $p > n$, target $y \in \mathbb{R}^n$. The loss is $L(w) = \frac{1}{2}\|Xw - y\|^2$. How many solutions are there with $L(w) = 0$? Describe the solution set geometrically.

2. **The GD update.** Write down the GD update for this loss. What's $\nabla L(w)$ explicitly?

3. **The conjecture.** If you start at $w_0 = 0$, where do successive GD iterates live? Try unrolling one or two steps. Does the iterate $w_k$ have any structural property in common with the rows of $X$?

4. **What changes with $w_0 \neq 0$?** Predict: if you start at a generic random $w_0$, will GD still converge to the min-norm solution? If not, what does it converge to?

5. **Connection to Week 10.** The Dinh et al. argument said static Hessian-sharpness is parameterization-dependent and can't explain generalization. Is the row-space argument parameterization-dependent? Does it rely on the Euclidean inner product anywhere?

Save these as `Week11_S1_PreReconstruction` — same format as your Week 10 S1 pre-reading conjectures.

### Work (45–60m)

After your reconstruction, work through the proof properly.

1. **Restate the setup.** Underdetermined system $Xw = y$, $p > n$. The set of zero-loss solutions is the affine subspace $\{w : Xw = y\}$, which (assuming $X$ has rank $n$) has dimension $p - n$. Infinitely many solutions.

2. **The GD update.** $\nabla L(w) = X^T(Xw - y)$. So
$$w_{k+1} = w_k - \alpha X^T(Xw_k - y)$$
The crucial observation: the update direction $-\alpha X^T(Xw_k - y)$ is a linear combination of the *columns* of $X^T$ — i.e., the rows of $X$. Every GD step moves $w$ by a vector in $\text{rowspace}(X)$.

3. **Induction.** If $w_0 = 0 \in \text{rowspace}(X)$ (trivially — the zero vector is in every subspace), then $w_1 = w_0 + \Delta_0$ where $\Delta_0 \in \text{rowspace}(X)$, so $w_1 \in \text{rowspace}(X)$. By induction, $w_k \in \text{rowspace}(X)$ for all $k$.

4. **The min-norm solution lives in the row space.** Decompose $\mathbb{R}^p = \text{rowspace}(X) \oplus \text{null}(X)$ (orthogonal decomposition under the Euclidean inner product). Any solution $w$ to $Xw = y$ can be written $w = w_{\parallel} + w_{\perp}$ where $w_{\parallel} \in \text{rowspace}(X)$ and $w_{\perp} \in \text{null}(X)$. Since $Xw_{\perp} = 0$, you have $Xw_{\parallel} = y$ — the row-space component alone solves the system. And because $\|w\|^2 = \|w_{\parallel}\|^2 + \|w_{\perp}\|^2$ (Pythagoras under the inner product), the minimum-norm solution has $w_{\perp} = 0$. So the min-norm solution is the unique solution in $\text{rowspace}(X)$.

5. **Conclusion.** GD from zero stays in $\text{rowspace}(X)$ forever. If it converges to *a* zero-loss solution (which it does, with $\alpha$ small enough), and the only zero-loss solution in $\text{rowspace}(X)$ is the min-norm one, then GD → min-norm.

6. **Closed form for the min-norm solution.** Solve $X(X^T \beta) = y$ for $\beta$ — this gives $\beta = (XX^T)^{-1}y$. Then $w^* = X^T \beta = X^T(XX^T)^{-1}y$. This is the closed form to compare against in S2.

7. **What changes with $w_0 \neq 0$?** GD converges to $w_0 + (\text{rowspace component})$. More precisely: $w_\infty = w_{0,\perp} + w^*_{\text{min-norm,shifted}}$ where the null-space component of $w_0$ is preserved exactly because GD never moves in $\text{null}(X)$. So *the implicit bias is initialization-dependent.* This is one of the points the experiment in S2 should make vivid.

8. **Metric remark (plant the seed; harvest in S4).** The whole argument relies on the orthogonal decomposition $\mathbb{R}^p = \text{rowspace}(X) \oplus \text{null}(X)$, which is an orthogonal decomposition *under the Euclidean inner product*. If you ran "GD" using a different metric — i.e., preconditioned GD with some $M \neq I$ — the orthogonality would shift and the implicit bias would point to a different solution. The Week 5 metric framing applies here, but at the level of *which solution gets selected* rather than *what step direction is taken*.

### References (consult after the cold reconstruction)
- CS4787 Lecture 11 (Spring 2019) — overparameterized regime: `cs.cornell.edu/courses/cs4787/2019sp/notes/lecture11.pdf`
- Roger Grosse CSC421 notes — Grosse covers min-norm specifically; check his Lecture 7 or 8 slides
- Shalev-Shwartz & Ben-David, *Understanding Machine Learning*, Ch. 12–14 (generalization, learnability framing) — `cs.huji.ac.il/~shais/UnderstandingMachineLearning/`
- Your own Week 5 metric note for the inner-product framing

### Post-reconstruction review (10m)
Compare your pre-reconstruction answers to what you just derived. Note any that were wrong, vague, or surprised you. Specifically check: did you correctly predict that non-zero $w_0$ breaks the min-norm result? Did you spot the Euclidean-inner-product dependence?

### Output (last 10m)
Create note: **Week11_S1_ImplicitBias_Math**
- The row-space argument in 4–6 numbered steps
- The closed-form min-norm formula
- One paragraph: what "implicit bias" means in this context
- The non-zero initialization caveat
- The metric remark as a forward hook

---

## Session 2 (60–90m) — Code: the min-norm experiment

**Goal:** Demonstrate GD → min-norm numerically. Then probe the boundaries: what happens with non-zero init, with different optimizers, with different problem sizes? This is a small, clean experiment — no MLPs, no datasets, just numpy.

### Predictions before running anything (5m)

Write these down before any code runs:

1. With $n = 10$, $p = 100$, $w_0 = 0$, what do you expect $\|w_{\text{GD}}\| / \|w_{\text{min-norm}}\|$ to be? (Hint: see S1 step 5.)
2. With $w_0$ drawn from $\mathcal{N}(0, I)$, what do you expect that ratio to look like?
3. Does **momentum GD** still converge to min-norm from $w_0 = 0$? Argue from the form of the velocity update — is the update direction still a linear combination of rows of $X$?
4. Does **Adam** still converge to min-norm from $w_0 = 0$? Argue from its per-coordinate rescaling — does the row-space argument survive?

These predictions are the empirical version of S1's reconstruction. Run the experiment, then check.

### Work (45–60m)

Create `w11s2-code.py` — separate file from your prior weeks, following the per-week-file convention.

1. **Generate the problem.**
```python
import numpy as np
np.random.seed(0)
n, p = 10, 100
X = np.random.randn(n, p)
y = np.random.randn(n)
# closed-form min-norm solution
w_star = X.T @ np.linalg.solve(X @ X.T, y)
```

2. **Plain GD from zero.** Pick a sensible step size (e.g., $\alpha = 1 / \sigma_{\max}(X)^2$ or just $10^{-3}$ to start). Run for enough iterations to reach $L < 10^{-10}$. Compare $\|w\|$ and $\|w - w^*\|$ to your prediction.

3. **GD from random init.** Same step size, $w_0 \sim \mathcal{N}(0, 0.1^2 I)$. Compare $\|w_{\text{final}}\|$ and $\|w_{\text{final}} - w^*\|$. *And* check: does the null-space component of $w_0$ persist exactly in $w_{\text{final}}$? You can compute the null-space projection by $(I - X^T(XX^T)^{-1}X) w$.

4. **Momentum GD from zero.** Reuse your `MomentumSGD` class. Does it still find $w^*$? Why or why not, given your S2 prediction?

5. **Adam from zero.** Reuse your `Adam` class. Does it still find $w^*$? Look at $\|w_{\text{final}} - w^*\|$. If it doesn't match, can you explain why (per-coordinate rescaling = non-Euclidean preconditioning, which violates the orthogonality assumption in S1 step 4)?

6. **Optional stretch (if time):** vary $p$ at fixed $n$ — does the min-norm bias become more or less pronounced as overparameterization grows?

### Results table

Build a small table for your S2 note:

| Optimizer | $w_0$ | $\|w_{\text{final}}\|$ | $\|w_{\text{final}} - w^*\|$ | Loss at end |
|---|---|---|---|---|
| GD | zero | ? | ? | ? |
| GD | random | ? | ? | ? |
| Momentum | zero | ? | ? | ? |
| Adam | zero | ? | ? | ? |

### Output (last 10m)
Create note: **Week11_S2_ImplicitBias_Code**
- The results table
- One paragraph on which predictions held and which didn't
- A specific observation about Adam (this is the interesting one — your S1 metric remark predicts it shouldn't recover min-norm)

---

## Session 3 (60–90m) — Read: implicit regularization beyond the linear case

**Goal:** Understand what's known about implicit bias in deeper, non-linear settings — and, critically, what's *not* known. This is the session where the clean linear-regression story collides with the messy reality of deep learning.

### Pre-reading (5m, do now)

In your S3 note, before reading:

1. **Your hypothesis.** Does GD always prefer "simple" solutions? What does "simple" even mean when the model isn't linear?
2. **A prediction.** Will the Neyshabur et al. argument generalize cleanly from linear regression to deep networks, or will it require new ideas? Why?
3. **A connection.** How does the implicit-bias story relate to the Dinh et al. counterexample from Week 10? (Hint: Dinh et al. ruled out one *static* explanation; implicit bias is a *dynamic* candidate.)

### Reading plan — pick one primary, skim a second

**Primary option A (accessible, recommended for first pass):**
*Neyshabur, Tomioka, Srebro — "In Search of the Real Inductive Bias: On the Role of Implicit Regularization in Deep Learning"* (2014) — `arxiv.org/abs/1412.6614`
- Short paper (~10 pages). Argues empirically that the network architecture *plus the optimization algorithm* form the effective hypothesis class, and SGD's trajectory matters.
- Focus on: Sections 1–3 and the empirical setup. Don't get bogged down in the geometry-of-optimization formalism if it's not clicking.

**Primary option B (deeper, harder, more technically rewarding):**
*Soudry, Hoffer, Nacson, Gunasekar, Srebro — "The Implicit Bias of Gradient Descent on Separable Data"* (2018) — `arxiv.org/abs/1710.10345`
- Shows GD on logistic loss with separable data converges in direction to the max-margin (hard-margin SVM) solution. This is a *classification* analogue of the min-norm story.
- Focus on: the theorem statement (Section 3) and the proof sketch. The full proof is technical; understand the intuition first.

**Secondary skim (15m, optional):**
*Zhang, Bengio, Hardt, Recht, Vinyals — "Understanding Deep Learning Requires Rethinking Generalization"* (2017) — `arxiv.org/abs/1611.03530`
- The "memorizing random labels" paper. This is the empirical anchor for the entire implicit-bias research program: classical generalization theory can't explain why deep nets generalize, because the same networks can also memorize pure noise.
- This paper is on your Q1_Integration plan for this week. If you've already read it, skip; if not, the abstract + Section 1 + Section 2 is 10 minutes well spent.

**Recommendation:** Read Neyshabur et al. as the primary, skim Zhang et al. (it's the empirical setup that motivates all of this), and if you have energy left, look at the Soudry et al. theorem statement.

### Active reading (50–65m)

For your chosen primary paper:
1. Before each section, predict what the next claim will be based on the previous one.
2. Note every assumption: linearity? convexity? separability? specific loss function? initialization?
3. After each section, write one sentence in your own words on what was just proved.

For Zhang et al. (if you read it): the key question is "what does this paper *rule out*?" It rules out the hypothesis that the function class itself is what limits generalization. That's a negative result, which means the explanation has to come from somewhere else — i.e., the algorithm. That's the bridge to implicit bias.

### Output (last 10m)
Create note: **Week11_S3_Read_ImplicitBias**
- One paragraph: the main result of your primary paper
- A list of assumptions the result requires
- One question about what happens in non-linear (deep, non-separable) models — this is the open problem you should be aware of
- Updated answers to the pre-reading questions

---

## Session 4 (60–90m) — Write: "GD chooses among infinite solutions"

**Goal:** Produce a standalone explainer that pairs with your Week 10 landscape note. The pair tells a complete story: Week 10 ruled out a static explanation for generalization, Week 11 names a dynamic one.

### Target

~1200–1500 words. Five sections. One plot or table from S2. Cross-link the Week 10 note explicitly.

### Structure

1. **The setup: underdetermined systems have infinitely many solutions.** Set the stage: $p > n$ linear regression, the zero-loss affine subspace has dimension $p - n$, classical learning theory says you're doomed.

2. **GD from zero → min-norm: the row-space argument.** State and prove the result. Keep this tight — 3–4 paragraphs. The proof is short enough to include in full; do so.

3. **What this means: the optimizer is a regularizer.** This is the conceptual punchline. Without any explicit regularization term, the choice of *algorithm* + *initialization* picks out a specific solution from infinitely many equally good ones. The "inductive bias" of the learning system is the algorithm, not just the model.

4. **Caveats and dependencies.** This is where you earn intellectual credit. Cover:
   - Non-zero initialization changes the answer (your S2 experiment).
   - The result depends on the Euclidean inner product (Adam-from-zero doesn't recover min-norm — you've shown this).
   - For deep networks, the linear-regression argument doesn't apply. There are results for special cases (Soudry et al. for separable classification; matrix factorization has its own theory), but no general theorem for arbitrary deep nets.

5. **Forward: the open question.** The implicit-bias program is a research frontier. The Zhang et al. memorization result tells us we need *something* like implicit bias to explain why deep networks generalize. We know it exists empirically. We have partial theory for special cases. We don't have a unified theory. Be honest about this.

### Cross-links

In your Obsidian system, this note should connect to:
- `[[w10s3-read-landscape]]` — the static-geometry counterpart
- `[[w5s1-gd-euclidean]]` — the metric framing
- `[[w9s4-optimizer-cheat-sheet]]` — Adam's per-coordinate rescaling shows up as a deviation from min-norm in S2

### Output
Create note: **Week11_ImplicitBias_Explainer**

---

## Week 11 "done" checklist

- [ ] I can prove that GD from zero converges to the min-norm solution for underdetermined linear regression, without notes.
- [ ] I can state and explain what changes when $w_0 \neq 0$.
- [ ] My experiment confirms GD → min-norm numerically and shows the Adam deviation.
- [ ] I can explain "implicit bias" as a concept and why it matters for deep learning.
- [ ] I understand the Zhang et al. memorization result and why it motivates the implicit-bias program.
- [ ] My explainer note pairs cleanly with Week 10's landscape note.
- [ ] I can articulate the metric connection: orthogonal decomposition under the Euclidean inner product is what makes GD pick the L2-minimum solution, not (say) L1.

---

## Forward connections seeded this week

- **Week 12 (Convolution):** Conv is a *structural* inductive bias (locality, translation equivariance) baked into the model. Implicit bias is an inductive bias from the *algorithm*. Two flavors of the same idea — useful contrast for the Week 12 intro.
- **Week 13 (Attention):** Attention learns its inductive bias rather than hard-coding it. The metric-thread payoff is sharpest here, but you've now seen "metric choice picks the solution" in three places (steepest descent direction, optimizer geometry, implicit bias).
- **Q2 (Perception / representation learning):** Why representation learning works at all is largely an implicit-bias question. You'll come back to this.
- **Q3 (RL):** Policy gradient methods rely on parameterization choices that turn into implicit biases on the policy. The natural-gradient story (Amari, Kakade) is a direct continuation of the metric thread.

---

## Time budget estimate

| Session | Target time | Stretch time |
|---|---|---|
| S1 — row-space math | 60m | 90m |
| S2 — min-norm experiment | 60m | 90m |
| S3 — reading | 75m | 90m |
| S4 — explainer note | 75m | 90m |
| **Total** | **4h 30m** | **6h** |

S1 should be lighter than Week 10's S1 — the proof is short. If you finish S1 early, use the extra time to write out the metric remark carefully, since it's the link back to Week 5 and forward to Q3.

S2 is the most fun session this week. It's the first time in Q1 where the experiment is short enough to *fully* finish with multiple ablations. No MLP, no dataset wrangling — just numpy.

S3 has the most variance. Neyshabur is 10 pages, Soudry is 30+. Stick with Neyshabur unless you're well ahead.

S4 is the lightest writing session of the quarter so far, because the technical material is contained and your pair with Week 10 is already set up.
