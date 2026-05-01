---
created: 2026-04-20
title: Q1 · Week 10 · Session-by-Session Lesson Plan
tags: [lesson-plan, optimization, loss-landscape, generalization]
---

# Q1 · Week 10 · Session-by-Session Lesson Plan

## Context: Where You're Coming From

Week 9 closed cleanly. You derived Adam from first principles, built a full `Adam` class alongside `SGD` and `MomentumSGD`, ran ablations (bias correction on/off, epsilon sensitivity), and read the AdamW paper. The backprop explainer is at v0.7. No open carry-forward issues.

The through-line you've been building is **metric choice in optimization**: gradient descent uses the Euclidean metric implicitly (Week 5); momentum accumulates gradients in a way that implicitly reshapes the effective step direction (Weeks 6–7); Adam approximates a diagonal preconditioner — i.e., a parameter-wise metric rescaling (Week 9). All of these are choices about which directions to treat as "equivalent."

Week 10 pivots from *how optimizers move* to *what the terrain they move on looks like*. This is the first week where the objective is not to build a better optimizer but to understand why optimizers sometimes succeed despite the loss surface being non-convex and high-dimensional. It also sets up Week 11 (implicit bias) by introducing the overparameterized regime and Week 13 (attention) indirectly — once you've seen what a loss landscape is, you'll have the vocabulary for reasoning about why certain architectures train well.

---

## Week 10 outcome targets (ship by week's end)

- **Conceptual:** What the Hessian tells you about local geometry; what sharp and flat minima are *precisely*; why the flat-minima → generalization story is appealing but contested; what a 2D slice of a high-dimensional landscape can and cannot show you.
- **Math:** Second-order Taylor expansion and its eigenvalue analysis; filter-normalized direction construction (why you normalize per-filter, not per-network).
- **Code:** 1D and 2D loss slice plots using filter-normalized random directions. Run on your trained MLP from prior weeks. Compare: random init vs. trained; SGD-trained vs. Adam-trained.
- **Teaching artifact:** Interpretation note titled "Sharp and flat minima: what we know and don't know." Treat this as a genuinely scientific write-up — present the hypothesis, the evidence, and the Dinh et al. counterexample honestly.

The explainer (v0.7) does not need to grow this week unless you want to add a "landscape geometry" appendix. The primary writing artifact is the interpretation note.

---

## Session 1 (60–90m) — Math: Taylor expansion, the Hessian, and eigenvalue geometry

**Goal:** Formalize what "loss landscape shape at a point" means. Get the Hessian eigendecomposition fluent enough that you can read any paper that uses the phrase "flat minimum" and know what's being claimed.

### Prep (5m)

Before opening any reference: on a blank page, write the second-order Taylor expansion of $f(w + \Delta w)$ around $w$ from memory. Box the first-order term. Box the second-order term. Which term dominates near a minimum, and why?

### Work (55–75m)

**1. Second-order Taylor expansion (15m)**

Derive cleanly:
$$f(w + \Delta w) \approx f(w) + \nabla f(w)^\top \Delta w + \tfrac{1}{2} \Delta w^\top H \Delta w$$

Key observation: at a local minimum, $\nabla f(w^*) = 0$, so the local shape reduces to
$$f(w^* + \Delta w) - f(w^*) \approx \tfrac{1}{2} \Delta w^\top H \Delta w$$

This is a quadratic form. The Hessian $H$ is symmetric, so it has a full eigendecomposition $H = Q \Lambda Q^\top$. Writing $\Delta w = Q u$ in the eigenbasis gives
$$f(w^* + \Delta w) - f(w^*) \approx \tfrac{1}{2} \sum_i \lambda_i u_i^2$$

The landscape near a minimum is a sum of independent parabolas along the eigendirections. This is the **same decomposition** you used in Week 7 to analyze momentum per-direction — flag this connection explicitly in your notes.

**2. Eigenvalue interpretation (15m)**

- Large positive $\lambda_i$ → sharp curvature along that direction; small perturbations increase loss fast.
- Small positive $\lambda_i$ → flat direction; loss changes slowly.
- Zero $\lambda_i$ → degenerate; often reflects a symmetry (e.g., permutation symmetry between hidden units).
- Negative $\lambda_i$ → saddle point, not a minimum.

**Exercise:** If $H$ has eigenvalues $\{100, 10, 1, 0.01\}$, sketch what a 2D slice along the two *extremal* eigendirections would look like. Then sketch what a slice along the two *middle* eigendirections would look like. The visual contrast matters for S2.

**3. Defining sharp and flat minima (15m)**

Precise definitions to write down:
- A minimum is **sharp** if the largest Hessian eigenvalues are large — the loss basin is narrow along those directions.
- A minimum is **flat** if the Hessian eigenvalues are all small — the basin is broad.

The folk claim (Hochreiter & Schmidhuber 1997, later Keskar et al. 2017): flat minima generalize better because small perturbations to the weights don't meaningfully change the function. This has an MDL / Bayesian flavor — a flat minimum requires fewer bits to specify.

Write down — in your own words, before reading S3 — *why* you think this should or shouldn't be true. You'll revisit this in S3 after reading Dinh et al.

**4. The reparameterization problem (10m preview)**

Write a one-line note: *reparameterization changes flatness without changing the function*. Don't expand on this yet — it's the crux of Dinh et al. (2017), which is S3 reading. But seed the question: if I replace $w$ with $w' = c \cdot w$ for some constant $c$, what happens to the Hessian? What happens to the predictions?

### References

- GBC §8.2 — "Challenges in Neural Network Optimization" covers ill-conditioning, local minima, saddle points, plateaus, and cliffs. Skim for vocabulary.
- Boyd & Vandenberghe §9.1–9.2 — unconstrained second-order methods. You have this from earlier weeks.

### Output (last 10m)

Create note: **Week10_S1_Hessian_Landscape**

Include:
- The Taylor expansion, eigendecomposition, and per-direction parabola sum.
- Your written definitions of sharp and flat minima.
- Your pre-reading conjecture about the flat-minima → generalization claim.
- The reparameterization question as an open hook for S3.
- One sentence explicitly connecting this to the Week 7 per-direction momentum analysis.

---

## Session 2 (60–90m) — Code: loss slice visualization with filter-normalized directions

**Goal:** Produce 1D and 2D loss slice plots for your trained MLP. Use the filter-normalized direction construction from Li et al. so the plots are actually comparable across architectures and training runs.

### Prep (5m)

Find the trained model checkpoints from Weeks 8–9. You want at minimum:
- A model trained with `SGD`.
- A model trained with `Adam`.
- Ideally, both from the same initialization if you saved it, so the comparison is apples-to-apples. If not, train fresh models quickly at the top of the session.

### Work plan (60–75m)

**1. 1D loss slice (15m)**

The simplest possible landscape visualization. Pick a direction $d$ in parameter space, then evaluate
$$\phi(\alpha) = L(w^* + \alpha d)$$

for a range of $\alpha$ values (e.g., 50 points in $[-1, 1]$). Plot loss vs. $\alpha$.

Choice of direction matters:
- **Random direction:** see the "typical" local shape. Sample $d \sim \mathcal{N}(0, I)$ with the same shape as $w^*$.
- **Gradient direction:** see the steepest-ascent shape.
- **Direction between two minima:** if you have two trained models $w_A$ and $w_B$, the direction $d = w_B - w_A$ and $\alpha \in [0, 1]$ gives a linear interpolation — often more interesting than a random slice.

Start with a random direction. Don't filter-normalize yet.

**2. The scaling problem and filter normalization (20m)**

Run your random-direction slice. Observe that the shape is heavily influenced by the *magnitude* of $d$ relative to $w^*$. A weight tensor with small values will be dominated by the random direction; a large-valued tensor will barely move. The plot conflates curvature with scaling.

Li et al.'s fix (filter normalization): for each layer's filter (or weight tensor) $w_i$ and its corresponding direction component $d_i$, rescale:
$$d_i \leftarrow d_i \cdot \frac{\|w_i\|}{\|d_i\|}$$

This makes the direction magnitude match the parameter magnitude layer-by-layer. Implement this as a utility function and apply it before slicing.

Re-run the 1D slice with filter normalization. You should see a meaningfully different plot — the curvature should now reflect local geometry rather than random scaling artifacts.

**3. 2D loss slice (20m)**

Pick two filter-normalized random directions $d_1, d_2$. Evaluate
$$\phi(\alpha, \beta) = L(w^* + \alpha d_1 + \beta d_2)$$

on a grid (e.g., $25 \times 25$ for speed, or $50 \times 50$ if you're patient). Plot as a filled contour. Matplotlib's `contourf` is the right tool; `contour` overlaid on top gives you isolines.

Important practical note: if your loss is cross-entropy on a classification problem, use a log-scale colormap or take $\log(1 + L)$ before plotting. Otherwise the extreme values far from the minimum will flatten the entire visualization.

**4. Comparison runs (15m)**

Run 2D slices for:
- Random init (pre-training) — this is your "rough landscape" baseline.
- Trained with SGD.
- Trained with Adam.

Question to answer from the plots: does the SGD-trained minimum look flatter than the Adam-trained minimum? Be honest with yourself — this is one of the contested empirical claims in the literature, and your tiny MLP may or may not reproduce it. Noting "I cannot clearly distinguish these" is a legitimate and scientifically valuable answer.

### Output (last 10m)

Create notebook/file: **w10s2_landscape.py** (or `.marimo`).

Save plots:
- `w10_1d_slice_random.png` — naive random direction.
- `w10_1d_slice_filter_normalized.png` — after normalization; include in S4.
- `w10_2d_slice_sgd.png` and `w10_2d_slice_adam.png` — for S4 comparison.

Create note: **Week10_S2_LandscapeViz_Code**
- One paragraph on what filter normalization fixed (before vs. after comparison).
- One paragraph on the SGD vs. Adam comparison — what you saw, and what you can and can't claim from it.

---

## Session 3 (60–90m) — Read: landscape visualization and its critics

**Goal:** Understand what loss landscape visualizations can show and — critically — what they can't. Read Li et al. for the method and the headline findings; read Dinh et al. for the sharpest critique of the sharp/flat-minima → generalization story.

### Reading material

**Primary: Li et al. (2018), "Visualizing the Loss Landscape of Neural Nets"**
- `arxiv.org/abs/1712.09913`
- The paper that introduced filter-normalized directions. Key findings: skip connections dramatically smooth the loss landscape; wider networks have smoother landscapes; networks that generalize better tend to have flatter minima *in their visualization*.

**Primary: Dinh et al. (2017), "Sharp Minima Can Generalize For Deep Nets"**
- `arxiv.org/abs/1703.04933`
- The sharpest critique. Shows that for any flat minimum, you can reparameterize the network to make it arbitrarily sharp in Hessian terms — without changing the function it computes. This means "sharpness" (as measured by Hessian eigenvalues) is *not* a reparameterization-invariant property, and therefore cannot be the true mechanism behind generalization.

**Secondary: CS4787 Lecture 11 (Spring 2019), "Deep Neural Networks"**
- `cs.cornell.edu/courses/cs4787/2019sp/notes/lecture11.pdf`
- Bridges from convex optimization (where local = global and landscapes are well-behaved) into the overparameterized regime (where non-convexity dominates but training works anyway). Sets up Week 11.

### Active reading workflow

**Before reading (5m):** from S1, you wrote down your pre-reading conjecture about flat-minima → generalization. Do not edit it. List three things you think a 2D slice *cannot* show you about a 10,000-dimensional landscape.

**During reading Li et al. (25m):** focus on §4 (filter normalization derivation) and §6 (the skip-connection results). The ResNet vs. plain-network landscape comparisons in Figure 1 are the paper's most-shared visual — note what they do and don't prove.

**During reading Dinh et al. (20m):** the key result is Proposition 4 (scale-dependent reparameterization). Follow the argument: given a ReLU network, you can rescale layer $i$ by $c$ and layer $i+1$ by $1/c$ without changing the function, but the Hessian eigenvalues in parameter space change dramatically. Make sure you can explain this to yourself in one paragraph.

**During reading CS4787 L11 (15m):** focus on the overparameterization discussion and the connection to interpolation. Don't get pulled into the mean-field or NTK side discussions.

**After reading (5m):** revisit your pre-reading conjecture. What changed? Write down the revision, explicitly. This is for your own epistemic hygiene — tracking when you were wrong is how you learn to calibrate.

### Output (last 10m)

Create note: **Week10_S3_Read_Landscape**

Include:
- Your pre-reading conjecture, unedited.
- Your post-reading revision.
- One-sentence summary of the Li et al. method and its main empirical claim.
- One-sentence summary of the Dinh et al. counterargument.
- The three things a 2D slice can't show you (you wrote these before reading — did the readings change the list?).
- One forward hook for Week 11: what does the overparameterized regime have to do with implicit bias?

---

## Session 4 (60–90m) — Write: the sharp/flat minima interpretation note

**Goal:** Write a note that honestly presents what we know and don't know about landscape geometry and its relationship to generalization. This is your teaching artifact — write it as if you're going to hand it to someone who just finished Week 9 and is wondering why non-convex optimization ever works.

### Structure

The note should have five sections. Aim for ~1500 words total.

**1. Local geometry and the Hessian (250 words)**

- Taylor expansion and the per-direction parabola decomposition.
- Eigenvalues as curvature along principal directions.
- Why, at a minimum, first-order information vanishes and second-order dominates.
- Include a small diagram or reference to your 2D slice from S2 showing a contour plot with the two principal directions highlighted.

**2. Defining sharp and flat (200 words)**

- Precise definitions in terms of Hessian eigenvalues (or the spectral norm / trace as common proxies).
- The intuitive picture: a narrow basin vs. a wide basin.
- The information-theoretic / Bayesian-flavored appeal: flat minima encode less information, should generalize better.

**3. The empirical evidence (300 words)**

- Li et al.'s visualizations: skip connections flatten landscapes; generalization seems to correlate with flatness in their experiments.
- Keskar et al. (2017) on large-batch training producing sharper minima that generalize worse (reference this without needing to read it in full — Li et al. cite it).
- Your own S2 results: what did you see comparing SGD vs. Adam? Present the plots. Be specific about what they do and don't show on your tiny MLP.

**4. The counterargument: Dinh et al. (300 words)**

- The reparameterization result: Hessian-based sharpness is not invariant under function-preserving reparameterizations.
- Why this is a serious problem for the flat-minima → generalization hypothesis *as a causal claim*.
- What it does and doesn't invalidate: the empirical correlation between (some measures of) flatness and generalization may be real; the simple mechanistic story is not.

**5. What a 2D slice can and cannot tell you (250 words)**

- The three (or more) things from S3 that a 2D slice can't show: volume, connectivity between minima, direction-dependent structure that's averaged out by random slicing, etc.
- The role of filter normalization: necessary but not sufficient.
- The honest summary: landscape visualizations are useful *intuition pumps*, not *explanations*.

**6. Forward connection (100 words)**

One short paragraph: how does this set up Week 11? The overparameterized regime (where $p \gg n$) changes the picture — minima are not isolated points but extended manifolds, and *which* point on the manifold gradient descent selects is the implicit bias question.

### Output

- Note: **Week10_Landscape_Interpretation**
- Embed `w10_1d_slice_filter_normalized.png` in §1 or §2.
- Embed `w10_2d_slice_sgd.png` and `w10_2d_slice_adam.png` (side by side) in §3.

### Optional: add a landscape appendix to the backprop explainer

If you have time and energy, add a half-page appendix to `backprop_explainer-v0.7.md` called "Appendix: what does the landscape look like?" This is not required. The explainer is about *computing* gradients; the landscape content is ancillary. Only add it if you can do so without bloating the explainer's main narrative.

---

## Week 10 "done" checklist

- [ ] I can state the second-order Taylor expansion and explain what each term does at a minimum.
- [ ] I can explain the Hessian eigenvalue → per-direction curvature story in a single clean paragraph.
- [ ] I have a working 1D and 2D loss slice visualization using filter-normalized directions.
- [ ] I have comparison plots between SGD-trained and Adam-trained minima, and I can say honestly what they do and don't show.
- [ ] I can explain Dinh et al.'s reparameterization argument in one paragraph without notes.
- [ ] My interpretation note presents both the flat-minima hypothesis and its critique fairly.
- [ ] I have a forward hook ready for Week 11 on the overparameterized regime.

---

## Forward connections seeded this week

- **Week 11 (Implicit bias of GD):** The overparameterized regime from CS4787 L11 sets up the question — when $p > n$ and there are infinitely many zero-loss solutions, which one does GD pick? Landscape geometry is part of the answer (the minima form a connected manifold, not isolated points) but not the whole answer.
- **Week 12 (Convolution):** Li et al.'s result that skip connections smooth the landscape is a preview of why architectural choices matter for optimizability, not just expressivity.
- **Week 13 (Attention):** No direct connection, but the general lesson — that representational and geometric choices affect training dynamics — carries over.
- **Q2 (Generalization and representation learning):** the flat-minima debate is one entry point into the larger question of what makes a deep network generalize. You'll revisit it.

---

## Time budget estimate

| Session | Target time | Stretch time |
|---|---|---|
| S1 — Hessian math | 60m | 90m |
| S2 — slice code | 75m | 90m |
| S3 — reading | 75m | 90m |
| S4 — interpretation note | 75m | 90m |
| **Total** | **4h 45m** | **6h** |

S2 is the session most likely to overrun. If you're running long, cut the 2D SGD-vs-Adam comparison and keep just a single trained-model 2D slice — you can still write a meaningful S4 note from one clean plot.
