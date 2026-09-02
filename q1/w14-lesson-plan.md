---
created: 2026-08-18T00:00:00Z
id: 019ffb8c-4a10-7d21-b3e5-6c9a2f4e8d17
title: Q1 · Week 14 · Session-by-Session Lesson Plan
---

# Q1 · Week 14 · Session-by-Session Lesson Plan

## Context: Where You're Coming From

Week 13 closed clean across all four sessions. You derived scaled dot-product attention with the $\sqrt{d_k}$ variance argument, implemented stabilized softmax and masked attention in NumPy, ran the temperature-threshold and scaling ablations with multi-seed averaging, read Vaswani §1–3 with pre/post conjectures logged, and shipped `Week13_S4_Attention_Explainer.md` covering all six sections. The metric thread reached its intended terminus: attention as a *learned* bilinear form, $S_{ij} = x_i^\top (W_Q^\top W_K) x_j$, which is the same object you first met as a choice of inner product in Week 4.

Two things from Week 13 shape this week specifically.

First, the **label-check discipline** is now standing practice, instituted after a two-week pattern: correct index arithmetic paired with inverted labels (equivariance/invariance in Week 12, asymmetry attributed to $A$ rather than $S$ in Week 13 S1 and again in S3). The pattern is diagnostic — your derivations are sound and your vocabulary retrieval is not. That distinction matters for how you should self-test this week, and it's the reason S2 below is structured the way it is.

Second, you finish Q1 with **real debts on the books**. The Week 10 Adam comparison plot was never generated (`w10_trained_models.npz` has the trained weights; `loss_slice_2d` needs to run on `flat_adam`). The condition-number reparameterization question is parked. The vault has naming drift. Wrap week exists to either close these or consciously discharge them — carrying them silently into Q2 is the failure mode.

**A structural note on this week:** it deviates from the S1 math → S2 code → S3 read → S4 write cadence, because there's no new material. The original integrated plan ordered it cleanup → synthesis → self-test (optional) → Q2 planning. I'd argue for **swapping the self-test ahead of the synthesis writeup**, for two reasons: an optional last-third session is the one that gets skipped when the week runs long, and it's exactly the session with the highest information value at a quarter boundary. More importantly, the synthesis writeup is more honest when it's written *after* you find out what you can actually reconstruct cold. Otherwise you'll write the arc from your notes, and the notes will make it look like you know things you last knew in May.

Take the swap or leave it, but decide deliberately rather than by default.

---

## Week 14 outcome targets (ship by week's end)

- **Repo:** Every mini-implementation runs from a clean checkout. Three regression tests pass. A README that a stranger — or you in October — could follow to reproduce the main results.
- **Calibration:** An honest, closed-book measurement of what survived from Weeks 1–13, with named gaps.
- **Synthesis:** A 2–3 page writeup telling the backprop → geometry → optimization → generalization → building blocks arc, with the metric thread as its spine and a "what I got wrong" section that doesn't flinch.
- **Forward:** A Q2 launch note with concrete questions and a resource map — plus one strategic decision made explicitly (see S4).

---

## Session 1 (75–90m) — Repo + vault cleanup, and close the Week 10 debt

**Goal:** The codebase and the vault stop being a pile of session artifacts and become a thing you can navigate.

### Code inventory (15m)

Walk the tree and sort every file into one of three buckets: **keep and test**, **keep as archive** (it ran once, it made a plot, it's not load-bearing), **delete**. The `Value` engine, the optimizer classes, and the two mini-implementations are the load-bearing pieces; most of the `wNsM.py` marimo notebooks are archive.

Proposed structure:

```
mini/          conv1d.py, attention.py          (standalone, grad-checked)
engine/        value.py, optimizers.py, nn.py    (the autograd core + numpy interface)
experiments/   w10s2.py, w11s2.py, w12s2.py …    (marimo notebooks, archive)
tests/         test_gradcheck.py                  (the three tests below)
README.md
```

### The three tests (25m)

Not comprehensive coverage — a tripwire. If these pass, the core is intact:

1. **`Value` engine gradcheck** — finite differences against a small composite expression exercising every op you implemented.
2. **`conv1d_backward` gradcheck** — both $\nabla_x L$ and $\nabla_k L$, relative error $< 10^{-5}$ (this is already written; it just needs to live in `tests/`).
3. **Attention invariants** — attention rows sum to 1; masked positions contribute exactly zero; and the shape assertion $A \in \mathbb{R}^{n_q \times n_k}$. Assert the shape explicitly. That's the invariant you've slipped on more than once, and a test is a cheaper corrective than a flashcard.

### Close or discharge the Week 10 Adam plot (20m, hard-capped)

Load `w10_trained_models.npz`, run `loss_slice_2d` on `flat_adam`, produce the comparison against the SGD/momentum slices. **Time-box this at 20 minutes.** If the infrastructure has drifted and it doesn't come back cleanly, delete the debt: write one line in the Week 10 note saying the Adam slice was never generated and why, and move on. A stale TODO carried into Q2 costs more attention than the plot is worth.

### Vault hygiene (20m)

- **Naming drift.** The current vault contains one copy of each Week 11 artifact, using lowercase kebab-case (for example, `w11s1-implicit-bias-math.md`). Treat that as canonical and update stale TitleCase wiki-links rather than creating duplicate aliases.
- **Missing extensions.** Current audit found no extensionless files, so there is nothing to rename. If `Q1_revised_plan_weeks_7-14` or `w13s3-reading` reappears, rename it with its intended `.md` extension before linking to it.
- **Dangling links.** The Week 12 parameterized-linear-map placeholder resolves to `[[w4s1-gradients-levelsets]]`; update stale artifact links to their canonical filenames.
- **Build `Q1_Index`** — a map-of-content note linking all fourteen weeks by topic, with the four artifact types (S1 math / S2 code / S3 reading / S4 explainer) as columns. This is what you'll navigate from in Q2 instead of scrolling the file list.

### One decision to make and record

**Backprop Explainer v1.0.** It's been sitting at v0.8 since Week 11. Do the conv and attention explainers fold in as chapters, or does the explainer stay a backprop-mechanics document with conv/attention as standalone siblings? The Week 12 plan explicitly deferred this to now. Either answer is defensible — the argument for keeping it narrow is that "backprop explainer" that also covers architecture is really "my Q1 notes" wearing a misleading title. Decide, write the decision down, don't relitigate it in Q2.

### Output
Update: **`README.md`**, create **`Q1_Index`**

---

## Session 2 (75–90m) — Cold self-test

**Goal:** Measure what actually survived. Closed book, timed, graded afterward against your notes. Treat gaps as calibration anchors, exactly as in every S1 this quarter — the point is not to score well.

### Part A — The derivation challenge (40m)

Starting from $f(x) = \tfrac{1}{2} x^\top A x - b^\top x$ with $A \succ 0$:

1. Derive the GD update and its convergence rate in terms of $\kappa(A)$.
2. Derive the heavy-ball momentum update. State what it does to the effective rate and why the EMA framing explains it.
3. Derive the Adam update including bias correction. State precisely what "diagonal preconditioning" buys you on this quadratic, and what it *doesn't* — you have a concrete result to check yourself against here, since the Week 11 experiment showed **equalization, not concentration**, refuting both your prediction and mine.
4. The unifying claim: each optimizer is a choice of metric. Write $\Delta x = -\eta M^{-1} \nabla f$ and identify $M$ for each of the three.

### Part B — The shape and label gauntlet (25m)

Short-answer, targeting your two confirmed failure modes. No notes.

- **Shapes.** Given $Q \in \mathbb{R}^{n_q \times d_k}$, $K \in \mathbb{R}^{n_k \times d_k}$, $V \in \mathbb{R}^{n_k \times d_v}$: shapes of $S$, $A$, and the output. State the *reason* $A$'s shape cannot contain $d_v$, don't just write the answer.
- **Conv.** Output length for input $n$, filter $m$, padding $p$, stride $s$ — derived, not recalled. Then the index bounds of the backward pass accumulation loop for $\nabla_x L$.
- **Labels.** Convolution is translation-\_\_\_\_\_\_\_\_\_ (equivariant or invariant?) — and derive the expression *first*, then read the label off it. Which matrix is asymmetric in general, $S$ or $A$, and are those asymmetries independent?
- **Subspaces.** For GD from zero init on an underdetermined system, which subspace does the iterate live in, and which component is frozen?

### Part C — Grade and log (15m)

Score each item hit / partial / miss. For every miss, write one line: *what I said, what's correct, why I slipped*. This log feeds directly into S3's calibration section — and the misses are the most valuable content in the entire synthesis writeup.

**Alternative:** if you'd rather test breadth than depth, CS231n Assignment 1 (kNN, SVM, Softmax, 2-layer net from scratch) is the original suggestion. I'd steer you to the derivation challenge instead — A1 mostly re-tests implementation skills you've demonstrated repeatedly, whereas the failure mode you've actually exhibited is retrieval under cold conditions.

### Output
Create note: **`Week14_S2_Q1_SelfTest`**

---

## Session 3 (90m) — Synthesis writeup

**Goal:** The document that makes Q1 legible as one argument rather than fourteen weeks of sessions.

### Structure

1. **Backprop (Weeks 1–3)** — reverse-mode autodiff, VJPs, the computational graph as the object that makes gradients cheap.
2. **Geometry (Weeks 4–5)** — gradients are covectors; steepest descent is only defined relative to a metric; condition number and the ravine picture.
3. **Optimization (Weeks 6–9)** — momentum as EMA, Adam as diagonal preconditioning, AdamW's decoupling; every optimizer as a metric choice.
4. **Generalization (Weeks 10–11)** — loss landscape slices and their pitfalls, implicit bias, what the optimizer selects for free among infinitely many solutions.
5. **Building blocks (Weeks 12–13)** — conv as a fixed banded linear map, attention as a learned one; hard-coded vs. learned connectivity.
6. **The thread** — the inner product appears in defining the gradient, in choosing the descent direction, in adaptive preconditioning, and in attention similarity. State it as one claim and support it with the four instances.

### Two constraints on the writing

**Write section 6 first.** If the thread doesn't hold up when you state it plainly, you want to know that before you've written five sections leading to it.

**Include a calibration section.** Pull directly from S2's log plus the running misses: the Adam equalization surprise, the multi-head FLOP conjecture, the equivariance/invariance inversion, the $(n_q, d_v)$ shape slip, the single-seed ablation correction. Mark anything you couldn't reconstruct cold in S2 as an open gap in plain language rather than smoothing it over from notes. This section is the one that's genuinely hard to write and the one that will be worth reading in six months.

Keep to 2–3 pages. The compression is the exercise — if it runs to eight pages you've written a summary rather than a synthesis.

### Output
Create note: **`Q1_Synthesis`**

---

## Session 4 (75m) — Q2 launch: resource map + open questions

**Goal:** Start Q2 from a plan rather than from a blank Monday.

### The strategic decision (20m — do this first)

**Does Q2 stay in pure NumPy, or move to PyTorch?**

Q1's constraint — build everything from scratch — was the right call and it worked. Q2 is perception and representation learning, which means real image data, multi-layer CNNs, and eventually ViT. A pure-NumPy CNN on CIFAR is a multi-week engineering project that teaches you comparatively little you didn't learn writing `conv1d_backward`.

The defensible middle: keep the from-scratch discipline for the *first* instance of each new mechanism (write the 2D conv forward/backward once, grad-check it, then never again), and use PyTorch for anything that's scale rather than concept. Write down which side of that line each Q2 topic falls on now, while you're thinking about it structurally, rather than deciding it under time pressure in Week 3 of Q2.

### Open questions to carry (25m)

Already banked, needing sharpening:

- **2D positional encoding / ViT** — does the linear-offset structure of sinusoidal PE survive two axes, or does ViT just learn the encoding? (Week 13 S3)
- **Multi-head ≈ conv channels** — does the analogy hold, or does it break once you look at what each actually mixes over? (Week 13 S3)
- **Residual connections** ↔ loss-landscape smoothing — the backward link to `[[w10s3-read-landscape]]`, still unexplored. (Week 13 S3)
- **Condition number under reparameterization** — does $\kappa$ survive attacks more sophisticated than ReLU rescaling? Parked since Week 10; Q2's generalization thread is where it belongs. Either schedule it or formally retire it.
- **Hierarchical features** — how do CNNs actually build them, and what does the metric story say about a *stack* of layers rather than one? (Week 12 forward pointer)
- **Representation-learning objectives** — where do contrastive and generative objectives sit relative to the optimization story? Do they change the implicit bias?

Aim for 5–10 questions total, each specific enough that you could tell whether a week's work answered it.

### Resource map (30m)

- **CS231n** is the Q2 spine — you've used the notes piecemeal for Weeks 8, 9, and 12; map its lecture sequence onto a Q2 week plan.
- **CS224n** leftovers (Lecture 9: pretraining, GPT/BERT) if Q2 includes a transformer-for-vision arc.
- **CMU 10-703** stays flagged for Q3, not now.
- Identify the **one paper per topic** you'd read, not a reading list. Q1's papers worked because there were five of them across fourteen weeks.

### Output
Create note: **`Q2_Launch_Questions`**

---

## Week 14 "done" checklist

- [ ] Every mini-implementation runs from a clean checkout; three regression tests pass.
- [ ] README documents how to reproduce the main results.
- [ ] The Week 10 Adam slice is either generated or formally discharged in writing.
- [ ] Vault duplicates merged, extensions fixed, dangling links resolved, `Q1_Index` exists.
- [ ] Backprop Explainer v1.0 scope decision made and recorded.
- [ ] Self-test completed closed-book and graded, with a miss log.
- [ ] `Q1_Synthesis` states the metric thread as one claim and names the open gaps honestly.
- [ ] `Q2_Launch_Questions` has 5–10 concrete questions and the NumPy-vs-PyTorch decision written down.

---

## Time budget estimate

| Session | Target | Stretch |
|---|---|---|
| S1 — Repo + vault cleanup, Week 10 debt | 80m | 100m |
| S2 — Cold self-test | 80m | 90m |
| S3 — Synthesis writeup | 90m | 120m |
| S4 — Q2 launch | 75m | 90m |
| **Total** | **5h 25m** | **6h 40m** |

S1 is the session most likely to sprawl, because cleanup has no natural stopping point. The 20-minute cap on the Adam plot is the main defense; apply the same instinct to the vault work — deduplicate and index, don't reformat every note.

S3 is the one worth the stretch time if you have it. It's the artifact with the longest half-life.

---

## Forward connections seeded this week

- **Q2 (Perception / representation learning):** `Q2_Launch_Questions` is the input to the Q2 lesson plan. The NumPy-vs-PyTorch line drawn in S4 determines how the Q2 S2 sessions get scoped.
- **The metric thread continues:** it doesn't end at attention. Contrastive objectives are explicitly about learning a metric on the representation space — which makes Q2 the natural sequel rather than a fresh start.
- **Retired or carried:** whatever you decide about $\kappa$-under-reparameterization in S4, that's the last week it gets to be ambiguous.
