---
created: 2026-01-23 17:21
title: 
draft: 
tags: 
aliases: 
description:
---
# Q1 · Week 3 · Session-by-Session Lesson Plan

## Week 3 outcome targets (ship by week’s end)
- **Derivation:** Backprop for a 2-layer MLP **with mini-batch** and **cross-entropy** (choose either sigmoid+BCE or softmax+CE), with clean $\delta$ notation.
- **Code:** Your micrograd/mini-NN trains:
  - XOR **reliably**, and
  - one tiny **2-class** dataset (e.g., linearly separable points) using BCE/CE.
- **Verification:** 1 gradient-check harness (finite differences) that can test any parameter in your model.
- **Teaching artifact:** Add a new section to your explainer: **“VJP vs full Jacobian”** + one diagram showing derivatives “living on edges.”

---

## Session 1 (60–90m) — Math & derivations: loss layers + $\delta$ notation (the “error” story)
**Goal:** Make $\delta$ feel like the natural currency of backprop, not a random symbol.

### Prep (5m)
Pick one route (don’t do both this week):
- **Route A (binary):** $\hat{y}=\sigma(z)$ with **binary cross-entropy** $L(y,\hat{y})$  
- **Route B (multi-class):** $\hat{y}=\mathrm{softmax}(z)$ with **cross-entropy** $L(y,\hat{y})$

### Work (55–75m)
1. Write the computational chain explicitly:
   - $z_2 \to \hat{y} \to L$
2. Derive the key simplification:
   - Route A typically yields: $\delta_2 := \frac{\partial L}{\partial z_2}$ in a compact form (often $\hat{y}-y$ under common conventions).
   - Route B similarly yields: $\delta_2 = \hat{y}-y$ when $y$ is one-hot and $L$ is CE.
3. Backprop into the hidden layer:
   - $\delta_1 = (W_2^\top \delta_2)\odot f'(z_1)$ where $f$ is ReLU/tanh/etc.
4. Mini-batch clarity:
   - Show how gradients sum/average across batch: $\nabla W = \frac{1}{B}\sum_{i=1}^B \nabla W^{(i)}$
5. Shape sanity check for every line.

### Output (last 10m)
Create note: **Week3_S1_LossLayers_DeltaNotation**
- Box: $\delta_2$ formula
- Box: $\delta_1 = (W_2^\top\delta_2)\odot f'(z_1)$
- 3 bullets: “What $\delta$ *means*”

---

## Session 2 (60–90m) — Code from scratch: add BCE/CE + batch training + gradient check
**Goal:** Your code stops being “it kinda works” and starts being *trustworthy*.

### Work plan (60–75m)
1. Implement the loss route you chose:
   - Route A: sigmoid + BCE
   - Route B: softmax + CE (softmax can be implemented stably via log-sum-exp)
2. Add mini-batch support (minimal version):
   - Either loop samples and accumulate grads, or average loss over batch then backprop once (depending on your design).
3. Add a **finite-difference gradient check** utility:
   - Pick one parameter $p$
   - Approx: $\frac{\partial L}{\partial p} \approx \frac{L(p+\epsilon)-L(p-\epsilon)}{2\epsilon}$
   - Compare to autograd gradient; print relative error
4. “Torture test”:
   - Run grad-check on 2–3 randomly chosen params each run (small network, small data).

### Output (last 10m)
- Commit code.
- Note: **Week3_S2_Code_Notes**
  - “What I implemented”
  - “Known limitations”
  - “Next obvious refactor”

---

## Session 3 (60–90m) — Read actively: VJP, Jacobians, and “derivatives on edges”
**Goal:** Lock in the mental model: reverse-mode computes **vector–Jacobian products** (VJPs) without building giant Jacobians.

### Active reading workflow (60–75m)
1. Before reading: write a 5-line “what I believe VJP is.”
2. During reading: extract *invariants*:
   - reverse-mode starts from $\bar{L} = \frac{\partial L}{\partial L}=1$
   - each node pulls back an upstream vector via a local Jacobian transpose (conceptually)
3. After reading: rewrite one section in your notation:
   - Emphasize: you never need $J$ explicitly; you need $v^\top J$.

### Output (last 10m)
Create note: **Week3_S3_Read_VJP_Edges**
- 5 bullet “invariants”
- 1 diagram (even ASCII is fine)
- 5 questions for Week 4 (e.g., “How does this change with multiple outputs?”)

---

## Session 4 (60–90m) — Write/teach: update “Backprop explained to my future self” (v0.2)
**Goal:** Upgrade the explainer from “works” to “teaches.”

### Add these sections (aim for clarity, not length)
1. **$\delta$ notation** (what it tracks; why it’s convenient)
2. **Edges carry derivatives**
   - Show each edge has a local derivative; nodes cache values from forward pass
3. **VJP vs full Jacobian**
   - One paragraph: why reverse-mode is efficient for scalar loss

### Output
- Updated doc: **Backprop_Explainer_v0.2**
- Include one worked example using your Week 3 loss (BCE/CE)

---

## Optional Session 5 (45–75m) — Non-technical: “How knowledge is built” note #1
**Goal:** One tight note connecting philosophy/history to ML practice.

Pick one:
- **Popper:** what counts as a “falsifiable claim” in ML? (benchmarks, ablations, replication)
- **Kuhn:** when does “normal science” show up in ML? (incremental gains, shared toolchains)
- **Lakatos:** what’s the “hard core” vs “protective belt” in deep learning research programs?

### Output
Note: **Week3_Philosophy_ML_Culture_1** (5–10 bullets)

---

## Week 3 “done” checklist
- [ ] I can derive $\delta_2$ for my chosen loss (BCE or CE) cleanly.
- [ ] My training loop supports batches (even if simple).
- [ ] I ran a finite-difference gradient check and it passed for a few params.
- [ ] My explainer now includes $\delta$, edges, and a VJP paragraph.
