---
created: 2026-01-23 17:31
title: 
draft: 
tags: 
aliases: 
description:
---
## Week 2 outcome targets (ship by week’s end)
- **Derivation:** Clean, checked derivation of backprop for a **2-layer MLP** (explicit Jacobians / shapes).
- **Code:** micrograd-style engine supports enough ops to train a tiny MLP on **XOR** (scalar core, plus a thin vector wrapper or tiny matrix helper).
- **Reading notes:** 1 page of “core idea in my notation” + 5 high-quality questions.
- **Teaching artifact:** 1–2 page explainer: “Backprop as computation graph + local Jacobians” with one diagram.

---

## Session 1 (60–90m) — Math & derivations: backprop for a 2-layer MLP
**Goal:** Make backprop feel inevitable: local derivatives + chain rule bookkeeping.

### Prep (5m)
Pick the exact model to derive (write shapes everywhere):
- $x \in \mathbb{R}^d$
- $z_1 = W_1 x + b_1$, $a_1 = \mathrm{ReLU}(z_1)$
- $z_2 = W_2 a_1 + b_2$, $\hat{y} = \sigma(z_2)$ (or identity for regression)

### Work (55–75m)
1. **Write shapes everywhere** (prevents most mistakes).
2. Derive in reverse order:
   - $\frac{\partial \mathcal{L}}{\partial z_2}$ (for your chosen loss)
   - $\frac{\partial \mathcal{L}}{\partial W_2}$, $\frac{\partial \mathcal{L}}{\partial b_2}$
   - Backprop signal: $\delta_1 = \frac{\partial \mathcal{L}}{\partial z_1} = (W_2^\top \delta_2) \odot \mathrm{ReLU}'(z_1)$
   - Then $\frac{\partial \mathcal{L}}{\partial W_1}$, $\frac{\partial \mathcal{L}}{\partial b_1}$
3. **Sanity checks**
   - Dimensions match at every step
   - If ReLU is “off” (negative), gradients to that unit go to $0$
   - Numerical spot-check on one tiny example (e.g., $d=2$, hidden=$2$)

### Output (last 10m)
Create note: **Week2_S1_MLP_Backprop_Derivation**
- Box the final gradients
- Add a short “intuition” section (2–4 bullets)

---

## Session 2 (60–90m) — Code from scratch: micrograd → tiny MLP that learns XOR
**Goal:** Turn the derivation into a working training loop (even if tiny/ugly).

### Prep (5m)
Choose Week 2 scope:
- **Option A:** Scalar autograd core; build networks out of scalars (simplest).
- **Option B:** Add a thin Vector/Tensor convenience layer (arrays of Values) for dot products.

### Build plan (60–75m)
1. **Core autograd completeness**
   - Ensure ops: add, mul, pow, neg/sub, div, exp, log, tanh *or* sigmoid, relu
   - Topological sort + stable backward pass
2. **MLP module**
   - `Neuron` (weights = list of Values, bias = Value)
   - `Layer` (list of Neurons)
   - `MLP` (stack of Layers)
3. **Loss + training loop**
   - XOR dataset (4 points)
   - Loss: MSE or BCE (match Session 1 choice)
   - SGD update with a fixed learning rate
4. **One “trust but verify” test**
   - Finite-difference gradient check on one parameter (pick one weight)

### Output (last 10m)
Commit code + short README note:
- “What works / what’s hacky / what I’ll improve in Week 3”

---

## Session 3 (60–90m) — Read actively: backprop / autodiff as computation
**Goal:** Convert someone else’s explanation into *your* notation and mental model.

### Pick one reading target
- Backprop/autodiff chapter/section you already trust (UDL notes, autodiff overview, micrograd-style explanation, etc.)

### Active reading workflow (60–75m)
1. Write a **5-line summary from memory** *before* reading (what you think it says).
2. Read with one purpose: **extract invariants**
   - What must be true for reverse-mode to work?
   - Where do Jacobians appear implicitly?
3. Produce:
   - A rewritten derivation snippet in your notation (½ page)
   - **5 questions** that expose gaps (e.g., “What changes if outputs are vector-valued?”)

### Output (last 10m)
Create note: **Week2_S3_Read_Backprop_Autodiff**
- Include your 5 questions at the bottom (seed Week 3 prompts)

---

## Session 4 (60–90m) — Write/teach: “Backprop explained to my future self” (v0.1)
**Goal:** Start the quarter’s main write-up early; iterate weekly.

### Draft structure (fill during session)
1. **What backprop is** (1 paragraph): reverse-mode on a DAG
2. **What it computes** (3 bullets): gradients of scalar loss wrt many parameters efficiently
3. **How it works** (diagram + 5–10 bullets)
   - forward pass caches
   - local derivatives
   - reverse accumulation along edges
4. **One concrete worked example**
   - Use the exact 2-layer MLP from Session 1
   - Show the $\delta$ recursion and where ReLU gates gradients

### Output
- 1–2 page draft + one simple diagram (box-and-arrows is fine)

---

## Optional Session 5 (45–75m) — Non-technical: philosophy/history note tied to ML practice
**Goal:** 5–10 bullets connecting “how knowledge is built” to ML culture.

Pick one lens:
- **Popper:** falsifiability ↔ evaluation/benchmarks (what does it mean to “refute” a model claim?)
- **Kuhn:** paradigm shifts ↔ architecture shifts (CNN→Transformer), and what gets ignored
- **Lakatos:** research programs ↔ “scaling laws” progress vs brittle anomalies

**Output:** note: **Week2_Philosophy_Link_to_ML**

---

## Week 2 “done” checklist
- [ ] I can re-derive $\delta_1 = (W_2^\top\delta_2)\odot \mathrm{ReLU}'(z_1)$ without looking.
- [ ] My micrograd trains XOR to low loss (even if slow).
- [ ] I wrote 5 sharp reading questions.
- [ ] I produced a v0.1 explainer doc (not polished—exists).
