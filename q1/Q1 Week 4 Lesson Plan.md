---
created: 2026-01-30 15:45
title: 
draft: 
tags: 
aliases: 
description:
---
# Q1 · Week 4 · Session-by-Session Lesson Plan

## Week 4 outcome targets (ship by week’s end)
- **Conceptual mastery:** A rock-solid understanding of **why gradients are covectors**, how inner products let us *represent* them as vectors, and where that choice quietly enters ML code.
- **Math:** Clear derivation of gradients as **normals to level sets** and their role in optimization.
- **Code:** Extend your framework or experiments to *observe* gradient geometry (directions, magnitudes, saturation).
- **Teaching artifact:** A polished section in your explainer: **“Gradients as covectors, vectors by choice”** with one geometric diagram.

---

## Session 1 (60–90m) — Math & geometry: gradients, covectors, and level sets
**Goal:** Internalize what a gradient *is*, not just how to compute it.

### Prep (5m)
Pick a simple scalar function:
- $f(x,y) = x^2 + y^2$
- or $f(x,y) = x^2 + 2y^2$

### Work (55–75m)
1. **Derivative as a linear map**
   - Write $df_x : \mathbb{R}^n \to \mathbb{R}$
   - Emphasize: this is a **covector**, not a vector
2. **Level sets**
   - Define: $\{x \mid f(x) = c\}$
   - Show that $df_x(v)=0$ for all tangent vectors $v$ to the level set
3. **Normals**
   - Show the gradient is normal to the level set
   - Explain why “steepest ascent” follows the gradient
4. **Inner product choice**
   - Show how an inner product lets us *identify* a covector with a vector
   - Note: this identification is a **choice**, not a fact of nature

### Output (last 10m)
Create note: **Week4_S1_Gradients_LevelSets**
- One worked 2D example
- One short paragraph: “Why gradients aren’t really vectors”

---

## Session 2 (60–90m) — Code & experiments: gradient geometry in practice
**Goal:** See gradient geometry show up in real training behavior.

### Work plan (60–75m)
1. Pick a tiny model you already have:
   - Linear regression
   - Logistic regression
   - 1-hidden-layer MLP
2. Instrument your training loop to log:
   - Gradient norms $\|\nabla W\|$
   - Parameter update magnitudes
3. Run 2–3 experiments:
   - Different learning rates
   - Different initializations
   - Optional: rescale inputs and observe gradient scaling

### What to observe
- Exploding vs vanishing gradients
- How gradient *direction* stays meaningful even when magnitude changes
- How saturation (e.g., sigmoid) flattens level sets

### Output (last 10m)
Create note: **Week4_S2_GradientGeometry_Code**
- 3 observations tied back to geometry
- One “this surprised me” bullet

---

## Session 3 (60–90m) — Read actively: gradients, metrics, and optimization bias
**Goal:** Connect geometry to real optimization behavior.

### Reading focus (pick one angle)
- Gradients as covectors and metric choice
- Why SGD implicitly prefers certain solutions
- How parameterization affects optimization paths

### Active reading workflow (60–75m)
1. Before reading: write a 5-line belief about “what gradients point to.”
2. During reading:
   - Mark every place an inner product or norm is assumed
3. After reading:
   - Rewrite one argument in your own notation
   - Explicitly state where geometry enters

### Output (last 10m)
Create note: **Week4_S3_Read_GradientGeometry**
- 5 invariants or takeaways
- 5 questions for Week 5 (e.g., “What changes under a different metric?”)

---

## Session 4 (60–90m) — Write/teach: explainer v0.3 (geometry edition)
**Goal:** Turn intuition into something you could confidently teach.

### Add these sections to your explainer
1. **Gradients are covectors**
   - $df_x : \mathbb{R}^n \to \mathbb{R}$
   - Why ML libraries hide this fact
2. **Why gradients point “uphill”**
   - Level sets and normals
3. **Why gradient descent works**
   - Small steps along the normal reduce the function fastest
4. **Where choices sneak in**
   - Inner products
   - Parameterization
   - Scaling

### Output
- Updated doc: **Backprop_Explainer_v0.3**
- Include one hand-drawn (or ASCII) level-set diagram

---

## Optional Session 5 (45–75m) — Non-technical: geometry, models, and belief
**Goal:** Reflect on how representation shapes understanding.

Pick one lens:
- Geometry as a *lens* rather than truth
- Why coordinate systems matter in science
- Historical examples of “wrong coordinates, right physics”

### Output
Note: **Week4_Philosophy_Models_and_Geometry**

---

## Week 4 “done” checklist
- [ ] I can explain why gradients are covectors without handwaving.
- [ ] I can relate gradient descent to level sets and normals.
- [ ] I observed gradient geometry effects in real code.
- [ ] My explainer now includes a geometry-first explanation.

