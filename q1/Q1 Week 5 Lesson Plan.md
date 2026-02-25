## Q1 · Week 5 — **Gradient Descent as Geometry (First Contact with Optimization)**

**Quarter theme:** Foundations you truly own
**Week theme:** _Why gradient descent is the step it is_
**Status context:** You’ve just locked in backprop, VJPs, gradients-as-covectors, and numerical trust. Now we let the gradient _move something_.

This week is the hinge: derivatives stop being static objects and start becoming **flows**.

---

### What Week 5 Must Achieve (Non-Negotiables)

By the end of this week, you should be able to say—without bluffing:

- Gradient descent is **not a rule**, it’s the solution to a **constrained optimization problem**
- “Steepest descent” only makes sense **relative to a metric**
- Learning rate is a **trust-region parameter**, not a magic knob
- The ravine problem is not a pathology—it’s geometry complaining

If those sentences feel solid, the week worked.

---

## Session Structure (aligned with your weekly cadence)

### **Session 1 — Math & Geometry (60–90 min)**

**Gradient descent from first principles**

This is a paper-and-pencil session. No code. No libraries. No shortcuts.

You derive GD as:

$$
\min_{|\Delta x| \le \varepsilon} \nabla f(x)^\top \Delta x
$$

Key threads to hit:

- Taylor expansion → linear approximation → constrained minimization
- Why the solution is ( \Delta x \propto -\nabla f(x) )
- Where the **norm** sneaks in
- Why changing the norm changes the notion of “steepest”

**Explicit bridge to earlier weeks:**
The gradient is still a **covector**. The inner product is what turns it into a vector field you can follow.

**Output:**
A short note titled something like

> _“Gradient Descent Is Steepest Descent Under the Euclidean Metric”_

If that title feels obvious, you’re doing it right.

---

### **Session 2 — Code from Scratch (60–90 min)**

**Watch geometry happen**

You now make the math visible.

What you implement:

- Plain gradient descent (no momentum, no tricks)
- On **2D toy functions only**:
  - isotropic bowl
  - narrow ravine
  - saddle (optional but enlightening)

What you observe:

- Oscillation across the ravine
- Sensitivity to learning rate
- Why “just lower the LR” is an unsatisfying answer

No neural nets yet. This is physics class, not engineering lab.

**Output:**
Plots + a short paragraph per surface:

> “What geometry predicted, what I saw.”

---

### **Session 3 — Read Actively (60–90 min)**

**Conditioning, curvature, and why GD is slow**

This is a _selective_ reading session. You are hunting for one idea:

> Convergence rate depends on the **condition number** of the Hessian.

You should come away understanding:

- Why ravines exist in parameter space
- Why eigenvalues matter
- Why preconditioning is a geometric operation

Do not summarize the reading. Rewrite the idea in **your notation**.

**Output:**
A note answering:

> “Why does curvature limit how fast GD can move?”

---

### **Session 4 — Write / Teach (60–90 min)**

**Explainer v0.4**

You now update your running explainer document.

This week adds one conceptual layer:

- GD as constrained optimization
- Metric dependence of “steepest”
- Visual intuition for ravines

No momentum yet. No Adam. One idea, cleanly landed.

If you can explain this to your _future self_ without equations, you win.

---

### **Optional Session 5 — Non-Technical (light, but aligned)**

**Optimization as a universal idea**

Pick _one_ lens:

- Gradient flows in physics (heat, diffusion)
- Coordinate systems in mechanics
- Why “choice of metric” is a modeling decision everywhere

Write **5–10 bullets**, not an essay. This is seasoning, not the main dish.

---

## End-of-Week Checkpoint (Be Honest)

Week 5 is done only if you can:

- Derive GD without invoking authority
- Explain why learning rate depends on curvature
- Predict GD behavior on a loss surface _before_ running it
- Say what would change if the norm changed

If any of those feel shaky, repeat **Session 1**, not more reading.

---

## Why This Week Matters (Zoomed Out)

Everything coming next—momentum, Adam, natural gradient, second-order methods—is just answering the question Week 5 raises:

> _What if Euclidean geometry is the wrong geometry?_

You’ve now set the trap. Next week, momentum walks right into it.

When you’re ready, we’ll do **Week 6: Momentum as Geometry**, where gradients acquire inertia and optimization starts to feel like dynamics instead of bookkeeping.
