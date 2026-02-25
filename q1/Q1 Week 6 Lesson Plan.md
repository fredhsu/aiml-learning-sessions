**Theme:** *Vectorize the model (so it scales) + add momentum (so it trains better).*
You’ll end the week with (1) a batched MLP that doesn’t explode on shapes, and (2) an optimizer that doesn’t ping-pong in ravines.

#### Session 1 (60–90m) — Matrix gradients for `Y = XW + b`

* Re-derive the backward rules for batched matmul + bias add.
* Write a **shape contract** for each op: input shapes, output shape, and gradient shapes.
* Tiny test: finite-diff check on a 2×3×4 toy example (small enough to eyeball).

**Ship:** `Linear` layer math + shape-contract notes.

#### Session 2 (60–90m) — Implement vectorized ops + shape-bug tests

* Implement: `matmul`, `sum(axis=...)`, broadcasting add, and their backwards.
* Add tests specifically designed to catch silent broadcasting bugs.
* Run a 2-layer MLP forward/backward on a batch.

**Ship:** batched 2-layer MLP runs end-to-end.

#### Session 3 (60–90m) — Momentum as geometry (math + intuition)

* Derive momentum as an **EMA of gradients** and explain why it cancels zig-zag in ravines.
* Connect to the **heavy-ball** dynamics picture (discrete → continuous intuition).
* (Optional) Nesterov: one paragraph “lookahead gradient” intuition.

**Ship:** your “Momentum smooths the gradient field” section draft.

#### Session 4 (60–90m) — Add momentum (and compare trajectories)

* Implement SGD+momentum (optionally Nesterov) in your optimizer.
* Compare GD vs momentum on: ill-conditioned quadratic ravine + saddle.
* Track: loss + step norm or velocity norm.

**Ship:** plots/logs + “when it helps / when it hurts” bullets.

#### Optional Session 5 (30–60m) — One-page cheat sheet

* “Gradients are covectors → we store them as arrays via an inner product”
* “Shape contracts prevent 80% of bugs”
* “Momentum = smoothing + inertia”

Yep — and we can make it *do work* (not just “consume words”).

### Week 6 Reading Assignment (60–90m)

Pick **one** primary reading + **one** optional “intuition booster”.

**Primary (choose 1):**

1. **Goodfellow/Bengio/Courville (Deep Learning), Ch. 8 “Optimization for Training Deep Models”**
   Focus sections: conditioning, momentum, Nesterov, and why curvature/ill-conditioning causes zig-zag.
   This pairs perfectly with your ravine + condition number thread.

2. **CS231n notes on optimization (SGD, momentum, Nesterov)**
   Short, practical, and very “implementation-forward.” Great companion to your vectorization week because it keeps the mental model tied to code.

3. **Sutskever’s “On the importance of initialization and momentum…” (thesis / writeup sections)**
   This is the “momentum is not a hack” lens: how it interacts with curvature and why it often behaves like acceleration.

**Optional intuition booster (10–20m, choose 1):**

* A blog/notes piece on **heavy-ball ODE** intuition for momentum (discrete update ↔ damped dynamics).
* A short explainer on **EMA filtering** (momentum as a low-pass filter on gradient noise + oscillations).

---

### How to read it (so it sticks)

Write these *three* mini-outputs in your notes as you read:

1. **One equation, translated into your notation**
   Example target:
   ( v_{t+1}=\beta v_t + g_t,\quad x_{t+1}=x_t-\alpha v_{t+1} )

2. **One “cause → effect” sentence about ravines**
   Something like: *ill-conditioning → gradients point across valley → oscillation → momentum averages/cancels the alternating component.*

3. **One “failure mode” note**
   When does momentum get worse? (overshoot, too-high β/α, sharp minima, etc.)

---

### Where it plugs into the merged Week 6 plan

* **Session 3** becomes “Reading + momentum math”: read first, then derive/annotate with your own words.
* **Session 4** becomes “prove it in code”: momentum vs GD trajectories on the ravine/saddle.

This keeps Week 6 coherent: **shapes make it run, momentum makes it behave**.
