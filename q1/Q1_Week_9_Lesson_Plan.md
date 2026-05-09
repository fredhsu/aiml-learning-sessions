# Week 9 — Adam and Adaptive Learning Rates

The big picture: Adam as momentum + diagonal preconditioning, connecting back to your metric/inner product thread from Weeks 4–5. The punchline you're building toward is that every optimizer you've studied is really a different choice of effective inner product.

**Session breakdown:**

**S1 (Math)** — Derive Adam from first principles via the AdaGrad → RMSProp → Adam progression. The prep reading is CS4787 Lecture 8 on preconditioning and adaptive learning rates. You'll work through bias correction derivation and land on Adam as a diagonal approximation to the inverse Hessian — the concrete payoff of your "steepest descent depends on the metric" insight.

**S2 (Code)** — Implement Adam with bias correction in your optimizer framework, then run a 3-way comparison (SGD vs. Momentum vs. Adam) on your toy problems. Key ablations: bias correction on/off, epsilon sensitivity, and performance on the isotropic bowl vs. the ravine.

**S3 (Read)** — Loshchilov & Hutter's AdamW paper as primary reading. The critical insight: Adam + L2 ≠ Adam + weight decay, because L2 modifies gradients _before_ Adam's adaptive scaling distorts the regularization. CS4787 Lecture 9 (variance reduction, Polyak averaging) is optional but plants seeds for RL later.

**S4 (Write)** — A 1-page optimizer cheatsheet covering SGD, Momentum, Adam, AdamW with default hyperparameters, failure modes, decision criteria, and the unifying metric connection.

**Done checklist:**

- Can derive Adam's update and explain each component's role
- Can explain why Adam is a diagonal preconditioner (metric connection)
- Know when AdamW vs. Adam + L2 and why
- Comparison plot clearly shows SGD/Momentum/Adam tradeoffs
