Perfect — mini-implementations means Weeks 12–13 stay “conceptual + tested” instead of turning into a framework-architecture saga.

Here’s the same merged plan starting Week 7, with **Weeks 12–13 explicitly scoped to standalone mini-impls** (no need to weave them into your `Value` engine).

## Combined Q1 Plan (Week 7 onward)

### Week 7 — SGD + Momentum (ship an optimizer API)

- **Deliverables**
  - `Optimizer` abstraction + `SGD`, `Momentum`
  - One ravine-ish toy where momentum clearly helps (loss vs steps)
  - 1–2 page note: momentum as EMA / heavy ball

- **Sessions**
  - S1: derive momentum update + ravine intuition
  - S2: implement + run comparison
  - S3: read one momentum explainer; extract 3 rules
  - S4: write + add plots/screenshots

### Week 8 — Train/val + metrics + weight decay

- **Deliverables**
  - train/val split, accuracy (or your chosen metric)
  - L2 weight decay toggle
  - short “training lies” diagnostic note

- **Sessions**
  - S1: derive L2 gradient contribution
  - S2: implement eval loop + metric + decay
  - S3: read regularization/overfit section
  - S4: write diagnostic checklist

### Week 9 — Adam (recommended if Week 7–8 are stable)

- **Deliverables**
  - Adam w/ bias correction, epsilon
  - compare SGD vs momentum vs Adam on same task
  - 1-page “when to use what” cheatsheet

- **Sessions**
  - S1: Adam as momentum + RMS scaling (diagonal preconditioner / “metric” story)
  - S2: implement + small ablations (bias correction on/off)
  - S3: read about AdamW / failure modes
  - S4: write cheatsheet

### Week 10 — Loss landscape slices (working theory: sharp/flat)

- **Deliverables**
  - 1D/2D parameter-direction loss slice plot(s)
  - note: what sharp/flat means + caveats

- **Sessions**
  - S1: Taylor/Hessian local picture
  - S2: implement “pick direction(s) → evaluate loss along α”
  - S3: read Li et al. landscape viz summary
  - S4: write interpretation + pitfalls

### Week 11 — Implicit bias of GD (underdetermined regression experiment)

- **Deliverables**
  - p>n linear regression: GD-from-zero → min-norm solution experiment
  - short explainer: “GD chooses among infinite solutions”

- **Sessions**
  - S1: why GD stays in row space (min-norm)
  - S2: code experiment + vary init
  - S3: read one implicit-bias reference
  - S4: write up results

### Week 12 — Convolution mini-implementation (standalone, grad-checked)

**Scope**: pure Python (loops OK), tiny sizes, grad-check via finite differences.

- **Deliverables**
  - minimal conv forward (1D or tiny 2D)
  - finite-diff gradient check for weights + input
  - 1-page note: conv as structured linear operator / weight sharing

- **Sessions**
  - S1: conv as sliding dot-product; Toeplitz intuition
  - S2: implement + grad-check harness
  - S3: read CNN inductive bias short piece
  - S4: write note + include one diagram

### Week 13 — Attention mini-implementation (standalone + visualization)

**Scope**: pure Python, softmax stabilization, show attention weights.

- **Deliverables**
  - scaled dot-product attention: `softmax(QK^T / sqrt(d)) V`
  - toy task + attention heatmap
  - 1-page note: attention as differentiable lookup / soft addressing

- **Sessions**
  - S1: derive attention + why scaling matters
  - S2: implement + visualize weights
  - S3: read AIAYN sections 1–3; answer “why Q/K/V?”
  - S4: write explainer

### Week 14 — Q1 wrap/polish (buffer week if needed)

- **Deliverables**
  - repo cleanup + tests + README “how to reproduce”
  - final synthesis writeup: your “backprop + optimization + generalization” story arc
  - Q2 launch questions (transformers-focused)

If you follow your weekly cadence template (S1 math → S2 code → S3 read → S4 write), this slots in cleanly.

One small “mentor nudge”: the plan works best if every week produces **one artifact you could show another human** (plot, note, mini-report). That keeps it from dissolving into vibes.

Next up, since you’re at Week 6 now: **Week 7 Session 1** is basically “momentum as EMA + ravine intuition.” We can jump straight into that whenever you’re ready.
