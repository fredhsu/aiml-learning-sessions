---
created: 2026-06-04
id: 019e7c3f-2b1a-7c44-9def-1a2b3c4d5e6f
tags:
- lesson-plan
- convolution
- inductive-bias
- linear-operators
title: Q1 · Week 12 · Session-by-Session Lesson Plan
---

# Q1 · Week 12 · Session-by-Session Lesson Plan

## Context: Where You're Coming From

Week 11 closed completely. You proved the row-space theorem on paper (GD from zero → min-norm; non-zero init freezes a null-space component), built `w11s2.py` with four optimizers and verified the result to 8 decimal places, read Neyshabur as primary with a Zhang skim, and shipped *both* S4 options — the standalone explainer and the v0.8 section of the backprop explainer. No open carry-forward issues on the implicit-bias material itself.

Week 11 also produced one result worth carrying in your head as you start fresh material: the Adam finding was **equalization, not concentration**. You and I both predicted it the wrong way going in, and the experiment refuted us. That's the cleanest example so far of why the "predict before running" discipline earns its keep — the math heuristic alone would have left you confidently wrong.

There is one explicit thread to pick up this week: the **four fundamental subspaces** (Strang Chapter 3). Week 11 leaned hard on the orthogonal decomposition $\mathbb{R}^p = \text{rowspace}(X) \oplus \text{null}(X)$, but you used it operationally rather than from a settled foundation. Week 12 is the right place to close that gap, because convolution is a *structured* linear operator and the cleanest way to understand its structure is through the column space and row space of its Toeplitz matrix. The subspace review isn't a detour — it's the lens for the whole week.

The metric thread continues, but the through-line shifts. Weeks 5–11 were about **the algorithm**: which step direction, which solution selected. Convolution is the first week about **the model** — an inductive bias baked into the architecture rather than emerging from the optimizer's trajectory. The Week 11 forward hook named the contrast directly: implicit bias is a bias from the *algorithm*; convolution is a bias from the *structure*. Holding both in view is the point of starting the model-side of Q1 here.

The Week 10 carry-forward — whether condition number $\kappa$ survives reparameterization attacks beyond simple ReLU rescaling — is *not* a Week 12 topic. Leave it parked; it belongs to the landscape/generalization thread you'll revisit in Q2, not the conv week.

Scope discipline for the week: this is a **standalone mini-implementation**. Pure Python, loops are fine, tiny sizes, finite-difference grad-check. No need to wire it into your `Value` engine — that integration is a Week 14 cleanup task, not a Week 12 goal.

---

## Week 12 outcome targets (ship by week's end)

- **Conceptual:** Convolution as a structured linear operator with weight sharing; the Toeplitz view; what inductive bias convolution encodes (locality, translation equivariance, parameter sharing) and — equally important — what problems it's a *bad* prior for.
- **Math:** Convolution written as $y = Kx$ with $K$ banded-Toeplitz; the output-size formula $(n - m + 2p)/s + 1$; the four fundamental subspaces of $K$ (Strang Ch. 3) and what the column space / null space mean for a conv layer; weight sharing as a Jacobian constraint, including the gradient-summing structure of $\partial L / \partial k$.
- **Code:** Minimal 1D conv forward and backward in pure Python (loops OK), with a finite-difference gradient check on *both* $\nabla_x L$ and $\nabla_k L$. 2D is a conceptual extension only unless you're ahead.
- **Teaching artifact:** A standalone note, "Convolution as a structured linear operator," that connects conv back to the Week 4 linear-map story and forward to attention (Week 13). One diagram: the filter sliding across the input aligned with the rows of the Toeplitz matrix.

---

## Session 1 (60–90m) — Math: convolution as structured matrix multiply

**Goal:** Be able to write any 1D convolution as an explicit Toeplitz matrix multiply, derive the output-size formula from scratch, and say precisely what the four fundamental subspaces of that matrix mean for the layer. By the end you should be able to reconstruct the whole picture on a blank page.

### Cold reconstruction prompts (do these before reading anything — 10–15m)

Write your answers in a fresh note before consulting CS231n, Strang, or any reference. Calibration anchors for the post-S1 review.

1. **The operation.** 1D convolution: input $x \in \mathbb{R}^n$, filter $k \in \mathbb{R}^m$. Write down a single output entry $y_i$ as a sum. (Don't worry yet about cross-correlation vs. true convolution — just commit to one convention and note which.)

2. **As a matrix.** You claim $y = Kx$ for some matrix $K$. What are the dimensions of $K$? Sketch its structure for $n=5, m=3$, no padding, stride 1. What's special about its diagonals?

3. **Output size.** Without looking it up: for input length $n$, filter length $m$, padding $p$, stride $s$ — how long is the output? Derive it, don't recall it. Sanity-check against your $n=5, m=3$ case.

4. **The subspace question (Strang Ch. 3 setup).** $K$ is a linear map $\mathbb{R}^n \to \mathbb{R}^{n-m+1}$ (valid conv, stride 1). What are its four fundamental subspaces? In particular: is $K$ full rank? What lives in $\text{null}(K)$ — i.e., what input signals does this convolution annihilate? (Think about what a smoothing filter does to a high-frequency input.)

5. **Weight sharing as a constraint.** A fully connected layer of the same input/output size has how many free parameters? The conv layer has how many? Where, structurally, does the difference live — what constraint is imposed on the entries of $K$?

6. **The metric/algorithm contrast.** Week 11 was about a bias from the algorithm. Convolution is a bias from the model. State in one sentence what's being assumed about the *data* when you choose convolution over a dense layer.

Save these as `Week12_S1_PreReconstruction` — same format as your Week 10/11 S1 conjectures.

### Work (45–60m)

After the cold reconstruction, work it through properly.

1. **1D convolution as a sliding dot product.** Fix the convention (most ML "conv" is actually cross-correlation; state it): $y_i = \sum_{j=0}^{m-1} k_j \, x_{i+j}$.

2. **The Toeplitz form.** Write $y = Kx$ where $K \in \mathbb{R}^{(n-m+1) \times n}$ is banded-Toeplitz — each row is the filter shifted one step right, zeros elsewhere. Do the $n=5, m=3$ case by hand in full so the band structure is concrete.

3. **Output size formula.** $(n - m + 2p)/s + 1$. Derive each piece: padding adds $2p$ to the effective input length; stride divides the number of valid positions; the $+1$ counts the starting position. Check it reproduces your worked example.

4. **Four fundamental subspaces of $K$ (Strang Ch. 3 — close the gap).** This is the part Week 11 left underbuilt.
   - **Column space** $C(K) \subseteq \mathbb{R}^{n-m+1}$: the set of achievable outputs. For valid conv with a generic filter, this is all of $\mathbb{R}^{n-m+1}$ (the map is onto, since $K$ has full row rank for a non-degenerate filter).
   - **Null space** $N(K) \subseteq \mathbb{R}^n$: inputs the conv sends to zero. Dimension $n - (n-m+1) = m-1$. These are the signals the filter is blind to — for a smoothing/averaging kernel, high-frequency components live near here. This is the concrete payoff of the subspace lens: *the null space is the filter's blind spot.*
   - **Row space** $C(K^T)$: the orthogonal complement of the null space inside $\mathbb{R}^n$ — the input directions the layer actually "sees."
   - **Left null space** $N(K^T)$: trivial here when $K$ is onto.
   - Tie this back explicitly to Week 11: the rowspace/nullspace split you used for the min-norm proof is *the same decomposition*, now applied to a structured operator instead of a data matrix.

5. **Weight sharing as a Jacobian constraint.**
   - Dense layer: every entry of $W$ is an independent parameter ($n(n-m+1)$ of them).
   - Conv layer: only $m$ free parameters; $K$ is forced to be Toeplitz. Many entries of the Jacobian are *tied to the same underlying weight*.
   - Two distinct Jacobians to keep straight: $\partial y/\partial x = K$ (the Toeplitz matrix itself), and $\partial L/\partial k$, which **sums** the upstream gradient over every position where $k_j$ was applied. Write out $\partial L/\partial k_j = \sum_i (\partial L/\partial y_i)\, x_{i+j}$ — this sum is the signature of weight sharing and is exactly what your S2 backward pass has to implement.

6. **2D extension (conceptual only).** The filter becomes 2D; $K$ becomes doubly-block-Toeplitz. The principle is identical — don't derive it in full, just note the shape.

### References (consult after the cold reconstruction)
- CS231n "Convolutional Neural Networks" notes — `cs231n.github.io/convolutional-networks/` — spatial arrangement (stride, padding, output size), parameter sharing, the sliding-filter framing.
- Strang, *Introduction to Linear Algebra*, Chapter 3 — the four fundamental subspaces. This is the gap-closer; read the column-space / null-space sections against your $K$ matrix specifically.
- Goodfellow, Bengio, Courville Ch. 9.1–9.2 — the formal convolution-and-Toeplitz treatment, if you want the textbook version.
- Your own `[[Week11_S1_ImplicitBias_Math]]` — the rowspace/nullspace decomposition you're now generalizing.

### Post-reconstruction review (10m)
Compare your pre-reconstruction answers to what you derived. Flag specifically: did you correctly identify what lives in $\text{null}(K)$? Did you derive the output-size formula or recall it? Did you predict the gradient-summing structure of $\partial L/\partial k$, or did it surprise you? The null-space question is the one most likely to expose a gap — it's the bridge between "conv is a matrix" and "conv is a *prior*."

### Output (last 10m)
Create note: **Week12_S1_Conv_Math**
- The Toeplitz form with the $n=5, m=3$ worked example
- The output-size formula with its derivation
- The four fundamental subspaces of $K$, with the null-space-as-blind-spot interpretation
- The weight-sharing Jacobian constraint and the $\partial L/\partial k$ summing structure
- One line linking back to the Week 11 rowspace/nullspace decomposition

---

## Session 2 (60–90m) — Code: implement conv + grad-check

**Goal:** A working 1D conv forward and backward in pure Python, with finite-difference gradient checks passing on both $\nabla_x L$ and $\nabla_k L$.

### Predict before running (do this before writing the backward pass — 5–10m)

Record these in the notebook header, the same way you've done since Week 8:

1. **Forward sanity.** For a length-5 input of all ones and a length-3 averaging filter $k = [1/3, 1/3, 1/3]$, what's the output? (You should be able to say it exactly.)
2. **Gradient-check magnitude.** What relative error do you expect from a centered finite-difference check on a correct backward pass? (Recall your Week 8 grad-checks — what order of magnitude was "passing"?)
3. **The failure mode you're most likely to hit.** Predict it. (Off-by-one in the valid range? Forgetting to *sum* the filter gradient across positions? A flip between conv and cross-correlation in the backward pass?) Write down which one you'd bet on, then see if you're right.

### Work (50–70m)

Roll your own — this matches your build-from-scratch style and is small enough to fully finish.

1. **Forward.** `conv1d_forward(x, k)` with explicit loops. Valid convolution, stride 1 to start. Test against the all-ones / averaging-filter prediction above.
2. **Backward.** Derive and implement both gradients from the Toeplitz view:
   - $\partial L/\partial x$ — this is $K^T$ applied to the upstream gradient (a "full" correlation / flipped convolution). Watch the boundary handling.
   - $\partial L/\partial k_j = \sum_i g_i \, x_{i+j}$ — the summing structure from S1. This is the one your prediction flagged; check it carefully.
3. **Gradient check.** Centered finite differences on both gradients. Use a random small $x$, random small $k$, and a scalar loss (e.g. sum of outputs, or a random linear combination). Report relative error for each; you're looking for $\sim 10^{-6}$ or better.
4. **One extension if time allows.** Add `stride` and/or `padding` to the forward pass and re-run the output-size formula as an assertion (`assert len(y) == (n - m + 2*p)//s + 1`). This turns your S1 formula into a runtime check.

If you'd rather use a sturdier harness than your own finite-diff scaffold, CS231n Assignment 2's "Convolutional Networks" notebook (`cs231n.github.io/assignments2024/assignment2/`) has 2D conv forward/backward with built-in numerical-gradient utilities and CIFAR-10. Scope control: only the conv forward/backward sections — skip the PyTorch and visualization parts. But your own 1D version is the better fit for this week's "build it yourself" intent; treat CS231n as the fallback, not the default.

### Compare prediction to result (5m)
Did the failure mode you predicted actually bite? If a *different* bug hit, that's the calibration signal worth recording — same as the Adam-equalization surprise from Week 11. Note it in the writeup honestly.

### Output (last 10m)
Create note: **Week12_S2_Conv_Code** (notebook `w12s2.py`)
- Working forward + backward
- Grad-check relative errors for both $\nabla_x L$ and $\nabla_k L$ (pass/fail with magnitudes)
- The predicted-vs-actual bug note

---

## Session 3 (60–90m) — Read: what inductive bias convolution encodes

**Goal:** Understand what assumptions convolution bakes in, where translation equivariance is doing the work, and — the question most people skip — when convolution is the *wrong* prior.

### Pre-reading conjectures (10m)

Before opening anything, write:
1. **Three properties of images** that make convolution a good prior. (Locality? Stationarity of statistics? Translation invariance of the label?)
2. **The data assumption, stated as a bet.** Convolution assumes the useful features are the same regardless of *where* they appear. Name a data type where that assumption is *false* — where absolute position matters — and predict that conv would underperform a dense or position-aware model there.
3. **Equivariance vs. invariance.** Predict the difference. Which one does a conv *layer* give you, and which one needs pooling or a later operation to achieve?

### Reading (pick one — 45–60m)

- **Option A (recommended):** CS231n "Understanding and Visualizing CNNs" — `cs231n.github.io/understanding-cnn/`. What CNNs learn layer by layer, deconvnet visualizations, t-SNE of features. Most concrete picture of the bias in action.
- **Option B (formal):** GBC Ch. 9.1–9.4 — convolution, pooling, and translation equivariance treated rigorously. Best if you want the equivariance statement made precise.
- **Option C (historical, shortest):** LeCun et al., "Gradient-Based Learning Applied to Document Recognition" (1998) — the original motivation for weight sharing and local receptive fields.

### Active reading workflow
1. Note every place translation equivariance is explicitly invoked.
2. Watch for the equivariance→invariance step (where does pooling enter, and why).
3. After: write one sentence on a problem domain where convolution is a *bad* prior — the answer to your pre-reading bet #2.

### Output (last 10m)
Create note: **Week12_S3_Read_CNN**
- One paragraph: what bias convolution encodes, in your own words
- Where equivariance does the work, and how invariance is recovered
- The "bad prior" example — the honest limit of the architecture
- Updated answers to the pre-reading conjectures

---

## Session 4 (60–90m) — Write: "Convolution as a structured linear operator"

**Goal:** A standalone explainer that places convolution in the Week 4 linear-map framework and sets up the Week 13 attention contrast. This note pairs with your implicit-bias explainer the way Week 10's landscape note pairs with Week 11's — together they cover the two sources of inductive bias (structure vs. algorithm).

### Target
~1200–1800 words. One diagram. One result/check from S2. Cross-link Week 4, Week 11, and forward to Week 13.

### Structure
1. **Convolution as a constrained matrix multiply.** $y = Kx$, $K$ banded-Toeplitz. Lead with the picture, then the formula. This is the Week 4 "parameterized linear map" story with a structural constraint added.
2. **Weight sharing as parameter tying.** The $m$-vs-$n(n-m+1)$ parameter count; the tied Jacobian; the gradient-summing structure of $\partial L/\partial k$. Make the point that the constraint *is* the inductive bias — fewer parameters, but more importantly a specific shape imposed on what the layer can represent.
3. **The four subspaces, concretely.** What $\text{null}(K)$ means (the filter's blind spot), what $C(K)$ means (achievable outputs). Tie back explicitly to the Week 11 rowspace/nullspace decomposition: same linear algebra, applied to a structured operator instead of a data matrix.
4. **The inductive bias.** Locality, translation equivariance, parameter sharing — and the honest limit from S3: the data assumption convolution makes, and where it fails.
5. **Forward to attention (Week 13).** The one-sentence hook: convolution *hard-codes* its connectivity pattern (a fixed band); attention *learns* the connectivity pattern from the data (a content-dependent, dense-but-soft band). Both are linear maps from values to outputs; they differ in how the mixing weights are determined. This is the pivot that makes Week 13 land.

### The diagram
ASCII or hand-drawn: the filter sliding across the input, with each position aligned to the corresponding row of the Toeplitz matrix. The goal is that "sliding dot product" and "banded matrix" read as obviously the same object.

### Cross-links
- `[[Week4_...]]` — the parameterized linear map (find your Week 4 note's exact title)
- `[[Week11_S1_ImplicitBias_Math]]` — the rowspace/nullspace decomposition you're reusing
- `[[Week12_S1_Conv_Math]]` — the Toeplitz derivation and subspaces
- Forward stub to Week 13 attention

### Output
Create note: **Week12_Conv_Explainer**

(Backprop explainer stays at v0.8 this week — convolution is a model-structure topic, not a backprop-mechanics one, so it doesn't need to fold into the running document. If you want it in there eventually, that's a Week 14 polish call.)

---

## Week 12 "done" checklist

- [ ] I can write a 1D convolution as a Toeplitz matrix multiply and derive the output-size formula without notes.
- [ ] I can name all four fundamental subspaces of the conv matrix $K$ and say what $\text{null}(K)$ means for the layer.
- [ ] I can explain weight sharing as a Jacobian constraint and derive the gradient-summing structure of $\partial L/\partial k$.
- [ ] My 1D conv forward and backward pass finite-difference grad-checks on both $\nabla_x L$ and $\nabla_k L$.
- [ ] I can articulate the inductive bias convolution encodes — and name a problem where it's the *wrong* prior.
- [ ] I can state the convolution-vs-attention contrast in one sentence (hard-coded vs. learned connectivity).
- [ ] The Strang Ch. 3 four-subspaces gap from Week 11 is closed.

---

## Forward connections seeded this week

- **Week 13 (Attention):** The payoff of S4's closing section. Attention is convolution with the connectivity learned instead of fixed; the metric thread reaches its sharpest point when you see the query-key inner product as the thing that *chooses* the mixing weights. You've now seen "structure as bias" (conv) right before "learned structure" (attention).
- **Week 14 (Q1 wrap):** `conv1d.py` goes in the `mini/` folder; the `conv1d_backward` grad-check ($<10^{-5}$ relative error) is one of the three repo tests. The conv explainer is a candidate section for the Backprop Explainer v1.0 if you decide to fold it in.
- **Q2 (Perception / representation learning):** Convolution is the entry architecture. The "what makes a good prior for this data" question you wrote in S3 is the Q2 question in miniature. ViT (attention for images) is the natural conv-vs-attention sequel.
- **Carried, not closed:** The Week 10 question — does $\kappa$ survive reparameterization beyond ReLU rescaling — is still open and still parked. It belongs to the Q2 generalization thread, not here.

---

## Time budget estimate

| Session | Target time | Stretch time |
|---|---|---|
| S1 — Toeplitz math + four subspaces | 75m | 90m |
| S2 — conv forward/backward + grad-check | 75m | 90m |
| S3 — CNN inductive bias reading | 60m | 90m |
| S4 — structured-operator explainer | 75m | 90m |
| **Total** | **4h 45m** | **6h** |

S1 is heavier than Week 11's S1 because you're folding in the Strang four-subspaces review on top of the conv derivation — budget the full 75m and don't rush the null-space interpretation, since it's the conceptual hinge for both S3 and S4.

S2 is the session most likely to overrun, as usual for a from-scratch backward pass. The likely time sink is the boundary handling in $\partial L/\partial x$. If you're running long, ship a correct *valid*-conv (stride 1, no padding) version with passing grad-checks and leave stride/padding as a stretch — a clean minimal implementation beats a half-debugged general one.

S4 should be smooth: the technical content is contained, the diagram is the only new artifact, and the Week 13 hook practically writes itself once S1's subspace picture is solid.
