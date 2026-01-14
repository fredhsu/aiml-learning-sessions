# Q1 · Week 1 · Session 2

## **Building the Smallest Autodiff Engine (Scalar Only)**

**Session goal (read this first):**

> _I understand backprop because I implemented it for scalar expressions and watched gradients flow._

No vectors. No matrices. No PyTorch. Just numbers and the chain rule.

---

## ⏱ Time Box (90 minutes total)

### 0–10 min — Mental Setup (No Code)

**Write this on paper or notes:**

We want to compute gradients for expressions like:

```
z = x * y + y
```

What we need:

- A **computation graph**
- A way to store:
  - value
  - gradient
  - how this value was computed

If this feels “obvious,” good. If not, also good.

---

### 10–25 min — Define the Core Abstraction

**Key idea:** Every number is a _node_ in a graph.

You will implement a `Value` object with:

- `data`: the scalar value
- `grad`: gradient w.r.t. final output
- `_prev`: parents in the graph
- `_backward`: function to propagate gradients

**Checkpoint (don’t code yet):**
Ask yourself:

> “If I know ∂L/∂z, how do I compute ∂L/∂x?”

Answer: chain rule → that logic lives in `_backward`.

---

### 25–55 min — Minimal Implementation (Core of the Session)

Write a **very small** class (≈30–40 lines). Pseudocode structure:

```python
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op
```

Then implement:

#### Addition

```python
def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')
    def _backward():
        self.grad += out.grad
        other.grad += out.grad
    out._backward = _backward
    return out
```

#### Multiplication

```python
def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')
    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
    out._backward = _backward
    return out
```

**Key insight to pause on:**
The backward rules are just partial derivatives.

---

### 55–70 min — Backward Pass (Graph Traversal)

We need to call `_backward` **in reverse topological order**.

Implement:

- A DFS to build a topo order
- Set final node’s gradient to 1
- Iterate backwards and call `_backward`

**Conceptual check:**

> Why does the final node’s gradient start at 1?

Answer: ∂L/∂L = 1.

---

### 70–85 min — Test with a Tiny Example

Run this **by hand first**, then in code:

```
x = Value(2.0)
y = Value(3.0)
z = x * y + y
z.backward()
```

Expected:

- `z.data = 9`
- `dz/dx = y = 3`
- `dz/dy = x + 1 = 3`

If numbers don’t match:

- Print intermediate grads
- Trace one edge at a time

This debugging is the _learning_.

---

### 85–90 min — Write the Takeaway (Mandatory)

Write **3–5 sentences**, no more:

- What surprised you?
- What felt simpler than expected?
- What still feels fuzzy?

Example:

> “Backprop is just local derivatives plus bookkeeping. The only ‘magic’ is graph traversal order.”

---

## ✅ Definition of “Done”

You are finished when:

- Gradients match hand calculations
- You can explain _why_ each backward rule is correct
- You did **not** add vectors, NumPy, or clever abstractions

---

## 🔒 Guardrails (Very Important)

Do **not**:

- Add ReLU yet
- Generalize to tensors
- Optimize or refactor
- Look at PyTorch source

That comes later.

---

## What Session 3 Will Build On

Next session will:

- Add a nonlinearity (ReLU or tanh)
- Show how gradient flow can _die_
- Make the leap from math → training intuition

If you want, I can also give you:

- A **one-page handwritten-style derivation** to follow alongside the code
- A **common bug checklist** for this session
- A **15-minute “rescue version”** if you’re short on time

Just say the word.
