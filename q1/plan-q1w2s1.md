Below is a **concrete, time-boxed lesson plan** for
**Q1 · Week 2 · Session 1**, designed to produce the _first real learning loop_ with zero abstractions.

---

# Q1 · Week 2 · Session 1

## **Loss Functions & the First Learning Loop**

**Session goal:**

> _I can watch parameters change because gradients tell them to._

This is the moment where backprop stops being theory.

---

## ⏱ Time Box (75–90 minutes)

---

## 0–10 min — Conceptual Setup (No Code)

Write this sentence at the top of your notes:

> **A loss function turns “wrongness” into a number gradients can push against.**

Answer briefly:

- What does it mean for a model to be “wrong”?
  The prediction doesn't match the truth

- Why can’t we optimize accuracy directly?
  It is too hard to calculate directly.

(You’ll converge on: differentiability.)

---

## 10–25 min — Define the Simplest Possible Problem

We’ll fit **one parameter**.

### Model

[
\hat{y} = w \cdot x
]

### Loss (Squared Error)

[
L = (\hat{y} - y)^2
]

Choose:

- `x = 2`
- `y = 4`
- initial `w = -1`

**By hand (do this):**

- Compute `ŷ`
- Compute `L`
- Predict: should `w` increase or decrease?

---

## 25–45 min — Implement Squared Loss in Your Engine

Add:

```python
def square(self):
    out = Value(self.data ** 2, (self,), 'square')

    def _backward():
        self.grad += 2 * self.data * out.grad

    out._backward = _backward
    return out
```

Then build the graph:

```python
x = Value(2.0)
y = Value(4.0)
w = Value(-1.0)

y_hat = w * x
loss = (y_hat - y).square()
loss.backward()
```

**Check manually:**

- Is `w.grad` positive or negative?
- Does that match your intuition?

---

## 45–60 min — Gradient Descent (One Step Only)

Write:

[
w \leftarrow w - \eta \frac{\partial L}{\partial w}
]

Choose:

- `η = 0.1`

Apply **exactly one update**:

```python
w.data -= 0.1 * w.grad
```

Recompute:

- new `ŷ`
- new `L`

Write:

> Did the loss go down?

---

## 60–75 min — Loop It (Still Tiny)

Wrap the update in a loop:

```python
for step in range(10):
    loss.backward()
    w.data -= 0.1 * w.grad
    w.grad = 0.0
```

Print:

- `step`
- `w`
- `loss`

**Pause and observe:**

- Is `w` moving smoothly?
- Does learning slow down?

---

## 75–85 min — Failure Mode Awareness

Answer in writing:

1. What happens if `η` is too large?
2. Why must you reset `w.grad` each step?
3. What does it mean if loss explodes?

These answers generalize directly to deep nets.

---

## 85–90 min — Synthesis (Mandatory)

Write 3 bullets titled:

> **“What Training Actually Is”**

Good bullets:

- Loss defines direction
- Gradients define sensitivity
- Updates are blind but systematic

---

## ✅ Definition of “Done”

You are done when:

- You can predict the **sign** of gradients
- Loss decreases over iterations
- You understand why resetting gradients is necessary

---

## 🚫 Anti-Goals

Do **not**:

- Add datasets
- Add vectors
- Add multiple parameters
- Add ReLU yet

That comes next.

---

## What Comes Next

**Week 2 · Session 2**

- Two parameters
- Linear regression
- First contour plots (mentally)

If you want, I can:

- Extend this to **multiple data points**
- Add a **gradient-checking exercise**
- Compare squared loss vs absolute loss

Just say where to go.
