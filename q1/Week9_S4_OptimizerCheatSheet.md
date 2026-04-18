# Week 9 session 4 - Optimizer Cheat Sheet

## [[SGD]]

$$
w_{t+1} = w_t - \alpha \nabla L(w_t)
$$

### Default hyperparameters

- Learning rate $\alpha$ : 0.01 to 0.1

### Failure modes

- Slow convergence when learning rate is too low or high
- High conditioning causes oscillations across the steep direction

### When to use

- Debugging, simple baseline
- When generalization matters more than ease of optimization (such as vision)

## [[Momentum]]

$$
g_t = \nabla L(w_t)
m_t = \beta m_{t-1} + g_t
w_{t+1} = w_t - \alpha m_t
$$

### Default hyperparameters

- Learning rate $\alpha$ : 0.01 to 0.1
- Momentum $\beta$ : 0.9

### Failure modes

- Momentum too high/low could slow updates
- Momentum does not adapt per parameter
- Momentum can overshoot when the loss landscape changes direction

### When to use

- Preferred over vanilla SGD, can be used when SGD would be a good fit.
- Often paired with learning rate schedule

## [[Adam]]

$$

m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
\hat{m}_t = m_t/(1-\beta_1^t), \hat{v}_t = v_t/(1-\beta_2^t)
w_t=w_{t-1} - \alpha \hat{m}_t / (\sqrt{\hat{v}_t}+\epsilon)

$$

### Default hyperparameters

- Learning rate $\alpha$ : 1e-3
- $\beta1$ : 0.9
- $\beta2$ : 0.999

### Failure modes

- May not generalize well for specific cases such as vision
- SGD may outperform for simple use cases
- L2 regularization gets distorted by adaptive scaling — use AdamW instead when regularization is needed

### When to use

- Quick progress, uneven gradient scales, sparse or noisy problems
- Can handle ill conditioned problems better than SGD

## [[AdamW]]

$$

m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
\hat{m}_t = m_t/(1-\beta_1^t), \hat{v}_t = v_t/(1-\beta_2^t)
w_{t+1} =w_t - \alpha \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) - \alpha \lambda w_t

$$

### Default hyperparameters

- Learning rate $\alpha$ : 1e-3
- $\beta1$ : 0.9
- $\beta2$ : 0.999
- Weight decay $\lambda$ : 0.01 (range from 1e-3 to 0.1)

### Failure modes

- Same as Adam aside from the L2 Regularization issue

### When to use

- General improvement to Adam that addresses the L2 Regularization as weight decay issue
- AdamW is often the default in modern deep learning code

## Unifying metric thread

Going from [[SGD]] -> [[Momentum]] -> [[Adam]] can be seen as progressively adding richer preconditioning adjustments to the past choice to adjust for poorly conditioned loss landscapes [[2026-02-09 - Session1 - GD Euclidean]]. SGD uses the identity (Euclidean), Momentum uses the identity but smooths the gradient direction over time, and Adam introduces a diagonal preconditioner $diag(\frac{1}{\sqrt{\hat{v}_t}})$ based on the running second moment of gradients, helps when coordinates have different effective scales.

The preconditioning adjustments change the inner product by changing the metric. For a standard SGD, we use the identity matrix to produce the standard inner product with Euclidean metric. But with Adam, we changed the preconditioning and now use the diagonal matrix $diag(1/\sqrt{\hat{v}_t})$ and changes the inner product to $u^T M v$ with steepest direction: $\delta w = -M^{-1} \nabla L$. Note for Momentum we don't change the metric, we just smooth the gradient over time.

Tying it all together, [[backprop]] gives the [[covector]] that eats a direction and returns a directional derivative (same for all optimizers). The optimizer choice determines which metric converts it to a step, or how we convert the covector into a [[vector]].
