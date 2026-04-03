# Adam Math

## Progression

### AdaGrad

- Divides by the square root of the sum of squares of the grad.
- Accumulate squared grad by summation, but this will only grow since it's a squared value. Using the square root keeps the scaling in line.
- By dividing by large numbers in the denominator, we penalize larger gradients. However, this could lead to the learning rate eventually trending towards zero.

### RMSProp

- It improves upon AdaGrad by using the EMA of the squared grad instead of the sum to decay old information. Unlike AdaGrad the denominator can shrink as old gradients die out, keeping the effective learning rate in a useful range.

### Adam

- Builds on the previous two but adds momentum and bias correction.

## Adam update

1. $m_t= \beta_1 m_{t-1} + (1-\beta_1})g_t$ update to the first moment
2. $v_t= \beta_2 v_{t-1} + (1-\beta_2})g_t^2$ update to the second moment using squared grad
3. $\hat{m}_t = m_t/(1-\beta_1^t), \hat{v}_t = v_t/(1-\beta_2^t)$ Bias correction for both moments
4. $w_t=w_{t-1} - \alpha \hat{m}_t / (sqrt{\hat{v}_t}+\epsilon)$ Update step
   $1/sqrt(\hat{v}_t)$ normalizes the update, reducing large gradients and boosting small gradients

## Bias Correction

The division by $(1-\beta^t)$ helps with the estimation of $E[g_t^2]$ with $v_t$ since if we expand the summation it is off by a factor of $(1-\beta^t)$. Without this correction the early steps have a small $\sqrt(v_t)$ which means a small denominator that can make updates too large.

## Adam as diagonal preconditioning

Generally preconditioning the gradient step can be seen as multiplying a matrix $P$: $w_{t+1} = w_t - \alpha P g_t$. So we can look at Adam as one implementation of the matrix $P$. Using the ideal matrix $P=H^{-1}$ would lead to inefficient storage and computation, so Adam uses a diagonal: $P=diag(\frac{1}{\sqrt{v_t}})$ to make it O(n). This compromise captures the curvature of the individual parameters, but loses the correlation between the parameters.

## The metric thread

Going from SGD -> Momentum -> Adam can be seen as progressively adding richer preconditioning adjustments to the past choice to adjust for poorly conditioned loss landscapes [[Week5_S1]].
