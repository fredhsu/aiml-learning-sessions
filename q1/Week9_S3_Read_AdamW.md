# Week 9 Session 3 - AdamW reading

## Key point

L2 regularization is not the same as weight decay for adaptive gradient methods. In standard Adam with L2 regularization, the penalty term becomes entangled with Adam’s per-parameter adaptive scaling. By decoupling weight decay, AdamW is able to get much better results for tasks that Adam did not perform well in. In particular, for image classification SGD performed far better than Adam with regularization, but with AdamW they were able to get results similar or better than SGD. L2 regularization modifies the objective, while weight decay modifies the optimization dynamics; in plain SGD these happen to coincide, but in Adam they no longer do.

## Why L2 + Adam != weight decay + Adam

For normal Adam with L2 Regularization, $\lambda w_t$ is added to the gradient then gets divided by $\sqrt{\hat{v}_t}$, so the regularization gets rescaled per parameter. This is no longer the same as weight decay, so to perform weight decay properly, the $\lambda w$ term should be added _after_ the Adam step. The update formula makes this apparent:

$$
w_{t+1} =w_t - \alpha \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) - \alpha \lambda w_t
$$

## Decision criteria

In the paper AdamW proves to be robust and makes for a good default optimizer. AdamW also has the advantage of decoupling the hyperparameters so we can tune the learning rate and weight decay more independently, making the hyperparameter search easier. There may be some instances for vision tasks where SGD+momentum generalizes better.
