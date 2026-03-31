# Diagnostic Checklist

## Sanity checks

1. With random initialization does the loss match the theoretical expected value?
2. Overfit with a single training example to near-zero loss.

## Five failure modes

| Observation                                      | Cause                                                                      | Check                                          |
| ------------------------------------------------ | -------------------------------------------------------------------------- | ---------------------------------------------- |
| Loss not decreasing from the start               | Learning rate too high/low or computation bug                              | Overfit on a small sample and see if it works  |
| Train loss decreasing but val loss not improving | Overfitting                                                                | Check shape of divergence / data leakage       |
| Both losses decreasing, but very slowly          | small learning rate, poor initialization                                   | Check update/weight ratio, should be near 1e-3 |
| Loss becomes NaN or Inf                          | LR too high causing divergence                                             | Log gradient norm, check $\alpha \lambda < 1$  |
| Accuracy flat despite loss decreasing            | improvements to loss are not enough to trigger prediction change/accuracy, | Check scaling or per class accuracy            |
