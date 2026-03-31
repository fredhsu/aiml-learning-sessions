# W8S3 Reading - Training diagnostics

## Things to check when training isn't working (Pre-read)

- Check if the loss still decreases when learning rate is small and momentum/ regularization are zeroed.
- See if the loss has started to continuously increase indicating divergence.
- Check the training data to see if it is properly constructed

## Training diagnostics (Post-read)

Reading: [Learning and Evaluation - cs231n](cs231n.github.io/neural-networks-3/)

- Gradient checks
  - Compare the analytic gradient to numerical gradient
  - Be careful of precision issues
  - Isolate gradient checking from other optimizations
  - Use relative error with a ratio vs the absolute values
- Sanity check before training
  - Make sure you get correct loss with small parameters
  - Increasing regularization should increase loss
  - Overfit a small set of data
- Babysitting the learning process
  - Graph loss function and train/val accuracy
  - Plot activation/gradient histograms to find incorrect initialization
  - Track the update/weight ratio $\frac{\alpha ||\nabla L||}{||w||}$ which should be around 1e-3. **new to me**
- Hyperparameter optimization
  - Search on a log scale
  - Prefer random search to grid search **new to me**
  - Search from coarse to fine

## Early stopping <> L2 regularization

- For a quadratic loss, stopping early and regularizing with λ can produce the same solution under the right conditions.
- Both early stopping and L2 regularization are forms of implicit bias — they select among solutions not by the loss function alone but by the dynamics of the optimization procedure.
