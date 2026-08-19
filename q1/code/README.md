# Q1 code experiments

Small, from-scratch implementations and the notebooks used during Q1.

- `engine/` — scalar autograd engine, neural-network helpers, and NumPy optimizers.
- `mini/` — standalone, grad-checked implementations.
- `experiments/` — archived marimo notebooks used to generate session plots.
- `tests/` — regression tests for the engine and optimizers.

## Reproduce from a clean checkout

This project requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repository-url>
cd q1/code
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run python mini/conv1d.py
```

The test suite verifies the autograd engine against finite differences and checks the optimizer update rules. The convolution command runs its forward checks and finite-difference checks for both input and kernel gradients.

To open an archived plot notebook:

```bash
uv run marimo edit experiments/w10s2-nb.py
```
