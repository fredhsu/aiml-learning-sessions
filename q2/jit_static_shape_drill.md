# JIT/static-shape drill — first attempt

Complete this before running `jit_static_shape_drill.py`.

## Prediction

- Score: 3 / 5
- Time: 15 min
- Most likely failure mode, and the exact symptom it would produce: `logsumexp` reduces on `axis=0` rather than the class axis, so values remain finite but the Optax parity loss difference exceeds `1e-6`.
- Predeclared reference-check settings: loss absolute tolerance `1e-6`; central finite-difference step `h=1e-2`; maximum absolute gradient-error tolerance `2e-4`.

Scope note: this card is predicted at 15 minutes. If the prediction exceeds that, split it.

## Shape trace

| Expression                               | Shape  | Why                       |
| ---------------------------------------- | ------ | ------------------------- |
| `x`                                      | BxD    | batch x features          |
| `x_i`                                    | D      | features                  |
| `params["W"]`                            | DxC    | Feature x output          |
| `params["b"]`                            | C      | match output              |
| `one_logits(params, x_i)`                | C      | x_i @ W + b               |
| `batched_logits(params, x)`              | BxC    | x@W + b                   |
| `logits` in `stable_loss`                | BxC    | logits from batches       |
| per-row log-normaliser                   | Bx1    |                           |
| gathered correct-class log-probabilities | Bx1    | filtered to correct class |
| loss                                     | Scalar | mean over all values      |
| `grads["W"]`                             | DxC    | match params              |
| `grads["b"]`                             | C      | match params              |

## `vmap` invariants before execution

Answer before writing `in_axes`, and do not run the file to check.

- Chosen `in_axes` for `batched_logits`, and why each entry is what it is:
  (None,0) which map to (params,x_i). Shares params, maps batch examples
- What `in_axes=(0, 0)` would do here, and whether it fails loudly or silently:
  Fails loudly due to mapped axis of W, b, x having incompatible sizes. The current sizes would cause it to fail.
- What `in_axes=(None, None)` would do here, and whether it fails loudly or silently:
  Fails loudly, vmap needs at least one mapped axis, in this case it would fail since vmap maps no axis
- Which of the three wrong-but-runnable options is most dangerous, and why:
  (0,0) as it could succeed by chance of the axis being compatible

## JIT invariants before execution

- Static class-count source: `params['W'].shape[1]`
- Why it is available while tracing: known from abstract parameter shape during this trace
- Why `jnp.unique(y)` without a static output size fails under `jit`: the number of unique y values is determined during computation
- Predicted outcome of `bad_unique_count(y)` and expected exception text/type: ConcretizationTypeError
- Predicted outcome when `update` is later called with `x2: (8, D)`:
  Succeeds and returns scalar loss, will trigger a shape signature compilation

## Rubric — 5 points

Predicting a score is only meaningful against a stated rubric. Score each point binary.

| Point | Evidence                                                                                                                                                     |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1     | Shape trace is complete and correct before any execution.                                                                                                    |
| 1     | `in_axes` is chosen correctly from the contract, and both wrong alternatives are predicted correctly including whether each fails loudly or silently.        |
| 1     | `stable_loss` and the jitted `update` run, preserve pytree shapes, and produce finite values on both `(B, D)` and `(8, D)`.                                  |
| 1     | The `jnp.unique`-under-`jit` failure is predicted before execution with the correct exception type and the correct violated invariant.                       |
| 1     | Independent reference check passes: loss matches optax and `grads["W"]` matches a finite-difference estimate, both within a tolerance stated before running. |

## Post-run evidence (complete only after the first attempt is committed)

- First-attempt commit:
- Command:
- Output:
- `bad_unique_count` exception:
- Did `(8, D)` update succeed? What happened regarding compilation?
- `in_axes` actually chosen, and whether the alternatives behaved as predicted:
- Independent reference check: loss compared against `optax.softmax_cross_entropy_with_integer_labels`, and gradients against a finite-difference estimate. Command and parity result:
- Actual score / time (record the actual/predicted time ratio):
- Failure-mode prediction: hit / miss, with what actually broke first:
- Attempt errors and rationale. Use one letter from `K/R/M/D/P/F/T/C`, not prose:
  - `K` never encoded · `R` was encoded, unavailable now · `M` confident wrong model
  - `D` correct procedure from a neighbouring case · `P` knowledge present, hands wrong
  - `F` accurate but too slow · `T` failed to transfer to a changed surface · `C` process/care slip
  - Test for `P`: asked right now, could you state the correct procedure without looking it up? If not, it is `R` or `K`.
