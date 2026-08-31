# Task 3 — leakage-safe tabular vertical slice

## Pre-run experiment contract

- Dataset contract: 240 IID rows, 4 numeric features, binary label, intentional class imbalance.
- Split unit and justification: row level - rows are IID
- Fixed seed: use 7 as seed for testing
- Train/test sizes: 70% train / 30% test
- Data lineage: hold out `y_test`; split → fit scaler on `x_train` → transform train and test with the same training statistics → train on transformed training data → evaluate once on held-out test data.
- Preprocessing fit boundary: fit feature mean/std on X_train only; transform train and test
- Trivial baseline: predict the majority label observed in y_train
- Learned baseline and fixed hyperparameters: two class linear softmax model, W scale: 0.01, zero bias, full-batch SGD, learning rate 0.1, 20 steps
- Primary metric and defence: (recall_0 + recall_1)/2
- Error slice: report recall separately for class 0 and class 1
- balanced accuracy used because class imbalance makes ordinary accuracy misleading
- Predeclared success target: learned balanced accuracy >= 0.80 and >= trivial baseline + 0.25
- Expected leakage failure signature: fitting the scaler on all 240 rows rather than only 168 training rows would make the scaler's fitted-row lineage include held-out inputs.
- Reproduction command: uv run python task3_tabular_experiment.py

## Post-run results

- Train class counts: class 0 = 128; class 1 = 40.
- Test class counts: class 0 = 52; class 1 = 20.
- Majority class: 0.
- Majority-baseline balanced accuracy: 0.5.
- Learned-model balanced accuracy: 1.0.
- Learned-model recall, class 0: 1.0.
- Learned-model recall, class 1: 1.0.
- Target met: yes; 1.0 ≥ 0.80 and 1.0 − 0.5 = 0.5 ≥ 0.25.
- Leakage audit: scaler fit received 168 training rows only; `test_task3_tabular_experiment.py` instruments the fit boundary and verifies `[168, 168]` across two fixed-seed runs. Test labels are used only by held-out metric/error-slice evaluation.
- Verification: `uv run pytest -q test_task3_tabular_experiment.py` → `4 passed in 3.55s`; `uv run python task3_tabular_experiment.py` reproduced the seed-7 result above on CPU.
- Attempt errors: `P` — initial metric implementation, class-recall vectorisation, missing model bias, and incomplete data-lineage wiring required focused contract feedback. `C` — the preserved first-attempt commit included unrelated/generated files.
- Assistance: `scaffolded` — reused/adapted the prior classifier mechanism, received repeated contract reviews, and used tutor-supplied post-commit tests.
- Elapsed time: approximately 2 hours, completed piecemeal; prediction was 45 minutes.
