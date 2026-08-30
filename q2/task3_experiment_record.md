# Task 3 — leakage-safe tabular vertical slice

## Pre-run experiment contract

- Dataset contract: 240 IID rows, 4 numeric features, binary label, intentional class imbalance.
- Split unit and justification: row level - rows are IID
- Fixed seed: use 7 as seed for testing
- Train/test sizes: 70% train / 30% test
- Data lineage: hold out y_test, do not use in training, split -> fit scalar on train -> transform both
- Preprocessing fit boundary: fit feature mean/std on X_train only; transform train and test
- Trivial baseline: predict the majority label observed in y_train
- Learned baseline and fixed hyperparameters: two class linear softmax model, W scale: 0.01, zero bias, full-batch SGD, learning rate 0.1, 20 steps
- Primary metric and defence: (recall_0 + recall_1)/2
- Error slice: report recall separately for class 0 and class 1
- balanced accuracy used because class imbalance makes ordinary accuracy misleading
- Predeclared success target: learned balanced accuracy >= 0.80 and >= trivial baseline + 0.25
- Expected leakage failure signature: fitting the scalar (240 rows) before splitting contaminates held-out (168 training rows) evaluation
- Reproduction command: uv run python task3_tabular_experiment.py

## Post-run results

- Test class counts:
- Majority-baseline balanced accuracy:
- Learned-model balanced accuracy:
- Learned-model recall, class 0:
- Learned-model recall, class 1:
- Target met:
- Leakage audit:
- Attempt errors:
- Assistance:
