# Session 1 - Linear Classifier shape trace

Let B = batch size, D = features, C = classes

X: (B, D)
W: (D, C)
b: (C,)
y: (B,) integer class IDs

Z = X @ W + b: (B,C)

log_normalizer = logsumexp(Z, axis=1, keepdims=True): (B,1)
log_probs = Z - log_normalizer: (B, C)
correct_log_probs = take_along_axis(log_probs, y[:, None], axis=1): (B, 1)

L = mean(-correct_log_probs): ()

p = softmax(Z)
dL/dZ = (p - one_hot(y, C)) / B: (B, C)
dL/dW = X.T @ (dL/dZ): (D, C)
dL/db = sum(dL/dZ, axis=0): (C,)

W <- W - learning_rate * dL/dW
b <- b - learning_rate * dL/db
