uv run python - <<'PY'
import jax
import jax.numpy as jnp
from session_01_linear_classifier import train, linear_logits

p, x, y, initial, final = train(jax.random.key(42))
acc = jnp.mean(jnp.argmax(linear_logits(p, x), axis=1) == y)
print(f"initial_loss={float(initial):.6f}")
print(f"final_loss={float(final):.6f}")
print(f"accuracy={float(acc):.3f}")
PY
