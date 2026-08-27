"""Seeded Phase 0 debugging variants.

These are deliberately wrong.  Repair each variant only after preserving the
first diagnostic attempt and after predicting its failure signature in notes.
"""

import jax
import jax.numpy as jnp

from phase_0_diagnostic_attempt import linear_logits


def variant_1_global_reduction(params, x: jax.Array, y: jax.Array) -> jax.Array:
    """Deliberately normalises over every logit in the whole batch."""
    logits = linear_logits(params, x)
    log_normalizer = jax.nn.logsumexp(logits, axis=1, keepdims=True)
    log_probs = logits - log_normalizer
    correct = jnp.take_along_axis(log_probs, y[:, None], axis=1)
    return jnp.mean(-correct)


def variant_2_batch_shaped_params(key: jax.Array, x: jax.Array, n_classes: int):
    """Deliberately derives W/b dimensions from the batch surface."""
    batch_size, n_features = x.shape
    return {
        "W": jax.random.normal(key, (n_features, n_classes)),
        "b": jnp.zeros(n_classes),
    }


@jax.jit
def variant_3_jitted_loss(params, x: jax.Array, y: jax.Array) -> jax.Array:
    """Deliberately creates a data-dependent output shape while jitted."""
    n_classes = len(params["b"])
    logits = linear_logits(params, x)
    log_probs = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    targets = jax.nn.one_hot(y, n_classes)
    return -jnp.mean(jnp.sum(targets * log_probs, axis=-1))
