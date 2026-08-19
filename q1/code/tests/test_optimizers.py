"""Regression tests for the maintained NumPy optimizers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.optimizers import Adam, MomentumSGD, SGD  # noqa: E402


class OptimizerTests(unittest.TestCase):
    def test_sgd_applies_gradient_and_decoupled_weight_decay(self) -> None:
        params = np.array([1.0, -2.0])
        updated = SGD(learning_rate=0.1, weight_decay=0.2).step(
            params, np.array([0.5, -1.0])
        )

        np.testing.assert_allclose(updated, [0.93, -1.86])
        np.testing.assert_allclose(params, [1.0, -2.0])

    def test_momentum_accumulates_velocity_and_can_reset_it(self) -> None:
        optimizer = MomentumSGD(learning_rate=0.1, beta=0.5)
        first = optimizer.step(np.array([1.0, -2.0]), np.array([2.0, -4.0]))
        second = optimizer.step(first, np.array([4.0, -2.0]))

        np.testing.assert_allclose(first, [0.8, -1.6])
        np.testing.assert_allclose(second, [0.3, -1.2])

        optimizer.reset_state()
        restarted = optimizer.step(np.array([1.0, -2.0]), np.array([2.0, -4.0]))
        np.testing.assert_allclose(restarted, first)

    def test_adam_uses_bias_correction_and_reset_restarts_time(self) -> None:
        optimizer = Adam(learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8)
        params = np.array([1.0, -2.0])
        first = optimizer.step(params, np.array([2.0, -4.0]))

        # At t=1, bias correction makes m_hat=g and v_hat=g**2.
        np.testing.assert_allclose(first, [0.9, -1.9])
        self.assertEqual(optimizer.step_count, 1)

        optimizer.step(first, np.array([1.0, -1.0]))
        optimizer.reset_state()
        restarted = optimizer.step(params, np.array([2.0, -4.0]))
        np.testing.assert_allclose(restarted, first)
        self.assertEqual(optimizer.step_count, 1)

    def test_rejects_mismatched_parameter_and_gradient_shapes(self) -> None:
        with self.assertRaises(ValueError):
            SGD(learning_rate=0.1).step(np.zeros(2), np.zeros(3))


if __name__ == "__main__":
    unittest.main()
