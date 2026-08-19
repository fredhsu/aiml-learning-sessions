"""Regression tests for the maintained scalar autograd engine."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.nn import create_mlp  # noqa: E402
from engine.value import cross_entropy_from_logits, values  # noqa: E402


class ValueEngineTests(unittest.TestCase):
    def test_composite_expression_matches_finite_differences(self) -> None:
        def scalar_loss(x: float, y: float) -> float:
            sigmoid_x = 1.0 / (1.0 + math.exp(-x))
            term = (
                math.log(x * y + 3.0)
                + x / y
                + (2.0 - x)
                + 2.0 / x
                - 0.25 * x
                + x**3
                + sigmoid_x
                + math.exp(x) * 0.1
                + (3.0 + x)
                + max(0.0, x - 0.1)
            )
            return term**2

        x, y = values([1.25, -0.4], label="input")
        term = (
            (x * y + 3.0).log()
            + x / y
            + (2.0 - x)
            + 2.0 / x
            - 0.25 * x
            + x**3
            + x.sigmoid()
            + 0.1 * x.exp()
            + (3.0 + x)
            + (x - 0.1).relu()
        )
        loss = term.square()
        loss.backward()

        epsilon = 1e-6
        dx = (scalar_loss(x.data + epsilon, y.data) - scalar_loss(x.data - epsilon, y.data)) / (2 * epsilon)
        dy = (scalar_loss(x.data, y.data + epsilon) - scalar_loss(x.data, y.data - epsilon)) / (2 * epsilon)

        self.assertAlmostEqual(x.grad, dx, places=5)
        self.assertAlmostEqual(y.grad, dy, places=5)

    def test_mlp_exposes_parameters_and_backpropagates(self) -> None:
        model = create_mlp(2, [3, 2], seed=0)
        logits = model(values([0.5, -1.0], label="x"))
        loss = cross_entropy_from_logits(logits, target_index=1)
        loss.backward()

        parameters = model.parameters()
        self.assertEqual(len(parameters), 17)
        self.assertTrue(all(math.isfinite(parameter.grad) for parameter in parameters))

        model.zero_grad()
        self.assertTrue(all(parameter.grad == 0.0 for parameter in parameters))


if __name__ == "__main__":
    unittest.main()
