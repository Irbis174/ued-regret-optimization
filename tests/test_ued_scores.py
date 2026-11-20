import unittest

import jax.numpy as jnp
import numpy as np

from jaxued.utils import compute_max_returns, max_mc, positive_value_loss


class TestUedScores(unittest.TestCase):
    def test_scores_on_two_levels(self):
        # The columns are a solved level and a level that timed out.
        dones = jnp.array([[False, False], [False, False], [True, True]])
        rewards = jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        values = jnp.array([[0.2, 0.2], [0.3, 0.2], [0.5, 0.2]])
        advantages = jnp.array([[0.2, 0.4], [-0.1, 0.2], [0.3, -0.1]])

        max_returns = compute_max_returns(dones, rewards)
        max_mc_scores = max_mc(dones, values, max_returns)
        pvl_scores = positive_value_loss(dones, advantages)

        np.testing.assert_allclose(max_returns, [1.0, 0.0])
        np.testing.assert_allclose(max_mc_scores, [2 / 3, -0.2])
        np.testing.assert_allclose(pvl_scores, [1 / 6, 0.2])
