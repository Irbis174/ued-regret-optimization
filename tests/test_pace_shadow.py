"""CPU regression tests for the independent PACE shadow scoring kernel."""
import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from jaxued.pace_shadow import (
    gradient_norm_score,
    pace_score,
    sgd_provisional_update,
    shadow_scores_scan,
)


class PaceShadowTests(unittest.TestCase):
    def test_exact_squared_delta(self):
        before = {"actor": jnp.array([1.0, 2.0]), "critic": jnp.array([3.0])}
        after = {"actor": jnp.array([2.0, 4.0]), "critic": jnp.array([3.0])}
        self.assertAlmostEqual(float(pace_score(before, after, step_size=0.5)), 10.0)

    def test_rejects_invalid_alpha(self):
        with self.assertRaises(ValueError):
            pace_score({"w": jnp.ones(2)}, {"w": jnp.zeros(2)}, step_size=0)

    def test_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            pace_score({"w": jnp.ones(2)}, {"w": jnp.zeros(3)}, step_size=0.1)

    def test_selector_excludes_critic(self):
        before = {"actor": jnp.array([1.0]), "critic": jnp.array([0.0])}
        after = {"actor": jnp.array([2.0]), "critic": jnp.array([100.0])}
        self.assertAlmostEqual(
            float(pace_score(before, after, step_size=0.5, select=lambda x: x["actor"])), 2.0
        )

    def test_pure_sgd_and_gradient_identity(self):
        params = {"w": jnp.array([1.0, 2.0])}
        grads = {"w": jnp.array([3.0, -4.0])}
        alpha = 0.1
        after = sgd_provisional_update(params, grads, step_size=alpha)
        score = pace_score(params, after, step_size=alpha)
        self.assertAlmostEqual(float(score), 2.5, places=5)
        self.assertAlmostEqual(float(score), float(gradient_norm_score(grads, step_size=alpha)), places=5)
        self.assertTrue(jnp.array_equal(params["w"], jnp.array([1.0, 2.0])))

    def test_independent_levels_and_jit(self):
        params = {"w": jnp.array([1.0, -1.0])}
        # Each element is the fixed "trajectory" for one level.
        x = jnp.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])

        def update(old, one_level):
            def loss(p):
                return jnp.square(jnp.dot(p["w"], one_level) - 1.0)
            grad = jax.grad(loss)(old)
            return sgd_provisional_update(old, grad, step_size=0.1)

        fn = jax.jit(lambda p, batch: shadow_scores_scan(p, batch, update, step_size=0.1))
        scores = fn(params, x)
        self.assertEqual(scores.shape, (3,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(scores))))
        self.assertFalse(bool(jnp.all(scores == scores[0])))
        self.assertTrue(jnp.array_equal(params["w"], jnp.array([1.0, -1.0])))
        self.assertTrue(bool(jnp.array_equal(scores, fn(params, x))))

    def test_works_with_nested_pytree(self):
        params = {"a": {"w": jnp.array([1.0, 2.0])}, "b": jnp.array(0.25)}
        after = jax.tree_util.tree_map(lambda p: p + 0.2, params)
        score = jax.jit(lambda x, y: pace_score(x, y, step_size=0.1))(params, after)
        self.assertAlmostEqual(float(score), 1.2, places=4)


if __name__ == "__main__":
    unittest.main()
