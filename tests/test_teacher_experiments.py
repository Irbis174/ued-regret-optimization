"""JAX CPU tests for additive Teacher research helpers, no PPO training."""
import unittest

import jax
import jax.numpy as jnp

from jaxued.teacher_experiments import (
    har_score,
    sample_replay_capped,
    update_progress_memory,
)


class TeacherExperimentTests(unittest.TestCase):
    def test_duplicates_aggregate_before_ema(self):
        fast, slow, init = jax.jit(update_progress_memory)(
            jnp.zeros(4), jnp.zeros(4), jnp.zeros(4, dtype=bool),
            jnp.array([1, 1, 2, -1]),
            jnp.array([1, 0, 0, 1]),
            jnp.array([1, 1, 0, 1]),
        )
        self.assertAlmostEqual(float(fast[1]), 0.5)
        self.assertAlmostEqual(float(slow[1]), 0.5)
        self.assertTrue(bool(init[1]))
        self.assertFalse(bool(init[0]))
        self.assertFalse(bool(init[2]))

    def test_first_observation_no_fake_progress(self):
        fast, slow, _ = jax.jit(update_progress_memory)(
            jnp.zeros(3), jnp.zeros(3), jnp.zeros(3, dtype=bool),
            jnp.array([1]), jnp.array([1]), jnp.array([1]),
        )
        self.assertAlmostEqual(float(fast[1] - slow[1]), 0.0)

    def test_missing_completed_episode_does_not_update(self):
        fast, slow, init = jax.jit(update_progress_memory)(
            jnp.array([0.2, 0.8]), jnp.array([0.1, 0.4]),
            jnp.array([True, True]), jnp.array([1]),
            jnp.array([0]), jnp.array([0]),
        )
        self.assertAlmostEqual(float(fast[1]), 0.8, places=6)
        self.assertAlmostEqual(float(slow[1]), 0.4, places=6)
        self.assertTrue(bool(init[1]))

    def test_positive_learning_progress_and_later_regression(self):
        fn = jax.jit(update_progress_memory)
        fast, slow, init = jnp.zeros(2), jnp.zeros(2), jnp.zeros(2, dtype=bool)
        fast, slow, init = fn(fast, slow, init, jnp.array([1]), jnp.array([0]), jnp.array([1]))
        fast, slow, init = fn(fast, slow, init, jnp.array([1]), jnp.array([1]), jnp.array([1]))
        self.assertGreater(float(fast[1]), float(slow[1]))
        for _ in range(13):
            fast, slow, init = fn(fast, slow, init, jnp.array([1]), jnp.array([0]), jnp.array([1]))
        self.assertLess(float(fast[1]), float(slow[1]))

    def test_history_weight_zero_recovers_baseline(self):
        mx = jnp.array([0, 1, -1], dtype=jnp.float32)
        successes = jnp.array([2, 0, 1], dtype=jnp.float32)
        completed = jnp.array([4, 4, 2], dtype=jnp.float32)
        score = jax.jit(lambda: har_score(
            mx, successes, completed, jnp.ones(3), jnp.zeros(3), history_weight=0.0
        ))()
        p = successes / completed
        w = jnp.minimum(completed / 4, 1)
        expected = (1 - w) * jax.nn.sigmoid(mx) + w * 4 * p * (1 - p)
        self.assertTrue(bool(jnp.allclose(score, expected, atol=1e-7)))

    def test_no_history_uses_maxmc_fallback(self):
        score = jax.jit(lambda: har_score(
            jnp.array([1.]), jnp.array([0.]), jnp.array([0.]),
            jnp.array([0.]), jnp.array([0.])
        ))()
        self.assertAlmostEqual(float(score[0]), float(jax.nn.sigmoid(1.)), places=6)

    def test_history_affects_score(self):
        scorer = jax.jit(lambda fast: har_score(
            jnp.array([1.0]), jnp.array([2.0]), jnp.array([4.0]),
            jnp.array([fast]), jnp.array([0.2]), history_weight=0.2,
        ))
        self.assertGreater(float(scorer(0.8)[0]), float(scorer(0.2)[0]))

    def test_sampler_cap_and_jit(self):
        w = jnp.array([0.61, 0.10, 0.05, 0.04, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03])
        frontier = jnp.array([True] * 4 + [False] * 6)
        fn = jax.jit(lambda rng: sample_replay_capped(
            rng, w, frontier, batch_size=32, max_copies=8
        ))
        for seed in range(5):
            indices, counts = fn(jax.random.PRNGKey(seed))
            self.assertEqual(indices.shape, (32,))
            self.assertEqual(int(counts.sum()), 32)
            self.assertLessEqual(int(counts.max()), 8)
            self.assertTrue(bool(jnp.all(w[indices] > 0)))

    def test_sampler_falls_back_if_cap_impossible(self):
        w = jnp.array([0.8, 0.2, 0.0])
        fn = jax.jit(lambda rng: sample_replay_capped(
            rng, w, jnp.array([True, False, False]), batch_size=32, max_copies=8
        ))
        indices, counts = fn(jax.random.PRNGKey(7))
        self.assertEqual(indices.shape, (32,))
        self.assertEqual(int(counts.sum()), 32)
        self.assertTrue(bool(jnp.all(w[indices] > 0)))

    def test_diversity_of_concentrated_weights(self):
        w = jnp.array([0.61] + [0.39 / 19] * 19)
        frontier = jnp.array([True] * 6 + [False] * 14)
        fn = jax.jit(lambda rng: sample_replay_capped(
            rng, w, frontier, batch_size=32, max_copies=8
        ))
        uniques = []
        for seed in range(10):
            indices, _ = fn(jax.random.PRNGKey(seed))
            uniques.append(len(set(map(int, indices))))
        self.assertGreater(sum(uniques) / len(uniques), 5.0)


if __name__ == '__main__':
    unittest.main()
