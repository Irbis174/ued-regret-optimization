import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jaxued import learnability
from jaxued.level_sampler import LevelSampler
from jaxued.utils import compute_max_returns


def evidence(successes, episodes, max_return=0.0, maxmc=0.0):
    return learnability.RolloutEvidence(*map(jnp.asarray, (successes, episodes, max_return, maxmc)))


class TestLearnability(unittest.TestCase):
    def test_counts_only_completed_episodes(self):
        dones = jnp.array([[True, False], [False, True], [True, False]])
        rewards = jnp.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.9]])
        result = learnability.rollout_evidence(dones, rewards, jnp.zeros_like(rewards), compute_max_returns(dones, rewards))
        np.testing.assert_array_equal(result.episodes, [2, 1])
        np.testing.assert_array_equal(result.successes, [1, 0])

    def test_mature_score_peaks_at_half_success(self):
        score = jax.jit(lambda row: learnability.record_evidence(learnability.empty_history(), -jnp.inf, row, 0)[0])
        np.testing.assert_allclose([score(evidence(k, 4)) for k in (0, 2, 4)], [0.0, 1.0, 0.0])

    def test_cold_start_and_old_evidence(self):
        record = jax.jit(learnability.record_evidence)
        score, history = record(learnability.empty_history(), -jnp.inf, evidence(1, 1, 0.8), 0)
        self.assertAlmostEqual(float(score), 0.375)
        _, history = record(history, score, evidence(0, 1), learnability.HALF_LIFE)
        self.assertAlmostEqual(float(history["successes"]), 0.5)
        self.assertAlmostEqual(float(history["episodes"]), 1.5)
        self.assertAlmostEqual(float(history["max_return"]), 0.8)
        unchanged_score, unchanged_history = record(history, 0.25, evidence(0, 0), 999)
        self.assertAlmostEqual(float(unchanged_score), 0.25)
        self.assertEqual(int(unchanged_history["last_round"]), learnability.HALF_LIFE)

    def test_duplicate_insert_and_replay_keep_all_evidence(self):
        level_sampler = LevelSampler(capacity=4, duplicate_check=True)
        sampler = level_sampler.initialize(jnp.int32(0), learnability.empty_history())
        rows = evidence(jnp.array([1, 0, 0]), jnp.ones(3, dtype=jnp.int32), jnp.array([1.0, 0.0, 0.0]), jnp.zeros(3))
        sampler = learnability.insert_batch(level_sampler, sampler, jnp.array([1, 1, 2]), rows, 0)
        self.assertEqual(int(sampler["size"]), 2)
        np.testing.assert_array_equal(sampler["levels_extra"]["episodes"][:2], [2, 1])
        rows = evidence(jnp.array([1, 0]), jnp.ones(2, dtype=jnp.int32), jnp.array([1.0, 0.0]), jnp.zeros(2))
        sampler = learnability.update_batch(level_sampler, sampler, jnp.array([0, 0]), rows, 0)
        self.assertEqual(float(sampler["levels_extra"]["episodes"][0]), 4)
        self.assertEqual(float(sampler["levels_extra"]["successes"][0]), 2)
        weights = np.asarray(level_sampler.level_weights(sampler))
        self.assertTrue(np.all(weights[:2] > 0))
        np.testing.assert_array_equal(weights[2:], 0)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
