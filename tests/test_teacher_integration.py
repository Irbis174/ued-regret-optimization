import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from jaxued import learnability, teacher_experiments
from jaxued.level_sampler import LevelSampler


def evidence(successes, episodes, maxmc=0.0):
    episodes = jnp.asarray(episodes)
    return learnability.RolloutEvidence(
        jnp.asarray(successes), episodes, jnp.zeros(episodes.shape),
        jnp.broadcast_to(jnp.float32(maxmc), episodes.shape),
    )


class TeacherIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.api = LevelSampler(4, duplicate_check=True)

    def insert(self, sampler, levels, observations, round_index=0, weight=0.2):
        return jax.jit(lambda state, batch, rows: learnability.insert_batch(
            self.api, state, batch, rows, round_index, history_weight=weight
        ))(sampler, jnp.array(levels), observations)

    def test_har_updates_once_per_round_for_repeated_indices(self):
        state = self.api.initialize(jnp.int32(0), learnability.empty_history(track_progress=True))
        state = self.insert(state, [1, 1], evidence([1, 0], [1, 1]))
        extra = state["levels_extra"]
        self.assertEqual(int(state["size"]), 1)
        self.assertTrue(bool(extra["initialized"][0]))
        self.assertEqual(float(extra["fast_ema"][0]), 0.5)
        self.assertEqual(float(extra["slow_ema"][0]), 0.5)

        update = jax.jit(lambda state, rows, t: learnability.update_batch(
            self.api, state, jnp.array([0, 0]), rows, t, history_weight=0.2
        ))
        state = update(state, evidence([2, 0], [2, 1]), 1)
        extra = state["levels_extra"]
        self.assertAlmostEqual(float(extra["fast_ema"][0]), 0.55, places=6)
        self.assertAlmostEqual(float(extra["slow_ema"][0]), 0.5 + 0.05 / 6, places=6)
        expected = float(teacher_experiments.har_score(
            0., extra["successes"][0], extra["episodes"][0], extra["fast_ema"][0], extra["slow_ema"][0]
        ))
        self.assertAlmostEqual(float(state["scores"][0]), expected, places=6)
        unchanged = update(state, evidence([0, 0], [0, 0]), 2)
        for old, new in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(unchanged)):
            np.testing.assert_array_equal(old, new)

    def test_har_memory_resets_on_eviction_and_round_trips(self):
        self.api = LevelSampler(1, duplicate_check=True)
        state = self.api.initialize(jnp.int32(0), learnability.empty_history(track_progress=True))
        state = self.insert(state, [1], evidence([0], [1]))
        state = self.insert(state, [2], evidence([1], [1], 2.0), round_index=1)
        self.assertEqual(int(state["levels"][0]), 2)
        extra = state["levels_extra"]
        for key in ("fast_ema", "slow_ema", "episodes", "successes"):
            np.testing.assert_array_equal(extra[key], [1.0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher"
            checkpointer = ocp.StandardCheckpointer()
            try:
                checkpointer.save(path, {"sampler": state})
                restored = checkpointer.restore(path)["sampler"]
            finally:
                checkpointer.close()
        for old, new in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(restored)):
            np.testing.assert_array_equal(old, new)

    def test_weight_zero_matches_recent_learnability(self):
        baseline = self.api.initialize(jnp.int32(0), learnability.empty_history())
        har = self.api.initialize(jnp.int32(0), learnability.empty_history(track_progress=True))
        rows = evidence([1, 0, 1], [1, 1, 2], 0.3)
        baseline = self.insert(baseline, [1, 1, 2], rows, weight=None)
        har = self.insert(har, [1, 1, 2], rows, weight=0.0)
        for weight, state in ((None, baseline), (0.0, har)):
            updated = learnability.update_batch(
                self.api, state, jnp.array([1, 0, 0]), rows, 256, history_weight=weight
            )
            if weight is None:
                expected = updated
            else:
                np.testing.assert_array_equal(updated["scores"], expected["scores"])
                for key in expected["levels_extra"]:
                    np.testing.assert_array_equal(updated["levels_extra"][key], expected["levels_extra"][key])

    def test_frontier_reweighting_and_zero_weight_fallback(self):
        weights = jnp.array([0.6, 0.1, 0.2, 0.1, 0.0])
        frontier = jnp.array([True, True, False, False, False])
        fn = jax.jit(lambda eligible: teacher_experiments.capped_replay_weights(
            weights, frontier, eligible, preserve_frontier_mass=True
        ))
        probs = fn(jnp.array([False, True, True, True, False]))
        self.assertTrue(bool(jnp.isfinite(probs).all()))
        np.testing.assert_allclose([probs.sum(), probs[:2].sum()], [1.0, 0.7])
        np.testing.assert_array_equal(fn(jnp.array([False, False, False, False, True])), [0, 0, 0, 0, 1])

    def test_capped_replay_updates_timestamps_and_never_draws_empty_slots(self):
        api = LevelSampler(8, replay_max_copies=3, replay_preserve_frontier_mass=True)
        sampler = api.initialize(jnp.int32(-99))
        sampler, _ = api.insert_batch(sampler, jnp.arange(4), jnp.array([9.0, 2.0, 1.0, 0.0]))
        draw = jax.jit(lambda state, key: api.sample_replay_levels(state, key, 12))
        state, (indices, levels) = draw(sampler, jax.random.PRNGKey(4))
        np.testing.assert_array_equal(levels, indices)
        np.testing.assert_array_equal(np.bincount(indices), [3, 3, 3, 3])
        self.assertEqual(int(state["episode_count"]), 16)
        stamps = np.asarray(sampler["timestamps"]).copy()
        for timestamp, index in enumerate(indices, start=5):
            stamps[int(index)] = timestamp
        np.testing.assert_array_equal(state["timestamps"], stamps)

    def test_small_and_empty_buffer_fallback(self):
        api = LevelSampler(5, replay_prob=1, minimum_fill_ratio=0, replay_max_copies=2)
        sampler = api.initialize(jnp.int32(-99))
        key = jax.random.PRNGKey(7)
        draw = jax.jit(lambda state: api.sample_replay_levels(state, key, 6))
        self.assertFalse(bool(api.sample_replay_decision(sampler, key)))
        empty, (indices, _) = draw(sampler)
        np.testing.assert_array_equal(indices, [-1] * 6)
        self.assertEqual(int(empty["episode_count"]), 0)
        sampler, _ = api.insert(sampler, jnp.int32(8), 0.25)
        updated, (indices, levels) = draw(sampler)
        np.testing.assert_array_equal(indices, [0] * 6)
        np.testing.assert_array_equal(levels, [8] * 6)
        self.assertEqual(int(updated["timestamps"][0]), 7)

    def test_disabled_replay_matches_direct_scan(self):
        sampler = self.api.initialize(jnp.int32(0))
        sampler, _ = self.api.insert_batch(sampler, jnp.array([1, 2]), jnp.array([0.0, 0.1]))
        key = jax.random.PRNGKey(8)
        expected = jax.lax.scan(self.api.sample_replay_level, sampler, jax.random.split(key, 10))
        actual = self.api.sample_replay_levels(sampler, key, 10)
        for old, new in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(actual)):
            np.testing.assert_array_equal(old, new)
