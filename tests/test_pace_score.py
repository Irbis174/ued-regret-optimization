import unittest

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from jaxued.pace_score import compute_pace_pg_scores, rollout_statistics


def recurrent_policy(params, inputs, carry):
    obs, last_dones = inputs

    def step(memory, row):
        signal, reset = row
        memory = jnp.where(reset, 0, memory) + signal
        return memory, memory

    carry, memory = jax.lax.scan(step, carry, (obs["signal"], last_dones))
    logits = params["actor"] * memory
    return carry, distrax.Categorical(logits=jnp.stack([logits, -logits], axis=-1)), params["critic"] * memory


class TestPaceScore(unittest.TestCase):
    def setUp(self):
        self.params = {"actor": jnp.float32(0.31), "critic": jnp.float32(-0.77)}
        self.obs = {"signal": jnp.array([[0.3, 0.4, 0.1], [1.3, 0.8, 0.2], [-0.5, 0.3, 1.0]])}
        self.actions = jnp.array([[0, 1, 0], [0, 0, 1], [1, 0, 1]])
        self.dones = jnp.array([[True, False, False], [False, True, False], [True, True, True]])
        self.advantages = jnp.array([[1.0, -1.0, 0.2], [0.2, 0.4, -0.3], [0.9, 0.5, -0.7]])
        self.carry = jnp.array([0.7, -0.4, 0.3])

    def scorer(self, **options):
        return jax.jit(lambda params, *batch: compute_pace_pg_scores(recurrent_policy, params, *batch, **options))

    def test_norm_matches_clipped_ppo_at_initial_policy_and_virtual_sgd(self):
        scores = self.scorer()(self.params, self.obs, self.actions, self.dones, self.advantages, self.carry)
        self.assertEqual(scores.shape, (3,))
        self.assertEqual(scores.dtype, jnp.float32)
        self.assertTrue(np.isfinite(scores).all() and (scores >= 0).all())
        self.assertEqual(len(np.unique(scores)), 3)
        norm_adv = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-5)
        last_dones = jnp.concatenate([jnp.zeros_like(self.dones[:1]), self.dones[:-1]], axis=0)
        _, old, _ = recurrent_policy(self.params, (self.obs, last_dones), self.carry)
        old_log_prob = old.log_prob(self.actions)
        alpha = 0.07
        for index, score in enumerate(scores):
            def clipped(params):
                _, policy, _ = recurrent_policy(params, (self.obs, last_dones), self.carry)
                ratio = jnp.exp(policy.log_prob(self.actions)[:, index] - old_log_prob[:, index])
                adv = norm_adv[:, index]
                return -jnp.minimum(ratio * adv, jnp.clip(ratio, 0.8, 1.2) * adv).mean()

            grad = jax.grad(clipped)(self.params)
            self.assertEqual(grad["critic"], 0.0)
            np.testing.assert_allclose(score, grad["actor"] ** 2, rtol=1e-5)
            delta = self.params["actor"] - alpha * grad["actor"] - self.params["actor"]
            np.testing.assert_allclose(delta ** 2 / alpha, alpha * score, rtol=1e-5)

    def test_permutation_chunking_and_isolation_at_fixed_advantage_statistics(self):
        batch = (self.obs, self.actions, self.dones, self.advantages, self.carry)
        serial = self.scorer()
        scores = serial(self.params, *batch)
        np.testing.assert_allclose(self.scorer(chunk_size=2)(self.params, *batch), scores, rtol=1e-5)
        order = jnp.array([2, 0, 1])
        reordered = ({"signal": self.obs["signal"][:, order]},
                     *(x[:, order] for x in batch[1:-1]), self.carry[order])
        np.testing.assert_allclose(serial(self.params, *reordered), scores[order], rtol=1e-5)
        changed = {"signal": self.obs["signal"].at[:, 1].set(2.0)}
        new_scores = serial(self.params, changed, *batch[1:])
        np.testing.assert_array_equal(new_scores[::2], scores[::2])
        self.assertNotEqual(new_scores[1], scores[1])

    def test_zero_advantage_entropy_ablation_and_train_state_purity(self):
        state = TrainState.create(apply_fn=recurrent_policy, params=self.params, tx=optax.adam(0.1))
        before = [np.array(x) for x in jax.tree_util.tree_leaves(state)]
        batch = (self.obs, self.actions, self.dones, jnp.zeros_like(self.advantages), self.carry)
        np.testing.assert_array_equal(self.scorer()(state.params, *batch), jnp.zeros(3))
        self.assertTrue((self.scorer(entropy_coeff=0.01)(state.params, *batch) > 0).all())
        for expected, actual in zip(before, jax.tree_util.tree_leaves(state)):
            np.testing.assert_array_equal(expected, actual)

    def test_diagnostics_report_nonfinite_scores_and_successes(self):
        scores = jnp.array([0.0, jnp.nan, 0.5])
        rewards = jnp.eye(3)
        stats = jax.jit(rollout_statistics)(scores, self.advantages, self.dones, rewards)
        self.assertAlmostEqual(float(stats["pace/nonfinite_fraction"]), 1 / 3)
        self.assertAlmostEqual(float(stats["pace/duplicate_score_fraction"]), 1 / 3)
        self.assertAlmostEqual(float(stats["pace/levels_with_success_fraction"]), 1.0)
        self.assertTrue(all(np.isfinite(value) for value in stats.values()))
