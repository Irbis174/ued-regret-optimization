"""Check the shadow objective against the current Maze PPO implementation."""

from functools import partial
import unittest

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from examples.maze_plr import ActorCritic, update_actor_critic_rnn
from jaxued import pace_shadow
from jaxued.environments.maze.env import Observation
from jaxued.mutation_checkpoint import validate_config


CLIP = 0.2
ENTROPY = 1e-3
ALPHA = 1e-4


class MazeShadowLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = jax.random.PRNGKey(12)
        t, b = 7, 3
        cls.net = ActorCritic(7)
        cls.obs = Observation(
            image=jax.random.uniform(rng, (t, b, 5, 5, 3)),
            agent_dir=jnp.arange(t * b).reshape(t, b) % 4,
        )
        cls.carry = tuple(
            jax.random.normal(jax.random.fold_in(rng, i), (b, 256)) * 0.1 for i in (1, 2)
        )
        cls.dones = jnp.zeros((t, b), dtype=bool).at[1, 0].set(True).at[3, 2].set(True).at[-1, 1].set(True)
        # Prepend explicitly so a terminal final step cannot reset the first input.
        cls.last_dones = jnp.concatenate((jnp.zeros((1, b), dtype=bool), cls.dones[:-1]), axis=0)
        cls.params = cls.net.init(rng, (cls.obs, cls.last_dones), cls.carry)
        _, policy, values = cls.net.apply(cls.params, (cls.obs, cls.last_dones), cls.carry)
        cls.actions = jnp.arange(t * b).reshape(t, b) % 7
        # Fixed offsets exercise both sides of PPO clipping.
        cls.log_probs = policy.log_prob(cls.actions) + 0.2 * (jnp.arange(t * b).reshape(t, b) % 5 - 2)
        cls.advantages = jnp.arange(t, dtype=jnp.float32)[:, None] ** 2 + jnp.array([-15., 7., 34.])
        cls.values = values
        cls.slots = jnp.arange(b)
        cls.score = staticmethod(jax.jit(partial(
            pace_shadow.rollout_scores, cls.net.apply, step_size=ALPHA, clip_eps=CLIP, entropy_coeff=ENTROPY,
        )))

    def reference_slot_loss(self, params, slot):
        # Rebuild the old minibatch actor formula on fixed inputs using all
        # slots together; average only the specified slot at the last step.
        _, policy, _ = self.net.apply(params, (self.obs, self.last_dones), self.carry)
        ratio = jnp.exp(policy.log_prob(self.actions) - self.log_probs)
        adv = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-5)
        actor = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - CLIP, 1 + CLIP) * adv)
        return actor[:, slot].mean() - ENTROPY * policy.entropy()[:, slot].mean()

    def test_mean_of_slot_losses_matches_real_ppo_actor_term(self):
        state = TrainState.create(apply_fn=self.net.apply, params=self.params, tx=optax.sgd(1e-4))
        _, (_, (_, clip_loss, entropy)) = update_actor_critic_rnn(
            jax.random.PRNGKey(33), state, self.carry,
            (self.obs, self.actions, self.dones, self.log_probs, self.values,
             self.values + self.advantages, self.advantages),
            num_envs=3, n_steps=7, n_minibatch=1, n_epochs=1,
            clip_eps=CLIP, entropy_coeff=ENTROPY, critic_coeff=0.5, update_grad=False,
        )
        adv = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-5)
        losses = []
        for slot in range(3):
            obs = jax.tree_util.tree_map(lambda x: x[:, slot], self.obs)
            carry = jax.tree_util.tree_map(lambda x: x[slot], self.carry)
            one_level = (carry, obs, self.actions[:, slot], self.last_dones[:, slot],
                         self.log_probs[:, slot], adv[:, slot])
            losses.append(pace_shadow.policy_loss(
                self.net.apply, self.params, one_level, clip_eps=CLIP, entropy_coeff=ENTROPY,
            ))
        np.testing.assert_allclose(np.mean(losses), (clip_loss - ENTROPY * entropy).item(), atol=1e-7)

    def test_scores_match_independent_per_slot_sgd(self):
        scores = self.score(self.params, self.carry, self.obs, self.actions, self.dones, self.log_probs,
                            self.advantages, self.slots)
        expected = []
        for slot in range(3):
            grads = jax.grad(self.reference_slot_loss)(self.params, slot)["params"]
            for name in ("critic0", "critic1"):
                for leaf in jax.tree_util.tree_leaves(grads[name]):
                    np.testing.assert_array_equal(leaf, 0)
            policy = pace_shadow.maze_policy_params(self.params)
            self.assertEqual(set(policy), {"Conv_0", "scalar_embed", "OptimizedLSTMCell_0", "actor0", "actor1"})
            for name in ("Conv_0", "scalar_embed", "OptimizedLSTMCell_0"):
                self.assertTrue(any(np.any(x != 0) for x in jax.tree_util.tree_leaves(grads[name])))
            delta_squares = []
            for name, old in policy.items():
                for weight, grad in zip(jax.tree_util.tree_leaves(old), jax.tree_util.tree_leaves(grads[name])):
                    weight, grad = np.asarray(weight), np.asarray(grad)
                    moved = weight - np.float32(ALPHA) * grad
                    delta_squares.append(np.square(moved - weight).sum())
            expected.append(sum(delta_squares) / ALPHA)
        np.testing.assert_allclose(scores, expected, rtol=2e-4, atol=1e-9)
        self.assertEqual(np.unique(scores).size, 3)
        self.assertTrue(np.all(np.isfinite(scores) & (scores > 0)))

    def test_other_trajectories_and_critic_weights_do_not_enter_slot_gradient(self):
        slots = jnp.array([0])
        args = (self.carry, self.obs, self.actions, self.dones, self.log_probs, self.advantages, slots)
        score = self.score(self.params, *args)
        params = {"params": dict(self.params["params"])}
        for name in ("critic0", "critic1"):
            params["params"][name] = jax.tree_util.tree_map(lambda x: x + 100, params["params"][name])
        edited_obs = jax.tree_util.tree_map(lambda x: x.at[:, 1:].set(0), self.obs)
        other = self.score(params, self.carry, edited_obs, self.actions.at[:, 1:].set(0), self.dones,
                           self.log_probs.at[:, 1:].set(-10), self.advantages, slots)
        np.testing.assert_array_equal(score, other)

    def test_slots_and_resume(self):
        self.assertEqual(pace_shadow.level_slots(32, 4), (0, 8, 16, 24))
        with self.assertRaises(ValueError):
            pace_shadow.level_slots(3, 4)
        validate_config({}, {"pace_shadow": True, "pace_shadow_every": 1, "pace_shadow_levels": 4,
                             "pace_shadow_alpha": ALPHA, "pace_shadow_loss": "policy_only",
                             "pace_shadow_log": "new.jsonl"}, False)

    def test_rollout_comparisons_keep_slot_order_and_skip_unfinished_episodes(self):
        dones = jnp.zeros((4, 3), dtype=bool).at[1::2, 0].set(True)
        rewards = jnp.zeros((4, 3)).at[1, 0].set(1).at[-1, 2].set(1)
        values = jnp.full((4, 3), 0.25)
        advantages = jnp.broadcast_to(jnp.array([2., -3., 7.]), (4, 3))
        observed = pace_shadow.rollout_comparisons(dones, rewards, values, advantages, jnp.array([2, 0]))
        np.testing.assert_array_equal(observed["completed_episodes"], [0, 2])
        np.testing.assert_array_equal(observed["solved_episodes"], [0, 1])
        np.testing.assert_allclose(observed["maxmc_rollout"], [-np.inf, 0.75])
        np.testing.assert_allclose(observed["pvl"], [-np.inf, 2])
        np.testing.assert_allclose(observed["recent_rollout"], [-np.inf, 0.5 + 0.5 / (1 + np.exp(-0.75))])
