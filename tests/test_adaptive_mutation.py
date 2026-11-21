import ast
import json
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import struct

from jaxued import adaptive_mutation as am, mutation_checkpoint
from jaxued.environments.maze import make_level_generator, make_level_mutator_minimax


@struct.dataclass
class ExampleState:
    sampler: dict
    params: dict
    num_dr_updates: int
    num_replay_updates: int
    num_mutation_updates: int


class AdaptiveMutationTests(unittest.TestCase):
    def test_operators_valid_reproducible_and_same_as_fixed_edit_five(self):
        keys = jax.random.split(jax.random.PRNGKey(15), 32)
        parents = jax.vmap(make_level_generator(13, 13, 25))(keys)
        keys = jax.random.split(jax.random.PRNGKey(27), 32)
        operator = jnp.arange(32, dtype=jnp.int32) % 4
        mutate = make_level_mutator_minimax(100)
        batch = jax.jit(lambda keys, ids: am.mutate_batch(mutate, keys, parents, ids, am.DEFAULT_ARMS))
        first = batch(keys, operator)
        repeated = batch(keys, operator)
        for x, y in zip(jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(repeated)):
            np.testing.assert_array_equal(x, y)
        self.assertTrue(bool(jax.vmap(am.valid_child)(parents, first).all()))
        other = batch(jax.random.split(jax.random.PRNGKey(888), 32), operator)
        self.assertTrue(bool((first.wall_map != other.wall_map).any() | (first.goal_pos != other.goal_pos).any()))
        fixed = jax.vmap(mutate, (0, 0, None))(keys, parents, 5)
        fifth = batch(keys, jnp.full(32, 2))
        for x, y in zip(jax.tree_util.tree_leaves(fixed), jax.tree_util.tree_leaves(fifth)):
            np.testing.assert_array_equal(x, y)

    def test_fitness_rejects_invalid_parent_and_batch_duplicates(self):
        valid = jnp.array([False, True, True, True, True, True, True])
        parent_duplicate = jnp.array([False, True, False, False, False, False, False])
        same = jnp.eye(7, dtype=bool).at[2, 3].set(True).at[3, 2].set(True)
        stats = valid, parent_duplicate, same, jnp.ones(7)
        scores = jnp.array([99., 99., 99., 98., 1., 0., 1.])
        result = jax.jit(lambda s: am.assess(s, stats, "batch_rank", 169))(scores)
        np.testing.assert_array_equal(result, [0, 0, 0, 0, .75, 0, .75])
        diagnostic = am.assess(scores, stats, "changed_cells", 169)
        np.testing.assert_allclose(diagnostic, np.array([0, 0, 0, 0, 1, 1, 1]) / 169)

    def test_rank_handles_ties_missing_scores_and_singleton(self):
        rank = jax.jit(am.rank_rewards)
        np.testing.assert_array_equal(rank(jnp.ones(4), jnp.ones(4, dtype=bool)), [.5] * 4)
        scores = jnp.array([jnp.nan, -jnp.inf, jnp.inf, 3.0])
        np.testing.assert_array_equal(rank(scores, jnp.ones(4, dtype=bool)), [0, 0, 0, .5])
        np.testing.assert_array_equal(rank(scores, jnp.zeros(4, dtype=bool)), [0] * 4)

    def test_cold_start_covers_all_arms_before_exploitation(self):
        key = jax.random.PRNGKey(7)
        cold = jax.jit(lambda key: am.sample_arms(key, am.initialize(4), 32, 0.0))
        result = cold(key)
        np.testing.assert_array_equal(np.sort(result[:4]), np.arange(4))
        np.testing.assert_array_equal(result, cold(key))
        self.assertEqual(len(np.unique(am.sample_uniform(key, 128, 4))), 4)

    def test_epsilon_exploitation_and_random_ties(self):
        state = {**am.initialize(4), "count": jnp.ones(4, dtype=jnp.int32)}
        best = {**state, "ema_reward": jnp.array([0., .1, .9, .3])}
        draw = jax.jit(lambda s, e: am.sample_arms(jax.random.PRNGKey(142), s, 256, e))
        np.testing.assert_array_equal(draw(best, 0.0), np.full(256, 2))
        self.assertEqual(len(np.unique(draw(best, 1.0))), 4)
        self.assertEqual(len(np.unique(draw(state, 0.0))), 4)

    def test_ema_updates_once_per_arm_and_ignores_unused(self):
        state = {**am.initialize(4), "ema_reward": jnp.array([.6, .1, .8, .3])}
        state = jax.jit(am.update)(state, jnp.array([0, 0, 1, 3]), jnp.array([1., .2, .5, 0.]), .1, 8)
        np.testing.assert_array_equal(state["count"], [2, 1, 0, 1])
        np.testing.assert_array_equal(state["last_selected"], [8, 8, -1, 8])
        np.testing.assert_allclose(state["reward_sum"], [1.2, .5, 0, 0], atol=1e-6)
        np.testing.assert_allclose(state["ema_reward"], [.6, .14, .8, .27], atol=1e-6)

    def test_preferences_adapt_after_reward_shift(self):
        key, state = jax.random.PRNGKey(74), am.initialize(4)
        pull = jax.jit(lambda key, state: am.sample_arms(key, state, 32, .15))
        update = jax.jit(am.update)
        for wanted in (1, 3):
            for step in range(96):
                key, next_key = jax.random.split(key)
                chosen = pull(next_key, state)
                state = update(state, chosen, (chosen == wanted).astype(jnp.float32), .1, step)
            self.assertEqual(int(state["ema_reward"].argmax()), wanted)
            self.assertGreater(float((pull(key, state) == wanted).mean()), .7)
        self.assertTrue(bool(jnp.isfinite(state["ema_reward"]).all()))

    def test_invalid_slot_substitutes_parent_without_buffer_insertion(self):
        def add(sampler, levels, values):
            return sampler + (levels * values.sum(axis=0)).sum()
        result = jax.jit(lambda mask: am.insert_valid(
            add, 0, jnp.array([7, 13, 5]), mask, (jnp.ones((2, 3), dtype=jnp.int32),)
        ))(jnp.array([True, False, True]))
        self.assertEqual(int(result), 24)
        actual = am.safe_children(jnp.array([1, 2, 3]), jnp.array([9, 8, 7]), jnp.array([True, False, True]))
        np.testing.assert_array_equal(actual, [9, 2, 7])

    def test_resume_restores_rng_history_and_next_selection(self):
        bandit = am.update(am.initialize(4), jnp.arange(4), jnp.array([.1, .9, .5, .2]), .1, 2)
        state = ExampleState({"mutation_bandit": bandit}, {"weight": jnp.ones(2)}, 1, 1, 1)
        template = state.replace(sampler={"mutation_bandit": am.initialize(4)})
        config = {"mutation_selection": "bandit", "use_accel": True, "num_updates": 6, "seed": 0}
        key = jax.random.PRNGKey(65)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
            checkpointer = ocp.StandardCheckpointer()
            try:
                checkpointer.save(directory / "models/0/default", mutation_checkpoint.payload(key, state))
            finally:
                checkpointer.close()
            rng, restored, completed = mutation_checkpoint.restore(directory, -1, template, config)
            with self.assertRaisesRegex(ValueError, "bandit_epsilon"):
                mutation_checkpoint.restore(directory, 0, template, {**config, "bandit_epsilon": .7})
        self.assertEqual(completed, 3)
        np.testing.assert_array_equal(rng, key)
        restored_bandit = restored.sampler["mutation_bandit"]
        for name in bandit:
            np.testing.assert_array_equal(bandit[name], restored_bandit[name])
        np.testing.assert_array_equal(am.sample_arms(key, bandit, 32, .15),
                                      am.sample_arms(rng, restored_bandit, 32, .15))

    def test_legacy_fixed_migration_is_explicit(self):
        from flax.serialization import to_state_dict

        state = ExampleState({}, {"weight": jnp.ones(2)}, 1, 1, 1)
        template = state.replace(sampler={"mutation_bandit": am.initialize(4)})
        config = {"use_accel": True, "num_updates": 6, "seed": 0}
        request = {**config, "mutation_selection": "bandit", "initialize_bandit_on_resume": True}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
            checkpointer = ocp.StandardCheckpointer()
            try:
                checkpointer.save(directory / "models/0/default", to_state_dict(state))
            finally:
                checkpointer.close()
            with self.assertRaisesRegex(ValueError, "mutation_selection"):
                mutation_checkpoint.restore(directory, 0, template, {**request, "initialize_bandit_on_resume": False})
            with self.assertRaisesRegex(ValueError, "legacy checkpoint"):
                mutation_checkpoint.restore(directory, 0, template, request)
            _, result, _ = mutation_checkpoint.restore(directory, 0, template, {**request, "allow_legacy_resume": True})
        np.testing.assert_array_equal(result.sampler["mutation_bandit"]["count"], np.zeros(4))
        np.testing.assert_array_equal(result.params["weight"], state.params["weight"])

    def test_mutation_branch_contains_one_student_rollout(self):
        script = Path(__file__).resolve().parents[1] / "examples/maze_plr.py"
        tree = ast.parse(script.read_text(encoding="utf-8"))
        branch = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "on_mutate_levels")
        rollouts = [node for node in ast.walk(branch) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == "sample_trajectories_rnn"]
        self.assertEqual(len(rollouts), 1)
