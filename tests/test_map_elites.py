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

from jaxued import learnability, map_elites as qd, mutation_checkpoint
from jaxued.environments.maze.level import Level


@struct.dataclass
class ExampleState:
    sampler: dict
    params: dict
    num_dr_updates: int = 1
    num_replay_updates: int = 1
    num_mutation_updates: int = 1


class TestMapElites(unittest.TestCase):
    def setUp(self):
        self.corridor = Level.from_str("#####\n#>..#\n###.#\n###G#\n#####")
        self.room = Level.from_str("#####\n#>..#\n#...#\n#..G#\n#####")
        self.other = self.corridor.replace(agent_dir=jnp.uint8(1))
        self.empty = qd.initialize_archive(self.corridor, (1, 1))
        self.key = jax.random.PRNGKey(195)
        self.update = jax.jit(lambda archive, levels, quality, obs, step: qd.update_archive_batch(
            archive, levels, quality, obs, step, jnp.array([]), jnp.array([]), 4
        ))

    def assert_tree_equal(self, left, right):
        self.assertEqual(jax.tree_util.tree_structure(left), jax.tree_util.tree_structure(right))
        for x, y in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
            np.testing.assert_array_equal(x, y)

    def submit(self, archive, levels, scores, step, obs=None):
        if obs is None:
            obs = [1] * len(scores)
        return self.update(archive, Level.stack(levels), jnp.array(scores), jnp.array(obs), step)

    def test_descriptors_with_padding_and_playable_border(self):
        padded = self.corridor.replace(wall_map=jnp.pad(self.corridor.wall_map, ((0, 2), (0, 2))))
        opening = self.corridor.replace(wall_map=self.corridor.wall_map.at[0, 1].set(False))
        levels = Level.stack([self.corridor, opening, self.room])
        actual = jax.jit(jax.vmap(qd.compute_descriptors))(levels)
        np.testing.assert_allclose(actual, [[4/9, 2/5], [4/9, 1/5], [0., 0.]])
        np.testing.assert_allclose(qd.compute_descriptors(padded), actual[0])

    def test_niche_index_is_row_major_with_inclusive_lower_edges(self):
        descriptors = jnp.array([[0., 0.], [.199, .5], [.2, .5], [.4, 1.], [1., 1.]])
        actual = jax.jit(qd.get_niche_index)(descriptors, jnp.array([.2, .4]), jnp.array([.5]))
        np.testing.assert_array_equal(actual, [0, 1, 3, 5, 5])

    def test_fill_replace_and_refresh_quality_downward(self):
        archive = self.submit(self.empty, [self.corridor], [.5], 0)
        self.assertEqual((int(archive["occupied"].sum()), int(archive["num_updates"][0])), (1, 0))
        rejected = self.submit(archive, [self.other], [.4], 1)
        self.assert_tree_equal(archive, rejected)
        refreshed = self.submit(archive, [self.corridor, self.corridor], [.2, .2], 2, [1, 3])
        self.assertAlmostEqual(float(refreshed["quality"][0]), .2)
        self.assertEqual((int(refreshed["last_evaluated"][0]), int(refreshed["observations"][0])), (2, 3))
        self.assertEqual(int(refreshed["num_updates"][0]), 0)
        replaced = self.submit(refreshed, [self.other], [.3], 3)
        self.assertEqual(int(replaced["levels"].agent_dir[0]), 1)
        self.assertEqual(int(replaced["num_updates"][0]), 1)
        same_batch = self.submit(archive, [self.other, self.corridor], [.4, .2], 1)
        self.assertEqual(int(same_batch["levels"].agent_dir[0]), 1)

    def test_collision_order_and_ties(self):
        a = self.submit(self.empty, [self.other, self.corridor, self.other], [.9, .9, .9], 0)
        b = self.submit(self.empty, [self.corridor, self.other], [.9, .9], 0)
        self.assert_tree_equal(a, b)
        self.assertEqual(int(a["levels"].agent_dir[0]), 0)
        c = self.submit(self.empty, [self.corridor, self.other], [.1, .9], 0)
        self.assertEqual(int(c["levels"].agent_dir[0]), 1)
        tied = self.submit(c, [self.corridor], [.9], 0)
        self.assert_tree_equal(b["levels"], tied["levels"])
        self.assertEqual(int(tied["num_updates"][0]), 1)

    def test_ignore_incomplete_nonfinite_and_invalid_candidates(self):
        invalid = self.other.replace(agent_pos=self.other.goal_pos)
        unchanged = self.submit(self.empty, [self.corridor, self.other, invalid, self.room],
                                [1., np.nan, 1., np.inf], 0, [0, 1, 1, 1])
        self.assert_tree_equal(unchanged, self.empty)
        self.assertTrue(np.isfinite(list(qd.archive_statistics(unchanged, 0, 4).values())).all())

    def test_same_level_is_pooled_before_quality_comparison(self):
        levels = Level.stack([self.corridor, self.corridor, self.other, self.room])
        evidence = learnability.RolloutEvidence(jnp.array([1, 0, 0, 0]), jnp.array([1, 2, 2, 0]),
                                                jnp.array([.8, 0., 0., 0.]),
                                                jnp.array([.7, -.3, .3, -jnp.inf]))
        scores, obs = jax.jit(qd.rollout_quality)(levels, evidence, 12)
        pooled = learnability.RolloutEvidence(1, 3, .8, (.7 + 2 * .5) / 3)
        expected = learnability.record_evidence(learnability.empty_history(), 0., pooled, 12)[0]
        np.testing.assert_array_equal(obs, [3, 3, 2, 0])
        np.testing.assert_allclose(scores[:2], expected)
        self.assertEqual(float(scores[-1]), 0.)
        self.assertTrue(np.isfinite(scores).all())

    def test_stale_replacement_and_parent_fallback(self):
        archive = self.submit(self.empty, [self.corridor], [.9], 0)
        replay = Level.stack([self.other] * 3)
        for memory in (self.empty, archive):
            parents, indices = qd.sample_archive_parents(memory, replay, self.key, 5, 4, 1.)
            self.assert_tree_equal(parents, replay)
            np.testing.assert_array_equal(indices, [-1, -1, -1])
        self.assertEqual(int(qd.archive_statistics(archive, 5, 4)["qd/stale_elite_count"]), 1)
        archive = self.submit(archive, [self.other], [.1], 5)
        self.assertEqual(int(archive["last_evaluated"][0]), 5)
        self.assertAlmostEqual(float(archive["quality"][0]), .1)

    def test_uniform_niches_and_per_slot_mix_independent_of_replay(self):
        archive = qd.initialize_archive(self.corridor, (2, 1))
        archive = qd.update_archive_batch(archive, Level.stack([self.room, self.corridor]),
                                          jnp.array([.9, .1]), jnp.array([1, 1]), 0,
                                          jnp.array([.2]), jnp.array([]), 4)
        replay = Level.stack([self.other] * 512)
        alternate = replay.replace(agent_dir=jnp.full_like(replay.agent_dir, 3))
        sample = jax.jit(qd.sample_archive_parents)
        parents, indices = sample(archive, replay, self.key, 0, 4, .5)
        others, same = sample(archive, alternate, self.key, 0, 4, .5)
        np.testing.assert_array_equal(indices, same)
        mask = np.asarray(indices >= 0)
        self.assertAlmostEqual(mask.mean(), .5, delta=.08)
        self.assertAlmostEqual(float((indices[mask] == 0).mean()), .5, delta=.08)
        np.testing.assert_array_equal(parents.agent_dir[~mask], replay.agent_dir[~mask])
        np.testing.assert_array_equal(others.agent_dir[~mask], alternate.agent_dir[~mask])
        np.testing.assert_array_equal(qd.fingerprints(parents)[mask], qd.fingerprints(others)[mask])
        self.assert_tree_equal(sample(archive, replay, self.key, 0, 4, 0.)[0], replay)

    def test_random_control_ignores_finite_quality(self):
        levels = Level.stack([self.corridor, self.corridor, self.other, self.room])
        def apply(scores):
            return qd.update_archive_batch(self.empty, levels, scores, jnp.ones(4, jnp.int32),
                                           0, jnp.array([]), jnp.array([]), 4, random_key=self.key)
        left = jax.jit(apply)(jnp.array([.1, .1, .3, .9]))
        right = jax.jit(apply)(jnp.array([.9, .9, .3, .1]))
        self.assert_tree_equal(left["levels"], right["levels"])
        self.assertEqual(int(left["occupied"].sum()), 1)

    def test_orbax_restores_full_archive_rng_and_next_parent_selection(self):
        size = 25
        template_archive = qd.initialize_archive(self.corridor)
        archive = {**template_archive, "occupied": jnp.ones(size, bool),
                   "quality": jnp.linspace(0., 1., size), "last_evaluated": jnp.arange(size),
                   "num_updates": jnp.arange(size) * 2, "observations": jnp.arange(size) + 1,
                   "levels": template_archive["levels"].replace(agent_dir=jnp.arange(size, dtype=jnp.uint8) % 4)}
        state = ExampleState({"qd_archive": archive, "qd_max_age": jnp.int32(1024)}, {"w": jnp.ones(2)})
        template = state.replace(sampler={**state.sampler, "qd_archive": template_archive})
        config = {"use_accel": True, "qd_mode": "map_elites"}
        restored_rng, restored, rounds = self.checkpoint_roundtrip(state, template, config, config)
        self.assertEqual(rounds, 3)
        self.assert_tree_equal(state, restored)
        np.testing.assert_array_equal(restored_rng, self.key)
        replay = Level.stack([self.other] * 32)
        for left, right in zip(qd.sample_archive_parents(archive, replay, self.key, 25, 1024, .5),
                               qd.sample_archive_parents(restored.sampler["qd_archive"], replay,
                                                         restored_rng, 25, 1024, .5)):
            self.assert_tree_equal(left, right)
        for key, value in (("qd_parent_prob", .6), ("qd_max_age", 10), ("qd_wall_edges", [.1, .2, .3, .4])):
            with self.assertRaisesRegex(ValueError, key):
                mutation_checkpoint.validate_config(config, {**config, key: value}, False)

    def test_old_fixed_orbax_checkpoint_requires_explicit_migration(self):
        state = ExampleState({}, {"w": jnp.ones(2)})
        config = {"use_accel": True}
        rng, restored, _ = self.checkpoint_roundtrip(state, state, config, {**config, "qd_mode": "off"})
        self.assert_tree_equal(state, restored)
        np.testing.assert_array_equal(rng, self.key)
        template = state.replace(sampler={"qd_archive": qd.initialize_archive(self.corridor),
                                          "qd_max_age": jnp.int32(1024)})
        request = {**config, "qd_mode": "map_elites"}
        with self.assertRaisesRegex(ValueError, "qd_mode"):
            self.checkpoint_roundtrip(state, template, config, request)
        _, migrated, _ = self.checkpoint_roundtrip(state, template, config,
                                                    {**request, "initialize_qd_on_resume": True})
        self.assert_tree_equal(migrated, template)

    def checkpoint_roundtrip(self, state, template, saved, requested):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "config.json").write_text(json.dumps(saved), encoding="utf-8")
            checkpointer = ocp.StandardCheckpointer()
            try:
                checkpointer.save(directory / "models/0/default", mutation_checkpoint.payload(self.key, state))
            finally:
                checkpointer.close()
            return mutation_checkpoint.restore(directory, -1, template, requested)

    def test_training_branches_preserve_one_rollout_each(self):
        script = Path(__file__).resolve().parents[1] / "examples/maze_plr.py"
        tree = ast.parse(script.read_text(encoding="utf-8"))
        branches = {"on_new_levels", "on_replay_levels", "on_mutate_levels"}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in branches:
                count = sum(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                            and call.func.id == "sample_trajectories_rnn" for call in ast.walk(node))
                self.assertEqual(count, 1, node.name)
                branches.remove(node.name)
        self.assertFalse(branches)
