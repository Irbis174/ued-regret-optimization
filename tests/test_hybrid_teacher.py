import ast
import json
from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import struct

from jaxued import hybrid_teacher as hybrid, learnability, mutation_checkpoint, teacher_parity
from jaxued.environments.maze.level import Level
from jaxued.level_sampler import LevelSampler
from jaxued.utils import compute_max_returns, max_mc


@struct.dataclass
class State:
    params: dict
    sampler: dict
    opt_state: tuple
    step: int
    num_dr_updates: int
    num_replay_updates: int
    num_mutation_updates: int


class TestHybridTeacher(unittest.TestCase):
    def setUp(self):
        self.level = Level.from_str("#####\n#>..#\n#...#\n#..G#\n#####")
        self.other = self.level.replace(agent_dir=jnp.uint8(1))
        self.sampler = LevelSampler(4, duplicate_check=True, staleness_coeff=.3)

    def test_schedule_jit_and_vmap(self):
        rounds = jnp.array([0, 250, 500, 625, 750, 1000])
        for mode, expected in (
            ("recent_control", [0, 0, 0, 0, 0, 0]),
            ("smooth", [0, 0, 0, .5, 1, 1]),
            ("hard_switch", [0, 0, 1, 1, 1, 1]),
        ):
            actual = jax.jit(jax.vmap(lambda r: hybrid.coefficient(r, mode, 500, 750)))(rounds)
            np.testing.assert_array_equal(actual, expected)

    def test_recent_priority_and_replay_parity_with_duplicates(self):
        levels = Level.stack([self.level, self.other, self.level])
        dones = jnp.array([[False, True, False], [True, False, True], [True, True, False]])
        rewards = jnp.array([[0., 0., 0.], [.8, 0., .9], [0., .7, 0.]])
        values = jnp.array([[.11, .22, .33], [.18, .28, .38], [.31, .41, .51]])
        evidence = learnability.rollout_evidence(dones, rewards, values, compute_max_returns(dones, rewards))
        plain = self.sampler.initialize(self.level, learnability.empty_history())
        dual = self.sampler.initialize(self.level, hybrid.empty_history())
        plain = jax.jit(lambda s: learnability.insert_batch(self.sampler, s, levels, evidence, 0))(plain)
        dual = jax.jit(lambda s: hybrid.insert_batch(self.sampler, s, levels, evidence, 0, 0.))(dual)
        self.assertEqual(int(dual["size"]), 2)
        np.testing.assert_allclose(dual["levels_extra"]["maxmc_max_return"][:2], [.9, .7])

        key = jax.random.PRNGKey(195)
        replay = jax.jit(lambda s: self.sampler.sample_replay_levels(s, key, 3))
        plain, a = replay(plain)
        dual, b = replay(dual)
        self.assertEqual(teacher_parity.tree_digests(a), teacher_parity.tree_digests(b))
        forced = jnp.array([0, 0, 1])
        reference = jnp.maximum(dual["levels_extra"]["maxmc_max_return"][forced], evidence.max_return)
        raw = max_mc(dones, values, reference)
        plain = jax.jit(lambda s: learnability.update_batch(self.sampler, s, forced, evidence, 4))(plain)
        dual = jax.jit(lambda s: hybrid.update_batch(self.sampler, s, forced, evidence, dones, values, 4, 0.))(dual)
        shared = {**dual, "levels_extra": {k: dual["levels_extra"][k] for k in plain["levels_extra"]}}
        self.assertEqual(teacher_parity.tree_digests(plain), teacher_parity.tree_digests(shared))
        np.testing.assert_allclose(dual["levels_extra"]["maxmc_score"][:2], np.asarray(raw)[[1, 2]], atol=1e-7)

    def test_replacement_resets_both_histories_and_preserves_rejected_slot(self):
        api = LevelSampler(1, duplicate_check=True)
        state = api.initialize(self.level, hybrid.empty_history())
        insert = jax.jit(lambda s, level, values, step: hybrid.insert_batch(api, s, level, values, step, 0.))
        incomplete = learnability.RolloutEvidence(jnp.array([0]), jnp.array([0]), jnp.array([0.]), jnp.array([-jnp.inf]))
        moderate = learnability.RolloutEvidence(jnp.array([0]), jnp.array([3]), jnp.array([0.]), jnp.array([0.]))
        strong = learnability.RolloutEvidence(jnp.array([1]), jnp.array([2]), jnp.array([.8]), jnp.array([.7]))
        state = insert(state, Level.stack([self.level]), moderate, 0)
        rejected = insert(state, Level.stack([self.other]), incomplete, 1)
        np.testing.assert_array_equal(rejected["levels"].agent_dir, [0])
        self.assertEqual(teacher_parity.tree_digests(rejected["levels_extra"]),
                         teacher_parity.tree_digests(state["levels_extra"]))
        state = insert(rejected, Level.stack([self.other]), strong, 2)
        np.testing.assert_array_equal(state["levels"].agent_dir, [1])
        np.testing.assert_array_equal(state["levels_extra"]["episodes"], [2])
        np.testing.assert_allclose(state["levels_extra"]["maxmc_score"], [.7])
        np.testing.assert_array_equal(state["levels_extra"]["last_round"], [2])

    def test_all_slots_reweighted_and_native_rank_handles_sigmoid_ties(self):
        state = self.sampler.initialize(self.level, hybrid.empty_history())
        state["size"] = jnp.int32(3)
        state["levels_extra"].update(
            recent_score=jnp.array([.9, .6, .2, .99]),
            recent_initialized=jnp.ones(4, bool),
            maxmc_initialized=jnp.array([True, True, False, True]),
            maxmc_score=jnp.array([24., 30., 0., 500.]),
        )
        blended = jax.jit(hybrid.reweight)(state, .5)
        np.testing.assert_allclose(blended["scores"][:3], [.95, .8, .1], atol=1e-6)
        displayed = hybrid.reweight(state, 1.)
        np.testing.assert_array_equal(displayed["scores"][:3], [1., 1., 0.])
        native = hybrid.reweight(state, 1., raw_rank=True)
        np.testing.assert_array_equal(native["scores"], [24., 30., -np.inf, -np.inf])
        weights = self.sampler.level_weights(native)
        self.assertEqual(int(weights.argmax()), 1)
        self.assertEqual(float(weights[-1]), 0.)
        self.assertAlmostEqual(float(weights.sum()), 1.)
        self.assertTrue(np.isfinite(weights).all())
        np.testing.assert_array_equal(hybrid.reweight(state, 0.)["scores"][:3], state["levels_extra"]["recent_score"][:3])
        np.testing.assert_array_equal(blended["timestamps"], state["timestamps"])

    def test_incomplete_replay_preserves_recent_and_masks_native(self):
        pooled = learnability.RolloutEvidence(jnp.array([1]), jnp.array([2]), jnp.array([.8]), jnp.array([.3]))
        state = self.sampler.initialize(self.level, hybrid.empty_history())
        state = hybrid.insert_batch(self.sampler, state, Level.stack([self.level]), pooled, 0, 0.)
        dones, values = jnp.zeros((2, 1), bool), jnp.ones((2, 1))
        evidence = learnability.rollout_evidence(dones, jnp.zeros_like(values), values, jnp.zeros(1))
        updated = hybrid.update_batch(self.sampler, state, jnp.array([0]), evidence, dones, values, 1, 1., raw_rank=True)
        self.assertEqual(float(updated["levels_extra"]["recent_score"][0]), float(state["scores"][0]))
        self.assertFalse(bool(updated["levels_extra"]["maxmc_initialized"][0]))
        self.assertTrue(float(updated["scores"][0]) == -np.inf)
        self.assertEqual(float(hybrid.reweight(updated, 1.)["scores"][0]), 0.)

    def test_checkpoint_fork_restores_history_rng_optimizer_and_horizon(self):
        bank = self.sampler.initialize(self.level, hybrid.empty_history())
        state = State({"w": jnp.ones(1)}, bank, (jnp.int32(10), jnp.ones(1)), 10, 1, 2, 0)
        key = jax.random.PRNGKey(42)
        prefix = {**hybrid.DEFAULTS, "dual_score_tracking": True, "hybrid_start_round": 3, "hybrid_end_round": 4,
                  "score_function": "recent_learnability", "num_updates": 6, "lr": .0001}
        fork = {**prefix, "score_function": "hybrid", "fork_hybrid_on_resume": True}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "config.json").write_text(json.dumps(prefix), encoding="utf-8")
            writer = ocp.StandardCheckpointer()
            try:
                writer.save(folder / "models/0/default", mutation_checkpoint.payload(key, state))
            finally:
                writer.close()
            for mode in hybrid.MODES:
                request = {**fork, "hybrid_mode": mode}
                restored, continuation, completed = mutation_checkpoint.restore(folder, 0, state, request)
                self.assertEqual(completed, 3)
                self.assertEqual(teacher_parity.tree_digests((restored, continuation)),
                                 teacher_parity.tree_digests((key, state)))
            with self.assertRaisesRegex(ValueError, "num_updates"):
                mutation_checkpoint.restore(folder, 0, state, {**fork, "num_updates": 3})
            with self.assertRaisesRegex(ValueError, "hybrid_mode"):
                mutation_checkpoint.restore(folder, 0, state, {**fork, "fork_hybrid_on_resume": False,
                                                                 "hybrid_mode": "smooth"})
            (folder / "config.json").write_text(json.dumps({**prefix, "dual_score_tracking": False}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dual-score"):
                mutation_checkpoint.restore(folder, 0, state, fork)

    def test_training_code_preserves_rollout_count(self):
        source = Path(__file__).resolve().parents[1] / "examples" / "maze_plr.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        branches = {"on_new_levels", "on_replay_levels", "on_mutate_levels"}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in branches:
                calls = (call.func for call in ast.walk(node) if isinstance(call, ast.Call))
                self.assertEqual(sum(isinstance(fn, ast.Name) and fn.id == "sample_trajectories_rnn"
                                     for fn in calls), 1)
                branches.remove(node.name)
        self.assertFalse(branches)
