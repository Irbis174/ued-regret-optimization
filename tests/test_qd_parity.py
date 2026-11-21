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

from examples.maze_qd_parity import compare_traces, load_trace
from jaxued import learnability, map_elites, mutation_checkpoint, teacher_parity
from jaxued.environments.maze.level import Level
from jaxued.level_sampler import LevelSampler


@struct.dataclass
class State:
    sampler: dict
    params: dict
    opt_state: tuple
    step: int
    update_state: int
    num_dr_updates: int
    num_replay_updates: int
    num_mutation_updates: int
    dr_last_level_batch: Level
    replay_last_level_batch: Level
    mutation_last_level_batch: Level


class TestQDParity(unittest.TestCase):
    def setUp(self):
        self.level = Level.from_str("#####\n#>..#\n#...#\n#..G#\n#####")
        self.levels = Level.stack([self.level, self.level.replace(agent_dir=jnp.uint8(2))])
        self.key = jax.random.PRNGKey(413)
        self.api = LevelSampler(4, duplicate_check=True)
        self.base = self.api.initialize(self.level, learnability.empty_history())
        self.archive = map_elites.initialize_archive(self.level)

    def state(self, sampler):
        return State(sampler, {"weight": jnp.array([1., 2.])}, (jnp.zeros(2),), 5, 1, 1, 1, 0,
                     self.levels, self.levels, self.levels)

    def test_trace_and_archive_are_passive_in_jitted_replay(self):
        evidence = learnability.RolloutEvidence(jnp.array([1, 0]), jnp.ones(2, dtype=jnp.int32),
                                                jnp.array([.8, 0.]), jnp.array([.7, .3]))
        base = learnability.insert_batch(self.api, self.base, self.levels, evidence, 0)
        with tempfile.TemporaryDirectory() as tmp:
            trace = teacher_parity.Trace(Path(tmp) / "trace.jsonl")

            def train(state, instrument):
                key, sampling_key = jax.random.split(self.key)
                sampler, (indices, levels) = self.api.sample_replay_levels(state.sampler, sampling_key, 2)
                sampler = learnability.update_batch(self.api, sampler, indices, evidence, 1)
                if instrument:
                    score, obs = map_elites.rollout_quality(levels, evidence, 1)
                    archive = map_elites.update_archive_batch(
                        sampler["qd_archive"], levels, score, obs, 1, *map_elites.DEFAULT_EDGES, 1024
                    )
                    sampler = {**sampler, "qd_archive": archive}
                    jax.debug.callback(trace.buffer, 1, teacher_parity.base_sampler(sampler), ordered=True)
                return key, state.replace(sampler=sampler), indices, levels

            run = jax.jit(train, static_argnums=1)
            a, fixed, ai, al = run(self.state(base), False)
            b, shadow, bi, bl = run(self.state({**base, "qd_archive": self.archive}), True)
            jax.effects_barrier()
            self.assertEqual(teacher_parity.tree_digests((a, ai, al)), teacher_parity.tree_digests((b, bi, bl)))
            projected = shadow.replace(sampler=teacher_parity.base_sampler(shadow.sampler))
            self.assertEqual(teacher_parity.tree_digests(fixed), teacher_parity.tree_digests(projected))
            self.assertGreater(int(shadow.sampler["qd_archive"]["occupied"].sum()), 0)
            row, = load_trace(trace.path)
            self.assertEqual(row["sampler"], teacher_parity.tree_digests(fixed.sampler))

    def test_callbacks_keep_order_exact_replay_indices_and_full_levels(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = teacher_parity.Trace(Path(tmp) / "trace.jsonl")

            def run(levels):
                jax.debug.callback(trace.round_start, 0, 1, self.key, self.key, ordered=True)
                jax.debug.callback(lambda v: trace.selection(0, "replay", self.key, v, [7, 4], None),
                                   levels, ordered=True)
                jax.debug.callback(lambda v: trace.selection(1, "mutation", self.key, v, None, v),
                                   levels, ordered=True)
                return levels

            jax.block_until_ready(jax.jit(run)(self.levels))
            jax.effects_barrier()
            start, replay, mutation = load_trace(trace.path)
            self.assertEqual(start["branch"], "replay")
            self.assertEqual(replay["replay_indices"], [7, 4])
            self.assertEqual(mutation["mutation_parent_indices"], [7, 4])
            self.assertEqual(mutation["parents"], replay["levels"])
            size = int(np.prod(replay["levels"]["wall_shape"][1:]))
            decoded = [np.unpackbits(np.frombuffer(bytes.fromhex(row), dtype=np.uint8))[:size]
                       for row in replay["levels"]["walls_hex"]]
            np.testing.assert_array_equal(np.array(decoded), self.levels.wall_map.reshape(2, -1))
            self.assertEqual(len(set(replay["levels"]["identities"])), 2)
            result = compare_traces([start, replay, mutation], [start, replay, {**mutation, "rng": [0, 0]}])
            self.assertEqual(result["first"], {"stage": "selection", "round": 1, "fields": ["rng"]})

    def test_checkpoint_resume_ignores_trace_destination(self):
        original = self.state({**self.base, "qd_archive": self.archive, "qd_max_age": jnp.int32(1024)})
        config = {"use_accel": True, "qd_mode": "shadow", "debug_teacher_parity": "old.jsonl"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            writer = ocp.StandardCheckpointer()
            try:
                writer.save(root / "models/0/default", mutation_checkpoint.payload(self.key, original))
            finally:
                writer.close()
            key, actual, rounds = mutation_checkpoint.restore(root, -1, original,
                                                               {**config, "debug_teacher_parity": "new.jsonl"})
            self.assertEqual(rounds, 2)
            self.assertEqual(teacher_parity.tree_digests((key, actual)),
                             teacher_parity.tree_digests((self.key, original)))
            first = teacher_parity.Trace(root / "first.jsonl")
            first.snapshot(original, self.key, rounds, "initial")
            second = teacher_parity.Trace(root / "resumed.jsonl")
            second.snapshot(actual, key, rounds, "initial")
            self.assertEqual(load_trace(first.path), load_trace(second.path))

    def test_no_additional_training_rollout_or_prng_calls_in_trace_code(self):
        main = Path(__file__).resolve().parents[1] / "examples/maze_plr.py"
        tree = ast.parse(main.read_text(encoding="utf-8"))
        functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        for name in ("on_new_levels", "on_replay_levels", "on_mutate_levels"):
            calls = [node.func for node in ast.walk(functions[name]) if isinstance(node, ast.Call)]
            self.assertEqual(sum(isinstance(fn, ast.Name) and fn.id == "sample_trajectories_rnn"
                                 for fn in calls), 1)
        for name in ("trace_selection", "trace_rollout", "trace_sampler"):
            self.assertFalse(any(isinstance(node, ast.Attribute) and node.attr == "random"
                                 for node in ast.walk(functions[name])))
