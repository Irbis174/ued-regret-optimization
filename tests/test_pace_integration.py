"""Opt-in technical smoke for real GPU-sized New, Replay and Mutation rounds."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from time import perf_counter
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import wandb

from jaxued.environments.maze.level import prefabs
from jaxued.level_sampler import LevelSampler


TRAINER = Path(__file__).resolve().parents[1] / "examples" / "maze_plr.py"
SMOKE_NAME = "PaceSmokeMaze"
SMOKE_MAP = "\n".join([">.G".ljust(13, "#")] + ["#" * 13] * 12)


@unittest.skipUnless(os.getenv("JAXUED_RUN_PACE_SMOKE") == "1", "enable with JAXUED_RUN_PACE_SMOKE=1")
class TestPaceIntegration(unittest.TestCase):
    def run_trainer(self, args):
        with patch.object(sys, "argv", [str(TRAINER), *args]), patch.dict(os.environ, {"WANDB_MODE": "offline"}):
            try:
                runpy.run_path(str(TRAINER), run_name="__main__")
            finally:
                wandb.finish()

    def assert_same_student(self, first, second):
        for key in ("params", "opt_state", "step"):
            expected, structure = jax.tree_util.tree_flatten(first[key])
            actual, other = jax.tree_util.tree_flatten(second[key])
            self.assertEqual(structure, other)
            for a, b in zip(expected, actual):
                np.testing.assert_array_equal(a, b)

    def check_run(self, reader, name, score, accel):
        root = Path("checkpoints", name, "0").resolve()
        args = [
            "--run_name", name, "--seed", "0", "--num_updates", "3", "--eval_freq", "1",
            "--checkpoint_save_interval", "1", "--level_buffer_capacity", "64",
            "--minimum_fill_ratio", "0.25", "--replay_prob", "1",
            "--no-exploratory_grad_updates", "--score_function", score,
            "--eval_levels", SMOKE_NAME, "--eval_num_attempts", "2",
            "--use_accel" if accel else "--no-use_accel",
        ]
        if score == "pace_pg":
            args.append("--pace_log_diagnostics")
        start = perf_counter()
        self.run_trainer(args)
        wall_seconds = perf_counter() - start
        print(f"{name} completed training and checkpoint writes in {wall_seconds:.2f}s", flush=True)

        counters = ("num_dr_updates", "num_replay_updates", "num_mutation_updates", "step")
        expected = [(1, 0, 0, 0), (1, 1, 0, 5), (1, 1, 1, 5) if accel else (1, 2, 0, 10)]
        states = [reader.restore(root / "models" / str(i) / "default") for i in range(3)]
        self.assertEqual([tuple(int(state[k]) for k in counters) for state in states], expected)
        self.assertTrue(all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(states[-1]["params"])))
        if accel:
            self.assert_same_student(states[1], states[2])

        sampler = states[-1]["sampler"]
        capacity = len(sampler["scores"])
        sampler_api = LevelSampler(capacity, staleness_coeff=0.3, prioritization_params={"temperature": 0.3})
        weights = np.asarray(sampler_api.level_weights(jax.tree_util.tree_map(jnp.asarray, sampler)))
        size = int(sampler["size"])
        self.assertTrue(np.isfinite(weights).all() and (weights >= 0).all())
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        np.testing.assert_array_equal(weights[size:], 0)

        metrics = [json.loads(line) for line in (root / "teacher_metrics.jsonl").read_text().splitlines()]
        if score == "pace_pg":
            for row in metrics:
                self.assertEqual(row["pace/nonfinite_fraction"], 0.0)
                self.assertEqual(row["level_sampler/nonfinite_active_scores"], 0)
                self.assertGreater(row["pace/max_score"], 0)
            active = np.asarray(sampler["scores"][:size])
            self.assertTrue((active >= 0).all() and np.unique(active).size > 1)
            self.run_trainer([
                "--mode", "eval", "--checkpoint_directory", str(root),
                "--eval_levels", SMOKE_NAME, "--eval_num_attempts", "2",
            ])
            with np.load(Path("results", name, "0", "results.npz")) as result:
                self.assertEqual(result["levels"].tolist(), [SMOKE_NAME])
                self.assertTrue(np.isfinite(result["cum_rewards"]).all())

        return states[0], {"wall_seconds": wall_seconds, "counters": dict(zip(counters, expected[-1])),
                           "buffer_size": size, "metrics": metrics}

    def test_checkpoint_paths_and_optimizer_steps(self):
        destination = os.getenv("JAXUED_PACE_SMOKE_DIR")
        if destination:
            destination = Path(destination).resolve()
            destination.mkdir(parents=True)
        previous_dir = Path.cwd()
        with (nullcontext(destination) if destination else tempfile.TemporaryDirectory()) as workdir:
            os.chdir(workdir)
            reader = ocp.StandardCheckpointer()
            summary = {"devices": [str(device) for device in jax.devices()], "runs": {}}
            try:
                with patch.dict(prefabs, {SMOKE_NAME: SMOKE_MAP}):
                    for accel, method in [(False, "plr"), (True, "accel")]:
                        baseline_name, pace_name = f"maxmc-{method}", f"pace-{method}"
                        baseline, summary["runs"][baseline_name] = self.check_run(reader, baseline_name, "MaxMC", accel)
                        pace, summary["runs"][pace_name] = self.check_run(reader, pace_name, "pace_pg", accel)
                        self.assert_same_student(baseline, pace)
                Path("summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            finally:
                try:
                    reader.close()
                    wandb.teardown()
                finally:
                    os.chdir(previous_dir)
