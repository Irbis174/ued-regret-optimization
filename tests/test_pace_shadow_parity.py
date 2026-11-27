"""Opt-in, three-round ACCEL parity audit with independent trainer processes."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter
import unittest

import orbax.checkpoint as ocp

from examples.maze_qd_parity import compare_checkpoint, compare_traces, load_trace


ROOT = Path(__file__).resolve().parents[1]
CHILD = """
import json
import runpy
import jax
from examples.maze_qd_parity import install_monitoring
install_monitoring(standard=False)
runpy.run_path(sys.argv.pop(1), run_name="__main__")
device = jax.devices()[0]
print("PACE_PROFILE " + json.dumps({"device": str(device), "memory_bytes": device.memory_stats()}))
"""


@unittest.skipUnless(os.getenv("JAXUED_PACE_SHADOW_AUDIT"), "set a fresh JAXUED_PACE_SHADOW_AUDIT directory")
class MazeShadowParityTest(unittest.TestCase):
    def test_new_replay_mutation(self):
        output = Path(os.environ["JAXUED_PACE_SHADOW_AUDIT"]).resolve()
        output.mkdir(parents=True, exist_ok=False)
        gpu = os.environ.get("JAX_PLATFORMS", "") in ("cuda", "cuda,cpu")
        num_envs = 32 if gpu else 4
        names = ("disabled-a", "disabled-b", "shadow") if gpu else ("disabled-a", "shadow")
        common = [
            "--use_accel", "--score_function", "MaxMC", "--seed", "0", "--num_updates", "3",
            "--eval_freq", "1", "--eval_levels", "QDParitySmoke", "--eval_num_attempts", "1",
            "--checkpoint_save_interval", "1", "--num_train_envs", str(num_envs),
            "--level_buffer_capacity", str(2 * num_envs), "--minimum_fill_ratio", "0.25", "--replay_prob", "1",
        ]
        env = {
            **os.environ, "WANDB_MODE": "disabled",
            "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "src"))),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
        timing = {}
        for name in names:
            command = [sys.executable, "-B", "-c", "import sys\n" + CHILD,
                       str(ROOT / "examples/maze_plr.py"), *common, "--run_name", name]
            if not gpu:
                command += ["--debug_teacher_parity", str(output / f"{name}-trace.jsonl")]
            if name == "shadow":
                command += ["--pace_shadow", "--pace_shadow_every", "1",
                            "--pace_shadow_log", str(output / "pace_shadow.jsonl")]
            elif name == "disabled-b":
                command += ["--pace_shadow=false"]
            print(f"Starting {name}", flush=True)
            log_path = output / f"{name}.log"
            start = perf_counter()
            with log_path.open("x", encoding="utf-8") as log:
                run = subprocess.run(command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT)
            self.assertEqual(run.returncode, 0, log_path.read_text(encoding="utf-8")[-5000:])
            profile = next(line.split(" ", 1)[1] for line in log_path.read_text(encoding="utf-8").splitlines()
                           if line.startswith("PACE_PROFILE "))
            timing[name] = {"process_wall_seconds": perf_counter() - start, **json.loads(profile)}

        rows = load_trace(output / "pace_shadow.jsonl")
        self.assertEqual([row["branch"] for row in rows], ["new", "replay", "mutation"])
        self.assertEqual([row["optimizer_steps_before"] for row in rows], [0, 0, 5])
        self.assertEqual([row["env_steps"] for row in rows], [256 * num_envs * i for i in (1, 2, 3)])
        for row in rows:
            self.assertEqual(row["finite_ratio"], 1)
            self.assertEqual([level["slot"] for level in row["levels"]], list(range(0, num_envs, num_envs // 4)))
            self.assertTrue(all(level["score"] >= 0 and level["completed_episodes"] >= 1 for level in row["levels"]))

        counters = ("num_dr_updates", "num_replay_updates", "num_mutation_updates", "step")
        summary = {"seed": 0, "num_envs": num_envs, "steps_per_env": 256, "trainer_flags": common,
                   "timing": timing, "checkpoints": {}}
        reader = ocp.StandardCheckpointer()
        try:
            for index, expected in enumerate(((1, 0, 0, 0), (1, 1, 0, 5), (1, 1, 1, 5))):
                states = {}
                for name in names:
                    path = output / "checkpoints" / name / "0" / "models" / str(index) / "default"
                    state = reader.restore(path)
                    self.assertEqual(tuple(int(state[key]) for key in counters), expected)
                    states[name] = state
                comparisons = {name: compare_checkpoint(states["disabled-a"], state)
                               for name, state in states.items() if name != "disabled-a"}
                if gpu:
                    comparisons["shadow-vs-disabled-b"] = compare_checkpoint(states["disabled-b"], states["shadow"])
                for comparison in comparisons.values():
                    self.assertNotIn("runner_rng", comparison["changed_fields"])
                    self.assertNotIn("step", comparison["changed_fields"])
                summary["checkpoints"][index + 1] = {"state_counters": dict(zip(counters, expected)),
                                                      "comparisons": comparisons}
        finally:
            reader.close()
        if not gpu:
            summary["trace"] = compare_traces(*(load_trace(output / f"{name}-trace.jsonl") for name in names))
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if not gpu:
            self.assertIsNone(summary["trace"]["first"])
            self.assertTrue(all(not info["comparisons"]["shadow"]["changed_fields"]
                                for info in summary["checkpoints"].values()))
