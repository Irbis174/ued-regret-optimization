"""Exercise fixed, uniform and resumed bandit ACCEL on a tiny technical schedule."""

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
import wandb

from jaxued.environments.maze.level import prefabs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id or not all(c.isalnum() or c in "-_" for c in args.run_id):
        parser.error("invalid run ID")
    os.environ["WANDB_MODE"] = "offline"
    rows = prefabs["TrivialMaze"].splitlines()
    prefabs["AMSmokeMaze"] = "\n".join([row.ljust(13, "#") for row in rows] + ["#" * 13] * (13 - len(rows)))
    common = [
        "--use_accel", "--seed", "0", "--eval_freq", "3", "--eval_levels", "AMSmokeMaze",
        "--eval_num_attempts", "2", "--checkpoint_save_interval", "1",
        "--level_buffer_capacity", "64", "--minimum_fill_ratio", "0.25", "--replay_prob", "1",
    ]
    previous = f"checkpoints/{args.run_id}-bandit-start/0"
    configs = (
        ("fixed", 3, 0, ["--score_function", "MaxMC", "--num_edits", "5"]),
        ("uniform", 3, 0, ["--score_function", "pvl", "--mutation_selection", "uniform"]),
        ("bandit-start", 6, 0, ["--score_function", "recent_learnability", "--adaptive_mutation",
                               "--stop_after_updates", "3"]),
        ("bandit-resume", 6, 1, ["--score_function", "recent_learnability", "--adaptive_mutation",
                                "--resume_checkpoint_directory", previous]),
    )
    output = Path("results/adaptive-mutation-diagnostics") / f"{args.run_id}.json"
    if output.exists() or any((Path("checkpoints") / f"{args.run_id}-{key}" / "0").exists()
                              for key, _, _, _ in configs):
        parser.error("smoke outputs already exist")
    summary, memories = {}, {}
    # With the technical schedule: new, replay, mutate, replay, mutate, replay.
    expected = [(1, 1, 1, 5), (1, 3, 2, 15)]
    fields = ("num_dr_updates", "num_replay_updates", "num_mutation_updates", "step")
    reader = ocp.StandardCheckpointer()
    try:
        reference = None
        for arm, budget, index, options in configs:
            name = f"{args.run_id}-{arm}"
            path = Path("checkpoints") / name / "0"
            argv = sys.argv
            sys.argv = ["examples/maze_plr.py", "--run_name", name, "--num_updates", str(budget), *common, *options]
            print("Smoke", arm, jax.devices(), flush=True)
            try:
                runpy.run_path(sys.argv[0], run_name="__main__")
                wandb_dir = str(Path(wandb.run.dir).parent)
            finally:
                sys.argv = argv
                wandb.finish()
            state = reader.restore((path / "models" / str(index) / "default").resolve())
            counters = tuple(int(state[key]) for key in fields)
            assert counters == expected[index], (arm, counters)
            leaves, tree = jax.tree_util.tree_flatten(state["params"])
            structure = (tree, [x.shape for x in leaves])
            if reference is None:
                reference = structure
            assert reference == structure and all(np.isfinite(x).all() for x in leaves)
            assert state["runner_rng"].shape == (2,)
            metrics = json.loads((path / "teacher_metrics.jsonl").read_text().splitlines()[-1])
            assert np.isfinite(list(metrics.values())).all()
            assert metrics["level_sampler/nonfinite_active_scores"] == 0
            assert metrics["level_sampler/nonfinite_active_weights"] == 0
            if arm.startswith("bandit"):
                bandit = {key: np.asarray(value) for key, value in state["sampler"]["mutation_bandit"].items()}
                assert np.all(bandit["count"] > 0) and int(bandit["count"].sum()) == counters[2] * 32
                assert np.isfinite(bandit["ema_reward"]).all() and np.all(bandit["reward_sum"] >= 0)
                assert np.any(bandit["ema_reward"] > 0)
                memories[arm] = bandit
            else:
                assert "mutation_bandit" not in state["sampler"]
            if arm == "uniform":
                assert sum(metrics[f"adaptive_mutation/operator_{i}_count"] for i in range(4)) == 32
            summary[arm] = {
                "checkpoint": str(path / "models" / str(index) / "default"), "wandb_dir": wandb_dir,
                "counters": dict(zip(fields, counters)), "env_steps": sum(counters[:3]) * 32 * 256,
                "metrics": metrics,
            }
            print("Smoke OK", arm, counters, flush=True)
    finally:
        reader.close()
    before, after = (memories[arm] for arm in ("bandit-start", "bandit-resume"))
    assert np.all(after["count"] >= before["count"])
    assert int((after["count"] - before["count"]).sum()) == 32
    assert np.all(after["reward_sum"] >= before["reward_sum"])
    assert int(after["last_selected"].max()) == 4
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    print("Saved", output, flush=True)


if __name__ == "__main__":
    main()
