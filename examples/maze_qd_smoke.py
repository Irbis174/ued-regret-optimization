"""Check QD-ACCEL on a tiny technical schedule with procedural training levels."""

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


def level_identities(levels, size):
    rows = np.column_stack([np.asarray(levels[key])[:size].reshape(size, -1)
                            for key in ("wall_map", "agent_pos", "goal_pos", "agent_dir", "width", "height")])
    return {row.tobytes() for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id or not all(c.isalnum() or c in "-_" for c in args.run_id):
        parser.error("invalid run ID")
    os.environ["WANDB_MODE"] = "offline"
    rows = prefabs["TrivialMaze"].splitlines()
    prefabs["QDSmokeMaze"] = "\n".join([row.ljust(13, "#") for row in rows] + ["#" * 13] * (13 - len(rows)))
    common = [
        "--use_accel", "--seed", "0", "--score_function", "recent_learnability", "--num_edits", "5",
        "--no-exploratory_grad_updates", "--num_updates", "6", "--eval_freq", "3",
        "--eval_levels", "QDSmokeMaze", "--eval_num_attempts", "2", "--checkpoint_save_interval", "1",
        "--level_buffer_capacity", "64", "--minimum_fill_ratio", "0.25", "--replay_prob", "1",
    ]
    source = f"checkpoints/{args.run_id}-map_elites/0"
    stop = ["--stop_after_updates", "3"]
    configs = (
        ("fixed", "off", 0, stop),
        ("shadow", "shadow", 0, stop),
        ("map_elites", "map_elites", 0, stop),
        ("random_archive", "random_archive", 0, stop),
        ("resumed", "map_elites", 1, ["--resume_checkpoint_directory", source]),
    )
    output = Path("results/quality-diversity-diagnostics") / f"{args.run_id}.json"
    if output.exists() or any((Path("checkpoints") / f"{args.run_id}-{arm}" / "0").exists()
                              for arm, *_ in configs):
        parser.error("smoke outputs already exist")
    fields = ("num_dr_updates", "num_replay_updates", "num_mutation_updates", "step")
    expected = [(1, 1, 1, 5), (1, 3, 2, 15)]
    summary = {"devices": [str(device) for device in jax.devices()], "arms": {}}
    reader = ocp.StandardCheckpointer()
    try:
        for arm, mode, index, options in configs:
            name = f"{args.run_id}-{arm}"
            path = Path("checkpoints") / name / "0"
            argv = sys.argv
            sys.argv = ["examples/maze_plr.py", "--run_name", name, "--qd_mode", mode, *common, *options]
            print("Smoke", arm, jax.devices(), flush=True)
            try:
                runpy.run_path(sys.argv[0], run_name="__main__")
                wandb_dir = str(Path(wandb.run.dir).parent)
            finally:
                sys.argv = argv
                wandb.finish()
            checkpoint = path / "models" / str(index) / "default"
            state = reader.restore(checkpoint.resolve())
            counters = tuple(int(state[key]) for key in fields)
            assert counters == expected[index], (arm, counters)
            assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(state["params"]))
            assert state["runner_rng"].shape == (2,)
            sampler = state["sampler"]
            archive = sampler.get("qd_archive")
            assert (archive is None) == (mode == "off")
            assert "mutation_bandit" not in sampler
            result = {
                "checkpoint": str(checkpoint), "wandb_dir": wandb_dir,
                "counters": dict(zip(fields, counters)), "env_steps": sum(counters[:3]) * 32 * 256,
            }
            if index == 0:
                previous = level_identities(state["dr_last_level_batch"], 32)
                previous |= level_identities(state["replay_last_level_batch"], 32)
                children = level_identities(state["mutation_last_level_batch"], 32) - previous
                inserted = len(children & level_identities(sampler["levels"], int(sampler["size"])))
                assert inserted > 0, arm
                result["new_children_in_sampler"] = inserted
            if mode == "off":
                fixed = state
            else:
                metrics = json.loads((path / "teacher_metrics.jsonl").read_text().splitlines()[-1])
                assert np.isfinite(list(metrics.values())).all()
                assert metrics["level_sampler/nonfinite_active_scores"] == 0
                assert metrics["level_sampler/nonfinite_active_weights"] == 0
                assert len(archive["occupied"]) == 25 and int(archive["occupied"].sum()) > 1
                assert np.isfinite(archive["quality"]).all()
                assert metrics["qd/valid_children_ratio"] == 1
                if mode == "shadow":
                    assert metrics["qd/parents_from_archive"] == 0
                    np.testing.assert_array_equal(state["runner_rng"], fixed["runner_rng"])
                    for field in ("dr_last_level_batch", "replay_last_level_batch", "mutation_last_level_batch"):
                        for key, value in state[field].items():
                            np.testing.assert_array_equal(value, fixed[field][key])
                    differences = [np.max(np.abs(np.asarray(x) - np.asarray(y)))
                                   for x, y in zip(jax.tree_util.tree_leaves(state["params"]),
                                                   jax.tree_util.tree_leaves(fixed["params"]))]
                    result["fixed_shadow_rng_and_levels_equal"] = True
                    result["max_absolute_param_difference_from_fixed"] = float(max(differences))
                else:
                    assert 0 < metrics["qd/parents_from_archive"] <= 32
                if arm == "map_elites":
                    before = archive
                elif arm == "resumed":
                    occupied = np.asarray(before["occupied"])
                    assert np.asarray(archive["occupied"])[occupied].all()
                    assert (archive["last_evaluated"] >= before["last_evaluated"]).all()
                    assert (archive["num_updates"] >= before["num_updates"]).all()
                    assert int(archive["last_evaluated"].max()) > int(before["last_evaluated"].max())
                result["metrics"] = metrics
            summary["arms"][arm] = result
            print("Smoke OK", arm, counters, flush=True)
    finally:
        reader.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    print("Saved", output, flush=True)


if __name__ == "__main__":
    main()
