"""Read-only curriculum audit for local Orbax 0.5.3 archives on CPU."""

import argparse
from contextlib import closing
import csv
import json
from pathlib import Path

from etils import epath
import jax
import numpy as np
import orbax.checkpoint as ocp

from jaxued.level_sampler import LevelSampler
from jaxued.pace_validation import fingerprint, shortest_path


def audit(seed_dir, step, checkpointer, handler):
    config = json.loads((seed_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = epath.Path(seed_dir.resolve() / "models" / str(step) / "default")
    # Orbax 0.5.3 public metadata deserializes the original CUDA sharding on CPU.
    structure, _ = handler._get_internal_metadata(checkpoint)
    restore_args = jax.tree_util.tree_map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), structure)
    record = checkpointer.restore(checkpoint, args=ocp.args.PyTreeRestore(restore_args=restore_args))
    buffer = record["sampler"]
    size = int(buffer["size"])
    levels = buffer["levels"]
    rows = list(zip(levels["wall_map"][:size], levels["agent_pos"][:size], levels["goal_pos"][:size]))
    lengths = np.array([shortest_path(w, a, g) for w, a, g in rows])
    scores = buffer["scores"][:size]
    legal = lengths >= 0
    ranked = np.argsort(-scores, kind="stable")
    top = ranked[:32]
    sampler = LevelSampler(config["level_buffer_capacity"], config["replay_prob"], config["staleness_coeff"],
                           config["minimum_fill_ratio"], config["prioritization"],
                           {"temperature": config["temperature"], "k": config["topk_k"]})
    p = np.asarray(sampler.level_weights(buffer))[:size]
    return {
        "run_name": config["run_name"], "seed": config["seed"],
        "num_updates": sum(int(record[f"num_{kind}_updates"]) for kind in ("dr", "replay", "mutation")),
        "optimizer_updates": int(record["step"]), "buffer_size": size,
        "unique_layouts": len({fingerprint(w, a, g) for w, a, g in rows}),
        "solvable_fraction": float(legal.mean()), "top32_solvable_fraction": float(legal[top].mean()),
        "weighted_solvable_fraction": float(p @ legal),
        "mean_solvable_path": float(np.mean(lengths[legal])) if legal.any() else None,
        "top32_mean_solvable_path": float(np.mean(lengths[top][legal[top]])) if legal[top].any() else None,
        "top1_replay_weight": float(p[ranked[0]]), "ess": float(1 / (p @ p)),
        "finite_scores": int(np.isfinite(scores).sum()), "checkpoint": str(checkpoint),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("seed_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    handler = ocp.PyTreeCheckpointHandler()
    with closing(ocp.Checkpointer(handler)) as checkpointer:
        results = [audit(directory, step, checkpointer, handler)
                   for directory in args.seed_dirs for step in (17, 119)]
    with args.output.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
