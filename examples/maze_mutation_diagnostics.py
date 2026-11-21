"""Measure the geometry of ACCEL mutations on procedural levels, without a Student."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxued import adaptive_mutation as am
from jaxued.environments.maze import MazeSolved, make_level_generator, make_level_mutator_minimax


def identity(levels):
    return np.column_stack([
        np.asarray(leaf).reshape(levels.wall_map.shape[0], -1)
        for leaf in jax.tree_util.tree_leaves(levels)
    ])


def distribution(values):
    return dict(zip(("min", "q25", "median", "q75", "max"), np.percentile(values, [0, 25, 50, 75, 100]).tolist())) if len(values) else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=17431)
    parser.add_argument("--parents", type=int, default=96)
    parser.add_argument("--samples-per-parent", type=int, default=4)
    args = parser.parse_args()
    if args.parents < 1 or args.samples_per_parent < 1:
        parser.error("parent and sample counts must be positive")
    if args.output.exists():
        parser.error("output directory already exists")

    level_key, edit_key = jax.random.split(jax.random.PRNGKey(args.seed))
    generator = make_level_generator(13, 13, 25)
    parents = jax.jit(jax.vmap(generator))(jax.random.split(level_key, args.parents))
    parents = jax.tree_util.tree_map(lambda x: jnp.repeat(x, args.samples_per_parent, axis=0), parents)
    count = args.parents * args.samples_per_parent
    keys = jax.random.split(edit_key, count)
    parent_ids = np.repeat(np.arange(args.parents), args.samples_per_parent)
    parent_identity = identity(parents)
    mutate = make_level_mutator_minimax(am.MAX_EDITS)
    generate = jax.jit(lambda indices: am.mutate_batch(mutate, keys, parents, indices, am.DEFAULT_ARMS))
    jax.block_until_ready(generate(jnp.zeros(count, dtype=jnp.int32)))

    solver = MazeSolved(max_height=13, max_width=13)
    horizon = solver.default_params.max_steps_in_episode
    distances = jax.jit(jax.vmap(lambda level: solver.min_steps_to_goal(solver.init_state_from_level(level))))
    validity = jax.jit(jax.vmap(am.valid_child))
    parent_steps = np.asarray(distances(parents))
    summary = {"seed": args.seed, "parents": args.parents, "samples_per_parent": args.samples_per_parent,
               "device": str(jax.devices()[0]), "jax_version": jax.__version__, "operators": []}
    arrays = []
    for operator, edits in enumerate(am.DEFAULT_ARMS):
        indices = jnp.full(count, operator, dtype=jnp.int32)
        start = time.perf_counter()
        children = jax.block_until_ready(generate(indices))
        elapsed = time.perf_counter() - start
        valid = np.asarray(validity(parents, children))
        fingerprints = identity(children)
        same_parent = (fingerprints == parent_identity).all(axis=1)
        steps = np.asarray(distances(am.safe_children(parents, children, valid)))
        steps = np.where(valid, steps, np.inf)
        solved = steps <= horizon
        both = solved & (parent_steps <= horizon)
        walls = np.asarray(children.wall_map)
        parent_walls = np.asarray(parents.wall_map)
        changed = (walls != parent_walls).sum(axis=(1, 2))
        wall_delta = walls.sum(axis=(1, 2)) - parent_walls.sum(axis=(1, 2))
        by_parent = np.column_stack((parent_ids, fingerprints))[valid]
        unique = len(np.unique(fingerprints[valid], axis=0))
        unique_by_parent = len(np.unique(by_parent, axis=0))
        row = {
            "id": operator, "edits": edits, "generated": count, "valid_ratio": float(valid.mean()),
            "unique_ratio": unique / count, "within_parent_unique_ratio": unique_by_parent / count,
            "identical_to_parent_ratio": float(same_parent.mean()),
            "mean_changed_cells": float(changed.mean()), "changed_cells": distribution(changed),
            "mean_wall_count_change": float(wall_delta.mean()), "wall_count_change": distribution(wall_delta),
            "goal_moved_ratio": float(np.any(np.asarray(children.goal_pos) != np.asarray(parents.goal_pos), axis=1).mean()),
            "reachable_within_horizon_ratio": float(solved.mean()),
            "shortest_path_steps": distribution(steps[solved]),
            "shortest_path_delta": distribution(steps[both] - parent_steps[both]),
            "generation_wall_seconds": elapsed,
        }
        print(json.dumps(row), flush=True)
        summary["operators"].append(row)
        arrays.append((valid, same_parent, changed, wall_delta, steps))

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.output / "metrics.npz", parent_id=parent_ids, parent_shortest_path=parent_steps,
        **{name: np.stack(column) for name, column in zip(
            ("valid", "same_parent", "changed_cells", "wall_delta", "shortest_path"), zip(*arrays)
        )},
    )


if __name__ == "__main__":
    main()
