"""Calibrate Maze descriptors and exercise an archive without Student rollouts."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

from jaxued import adaptive_mutation as am, map_elites as qd
from jaxued.environments.maze import MazeSolved, make_level_generator, make_level_mutator_minimax
from jaxued.environments.maze.level import Level


def calibrate_edges(values, bins):
    quantiles = np.quantile(values, np.arange(1, bins) / bins).astype(np.float32)
    counts = np.bincount(np.searchsorted(quantiles, values, side="right"), minlength=bins)
    if np.all(counts > 0):
        return quantiles, "quantiles"
    unique, frequency = np.unique(values, return_counts=True)
    bins = min(bins, len(unique))
    cdf = frequency.cumsum()[:-1] / len(values)
    cuts = []
    for target in range(1, bins):
        first = cuts[-1] + 1 if cuts else 0
        last = len(unique) - (bins - target)
        cuts.append(first + int(np.abs(cdf[first:last] - target / bins).argmin()))
    cuts = np.array(cuts, dtype=int)
    actual = (unique[cuts] + unique[cuts + 1]) * np.float32(0.5)
    return actual, "distinct midpoint nearest quantile"


def save_archive_view(directory, stem, occupied, quality, levels):
    rows, cols = occupied.shape
    x0, y0, cell = 46, 44, 56
    image = Image.new("RGB", (x0 + cols * cell + 12, y0 + rows * cell + 12), "white")
    draw = ImageDraw.Draw(image)
    draw.text((x0, 8), "Elite quality [0, 1] | Dead-end bins", fill="#18252e")
    draw.text((4, y0 - 30), "Wall\nbins", fill="#18252e")
    for col in range(cols):
        draw.text((x0 + col * cell + 24, y0 - 15), str(col), fill="#18252e")
    for row in range(rows):
        y = y0 + row * cell
        draw.text((22, y + 22), str(row), fill="#18252e")
        for col in range(cols):
            x = x0 + col * cell
            value = float(quality[row, col])
            if occupied[row, col]:
                color = tuple(int(light + value * (dark - light))
                              for light, dark in zip((227, 238, 250), (20, 61, 106)))
                label, foreground = f"{value:.2f}", "white" if value > .6 else "#18252e"
            else:
                color, label, foreground = "#f2f3f4", "-", "#5c6470"
            draw.rectangle((x, y, x + cell - 2, y + cell - 2), fill=color)
            draw.text((x + 14, y + 22), label, fill=foreground)
    image.save(directory / f"{stem}-quality.png")
    examples = []
    for index in np.flatnonzero(occupied.ravel())[:5]:
        level = Level(**{key: values[index] for key, values in levels.items()})
        examples.append(f"Niche {index} ({index // cols}, {index % cols})\n{level.to_str()}")
    (directory / f"{stem}-elites.txt").write_text("\n\n".join(examples) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=57314)
    parser.add_argument("--levels", type=int, default=4096)
    args = parser.parse_args()
    if args.levels < 128 or args.levels % 32:
        parser.error("--levels must be a multiple of 32 and at least 128")
    if args.output.exists():
        parser.error("output directory already exists")

    generate_key, select_key, mutate_key = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    generator = make_level_generator(13, 13, 25)
    levels = jax.jit(jax.vmap(generator))(jax.random.split(generate_key, args.levels))
    describe = jax.jit(jax.vmap(qd.compute_descriptors))
    jax.block_until_ready(describe(levels))
    start = time.perf_counter()
    descriptors = np.asarray(describe(levels))
    seconds = time.perf_counter() - start
    calibrations = [calibrate_edges(column, 5) for column in descriptors.T]
    wall_edges, deadend_edges = (jnp.asarray(edges) for edges, _ in calibrations)
    bins = len(wall_edges) + 1, len(deadend_edges) + 1
    placeholder = jax.tree_util.tree_map(lambda x: x[0], levels)
    archive = qd.initialize_archive(placeholder, bins)

    @jax.jit
    def update(archive, levels, round_index):
        score = 1.0 - jnp.abs(jax.vmap(qd.compute_descriptors)(levels)[:, 0] - 0.22)
        return qd.update_archive_batch(
            archive, levels, score, jnp.ones(len(score), dtype=jnp.int32),
            round_index, wall_edges, deadend_edges, 1024,
        )

    for start in range(0, args.levels, 32):
        batch = jax.tree_util.tree_map(lambda x: x[start:start + 32], levels)
        archive = update(archive, batch, jnp.int32(start // 32))

    replay = jax.tree_util.tree_map(lambda x: x[:128], levels)
    elite_parents, chosen = qd.sample_archive_parents(archive, replay, select_key, args.levels // 32 - 1, 1024, 1.0)
    keys = jax.random.split(mutate_key, 128)
    mutator = jax.jit(jax.vmap(make_level_mutator_minimax(am.MAX_EDITS), (0, 0, None)))
    solver = MazeSolved(max_height=13, max_width=13)
    path_length = jax.jit(jax.vmap(lambda x: solver.min_steps_to_goal(solver.init_state_from_level(x))))
    comparisons = {}
    for name, parents in (("procedural_parents", replay), ("archive_parents", elite_parents)):
        children = mutator(keys, parents, 5)
        valid, unchanged, _, changes = am.candidate_stats(parents, children)
        lengths = np.asarray(path_length(am.safe_children(parents, children, valid)))
        reachable = np.asarray(valid) & (lengths <= solver.default_params.max_steps_in_episode)
        comparisons[name] = {
            "slots": 128, "unique_parents": int(qd.unique_count(parents)),
            "unique_children": int(qd.unique_count(children)), "invalid_children_ratio": float((~valid).mean()),
            "unchanged_children_ratio": float(unchanged.mean()), "mean_changed_cells": float(changes.mean()),
            "reachable_children_ratio": float(reachable.mean()),
            "median_shortest_path": float(np.median(lengths[reachable])) if reachable.any() else None,
            "child_descriptor_bins": int(len(np.unique(qd.get_niche_index(describe(children), wall_edges, deadend_edges)))),
        }

    occupied = np.asarray(archive["occupied"])
    qualities = np.asarray(archive["quality"]).reshape(bins)
    report = {
        "seed": args.seed, "n_levels": args.levels, "bins": bins, "descriptor_wall_seconds": seconds,
        "unique_levels": int(len(np.unique(np.asarray(qd.fingerprints(levels)), axis=0))),
        "descriptors": {
            name: {"edges": edges.tolist(), "method": method, "unique_values": len(np.unique(column)),
                   "min_q25_median_q75_max": np.percentile(column, [0, 25, 50, 75, 100]).tolist()}
            for name, (edges, method), column in zip(("wall_density", "deadend_ratio"), calibrations, descriptors.T)
        },
        "niche_counts": np.bincount(
            qd.get_niche_index(descriptors, wall_edges, deadend_edges), minlength=np.prod(bins)
        ).reshape(bins).tolist(),
        "archive": {k: float(v) for k, v in qd.archive_statistics(archive, args.levels // 32 - 1, 1024).items()},
        "synthetic_fitness": "1 - abs(interior_wall_density - 0.22)",
        "comparison": comparisons,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.output / "descriptors.npz", descriptors=descriptors, quality_grid=qualities,
                        occupied=occupied.reshape(bins), selected_niches=chosen)
    levels = {key: np.asarray(getattr(archive["levels"], key)) for key in Level.__dataclass_fields__}
    save_archive_view(args.output, "archive", occupied.reshape(bins), qualities, levels)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
