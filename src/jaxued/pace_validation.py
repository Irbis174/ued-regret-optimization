"""Fixed procedural validation and host-side curriculum diagnostics."""

from collections import deque
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxued.environments.maze import Level


EXPERT_LEVELS = (
    "SixteenRooms", "SixteenRooms2", "Labyrinth", "LabyrinthFlipped",
    "Labyrinth2", "StandardMaze", "StandardMaze2", "StandardMaze3",
)
GENERATOR_SEED = 921572
EVAL_SEED = 65899
ATTEMPTS = 4
TIERS = ("easy", "medium", "hard", "long")


def shortest_path(walls, start, goal):
    start, goal = tuple(map(int, start)), tuple(map(int, goal))
    h, w = walls.shape
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        (x, y), distance = queue.popleft()
        if (x, y) == goal:
            return distance
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and not walls[ny, nx] and (nx, ny) not in seen:
                queue.append(((nx, ny), distance + 1))
                seen.add((nx, ny))
    return -1


def fingerprint(walls, start, goal):
    return np.packbits(walls).tobytes() + bytes(map(int, (*start, *goal)))


def generate(path, per_tier=32, seed=GENERATOR_SEED):
    rng = np.random.default_rng(seed)
    levels, seen = [], set()
    for tier, (size, density, low, high) in enumerate(((9, .12, 4, 8), (10, .20, 9, 14),
                                                      (11, .27, 15, 21), (12, .31, 22, 60))):
        accepted = 0
        for _ in range(20000):
            wall = rng.random((size, size)) < density
            start, goal = np.argwhere(~wall)[rng.choice((~wall).sum(), 2, replace=False)][:, ::-1]
            distance = shortest_path(wall, start, goal)
            if not low <= distance <= high:
                continue
            padded = np.pad(wall, ((0, 13 - size), (0, 13 - size)), constant_values=True)
            key = fingerprint(padded, start, goal)
            if key in seen:
                continue
            seen.add(key)
            levels.append((padded, goal, start, rng.integers(4), size, size, distance, tier))
            accepted += 1
            if accepted == per_tier:
                break
        if accepted != per_tier:
            raise RuntimeError(f"could not fill validation tier {TIERS[tier]}")

    # Performed after generation as an exact collision assertion only.
    expert = Level.load_prefabs(EXPERT_LEVELS)
    for i in range(len(EXPERT_LEVELS)):
        key = fingerprint(np.asarray(expert.wall_map[i]), np.asarray(expert.agent_pos[i]),
                          np.asarray(expert.goal_pos[i]))
        if key in seen:
            raise ValueError("procedural validation collides with an expert layout")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("wall_map", "goal_pos", "agent_pos", "agent_dir", "width", "height", "distance", "tier")
    dtypes = (bool, np.uint32, np.uint32, np.uint8, np.int32, np.int32, np.int32, np.int32)
    with path.open("xb") as file:
        np.savez_compressed(file, **{key: np.array(values, dtype=dtype)
                                   for key, dtype, values in zip(fields, dtypes, zip(*levels))})
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".json").write_text(json.dumps({
        "seed": seed, "eval_seed": EVAL_SEED, "attempts": ATTEMPTS, "sha256": digest,
        "tiers": {tier: per_tier for tier in TIERS},
    }, indent=2) + "\n", encoding="utf-8")


def load(path):
    with np.load(path, allow_pickle=False) as data:
        levels = Level(**{key: jnp.asarray(data[key]) for key in Level.__dataclass_fields__})
        tiers = np.asarray(data["tier"])
    return levels, tiers


def metrics(returns, tiers):
    solved = returns > 0
    by_tier = {f"eval/validation_tier_{name}_solve_rate": float(solved[:, tiers == i].mean())
               for i, name in enumerate(TIERS)}
    return {
        **by_tier, "eval/validation_solve_rate": float(solved.mean()),
        "eval/validation_hard_solve_rate": float(solved[:, tiers >= 2].mean()),
        "eval/validation_episode_return": float(returns.mean()),
    }


def buffer_metrics(sampler, validation_levels, path):
    size = int(sampler["size"])
    levels = jax.device_get(jax.tree_util.tree_map(lambda x: x[:size], sampler["levels"]))
    keys = [fingerprint(w, a, g) for w, a, g in zip(levels.wall_map, levels.agent_pos, levels.goal_pos)]
    valid_keys = {fingerprint(w, a, g) for w, a, g in zip(
        *jax.device_get((validation_levels.wall_map, validation_levels.agent_pos, validation_levels.goal_pos)))}
    if valid_keys.intersection(keys):
        raise ValueError("training buffer contains a validation layout")
    indices = np.linspace(0, size - 1, min(256, size), dtype=int)
    walls = levels.wall_map[indices]
    distances = np.array([shortest_path(w, a, g) for w, a, g in zip(
        walls, levels.agent_pos[indices], levels.goal_pos[indices])])
    extra = {key: np.asarray(value[:size]) for key, value in sampler["levels_extra"].items()}
    np.savez_compressed(path, scores=np.asarray(sampler["scores"][:size]), distances=distances,
                        sampled_indices=indices, **extra)
    return {
        "buffer/unique_levels": len(set(keys)),
        "buffer/solvable_fraction": float((distances >= 0).mean()),
        "buffer/mean_shortest_path": float(distances[distances >= 0].mean()) if (distances >= 0).any() else 0.,
        "buffer/mean_wall_count": float(walls.sum(axis=(1, 2)).mean()),
    }
