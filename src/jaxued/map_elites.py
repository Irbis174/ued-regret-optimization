"""Bounded JAX MAP-Elites archive for the Maze Teacher."""

import jax
import jax.numpy as jnp

from jaxued import adaptive_mutation, learnability


# Measured on 4,096 levels from make_level_generator(13, 13, 25), seed 57314.
DEFAULT_EDGES = (
    (0.12396693974733353, 0.13223139941692352, 0.14049586653709412, 0.1570248007774353),
    (0.004464285913854837, 0.008968790993094444, 0.00956959743052721, 0.018957771360874176),
)
DEFAULT_CONFIG = {
    "qd_mode": "off",
    "qd_bins": [5, 5],
    "qd_wall_edges": list(DEFAULT_EDGES[0]),
    "qd_deadend_edges": list(DEFAULT_EDGES[1]),
    "qd_parent_prob": 0.5,
    "qd_max_age": 1024,
}


def compute_descriptors(level):
    walls = jnp.asarray(level.wall_map, dtype=bool)
    y, x = jnp.indices(walls.shape)
    interior = (x > 0) & (x < level.width - 1) & (y > 0) & (y < level.height - 1)
    walkable = ~walls & (x < level.width) & (y < level.height)
    padded = jnp.pad(walkable.astype(jnp.int32), 1)
    neighbors = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
    density = (interior & walls).sum() / jnp.maximum(interior.sum(), 1)
    deadends = (interior & walkable & (neighbors == 1)).sum() / jnp.maximum((interior & walkable).sum(), 1)
    return jnp.stack((density, deadends))


def get_niche_index(descriptors, wall_edges, deadend_edges):
    wall = jnp.searchsorted(jnp.asarray(wall_edges), descriptors[..., 0], side="right")
    deadends = jnp.searchsorted(jnp.asarray(deadend_edges), descriptors[..., 1], side="right")
    return wall * (len(deadend_edges) + 1) + deadends


def initialize_archive(placeholder, bins=(5, 5)):
    size = bins[0] * bins[1]
    return {
        "occupied": jnp.zeros(size, dtype=bool),
        "levels": jax.tree_util.tree_map(lambda x: jnp.broadcast_to(jnp.asarray(x), (size,) + jnp.shape(x)), placeholder),
        "quality": jnp.zeros(size, dtype=jnp.float32),
        "last_evaluated": jnp.full(size, -1, dtype=jnp.int32),
        "num_updates": jnp.zeros(size, dtype=jnp.int32),
        "observations": jnp.zeros(size, dtype=jnp.int32),
    }


def fingerprints(levels):
    """Lossless flattened level identity, including start, direction and goal."""
    size = levels.wall_map.shape[0]
    return jnp.concatenate([x.reshape(size, -1).astype(jnp.int32)
                            for x in jax.tree_util.tree_leaves(levels)], axis=1)


def _lex_less(left, right):
    differs = left != right
    first = differs.argmax(axis=-1)[..., None]
    return jnp.take_along_axis(differs & (left < right), first, axis=-1)[..., 0]


def unique_count(levels):
    rows = fingerprints(levels)
    same = (rows[:, None] == rows[None, :]).all(axis=-1)
    return (~jnp.tril(same, k=-1).any(axis=1)).sum()


def rollout_quality(levels, evidence, round_index):
    """Score one observation window, pooling identical levels in this batch.

    All candidates start with empty history for the archive comparison. The
    LevelSampler's independent accumulated histories are preserved.
    """
    rows = fingerprints(levels)
    same = (rows[:, None] == rows[None, :]).all(axis=-1)
    completed = same & (evidence.episodes[None, :] > 0)
    n = jnp.where(completed, evidence.episodes, 0).sum(axis=1)
    k = jnp.where(completed, evidence.successes, 0).sum(axis=1)
    best = jnp.where(completed, evidence.max_return, 0).max(axis=1)
    # MaxMC averages over completed episodes and is affine in max_return.
    centered = jnp.where(evidence.episodes > 0, evidence.maxmc - evidence.max_return, 0)
    score = best + jnp.where(completed, centered * evidence.episodes, 0).sum(axis=1) / jnp.maximum(n, 1)
    pooled = learnability.RolloutEvidence(k, n, best, score)
    quality = jax.vmap(lambda row: learnability.record_evidence(
        learnability.empty_history(), 0., row, round_index
    )[0])(pooled)
    return quality, n.astype(jnp.int32)


def update_archive_batch(archive, levels, quality, observations, round_index, wall_edges, deadend_edges,
                         max_age, *, random_key=None):
    """Refresh observed incumbents, then compete within each niche."""
    rows = fingerprints(levels)
    incumbent = fingerprints(archive["levels"])
    occupied = archive["occupied"]
    slots = jnp.arange(occupied.shape[0])
    niche = get_niche_index(jax.vmap(compute_descriptors)(levels), wall_edges, deadend_edges)
    structurally_valid = jax.vmap(adaptive_mutation.valid_child)(levels, levels)
    valid = (observations > 0) & jnp.isfinite(quality) & structurally_valid
    in_niche = (slots[:, None] == niche[None, :]) & valid
    same = (incumbent[:, None] == rows[None, :]).all(axis=-1)
    observed = occupied[:, None] & in_niche & same
    refresh = observed.any(axis=1)
    best_observed = jnp.where(observed, quality, -jnp.inf).max(axis=1, keepdims=True)
    fresh_index = jnp.where(observed & (quality == best_observed), observations, -1).argmax(axis=1)
    prior_quality = jnp.where(refresh, quality[fresh_index], archive["quality"])
    prior_time = jnp.where(refresh, round_index, archive["last_evaluated"])
    prior_observations = jnp.where(refresh, observations[fresh_index], archive["observations"])
    fresh = occupied & (round_index - prior_time <= max_age)

    if random_key is None:
        # Rank all candidates with a deterministic lexicographic tie-break.
        equal_score = quality[:, None] == quality[None, :]
        lex = _lex_less(rows[:, None], rows[None, :])
        equal_level = (rows[:, None] == rows[None, :]).all(axis=-1)
        more_observed = observations[:, None] > observations[None, :]
        better = (quality[:, None] > quality[None, :]) | (equal_score & (lex | (equal_level & more_observed)))
        rank = better.sum(axis=0)
        winner = jnp.where(in_niche, rank, len(quality) + 1).argmin(axis=1)
        competitor = quality[winner]
        better_than_incumbent = (competitor > prior_quality) | (
            (competitor == prior_quality) & _lex_less(rows[winner], incumbent)
        )
        replace = in_niche.any(axis=1) & (~fresh | better_than_incumbent)
    else:
        # One uniform draw over unique challengers and the fresh incumbent.
        same_candidate = (rows[:, None] == rows[None, :]).all(axis=-1)
        first = ~jnp.tril(same_candidate & valid[None, :], k=-1).any(axis=1)
        challengers = in_niche & first & ~(fresh[:, None] & same)
        choices = jnp.concatenate((challengers, jnp.ones((len(slots), 1), dtype=bool)), axis=1)
        choices = choices.at[:, -1].set(fresh | ~challengers.any(axis=1))
        draw = jax.random.categorical(random_key, jnp.where(choices, 0., -jnp.inf), axis=-1)
        replace = draw < len(quality)
        winner = jnp.minimum(draw, len(quality) - 1)

    def pick(old, new):
        mask = replace.reshape((len(slots),) + (1,) * (old.ndim - 1))
        return jnp.where(mask, new[winner], old)

    return {
        "occupied": occupied | replace,
        "levels": jax.tree_util.tree_map(pick, archive["levels"], levels),
        "quality": jnp.where(replace, quality[winner], prior_quality),
        "last_evaluated": jnp.where(replace, round_index, prior_time),
        "num_updates": archive["num_updates"] + (occupied & replace).astype(jnp.int32),
        "observations": jnp.where(replace, observations[winner], prior_observations),
    }


def sample_archive_parents(archive, replay_parents, key, round_index, max_age, parent_prob):
    available = archive["occupied"] & (round_index - archive["last_evaluated"] <= max_age)
    use_key, niche_key = jax.random.split(key)
    batch_size = replay_parents.wall_map.shape[0]
    use = jax.random.bernoulli(use_key, parent_prob, (batch_size,)) & available.any()
    safe = available.at[0].set(available[0] | ~available.any())
    indices = jax.random.categorical(niche_key, jnp.where(safe, 0., -jnp.inf), shape=(batch_size,)).astype(jnp.int32)

    def choose(elites, parents):
        return jnp.where(use.reshape((batch_size,) + (1,) * (parents.ndim - 1)), elites[indices], parents)

    parents = jax.tree_util.tree_map(choose, archive["levels"], replay_parents)
    return parents, jnp.where(use, indices, -1)


def archive_statistics(archive, round_index, max_age):
    occupied = archive["occupied"]
    age = jnp.maximum(round_index - archive["last_evaluated"], 0)
    count = occupied.sum()
    den = jnp.maximum(count, 1)
    descriptors = jax.vmap(compute_descriptors)(archive["levels"])
    means = jnp.where(occupied[:, None], descriptors, 0).sum(axis=0) / den
    return {
        "qd/archive_coverage": occupied.mean(),
        "qd/occupied_niches": count,
        "qd/elite_replacements": archive["num_updates"].sum(),
        "qd/mean_elite_quality": jnp.where(occupied, archive["quality"], 0).sum() / den,
        "qd/mean_elite_age": jnp.where(occupied, age, 0).sum() / den,
        "qd/stale_elite_count": (occupied & (age > max_age)).sum(),
        "qd/mean_elite_observations": jnp.where(occupied, archive["observations"], 0).sum() / den,
        "qd/descriptor_wall_density": means[0],
        "qd/descriptor_deadend_ratio": means[1],
    }
