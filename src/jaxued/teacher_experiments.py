"""JIT-compatible teacher-only helpers for HAR and capped replay."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def update_progress_memory(
    fast: jax.Array,
    slow: jax.Array,
    initialized: jax.Array,
    level_indices: jax.Array,
    new_successes: jax.Array,
    new_completed: jax.Array,
    *,
    fast_rate: float = 0.3,
    slow_rate: float = 0.05,
):
    """Update fast/slow success EMAs, aggregating repeated buffer indices.

    All memory arrays have shape [capacity]. New episode counts have shape
    [batch]. A negative or out-of-range index is ignored. Unfinished episodes
    contribute nothing to new_completed; no completed episode means no update.

    On the first valid observation, fast and slow are initialized identically
    to prevent spurious learning-progress scores.

    IMPORTANT: initialize/reset fast, slow and initialized when replacing an
    existing buffer slot with a *new* level. The caller owns this lifecycle.
    """
    capacity = fast.shape[0]
    indices = jnp.asarray(level_indices, dtype=jnp.int32)
    successes = jnp.asarray(new_successes, dtype=jnp.float32)
    completed = jnp.asarray(new_completed, dtype=jnp.float32)
    valid = (indices >= 0) & (indices < capacity) & (completed > 0)
    safe_indices = jnp.where(valid, indices, 0)

    summed_successes = jnp.zeros((capacity,), dtype=jnp.float32).at[safe_indices].add(
        jnp.where(valid, jnp.minimum(jnp.maximum(successes, 0), completed), 0.0)
    )
    summed_completed = jnp.zeros((capacity,), dtype=jnp.float32).at[safe_indices].add(
        jnp.where(valid, completed, 0.0)
    )
    observed = summed_completed > 0
    p = jnp.clip(summed_successes / jnp.maximum(summed_completed, 1.0), 0.0, 1.0)
    fast_new = jnp.where(initialized, (1 - fast_rate) * fast + fast_rate * p, p)
    slow_new = jnp.where(initialized, (1 - slow_rate) * slow + slow_rate * p, p)
    return (
        jnp.where(observed, fast_new, fast),
        jnp.where(observed, slow_new, slow),
        initialized | observed,
    )


def har_score(
    maxmc: jax.Array,
    recent_successes: jax.Array,
    recent_completed: jax.Array,
    fast: jax.Array,
    slow: jax.Array,
    *,
    history_weight: float = 0.2,
    warmup_episodes: float = 4.0,
):
    """Blend logged ``recent_learnability`` with positive progress.

    The caller passes ALREADY AGE-DECAYED recent_successes and recent_completed
    from its original Teacher implementation (this function does not decay).

    ``history_weight=0`` exactly recovers the recorded recent-learnability
    formula, up to normal floating-point precision.
    """
    maxmc = jnp.asarray(maxmc, dtype=jnp.float32)
    successes = jnp.asarray(recent_successes, dtype=jnp.float32)
    completed = jnp.asarray(recent_completed, dtype=jnp.float32)
    p = jnp.clip(successes / jnp.maximum(completed, 1e-8), 0.0, 1.0)
    w = jnp.clip(completed / warmup_episodes, 0.0, 1.0)
    learnability = 4 * p * (1 - p)
    progress = jnp.clip(jnp.asarray(fast) - jnp.asarray(slow), 0.0, 1.0)
    return (1.0 - w) * jax.nn.sigmoid(maxmc) + w * (
        (1.0 - history_weight) * learnability + history_weight * progress
    )


def capped_replay_weights(level_weights, frontier, eligible, *, preserve_frontier_mass):
    """Reweight a single draw while keeping the original group masses when possible."""
    base = jnp.where(jnp.isfinite(level_weights) & (level_weights > 0), level_weights, 0.0)
    weights = jnp.where(eligible, base, 0.0)
    total = weights.sum()
    uniform = eligible / jnp.maximum(eligible.sum(), 1)
    probs = jnp.where(total > 0, weights / jnp.where(total > 0, total, 1), uniform)
    if preserve_frontier_mass:
        def preserve():
            original_mass = jnp.where(frontier, base, 0.0).sum() / base.sum()
            return jnp.where(frontier, fg * (original_mass / fg_mass),
                             other * ((1 - original_mass) / other_mass))

        fg = jnp.where(frontier, weights, 0.0)
        other = jnp.where(~frontier, weights, 0.0)
        fg_mass, other_mass = fg.sum(), other.sum()
        probs = jax.lax.cond((fg_mass > 0) & (other_mass > 0), preserve, lambda: probs)
    return probs


def frontier_mask(sampler):
    """Identify active levels with evidence of both successes and failures."""
    active = jnp.arange(sampler["scores"].shape[0]) < sampler["size"]
    extra = sampler.get("levels_extra", {})
    if "successes" not in extra:
        return jnp.zeros_like(active)
    return active & (extra["successes"] > 0) & (extra["successes"] < extra["episodes"])


def replay_batch_stats(indices, frontier):
    """Count, distinct levels, maximum multiplicity, frontier share and batch ESS."""
    equal = indices[:, None] == indices[None, :]
    counts = equal.sum(axis=1)
    unique = (~jnp.tril(equal, k=-1).any(axis=1)).sum()
    size = indices.shape[0]
    return jnp.array([1, unique, counts.max(), frontier[indices].mean(), size**2 / counts.sum()],
                     dtype=jnp.float32)


def sample_replay_capped(
    key: jax.Array,
    level_weights: jax.Array,
    frontier_mask: jax.Array,
    *,
    batch_size: int = 32,
    max_copies: int = 8,
    preserve_frontier_mass: bool = True,
):
    """Sample replay indices from a *snapshot*, with optional per-index cap.

    If too few positive-weight indices exist for the cap, fall back to ordinary
    replacement sampling. This is useful for small technical smoke tests.

    Within each draw, frontier / non-frontier probability mass is reweighted
    toward the original group masses, when both groups are still available.
    This is NOT a fixed quota of frontier items in the final batch.

    IMPORTANT: this helper does NOT update LevelSampler staleness/timestamps.
    The fork's actual sampler must retain its state update on integration.

    At least one positive-weight entry is required. Empty buffers must not call
    this sampler; JaxUED normally prevents replay before the fill threshold.
    """
    w = jnp.asarray(level_weights, dtype=jnp.float32)
    frontier = jnp.asarray(frontier_mask, dtype=jnp.bool_)
    w = jnp.where(jnp.isfinite(w) & (w > 0), w, 0.0)
    w = w / jnp.maximum(jnp.sum(w), 1e-20)
    can_cap = jnp.sum(w > 0) * max_copies >= batch_size

    def select(carry, _):
        rng, counts = carry
        rng, draw_rng = jax.random.split(rng)
        eligible = (w > 0) & ((counts < max_copies) | ~can_cap)
        probs = capped_replay_weights(w, frontier, eligible, preserve_frontier_mass=preserve_frontier_mass)
        logits = jnp.where(probs > 0, jnp.log(probs), -jnp.inf)
        idx = jax.random.categorical(draw_rng, logits)
        return (rng, counts.at[idx].add(1)), idx

    (_, counts), indices = jax.lax.scan(
        select, (key, jnp.zeros_like(w, dtype=jnp.int32)), None, length=batch_size
    )
    return indices, counts
