"""Teacher scoring from the last actor layer and a shared replay percentile."""

import jax
import jax.numpy as jnp

from jaxued import pace_adaptive


HYBRID = "hybrid_pace_lite_floor"
STANDALONE = "pace_lite_floor"
METHODS = (HYBRID, STANDALONE)


def head_gradients(features, logits, actions, advantages, mask, *, bias=True):
    """Mean policy-gradient kernel and bias per environment in a [T, B] rollout."""
    features, logits, advantages = jax.lax.stop_gradient((features, logits, advantages))
    valid = (mask.astype(bool) & jnp.isfinite(advantages)
             & jnp.isfinite(features).all(axis=-1) & jnp.isfinite(logits).all(axis=-1)
             & (actions >= 0) & (actions < logits.shape[-1]))
    clean_logits = jnp.where(valid[..., None], logits, 0)
    probs = jax.nn.softmax(clean_logits, axis=-1)
    delta = jnp.where(valid, advantages, 0)[..., None] * (probs - jax.nn.one_hot(actions, logits.shape[-1]))
    features = jnp.where(valid[..., None], features, 0)
    count = valid.sum(axis=0)
    denom = jnp.maximum(count, 1)
    kernel = jnp.einsum("tbh,tba->bha", features, delta, precision=jax.lax.Precision.HIGHEST)
    return kernel / denom[:, None, None], delta.sum(axis=0) / denom[:, None] if bias else None, count


def scores(features, logits, actions, advantages):
    """Use the full AutoReplay rollout, including terminal steps and the GAE tail."""
    mask = (jnp.isfinite(advantages) & jnp.isfinite(features).all(axis=-1)
            & jnp.isfinite(logits).all(axis=-1) & (actions >= 0) & (actions < logits.shape[-1]))
    count = jnp.maximum(mask.sum(), 1)
    mean = jnp.where(mask, advantages, 0).sum() / count
    centered = jnp.where(mask, advantages - mean, 0)
    normalized = centered / (jnp.sqrt(jnp.square(centered).sum() / count) + 1e-5)
    kernel, bias, observations = head_gradients(features, logits, actions, normalized, mask)
    squared = jnp.square(kernel).sum(axis=(1, 2)) + jnp.square(bias).sum(axis=1)
    return jnp.where(observations > 0, squared, jnp.nan)


def percentiles(values, usable):
    """Mid-rank percentiles under a single masked empirical CDF."""
    reference, order = jax.lax.sort_key_val(jnp.where(usable, values, jnp.inf),
                                           jnp.arange(values.size, dtype=jnp.int32))
    different = reference[1:] != reference[:-1]
    position = jnp.arange(values.size, dtype=jnp.int32)
    first = jax.lax.cummax(jnp.where(jnp.concatenate((jnp.array([True]), different)), position, 0), 0)
    last = jax.lax.cummin(jnp.where(jnp.concatenate((different, jnp.array([True]))), position, values.size),
                         0, reverse=True)
    sorted_pct = (first + last + 1) / (2 * jnp.maximum(usable.sum(), 1))
    return jnp.where(usable, jnp.zeros_like(sorted_pct).at[order].set(sorted_pct), 0)


def synchronize(sampler, incoming, options, hybrid):
    """Use one reference for occupied slots and the incoming rollout."""
    stored = sampler["levels_extra"]
    capacity = sampler["scores"].size
    squared = jnp.concatenate((stored["grad_squared"], incoming["grad_squared"]))
    occupied = jnp.arange(capacity) < sampler["size"]
    active = jnp.concatenate((occupied, jnp.ones(incoming["grad_squared"].shape, dtype=bool)))
    valid = active & jnp.isfinite(squared) & (squared >= 0)
    _, optimal, floor = pace_adaptive.components(squared, sampler["pace_threshold"],
                                                options["pace_shape"], options["pace_eps"])
    beta = options["pace_floor_weight"]
    pace = (1 - beta) * optimal + beta * floor
    if hybrid:
        maxmc = jnp.concatenate((stored["maxmc"], incoming["maxmc"]))
        alpha = options["hybrid_pace_weight"]
        mixed = ((1 - alpha) * percentiles(maxmc, valid & jnp.isfinite(maxmc))
                 + alpha * percentiles(pace, valid))
    else:
        mixed = pace
    scores = jnp.where(valid, mixed, -jnp.inf)
    return {**sampler, "scores": scores[:capacity]}, scores[capacity:]


def insert_batch(level_sampler, sampler, levels, scores, extra):
    """Skip invalid candidates entirely, including duplicate inserts."""
    def insert(state, item):
        level, score, observation = item
        updated = jax.lax.cond(jnp.isfinite(score),
                               lambda: level_sampler.insert(state, level, score, observation)[0],
                               lambda: state)
        return updated, None

    return jax.lax.scan(insert, sampler, (levels, scores, extra))[0]
