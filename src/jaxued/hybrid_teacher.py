"""Track the two native priorities on the same occupied LevelSampler slots."""

import jax
import jax.numpy as jnp

from jaxued import learnability
from jaxued.utils import max_mc


MODES = ("recent_control", "hard_switch", "smooth")
DEFAULTS = {
    "dual_score_tracking": False, "hybrid_mode": "recent_control",
    "hybrid_start_round": 500, "hybrid_end_round": 750,
}


def coefficient(round_index, mode, start, end):
    if mode == "recent_control":
        return jnp.float32(0)
    if mode == "hard_switch":
        return (jnp.asarray(round_index) >= start).astype(jnp.float32)
    return jnp.clip((jnp.asarray(round_index, dtype=jnp.float32) - start) / (end - start), 0., 1.)


def config_coefficient(config, round_index):
    return coefficient(round_index, config["hybrid_mode"], config["hybrid_start_round"], config["hybrid_end_round"])


def empty_history():
    return {
        **learnability.empty_history(),
        "recent_score": jnp.float32(0), "recent_initialized": jnp.bool_(False),
        "maxmc_score": jnp.float32(0), "maxmc_initialized": jnp.bool_(False),
        "maxmc_max_return": jnp.float32(0),
    }


def priorities(history, weight, *, raw_rank=False):
    recent = history["recent_score"]
    native = jnp.where(history["maxmc_initialized"], history["maxmc_score"], -jnp.inf)
    mixed = (1 - weight) * recent + weight * jax.nn.sigmoid(native)
    mixed = jnp.where(weight == 0, recent, mixed)
    # The native key preserves rank even when float32 sigmoid saturates.
    if raw_rank:
        mixed = jnp.where(weight == 1, native, mixed)
    return jnp.where(history["recent_initialized"], mixed, -jnp.inf)


def reweight(sampler, weight, *, raw_rank=False):
    active = jnp.arange(sampler["scores"].shape[0]) < sampler["size"]
    scores = priorities(sampler["levels_extra"], weight, raw_rank=raw_rank)
    return {**sampler, "scores": jnp.where(active, scores, -jnp.inf)}


def record(history, evidence, native_return, native_score, round_index):
    recent, updated = learnability.record_evidence(history, history["recent_score"], evidence, round_index)
    native_valid = jnp.isfinite(native_score)
    return {
        **updated,
        "recent_score": recent,
        "recent_initialized": history["recent_initialized"] | (evidence.episodes > 0),
        "maxmc_max_return": native_return,
        "maxmc_score": jnp.where(native_valid, native_score, 0.),
        "maxmc_initialized": native_valid,
    }


def insert_batch(level_sampler, sampler, levels, evidence, round_index, weight, *, raw_rank=False):
    def insert(sampler, row):
        level, observation = row
        history = empty_history()
        if level_sampler.duplicate_check:
            index = level_sampler.find(sampler, level)
            history = jax.lax.cond(index >= 0, lambda: level_sampler.get_levels_extra(sampler, index), lambda: history)
        history = record(history, observation, observation.max_return, observation.maxmc, round_index)
        return level_sampler.insert(sampler, level, priorities(history, weight, raw_rank=raw_rank), history)

    return jax.lax.scan(insert, sampler, (levels, evidence))[0]


def update_batch(level_sampler, sampler, indices, evidence, dones, values, round_index, weight, *, raw_rank=False):
    # Original MaxMC freezes historical best returns before processing repeated replay indices.
    native_returns = jnp.maximum(sampler["levels_extra"]["maxmc_max_return"][indices], evidence.max_return)
    native_scores = max_mc(dones, values, native_returns)

    def update(sampler, row):
        index, observation, native_return, native_score = row
        history = level_sampler.get_levels_extra(sampler, index)
        history = record(history, observation, native_return, native_score, round_index)
        updated = level_sampler.update(sampler, index, priorities(history, weight, raw_rank=raw_rank), history)
        return updated, None

    return jax.lax.scan(update, sampler, (indices, evidence, native_returns, native_scores))[0]


def statistics(sampler, weight):
    extra = sampler["levels_extra"]
    active = jnp.arange(sampler["scores"].shape[0]) < sampler["size"]
    observed = active & extra["maxmc_initialized"]
    return {
        "hybrid/lambda": weight,
        "hybrid/mean_recent": jnp.where(active, extra["recent_score"], 0).sum() / jnp.maximum(active.sum(), 1),
        "hybrid/mean_maxmc": jnp.where(observed, extra["maxmc_score"], 0).sum() / jnp.maximum(observed.sum(), 1),
        "hybrid/mean_priority": jnp.where(active, sampler["scores"], 0).sum() / jnp.maximum(active.sum(), 1),
        "hybrid/maxmc_observed": observed.sum(),
    }
