"""Estimate level learnability from recent episodes already collected by PLR."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jaxued.utils import max_mc
from jaxued.teacher_experiments import har_score, update_progress_memory


# Age in curriculum rounds, including rounds where the student is frozen.
HALF_LIFE = 256
FULL_WEIGHT_EPISODES = 4


class RolloutEvidence(NamedTuple):
    successes: jax.Array
    episodes: jax.Array
    max_return: jax.Array
    maxmc: jax.Array


def empty_history(*, track_progress=False):
    history = {
        "successes": jnp.float32(0),
        "episodes": jnp.float32(0),
        "max_return": jnp.float32(-jnp.inf),
        "last_round": jnp.int32(0),
    }
    if track_progress:
        history.update(fast_ema=jnp.float32(0), slow_ema=jnp.float32(0), initialized=jnp.bool_(False))
    return history


def rollout_evidence(dones, rewards, values, max_returns):
    """Each completion contributes a solve or a timeout; unfinished tails contribute neither."""
    return RolloutEvidence(
        (dones & (rewards > 0)).sum(axis=0),
        dones.sum(axis=0),
        max_returns,
        max_mc(dones, values, max_returns),
    )


def record_evidence(history, previous_score, evidence, round_index, *, history_weight=None):
    """Refresh the level's estimate and its priority when an episode finishes."""
    def update():
        retention = jnp.exp2(-(round_index - history["last_round"]) / HALF_LIFE)
        successes = retention * history["successes"] + evidence.successes
        episodes = retention * history["episodes"] + evidence.episodes
        max_return = jnp.maximum(history["max_return"], evidence.max_return)
        p = successes / episodes
        learnability = 4 * p * (1 - p)
        weight = jnp.minimum(episodes / FULL_WEIGHT_EPISODES, 1.0)
        # MaxMC is affine in max_return. Adjust it for the saved best return
        # and put the fallback in the same [0, 1] range as learnability.
        maxmc = evidence.maxmc + (max_return - evidence.max_return)
        if history_weight is None or history_weight == 0:
            fallback = jax.nn.sigmoid(maxmc)
            score = (1 - weight) * fallback + weight * learnability
        else:
            score = har_score(maxmc, successes, episodes, history["fast_ema"], history["slow_ema"],
                              history_weight=history_weight, warmup_episodes=FULL_WEIGHT_EPISODES)
        return score, {
            **history,
            "successes": successes,
            "episodes": episodes,
            "max_return": max_return,
            "last_round": round_index,
        }

    return jax.lax.cond(evidence.episodes > 0, update, lambda: (previous_score, history))


def insert_batch(level_sampler, sampler, levels, evidence, round_index, *, history_weight=None):
    track_progress = history_weight is not None
    grouped = (evidence.successes, evidence.episodes) if track_progress else None
    if track_progress and level_sampler.duplicate_check:
        # Associate identical new/mutated levels before they receive buffer indices.
        size = evidence.episodes.shape[0]
        same = jnp.ones((size, size), dtype=bool)
        for leaf in jax.tree_util.tree_leaves(levels):
            flat = leaf.reshape(size, -1)
            same &= (flat[:, None, :] == flat[None, :, :]).all(axis=-1)
        grouped = tuple(jnp.where(same, x[None, :], 0).sum(axis=1) for x in grouped)

    def insert(sampler, row):
        level, observation, progress = row
        history, score = empty_history(track_progress=track_progress), jnp.float32(-jnp.inf)
        if level_sampler.duplicate_check:
            index = level_sampler.find(sampler, level)
            history, score = jax.lax.cond(
                index >= 0,
                lambda: (level_sampler.get_levels_extra(sampler, index), sampler["scores"][index]),
                lambda: (history, score),
            )
        if track_progress:
            successes, episodes = progress
            first = (~history["initialized"] | (history["last_round"] != round_index)) & (observation.episodes > 0)
            fast, slow, initialized = update_progress_memory(
                history["fast_ema"][None], history["slow_ema"][None], history["initialized"][None],
                jnp.zeros(1, dtype=jnp.int32), successes[None], jnp.where(first, episodes, 0)[None],
            )
            history = {**history, "fast_ema": fast[0], "slow_ema": slow[0], "initialized": initialized[0]}
        score, history = record_evidence(history, score, observation, round_index, history_weight=history_weight)
        return level_sampler.insert(sampler, level, score, history)

    return jax.lax.scan(insert, sampler, (levels, evidence, grouped))[0]


def update_batch(level_sampler, sampler, indices, evidence, round_index, *, history_weight=None):
    if history_weight is not None:
        history = sampler["levels_extra"]
        fast, slow, initialized = update_progress_memory(
            history["fast_ema"], history["slow_ema"], history["initialized"],
            indices, evidence.successes, evidence.episodes,
        )
        history = {**history, "fast_ema": fast, "slow_ema": slow, "initialized": initialized}
        sampler = {**sampler, "levels_extra": history}

    def update(sampler, row):
        index, observation = row
        # Sampling with replacement can produce the same index more than once.
        # Read the running history so every rollout is counted exactly once.
        history = level_sampler.get_levels_extra(sampler, index)
        score, history = record_evidence(
            history, sampler["scores"][index], observation, round_index, history_weight=history_weight
        )
        return level_sampler.update(sampler, index, score, history), None

    return jax.lax.scan(update, sampler, (indices, evidence))[0]
