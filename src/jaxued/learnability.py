"""Estimate level learnability from recent episodes already collected by PLR."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jaxued.utils import max_mc


# Age in curriculum rounds, including rounds where the student is frozen.
HALF_LIFE = 256
FULL_WEIGHT_EPISODES = 4


class RolloutEvidence(NamedTuple):
    successes: jax.Array
    episodes: jax.Array
    max_return: jax.Array
    maxmc: jax.Array


def empty_history():
    return {
        "successes": jnp.float32(0),
        "episodes": jnp.float32(0),
        "max_return": jnp.float32(-jnp.inf),
        "last_round": jnp.int32(0),
    }


def rollout_evidence(dones, rewards, values, max_returns):
    """Each completion contributes a solve or a timeout; unfinished tails contribute neither."""
    return RolloutEvidence(
        (dones & (rewards > 0)).sum(axis=0),
        dones.sum(axis=0),
        max_returns,
        max_mc(dones, values, max_returns),
    )


def record_evidence(history, previous_score, evidence, round_index):
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
        fallback = jax.nn.sigmoid(evidence.maxmc + (max_return - evidence.max_return))
        score = (1 - weight) * fallback + weight * learnability
        return score, {
            "successes": successes,
            "episodes": episodes,
            "max_return": max_return,
            "last_round": round_index,
        }

    return jax.lax.cond(evidence.episodes > 0, update, lambda: (previous_score, history))


def insert_batch(level_sampler, sampler, levels, evidence, round_index):
    def insert(sampler, row):
        level, observation = row
        history, score = empty_history(), jnp.float32(-jnp.inf)
        if level_sampler.duplicate_check:
            index = level_sampler.find(sampler, level)
            history, score = jax.lax.cond(
                index >= 0,
                lambda: (level_sampler.get_levels_extra(sampler, index), sampler["scores"][index]),
                lambda: (history, score),
            )
        score, history = record_evidence(history, score, observation, round_index)
        return level_sampler.insert(sampler, level, score, history)

    return jax.lax.scan(insert, sampler, (levels, evidence))[0]


def update_batch(level_sampler, sampler, indices, evidence, round_index):
    def update(sampler, row):
        index, observation = row
        # Sampling with replacement can produce the same index more than once.
        # Read the running history so every rollout is counted exactly once.
        history = level_sampler.get_levels_extra(sampler, index)
        score, history = record_evidence(history, sampler["scores"][index], observation, round_index)
        return level_sampler.update(sampler, index, score, history), None

    return jax.lax.scan(update, sampler, (indices, evidence))[0]
