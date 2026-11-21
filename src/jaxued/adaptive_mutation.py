"""Teacher-only operator selection and fitness for Adaptive Mutation ACCEL."""

import jax
import jax.numpy as jnp

DEFAULT_ARMS = (1, 3, 5, 10)
MAX_EDITS = 100


def initialize(num_arms):
    return {
        "count": jnp.zeros(num_arms, dtype=jnp.int32),
        "ema_reward": jnp.zeros(num_arms, dtype=jnp.float32),
        "reward_sum": jnp.zeros(num_arms, dtype=jnp.float32),
        "last_selected": jnp.full(num_arms, -1, dtype=jnp.int32),
    }


def sample_arms(key, state, batch_size, epsilon):
    """Draw epsilon-greedy actions, reserving pulls for unseen arms in this batch."""
    n = state["count"].shape[0]
    best = state["ema_reward"] == state["ema_reward"].max()

    def draw(counts, key):
        coin, pick = jax.random.split(key)
        unseen = counts == 0
        explore = jax.random.uniform(coin) < epsilon
        allowed = jnp.where(unseen.any(), unseen, best | explore)
        index = jax.random.categorical(pick, jnp.where(allowed, 0., -jnp.inf)).astype(jnp.int32)
        return counts.at[index].add(1), index

    return jax.lax.scan(draw, state["count"], jax.random.split(key, batch_size))[1]


def sample_uniform(key, batch_size, num_arms):
    return jax.random.randint(key, (batch_size,), 0, num_arms, dtype=jnp.int32)


def update(state, indices, rewards, alpha, round_index):
    """Apply one EMA update per used arm with that arm's mean batch reward."""
    counts = jnp.zeros_like(state["count"]).at[indices].add(1)
    sums = jnp.zeros_like(state["reward_sum"]).at[indices].add(rewards)
    means = sums / jnp.maximum(counts, 1)
    estimate = (1 - alpha) * state["ema_reward"] + alpha * means
    return {
        "count": state["count"] + counts,
        "reward_sum": state["reward_sum"] + sums,
        "ema_reward": jnp.where(counts > 0, estimate, state["ema_reward"]),
        "last_selected": jnp.where(counts > 0, round_index, state["last_selected"]),
    }


def mutate_batch(mutator, keys, parents, indices, arms):
    return jax.vmap(mutator, (0, 0, 0))(keys, parents, jnp.asarray(arms, dtype=jnp.int32)[indices])


def valid_child(parent, child):
    y, x = jnp.indices(child.wall_map.shape)
    inside = (x < child.width) & (y < child.height)
    return (child.is_well_formatted() & (child.width == parent.width) & (child.height == parent.height)
            & (child.agent_dir < 4) & jnp.all(inside | child.wall_map))


def candidate_stats(parents, children):
    """Return structural validity, parent duplicates, pairwise identity and wall changes."""
    size = parents.wall_map.shape[0]
    matches_parent = jnp.ones(size, dtype=bool)
    same = jnp.ones((size, size), dtype=bool)
    for parent, child in zip(jax.tree_util.tree_leaves(parents), jax.tree_util.tree_leaves(children)):
        parent, child = parent.reshape(size, -1), child.reshape(size, -1)
        matches_parent &= (parent == child).all(axis=1)
        same &= (child[:, None, :] == child[None, :, :]).all(axis=-1)
    changes = jnp.sum(parents.wall_map != children.wall_map, axis=(1, 2))
    return jax.vmap(valid_child)(parents, children), matches_parent, same, changes


def rank_rewards(scores, eligible):
    """Ascending midpoint ranks on eligible finite scores; singleton gets 0.5."""
    eligible = eligible & jnp.isfinite(scores)
    left, right = scores[:, None], scores[None, :]
    lower = ((left > right) & eligible[None, :]).sum(axis=1)
    tied = ((left == right) & eligible[None, :]).sum(axis=1)
    size = eligible.sum()
    rank = (lower + 0.5 * (tied - 1)) / jnp.maximum(size - 1, 1)
    return jnp.where(eligible, jnp.where(size == 1, 0.5, rank), 0.0)


def assess(scores, stats, strategy, area):
    valid, parent_duplicate, same, changes = stats
    novel = valid & ~parent_duplicate
    # All colliding novel candidates lose reward, independent of their position.
    eligible = novel & ((same & novel[None, :]).sum(axis=1) == 1)
    if strategy == "changed_cells":
        return jnp.where(eligible, changes / area, 0.0)
    return rank_rewards(scores, eligible)


def batch_metrics(stats, rewards, indices, num_arms):
    valid, _, same, changes = stats
    unique = valid & ~jnp.tril(same & valid[None, :], k=-1).any(axis=1)
    counts = jnp.zeros(num_arms, dtype=jnp.float32).at[indices].add(1)
    return jnp.concatenate((jnp.array([1, rewards.mean(), unique.mean(), (~valid).mean(), changes.mean()]),
                            counts))


def safe_children(parents, children, valid):
    """Substitute parents for invalid slots without adding any environment work."""
    return jax.vmap(lambda parent, child, ok: jax.lax.cond(ok, lambda: child, lambda: parent))(
        parents, children, valid
    )


def insert_valid(insert_batch, sampler, children, valid, rollout):
    """Use ordinary insertion, skipping invalid slots even on the deduplication path."""
    def insert_individually():
        def visit(sampler, row):
            level, accept, trajectory = row
            def insert():
                batch = jax.tree_util.tree_map(lambda x: x[None], level)
                return insert_batch(sampler, batch, *(x[:, None] for x in trajectory))
            return jax.lax.cond(accept, insert, lambda: sampler), None

        per_slot = tuple(jnp.swapaxes(x, 0, 1) for x in rollout)
        return jax.lax.scan(visit, sampler, (children, valid, per_slot))[0]

    return jax.lax.cond(valid.all(), lambda: insert_batch(sampler, children, *rollout), insert_individually)
