"""Per-level policy gradient scores for recurrent rollouts."""

import jax
import jax.numpy as jnp


def compute_pace_pg_scores(
    apply_fn, params, obs, actions, dones, advantages, init_hstate, *,
    chunk_size=1, entropy_coeff=0.0,
):
    """Return one squared actor-gradient norm per environment in a [T, B] rollout."""
    if chunk_size < 1:
        raise ValueError("PACE chunk size must be positive")
    if actions.ndim != 2 or not all(actions.shape) or not actions.shape == dones.shape == advantages.shape:
        raise ValueError("PACE expects matching, nonempty [time, environment] arrays")

    last_dones = jnp.roll(dones, 1, axis=0).at[0].set(False)
    normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
    rollout = jax.lax.stop_gradient((obs, actions, last_dones, normalized))
    init_hstate = jax.lax.stop_gradient(init_hstate)
    params = jax.lax.stop_gradient(params)

    def score_one(index):
        obs_i, actions_i, last_dones_i, normalized_i = jax.tree_util.tree_map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, index, 1, axis=1), rollout
        )
        hstate_i = jax.tree_util.tree_map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, index, 1, axis=0), init_hstate
        )

        def actor_loss(params):
            _, policy, _ = apply_fn(params, (obs_i, last_dones_i), hstate_i)
            loss = -(normalized_i * policy.log_prob(actions_i)).mean()
            if entropy_coeff:
                loss -= entropy_coeff * policy.entropy().mean()
            return loss

        grads = jax.grad(actor_loss)(params)
        return sum(jnp.square(g.astype(jnp.float32)).sum() for g in jax.tree_util.tree_leaves(grads))

    batch_size = actions.shape[1]
    indices = jnp.arange(batch_size)
    if chunk_size == 1:
        return jax.lax.map(score_one, indices)
    chunk_size = min(chunk_size, batch_size)
    # Pad the final chunk. Repeating a score consumes no environment steps or RNG.
    padding = (-batch_size) % chunk_size
    chunks = jnp.pad(indices, (0, padding), mode="edge").reshape(-1, chunk_size)
    return jax.lax.map(jax.vmap(score_one), chunks).reshape(-1)[:batch_size]


def rollout_statistics(scores, advantages, dones, rewards):
    """Summarize a score batch, reporting nonfinite entries before sanitization."""
    finite = jnp.isfinite(scores)
    safe = jnp.where(finite, scores, 0)
    ordered = jnp.sort(safe)
    quantiles = jnp.quantile(ordered, jnp.array([0.1, 0.5, 0.9]))
    return {
        "pace/mean_score": safe.mean(),
        "pace/mean_batch_score_p10": quantiles[0],
        "pace/mean_batch_score_p50": quantiles[1],
        "pace/mean_batch_score_p90": quantiles[2],
        "pace/max_score": ordered[-1],
        "pace/nonfinite_fraction": (~finite).mean(),
        "pace/zero_score_fraction": (safe <= 1e-12).mean(),
        "pace/duplicate_score_fraction": (ordered[1:] == ordered[:-1]).sum() / scores.size,
        "pace/advantage_std": advantages.std(),
        "pace/levels_with_success_fraction": (dones & (rewards > 0)).any(axis=0).mean(),
    }
