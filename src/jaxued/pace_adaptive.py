"""Adaptive priorities over the original PACE squared actor-gradient norm."""

import math

import jax.numpy as jnp


DEFAULTS = {
    "pace_quantile": 0.75,
    "pace_floor_weight": 0.20,
    "pace_threshold_ema": 0.95,
    "pace_shape": 1.0,
    "pace_eps": 1e-8,
    "pace_warmup_fraction": 0.0,
    "pace_fixed_threshold": None,
    "hybrid_pace_weight": 0.20,
}
METHODS = ("pace_adaptive_floor", "pace_adaptive", "pace_optimal", "pace_clipped", "pace_hybrid")
PERCENTILES = (25, 50, 75, 90, 95, 99)


def validate(config):
    options = {**DEFAULTS, **{key: config[key] for key in DEFAULTS if key in config}}
    for key, value in options.items():
        if value is None and key == "pace_fixed_threshold":
            continue
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        if key in ("pace_shape", "pace_eps", "pace_fixed_threshold"):
            if not 0 < value <= 1e18 or (key == "pace_eps" and value < 1e-19):
                raise ValueError(f"invalid positive float32 scale for {key}")
        elif not 0 <= value <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    if config["score_function"] == "pace_optimal" and options["pace_fixed_threshold"] is None:
        raise ValueError("pace_optimal requires --pace_fixed_threshold from training diagnostics")
    return options


def empty_extra(hybrid):
    extra = {"max_return": jnp.float32(-jnp.inf), "grad_squared": jnp.float32(0)}
    if hybrid:
        extra["maxmc"] = jnp.float32(-jnp.inf)
    return extra


def grad_statistics(squared, quantile):
    valid = jnp.isfinite(squared) & (squared >= 0)
    norms = jnp.sqrt(jnp.where(valid, squared, 0))
    sorted_norms = jnp.sort(jnp.where(valid, norms, jnp.inf))
    count = valid.sum()
    max_index = jnp.maximum(count - 1, 0)

    def at(q):
        rank = q * max_index
        low, high = rank.astype(jnp.int32), jnp.ceil(rank).astype(jnp.int32)
        a = jnp.where(count > 0, sorted_norms[low], 0)
        b = jnp.where(count > 0, sorted_norms[high], 0)
        return a + (b - a) * (rank - low)

    return valid, norms, at(quantile), jnp.stack([at(p / 100) for p in PERCENTILES])


def adapt_threshold(previous, current, ema, eps):
    """Keep zero as the uninitialized sentinel until a positive quantile exists."""
    proposal = jnp.where(previous > 0, ema * previous + (1 - ema) * current, current)
    # A completely zero or unusable batch gives no scale evidence.
    return jnp.where(current > 0, jnp.maximum(proposal, eps), previous)


def components(squared, threshold, shape, eps):
    valid = jnp.isfinite(squared) & (squared >= 0)
    # The source already returns squared norms. Cap x only for float32 safety.
    scale = jnp.maximum(threshold, eps)
    x = jnp.minimum((jnp.where(valid, squared, 0) / scale) / scale, 1e12)
    log_optimal = jnp.log(jnp.maximum(x, jnp.finfo(jnp.float32).tiny)) + 1 - x
    optimal = jnp.where(x > 0, jnp.exp(shape * jnp.minimum(log_optimal, 0)), 0)
    return x, optimal, x / (1 + x)


def priority(extra, threshold, options, method, warmup=False):
    squared = extra["grad_squared"]
    if method == "pace_clipped":
        return jnp.minimum(squared, jnp.square(jnp.maximum(threshold, options["pace_eps"])))
    _, optimal, floor = components(squared, threshold, options["pace_shape"], options["pace_eps"])
    beta = 0.0 if method in ("pace_adaptive", "pace_optimal") else options["pace_floor_weight"]
    pace = (1 - beta) * optimal + beta * floor
    if method == "pace_hybrid":
        alpha = options["hybrid_pace_weight"]
        # A fixed, shared map preserves comparability with every stored level.
        maxmc_unit = 0.5 + jnp.arctan(extra["maxmc"] / 0.1) / jnp.pi
        return (1 - alpha) * maxmc_unit + alpha * pace
    return jnp.where(warmup, squared, pace)


def reweight(sampler, threshold, options, method, warmup=False):
    active = jnp.arange(sampler["scores"].size) < sampler["size"]
    scores = priority(sampler["levels_extra"], threshold, options, method, warmup)
    return {**sampler, "pace_threshold": threshold, "scores": jnp.where(active, scores, -jnp.inf)}


def prepare(sampler, squared, options, method, warmup=False, *, rescore=True):
    nonfinite = (~jnp.isfinite(squared)).mean()
    valid, norms, current, quantiles = grad_statistics(squared, options["pace_quantile"])
    fixed = options["pace_fixed_threshold"]
    threshold = (jnp.float32(fixed) if fixed is not None else
                 adapt_threshold(sampler["pace_threshold"], current,
                                 options["pace_threshold_ema"], options["pace_eps"]))
    sampler = (reweight(sampler, threshold, options, method, warmup) if rescore
               else {**sampler, "pace_threshold": threshold})
    squared = jnp.where(valid, squared, 0)
    x, optimal, floor = components(squared, threshold, options["pace_shape"], options["pace_eps"])
    denom = jnp.maximum(valid.sum(), 1)
    mean = norms.sum() / denom
    # Scale before squaring to keep dispersion finite at float32 extremes.
    max_norm = jnp.maximum(norms.max(), options["pace_eps"])
    std = max_norm * jnp.sqrt(jnp.where(valid, (norms / max_norm - mean / max_norm) ** 2, 0).sum() / denom)
    stats = {
        "pace/grad_norm_mean": mean, "pace/grad_norm_std": std,
        "pace/grad_norm_min": jnp.where(valid.any(), jnp.where(valid, norms, jnp.inf).min(), 0),
        "pace/grad_norm_max": norms.max(), "pace/threshold": threshold,
        "pace/threshold_current_quantile": current,
        "pace/normalized_grad_mean": jnp.sqrt(x).mean(), "pace/normalized_grad_std": jnp.sqrt(x).std(),
        "pace/optimal_score_mean": optimal.mean(), "pace/floor_score_mean": floor.mean(),
        "pace/above_threshold_fraction": (valid & (norms > threshold)).sum() / denom,
        "pace/nonfinite_fraction": nonfinite, "pace/invalid_fraction": (~valid).mean(),
    }
    stats.update({f"pace/grad_norm_p{p}": v for p, v in zip(PERCENTILES, quantiles)})
    return sampler, squared, stats


def rollout_metrics(scores):
    score_scale = jnp.maximum(scores.max(), 1e-30)
    return {
        "pace/score_mean": (scores / score_scale).mean() * score_scale,
        "pace/score_std": (scores / score_scale).std() * score_scale,
        "pace/score_min": scores.min(), "pace/score_max": scores.max(),
        "pace/near_zero_score_fraction": (scores <= 1e-8).mean(),
    }
