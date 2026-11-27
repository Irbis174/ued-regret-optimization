import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxued import pace_adaptive as pace
from jaxued.pace_validation import metrics as validation_metrics
from jaxued.level_sampler import LevelSampler


OPTIONS = pace.DEFAULTS
METHOD = "pace_adaptive_floor"


def test_shapes_peaks_floor_and_invalid_inputs_under_jit_and_vmap():
    norms = jnp.array([0., 1e-7, .8, 1., 1.2, 1e19, jnp.nan, jnp.inf, -1.])
    squared = jnp.where(norms < 0, norms, norms ** 2)
    run = jax.jit(jax.vmap(lambda values: pace.components(values, 1., 1., 1e-8)))
    x, optimal, floor = run(jnp.stack([squared] * 2))
    assert x.shape == (2, len(norms))
    assert np.isfinite([x, optimal, floor]).all()
    np.testing.assert_allclose(optimal[:, 3], 1)
    assert int(optimal[0].argmax()) == 3
    score = .8 * optimal + .2 * floor
    assert score[0, :6].argmax() in (3, 4)
    assert score[0, 5] == pytest.approx(.2)
    np.testing.assert_array_equal(score[:, jnp.array([0, 6, 7, 8])], 0)
    assert np.isfinite(pace.components(jnp.float32(3e38), 0., 1., 1e-19)).all()


def test_quantile_and_ema_ignore_invalid_and_do_not_prime_on_zero():
    def update(prior, scores):
        _, _, current, _ = pace.grad_statistics(scores, .75)
        return pace.adapt_threshold(prior, current, .95, 1e-8)

    update = jax.jit(update)
    assert update(0., jnp.zeros(6)) == 0.
    assert update(0., jnp.array([jnp.inf, jnp.nan, -1, 1., 4., 9.])) == 2.5
    assert update(2.5, jnp.array([16.] * 6)) == pytest.approx(2.575)
    assert update(2.5, jnp.zeros(6)) == 2.5
    assert update(0., jnp.array([1e-20] * 6)) == pytest.approx(1e-8)


def test_hard_validation_tier_and_hard_aggregate_are_separate():
    result = validation_metrics(np.array([[0., 0., 1., 0.]]), np.arange(4))
    assert result["eval/validation_tier_hard_solve_rate"] == 1.
    assert result["eval/validation_tier_long_solve_rate"] == 0.
    assert result["eval/validation_hard_solve_rate"] == .5


def test_threshold_reweights_old_slots_and_preserves_staleness(tmp_path):
    import orbax.checkpoint as ocp

    sampler = LevelSampler(4, .8, .3, .5, "rank", {"temperature": .3}, duplicate_check=True)
    state = sampler.initialize(jnp.int32(0), pace.empty_extra(False))
    state["pace_threshold"] = jnp.float32(1.)
    state, _ = sampler.insert_batch(state, jnp.array([1, 2, 3]), jnp.array([.9, .7, .3]), {
        "grad_squared": jnp.array([1., 4., 9.]), "max_return": jnp.zeros(3),
    })
    options = {**OPTIONS, "pace_threshold_ema": 0.}
    prepare = jax.jit(lambda s: pace.prepare(s, jnp.array([16., 16.]), options, METHOD))
    updated, _, metrics = prepare(state)
    assert updated["pace_threshold"] == 4.
    assert int(updated["scores"].argmax()) == 2
    np.testing.assert_array_equal(updated["timestamps"], state["timestamps"])
    assert jnp.isneginf(updated["scores"][3])
    assert np.isfinite(list(metrics.values())).all()
    weights = sampler.level_weights(updated)
    assert (weights >= 0).all() and weights[3] == 0
    assert weights.sum() == pytest.approx(1.)
    key = jax.random.PRNGKey(7)
    checkpointer = ocp.StandardCheckpointer()
    path = tmp_path / "checkpoint"
    try:
        checkpointer.save(path, {"sampler": updated, "rng": key})
        restored = checkpointer.restore(path)
    finally:
        checkpointer.close()
    assert restored["sampler"]["pace_threshold"] == 4.
    original = sampler.sample_replay_levels(updated, key, 4)[1][0]
    resumed = sampler.sample_replay_levels(restored["sampler"], restored["rng"], 4)[1][0]
    np.testing.assert_array_equal(original, resumed)
