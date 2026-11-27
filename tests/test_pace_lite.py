import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxued import pace_adaptive, pace_lite


@pytest.mark.parametrize("use_bias", [True, False])
def test_head_gradients_match_autodiff(use_bias):
    h = jax.random.normal(jax.random.PRNGKey(21), (8, 3, 5))
    w = jax.random.normal(jax.random.PRNGKey(22), (5, 4))
    b = jnp.array([1., -1., 1e3, -1e3]) if use_bias else jnp.zeros(4)
    a = jnp.arange(24).reshape(8, 3) % 4
    gae = jnp.linspace(-1., 1., 24).reshape(8, 3).at[:, 0].set(0).at[0, 1].set(jnp.inf)
    mask = jnp.ones((8, 3), dtype=bool).at[-1, 1].set(False).at[:, 2].set(False)
    valid = mask & jnp.isfinite(gae)
    h = h.at[0, 1, 0].set(jnp.nan)
    gae_clean = jnp.where(valid, gae, 0)
    h_clean = jnp.where(valid[..., None], h, 0)

    def loss(kernel, bias, features, actions, advantage, mask):
        logits = features @ kernel + (bias if use_bias else 0)
        selected = jnp.take_along_axis(jax.nn.log_softmax(logits), actions[:, None], axis=-1)[:, 0]
        return -(advantage * selected).sum() / jnp.maximum(mask.sum(), 1)

    expected_w, expected_b = jax.vmap(jax.grad(loss, argnums=(0, 1)), (None, None, 1, 1, 1, 1))(
        w, b, h_clean, a, gae_clean, valid,
    )
    actual_w, actual_b, count = jax.jit(lambda *xs: pace_lite.head_gradients(*xs, bias=use_bias))(
        h, h @ w + b, a, gae, mask,
    )
    np.testing.assert_allclose(expected_w, actual_w, atol=1e-6, rtol=1e-6)
    if use_bias:
        np.testing.assert_allclose(expected_b, actual_b, atol=1e-6, rtol=1e-6)
    else:
        assert actual_b is None
    np.testing.assert_array_equal(count, valid.sum(axis=0))
    assert np.isnan(pace_lite.scores(h, h @ w + b, a.at[:, 2].set(-1), gae)[2])


def test_gradient_cancellation_and_vmap():
    h = jnp.ones((2, 1, 2))
    logits = jnp.zeros((2, 1, 3))
    actions = jnp.zeros((2, 1), dtype=jnp.int32)
    gae = jnp.array([[1.], [-1.]])
    result = jax.jit(jax.vmap(pace_lite.scores))(
        h[None], logits[None], actions[None], gae[None],
    )
    np.testing.assert_array_equal(result, [[0.]])


def test_shared_midranks_and_invalid_slots():
    stored = {"grad_squared": jnp.array([1., 9., 90.]), "maxmc": jnp.array([3., 2., 90.])}
    entries = {"grad_squared": jnp.array([1., jnp.nan]), "maxmc": jnp.array([3., 100.])}
    sampler = {"scores": jnp.array([10., 10., -jnp.inf]), "levels_extra": stored,
               "size": jnp.array(2), "pace_threshold": jnp.array(1.)}
    result, candidates = jax.jit(lambda s, e: pace_lite.synchronize(s, e, pace_adaptive.DEFAULTS, True))(
        sampler, entries,
    )
    assert result["scores"][0] == candidates[0]
    assert candidates[0] > result["scores"][1]
    assert jnp.isneginf(result["scores"][2]) and jnp.isneginf(candidates[1])
    np.testing.assert_allclose(pace_lite.percentiles(jnp.array([1., 1., 2., 3.]), jnp.array([1, 1, 1, 0], bool)),
                               [1/3, 1/3, 5/6, 0])


def test_percentiles_match_reference_across_full_buffer():
    rng = np.random.default_rng(17)
    values = rng.integers(-80, 81, 4032).astype(np.float32)
    usable = rng.random(values.size) > .15
    reference = np.sort(values[usable])
    expected = (np.searchsorted(reference, values, "left") + np.searchsorted(reference, values, "right"))
    expected = np.where(usable, expected / (2 * len(reference)), 0)
    np.testing.assert_allclose(jax.jit(pace_lite.percentiles)(values, usable), expected, atol=1e-7)
