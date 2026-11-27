"""Profile synchronized stages and compare scores on one paired Maze rollout."""

import argparse
from contextlib import closing
import json
from pathlib import Path
from time import perf_counter

from etils import epath
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from scipy.stats import pearsonr, spearmanr

from maze_plr import (ActorCritic, BaseTrainState, compute_gae, evaluate_rnn,
                      sample_trajectories_rnn, update_actor_critic_rnn)
from jaxued import pace_adaptive, pace_lite, pace_score, pace_validation
from jaxued.environments.maze import Level, Maze
from jaxued.level_sampler import LevelSampler
from jaxued.utils import compute_max_returns, max_mc
from jaxued.wrappers import AutoReplayWrapper


def profile(checkpoint, validation, output, repeat):
    config = json.loads((checkpoint.parent.parent.parent / "config.json").read_text(encoding="utf-8"))
    handler = ocp.PyTreeCheckpointHandler()
    path = epath.Path(checkpoint.resolve())
    with closing(ocp.Checkpointer(handler)) as reader:
        # Restoring host arrays also allows an isolated CPU run on a GPU checkpoint.
        restore_args = jax.tree_util.tree_map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray),
                                             handler._get_internal_metadata(path)[0])
        archive = reader.restore(path, args=ocp.args.PyTreeRestore(restore_args=restore_args))
    record = jax.device_put(archive)
    raw_sampler = record["sampler"]
    stages = {}

    def timed(name, fn, *args):
        jax.block_until_ready(args)
        start = perf_counter()
        compiled = jax.jit(fn).lower(*args).compile()
        compile_time = perf_counter() - start
        jax.block_until_ready(compiled(*args))
        seconds = []
        for _ in range(repeat):
            start = perf_counter()
            result = jax.block_until_ready(compiled(*args))
            seconds.append(perf_counter() - start)
        stages[name] = {"compile_s": compile_time, "median_ms": 1000 * float(np.median(seconds))}
        return result

    env = Maze(agent_view_size=config["agent_view_size"], normalize_obs=True)
    replay_env = AutoReplayWrapper(env)
    params = replay_env.default_params
    model = ActorCritic(env.action_space(params).n)
    n, steps = config["num_train_envs"], config["num_steps"]
    optimizer = optax.chain(optax.clip_by_global_norm(config["max_grad_norm"]),
                           optax.adam(lambda step: config["lr"] * (
                               1 - (step // (config["num_minibatches"] * config["epoch_ppo"])) / config["num_updates"]
                           ), eps=1e-5))
    policy = BaseTrainState.create(params=record["params"], apply_fn=model.apply, tx=optimizer)
    policy = policy.replace(opt_state=serialization.from_state_dict(policy.opt_state, record["opt_state"]),
                            step=record["step"])
    levels = Level(**jax.tree_util.tree_map(lambda a: a[:n], raw_sampler["levels"]))
    key, reset = jax.random.split(jax.random.PRNGKey(891))
    reset_obs, reset_state = jax.vmap(replay_env.reset_to_level, (0, 0, None))(jax.random.split(reset, n), levels, params)
    carry = model.initialize_carry((n,))

    def rollout(rng, state, captured):
        return sample_trajectories_rnn(rng, replay_env, params, state, carry, reset_obs, reset_state, n, steps,
                                       record_actor_head=captured)

    plain = timed("rollout", lambda k, p: rollout(k, p, False), key, policy)
    tapped = timed("rollout_with_actor_head", lambda k, p: rollout(k, p, True), key, policy)
    for a, b in zip(jax.tree_util.tree_leaves((plain[0], plain[1][:-1])),
                    jax.tree_util.tree_leaves((tapped[0], tapped[1][:-1]))):
        np.testing.assert_array_equal(a, b)
    obs, actions, rewards, dones, lp, values, info = tapped[1]
    h, logits = info["actor_head"]
    advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], tapped[0][-1], values, rewards, dones)
    mc = timed("maxmc", lambda d, r, v: max_mc(d, v, compute_max_returns(d, r)), dones, rewards, values)
    original = timed("pace_pg", lambda p, o, a, d, adv: pace_score.compute_pace_pg_scores(
        model.apply, p, o, a, d, adv, carry), policy.params, obs, actions, dones, advantages)
    lite = timed("pace_lite", pace_lite.scores, h, logits, actions, advantages)
    options = {**pace_adaptive.DEFAULTS, "pace_floor_weight": .30}
    sampler_api = LevelSampler(config["level_buffer_capacity"], config["replay_prob"], config["staleness_coeff"],
                               config["minimum_fill_ratio"], "rank", {"temperature": config["temperature"]},
                               duplicate_check=config["buffer_duplicate_check"])
    normalized = {**raw_sampler, "levels": Level(**raw_sampler["levels"])}
    upgraded = {**normalized, "pace_threshold": jnp.float32(0), "levels_extra": {
        **normalized["levels_extra"], "maxmc": normalized["scores"],
        "grad_squared": jnp.resize(lite, normalized["scores"].shape),
    }}
    extra = dict(max_return=compute_max_returns(dones, rewards), maxmc=mc, grad_squared=lite)

    def reweight(state, sample):
        state, _, _ = pace_adaptive.prepare(state, sample["grad_squared"], options, pace_lite.HYBRID, rescore=False)
        return pace_lite.synchronize(state, sample, options, True)

    reweighted, priorities = timed("threshold_and_common_percentiles", reweight, upgraded, extra)
    timed("replay_sampling", lambda s, k: sampler_api.sample_replay_levels(s, k, n), reweighted, key)
    timed("buffer_insertion", lambda s, l, p, e: pace_lite.insert_batch(sampler_api, s, l, p, e),
          reweighted, levels, priorities, extra)
    timed("ppo", lambda k, p, b: update_actor_critic_rnn(
        k, p, carry, b, n, steps, config["num_minibatches"], config["epoch_ppo"],
        config["clip_eps"], config["entropy_coeff"], config["critic_coeff"], update_grad=True,
    ), key, policy, (obs, actions, dones, lp, values, targets, advantages))

    def evaluate(state, levels):
        def attempt(rng):
            obs, initial = jax.vmap(env.reset_to_level, (0, 0, None))(
                jax.random.split(rng, levels.agent_dir.size), levels, params,
            )
            _, rewards, lengths = evaluate_rnn(rng, env, params, state,
                model.initialize_carry((levels.agent_dir.size,)), obs, initial, params.max_steps_in_episode)
            return (rewards * (jnp.arange(params.max_steps_in_episode)[:, None] < lengths)).sum(axis=0)
        return jax.lax.map(attempt, jax.random.split(jax.random.PRNGKey(pace_validation.EVAL_SEED), pace_validation.ATTEMPTS))

    timed("validation", evaluate, policy, pace_validation.load(validation)[0])
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
    head = policy.params["params"]["actor1"]
    actual_w, actual_b, _ = pace_lite.head_gradients(h, logits, actions, advantages, jnp.ones_like(dones))

    def loss(kernel, bias, h, actions, a):
        lp = jax.nn.log_softmax(h @ kernel + bias)
        return -(a * jnp.take_along_axis(lp, actions[:, None], axis=1)[:, 0]).mean()

    expected_w, expected_b = jax.vmap(jax.grad(loss, (0, 1)), (None, None, 1, 1, 1))(
        head["kernel"], head["bias"], h, actions, advantages,
    )
    expected = np.concatenate((np.asarray(expected_w).ravel(), np.asarray(expected_b).ravel()))
    actual = np.concatenate((np.asarray(actual_w).ravel(), np.asarray(actual_b).ravel()))
    expected_norm = np.square(expected_w).sum(axis=(1, 2)) + np.square(expected_b).sum(axis=1)
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=2e-4)
    np.testing.assert_allclose(lite, expected_norm, atol=3e-6, rtol=2e-4)
    comparison = {}
    for name, a, b in (("maxmc_vs_lite", mc, lite), ("maxmc_vs_pace", mc, original), ("pace_vs_lite", original, lite)):
        valid = np.isfinite(a) & np.isfinite(b)
        a, b = np.asarray(a)[valid], np.asarray(b)[valid]
        k = min(8, len(a))
        comparison[name] = {"pearson": float(pearsonr(a, b)[0]), "spearman": float(spearmanr(a, b)[0]),
                            "top_k": k, "overlap": len(set(np.argsort(a)[-k:]) & set(np.argsort(b)[-k:]))}
    payload = {"device": str(jax.devices()[0]), "rounds_checkpoint": 5000, "envs": n, "steps": steps,
               "repeat": repeat, "stages": stages, "scores": comparison,
               "memory": jax.devices()[0].memory_stats(),
               "rollout_env_steps_per_s": n * steps * 1000 / stages["rollout"]["median_ms"],
               "gradients": {"max_abs_error": float(np.max(np.abs(actual - expected))),
                             "relative_l2_error": float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)),
                             "cosine": float(np.dot(actual, expected) / (np.linalg.norm(actual) * np.linalg.norm(expected))),
                             "max_squared_norm_error": float(np.max(np.abs(np.asarray(lite) - expected_norm)))}}
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    profile(args.checkpoint, args.validation, args.output, args.repeat)
