"""Compare first-visit Maze PLR scores on paired, inference-only rollouts."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import struct
from scipy.stats import spearmanr

from examples.maze_plr import ActorCritic, compute_gae, compute_score, sample_trajectories_rnn
from jaxued.environments.maze import Maze, MazeSolved, make_level_generator
from jaxued.pace_score import compute_pace_pg_scores
from jaxued.utils import compute_max_returns
from jaxued.wrappers import AutoReplayWrapper


@struct.dataclass
class Policy:
    params: dict
    apply_fn: object = struct.field(pytree_node=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--levels", default=256, type=int)
    parser.add_argument("--attempts", default=8, type=int)
    parser.add_argument("--level-seed", default=73129, type=int)
    parser.add_argument("--rollout-seed", default=40961, type=int)
    parser.add_argument("--pace", action="store_true", help="also score and profile PACE-PG on the same rollouts")
    parser.add_argument("--pace-chunk-size", default=1, type=int)
    parser.add_argument("--progress-checkpoint", type=int, help="later checkpoint index in the same run")
    args = parser.parse_args()

    config = json.loads((args.checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    batch_size, horizon = config["num_train_envs"], config["num_steps"]
    env = AutoReplayWrapper(Maze(agent_view_size=config["agent_view_size"], normalize_obs=True))
    env_params = env.default_params
    if args.levels <= 0 or args.levels % batch_size or args.attempts < 1:
        parser.error("--levels must be a positive multiple of the batch size and --attempts positive")
    if args.pace and (args.pace_chunk_size < 1 or config["num_minibatches"] != 1):
        parser.error("PACE requires a positive chunk size and num_minibatches=1")
    if args.progress_checkpoint is not None and args.progress_checkpoint <= args.checkpoint:
        parser.error("--progress-checkpoint must be later than --checkpoint")
    if horizon < env_params.max_steps_in_episode:
        parser.error("checkpoint rollout horizon must cover a full Maze episode")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    model = ActorCritic(env.action_space(env_params).n)
    checkpoint_path = args.checkpoint_dir / "models" / str(args.checkpoint) / "default"
    reader = ocp.StandardCheckpointer()
    try:
        checkpoint = reader.restore(checkpoint_path.resolve())
        later = args.checkpoint_dir / "models" / str(args.progress_checkpoint) / "default"
        progress = reader.restore(later.resolve()) if args.progress_checkpoint is not None else None
    finally:
        reader.close()
    policy = Policy(checkpoint["params"], model.apply)
    progress_policy = None if progress is None else Policy(progress["params"], model.apply)

    generator = make_level_generator(13, 13, config["n_walls"])
    level_keys = jax.random.split(jax.random.PRNGKey(args.level_seed), args.levels)
    levels = jax.jit(jax.vmap(generator))(level_keys)
    solved_env = MazeSolved(max_height=13, max_width=13)

    def shortest_path(level):
        state = solved_env.init_state_from_level(level)
        return solved_env.min_steps_to_goal(state)

    min_steps = np.asarray(jax.jit(jax.vmap(shortest_path))(levels))

    @jax.jit
    def score_batch(rng, levels, policy):
        reset_key, rollout_key = jax.random.split(rng)
        obs, state = jax.vmap(env.reset_to_level, (0, 0, None))(
            jax.random.split(reset_key, batch_size), levels, env_params
        )
        initial_carry = model.initialize_carry((batch_size,))
        result, trajectory = sample_trajectories_rnn(
            rollout_key, env, env_params, policy, initial_carry, obs, state, batch_size, horizon
        )
        obs, actions, rewards, dones, _, values, _ = trajectory
        advantages, _ = compute_gae(config["gamma"], config["gae_lambda"], result[-1], values, rewards, dones)
        max_returns = compute_max_returns(dones, rewards)
        maxmc = compute_score({"score_function": "MaxMC"}, dones, values, max_returns, advantages)
        pvl = compute_score({"score_function": "pvl"}, dones, values, max_returns, advantages)
        first_done = dones.argmax(axis=0)
        solved = rewards[first_done, jnp.arange(batch_size)] > 0
        if args.pace:
            return maxmc, pvl, solved, (obs, actions, dones, advantages, initial_carry)
        return maxmc, pvl, solved

    @jax.jit
    def pace_batch(weights, inputs):
        return compute_pace_pg_scores(model.apply, weights, *inputs, chunk_size=args.pace_chunk_size)

    compiled = None
    compile_seconds = None
    score_times = []

    def evaluate_batch(key, batch):
        nonlocal compiled, compile_seconds
        maxmc, pvl, solved, *extra = score_batch(key, batch, policy)
        if not args.pace:
            return jax.device_get((maxmc, pvl, solved))
        inputs, = extra
        jax.block_until_ready(inputs)
        if compiled is None:
            start = perf_counter()
            compiled = pace_batch.lower(policy.params, inputs).compile()
            compile_seconds = perf_counter() - start
        start = perf_counter()
        pace = compiled(policy.params, inputs).block_until_ready()
        score_times.append(perf_counter() - start)
        return jax.device_get((maxmc, pvl, solved, pace))

    print(f"{jax.devices()} | {checkpoint_path} | optimizer step {int(checkpoint['step'])}", flush=True)
    # Draw one score rollout and independent success-estimation rollouts for each batch.
    rollout_key = jax.random.PRNGKey(args.rollout_seed)
    score_batches, evaluation_batches = [], []
    repeated_pace, progress_attempts = [], []
    for start in range(0, args.levels, batch_size):
        batch = jax.tree_util.tree_map(lambda x: x[start:start + batch_size], levels)
        keys = jax.random.split(jax.random.fold_in(rollout_key, start), args.attempts + 1)
        score_batches.append(evaluate_batch(keys[0], batch))
        repeats = [evaluate_batch(key, batch) for key in keys[1:]]
        evaluation_batches.append(np.stack([result[2] for result in repeats]))
        if args.pace:
            repeated_pace.append(np.stack([result[3] for result in repeats]))
        if progress_policy is not None:
            progress_attempts.append(np.stack([np.asarray(score_batch(key, batch, progress_policy)[2]) for key in keys[1:]]))

    maxmc, pvl, score_solved, *additional = (np.concatenate(columns) for columns in zip(*score_batches))
    eval_solved = np.concatenate(evaluation_batches, axis=1)
    later_solved = np.concatenate(progress_attempts, axis=1) if progress_attempts else None
    if not (np.isfinite(maxmc).all() and np.isfinite(pvl).all()):
        raise RuntimeError("non-finite scores on complete rollouts")
    solve_rate = eval_solved.mean(axis=0)
    reachable = min_steps <= env_params.max_steps_in_episode
    if np.any(eval_solved[:, ~reachable]) or np.any(score_solved[~reachable]) or (
        later_solved is not None and np.any(later_solved[:, ~reachable])
    ):
        raise RuntimeError("a solution was recorded outside the shortest-path bound")

    top_k = max(1, args.levels // 10)
    maxmc_top = np.argsort(-maxmc, kind="stable")[:top_k]
    pvl_top = np.argsort(-pvl, kind="stable")[:top_k]
    correlation = float(spearmanr(maxmc, pvl).statistic) if np.ptp(maxmc) and np.ptp(pvl) else None

    def group_stats(indices):
        stats = {
            "levels": int(len(indices)),
            "reachable": int(reachable[indices].sum()),
            "solves": int(eval_solved[:, indices].sum()),
            "levels_solved_at_least_once": int((solve_rate[indices] > 0).sum()),
            "mean_solve_rate": float(solve_rate[indices].mean()),
        }
        if later_solved is not None:
            stats["later_mean_solve_rate"] = float(later_solved[:, indices].mean())
            stats["mean_solve_rate_change"] = stats["later_mean_solve_rate"] - stats["mean_solve_rate"]
        return stats

    summary = {
        "checkpoint": str(checkpoint_path),
        "optimizer_step": int(checkpoint["step"]),
        "level_seed": args.level_seed,
        "rollout_seed": args.rollout_seed,
        "rollout_steps": horizon,
        "eval_attempts_per_level": args.attempts,
        "score_first_episode_solves": int(score_solved.sum()),
        "spearman": correlation,
        "top_k": top_k,
        "top_overlap": int(np.intersect1d(maxmc_top, pvl_top).size),
        "maxmc_cutoff_ties": int((maxmc == maxmc[maxmc_top[-1]]).sum()),
        "pvl_cutoff_ties": int((pvl == pvl[pvl_top[-1]]).sum()),
        "all": group_stats(np.arange(args.levels)),
        "maxmc_top": group_stats(maxmc_top),
        "pvl_top": group_stats(pvl_top),
    }
    arrays = {}
    if later_solved is not None:
        summary["progress_checkpoint"] = str(later)
        summary["progress_optimizer_step"] = int(progress["step"])
        arrays["later_solved"] = later_solved
    if args.pace:
        pace, = additional
        repeat_scores = np.concatenate(repeated_pace, axis=1)
        if not (np.isfinite(pace).all() and np.isfinite(repeat_scores).all()):
            raise RuntimeError("non-finite PACE scores")

        def rank_correlation(first, second):
            if len(first) < 2 or not np.ptp(first) or not np.ptp(second):
                return None
            return float(spearmanr(first, second).statistic)

        pace_top = np.argsort(-pace, kind="stable")[:top_k]
        memory = compiled.memory_analysis()
        summary["pace"] = {
            "devices": [str(device) for device in jax.devices()],
            "jax_version": jax.__version__,
            "chunk_size": args.pace_chunk_size,
            "distinct_scores": int(np.unique(pace).size),
            "zero_fraction": float((pace <= 1e-12).mean()),
            "spearman_maxmc": rank_correlation(pace, maxmc),
            "spearman_pvl": rank_correlation(pace, pvl),
            "spearman_independent_solve_rate": rank_correlation(pace, solve_rate),
            "spearman_same_level_repeat": [rank_correlation(pace, repeat) for repeat in repeat_scores],
            "median_per_level_score_std": float(np.median(np.vstack([pace, repeat_scores]).std(axis=0))),
            "top_overlap_maxmc": int(np.intersect1d(pace_top, maxmc_top).size),
            "top_overlap_pvl": int(np.intersect1d(pace_top, pvl_top).size),
            "top": group_stats(pace_top),
            "compile_seconds": compile_seconds,
            "first_call_seconds": score_times[0],
            "median_warm_batch_seconds": float(np.median(score_times[1:])),
            "xla_memory_bytes": None if memory is None else {
                "arguments": memory.argument_size_in_bytes,
                "outputs": memory.output_size_in_bytes,
                "temporaries": memory.temp_size_in_bytes,
                "input_output_alias": memory.alias_size_in_bytes,
            },
        }
        if later_solved is not None:
            change = later_solved.mean(axis=0) - solve_rate
            summary["pace"]["spearman_progress_reachable"] = rank_correlation(pace[reachable], change[reachable])
        arrays.update(pace=pace, pace_repeats=repeat_scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as file:
        np.savez_compressed(
            file, maxmc=maxmc, pvl=pvl, score_solved=score_solved, eval_solved=eval_solved,
            min_steps=min_steps, wall_map=levels.wall_map, agent_pos=levels.agent_pos,
            agent_dir=levels.agent_dir, goal_pos=levels.goal_pos, summary=json.dumps(summary),
            **arrays,
        )
    print(json.dumps(summary, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
