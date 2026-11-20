"""Compare first-visit Maze PLR scores on paired, inference-only rollouts."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import struct
from scipy.stats import spearmanr

from examples.maze_plr import ActorCritic, compute_gae, compute_score, sample_trajectories_rnn
from jaxued.environments.maze import Maze, MazeSolved, make_level_generator
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
    args = parser.parse_args()

    config = json.loads((args.checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    batch_size, horizon = config["num_train_envs"], config["num_steps"]
    env = AutoReplayWrapper(Maze(agent_view_size=config["agent_view_size"], normalize_obs=True))
    env_params = env.default_params
    if args.levels <= 0 or args.levels % batch_size or args.attempts < 1:
        parser.error("--levels must be a positive multiple of the batch size and --attempts positive")
    if horizon < env_params.max_steps_in_episode:
        parser.error("checkpoint rollout horizon must cover a full Maze episode")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    model = ActorCritic(env.action_space(env_params).n)
    checkpoint_path = args.checkpoint_dir / "models" / str(args.checkpoint) / "default"
    reader = ocp.StandardCheckpointer()
    try:
        checkpoint = reader.restore(checkpoint_path.resolve())
    finally:
        reader.close()
    policy = Policy(checkpoint["params"], model.apply)

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
        _, _, rewards, dones, _, values, _ = trajectory
        advantages, _ = compute_gae(config["gamma"], config["gae_lambda"], result[-1], values, rewards, dones)
        max_returns = compute_max_returns(dones, rewards)
        maxmc = compute_score({"score_function": "MaxMC"}, dones, values, max_returns, advantages)
        pvl = compute_score({"score_function": "pvl"}, dones, values, max_returns, advantages)
        first_done = dones.argmax(axis=0)
        solved = rewards[first_done, jnp.arange(batch_size)] > 0
        return maxmc, pvl, solved

    print(f"{jax.devices()} | {checkpoint_path} | optimizer step {int(checkpoint['step'])}", flush=True)
    # Draw one score rollout and independent success-estimation rollouts for each batch.
    rollout_key = jax.random.PRNGKey(args.rollout_seed)
    score_batches, evaluation_batches = [], []
    for start in range(0, args.levels, batch_size):
        batch = jax.tree_util.tree_map(lambda x: x[start:start + batch_size], levels)
        keys = jax.random.split(jax.random.fold_in(rollout_key, start), args.attempts + 1)
        score_batches.append(jax.device_get(score_batch(keys[0], batch, policy)))
        evaluation_batches.append(np.stack([np.asarray(score_batch(key, batch, policy)[2]) for key in keys[1:]]))

    maxmc, pvl, score_solved = (np.concatenate(columns) for columns in zip(*score_batches))
    eval_solved = np.concatenate(evaluation_batches, axis=1)
    if not (np.isfinite(maxmc).all() and np.isfinite(pvl).all()):
        raise RuntimeError("non-finite scores on complete rollouts")
    solve_rate = eval_solved.mean(axis=0)
    reachable = min_steps <= env_params.max_steps_in_episode
    if np.any(eval_solved[:, ~reachable]) or np.any(score_solved[~reachable]):
        raise RuntimeError("a solution was recorded outside the shortest-path bound")

    top_k = max(1, args.levels // 10)
    maxmc_top = np.argsort(-maxmc, kind="stable")[:top_k]
    pvl_top = np.argsort(-pvl, kind="stable")[:top_k]
    correlation = float(spearmanr(maxmc, pvl).statistic) if np.ptp(maxmc) and np.ptp(pvl) else None

    def group_stats(indices):
        return {
            "levels": int(len(indices)),
            "reachable": int(reachable[indices].sum()),
            "solves": int(eval_solved[:, indices].sum()),
            "levels_solved_at_least_once": int((solve_rate[indices] > 0).sum()),
            "mean_solve_rate": float(solve_rate[indices].mean()),
        }

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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as file:
        np.savez_compressed(
            file, maxmc=maxmc, pvl=pvl, score_solved=score_solved, eval_solved=eval_solved,
            min_steps=min_steps, wall_map=levels.wall_map, agent_pos=levels.agent_pos,
            agent_dir=levels.agent_dir, goal_pos=levels.goal_pos, summary=json.dumps(summary),
        )
    print(json.dumps(summary, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
