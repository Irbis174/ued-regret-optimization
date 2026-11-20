"""Run a matched PLR score pilot and evaluate every checkpoint on procedural levels."""

import argparse
import json
import runpy
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import wandb

from examples.maze_plr import ActorCritic, evaluate_rnn
from jaxued.environments.maze import Maze, MazeSolved, make_level_generator
from jaxued.environments.maze.level import Level, prefabs


SCORES = ("MaxMC", "pvl", "recent_learnability")
WALL_DRAWS = (25, 40, 55)
LEVELS_PER_GROUP = 16
LEVEL_SEED = 84061
EVAL_SEED = 66317
EVAL_ATTEMPTS = 16


def validation_levels():
    solver = MazeSolved(max_height=13, max_width=13)

    def distance(level):
        return solver.min_steps_to_goal(solver.init_state_from_level(level))

    distances = jax.jit(jax.vmap(distance))
    selected, shortest_paths = [], []
    for wall_draws in WALL_DRAWS:
        key = jax.random.fold_in(jax.random.PRNGKey(LEVEL_SEED), wall_draws)
        candidates = jax.jit(jax.vmap(make_level_generator(13, 13, wall_draws)))(jax.random.split(key, 64))
        steps = np.asarray(distances(candidates))
        indices = np.flatnonzero(steps <= solver.default_params.max_steps_in_episode)[:LEVELS_PER_GROUP]
        if len(indices) != LEVELS_PER_GROUP:
            raise RuntimeError(f"not enough solvable validation levels with {wall_draws} wall draws")
        selected.append(jax.tree_util.tree_map(lambda x: x[indices], candidates))
        shortest_paths.extend(steps[indices].astype(int))
    levels = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs), *selected)
    return levels, np.asarray(shortest_paths)


def make_evaluator():
    maze = Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)
    params = maze.default_params
    model = ActorCritic(maze.action_space(params).n)
    horizon = params.max_steps_in_episode

    @jax.jit
    def evaluate(policy_params, levels, keys):
        policy = SimpleNamespace(params=policy_params, apply_fn=model.apply)
        batch_size = levels.agent_dir.shape[0]

        def attempt(key):
            reset_key, rollout_key = jax.random.split(key)
            obs, state = jax.vmap(maze.reset_to_level, (0, 0, None))(
                jax.random.split(reset_key, batch_size), levels, params
            )
            _, rewards, lengths = evaluate_rnn(
                rollout_key, maze, params, policy, model.initialize_carry((batch_size,)), obs, state, horizon
            )
            return (rewards * (jnp.arange(horizon)[:, None] < lengths)).sum(axis=0), lengths

        return jax.vmap(attempt)(keys)

    return evaluate


def train(score, run_name, args, monitoring_names):
    argv = sys.argv
    sys.argv = [
        "examples/maze_plr.py", "--run_name", run_name, "--seed", str(args.seed),
        "--num_updates", str(args.rounds), "--eval_freq", str(args.eval_freq),
        "--checkpoint_save_interval", "1", "--max_number_of_checkpoints", str(args.rounds // args.eval_freq),
        "--score_function", score, "--no-use_accel", "--no-exploratory_grad_updates",
        "--eval_num_attempts", "4", "--eval_levels", *monitoring_names,
    ]
    start = time.perf_counter()
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        sys.argv = argv
        wandb.finish()
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=250)
    parser.add_argument("--eval-freq", type=int, default=125)
    args = parser.parse_args()
    if args.rounds <= 0 or args.eval_freq <= 0 or args.rounds % args.eval_freq:
        parser.error("--rounds must be a positive multiple of --eval-freq")
    if not args.run_id or not all(c.isalnum() or c in "-_" for c in args.run_id):
        parser.error("--run-id must contain letters, numbers, '-' or '_'")
    run_names = {score: f"{args.run_id}-{score}" for score in SCORES}
    if any((Path("checkpoints") / name / str(args.seed)).exists() for name in run_names.values()):
        parser.error("one of the training run names already exists")
    output_dir = Path("results/score-pilot") / args.run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Pilot {args.run_id} on {jax.devices()}", flush=True)

    levels, shortest_paths = validation_levels()
    wall_draws = np.repeat(WALL_DRAWS, LEVELS_PER_GROUP)
    monitoring_names = []
    for index in (0, 1, 16, 17, 32, 33):
        level = jax.tree_util.tree_map(lambda x: x[index], levels)
        name = f"{args.run_id}-validation-{index}"
        prefabs[name] = level.to_str()
        monitoring_names.append(name)
    np.savez_compressed(
        output_dir / "validation.npz", wall_map=levels.wall_map, agent_pos=levels.agent_pos,
        agent_dir=levels.agent_dir, goal_pos=levels.goal_pos, wall_draws=wall_draws, shortest_paths=shortest_paths,
    )
    keys = jax.random.split(jax.random.PRNGKey(EVAL_SEED), EVAL_ATTEMPTS)
    evaluate = make_evaluator()
    report = {
        "training_seed": args.seed, "rounds": args.rounds, "eval_freq": args.eval_freq,
        "level_seed": LEVEL_SEED, "eval_seed": EVAL_SEED, "attempts_per_level": EVAL_ATTEMPTS,
        "monitoring_levels": monitoring_names, "scores": {},
    }

    reader = ocp.StandardCheckpointer()
    try:
        for score, run_name in run_names.items():
            print(f"Training {score}: {run_name}", flush=True)
            seconds = train(score, run_name, args, monitoring_names)
            rows, all_returns, all_lengths = [], [], []
            checkpoint_dir = Path("checkpoints") / run_name / str(args.seed)
            config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
            env_steps_per_round = config["num_train_envs"] * config["num_steps"]
            for index in range(args.rounds // args.eval_freq):
                state = reader.restore((checkpoint_dir / "models" / str(index) / "default").resolve())
                new = int(state["num_dr_updates"])
                replay = int(state["num_replay_updates"])
                rounds = (index + 1) * args.eval_freq
                if new + replay != rounds or int(state["num_mutation_updates"]) != 0:
                    raise RuntimeError(f"unexpected curriculum counters for {score} at round {rounds}")
                rewards, lengths = map(np.asarray, evaluate(state["params"], levels, keys))
                solves = rewards > 0
                rows.append({
                    "rounds": rounds, "env_steps": rounds * env_steps_per_round,
                    "new": new, "replay": replay, "optimizer_steps": int(state["step"]),
                    "solve_rate": float(solves.mean()), "solves": int(solves.sum()),
                    "mean_return": float(rewards.mean()),
                    "solve_rate_by_wall_draws": {str(n): float(solves[:, wall_draws == n].mean()) for n in WALL_DRAWS},
                })
                all_returns.append(rewards)
                all_lengths.append(lengths)
            np.savez_compressed(output_dir / f"{score}.npz", returns=all_returns, episode_lengths=all_lengths)
            report["scores"][score] = {"run_name": run_name, "wall_seconds": seconds, "checkpoints": rows}
            (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"{score}: {rows[-1]['solves']}/{EVAL_ATTEMPTS * len(wall_draws)} solved at {args.rounds} rounds", flush=True)
    finally:
        reader.close()


if __name__ == "__main__":
    main()
