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
from examples.maze_qd_diagnostics import save_archive_view
from jaxued import hybrid_teacher, map_elites
from jaxued.environments.maze import Maze, MazeSolved, make_level_generator
from jaxued.environments.maze.level import Level, prefabs
from jaxued.level_sampler import LevelSampler
from jaxued.teacher_experiments import frontier_mask


SCORES = ("MaxMC", "pvl", "recent_learnability")
CAPPED = ("--replay_max_copies", "8", "--replay_preserve_frontier_mass")
EXPERIMENTS = {
    "A": ("recent_learnability", ()),
    "B": ("recent_learnability", CAPPED),
    "C": ("history_aware_regret", ()),
    "D": ("history_aware_regret", CAPPED),
}
QD_EXPERIMENTS = {
    arm: ("recent_learnability", ("--qd_mode", mode))
    for arm, mode in (("fixed", "off"), ("shadow", "shadow"),
                      ("random_archive", "random_archive"), ("map_elites", "map_elites"))
}
WALL_DRAWS = (25, 40, 55)
LEVELS_PER_GROUP = 16
LEVEL_SEED = 84061
EVAL_SEED = 66317
EVAL_ATTEMPTS = 16


def validation_levels(level_seed=LEVEL_SEED):
    solver = MazeSolved(max_height=13, max_width=13)

    def distance(level):
        return solver.min_steps_to_goal(solver.init_state_from_level(level))

    distances = jax.jit(jax.vmap(distance))
    selected, shortest_paths = [], []
    for wall_draws in WALL_DRAWS:
        key = jax.random.fold_in(jax.random.PRNGKey(level_seed), wall_draws)
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


def train(score, run_name, args, monitoring_names, teacher_flags=(), *, accel=False):
    argv = sys.argv
    sys.argv = [
        "examples/maze_plr.py", "--run_name", run_name, "--seed", str(args.seed),
        "--num_updates", str(args.rounds), "--eval_freq", str(args.eval_freq),
        "--checkpoint_save_interval", "1", "--max_number_of_checkpoints", str(args.rounds // args.eval_freq),
        "--score_function", score, "--use_accel" if accel else "--no-use_accel",
        "--no-exploratory_grad_updates",
        *teacher_flags,
        "--eval_num_attempts", "4", "--eval_levels", *monitoring_names,
    ]
    start = time.perf_counter()
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
        wandb_dir = str(Path(wandb.run.dir).parent) if wandb.run is not None else None
    finally:
        sys.argv = argv
        wandb.finish()
    return time.perf_counter() - start, wandb_dir


def comparable_config(config):
    """Compare training settings, ignoring teacher switches and synthetic eval names."""
    excluded = {
        "run_name", "group_name", "score_function", "replay_max_copies",
        "replay_preserve_frontier_mass", "har_history_weight", "eval_levels",
        "mutation_selection", "adaptive_mutation", "bandit_strategy", "bandit_epsilon",
        "bandit_ema_alpha", "bandit_reward", "mutation_arms", "stop_after_updates",
        "resume_checkpoint_directory", "resume_checkpoint_step", "initialize_bandit_on_resume",
        "allow_legacy_resume", "qd_mode", "initialize_qd_on_resume", "debug_teacher_parity",
        "fork_hybrid_on_resume", "hybrid_mode",
    }
    defaults = map_elites.DEFAULT_CONFIG | hybrid_teacher.DEFAULTS
    normalized = {key: value for key, value in (defaults | config).items() if key not in excluded}
    normalized["eval_levels"] = [name.rsplit("-validation-", 1)[-1] for name in config["eval_levels"]]
    return normalized


def buffer_snapshot(state, config, *, weight_sampler=None):
    sampler = state["sampler"]
    size = int(sampler["size"])
    api = LevelSampler(
        config["level_buffer_capacity"], staleness_coeff=config["staleness_coeff"],
        prioritization=config["prioritization"],
        prioritization_params={"temperature": config["temperature"], "k": config["topk_k"]},
    )
    scores = np.asarray(sampler["scores"])[:size]
    weights = np.asarray(api.level_weights(sampler if weight_sampler is None else weight_sampler))
    if size and (not np.isfinite(scores).all() or not np.isfinite(weights).all()):
        raise FloatingPointError("nonfinite active priority or replay weight")
    if size:
        np.testing.assert_allclose(weights.sum(), 1, atol=1e-6)
        if np.any(weights < 0) or np.any(weights[size:] != 0):
            raise ValueError("invalid replay distribution")
    frontier = np.asarray(frontier_mask(sampler))
    last = state["replay_last_level_batch"]
    batch_size = config["num_train_envs"]
    counts = np.unique(np.column_stack([
        np.asarray(last[key]).reshape(batch_size, -1)
        for key in ("wall_map", "agent_pos", "agent_dir", "goal_pos")
    ]), axis=0, return_counts=True)[1] if int(state["num_replay_updates"]) else np.array([])
    return {
        "active": size,
        "frontier_fraction": float(frontier[:size].mean()) if size else 0.0,
        "frontier_weight": float(weights[frontier].sum()),
        "weight_ess": float(1 / np.square(weights).sum()) if size else 0.0,
        "last_batch_unique": len(counts),
        "last_batch_max_copies": int(counts.max()) if len(counts) else 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=250)
    parser.add_argument("--eval-freq", type=int, default=125)
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--teacher-experiments", action="store_true", help="Run HAR/replay diversity ablations")
    variants.add_argument("--mutation-experiments", action="store_true", help="Run fixed/uniform/bandit ACCEL")
    variants.add_argument("--qd-experiments", action="store_true", help="Run fixed/shadow/random/MAP-Elites ACCEL")
    parser.add_argument("--mutation-scores", nargs="+", choices=("MaxMC", "recent_learnability"),
                        default=("MaxMC", "recent_learnability"))
    parser.add_argument("--reuse-baseline", type=Path, help="Existing score-pilot result directory for the control arm")
    parser.add_argument("--experiment-arms", nargs="+", choices=("B", "C", "D"), default=("B", "C", "D"))
    args = parser.parse_args()
    if args.rounds <= 0 or args.eval_freq <= 0 or args.rounds % args.eval_freq:
        parser.error("--rounds must be a positive multiple of --eval-freq")
    if not args.run_id or not all(c.isalnum() or c in "-_" for c in args.run_id):
        parser.error("--run-id must contain letters, numbers, '-' or '_'")
    if args.reuse_baseline and not (args.teacher_experiments or args.qd_experiments):
        parser.error("--reuse-baseline requires --teacher-experiments or --qd-experiments")
    accel = args.mutation_experiments or args.qd_experiments
    baseline_arm = "fixed" if args.qd_experiments else "A"
    if args.teacher_experiments:
        if len(set(args.experiment_arms)) != len(args.experiment_arms):
            parser.error("duplicate experimental arms")
        specs = {key: EXPERIMENTS[key] for key in ("A", *args.experiment_arms)}
    elif args.mutation_experiments:
        specs = {
            f"{score}-{mode}": (score, ("--mutation_selection", mode))
            for score in args.mutation_scores
            for mode in ("fixed", "uniform", "bandit")
        }
    elif args.qd_experiments:
        specs = QD_EXPERIMENTS
    else:
        specs = {score: (score, ()) for score in SCORES}
    run_names = {key: f"{args.run_id}-{key}" for key in specs if key != baseline_arm or not args.reuse_baseline}
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
    validation = dict(wall_map=levels.wall_map, agent_pos=levels.agent_pos, agent_dir=levels.agent_dir,
                      goal_pos=levels.goal_pos, wall_draws=wall_draws, shortest_paths=shortest_paths)
    np.savez_compressed(output_dir / "validation.npz", **validation)
    keys = jax.random.split(jax.random.PRNGKey(EVAL_SEED), EVAL_ATTEMPTS)
    evaluate = make_evaluator()
    report = {
        "training_seed": args.seed, "rounds": args.rounds, "eval_freq": args.eval_freq,
        "level_seed": LEVEL_SEED, "eval_seed": EVAL_SEED, "attempts_per_level": EVAL_ATTEMPTS,
        "monitoring_levels": monitoring_names, "scores": {},
    }
    reference = None
    if args.reuse_baseline:
        archive = json.loads((args.reuse_baseline / "summary.json").read_text(encoding="utf-8"))
        for key in ("training_seed", "rounds", "eval_freq", "level_seed", "eval_seed", "attempts_per_level"):
            if report[key] != archive[key]:
                raise ValueError(f"baseline protocol differs: {key}")
        with np.load(args.reuse_baseline / "validation.npz") as data:
            for key, array in validation.items():
                np.testing.assert_array_equal(array, data[key], err_msg=f"baseline validation differs: {key}")
        reference_key = "recent_learnability-fixed" if args.qd_experiments else "recent_learnability"
        reference = archive["scores"][reference_key]
        run_names[baseline_arm] = reference["run_name"]

    reader = ocp.StandardCheckpointer()
    expected_config = None
    try:
        for arm, (score, teacher_flags) in specs.items():
            run_name = run_names[arm]
            reused = arm == baseline_arm and reference is not None
            if reused:
                print(f"Reusing {run_name}", flush=True)
                seconds, wandb_dir = reference["wall_seconds"], reference.get("wandb_dir")
            else:
                print(f"Training {arm}: {run_name}", flush=True)
                seconds, wandb_dir = train(
                    score, run_name, args, monitoring_names, teacher_flags, accel=accel
                )
            rows, all_returns, all_lengths = [], [], []
            checkpoint_dir = Path("checkpoints") / run_name / str(args.seed)
            config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
            capped = "--replay_max_copies" in teacher_flags
            expected_selection = teacher_flags[-1] if args.mutation_experiments else "fixed"
            expected_qd = teacher_flags[-1] if args.qd_experiments else "off"
            if (config["score_function"] != score
                    or config["use_accel"] != accel
                    or config.get("mutation_selection", "fixed") != expected_selection
                    or config.get("qd_mode", "off") != expected_qd
                    or config.get("replay_max_copies", 0) != (8 if capped else 0)
                    or config.get("replay_preserve_frontier_mass", False) != capped):
                raise ValueError(f"unexpected teacher configuration for {arm}")
            common = comparable_config(config)
            if expected_config is None:
                expected_config = common
            elif common != expected_config:
                raise ValueError(f"training configuration differs for {arm}")
            metrics_file = checkpoint_dir / "teacher_metrics.jsonl"
            diagnostics = {}
            if metrics_file.exists():
                for line in metrics_file.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    diagnostics[int(row["num_updates"])] = row
            env_steps_per_round = config["num_train_envs"] * config["num_steps"]
            for index in range(args.rounds // args.eval_freq):
                state = reader.restore((checkpoint_dir / "models" / str(index) / "default").resolve())
                new = int(state["num_dr_updates"])
                replay = int(state["num_replay_updates"])
                mutation = int(state["num_mutation_updates"])
                rounds = (index + 1) * args.eval_freq
                if new + replay + mutation != rounds or (not accel and mutation):
                    raise RuntimeError(f"unexpected curriculum counters for {arm} at round {rounds}")
                if int(state["step"]) != replay * config["epoch_ppo"] * config["num_minibatches"]:
                    raise RuntimeError(f"unexpected optimizer steps for {arm} at round {rounds}")
                rewards, lengths = map(np.asarray, evaluate(state["params"], levels, keys))
                if not np.isfinite(rewards).all():
                    raise FloatingPointError(f"nonfinite validation return for {arm}")
                solves = rewards > 0
                rows.append({
                    "rounds": rounds, "env_steps": rounds * env_steps_per_round,
                    "new": new, "replay": replay, "mutation": mutation, "optimizer_steps": int(state["step"]),
                    "solve_rate": float(solves.mean()), "solves": int(solves.sum()),
                    "mean_return": float(rewards.mean()),
                    "solve_rate_by_wall_draws": {str(n): float(solves[:, wall_draws == n].mean()) for n in WALL_DRAWS},
                    "buffer": buffer_snapshot(state, config),
                    "teacher": diagnostics.get(rounds),
                })
                all_returns.append(rewards)
                all_lengths.append(lengths)
            np.savez_compressed(output_dir / f"{arm}.npz", returns=all_returns, episode_lengths=all_lengths)
            final_archive = state["sampler"].get("qd_archive")
            if final_archive is not None:
                shape = config["qd_bins"]
                grid = {key: np.asarray(value).reshape(shape) for key, value in final_archive.items() if key != "levels"}
                archive_levels = {key: np.asarray(value) for key, value in final_archive["levels"].items()}
                elite_levels = {f"level_{key}": value[grid["occupied"].ravel()]
                                for key, value in archive_levels.items()}
                np.savez_compressed(output_dir / f"{arm}-archive.npz", **grid, **elite_levels)
                save_archive_view(output_dir, arm, grid["occupied"], grid["quality"], archive_levels)
            report["scores"][arm] = {
                "run_name": run_name, "score_function": score, "wall_seconds": seconds, "checkpoints": rows,
                "wandb_dir": wandb_dir, "reused_checkpoint": reused,
            }
            (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"{arm}: {rows[-1]['solves']}/{EVAL_ATTEMPTS * len(wall_draws)} solved at {args.rounds} rounds", flush=True)
    finally:
        reader.close()


if __name__ == "__main__":
    main()
