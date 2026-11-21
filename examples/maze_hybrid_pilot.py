"""Fork one dual-score Recent prefix into three matched PLR continuation arms."""

import argparse
import json
import os
from pathlib import Path
import shutil

import jax
import numpy as np
import orbax.checkpoint as ocp

from examples.maze_score_pilot import (
    EVAL_ATTEMPTS, EVAL_SEED, LEVEL_SEED, LEVELS_PER_GROUP, WALL_DRAWS, buffer_snapshot,
    make_evaluator, train, validation_levels,
)
from jaxued import hybrid_teacher as hybrid, mutation_checkpoint, teacher_parity
from jaxued.environments.maze.level import prefabs
from jaxued.level_sampler import LevelSampler


SECONDARY_LEVEL_SEED = 110761
SECONDARY_EVAL_SEED = 112997
MONITOR_INDICES = (0, 1, 16, 17, 32, 33)
HISTORICAL = Path("results/score-pilot/plr-1000-s0-20261121-024350")


def read_config(path):
    return json.loads((path / "config.json").read_text(encoding="utf-8"))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rounds_completed(state):
    return sum(int(state[f"num_{phase}_updates"]) for phase in ("dr", "replay", "mutation"))


def read_models(reader, root, rounds):
    remaining = set(rounds)
    result = {}
    for path in sorted((root / "models").glob("*/default"), reverse=True):
        if not path.parent.name.isdecimal() or not path.is_dir():
            continue
        state = reader.restore(path.resolve())
        completed = rounds_completed(state)
        if completed in remaining:
            result[completed] = path, state
            remaining.remove(completed)
            if not remaining:
                break
    if remaining:
        raise ValueError(f"{root} has no checkpoints for rounds {sorted(remaining)}")
    return result


def valid_prefix(config, args):
    expected = {
        "dual_score_tracking": True, "hybrid_mode": "recent_control", "num_updates": args.rounds,
        "hybrid_start_round": args.resume_round, "hybrid_end_round": args.transition_end,
        "num_train_envs": 32, "num_steps": 256,
        "use_accel": False, "seed": args.seed, "eval_freq": args.eval_freq,
    }
    differences = sorted(key for key, value in expected.items() if config.get(key) != value)
    if config.get("score_function") not in ("recent_learnability", "hybrid"):
        differences.append("score_function")
    if differences:
        raise ValueError(f"incompatible common prefix: {', '.join(differences)}")


def validation_sets(directory, smoke):
    if smoke:
        name = "HybridSmoke"
        rows = prefabs["TrivialMaze"].splitlines()
        prefabs[name] = "\n".join([row.ljust(13, "#") for row in rows] + ["#" * 13] * (13 - len(rows)))
        return {}, [name]
    datasets = {}
    primary = None
    for name, level_seed, attempt_seed in (
        ("primary", LEVEL_SEED, EVAL_SEED), ("secondary", SECONDARY_LEVEL_SEED, SECONDARY_EVAL_SEED),
    ):
        levels, shortest_paths = validation_levels(level_seed)
        dataset = dict(wall_map=levels.wall_map, agent_pos=levels.agent_pos, agent_dir=levels.agent_dir,
                       goal_pos=levels.goal_pos, wall_draws=np.repeat(WALL_DRAWS, LEVELS_PER_GROUP),
                       shortest_paths=shortest_paths)
        if name == "primary":
            primary = levels
            if (HISTORICAL / "validation.npz").is_file():
                with np.load(HISTORICAL / "validation.npz") as old:
                    for key, values in dataset.items():
                        np.testing.assert_array_equal(values, old[key], err_msg=key)
        np.savez_compressed(directory / f"validation-{name}.npz", **dataset)
        keys = jax.random.split(jax.random.PRNGKey(attempt_seed), EVAL_ATTEMPTS)
        datasets[name] = (levels, keys)
    names = []
    for index in MONITOR_INDICES:
        name = f"hybrid-validation-{index}"
        prefabs[name] = jax.tree_util.tree_map(lambda x: x[index], primary).to_str()
        names.append(name)
    return datasets, names


def optimizer_and_budget(state, config, round_index):
    replay = int(state["num_replay_updates"])
    steps = int(state["step"])
    assert rounds_completed(state) == round_index and int(state["num_mutation_updates"]) == 0
    updates_per_replay = config["epoch_ppo"] * config["num_minibatches"]
    assert steps == replay * updates_per_replay
    for path, leaf in jax.tree_util.tree_flatten_with_path(state["opt_state"])[0]:
        if jax.tree_util.keystr(path).endswith("['count']"):
            assert int(leaf) == steps
    assert all(np.isfinite(leaf).all() for leaf in jax.tree_util.tree_leaves(state["params"]))
    return {
        "round": round_index, "new": int(state["num_dr_updates"]), "replay": replay,
        "optimizer_steps": steps, "environment_steps": round_index * config["num_train_envs"] * config["num_steps"],
        "next_learning_rate": config["lr"] * (1 - (steps // updates_per_replay) / config["num_updates"]),
    }


def describe(sampler, config, weight, out_file):
    size = int(sampler["size"])
    active = np.arange(config["level_buffer_capacity"]) < size
    extra = sampler["levels_extra"]
    assert np.asarray(extra["recent_initialized"])[active].all()
    assert not np.asarray(extra["recent_initialized"])[~active].any()
    assert not np.asarray(extra["maxmc_initialized"])[~active].any()
    recent = np.asarray(extra["recent_score"])[active]
    mc = np.asarray(extra["maxmc_score"])[active]
    observed = np.asarray(extra["maxmc_initialized"])[active]
    mixed = np.asarray(sampler["scores"])[active]
    if not np.isfinite(recent).all() or not np.isfinite(mc).all() or not np.isfinite(mixed).all():
        raise FloatingPointError("nonfinite dual scores")
    np.testing.assert_allclose(sampler["scores"], hybrid.reweight(sampler, weight)["scores"], atol=1e-7)
    view = hybrid.reweight(sampler, weight, raw_rank=config["prioritization"] == "rank")
    api = LevelSampler(
        config["level_buffer_capacity"], staleness_coeff=config["staleness_coeff"],
        prioritization=config["prioritization"],
        prioritization_params={"temperature": config["temperature"], "k": config["topk_k"]},
    )
    weights = np.asarray(api.level_weights(view))
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-6)
    assert (weights >= 0).all() and (weights[~active] == 0).all()
    orders = {key: np.argsort(-values, kind="stable") for key, values in (
        ("recent", recent), ("maxmc", np.where(observed, mc, -np.inf)), ("combined", np.asarray(view["scores"])[active]),
    )}
    k = min(10, size)
    overlap = len(set(orders["recent"][:k]) & set(orders["maxmc"][:k])) / k
    sorted_weights = np.sort(weights[active])[::-1]
    np.savez_compressed(out_file, weights=weights, active=active, recent=recent, maxmc=mc, observed=observed,
                        blended=mixed, **{f"{key}_rank": np.argsort(order) + 1 for key, order in orders.items()})
    return view, {
        **{key: float(value) for key, value in hybrid.statistics(sampler, weight).items()},
        "top10_overlap": overlap, "top1_weight": float(sorted_weights[0]),
        "top10_weight": float(sorted_weights[:k].sum()),
        "recent_range": [float(recent.min()), float(recent.max())],
        "maxmc_range": [float(mc[observed].min()), float(mc[observed].max())] if observed.any() else None,
    }


def draw_report(output, report):
    from matplotlib.figure import Figure

    panels = (
        ("Primary solve rate", (("solve_rate", ""),)),
        ("Mixing weight for next round", (("next_branch_lambda", ""),)),
        ("Mean priorities", (("hybrid/mean_recent", "Recent"), ("hybrid/mean_maxmc", "MaxMC"),
                             ("hybrid/mean_priority", "Mixed"))),
        ("Frontier fraction", (("frontier_fraction", ""),)),
        ("Effective replay levels", (("weight_ess", ""),)),
        ("Replay weight concentration", (("top1_weight", "Top 1"), ("top10_weight", "Top 10"))),
        ("Recent / MaxMC top-10 overlap", (("top10_overlap", ""),)),
    )
    figure = Figure(figsize=(12, 11), layout="constrained")
    axes = figure.subplots(4, 2).ravel()
    for axis, (title, series) in zip(axes, panels):
        for arm, rows in report["arms"].items():
            color = {"recent_control": "tab:blue", "hard_switch": "tab:orange", "smooth": "tab:green"}[arm]
            for (key, label), style in zip(series, ("-", "--", ":")):
                data = [(row["round"], row[key]) for row in rows if key in row]
                if data:
                    axis.plot(*zip(*data), style, color=color, marker="o", label=f"{arm} {label}".strip())
        axis.set(title=title, xlabel="Completed curriculum rounds")
        axis.grid(alpha=.25)
        if axis.lines:
            axis.legend(fontsize=7)
    axes[-1].set_visible(False)
    figure.savefig(output / "learning-and-replay.png", dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--resume-round", type=int)
    parser.add_argument("--transition-end", type=int)
    parser.add_argument("--eval-freq", type=int)
    parser.add_argument("--prefix-checkpoint", type=Path, help="Existing dual-score Recent run directory")
    parser.add_argument("--smoke", action="store_true", help="Run a six-round technical check with a smaller buffer")
    args = parser.parse_args()
    if not args.run_id or not all(c.isalnum() or c in "-_" for c in args.run_id):
        parser.error("invalid run ID")
    defaults = (6, 2, 4, 1) if args.smoke else (1000, 500, 750, 250)
    for key, default in zip(("rounds", "resume_round", "transition_end", "eval_freq"), defaults):
        if getattr(args, key) is None:
            setattr(args, key, default)
    if args.smoke and tuple(getattr(args, k) for k in ("rounds", "resume_round", "transition_end", "eval_freq")) != defaults:
        parser.error("smoke uses the fixed six-round technical schedule")
    start, end, finish, interval = args.resume_round, args.transition_end, args.rounds, args.eval_freq
    if not 0 < start < end < finish or interval <= 0 or any(r % interval for r in (start, end, finish)):
        parser.error("ordered start, end and finish must lie on evaluation boundaries")
    if args.prefix_checkpoint:
        valid_prefix(read_config(args.prefix_checkpoint), args)
    output = Path("results/hybrid-pilot") / args.run_id
    names = {arm: f"{args.run_id}-{arm}" for arm in ("prefix", *hybrid.MODES, "hard-resumed", "plain-recent")}
    roots = {arm: Path("checkpoints") / name / str(args.seed) for arm, name in names.items()}
    if output.exists() or any(path.exists() for path in roots.values()):
        parser.error("output or checkpoint directory already exists")
    output.mkdir(parents=True)
    os.environ["WANDB_MODE"] = "offline"
    datasets, monitoring = validation_sets(output, args.smoke)
    common = ("--hybrid_start_round", str(start), "--hybrid_end_round", str(end))
    if args.smoke:
        common += ("--level_buffer_capacity", "64", "--minimum_fill_ratio", ".25", "--replay_prob", "1")

    def run(arm, score, options):
        print("Training", arm, flush=True)
        return train(score, names[arm], args, monitoring, (*common, *options))

    report = {"seed": args.seed, "horizon": finish, "fork_round": start, "transition_end": end,
              "smoke": args.smoke, "devices": [str(device) for device in jax.devices()], "arms": {}, "runs": {},
              "validation_seeds": {"primary": [LEVEL_SEED, EVAL_SEED],
                                   "secondary": [SECONDARY_LEVEL_SEED, SECONDARY_EVAL_SEED]}}
    reader = ocp.StandardCheckpointer()
    try:
        prefix = args.prefix_checkpoint or roots["prefix"]
        if args.prefix_checkpoint is None:
            report["runs"]["prefix"] = run("prefix", "recent_learnability",
                                          ("--dual_score_tracking", "--stop_after_updates", str(start)))
        prefix_config = read_config(prefix)
        valid_prefix(prefix_config, args)
        source, state = read_models(reader, prefix, (start,))[start]
        if "maxmc_score" not in state["sampler"]["levels_extra"] or "runner_rng" not in state:
            raise ValueError("common prefix checkpoint lacks dual-score history or training PRNG")
        base = {"round": start, "optimizer_steps": int(state["step"]), "leaves": teacher_parity.tree_digests(state)}
        save_json(output / "common_initial_state.json", base)
        common_copy = output / "common_checkpoint"
        destination = common_copy / "models" / source.parent.name
        shutil.copytree(source.parent, destination)
        shutil.copyfile(prefix / "config.json", common_copy / "config.json")
        assert teacher_parity.tree_digests(reader.restore((destination / "default").resolve())) == base["leaves"]
        report["common_checkpoint"] = str(destination / "default")
        report["prefix_source"] = str(source)
        common_resume = ("--resume_checkpoint_directory", str(common_copy),
                         "--resume_checkpoint_step", source.parent.name, "--fork_hybrid_on_resume")
        checkpoints = list(range(start + interval, finish + 1, interval))
        evaluate = make_evaluator() if datasets else None
        for arm in hybrid.MODES:
            options = (*common_resume, "--hybrid_mode", arm)
            split = args.smoke and arm == "hard_switch"
            report["runs"][arm] = run(arm, "hybrid", (*options, "--stop_after_updates", str(end)) if split else options)
            assert json.loads((roots[arm] / "hybrid_initial_state.json").read_text()) == base
            config = read_config(roots[arm])
            if split:
                records = read_models(reader, roots[arm], [r for r in checkpoints if r <= end])
                mid_file, middle = records[end]
                resume = ("--resume_checkpoint_directory", str(roots[arm]), "--resume_checkpoint_step",
                          mid_file.parent.name, "--hybrid_mode", arm)
                report["runs"]["hard-resumed"] = run("hard-resumed", "hybrid", resume)
                restored = json.loads((roots["hard-resumed"] / "hybrid_initial_state.json").read_text())
                assert restored == {"round": end, "optimizer_steps": int(middle["step"]),
                                    "leaves": teacher_parity.tree_digests(middle)}
                records.update(read_models(reader, roots["hard-resumed"], [r for r in checkpoints if r > end]))
            else:
                records = read_models(reader, roots[arm], checkpoints)
            rows = []
            for round_index, (_, record) in [(start, (None, state)), *sorted(records.items())]:
                weight = hybrid.config_coefficient(prefix_config if round_index == start else config, round_index)
                view, scores = describe(record["sampler"], config, weight, output / f"{arm}-{round_index}-scores.npz")
                row = {**optimizer_and_budget(record, config, round_index), **scores,
                       **buffer_snapshot(record, config, weight_sampler=view),
                       "next_branch_lambda": float(hybrid.config_coefficient(config, round_index))}
                if not args.smoke:
                    for label, (levels, keys) in datasets.items():
                        returns, lengths = map(np.asarray, evaluate(record["params"], levels, keys))
                        if not np.isfinite(returns).all():
                            raise FloatingPointError("nonfinite validation returns")
                        successes = returns > 0
                        result = {"solve_rate": float(successes.mean()), "solves": int(successes.sum()),
                                  "mean_return": float(returns.mean()), "groups": {
                                      str(n): float(successes[:, i*LEVELS_PER_GROUP:(i+1)*LEVELS_PER_GROUP].mean())
                                      for i, n in enumerate(WALL_DRAWS)
                                  }}
                        row.update(result if label == "primary" else {"secondary": result})
                        np.savez_compressed(output / f"{arm}-{round_index}-{label}.npz",
                                            returns=returns, episode_lengths=lengths)
                rows.append(row)
            report["arms"][arm] = rows
            if args.smoke:
                assert [row["replay"] for row in rows] == [1, 2, 3, 4, 5]
                expected = {
                    "recent_control": [0., 0., 0., 0., 0.],
                    "hard_switch": [1., 1., 1., 1., 1.],
                    "smooth": [0., .5, 1., 1., 1.],
                }[arm]
                assert [row["next_branch_lambda"] for row in rows] == expected
            save_json(output / "summary.json", report)
        if args.smoke:
            report["runs"]["plain-recent"] = run("plain-recent", "recent_learnability", ())
            _, old = read_models(reader, roots["plain-recent"], (finish,))[finish]
            _, recent = read_models(reader, roots["recent_control"], (finish,))[finish]
            sampler = {**recent["sampler"], "levels_extra": {
                key: recent["sampler"]["levels_extra"][key] for key in old["sampler"]["levels_extra"]
            }}
            left = teacher_parity.tree_digests(old)
            right = teacher_parity.tree_digests({**recent, "sampler": sampler})
            report["plain_recent_comparison"] = {
                "changed_leaves": [key for key in left if left[key] != right[key]],
                "max_abs_param_delta": max(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(
                    jax.tree_util.tree_leaves(old["params"]), jax.tree_util.tree_leaves(recent["params"]))),
            }
    finally:
        reader.close()
    draw_report(output, report)
    lines = ["| Arm | Rounds | Solves / attempts | PPO steps | Environment steps | Stored λ | Next λ |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, rows in report["arms"].items():
        for row in rows:
            solves = f"{row['solves']}/{EVAL_ATTEMPTS * len(WALL_DRAWS) * LEVELS_PER_GROUP}" if "solves" in row else "—"
            lines.append(f"| {arm} | {row['round']} | {solves} | {row['optimizer_steps']} | "
                         f"{row['environment_steps']} | {row['hybrid/lambda']:.3f} | "
                         f"{row['next_branch_lambda']:.3f} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(output / "summary.json", report)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
