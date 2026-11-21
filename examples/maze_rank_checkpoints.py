"""Rank saved maze policies on the archived synthetic validation set."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from examples.maze_score_pilot import EVAL_ATTEMPTS, EVAL_SEED, make_evaluator
from jaxued.environments.maze import Level


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--round", required=True, type=int)
    parser.add_argument("runs", nargs="+", type=Path, help="run seed directories containing config.json and models")
    args = parser.parse_args()

    configs = [json.loads((run / "config.json").read_text(encoding="utf-8")) for run in args.runs]
    for field in ("seed", "num_updates", "num_train_envs", "num_steps", "eval_freq", "agent_view_size"):
        if any(config[field] != configs[0][field] for config in configs[1:]):
            parser.error(f"runs differ in {field}")
    config = configs[0]
    if config["agent_view_size"] != 5:
        parser.error("the archived evaluation protocol uses agent_view_size=5")
    interval = config["eval_freq"]
    if not 0 < args.round <= config["num_updates"] or args.round % interval:
        parser.error("--round must be a positive evaluation boundary within the training horizon")
    index = args.round // interval - 1

    with np.load(args.validation) as archive:
        count, height, width = archive["wall_map"].shape
        if (count, height, width) != (48, 13, 13):
            parser.error("expected 48 synthetic maps of size 13 by 13")
        levels = Level(
            wall_map=jnp.asarray(archive["wall_map"]),
            goal_pos=jnp.asarray(archive["goal_pos"]),
            agent_pos=jnp.asarray(archive["agent_pos"]),
            agent_dir=jnp.asarray(archive["agent_dir"]),
            width=jnp.full(count, width),
            height=jnp.full(count, height),
        )
    keys = jax.random.split(jax.random.PRNGKey(EVAL_SEED), EVAL_ATTEMPTS)
    evaluate = make_evaluator()
    print(
        f"seed={config['seed']} | round={args.round} | training horizon={config['num_updates']} | "
        f"validation={count} x {EVAL_ATTEMPTS} | eval seed={EVAL_SEED} | {jax.devices()[0]}",
        flush=True,
    )

    results = []
    reader = ocp.StandardCheckpointer()
    try:
        for run in args.runs:
            checkpoint = run / "models" / str(index) / "default"
            state = reader.restore(checkpoint.resolve())
            completed = sum(int(state[f"num_{branch}_updates"]) for branch in ("dr", "replay", "mutation"))
            if completed != args.round:
                raise ValueError(f"{checkpoint} contains round {completed}, expected {args.round}")
            returns, _ = evaluate(state["params"], levels, keys)
            returns = np.asarray(returns)
            if returns.shape != (EVAL_ATTEMPTS, count) or not np.isfinite(returns).all():
                raise ValueError(f"unexpected evaluation returns for {checkpoint}")
            results.append((int((returns > 0).sum()), run.resolve().parent.name, int(state["step"])))
    finally:
        reader.close()

    total = count * EVAL_ATTEMPTS
    print(f"{'Rank':>4}  {'Run':<55} {'Solved':>9} {'Solve rate':>11} {'Adam steps':>11}")
    previous = None
    rank = 0
    for position, (solved, name, steps) in enumerate(sorted(results, key=lambda row: (-row[0], row[1])), 1):
        if solved != previous:
            rank = position
        print(f"{rank:>4}  {name:<55} {solved:>3}/{total:<5} {solved / total:>11.2%} {steps:>11}")
        previous = solved


if __name__ == "__main__":
    main()
