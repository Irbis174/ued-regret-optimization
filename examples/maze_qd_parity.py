"""Run independent paired ACCEL parity checks and compare their traces."""

import argparse
from contextlib import nullcontext
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp

from examples.maze_score_pilot import EVAL_ATTEMPTS, EVAL_SEED, make_evaluator, validation_levels
from jaxued import teacher_parity
from jaxued.environments.maze.level import prefabs


SMOKE_LEVEL = "QDParitySmoke"
MONITORING_INDICES = (0, 1, 16, 17, 32, 33)
MONITORING_NAMES = [f"qd-parity-validation-{index}" for index in MONITORING_INDICES]
CHILD = (
    "from examples.maze_qd_parity import install_monitoring; "
    "import runpy, sys; install_monitoring(sys.argv.pop(1) == 'standard'); "
    "runpy.run_path('examples/maze_plr.py', run_name='__main__')"
)


def install_monitoring(standard):
    if standard:
        levels, _ = validation_levels()
        for name, index in zip(MONITORING_NAMES, MONITORING_INDICES):
            prefabs[name] = jax.tree_util.tree_map(lambda x: x[index], levels).to_str()
    else:
        rows = prefabs["TrivialMaze"].splitlines()
        prefabs[SMOKE_LEVEL] = "\n".join([row.ljust(13, "#") for row in rows] + ["#" * 13] * (13 - len(rows)))


def load_trace(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def compare_traces(left, right):
    if len(left) != len(right):
        raise ValueError("trace lengths differ")
    first, first_by_field = None, {}
    for a, b in zip(left, right):
        if (a["round"], a["stage"]) != (b["round"], b["stage"]):
            raise ValueError("trace stages differ")
        changed = sorted(key for key in a.keys() | b.keys() if a.get(key) != b.get(key))
        if changed:
            location = {"round": a["round"], "stage": a["stage"]}
            if first is None:
                first = {**location, "fields": changed}
            for key in changed:
                first_by_field.setdefault(key, location)
    return {"first": first, "first_by_field": first_by_field}


def compare_checkpoint(left, right):
    f = {**left, "sampler": teacher_parity.base_sampler(left["sampler"])}
    s = {**right, "sampler": teacher_parity.base_sampler(right["sampler"])}
    params = zip(jax.tree_util.tree_leaves(f["params"]), jax.tree_util.tree_leaves(s["params"]))
    return {
        "changed_fields": [key for key in sorted(f.keys() | s.keys()) if
                           teacher_parity.tree_digests(f.get(key)) != teacher_parity.tree_digests(s.get(key))],
        "max_abs_param_delta": max(float(np.abs(np.asarray(a) - np.asarray(b)).max()) for a, b in params),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--standard", action="store_true", help="Run the matched 250-round pilot schedule")
    parser.add_argument("--trace", action="store_true", help="Write per-round parity traces")
    parser.add_argument("--dump-hlo", action="store_true", help="Save HLO for initialization and train-and-evaluate")
    parser.add_argument("--smoke-envs", type=int, default=32, help="Parallel environments in each technical round")
    args = parser.parse_args()
    if not args.run_id or not all(c.isalnum() or c in "-_" for c in args.run_id):
        parser.error("invalid run ID")
    if args.smoke_envs < 1 or (args.standard and args.smoke_envs != 32):
        parser.error("--smoke-envs only changes the technical schedule and must be positive")
    output = Path("results/qd-shadow-parity") / args.run_id
    arms = ("fixed-a", "fixed-b", "shadow")
    if args.trace and not args.standard:
        arms += ("shadow-resumed",)
    directories = {arm: Path("checkpoints") / f"{args.run_id}-{arm}" / "0" for arm in arms}
    if output.exists() or any(p.exists() for p in directories.values()):
        parser.error("audit output already exists")
    output.mkdir(parents=True)
    os.environ["WANDB_MODE"] = "offline"
    budget, interval = (250, 125) if args.standard else (6, 3)
    common = [
        "--use_accel", "--score_function", "recent_learnability", "--num_edits", "5", "--seed", "0",
        "--no-exploratory_grad_updates", "--num_updates", str(budget), "--eval_freq", str(interval),
        "--checkpoint_save_interval", "1", "--max_number_of_checkpoints", "2",
    ]
    if args.standard:
        common += ["--eval_num_attempts", "4", "--eval_levels", *MONITORING_NAMES]
    else:
        common += ["--num_train_envs", str(args.smoke_envs), "--level_buffer_capacity", str(2 * args.smoke_envs),
                   "--minimum_fill_ratio", "0.25", "--replay_prob", "1",
                   "--eval_num_attempts", "2", "--eval_levels", SMOKE_LEVEL]
    for arm in arms:
        command = [sys.executable, "-B", "-c", CHILD, "standard" if args.standard else "smoke",
                   *common, "--run_name", f"{args.run_id}-{arm}",
                   "--qd_mode", "shadow" if arm.startswith("shadow") else "off"]
        if args.trace:
            command += ["--debug_teacher_parity", str(output / f"{arm}.jsonl")]
        if arm == "shadow-resumed":
            command += ["--resume_checkpoint_directory", str(directories["shadow"]), "--resume_checkpoint_step", "0"]
        elif not args.standard:
            command += ["--stop_after_updates", "3"]
        print(f"Starting {arm}", flush=True)
        dump = tempfile.TemporaryDirectory(prefix="qd-parity-hlo-") if args.dump_hlo else nullcontext()
        with dump as temporary, (output / f"{arm}.log").open("w", encoding="utf-8") as log:
            env = os.environ.copy()
            if temporary is not None:
                env["XLA_FLAGS"] = " ".join((env.get("XLA_FLAGS", ""), f"--xla_dump_to={temporary}",
                                             "--xla_dump_hlo_as_text",
                                             "--xla_dump_hlo_module_re=create_train_state|train_and_eval_step")).strip()
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=env)
            if temporary is not None:
                paths = sorted(Path(temporary).glob("*optimizations.txt"))
                for path in paths:
                    shutil.copyfile(path, output / f"{arm}-{path.name}")
                if not paths:
                    raise RuntimeError(f"XLA did not dump train-and-evaluate HLO for {arm}")
        print(f"Finished {arm}", flush=True)

    report = {"seed": 0, "standard": args.standard, "trace": args.trace, "devices": [str(d) for d in jax.devices()],
              "directories": {key: str(path) for key, path in directories.items()}}
    reader = ocp.StandardCheckpointer()
    try:
        indices = (0, 1) if args.standard else (0,)
        counters = ("num_dr_updates", "num_replay_updates", "num_mutation_updates", "step")
        pairs = (("fixed-a", "fixed-b", "fixed-b"), ("fixed-a", "shadow", "shadow"),
                 ("fixed-b", "shadow", "shadow-vs-fixed-b"))
        if args.standard:
            validation, _ = validation_levels()
            keys = jax.random.split(jax.random.PRNGKey(EVAL_SEED), EVAL_ATTEMPTS)
            evaluate = make_evaluator()
        def restore(arm, index):
            return reader.restore((directories[arm] / "models" / str(index) / "default").resolve())
        checkpoints = {}
        for index in indices:
            states = {arm: restore(arm, index) for arm in arms if arm != "shadow-resumed"}
            if not args.standard:
                assert all(tuple(int(state[key]) for key in counters) == (1, 1, 1, 5) for state in states.values())
            rows = {}
            for arm, state in states.items():
                row = {key: int(state[key]) for key in counters}
                assert sum(row[key] for key in counters[:3]) == interval * (index + 1)
                assert ("qd_archive" in state["sampler"]) == (arm == "shadow")
                if args.standard:
                    returns, lengths = map(np.asarray, evaluate(state["params"], validation, keys))
                    row["solves"] = int(np.sum(returns > 0))
                    np.savez_compressed(output / f"{arm}-{index + 1}.npz", returns=returns, episode_lengths=lengths)
                rows[arm] = row
            checkpoints[str((index + 1) * interval)] = {
                "arms": rows, "comparisons": {name: compare_checkpoint(states[a], states[b]) for a, b, name in pairs},
            }
        report["checkpoints"] = checkpoints
        if args.trace:
            traces = {arm: load_trace(output / f"{arm}.jsonl") for arm in arms}
            report["traces"] = {name: compare_traces(traces[a], traces[b]) for a, b, name in pairs}
            if "shadow-resumed" in traces:
                initial = traces["shadow-resumed"][0]
                boundary = traces["shadow"][-1]
                assert boundary["stage"] == "evaluation_boundary"
                assert {**boundary, "stage": "initial"} == initial
                resumed = restore("shadow-resumed", 1)
                assert tuple(int(resumed[key]) for key in counters) == (1, 3, 2, 15)
                assert int(resumed["sampler"]["qd_archive"]["occupied"].sum()) >= int(
                    states["shadow"]["sampler"]["qd_archive"]["occupied"].sum()
                )
                report["resume"] = {"restored_boundary_equal": True, "counters": dict(zip(counters, (1, 3, 2, 15)))}
    finally:
        reader.close()
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
