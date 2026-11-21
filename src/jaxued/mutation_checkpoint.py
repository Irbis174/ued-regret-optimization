"""Save and resume a Maze trainer with its exact next PRNG key."""

import json
from pathlib import Path

import jax
import orbax.checkpoint as ocp
from flax import serialization

from jaxued.adaptive_mutation import DEFAULT_ARMS
from jaxued.map_elites import DEFAULT_CONFIG as QD_DEFAULTS
from jaxued.hybrid_teacher import DEFAULTS as HYBRID_DEFAULTS


MUTATION_DEFAULTS = {
    "mutation_selection": "fixed",
    "bandit_strategy": "epsilon_greedy",
    "bandit_epsilon": 0.15,
    "bandit_ema_alpha": 0.1,
    "bandit_reward": "batch_rank",
    "mutation_arms": list(DEFAULT_ARMS),
}
IGNORED = {
    "project", "run_name", "group_name", "mode", "checkpoint_directory", "checkpoint_to_eval",
    "checkpoint_save_interval", "max_number_of_checkpoints", "adaptive_mutation",
    "resume_checkpoint_directory", "resume_checkpoint_step", "initialize_bandit_on_resume",
    "allow_legacy_resume", "stop_after_updates", "initialize_qd_on_resume", "debug_teacher_parity",
    "fork_hybrid_on_resume", "pace_log_diagnostics",
}


def payload(rng, train_state):
    """Keep params at the top level for the existing policy evaluator."""
    return {**serialization.to_state_dict(train_state), "runner_rng": rng}


def validate_config(saved, requested, initialize_bandit, initialize_qd=False, fork_hybrid=False):
    defaults = {
        **MUTATION_DEFAULTS,
        **QD_DEFAULTS,
        **HYBRID_DEFAULTS,
        "har_history_weight": 0.2,
        "replay_max_copies": 0,
        "replay_preserve_frontier_mass": False,
        "pace_score_chunk_size": 1,
        "pace_score_entropy": False,
    }
    saved = {**defaults, **saved}
    requested = {**defaults, **requested}
    migration = saved["mutation_selection"] == "fixed" and requested["mutation_selection"] == "bandit"
    if initialize_bandit and (not migration or not saved.get("use_accel")):
        raise ValueError("bandit initialization requires a fixed ACCEL checkpoint and a bandit destination")
    if initialize_qd and (
        initialize_bandit or saved["qd_mode"] != "off" or requested["qd_mode"] == "off"
        or not saved.get("use_accel") or saved["mutation_selection"] != "fixed"
    ):
        raise ValueError("QD initialization requires a fixed ACCEL checkpoint without QD and a QD destination")
    ignored = IGNORED | MUTATION_DEFAULTS.keys() if initialize_bandit else IGNORED
    if initialize_qd:
        ignored = ignored | QD_DEFAULTS.keys()
    if fork_hybrid:
        if (initialize_bandit or initialize_qd or not saved["dual_score_tracking"]
                or not requested["dual_score_tracking"] or saved["hybrid_mode"] != "recent_control"
                or saved.get("score_function") not in ("recent_learnability", "hybrid")
                or requested.get("score_function") != "hybrid"):
            raise ValueError("hybrid fork requires a dual-score Recent checkpoint and a hybrid destination")
        ignored = ignored | {"score_function", "hybrid_mode"}
    differences = sorted(
        key for key in saved.keys() | requested.keys()
        if key not in ignored and json.dumps(saved.get(key)) != json.dumps(requested.get(key))
    )
    if differences:
        raise ValueError(f"incompatible resume configuration: {', '.join(differences)}")


def restore(directory, step, template, config):
    directory = Path(directory).resolve()
    saved = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    initialize = config.get("initialize_bandit_on_resume", False)
    initialize_qd = config.get("initialize_qd_on_resume", False)
    fork_hybrid = config.get("fork_hybrid_on_resume", False)
    validate_config(saved, config, initialize, initialize_qd, fork_hybrid)
    models = directory / "models"
    if step == -1:
        steps = [int(p.name) for p in models.iterdir() if p.is_dir() and p.name.isdecimal()
                 and (p / "default").is_dir()]
        if not steps:
            raise FileNotFoundError(f"no usable checkpoints under {models}")
        step = max(steps)
    checkpointer = ocp.StandardCheckpointer()
    try:
        record = checkpointer.restore(models / str(step) / "default")
    finally:
        checkpointer.close()

    rng = record.pop("runner_rng", None)
    completed = sum(int(record[f"num_{name}_updates"]) for name in ("dr", "replay", "mutation"))
    if fork_hybrid and (rng is None or completed != config["hybrid_start_round"]):
        raise ValueError("hybrid fork requires its exact start-round checkpoint and original training PRNG")
    if rng is None:
        if not config.get("allow_legacy_resume", False):
            raise ValueError("legacy checkpoint has no training PRNG; use --allow_legacy_resume to reseed explicitly")
        rng = jax.random.fold_in(jax.random.PRNGKey(config["seed"]), completed)
        print("Legacy resume: training PRNG reseeded from seed and completed rounds.", flush=True)
    stored = record["sampler"].get("mutation_bandit")
    desired = template.sampler.get("mutation_bandit")
    if initialize:
        if stored is not None or desired is None:
            raise ValueError("checkpoint does not support explicit bandit initialization")
        record["sampler"]["mutation_bandit"] = serialization.to_state_dict(desired)
    elif (stored is None) != (desired is None):
        raise ValueError("checkpoint bandit state does not match the requested mode")
    elif stored is not None and any(stored[name].shape != value.shape for name, value in desired.items()):
        raise ValueError("checkpoint bandit state has incompatible arm count")
    stored_archive = record["sampler"].get("qd_archive")
    desired_archive = template.sampler.get("qd_archive")
    if initialize_qd:
        if stored_archive is not None or desired_archive is None:
            raise ValueError("checkpoint does not support explicit QD initialization")
        record["sampler"]["qd_archive"] = serialization.to_state_dict(desired_archive)
        record["sampler"]["qd_max_age"] = template.sampler["qd_max_age"]
    elif (stored_archive is None) != (desired_archive is None):
        raise ValueError("checkpoint archive does not match the requested mode")
    elif stored_archive is not None and stored_archive["occupied"].shape != desired_archive["occupied"].shape:
        raise ValueError("checkpoint archive has incompatible capacity")
    if config.get("dual_score_tracking", False):
        extra = record["sampler"]["levels_extra"]
        expected = template.sampler["levels_extra"]
        if expected.keys() != extra.keys() or any(extra[key].shape != value.shape for key, value in expected.items()):
            raise ValueError("checkpoint is missing compatible dual-score history")
    return rng, serialization.from_state_dict(template, record), completed
