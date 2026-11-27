"""Run the PACE v2 study sequentially in Ubuntu/WSL. See reports/pace_v2/README.md."""

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "pace_v2"
STATE = REPORT / "experiment_queue.json"
F03_FULL_REQUEST = REPORT / "f03_full_request.json"
SOURCE = REPORT / "source"
LITE_REPORT = ROOT / "reports" / "hybrid_pace_lite"
LITE_SOURCE = LITE_REPORT / "source"
LITE_METHOD = "hybrid_pace_lite_floor"
LITE_MANUAL = LITE_REPORT / "manual"
VALIDATION = REPORT / "validation.npz"
BASELINE_CONFIG = ROOT / "checkpoints" / "accel-s0" / "0" / "config.json"
STATUS_TERMINAL = {"COMPLETED", "FAILED", "SKIPPED"}
TARGET = 5000
HORIZON = 30000
SMOKE = 100
EXPERT_LEVELS = ("SixteenRooms", "SixteenRooms2", "Labyrinth", "LabyrinthFlipped",
                 "Labyrinth2", "StandardMaze", "StandardMaze2", "StandardMaze3")
BASELINE_FIELDS = (
    "lr", "max_grad_norm", "num_steps", "num_train_envs", "num_minibatches", "gamma", "epoch_ppo",
    "clip_eps", "gae_lambda", "entropy_coeff", "critic_coeff", "level_buffer_capacity", "replay_prob",
    "staleness_coeff", "temperature", "topk_k", "minimum_fill_ratio", "prioritization", "num_edits",
    "agent_view_size", "n_walls",
)


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic(path, content):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def write_json(path, item):
    atomic(path, json.dumps(item, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # The worker may still be writing the last record.
    return rows


def job(id_, method="pace_adaptive_floor", *, q=.75, floor=.2, ema=.95, shape=1., alpha=.2,
        seed=0, target=TARGET, reason="", parent=None):
    return dict(id=id_, method=method, q=q, floor=floor, ema=ema, shape=shape, alpha=alpha,
                seed=seed, target=target, reason=reason, parent=parent, status="PENDING", attempts=[])


def initialize():
    jobs = [
        job("S00", target=SMOKE, reason="Проверка CUDA на полном размере rollout перед F01."),
        job("F01", reason="Стартовая гипотеза."),
        job("F02", floor=.10, reason="Меньший вес Floor."),
        job("F03", floor=.30, reason="Больший вес Floor."),
        job("F04", q=.50, reason="Нижний квантиль."),
        job("F05", q=.90, reason="Верхний квантиль."),
        job("B00", "MaxMC", reason="ACCEL на том же бюджете, schedule и validation."),
    ]
    config = read_json(BASELINE_CONFIG)
    assert config["use_accel"] and not config["exploratory_grad_updates"] and config["buffer_duplicate_check"]
    assert config["num_updates"] == HORIZON and config["eval_freq"] == 250
    return {
        "jobs": jobs, "events": [], "winner": None, "stage": "main", "created": now(),
        "baseline": {key: config[key] for key in BASELINE_FIELDS},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source": read_json(SOURCE / "manifest.json"),
        "validation": read_json(VALIDATION.with_suffix(".json")),
    }


def event(state, message):
    state["events"].append({"at": now(), "message": message})
    print(f"{now()} {message}", flush=True)


def current_metric(task):
    if task["status"] != "COMPLETED":
        return None
    return task["attempts"][-1]["result"]["metrics"]


def ranked(tasks):
    usable = [task for task in tasks if current_metric(task) is not None]
    return sorted(usable, key=lambda task: (
        -current_metric(task)["eval/validation_solve_rate"],
        -current_metric(task)["eval/validation_hard_solve_rate"],
        -current_metric(task).get("eval/recent_solve_rate", 0),
        current_metric(task)["wall_clock_time"], task["id"],
    ))


def replicate(id_, source, reason, **changes):
    params = {key: source[key] for key in ("q", "floor", "ema", "shape", "alpha")}
    method = changes.pop("method", source["method"])
    return job(id_, method, reason=reason, **{**params, **changes})


def f03_full_jobs(state):
    source = next(task for task in state["jobs"] if task["id"] == "F03")
    metric = current_metric(source)
    if not metric or metric["num_updates"] != TARGET or metric["num_env_steps"] != TARGET * 8192:
        raise ValueError("F03 requires a completed, comparable 5000-round run")
    existing = {task["id"] for task in state["jobs"]}
    return [
        replicate(id_, source, "Полный бюджет F03 по запросу пользователя.", seed=seed, target=HORIZON,
                  parent="F03" if seed == 0 else None)
        for seed in range(3)
        if (id_ := f"F03-FULL{seed}") not in existing
    ]


def request_f03_full():
    state = read_json(STATE)
    jobs = f03_full_jobs(state)
    if not jobs:
        print("F03 seeds 0, 1, 2 are already recorded in the queue.")
        return
    if any(task["seed"] == 0 for task in jobs):
        source = next(task for task in state["jobs"] if task["id"] == "F03")
        attempt = source["attempts"][-1]
        train_result(read_json(Path(attempt["folder"]) / "spec.json"))
    write_json(F03_FULL_REQUEST, {"requested": now()})
    print("Requested F03 seeds 0, 1, 2. Seed 0 resumes from round 5000.")


def review_floor(state):
    if state.get("floor_reviewed"):
        return
    group = [task for task in state["jobs"] if task["id"] in ("F01", "F02", "F03")]
    if not all(task["status"] in STATUS_TERMINAL for task in group):
        return
    measurements = ", ".join(
        f"{task['id']} β={task['floor']}: {100 * current_metric(task)['eval/validation_solve_rate']:.2f}%"
        for task in ranked(group)
    )
    event(state, f"Срез абляции Floor на 5 000. {measurements or 'Валидные измерения отсутствуют.'}")
    state["floor_reviewed"] = True


def choose_next(state):
    if "lite_source" in state:
        choose_lite(state)
        return
    jobs = state["jobs"]
    by_id = {task["id"]: task for task in jobs}
    if state["winner"] or state["stage"] == "exhausted":
        return
    if not all(task["status"] in STATUS_TERMINAL for task in jobs):
        return
    standalones = ranked(task for task in jobs if task["id"].startswith("F"))
    if not standalones:
        event(state, "Нет технически валидных результатов PACE на 5000 раундах.")
        state["stage"] = "exhausted"
        return
    best = standalones[0]
    best_metrics = current_metric(best)
    baseline = current_metric(by_id["B00"])
    gap = (baseline["eval/validation_solve_rate"] - best_metrics["eval/validation_solve_rate"]
           if baseline else None)
    hard_gap = (baseline["eval/validation_hard_solve_rate"] - best_metrics["eval/validation_hard_solve_rate"]
                if baseline else None)
    competitive = gap is not None and (gap <= 0 or (
        gap <= .03 and hard_gap <= .03
        and baseline["eval/recent_solve_rate"] - best_metrics["eval/recent_solve_rate"] <= .03
    ))
    if state["stage"] == "main" and not competitive:
        jobs.extend((job("F06", floor=.05, reason="Контроль малого веса Floor."),
                     job("F07", "pace_adaptive", floor=0., reason="Абляция Floor.")))
        first_floor = ranked(by_id[id_] for id_ in ("F01", "F02", "F03"))
        if len(first_floor) >= 2 and first_floor[0]["floor"] == .30:
            gain = (current_metric(first_floor[0])["eval/validation_solve_rate"]
                    - current_metric(first_floor[1])["eval/validation_solve_rate"])
            if gain >= .03:
                jobs.append(replicate("F08", first_floor[0], "Floor 0.30 лидирует с запасом ≥3 п.п.", floor=.40))
        central = current_metric(by_id["F01"])
        if central and best["q"] in (.50, .90) and (
            best_metrics["eval/validation_solve_rate"] - central["eval/validation_solve_rate"] >= .03
        ):
            neighbor = .65 if best["q"] == .50 else .85
            jobs.append(replicate("F09", best, f"Соседний q={neighbor} после выигрыша ≥3 п.п. над F01.", q=neighbor))
        state["stage"] = "ablation"
        event(state, f"Лидер {best['id']}. Переходим к абляции Floor. Разрыв с ACCEL: {gap!r}.")
        return
    degraded = best_metrics["pace/nonfinite_fraction"] > .01 or (
        best_metrics["buffer/solvable_fraction"] < .5 and best_metrics["pace/near_zero_score_fraction"] > .5
    )
    if state["stage"] in ("main", "ablation") and ((gap is not None and gap > .03) or degraded):
        jobs.extend((replicate("H01", best, "Проверка лучшего PACE с 20% вкладом в MaxMC.", method="pace_hybrid"),
                     replicate("H04", best, "Гибридная абляция Floor.", method="pace_hybrid", floor=0.)))
        state["stage"] = "hybrid"
        event(state, f"Гибридный этап после самостоятельного поиска. Лидер {best['id']}, разрыв {gap!r}.")
        return
    hybrids = ranked(task for task in jobs if task["id"].startswith("H"))
    hybrid = hybrids[0] if hybrids else None
    winner = best
    reason = "Лучшая самостоятельная конфигурация по решаемости на 5000 раундах."
    if hybrid:
        improvement = current_metric(hybrid)["eval/validation_solve_rate"] - best_metrics["eval/validation_solve_rate"]
        if improvement >= .02:
            winner = hybrid
            reason = f"Гибрид улучшил конечную validation solve rate на {100 * improvement:.2f} п.п."
        else:
            reason += " Преимущество гибрида меньше 2 п.п."
    state["winner"] = winner["id"]
    state["stage"] = "full"
    state["selection_reason"] = reason
    jobs.append(replicate("FULL0", winner, reason, target=HORIZON, parent=winner["id"]))
    jobs.extend(replicate(f"FULL{seed}", winner, "Независимый полный seed победителя.", seed=seed, target=HORIZON)
                for seed in (1, 2))
    event(state, f"Выбран {winner['id']}. FULL0 продолжит полный checkpoint 5000. {reason}")


def lite_job(id_, alpha=.20, **kwargs):
    method = "pace_lite_floor" if alpha == 1 else LITE_METHOD
    return {**job(id_, method, alpha=alpha, floor=.30, **kwargs), "study": "lite"}


def choose_lite(state):
    jobs = state["jobs"]
    if LITE_MANUAL.exists() or state["winner"] or state["stage"] == "lite_exhausted" or any(
        task["status"] not in STATUS_TERMINAL for task in jobs
    ):
        return
    by_id = {task["id"]: task for task in jobs}
    if by_id["LS00"]["status"] != "COMPLETED":
        state["stage"] = "lite_exhausted"
        event(state, "CUDA smoke Lite завершился ошибкой. Причина в логах попыток LS00.")
        return
    if "H01" not in by_id:
        jobs.append(lite_job("H01", reason="Основной гибрид после CUDA smoke."))
        return
    baseline = current_metric(by_id["B00"])
    if "H02" not in by_id:
        metric = current_metric(by_id["H01"])
        detail = "Нет валидного результата." if not metric else (
            f"Validation {100 * metric['eval/validation_solve_rate']:.2f}%, "
            f"hard {100 * metric['eval/validation_hard_solve_rate']:.2f}%, "
            f"{metric['wall_clock_time'] / 60:.1f} мин."
        )
        event(state, f"H01. {detail} H02 проверит меньший вес Lite.")
        jobs.append(lite_job("H02", .10, reason="Чувствительность к меньшему весу Lite."))
        return
    if "H03" not in by_id and "L01" not in by_id:
        m1, m2 = (current_metric(by_id[key]) for key in ("H01", "H02"))
        rate = "eval/validation_solve_rate"
        if m1 and m2 and m1[rate] + .03 < min(m2[rate], baseline[rate]):
            skipped = lite_job("H03", .35, reason="Больший вес уже ухудшил качество между H02 и H01.")
            jobs.append({**skipped, "status": "SKIPPED"})
            event(state, "H03 пропущен после ухудшения на большем весе Lite. Переходим к L01.")
        else:
            jobs.append(lite_job("H03", .35, reason="Чувствительность к большему весу Lite."))
            return
    if "L01" not in by_id:
        jobs.append(lite_job("L01", 1., reason="Самостоятельный Lite-Floor при том же пороге."))
        return
    candidates = ranked(task for task in jobs if task.get("study") == "lite" and task["target"] == TARGET)
    if not candidates:
        state["stage"] = "lite_exhausted"
        event(state, "Все предварительные прогоны Lite завершились техническими ошибками.")
        return
    best_rate = current_metric(candidates[0])["eval/validation_solve_rate"]
    close = [task for task in candidates if best_rate - current_metric(task)["eval/validation_solve_rate"] <= .015]
    winner = min(close, key=lambda task: (
        -current_metric(task)["eval/validation_hard_solve_rate"],
        -current_metric(task)["eval/recent_solve_rate"],
        current_metric(task)["wall_clock_time"],
    ))
    state["winner"], state["stage"] = winner["id"], "lite_full"
    reason = "Лучший Lite-кандидат по validation с допуском 1,5 п.п. для сравнения hard, динамики и времени."
    state["selection_reason"] = reason
    for seed in range(3):
        jobs.append(lite_job(f"FULL{seed}", winner["alpha"], seed=seed, target=HORIZON,
                             parent=winner["id"] if seed == 0 else None, reason=reason))
    event(state, f"Выбран {winner['id']}. FULL0 продолжит checkpoint 5000.")


def identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] != "Z":
            return [int(pid), stat[19], Path("/proc/sys/kernel/random/boot_id").read_text().strip()]
    except (FileNotFoundError, ProcessLookupError):
        pass
    return None


def gpu_status():
    def query(option):
        return subprocess.run(["nvidia-smi", option, "--format=csv,noheader,nounits"],
                              check=True, capture_output=True, text=True, timeout=10).stdout.strip()

    try:
        devices = query("--query-gpu=index,uuid,memory.free,utilization.gpu").splitlines()
        if len(devices) != 1:
            return False, f"Ожидался один контролируемый GPU, найдено {len(devices)}"
        index, uuid, free, load = [part.strip() for part in devices[0].split(",")]
        computes = query("--query-compute-apps=pid,gpu_uuid")
        if computes:
            return False, f"GPU {uuid}, {free} MiB свободно, compute {computes.replace(chr(10), ' | ')}"
        # Catch trainers before their CUDA context is visible, including previously queued jobs.
        processes = subprocess.check_output(["ps", "-ww", "-eo", "pid=,args="], text=True)
        for row in processes.splitlines():
            pid, _, command = row.strip().partition(" ")
            args = command.split()
            if not args or "python" not in Path(args[0]).name or not identity(int(pid)):
                continue
            if any(Path(arg).name in {"maze_plr.py", "maze_dr.py", "maze_accel_baseline.py",
                                      "maze_plr_resume_s0.py", "queue_accel_s2.py"} for arg in args[1:]):
                return False, f"Текущий или ожидающий trainer PID {pid}, GPU {uuid}, {free} MiB свободно"
        if int(free) < 5000 or int(load) > 25:
            return False, f"GPU {uuid}, {free} MiB свободно, GPU util {load}%"
        return True, f"GPU {uuid}, {free} MiB свободно, GPU util {load}%"
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return False, f"Проверка nvidia-smi недоступна: {error}"


def checkpoint_dir(run_name, seed):
    return ROOT / "checkpoints" / run_name / str(seed)


def command(state, task, attempt):
    interval = SMOKE if task["target"] == SMOKE else 250
    lite = task.get("study") == "lite"
    name = f"{'pace-lite' if lite else 'pace-v2'}-{task['id'].lower()}-a{attempt}"
    args = [sys.executable, str((LITE_SOURCE if lite else SOURCE) / "examples" / "maze_plr.py"), "--run_name", name, "--use_accel",
            "--seed", str(task["seed"]), "--score_function", task["method"], "--num_updates", str(HORIZON),
            "--stop_after_updates", str(task["target"]), "--eval_freq", str(interval),
            "--pace_validation_freq", str(SMOKE if interval == SMOKE else 1000),
            "--pace_validation_file", str(VALIDATION), "--checkpoint_save_interval", "4"]
    for key, value in state["baseline"].items():
        args.extend((f"--{key}", str(value)))
    if (task["method"].startswith("pace_") and task["method"] != "pace_pg") or lite:
        for key, value in (("pace_quantile", task["q"]), ("pace_floor_weight", task["floor"]),
                           ("pace_threshold_ema", task["ema"]), ("pace_shape", task["shape"]),
                           ("hybrid_pace_weight", task["alpha"]), ("pace_warmup_fraction", 0.)):
            args.extend((f"--{key}", str(value)))
    if task["method"] == "pace_pg":
        args.append("--pace_log_diagnostics")
    if task["parent"]:
        parent = next(other for other in state["jobs"] if other["id"] == task["parent"])
        result = parent["attempts"][-1]["result"]
        args.extend(("--resume_checkpoint_directory", result["directory"],
                     "--resume_checkpoint_step", str(result["step"])))
    return name, args, task["target"] // interval - 1


def launch(state, task):
    attempt = len(task["attempts"]) + 1
    name, args, step = command(state, task, attempt)
    report = LITE_REPORT if task.get("study") == "lite" else REPORT
    folder = report / "runs" / task["id"] / f"attempt-{attempt}"
    folder.mkdir(parents=True, exist_ok=True)
    spec = dict(args=args, run_name=name, directory=str(checkpoint_dir(name, task["seed"])),
                step=step, target=task["target"], seed=task["seed"], method=task["method"], created=now())
    if task["target"] == HORIZON:
        spec["official"] = [sys.executable, args[1], "--mode", "eval", "--checkpoint_directory", spec["directory"],
                            "--checkpoint_to_eval", str(step), "--eval_num_attempts", "10",
                            "--eval_levels", *EXPERT_LEVELS, "--pace_eval_output", str(folder / "official")]
        # Retry only official evaluation once a full, validated checkpoint exists.
        for previous in reversed(task["attempts"]):
            old = Path(previous["folder"])
            if (old / "training_result.json").is_file():
                spec["official"][5] = read_json(old / "training_result.json")["directory"]
                spec["reuse_training"] = str(old / "training_result.json")
                break
    if task["parent"]:
        parent = next(item for item in state["jobs"] if item["id"] == task["parent"])
        spec["prefix_wall_clock"] = current_metric(parent)["wall_clock_time"]
    if task["id"] == "LS00":
        control = next(item for item in state["jobs"] if item["id"] == "B00")["attempts"][-1]["result"]
        model = Path(control["directory"]) / "models" / str(control["step"]) / "default"
        spec["profile"] = [sys.executable, str(LITE_SOURCE / "examples" / "pace_lite_profile.py"),
                           "--checkpoint", str(model), "--validation", str(VALIDATION),
                           "--output", str(folder / "profiling.json")]
    write_json(folder / "spec.json", spec)
    task["attempts"].append({"folder": str(folder), "run_name": name})
    task["status"] = "RUNNING"
    write_json(STATE, state)
    event(state, f"Старт {task['id']}, попытка {attempt}.")
    spawn_worker(folder)


def spawn_worker(folder):
    with (folder / "worker.log").open("ab") as log:
        subprocess.Popen([sys.executable, __file__, "--worker", str(folder)], cwd=ROOT,
                         stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)


def train_result(spec):
    import numpy as np

    directory = Path(spec["directory"])
    rows = read_rows(directory / "teacher_metrics.jsonl")
    validation = [row for row in rows if "eval/validation_solve_rate" in row]
    if not validation or validation[-1]["num_updates"] != spec["target"]:
        raise RuntimeError(f"missing validation at round {spec['target']}")
    row = validation[-1]
    if row["num_env_steps"] != spec["target"] * 8192 or row["gradient_updates"] <= 0:
        raise RuntimeError("unexpected training or optimizer budget")
    checkpoint = directory / "models" / str(spec["step"]) / "default"
    metadata = checkpoint / "_METADATA"
    if not metadata.is_file() or not (checkpoint / "manifest.ocdbt").is_file():
        raise RuntimeError("missing final Orbax checkpoint")
    keys = {tuple(item["key"] for item in node["key_metadata"])
            for node in read_json(metadata)["tree_metadata"].values()}
    if not {"runner_rng", "sampler", "opt_state", "params"} <= {key[0] for key in keys}:
        raise RuntimeError("final checkpoint lacks full training state")
    if spec.get("method", "MaxMC") not in ("MaxMC", "pace_pg") and ("sampler", "pace_threshold") not in keys:
        raise RuntimeError("final checkpoint lacks the adaptive threshold")
    with np.load(directory / f"validation-{spec['target']}.npz", allow_pickle=False) as data:
        returns = data["returns"]
        if returns.shape != (4, 128) or not np.isfinite(returns).all() or (
            (returns > 0).mean() != row["eval/validation_solve_rate"]
        ):
            raise RuntimeError("raw validation returns do not match the final metric")
    if not np.isfinite(list(row.values())).all() or row["level_sampler/nonfinite_active_weights"]:
        raise RuntimeError("invalid final training metrics")
    row["eval/recent_solve_rate"] = sum(v["eval/validation_solve_rate"] for v in validation[-3:]) / min(len(validation), 3)
    row["trajectory_wall_clock_time"] = row["wall_clock_time"] + spec.get("prefix_wall_clock", 0)
    return {"metrics": row, "directory": str(directory), "step": spec["step"]}


def worker(folder):
    with (folder / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        result_path = folder / "outcome.json"
        if result_path.exists():
            return
        if (folder / "worker.json").exists():
            write_json(result_path, {"status": "FAILED", "exit_code": None, "error": "worker interrupted", "at": now()})
            return
        spec = read_json(folder / "spec.json")
        source = Path(spec["args"][1]).parents[1]
        env = {**os.environ, "JAX_PLATFORMS": "cuda,cpu", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
               "CUDA_VISIBLE_DEVICES": "0", "WANDB_MODE": "disabled",
               "PYTHONPATH": str(source / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
        start = time.monotonic()
        code = None

        def run(args, name):
            nonlocal code
            with (folder / f"{name}.stdout.log").open("ab") as out, (folder / f"{name}.stderr.log").open("ab") as err:
                process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=out, stderr=err)
                write_json(folder / "worker.json", {"worker": identity(os.getpid()), "pid": process.pid,
                                                    "process": identity(process.pid), "phase": name, "at": now()})
                code = process.wait()
            if code:
                last = (folder / f"{name}.stderr.log").read_text(encoding="utf-8", errors="replace").splitlines()
                raise RuntimeError(f"{name} exit {code}: " + " | ".join(last[-5:]))

        try:
            previous = spec.get("reuse_training")
            if previous:
                result = read_json(Path(previous))
            else:
                if Path(spec["directory"]).exists():
                    raise FileExistsError(f"refusing to overwrite {spec['directory']}")
                run(spec["args"], "train")
                result = train_result(spec)
            write_json(folder / "training_result.json", result)
            if "profile" in spec:
                try:
                    run(spec["profile"], "profile")
                except RuntimeError as error:
                    result["profile_error"] = str(error)
            if "official" in spec:
                run(spec["official"], "official")
                import numpy as np

                with np.load(folder / "official" / "results.npz", allow_pickle=False) as data:
                    returns = data["cum_rewards"]
                    if tuple(data["levels"]) != EXPERT_LEVELS or returns.shape != (10, 8):
                        raise RuntimeError("invalid expert evaluation output")
                    solve = (returns > 0).mean(axis=0)
                    result["official"] = {
                        "solve_rate": float(solve.mean()), "mean_return": float(returns.mean()),
                        "per_level": dict(zip(EXPERT_LEVELS, map(float, solve))),
                        "path": str(folder / "official" / "results.npz"),
                    }
            outcome = dict(status="COMPLETED", exit_code=0, result=result)
        except Exception as error:
            outcome = dict(status="FAILED", exit_code=code, error=str(error))
        write_json(result_path, {**outcome, "elapsed": time.monotonic() - start, "at": now()})


def refresh(state, task):
    attempt = task["attempts"][-1]
    folder = Path(attempt["folder"])
    outcome = folder / "outcome.json"
    record = folder / "worker.json"
    if record.exists():
        attempt.update(read_json(record))
    if outcome.exists():
        attempt.update(read_json(outcome))
        task["status"] = "PENDING" if attempt["status"] == "FAILED" and len(task["attempts"]) < 3 else attempt["status"]
        event(state, f"{task['id']} {attempt['status']}. {attempt.get('error', '')}")
        if task["status"] == "PENDING" and task["parent"] and "resume" in attempt.get("error", "").lower():
            task["parent"] = None
            event(state, f"{task['id']} повторится с нуля после ошибки восстановления.")
    elif record.exists() and not any(identity(token[0]) == token for token in
                                     (attempt.get("worker"), attempt.get("process")) if token):
        # The worker lock prevents a duplicate even if the runner restarts mid-launch.
        spawn_worker(folder)
    elif not record.exists() and time.time() - (folder / "spec.json").stat().st_mtime > 45:
        spawn_worker(folder)


def check_snapshot(state):
    for root, manifest in ((SOURCE, state["source"]), (LITE_SOURCE, state.get("lite_source", {}))):
        for file, expected in manifest.items():
            if hashlib.sha256((root / file).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"research snapshot changed: {root / file}")
    if hashlib.sha256(VALIDATION.read_bytes()).hexdigest() != state["validation"]["sha256"]:
        raise RuntimeError("validation snapshot changed")


def freeze_source(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"research source already exists: {destination}")
    staging = destination.parent / f".source-{os.getpid()}"
    shutil.copytree(ROOT / "src" / "jaxued", staging / "src" / "jaxued",
                    ignore=shutil.ignore_patterns("__pycache__", ".DS_Store", "*.pyc"))
    (staging / "examples").mkdir()
    shutil.copy2(ROOT / "examples" / "maze_plr.py", staging / "examples" / "maze_plr.py")
    if destination == LITE_SOURCE:
        shutil.copy2(ROOT / "examples" / "pace_lite_profile.py", staging / "examples" / "pace_lite_profile.py")
    hashes = {str(file.relative_to(staging)): hashlib.sha256(file.read_bytes()).hexdigest()
              for file in sorted(staging.rglob("*")) if file.is_file()}
    write_json(staging / "manifest.json", hashes)
    staging.rename(destination)


def freeze():
    import importlib.metadata

    from jaxued.pace_validation import generate

    if STATE.exists():
        raise FileExistsError("research queue already exists")
    if not VALIDATION.exists():
        generate(VALIDATION)
    freeze_source(SOURCE)
    packages = ("jax", "jaxlib", "flax", "chex", "optax", "orbax-checkpoint", "numpy", "wandb", "matplotlib")
    write_json(REPORT / "environment.json", {"python": sys.version.split()[0], "interpreter": sys.executable,
                                            **{name: importlib.metadata.version(name) for name in packages}})


def run(once=False):
    from pace_research_report import render

    with (REPORT / "queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Research queue is already running.")
            return
        state = read_json(STATE) if STATE.exists() else initialize()
        if (LITE_SOURCE / "manifest.json").is_file() and "lite_source" not in state:
            if state["winner"] or any(task["id"] in {"H01", "H02", "H03", "L01", "LS00"} for task in state["jobs"]):
                raise RuntimeError("Lite campaign IDs conflict with the existing queue")
            controls = {task["id"]: task for task in state["jobs"] if task["id"] in {"B00", "F03"}}
            for control_id in ("B00", "F03"):
                control = controls[control_id]
                result = control["attempts"][-1]["result"]
                saved = read_json(Path(result["directory"]) / "config.json")
                if (control["status"] != "COMPLETED" or result["metrics"]["num_updates"] != TARGET
                        or result["metrics"]["num_env_steps"] != TARGET * 8192 or saved["seed"] != 0
                        or saved["num_updates"] != HORIZON
                        or Path(saved["pace_validation_file"]).resolve() != VALIDATION
                        or any(saved[key] != expected for key, expected in state["baseline"].items())):
                    raise RuntimeError(f"Lite control is incompatible: {control_id}")
            state["lite_source"] = read_json(LITE_SOURCE / "manifest.json")
            state["stage"] = "lite_search"
            state["jobs"].append(lite_job("LS00", target=SMOKE, reason="CUDA smoke на полном размере rollout."))
            event(state, "Добавлена кампания H-PACE-Lite после уже поставленных задач.")
        check_snapshot(state)
        state["runner"] = {"pid": os.getpid(), "identity": identity(os.getpid()), "at": now()}
        available_checks = 0
        while True:
            if F03_FULL_REQUEST.is_file():
                requested_jobs = f03_full_jobs(state)
                if requested_jobs:
                    state["jobs"].extend(requested_jobs)
                    event(state, "Добавлены полные прогоны F03 на seeds 0, 1, 2.")
            active = next((task for task in state["jobs"] if task["status"] == "RUNNING"), None)
            if active:
                refresh(state, active)
                _, state["gpu"] = gpu_status()
            else:
                if any(task["id"] == "S00" and task["status"] == "FAILED" for task in state["jobs"]):
                    for task in state["jobs"]:
                        if task["status"] not in STATUS_TERMINAL:
                            task["status"] = "SKIPPED"
                    state["stage"] = "exhausted"
                review_floor(state)
                choose_next(state)
                manual = LITE_MANUAL.exists()
                task = next((task for task in state["jobs"] if task["status"] not in STATUS_TERMINAL
                             and not (manual and task.get("study") == "lite")), None)
                if task:
                    available, state["gpu"] = gpu_status()
                    available_checks = available_checks + 1 if available else 0
                    task["status"] = "WAITING_FOR_GPU"
                    if available_checks >= 2:
                        launch(state, task)
                        available_checks = 0
            state["updated"] = now()
            write_json(STATE, state)
            try:
                render(state)
                if "lite_source" in state:
                    lite = {**state, "jobs": [task for task in state["jobs"]
                                              if task.get("study") == "lite" or task["id"] in {"B00", "F03"}]}
                    write_json(LITE_REPORT / "experiment_queue.json", lite)
                    render(lite, report=LITE_REPORT)
            except (OSError, ValueError) as error:
                print(f"{now()} Не удалось обновить отчёт: {error}", flush=True)
            if once or all(task["status"] in STATUS_TERMINAL for task in state["jobs"]):
                # Another pass is needed after the last search task to select and enqueue FULL0.
                if not once and not state["winner"] and state["stage"] not in ("exhausted", "lite_exhausted"):
                    continue
                return
            time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--freeze-lite", action="store_true")
    parser.add_argument("--enqueue-f03-full", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.freeze:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        freeze()
    elif args.freeze_lite:
        freeze_source(LITE_SOURCE)
    elif args.enqueue_f03_full:
        request_f03_full()
    else:
        run(args.once)
