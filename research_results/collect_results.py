"""Rebuild the JaxUED 30k report and CSVs from local evaluation artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from urllib.parse import quote

import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
ROUNDS = 30_000
FINAL = 119
CHECKPOINTS = (0, 17, 34, 51, 68, 85, 102, FINAL)
LEVELS = (
    "SixteenRooms", "SixteenRooms2", "Labyrinth", "LabyrinthFlipped",
    "Labyrinth2", "StandardMaze", "StandardMaze2", "StandardMaze3",
)
METHODS = ("DR", "PLR⊥", "ACCEL", "PACE-ACCEL")
ORGANIZERS = {"DR": (58, 16), "PLR⊥": (44, 8), "ACCEL": (67, 12)}
REPORTED = {
    ("DR", 0): (31.25, 0.1951, 204.50), ("DR", 1): (61.25, 0.3759, 162.59),
    ("PLR⊥", 0): (47.50, 0.2347, 197.99), ("PLR⊥", 1): (42.50, 0.2072, 204.24),
    ("ACCEL", 0): (40.00, 0.1874, 209.06), ("ACCEL", 1): (52.50, 0.2708, 189.36),
    ("PACE-ACCEL", 0): (38.75, 0.1907, 207.80),
}
PPO = {
    "lr": 1e-4, "max_grad_norm": 0.5, "num_train_envs": 32, "num_steps": 256,
    "num_minibatches": 1, "gamma": 0.995, "epoch_ppo": 5, "clip_eps": 0.2,
    "gae_lambda": 0.98, "entropy_coeff": 1e-3, "critic_coeff": 0.5,
}
TEACHER = {
    "exploratory_grad_updates": False, "level_buffer_capacity": 4000,
    "replay_prob": 0.8, "staleness_coeff": 0.3, "temperature": 0.3,
    "minimum_fill_ratio": 0.5, "prioritization": "rank", "buffer_duplicate_check": True,
}
SUMMARY_FIELDS = (
    "algorithm", "seed", "run_name", "status", "num_updates", "environment_steps",
    "checkpoint", "solve_rate", "mean_return", "mean_episode_length", "solved_episodes",
    "total_episodes", "evaluation_type", "checkpoint_path", "results_path",
    "source_archive", "verification_status",
)
LEVEL_FIELDS = (
    "algorithm", "seed", "run_name", "level", "solve_rate", "mean_return",
    "mean_episode_length", "solved_episodes", "total_episodes",
)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def link(path: Path, label: str | None = None) -> str:
    rel = relative(path)
    return f"[{label or path.name}](../{quote(rel, safe='/')})"


def read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def model_fingerprint(path: Path) -> str:
    """Bind a recorded Orbax load to the current contents of the checkpoint."""
    signature = hashlib.sha256()
    for leaf in sorted(path.rglob("*")):
        if leaf.is_file():
            signature.update(leaf.relative_to(path).as_posix().encode())
            signature.update(b"\0")
            with leaf.open("rb") as handle:
                signature.update(hashlib.file_digest(handle, "sha256").digest())
    return signature.hexdigest()


def read_audit():
    archive_for = {}
    wandb = defaultdict(list)
    orbax = {}
    resume = {}
    path = OUT / "import_log.txt"
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t", 3)
                if len(fields) != 4:
                    continue
                kind, archive, member, detail = fields
                if kind in {"COPIED", "IDENTICAL"}:
                    destination = detail.partition(" bytes=")[0]
                    if destination.endswith(("/config.json", "/results.npz")):
                        archive_for.setdefault(destination, archive)
                elif kind == "WANDB":
                    wandb[member].append(json.loads(detail))
                elif kind in {"ORBAX_OK", "ORBAX_ERROR"}:
                    orbax[member] = (kind, json.loads(detail))
                elif kind == "RESUME":
                    resume[member] = json.loads(detail)
    return archive_for, wandb, orbax, resume


def method_for(name: str, cfg: dict) -> str | None:
    if re.fullmatch(r"dr-s\d+(?:-.+)?", name) and "score_function" not in cfg:
        return "DR"
    if cfg.get("use_accel") and cfg.get("score_function") == "pace_pg":
        return "PACE-ACCEL"
    if cfg.get("score_function") == "MaxMC" and cfg.get("exploratory_grad_updates") is False:
        if name.startswith("accel-") and cfg.get("use_accel") is True:
            return "ACCEL"
        if name.startswith("plr-") and cfg.get("use_accel") is False:
            return "PLR⊥"
    return None


@dataclass
class Evaluation:
    path: Path
    rewards: np.ndarray
    lengths: np.ndarray

    def metrics(self, index=slice(None)):
        rewards = self.rewards[:, index]
        lengths = self.lengths[:, index]
        solved = int((rewards > 0).sum())
        return dict(
            solve_rate=solved / rewards.size,
            mean_return=float(rewards.mean()),
            mean_episode_length=float(lengths.mean()),
            solved_episodes=solved,
            total_episodes=int(rewards.size),
        )


@dataclass
class Run:
    algorithm: str
    seed: int
    name: str
    directory: Path
    config: dict
    indices: tuple[int, ...]
    updates: int
    archive: str = ""
    evaluation: Evaluation | None = None
    orbax: str = "not_tested"
    optimizer_steps: int | None = None
    rng_persisted: bool | None = None
    wandb: list[dict] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    parent: str | None = None
    first_index: int | None = None

    @property
    def complete(self):
        return FINAL in self.indices and self.updates == ROUNDS

    @property
    def status(self):
        if self.complete:
            return "completed_evaluated" if self.evaluation else "completed_no_evaluation"
        return "incomplete" if self.updates else "no_checkpoint_or_rounds"

    @property
    def checkpoint_index(self):
        return max(self.indices, default=None)

    @property
    def checkpoint(self):
        if self.checkpoint_index is None:
            return None
        return self.directory / "models" / str(self.checkpoint_index)


def read_progress(directory: Path, name: str, issues: list[str]) -> int:
    observed = 0
    path = directory / "teacher_metrics.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                issues.append(f"Неполная строка в {relative(path)}.")
                continue
            observed = max(observed, int(row.get("num_updates", 0)))
    console = ROOT / "logs" / f"{name}.log"
    if console.is_file():
        rounds = re.findall(r"Logging update:\s*(\d+)", console.read_text(encoding="utf-8", errors="replace"))
        observed = max(observed, max(map(int, rounds), default=0))
    return observed


def load_runs(archive_for, wandb, orbax, resume, issues):
    runs = []
    for config_path in sorted((ROOT / "checkpoints").glob("*/*/config.json")):
        try:
            cfg = read_config(config_path)
        except (OSError, ValueError) as error:
            issues.append(f"{relative(config_path)} — {error}")
            continue
        name = config_path.parent.parent.name
        if name.endswith("-standard-eval") or cfg.get("mode") != "train" or cfg.get("num_updates") != ROUNDS:
            continue
        algorithm = method_for(name, cfg)
        if algorithm is None:
            continue
        seed = cfg.get("seed")
        if str(seed) != config_path.parent.name or cfg.get("run_name") != name:
            issues.append(f"{relative(config_path)} — имя или seed расходятся с каталогом.")
            continue

        directory = config_path.parent
        indices = tuple(sorted(
            int(path.name) for path in (directory / "models").glob("*")
            if path.name.isdecimal() and (path / "default" / "_METADATA").is_file()
        ))
        declared_interval = int(cfg.get("eval_freq", 0))
        observed = read_progress(directory, name, issues)
        matching_wandb = list(wandb.get(name, []))
        if not matching_wandb:
            log = ROOT / "logs" / f"{name}.log"
            if log.is_file():
                for folder in dict.fromkeys(re.findall(
                    r"offline-run-\d{8}_\d{6}-[a-z0-9]+", log.read_text(encoding="utf-8", errors="replace")
                )):
                    run_id = folder.rsplit("-", 1)[-1]
                    path = ROOT / "wandb" / folder / f"run-{run_id}.wandb"
                    if path.is_file():
                        matching_wandb.append(dict(path=relative(path), state="binary_present", rounds=observed))
        updates = max(
            observed, (max(indices, default=-1) + 1) * declared_interval,
            *(int(row.get("rounds") or 0) for row in matching_wandb),
        )
        run = Run(
            algorithm, seed, name, directory, cfg, indices, updates,
            archive=archive_for.get(relative(config_path), ""),
            wandb=matching_wandb,
        )
        differences = [key for key, value in PPO.items() if cfg.get(key) != value]
        if algorithm != "DR":
            expected = dict(TEACHER)
            if cfg.get("use_accel"):
                expected["num_edits"] = 5
            differences += [key for key, value in expected.items() if cfg.get(key) != value]
            for key, accepted in (
                ("mutation_selection", {None, "fixed"}), ("qd_mode", {None, "off"}),
                ("adaptive_mutation", {None, False}), ("dual_score_tracking", {None, False}),
                ("pace_shadow", {None, False}), ("replay_max_copies", {None, 0}),
            ):
                if cfg.get(key) not in accepted:
                    differences.append(key)
        if algorithm == "PACE-ACCEL":
            if cfg.get("pace_score_chunk_size") != 1 or cfg.get("pace_score_entropy") is not False:
                differences.append("настройки PACE-PG")
        differences += [key for key, value in {"eval_freq": 250, "checkpoint_save_interval": 17}.items()
                        if cfg.get(key) != value]
        if differences:
            run.issues.append("Отличия от протокола — " + ", ".join(differences) + ".")
        if run.complete and indices != CHECKPOINTS:
            run.issues.append("Индексы сохранённых чекпойнтов отличаются от полного ряда.")
        if run.checkpoint:
            key = relative(run.checkpoint / "default")
            if key in orbax:
                kind, data = orbax[key]
                if kind == "ORBAX_ERROR":
                    run.orbax = "load_error"
                    run.issues.append(f"Orbax: {data['error']}")
                elif model_fingerprint(run.checkpoint / "default") == data["sha256"]:
                    run.orbax = "restored"
                    run.optimizer_steps = data["optimizer_steps"]
                    run.rng_persisted = data["rng_persisted"]
                else:
                    run.issues.append("Файлы чекпойнта изменились после проверки Orbax.")
        resume_data = resume.get(name, {})
        saved_directory = cfg.get("resume_checkpoint_directory")
        if saved_directory:
            run.parent = Path(saved_directory.replace("\\", "/")).parent.name
            run.first_index = cfg.get("resume_checkpoint_step")
        elif resume_data:
            run.parent = resume_data.get("parent")
            run.first_index = resume_data.get("index")
        runs.append(run)
    return runs


def attach_evaluations(runs, archive_for, aliases, issues):
    by_id = {(run.name, run.seed): run for run in runs}
    candidates = defaultdict(list)
    for path in sorted((ROOT / "results").rglob("results.npz")):
        if not path.parent.name.isdecimal():
            continue
        seed = int(path.parent.name)
        name = path.parent.parent.name
        # Evaluation aliases have their own config on WSL, or ship with a canonical "-standard-eval" name.
        name_from_folder = name.removesuffix("-standard-eval")
        eval_cfg = aliases.get((name, seed))
        run_name = eval_cfg.get("run_name", name_from_folder) if eval_cfg else name_from_folder
        run = by_id.get((run_name, seed))
        if not run:
            if name.endswith("-standard-eval"):
                issues.append(f"{relative(path)} — соответствующий конфиг обучения отсутствует.")
            continue
        if not run.complete:
            issues.append(f"{relative(path)} — финальный чекпойнт {FINAL} не подтверждён.")
            continue
        if eval_cfg and (eval_cfg.get("checkpoint_to_eval") not in (None, -1, FINAL)
                         or eval_cfg.get("seed") != seed
                         or tuple(eval_cfg.get("eval_levels", LEVELS)) != LEVELS):
            issues.append(f"{relative(path)} — параметры evaluator расходятся с протоколом.")
            continue
        if not eval_cfg and not name.endswith("-standard-eval"):
            issues.append(f"{relative(path)} — происхождение независимой оценки не подтверждено.")
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                labels = tuple(data["levels"])
                rewards, lengths = data["cum_rewards"], data["episode_lengths"]
                if labels != LEVELS or rewards.shape != (10, 8) or lengths.shape != (10, 8):
                    raise ValueError(f"карты или формы массивов расходятся с протоколом ({rewards.shape}, {lengths.shape})")
                if not np.isfinite(rewards).all() or not np.isfinite(lengths).all():
                    raise ValueError("массивы содержат NaN или Inf")
        except (KeyError, ValueError, OSError) as error:
            issues.append(f"{relative(path)} — {error}")
            continue
        candidates[run_name, seed].append(Evaluation(path, rewards, lengths))
    for key, items in candidates.items():
        selected = items[0]
        if any(not (np.array_equal(selected.rewards, item.rewards)
                    and np.array_equal(selected.lengths, item.lengths)) for item in items[1:]):
            issues.append(f"{key[0]} — обнаружены разные оценки: " + ", ".join(relative(item.path) for item in items))
            continue
        run = by_id[key]
        run.evaluation = selected
        run.archive = archive_for.get(relative(selected.path), run.archive)


def write_csv(name: str, fields, rows):
    path = OUT / name
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_row(run: Run) -> dict:
    cfg = run.config
    result = dict(
        algorithm=run.algorithm, seed=run.seed, run_name=run.name, status=run.status,
        num_updates=run.updates, environment_steps=run.updates * cfg["num_train_envs"] * cfg["num_steps"],
        checkpoint=run.checkpoint_index, checkpoint_path=relative(run.checkpoint) if run.checkpoint else "",
        results_path=relative(run.evaluation.path) if run.evaluation else "", source_archive=run.archive,
        evaluation_type="zero_shot_expert" if run.evaluation else "",
        verification_status=("npz_verified" if run.evaluation else "no_verified_npz") + ";orbax_" + run.orbax,
    )
    if run.evaluation:
        result.update(run.evaluation.metrics())
    return result


def select_per_seed(runs, issues):
    groups = defaultdict(list)
    for run in runs:
        groups[run.algorithm, run.seed].append(run)
    for key, items in groups.items():
        ranked = sorted(items, key=lambda r: (bool(r.evaluation), r.complete, r.updates), reverse=True)
        best = ranked[0]
        if sum(bool(r.evaluation) for r in ranked) > 1:
            issues.append(f"{key[0]}, seed {key[1]} — несколько независимых валидных оценок, агрегат пропущен.")
            continue
        yield key, best


def percentage(x):
    return f"{x * 100:.2f}%".replace(".", ",")


def number(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def signed_pp(x):
    return f"{x:+.2f}".replace(".", ",") + " п.п."


def duration(seconds):
    hours, remainder = divmod(round(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours} ч {minutes:02d} мин {seconds:02d} с"


def integers(x):
    return f"{x:,}".replace(",", " ")


def table(headers, rows):
    yield "| " + " | ".join(headers) + " |"
    yield "| " + " | ".join("---" for _ in headers) + " |"
    for row in rows:
        yield "| " + " | ".join(str(cell).replace("|", r"\|") for cell in row) + " |"


def report(runs, selected, attempts, resume, audit_issues):
    lines = [
        "# Результаты экспериментов JaxUED",
        "",
        "Бюджет полного прогона — 30 000 раундов × 32 среды × 256 шагов, или 245 760 000 шагов среды. "
        "Zero-shot оценка использует финальный чекпойнт 119 и по 10 попыток на восьми экспертных картах. "
        "Успех определяется как cum_rewards > 0.",
        "",
        "## Zero-shot mean solve rate",
        "",
    ]
    score_rows = []
    pending = []
    comparison = []
    discrepancies = []
    level_reference = {
        "DR": (100, 90, 0, 0, 0, 0, 0, 60),
        "PLR⊥": (100, 40, 100, 20, 40, 0, 80, 0),
        "ACCEL": (100, 80, 20, 0, 90, 0, 30, 0),
        "PACE-ACCEL": (60, 0, 90, 30, 80, 0, 0, 50),
    }
    for algorithm in METHODS:
        scores = []
        seed_cells = []
        for seed in range(3):
            key = algorithm, seed
            run = selected.get(key)
            prior = REPORTED.get(key)
            if run and run.evaluation:
                metrics = run.evaluation.metrics()
                score = metrics["solve_rate"]
                scores.append(score)
                seed_cells.append(percentage(score))
                actual = (round(score * 100, 2), round(metrics["mean_return"], 4),
                          round(metrics["mean_episode_length"], 2))
                if prior is not None and prior != actual:
                    discrepancies.append(f"{algorithm}, seed {seed}: ранее {prior}, по NPZ {actual}.")
                if seed == 0:
                    individual = tuple(round(run.evaluation.metrics(i)["solve_rate"] * 100)
                                       for i in range(len(LEVELS)))
                    if individual != level_reference[algorithm]:
                        discrepancies.append(f"{algorithm}, seed 0: solve rate по картам "
                                             f"{level_reference[algorithm]} ранее, {individual} по NPZ.")
            else:
                seed_cells.append(f"{number(prior[0])}%¹" if prior else "—")
                if run and run.complete:
                    next_step = "zero-shot оценка чекпойнта 119"
                elif run and run.updates:
                    next_step = f"полный бюджет и zero-shot оценка (последнее наблюдение {integers(run.updates)} раундов)"
                else:
                    next_step = "полное обучение и zero-shot оценка"
                pending.append(f"- **{algorithm}, seed {seed}** — {next_step}.")
        aggregate = f"{percentage(mean(scores))} ± {number(stdev(scores) * 100)} п.п." if len(scores) > 1 else (
            f"{percentage(scores[0])} (один seed)" if scores else "—"
        )
        score_rows.append((algorithm, *seed_cells, aggregate, len(scores)))
        if algorithm in ORGANIZERS:
            benchmark, deviation = ORGANIZERS[algorithm]
            difference = signed_pp(mean(scores) * 100 - benchmark) if scores else "—"
            comparison.append((algorithm, f"{benchmark}% ± {deviation} п.п.", difference))
    lines.extend(table(("Алгоритм", "Seed 0", "Seed 1", "Seed 2", "Mean ± Std", "n"), score_rows))
    lines.extend([
        "",
        "Mean и выборочное Std (ddof=1) рассчитаны только по подтверждённым оценкам завершённых прогонов. "
        "При одном seed приведено единичное значение. Тире обозначает отсутствие результата. "
        "¹ Ранее зарегистрированное значение без повторного подтверждения из NPZ, в агрегат не включено.",
        "",
        "## Ориентиры Т-Банка",
        "",
    ])
    lines.extend(table(("Алгоритм", "Т-Банк (n=3)", "Наше среднее минус ориентир"), comparison))
    spec = ROOT / "tbank_reference/task_spec/TASK_SPEC.md"
    lines.extend([
        "",
        "Разность приведена описательно. Число seeds в наших прогонах см. выше. "
        f"{link(spec, 'ТЗ и ориентиры организаторов')}",
        "",
        "## Метрики и артефакты по прогонам",
        "",
    ])
    rows = []
    for run in runs:
        data = run.evaluation.metrics() if run.evaluation else None
        ckpt = link(run.checkpoint, str(run.checkpoint_index)) if run.checkpoint else "—"
        source = link(run.evaluation.path, "NPZ") if run.evaluation else "—"
        if data:
            values = (percentage(data["solve_rate"]), number(data["mean_return"], 4),
                      number(data["mean_episode_length"]), f"{data['solved_episodes']}/{data['total_episodes']}")
        else:
            values = ("—", "—", "—", "—")
        state = {
            "completed_evaluated": "Обучение и оценка завершены",
            "completed_no_evaluation": "Обучение завершено",
            "incomplete": "Неполный бюджет",
            "no_checkpoint_or_rounds": "Раунды не подтверждены",
        }[run.status]
        rows.append((run.algorithm, run.seed, link(run.directory / "config.json", run.name), state,
                     integers(run.updates), ckpt, *values, source))
    lines.extend(table(("Алгоритм", "Seed", "Прогон", "Статус", "Раунды", "Чекпойнт", "Solve rate",
                        "Mean return", "Mean length", "Успехи", "Источник"), rows))
    lines.extend(["", "## Проверка конфигов, моделей и W&B", ""])
    rows = []
    for run in runs:
        orbax = {"restored": "Загружен, параметры конечны", "load_error": "Ошибка загрузки",
                 "not_tested": "Загрузка не проверялась"}[run.orbax]
        monitor = "экспертные" if tuple(run.config.get("eval_levels", ())) == LEVELS else "синтетические или другие"
        wandb_links = []
        for entry in run.wandb:
            path = ROOT / entry["path"]
            if path.is_file():
                label = path.stem
                if entry.get("mode") == "eval":
                    label += " (eval)"
                elif entry.get("rounds"):
                    label += f" ({integers(entry['rounds'])})"
                wandb_links.append(link(path, label))
        elapsed = next((row["runtime"] for row in reversed(run.wandb)
                        if row.get("mode") == "train" and row.get("runtime") is not None), None)
        rows.append((run.name, monitor, ", ".join(map(str, run.indices)) or "—", orbax,
                     integers(run.optimizer_steps) if run.optimizer_steps is not None else "—",
                     duration(elapsed) if elapsed is not None else "—",
                     ", ".join(dict.fromkeys(wandb_links)) or "—",
                     " ".join(run.issues) or "PPO и настройки Teacher совпадают с проверяемыми полями протокола"))
    lines.extend(table(("Прогон", "Карты мониторинга", "Индексы Orbax", "Проверка Orbax",
                        "Шаги Adam", "Время сегмента W&B", "W&B", "Конфигурация"), rows))
    lines.extend([
        "",
        "Проверка Orbax загружает сохранённые веса и проверяет их конечность. "
        "Отметка «Загружен» привязана к SHA-256 содержимого модели. Время W&B относится "
        "к указанному сегменту, у возобновлённых прогонов оно не включает предшествующее обучение.",
        "",
        "Конфиги проверены по общим настройкам Student и PPO, бюджету, частоте evaluation "
        "и сохранения, режиму ACCEL, regret-score, replay и мутациям. Мониторинг на экспертном "
        "наборе отмечен отдельно. Артефакты не позволяют установить, использовались ли результаты "
        "этого мониторинга при выборе гиперпараметров.",
        "",
        "## Возобновлённые обучения и попытки",
        "",
    ])
    for run in runs:
        note = resume.get(run.name)
        if note:
            lines.append(f"- **{run.name}** — {note['note']}")
        elif run.parent:
            index = "последний доступный" if run.first_index in (None, -1) else str(run.first_index)
            lines.append(f"- **{run.name}** — конфиг указывает на возобновление из {run.parent}, чекпойнт {index}.")
    for run in attempts:
        mode = "исходный сегмент возобновлённого обучения" if run.updates else "раунды не подтверждены"
        links = ", ".join(link(ROOT / item["path"]) for item in run.wandb if (ROOT / item["path"]).is_file())
        lines.append(f"- **{run.name}** — {mode}, {integers(run.updates)} наблюдаемых раундов"
                     + (f". {links}" if links else "."))
    lines.extend(["", "## Solve rate по лабиринтам", ""])
    evaluated = [run for run in runs if run.evaluation]
    for seed in sorted({run.seed for run in evaluated}):
        group = [run for run in evaluated if run.seed == seed and selected.get((run.algorithm, seed)) is run]
        if not group:
            continue
        lines.extend([f"### Seed {seed}", ""])
        lines.extend(table(("Лабиринт", *(run.algorithm for run in group)), (
            (name, *(percentage(run.evaluation.metrics(i)["solve_rate"]) for run in group))
            for i, name in enumerate(LEVELS)
        )))
        lines.append("")
    lines.extend(["[Return, длина эпизода и число успехов по каждой карте](results_per_level.csv)", "",
                  "## Предварительные выводы", ""])
    for seed in range(3):
        pace = selected.get(("PACE-ACCEL", seed))
        if not pace or not pace.evaluation:
            continue
        pace_score = pace.evaluation.metrics()["solve_rate"]
        comparison = []
        for method in METHODS[:3]:
            run = selected.get((method, seed))
            if run and run.evaluation:
                difference = (pace_score - run.evaluation.metrics()["solve_rate"]) * 100
                comparison.append(f"{method} {signed_pp(difference)}")
        if comparison:
            lines.append(f"- PACE-ACCEL на seed {seed} — {percentage(pace_score)}. Разница на том же seed — "
                         + ", ".join(comparison))
    pace_seed0 = selected.get(("PACE-ACCEL", 0))
    if pace_seed0 and pace_seed0.name == "pace-accel-30k-s0-20261125-160021":
        lines.append("- Synthetic Monitor Solve Rate для PACE-ACCEL seed 0 — 76,67% по шести "
                     "синтетическим картам. Эта метрика исключена из zero-shot агрегата. "
                     "[Журнал экспериментов](../EXPERIMENT_LOG.md)")
    lines.append("- Сравнение PACE-ACCEL с baseline пока ограничено общими seeds с готовой zero-shot оценкой. "
                 "Возобновлённые legacy-прогоны PLR⊥ нельзя считать побитово идентичными непрерывному обучению.")
    lines.extend(["", "## Оставшиеся эксперименты", "", *pending, "",
                  "## Сверка с ранее зарегистрированными значениями", ""])
    if discrepancies:
        lines.extend("- " + item for item in discrepancies)
    else:
        lines.append("Пересчитанные метрики совпали с ранее зарегистрированными значениями в доступных случаях, "
                     "с учётом точности их записи.")
    if audit_issues:
        lines.extend(["", "## Замечания при сборе", "", *("- " + x for x in audit_issues)])
    lines.extend([
        "", "## Обновление сводки", "",
        "Команда из корня репозитория в активной .venv", "",
        "~~~bash", "python research_results/collect_results.py", "~~~", "",
        "[Реестр архивов](archive_registry.md) · [Журнал импорта и проверок](import_log.txt)", "",
    ])
    return "\n".join(lines)


def main():
    archive_for, wandb, orbax, resume = read_audit()
    issues = []
    runs = load_runs(archive_for, wandb, orbax, resume, issues)
    aliases = {}
    for path in sorted((ROOT / "checkpoints").glob("*/*/config.json")):
        if not path.parent.name.isdecimal():
            continue
        try:
            cfg = read_config(path)
        except (OSError, ValueError):
            continue
        if cfg.get("mode") == "eval" or path.parent.parent.name.endswith("-standard-eval"):
            aliases[path.parent.parent.name, int(path.parent.name)] = cfg
    attach_evaluations(runs, archive_for, aliases, issues)
    active_names = {run.parent for run in runs if run.parent}
    attempts = [run for run in runs if run.name in active_names or not run.updates]
    attempt_names = {run.name for run in attempts}
    runs = sorted((r for r in runs if r.name not in attempt_names), key=lambda r: (METHODS.index(r.algorithm), r.seed, r.name))
    selected = dict(select_per_seed(runs, issues))
    per_level = []
    for run in runs:
        if run.evaluation:
            for index, level in enumerate(LEVELS):
                per_level.append(dict(algorithm=run.algorithm, seed=run.seed, run_name=run.name, level=level,
                                      **run.evaluation.metrics(index)))
    write_csv("results_summary.csv", SUMMARY_FIELDS, map(run_row, runs))
    write_csv("results_per_level.csv", LEVEL_FIELDS, per_level)
    (OUT / "RESULTS.md").write_text(report(runs, selected, attempts, resume, issues), encoding="utf-8", newline="\n")
    print(f"{sum(bool(r.evaluation) for r in runs)} verified zero-shot evaluations | {len(runs)} independent runs | {len(per_level)} per-level rows")


if __name__ == "__main__":
    main()
