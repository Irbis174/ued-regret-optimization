"""Write live research reports and draw figures from actual measurements."""

import csv
import io
import json
from pathlib import Path
import shlex

from pace_research_queue import (EXPERT_LEVELS, HORIZON, LITE_REPORT, REPORT, ROOT, TARGET, atomic,
                                 checkpoint_dir, command, current_metric, ranked, read_rows, write_json)


def percentage(rate):
    return "—" if rate is None else f"{100 * rate:.2f}%"


def render(state, report=REPORT):
    jobs = state["jobs"]
    lite = report == LITE_REPORT
    observed = {}
    candidates = ranked(job for job in jobs if job["target"] == TARGET and job["id"][0] in "FHL")
    standalone = next((job for job in candidates if job["id"].startswith(("F", "L"))), None)
    hybrid = next((job for job in candidates if job["id"].startswith("H")), None)
    base = next(job for job in jobs if job["id"] == "B00")
    original_dir = ROOT / "checkpoints" / "plr-s0" / "2"
    original = read_rows(original_dir / "teacher_metrics.jsonl")
    original_round = original[-1]["num_updates"] if original else "—"
    original_final = (original_dir / "models" / "119" / "default" / "_METADATA").is_file()
    progress = [
        "# H-PACE-Lite · ход экспериментов" if lite else "# PACE v2 · ход экспериментов", "",
        f"Обновлено {state['updated']}.",
        f"Исходный PLR⊥ seed 2, PID 14795 при аудите. Последняя метрика **{original_round} / 30 000**. "
        f"Финальный checkpoint {'найден' if original_final else 'ожидается'}.",
        f"Runner PID **{state['runner']['pid']}**. {state.get('gpu', '')}",
        f"Лучший самостоятельный **{standalone['id'] if standalone else '—'}**. "
        f"Лучший гибрид **{hybrid['id'] if hybrid else '—'}**.",
        f"ACCEL на том же validation **{percentage((current_metric(base) or {}).get('eval/validation_solve_rate'))}**.",
        f"Победитель **{state['winner'] or 'ещё не выбран'}**. {state.get('selection_reason', '')}", "",
        "| Задача | Состояние | Последний раунд | Validation | PID |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    ranking = [
        "# Рейтинг на 5 000 раундах", "",
        "Порядок определяет конечная validation solve rate. При равенстве учитываются сложные уровни, "
        "среднее трёх последних оценок и wall time. Основной поиск использует seed 0.", "",
        "| ID | Teacher | q | Floor | Вес Lite | Validation | Hard | Time |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for job in candidates + ([base] if current_metric(base) else []):
        metric = current_metric(job)
        q, floor = (job["q"], job["floor"]) if job["id"] != "B00" else ("—", "—")
        alpha = job["alpha"] if job.get("study") == "lite" else (0 if job["id"] == "B00" else "—")
        ranking.append(f"| {job['id']} | {job['method']} | {q} | {floor} | {alpha} | "
                       f"{percentage(metric['eval/validation_solve_rate'])} | "
                       f"{percentage(metric['eval/validation_hard_solve_rate'])} | {metric['wall_clock_time'] / 60:.1f} мин |")
    if not candidates:
        ranking.extend(("", "Пока нет подтверждённых результатов кандидатов на 5 000 раундах."))

    out = io.StringIO()
    fields = (
        "id", "run_name", "algorithm", "seed", "configuration", "git_commit", "num_updates", "environment_steps",
        "gradient_updates", "wall_clock_time", "status", "validation_solve_rate", "validation_hard_solve_rate",
        "train_solve_rate", "mean_episode_return", "checkpoint_path", "error_message", "official_solve_rate",
    )
    writer = csv.DictWriter(out, fields)
    writer.writeheader()
    commands = []
    for job in jobs:
        cached = []
        for number, attempt in enumerate(job["attempts"] or [{}], 1):
            rows, result = [], {}
            if attempt:
                spec = json.loads((Path(attempt["folder"]) / "spec.json").read_text(encoding="utf-8"))
                name, args = spec["run_name"], spec["args"]
                rows = read_rows(checkpoint_dir(name, job["seed"]) / "teacher_metrics.jsonl")
                result = attempt.get("result", {})
            else:
                name, args, _ = command(state, job, number)
            row = rows[-1] if rows else result.get("metrics", {})
            val = next((r for r in reversed(rows) if "eval/validation_solve_rate" in r), result.get("metrics", {}))
            if rows:
                cached = rows
            commands.append(f"# {job['id']} · попытка {number} · {job['reason']}\n{shlex.join(args)}")
            if attempt and "official" in spec:
                commands.append(shlex.join(spec["official"]))
            if attempt and "profile" in spec:
                commands.append(shlex.join(spec["profile"]))
            writer.writerow({
                "id": job["id"], "run_name": name, "algorithm": job["method"], "seed": job["seed"],
                "configuration": json.dumps(args[2:], ensure_ascii=False), "git_commit": state["git_commit"],
                "num_updates": row.get("num_updates"), "environment_steps": row.get("num_env_steps"),
                "gradient_updates": row.get("gradient_updates"), "wall_clock_time": row.get("wall_clock_time"),
                "status": attempt.get("status", job["status"]), "validation_solve_rate": val.get("eval/validation_solve_rate"),
                "validation_hard_solve_rate": val.get("eval/validation_hard_solve_rate"),
                "train_solve_rate": row.get("train/solve_rate"), "mean_episode_return": row.get("train/episode_return"),
                "checkpoint_path": str(Path(result["directory"]) / "models" / str(result["step"])) if result else None,
                "error_message": attempt.get("error"), "official_solve_rate": result.get("official", {}).get("solve_rate"),
            })
        observed[job["id"]] = cached
        pid = attempt.get("pid", "—") if job["status"] == "RUNNING" else "—"
        progress.append(f"| {job['id']} | {job['status']} | {row.get('num_updates', '—')} | "
                        f"{percentage(val.get('eval/validation_solve_rate'))} | {pid} |")

    progress.extend(("", "## Решения и события", ""))
    progress.extend(f"- {event['at']} · {event['message']}" for event in state["events"])
    progress.extend(("", "## Исторический официальный тест", "",
                     "Проверенные архивы из research_results/results_summary.csv. DR 46,25% (2 seeds), "
                     "PLR⊥ 45,00% (2 seeds), ACCEL 55,42% (3 seeds), исходный PACE-ACCEL 36,25% (3 seeds). "
                     "Эти значения относятся к полному бюджету."))
    atomic(report / "progress.md", "\n".join(progress) + "\n")
    atomic(report / "results.csv", out.getvalue())
    atomic(report / "ranking.md", "\n".join(ranking) + "\n")
    atomic(report / "experiment_commands.txt", "\n\n".join(commands) + "\n")

    if state["winner"]:
        winner = next(job for job in jobs if job["id"] == state["winner"])
        full = next(job for job in jobs if job["id"] == "FULL0")
        _, full_args, _ = command(state, full, max(1, len(full["attempts"])))
        write_json(report / "best_config.json", {
            "id": winner["id"], "method": winner["method"], "reason": state["selection_reason"],
            "validation": current_metric(winner), "git_commit": state["git_commit"],
            "source_hashes": state["lite_source"] if winner.get("study") == "lite" else state["source"],
            "validation_protocol": state["validation"], "full_seed0_command": full_args,
        })
        atomic(report / "best_algorithm.md", (
            f"# Победитель {winner['id']} · {winner['method']}\n\n{state['selection_reason']}\n\n"
            f"Конечная validation solve rate **{percentage(current_metric(winner)['eval/validation_solve_rate'])}**. "
            f"Статус FULL0 **{full['status']}**. Параметры и команда в best_config.json.\n"
        ))
    signature = [(job["status"], len(observed[job["id"]])) for job in jobs]
    stamp = report / ".figures-state.json"
    if not stamp.exists() or json.loads(stamp.read_text(encoding="utf-8")) != [list(pair) for pair in signature]:
        diagnostics(jobs, observed, report)
        write_json(stamp, signature)
    if lite:
        render_profiling(report, jobs)


def render_profiling(report, jobs):
    paths = [report / "profiling-cpu.json"]
    smoke = next(job for job in jobs if job["id"] == "LS00")
    paths.extend(Path(attempt["folder"]) / "profiling.json" for attempt in smoke["attempts"])
    profile_notes = ["# Профиль H-PACE-Lite", ""]
    validation_notes = ["# Проверка аналитического градиента", "",
                        "Проверка по jax.grad на зафиксированных признаках, действиях и advantages.", ""]
    for path in paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        stages = data["stages"]
        profile_notes.extend([
            f"## {data['device']}", "",
            f"Checkpoint ACCEL на 5 000 раундах, {data['envs']} сред × {data['steps']} шагов, "
            f"{data['repeat']} синхронизированных повторов. Rollout {data['rollout_env_steps_per_s']:.0f} шагов/с.", "",
            "| Этап | Компиляция, с | Медиана, мс |", "| --- | ---: | ---: |",
        ])
        profile_notes.extend(f"| {key} | {row['compile_s']:.3f} | {row['median_ms']:.3f} |"
                             for key, row in stages.items())
        memory = (data["memory"] or {}).get("peak_bytes_in_use")
        if memory is not None:
            profile_notes.extend(("", f"Пиковое значение аллокатора JAX в диагностике: {memory / 2**20:.1f} МиБ."))
        profile_notes.append("")
        validation_notes.extend((f"## {data['device']}", ""))
        validation_notes.extend(f"- {key} = {value:.8g}" for key, value in data["gradients"].items())
        validation_notes.extend(("", "| Метрики | Pearson | Spearman | Top-k |", "| --- | ---: | ---: | ---: |"))
        validation_notes.extend(
            f"| {key} | {row['pearson']:.3f} | {row['spearman']:.3f} | {row['overlap']} / {row['top_k']} |"
            for key, row in data["scores"].items()
        )
        validation_notes.append("")
    if len(profile_notes) == 2:
        profile_notes.append("Синхронизированный профиль ещё не записан.")
        validation_notes.append("Ожидаются результаты на фиксированном rollout.")
    for attempt in smoke["attempts"]:
        error = attempt.get("result", {}).get("profile_error")
        if error:
            profile_notes.extend(("", f"Ошибка отдельного профилирования: {error}"))
    atomic(report / "profiling.md", "\n".join(profile_notes) + "\n")
    atomic(report / "gradient_validation.md", "\n".join(validation_notes) + "\n")


def diagnostics(jobs, observed, report):
    import numpy as np

    curves = {job["id"]: observed[job["id"]] for job in jobs if job["target"] >= TARGET and observed[job["id"]]}
    notes = [
        "# Диагностика PACE v2", "",
        "Квантили нормы градиента усредняются по rollout-батчам за 250 раундов. Разрешимость и длина пути "
        "считаются по равномерной выборке до 256 слотов. Уникальность учитывает планировку, старт и цель "
        "во всём буфере. ESS рассчитан по итоговым replay-весам.", "",
    ]
    if not curves:
        notes.append("Метрики предварительных прогонов ещё не поступили.")
        atomic(report / "diagnostics.md", "\n".join(notes) + "\n")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = report / "figures"
    figures.mkdir(exist_ok=True)

    def plot(name, xlabel, ylabel, draw):
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        draw(ax)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.2)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=8, ncol=2)
        fig.savefig(figures / f"{name}.png", dpi=150)
        plt.close(fig)

    for name, key, label in (
        ("validation", "eval/validation_solve_rate", "Validation solve rate"),
        ("hard-validation", "eval/validation_hard_solve_rate", "Hard validation solve rate"),
        ("threshold", "pace/threshold", "Порог нормы градиента"),
        ("effective-sample-size", "buffer/effective_sample_size", "Replay ESS"),
        ("buffer-diversity", "buffer/unique_levels", "Уникальные лабиринты"),
        ("curriculum-difficulty", "buffer/mean_shortest_path", "Средняя длина кратчайшего пути"),
        ("wall-clock", "wall_clock_time", "Время сегмента, с"),
    ):
        points = {identifier: [(r["num_updates"], r[key]) for r in rows if key in r]
                  for identifier, rows in curves.items()}
        if any(points.values()):
            def draw(ax):
                for identifier, pairs in points.items():
                    if pairs:
                        ax.plot(*zip(*pairs), marker=".", label=identifier)
            plot(name, "Раунды сбора rollout", label, draw)

    search = ranked(job for job in jobs if job["target"] == TARGET)
    if search:
        plot("final-validation", "Кандидат", "Validation solve rate на 5 000",
             lambda ax: ax.bar([job["id"] for job in search],
                               [current_metric(job)["eval/validation_solve_rate"] for job in search]))
        if report == LITE_REPORT:
            def alpha_plot(ax):
                for item in search:
                    if item.get("study") == "lite" or item["id"] == "B00":
                        weight = item["alpha"] if item.get("study") == "lite" else 0.
                        ax.scatter(weight, current_metric(item)["eval/validation_solve_rate"], label=item["id"])
            plot("hybrid-weight", "Вес Lite", "Validation solve rate", alpha_plot)

            def pareto(ax):
                points = sorted(
                    (current_metric(item)["wall_clock_time"] / 60,
                     current_metric(item)["eval/validation_solve_rate"], item["id"]) for item in search
                )
                frontier = []
                for elapsed, quality, name in points:
                    ax.scatter(elapsed, quality, label=name)
                    if not frontier or quality > frontier[-1][1]:
                        frontier.append((elapsed, quality))
                ax.plot(*zip(*frontier), color="grey", alpha=.55)
            plot("pareto", "Время 5 000 раундов, мин", "Validation solve rate", pareto)
        for key, name in (("floor", "floor-weight"), ("q", "quantile")):
            selected = [job for job in search if job["id"].startswith("F")
                        and (job["q"] == .75 if key == "floor" else job["floor"] == .20)]
            if selected:
                def draw(ax):
                    for job in selected:
                        ax.scatter(job[key], current_metric(job)["eval/validation_solve_rate"], label=job["id"])
                plot(name, key, "Validation solve rate на 5 000", draw)

    buffers = {}
    for job in jobs:
        if job["id"] not in curves:
            continue
        directory = checkpoint_dir(job["attempts"][-1]["run_name"], job["seed"])
        paths = sorted(directory.glob("buffer-*.npz"), key=lambda p: int(p.stem.split("-")[1]))
        if paths:
            with np.load(paths[-1], allow_pickle=False) as data:
                buffers[job["id"]] = dict(data)
    for name, xkey, ykey, label, transform in (
        ("gradient-distributions", "grad_squared", None, "log10 нормы градиента",
         lambda x: np.log10(np.maximum(np.sqrt(x), 1e-12))),
        ("score-distributions", "scores", None, "Replay score", lambda x: x),
        ("score-vs-gradient", "grad_squared", "scores", "Норма градиента", np.sqrt),
        ("pace-vs-maxmc", "grad_squared", "maxmc", "Норма градиента", np.sqrt),
    ):
        items = {ident: data for ident, data in buffers.items() if xkey in data and (ykey is None or ykey in data)}
        if items:
            def draw(ax):
                for ident, data in items.items():
                    valid = np.isfinite(data[xkey])
                    if ykey is None:
                        values = transform(data[xkey][valid])
                        if len(values):
                            ax.hist(values, bins=30, density=True, histtype="step", label=ident)
                    else:
                        valid &= np.isfinite(data[ykey])
                        ax.scatter(transform(data[xkey][valid]), data[ykey][valid], s=3, alpha=.25, label=ident)
            plot(name, label, ykey or "Плотность", draw)

    for ident, data in buffers.items():
        score = data["scores"][data["sampled_indices"]]
        distance = data["distances"]
        valid = np.isfinite(score) & (distance >= 0)
        if valid.sum() >= 3 and np.ptp(score[valid]) > 0 and np.ptp(distance[valid]) > 0:
            corr = np.corrcoef(score[valid], distance[valid])[0, 1]
            notes.append(f"- {ident} · Pearson score / кратчайший путь на {valid.sum()} уровнях: {corr:.3f}.")

    full = [(job["seed"], job["attempts"][-1]["result"]["official"]) for job in jobs
            if job["target"] == HORIZON and job["status"] == "COMPLETED"]
    if full:
        solves = [metrics["solve_rate"] for _, metrics in full]
        std = f", σ выборки {100 * np.std(solves, ddof=1):.2f} п.п." if len(full) > 1 else ""
        notes.extend(("", "## Официальный zero-shot", "",
                      f"Среднее {percentage(np.mean(solves))} по {len(full)} seeds{std}.", "",
                      "| Seed | Mean solve rate | Mean return | " + " | ".join(EXPERT_LEVELS) + " |",
                      "| --- | ---: | ---: | " + " | ".join(["---:"] * len(EXPERT_LEVELS)) + " |"))
        for seed, item in full:
            rates = " | ".join(percentage(item["per_level"][name]) for name in EXPERT_LEVELS)
            notes.append(f"| {seed} | {percentage(item['solve_rate'])} | {item['mean_return']:.3f} | {rates} |")
        plot("official-zero-shot", "Полный бюджет", "Mean solve rate",
             lambda ax: ax.bar([f"Seed {seed}" for seed, _ in full] + ["PACE исходный", "ACCEL"],
                               solves + [.3625, .5541666667]))
    atomic(report / "diagnostics.md", "\n".join(notes) + "\n")
