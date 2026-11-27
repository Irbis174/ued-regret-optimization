from examples import pace_research_queue as queue


def finish(task, score, hard):
    task["status"] = "COMPLETED"
    task["attempts"] = [{"result": {"metrics": {
        "eval/validation_solve_rate": score,
        "eval/validation_hard_solve_rate": hard,
        "eval/recent_solve_rate": score,
        "pace/nonfinite_fraction": 0.,
        "pace/near_zero_score_fraction": 0.,
        "buffer/solvable_fraction": 1.,
        "wall_clock_time": 300.,
    }}}]


def state_for(floor, accel):
    state = {"jobs": [queue.job("F01"), queue.job("F02"), queue.job("F03", floor=.3), queue.job("B00", "MaxMC")],
             "events": [], "winner": None, "stage": "main"}
    for task, score in zip(state["jobs"], (floor, floor - .01, floor - .02, accel)):
        finish(task, score, score / 2)
    return state


def test_competitive_standalone_picks_full_continuation():
    state = state_for(.42, .44)
    queue.choose_next(state)
    assert state["winner"] == "F01"
    tasks = state["jobs"][-3:]
    assert [task["seed"] for task in tasks] == [0, 1, 2]
    assert all(task["target"] == 30000 for task in tasks)
    assert tasks[0]["parent"] == "F01"


def test_hybrid_launch_requires_completed_standalone_search():
    state = state_for(.31, .44)
    queue.choose_next(state)
    assert state["stage"] == "ablation"
    queue.choose_next(state)
    assert state["stage"] == "ablation"
    for task in state["jobs"][-2:]:
        finish(task, .30, .15)
    queue.choose_next(state)
    assert state["stage"] == "hybrid"
    finish(state["jobs"][-2], .37, .18)
    finish(state["jobs"][-1], .32, .16)
    queue.choose_next(state)
    assert state["winner"] == "H01"
    assert state["jobs"][-3]["parent"] == "H01"


def test_gpu_guard_checks_compute_and_memory(monkeypatch):
    def query(args, **kwargs):
        from subprocess import CompletedProcess

        key = args[1]
        return CompletedProcess(args, 0, telemetry[key], "")

    telemetry = {
        "--query-gpu=index,uuid,memory.free,utilization.gpu": "0, GPU-test, 7000, 4\n",
        "--query-compute-apps=pid,gpu_uuid": "123, GPU-test\n",
    }
    monkeypatch.setattr(queue.subprocess, "run", query)
    monkeypatch.setattr(queue.subprocess, "check_output", lambda *arg, **kw: "")
    assert not queue.gpu_status()[0]
    telemetry["--query-compute-apps=pid,gpu_uuid"] = ""
    assert queue.gpu_status()[0]
    telemetry["--query-gpu=index,uuid,memory.free,utilization.gpu"] = "0, GPU-test, 2300, 4\n"
    assert not queue.gpu_status()[0]


def test_lite_campaign_reuses_controls_and_can_select_standalone(monkeypatch, tmp_path):
    monkeypatch.setattr(queue, "LITE_MANUAL", tmp_path / "manual")
    control = queue.job("B00", "MaxMC")
    original = queue.job("F03", floor=.30)
    smoke = queue.lite_job("LS00", target=queue.SMOKE)
    state = {"jobs": [control, original, smoke], "stage": "lite_search", "winner": None, "events": [],
             "lite_source": {"mock": "hash"}}
    finish(control, .76, .58)
    finish(original, .67, .43)
    finish(smoke, .03, .01)
    queue.choose_next(state)
    assert state["jobs"][-1]["id"] == "H01"
    finish(state["jobs"][-1], .68, .45)
    queue.choose_next(state)
    assert state["jobs"][-1]["id"] == "H02"
    finish(state["jobs"][-1], .73, .55)
    queue.choose_next(state)
    assert state["jobs"][-2]["status"] == "SKIPPED"
    assert state["jobs"][-1]["id"] == "L01"
    finish(state["jobs"][-1], .735, .57)
    queue.choose_next(state)
    assert state["winner"] == "L01"
    full = state["jobs"][-3:]
    assert [task["seed"] for task in full] == [0, 1, 2]
    assert full[0]["parent"] == "L01"
    assert all(task["method"] == "pace_lite_floor" and task["alpha"] == 1 for task in full)


def test_f03_full_jobs_resume_only_seed_zero_and_prevent_duplicates():
    source = queue.job("F03", floor=.30)
    finish(source, .677, .430)
    source["attempts"][0]["result"]["metrics"].update(num_updates=5000, num_env_steps=40_960_000)
    state = {"jobs": [source]}
    full = queue.f03_full_jobs(state)
    assert [task["seed"] for task in full] == [0, 1, 2]
    assert [task["parent"] for task in full] == ["F03", None, None]
    assert all(task["target"] == 30000 and task["floor"] == .30 for task in full)
    state["jobs"].extend(full)
    assert queue.f03_full_jobs(state) == []
