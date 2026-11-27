import json
import os
import sys
import time
import runpy
import subprocess
from pathlib import Path
from datetime import datetime

BASE = Path.cwd().resolve()

WAIT_PID = int(sys.argv[1])

PREVIOUS = (
    BASE / "checkpoints"
    / "pace-accel-30k-s2-20261126-012135"
    / "2/models/119"
)

SOURCE = BASE / "checkpoints/accel-s0/0/config.json"

assert SOURCE.is_file(), SOURCE

config = json.loads(SOURCE.read_text())

assert config["use_accel"] is True
assert config["num_updates"] == 30000
assert config["score_function"] == "MaxMC"

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_name = f"accel-s2-{timestamp}"

# Получаем оригинальный baseline из Git,
# не изменяя нашу реализацию PACE-ACCEL.
script = BASE / "examples/maze_accel_baseline.py"

original = subprocess.check_output(
    [
        "git", "show",
        "refs/remotes/origin/main:examples/maze_plr.py"
    ],
    text=True
)

if script.exists():
    assert script.read_text() == original, script
else:
    script.write_text(original)

print("Waiting for PACE-ACCEL seed 2", flush=True)
print("PID:", WAIT_PID, flush=True)

while True:
    state = subprocess.run(
        ["ps", "-p", str(WAIT_PID), "-o", "stat="],
        capture_output=True,
        text=True
    ).stdout.strip()

    if not state or state.startswith("Z"):
        break

    time.sleep(60)

assert PREVIOUS.is_dir(), (
    "PACE-ACCEL завершился без финального чекпоинта. "
    "ACCEL не запущен."
)

print("PACE-ACCEL finished.", flush=True)
print("Starting ACCEL seed 2.", flush=True)

config.update({
    "seed": 2,
    "run_name": run_name,
    "mode": "train",
    "checkpoint_directory": None,
    "checkpoint_to_eval": -1,
})

assert not (
    BASE / "checkpoints" / run_name / "2"
).exists()

os.environ["WANDB_MODE"] = "offline"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORMS"] = "cuda"

sys.path.insert(0, str(BASE / "examples"))

print("Run:", run_name, flush=True)
print("Config:", config, flush=True)

module = runpy.run_path(str(script))
module["main"](config, project="JAXUED_TEST")

assert (
    BASE / "checkpoints" / run_name / "2/models/119"
).is_dir()

print("ACCEL seed 2 finished.", flush=True)
