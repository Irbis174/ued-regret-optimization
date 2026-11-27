#!/usr/bin/env bash
set -eu
cd -- "$(dirname -- "$0")/.."

if ! tmux has-session -t =pace-v2 2>/dev/null; then
    tmux new-session -d -s pace-v2 -c "$PWD" \
        'exec .venv/bin/python -u examples/pace_research_queue.py >>reports/pace_v2/runner.log 2>&1'
fi

# Keeps the WSL client open when launched detached from Windows.
while tmux has-session -t =pace-v2 2>/dev/null; do
    sleep 30
done
