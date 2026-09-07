#!/usr/bin/env bash
# Launch train.py fully detached so it survives the shell that started it.
# The tool/shell that launches this must NOT keep waiting on the child: the
# setsid + redirect of ALL three fds (stdin/out/err to log) is what lets the
# launching shell return immediately. This is the only supported way to start
# a background training run — a bare `python train.py &` holds the shell open.
#
# Pause (checkpoint + exit, frees GPU): kill -TERM "$(cat logs/pilot.pid)".
# Resume: ./.venv/bin/python train.py --resume ckpt/last.pt
# Monitor: tail -f logs/pilot.log ; watch nvidia-smi
#
# Usage:
#   ./run_training.sh [extra train.py args...]
set -euo pipefail
cd "$(dirname "$0")"

LOGDIR="logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/pilot.log"
PIDFILE="$LOGDIR/pilot.pid"

# default pilot recipe; override/extend with CLI args, e.g.
#   ./run_training.sh --budget 30000000 --micro-batch 8
ARGS=(--budget 50000000 --micro-batch 16 --accum 32 --eval-steps 75 --ckpt-steps 100 --log-steps 20)
ARGS+=("$@")

setsid ./.venv/bin/python train.py "${ARGS[@]}" >"$LOG" 2>&1 </dev/null &
echo $! > "$PIDFILE"
echo "launched train.py pid $(cat "$PIDFILE") log=$LOG args: ${ARGS[*]}"