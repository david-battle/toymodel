#!/bin/bash
# Launch training + watchdog in background, detached from shell.
# Usage: ./run_training_watchdog.sh [train.py args...]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

# Default training args (full 124M run, resuming from last checkpoint)
TRAIN_ARGS=(
    --resume ckpt/last.pt
    --n-layer 12
    --n-head 12
    --n-embd 768
    --block 512
    --budget 2500000000
    --micro-batch 4
    --accum 64
    --eval-steps 500
    --ckpt-steps 500
    --log-steps 20
)

# Allow override from command line
if [ $# -gt 0 ]; then
    TRAIN_ARGS=("$@")
fi

# Check for existing training process
if [ -f "$LOG_DIR/pilot.pid" ]; then
    OLD_PID=$(cat "$LOG_DIR/pilot.pid" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Training already running (PID $OLD_PID). Use 'kill -TERM $OLD_PID' to stop first."
        exit 1
    fi
fi

# Check for existing watchdog
if [ -f "$LOG_DIR/watchdog.pid" ]; then
    OLD_WD=$(cat "$LOG_DIR/watchdog.pid" 2>/dev/null || echo "")
    if [ -n "$OLD_WD" ] && kill -0 "$OLD_WD" 2>/dev/null; then
        echo "Watchdog already running (PID $OLD_WD). Stop it first."
        exit 1
    fi
fi

echo "Starting training with args: ${TRAIN_ARGS[*]}"

# Start training in background with setsid (detached)
setsid ./.venv/bin/python train.py "${TRAIN_ARGS[@]}" \
    >"$LOG_DIR/pilot.log" 2>&1 </dev/null &
TRAIN_PID=$!

echo $TRAIN_PID > "$LOG_DIR/pilot.pid"
echo "Training started (PID $TRAIN_PID)"

# Give training a moment to start and write first log lines
sleep 5

# Start watchdog in background with setsid
setsid ./.venv/bin/python watchdog.py \
    --threshold 10000 \
    --interval 30 \
    --restart-delay 10 \
    >"$LOG_DIR/watchdog.log" 2>&1 </dev/null &
WATCHDOG_PID=$!

echo $WATCHDOG_PID > "$LOG_DIR/watchdog.pid"
echo "Watchdog started (PID $WATCHDOG_PID)"

echo ""
echo "Both processes detached. Logs:"
echo "  Training: $LOG_DIR/pilot.log"
echo "  Watchdog: $LOG_DIR/watchdog.log"
echo ""
echo "To stop gracefully: kill -TERM $TRAIN_PID  (checkpoints and exits)"
echo "To stop watchdog:   kill -TERM $WATCHDOG_PID"